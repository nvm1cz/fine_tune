from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from .algorithms.config import load_algorithm_config
from .experiment_config.generation import ExperimentGenerator
from .experiment_config.runner import build_simulator


SCHEMA_VERSION = 1


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_case(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    case = job["case"]
    simulator, execution = build_simulator(case, job["algorithm_config"])
    metrics = simulator.run(
        stop_on_first_dead=bool(execution["stop_on_first_dead"]),
        stop_at_ft5=bool(execution["stop_at_first_5pct_dead"]),
        min_alive_ratio=execution["min_alive_ratio"],
        max_rounds=int(execution["rounds"]),
    )
    initial_energy = simulator.case.initial_energy * simulator.case.node_count
    rounds = []
    cumulative_packets = 0
    for index, (energy, dead, packets, delay) in enumerate(zip(
        metrics.residual_energy_by_round,
        metrics.dead_nodes_by_round,
        metrics.packets_received_by_round,
        metrics.delay_by_round,
    ), 1):
        cumulative_packets += int(packets)
        rounds.append({
            "round": index,
            "residual_energy_j": float(energy),
            "energy_consumed_j": float(initial_energy - energy),
            "alive_nodes": int(simulator.case.node_count - dead),
            "dead_nodes": int(dead),
            "packets_delivered_round": int(packets),
            "packets_delivered_cumulative": cumulative_packets,
            "delay_s_round": float(delay),
            "average_delay_s_round": float(delay / packets) if packets else None,
        })
    return {
        "case_key": job["case_key"],
        "algorithm_id": simulator.algorithm_id,
        "case_id": case["metadata"]["case_id"],
        "distribution": case["environment"]["distribution"],
        "seed": int(case["metadata"]["base_seed"]),
        "config_hash": case["metadata"]["config_hash"],
        "rounds_completed": len(rounds),
        "fnd_round": metrics.fnd_round,
        "hnd_round": metrics.hnd_round,
        "lnd_round": metrics.lnd_round,
        "final_residual_energy_j": float(metrics.residual_energy),
        "total_energy_consumed_j": float(initial_energy - metrics.residual_energy),
        "packets_generated": int(metrics.packets_generated),
        "packets_delivered": int(metrics.packets_delivered),
        "packet_delivery_ratio": metrics.packet_delivery_ratio,
        "average_e2e_delay_s": metrics.average_e2e_delay_s,
        "runtime_seconds": time.perf_counter() - started,
        "round_metrics": rounds,
        "status": "completed",
    }


def _write_derived(output: Path, results: list[dict[str, Any]]) -> None:
    summaries = []
    round_rows = []
    completed = []
    for result in results:
        summary = {key: value for key, value in result.items() if key != "round_metrics"}
        summaries.append(summary)
        completed.append({"case_key": result["case_key"], "status": result["status"]})
        for row in result["round_metrics"]:
            round_rows.append({
                "algorithm_id": result["algorithm_id"],
                "case_id": result["case_id"],
                "distribution": result["distribution"],
                "seed": result["seed"],
                **row,
            })
    _atomic_csv(output / "case_summary.csv", summaries, list(summaries[0]))
    _atomic_csv(output / "round_metrics.csv", round_rows, list(round_rows[0]))
    temporary = output / "completed_cases.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in completed:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / "completed_cases.jsonl")


def run_benchmark(
    root: Path,
    experiment_spec: Path,
    algorithm_config_path: Path,
    output: Path,
    *,
    workers: int = 4,
    max_wall_time_seconds: int = 39600,
    checkpoint_callback: Callable[[Path, str], None] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    algorithm_config = load_algorithm_config(algorithm_config_path)
    algorithm_id = algorithm_config["algorithm"]["id"]
    cases = ExperimentGenerator(experiment_spec).build_cases()
    cases = sorted(cases, key=lambda case: case["metadata"]["case_id"])
    spec = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_id": algorithm_id,
        "algorithm_config_hash": _hash(algorithm_config),
        "experiment_spec_hash": _hash(json.loads(experiment_spec.read_text())),
        "git_commit": _commit(root),
        "total_cases": len(cases),
    }
    manifest_path = output / "benchmark_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("schema_version", "algorithm_id", "algorithm_config_hash", "experiment_spec_hash", "git_commit"):
            if previous.get(key) != spec[key]:
                raise ValueError(f"benchmark checkpoint mismatch: {key}")
    results_path = output / "benchmark_results.jsonl"
    results: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                results[row["case_key"]] = row
    jobs = []
    for case in cases:
        key = f"{algorithm_id}|{case['metadata']['case_id']}|{case['metadata']['base_seed']}"
        if key not in results:
            jobs.append({"case_key": key, "case": case, "algorithm_config": algorithm_config})
    started = time.perf_counter()
    runtimes = [float(row["runtime_seconds"]) for row in results.values()]
    manifest = {**spec, "completed": False, "stop_reason": None, "completed_cases": len(results)}
    _atomic_json(manifest_path, manifest)

    def can_start() -> bool:
        estimate = max(runtimes, default=300.0)
        return time.perf_counter() - started + estimate < max_wall_time_seconds

    def record(row: dict[str, Any]) -> None:
        results[row["case_key"]] = row
        runtimes.append(float(row["runtime_seconds"]))
        _append(results_path, row)
        ordered = [results[key] for key in sorted(results)]
        _write_derived(output, ordered)
        manifest["completed_cases"] = len(results)
        _atomic_json(manifest_path, manifest)
        print(f"[benchmark] {len(results)}/{len(cases)} | {row['case_id']} | completed", flush=True)
        if checkpoint_callback:
            checkpoint_callback(output, f"benchmark {len(results)}/{len(cases)}")

    iterator = iter(jobs)
    stopped = False
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
        pending = {}
        while len(pending) < workers and can_start():
            try:
                job = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(_run_case, job)] = job
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                record(future.result())
            while len(pending) < workers:
                if not can_start():
                    stopped = True
                    break
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                pending[executor.submit(_run_case, job)] = job
    complete = len(results) == len(cases)
    manifest.update({"completed": complete, "stop_reason": None if complete else "time_budget" if stopped else "batch_incomplete", "completed_cases": len(results)})
    _atomic_json(manifest_path, manifest)
    if checkpoint_callback:
        checkpoint_callback(output, "benchmark completed" if complete else "benchmark time budget")
    return manifest
