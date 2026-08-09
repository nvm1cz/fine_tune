from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .algorithms.config import load_algorithm_config
from .benchmark import _run_case
from .experiment_config.generation import ExperimentGenerator


RECOVERY_SCHEMA_VERSION = 1


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            results[str(row["attempt_key"])] = row
    return results


def _write_summary(path: Path, results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for key in sorted(results):
        result = results[key]
        rows.append({k: v for k, v in result.items() if k != "round_metrics"})
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _validate_source_failures(
    source_benchmark_dir: Path,
    distribution: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    manifest_path = source_benchmark_dir / "benchmark_manifest.json"
    summary_path = source_benchmark_dir / "case_summary.csv"
    if not manifest_path.exists() or not summary_path.exists():
        raise ValueError("source benchmark must contain manifest and case summary")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("algorithm_id") != "eulc_ac_aco":
        raise ValueError("source benchmark is not the AC-ACO benchmark")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {(distribution, int(seed)) for seed in seeds}
    observed = {
        (str(row["distribution"]), int(row["seed"]))
        for row in rows
        if row.get("status") == "infeasible"
    }
    if not wanted.issubset(observed):
        raise ValueError(
            f"requested recovery cases are not recorded as infeasible: {sorted(wanted - observed)}"
        )
    return manifest


def run_ac_aco_recovery(
    root: Path,
    source_benchmark_dir: Path,
    output_dir: Path,
    *,
    seeds: Sequence[int],
    distribution: str,
    chaos_states: Sequence[float],
    budget: int,
    max_wall_time_seconds: int,
    checkpoint_callback: Callable[[Path, str], None] | None = None,
    case_runner: Callable[[dict[str, Any]], dict[str, Any]] = _run_case,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")
    if not chaos_states or any(not 0.0 < float(value) < 1.0 for value in chaos_states):
        raise ValueError("chaos states must be strictly between zero and one")
    source_manifest = _validate_source_failures(
        source_benchmark_dir, distribution, seeds
    )
    config = load_algorithm_config(
        root / "configs/benchmarks/best_eulc_ac_aco.yaml"
    )
    configured_budget = int(config["optimizer"]["fitness_evaluation_budget"])
    if int(budget) != configured_budget:
        raise ValueError(
            f"recovery budget must match frozen benchmark budget {configured_budget}"
        )
    all_cases = ExperimentGenerator(
        root / "configs/experiment_sets/benchmark_heldout.yaml"
    ).build_cases()
    wanted_seeds = {int(seed) for seed in seeds}
    cases = [
        case for case in all_cases
        if case["environment"]["distribution"] == distribution
        and int(case["metadata"]["base_seed"]) in wanted_seeds
    ]
    if len(cases) != len(wanted_seeds):
        raise ValueError("recovery case selection is incomplete")
    cases.sort(key=lambda case: int(case["metadata"]["base_seed"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "recovery_results.jsonl"
    summary_path = output_dir / "recovery_summary.csv"
    manifest_path = output_dir / "recovery_manifest.json"
    results = _read_results(results_path)
    specification = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "algorithm_id": "eulc_ac_aco",
        "source_benchmark_git_commit": source_manifest.get("git_commit"),
        "source_algorithm_config_hash": source_manifest.get("algorithm_config_hash"),
        "source_experiment_spec_hash": source_manifest.get("experiment_spec_hash"),
        "distribution": distribution,
        "seeds": sorted(wanted_seeds),
        "chaos_states": [float(value) for value in chaos_states],
        "fitness_evaluation_budget_per_optimizer_call": configured_budget,
        "recovery_policy": "first feasible deterministic chaos-state restart",
        "note": (
            "Project-defined recovery ablation; results do not replace the primary "
            "held-out benchmark."
        ),
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in specification.items():
            if previous.get(key) != value:
                raise ValueError(f"recovery checkpoint mismatch: {key}")
    started = time.perf_counter()
    runtimes = [float(row.get("runtime_seconds", 0.0)) for row in results.values()]

    def recovered(seed: int) -> bool:
        return any(
            int(row["seed"]) == seed and row.get("status") == "completed"
            for row in results.values()
        )

    stopped_for_time = False
    for case in cases:
        seed = int(case["metadata"]["base_seed"])
        if recovered(seed):
            continue
        for attempt_index, chaos_state in enumerate(chaos_states, start=1):
            attempt_key = f"{distribution}|{seed}|{attempt_index}|{float(chaos_state):.12g}"
            if attempt_key in results:
                if results[attempt_key].get("status") == "completed":
                    break
                continue
            estimate = max(runtimes, default=300.0)
            if time.perf_counter() - started + estimate >= max_wall_time_seconds:
                stopped_for_time = True
                break
            attempt_config = copy.deepcopy(config)
            attempt_config["optimizer"]["params"]["chaos_initial_state"] = float(
                chaos_state
            )
            row = case_runner({
                "case_key": f"recovery|{attempt_key}",
                "case": copy.deepcopy(case),
                "algorithm_config": attempt_config,
            })
            row.update({
                "attempt_key": attempt_key,
                "attempt_index": attempt_index,
                "chaos_initial_state": float(chaos_state),
                "fitness_evaluation_budget": configured_budget,
            })
            results[attempt_key] = row
            runtimes.append(float(row.get("runtime_seconds", 0.0)))
            _append(results_path, row)
            _write_summary(summary_path, results)
            print(
                f"[recovery] seed={seed} attempt={attempt_index}/"
                f"{len(chaos_states)} chaos={chaos_state} status={row['status']}",
                flush=True,
            )
            progress = {
                **specification,
                "completed": False,
                "stop_reason": None,
                "attempts_completed": len(results),
                "recovered_seeds": sorted(seed for seed in wanted_seeds if recovered(seed)),
            }
            _atomic_json(manifest_path, progress)
            if checkpoint_callback:
                checkpoint_callback(
                    output_dir,
                    f"recovery seed {seed} attempt {attempt_index} {row['status']}",
                )
            if row.get("status") == "completed":
                break
        if stopped_for_time:
            break
    recovered_seeds = sorted(seed for seed in wanted_seeds if recovered(seed))
    exhausted_seeds = sorted(
        seed for seed in wanted_seeds
        if not recovered(seed)
        and all(
            f"{distribution}|{seed}|{index}|{float(state):.12g}" in results
            for index, state in enumerate(chaos_states, start=1)
        )
    )
    completed = len(recovered_seeds) + len(exhausted_seeds) == len(wanted_seeds)
    manifest = {
        **specification,
        "completed": completed,
        "stop_reason": "time_budget" if stopped_for_time else None,
        "attempts_completed": len(results),
        "recovered_seeds": recovered_seeds,
        "unrecovered_exhausted_seeds": exhausted_seeds,
    }
    _atomic_json(manifest_path, manifest)
    if checkpoint_callback:
        checkpoint_callback(
            output_dir,
            "recovery complete" if completed else "recovery stopped: time_budget",
        )
    return manifest
