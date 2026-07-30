from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

from .algorithms.config import load_algorithm_config, validate_algorithm_config
from .experiment_config.generation import ExperimentGenerator
from .experiment_config.io import load_yaml, write_yaml
from .experiment_config.runner import build_simulator


TUNED_ALGORITHMS = ("eulc_pso", "eulc_ga", "eulc_ac_aco")


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _choice(rng: random.Random, values: list[Any]) -> Any:
    if not values:
        raise ValueError("tuning choice list cannot be empty")
    return copy.deepcopy(values[rng.randrange(len(values))])


def build_trial_config(
    base_config: dict[str, Any],
    algorithm_id: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    optimizer = config["optimizer"]
    population, iterations = sample.pop("budget_pair")
    optimizer["population_size"] = int(population)
    optimizer["max_iterations"] = int(iterations)
    optimizer["fitness_evaluation_budget"] = (
        int(population) * (int(iterations) + 1)
        if algorithm_id in {"eulc_pso", "eulc_ga"}
        else int(population) * int(iterations)
    )
    optimizer["params"].update(copy.deepcopy(sample))
    validate_algorithm_config(config)
    return config


def sample_trial_configs(
    base_config: dict[str, Any],
    algorithm_id: str,
    space: dict[str, Any],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if algorithm_id not in TUNED_ALGORITHMS:
        raise ValueError(f"unsupported tuning algorithm: {algorithm_id}")
    rng = random.Random(int(seed))
    configs: list[dict[str, Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(configs) < int(count):
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("search space has fewer unique configurations than requested")
        sample: dict[str, Any] = {"budget_pair": _choice(rng, space["budget_pairs"])}
        if algorithm_id == "eulc_pso":
            for key in (
                "inertia", "c1", "c2", "omega",
                "stagnation_restart_iterations", "stagnation_restart_fraction",
            ):
                sample[key] = _choice(rng, space[key])
            if float(sample["c1"]) + float(sample["c2"]) > 4.5:
                continue
        elif algorithm_id == "eulc_ga":
            sample["crossover_rate"] = _choice(rng, space["crossover_rate"])
            sample["mutation_sigma"] = _choice(rng, space["mutation_sigma"])
        else:
            sample["alpha"] = _choice(rng, space["alpha"])
            beta_start, beta_end = _choice(rng, space["beta_schedule"])
            rho_start, rho_end = _choice(rng, space["rho_schedule"])
            energy_w, eulc_w, centrality_w = _choice(rng, space["heuristic_weights"])
            sample.update({
                "beta_start": beta_start, "beta_end": beta_end,
                "rho_start": rho_start, "rho_end": rho_end,
                "chaos_strength": _choice(rng, space["chaos_strength"]),
                "stagnation_limit": _choice(rng, space["stagnation_limit"]),
                "heuristic_energy_weight": energy_w,
                "heuristic_eulc_weight": eulc_w,
                "heuristic_centrality_weight": centrality_w,
            })
        config = build_trial_config(base_config, algorithm_id, sample)
        fingerprint = stable_hash(config)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        configs.append({
            "algorithm_id": algorithm_id,
            "trial_id": f"{algorithm_id}__{fingerprint}",
            "config_hash": fingerprint,
            "config": config,
        })
    return configs


def select_stage_cases(
    cases: Iterable[dict[str, Any]],
    scenario_filter: dict[str, Any],
    seeds: Iterable[int],
) -> list[dict[str, Any]]:
    wanted_seeds = {int(seed) for seed in seeds}
    wanted_distributions = set(scenario_filter["distributions"])
    wanted_size = [int(v) for v in scenario_filter["deployment_size_m"]]
    selected = [
        copy.deepcopy(case)
        for case in cases
        if [int(v) for v in case["environment"]["deployment_size_m"]] == wanted_size
        and case["metadata"]["density"] == scenario_filter["density"]
        and case["environment"]["distribution"] in wanted_distributions
        and int(case["metadata"]["base_seed"]) in wanted_seeds
    ]
    if not selected:
        raise ValueError("scenario filter selected no experiment cases")
    return selected


def _run_trial_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        case = copy.deepcopy(job["case"])
        case["execution"]["rounds"] = int(job["rounds"])
        case["output"]["save_convergence"] = False
        simulator, execution = build_simulator(case, job["algorithm_config"])
        metrics = simulator.run(
            stop_on_first_dead=bool(execution["stop_on_first_dead"]),
            stop_at_ft5=bool(execution["stop_at_first_5pct_dead"]),
            min_alive_ratio=execution["min_alive_ratio"],
            max_rounds=int(execution["rounds"]),
        )
        initial_total = simulator.case.initial_energy * simulator.case.node_count
        energy_curve = metrics.residual_energy_by_round
        residual_auc = (
            statistics.mean(value / initial_total for value in energy_curve)
            if energy_curve and initial_total > 0 else 0.0
        )
        average_delay = metrics.average_e2e_delay_s
        return {
            **job["identity"],
            "status": "completed",
            "rounds_completed": len(energy_curve),
            "fnd_round": metrics.fnd_round,
            "fnd_censored": metrics.fnd_round is None,
            "survival_score": (
                int(metrics.fnd_round)
                if metrics.fnd_round is not None else len(energy_curve) + 1
            ),
            "residual_energy_auc": residual_auc,
            "final_residual_energy_fraction": (
                metrics.residual_energy / initial_total if initial_total > 0 else 0.0
            ),
            "packets_delivered": int(metrics.packets_delivered),
            "average_e2e_delay_s": (
                float(average_delay) if average_delay is not None else float("inf")
            ),
            "runtime_seconds": time.perf_counter() - started,
            "error": None,
        }
    except Exception as exc:
        return {
            **job["identity"], "status": "failed",
            "rounds_completed": 0, "fnd_round": None, "fnd_censored": True,
            "survival_score": 0, "residual_energy_auc": 0.0,
            "final_residual_energy_fraction": 0.0, "packets_delivered": 0,
            "average_e2e_delay_s": float("inf"),
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }


def aggregate_rankings(
    results: list[dict[str, Any]],
    expected_runs_per_trial: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["algorithm_id"], row["trial_id"]), []).append(row)
    rankings = []
    for (algorithm_id, trial_id), rows in grouped.items():
        completed = [row for row in rows if row["status"] == "completed"]
        delays = [
            float(row["average_e2e_delay_s"]) for row in completed
            if math.isfinite(float(row["average_e2e_delay_s"]))
        ]
        ranking = {
            "algorithm_id": algorithm_id,
            "trial_id": trial_id,
            "completed_runs": len(completed),
            "expected_runs": int(expected_runs_per_trial),
            "completion_rate": len(completed) / max(1, int(expected_runs_per_trial)),
            "median_survival_score": (
                statistics.median(row["survival_score"] for row in completed)
                if completed else 0.0
            ),
            "mean_residual_energy_auc": (
                statistics.mean(row["residual_energy_auc"] for row in completed)
                if completed else 0.0
            ),
            "std_residual_energy_auc": (
                statistics.pstdev(row["residual_energy_auc"] for row in completed)
                if len(completed) > 1 else 0.0
            ),
            "mean_packets_delivered": (
                statistics.mean(row["packets_delivered"] for row in completed)
                if completed else 0.0
            ),
            "mean_average_e2e_delay_s": statistics.mean(delays) if delays else float("inf"),
            "mean_runtime_seconds": (
                statistics.mean(row["runtime_seconds"] for row in completed)
                if completed else float("inf")
            ),
        }
        rankings.append(ranking)
    rankings.sort(key=lambda row: (
        row["algorithm_id"],
        -row["completion_rate"],
        -row["median_survival_score"],
        -row["mean_residual_energy_auc"],
        row["std_residual_energy_auc"],
        -row["mean_packets_delivered"],
        row["mean_average_e2e_delay_s"],
        row["mean_runtime_seconds"],
        row["trial_id"],
    ))
    for algorithm_id in TUNED_ALGORITHMS:
        rank = 0
        for row in rankings:
            if row["algorithm_id"] == algorithm_id:
                rank += 1
                row["rank"] = rank
    return rankings


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _append_durable_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _publish_progress(
    output_dir: Path,
    *,
    stage: str,
    stage_done: int,
    stage_total: int,
    overall_done: int,
    overall_total: int,
    latest: dict[str, Any] | None,
    stage_started: float,
) -> None:
    elapsed = max(0.0, time.perf_counter() - stage_started)
    remaining = max(0, stage_total - stage_done)
    eta_seconds = (
        elapsed / stage_done * remaining if stage_done > 0 else None
    )
    progress = {
        "stage": stage,
        "stage_completed": stage_done,
        "stage_total": stage_total,
        "stage_percent": 100.0 * stage_done / max(1, stage_total),
        "overall_completed": overall_done,
        "overall_total": overall_total,
        "overall_percent": 100.0 * overall_done / max(1, overall_total),
        "stage_elapsed_seconds": elapsed,
        "stage_eta_seconds": eta_seconds,
        "latest_algorithm": latest.get("algorithm_id") if latest else None,
        "latest_trial_id": latest.get("trial_id") if latest else None,
        "latest_case_id": latest.get("case_id") if latest else None,
        "latest_status": latest.get("status") if latest else None,
    }
    progress_path = output_dir / "progress.json"
    temporary = progress_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(progress_path)
    message = (
        f"[{stage}] {stage_done}/{stage_total} "
        f"({progress['stage_percent']:.1f}%) | overall "
        f"{overall_done}/{overall_total} ({progress['overall_percent']:.1f}%)"
    )
    if latest:
        message += (
            f" | {latest['algorithm_id']} | {latest['trial_id']} "
            f"| {latest['case_id']} | {latest['status']}"
        )
    print(message, flush=True)
    with (output_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_tuning(
    spec_path: Path,
    output_dir: Path,
    *,
    workers: int = 2,
    execute: bool = False,
    smoke: bool = False,
    algorithm_id: str | None = None,
    checkpoint_callback: Callable[[Path, str], None] | None = None,
) -> dict[str, Any]:
    root = spec_path.resolve().parents[2]
    spec = load_yaml(spec_path)
    all_cases = ExperimentGenerator(root / spec["experiment_set"]).build_cases()
    candidate_count = 1 if smoke else int(spec["candidates_per_algorithm"])
    stages = copy.deepcopy(spec["stages"])
    if smoke:
        stages = [{"name": "smoke", "rounds": 2, "seeds": [0], "keep": 1}]
    selected_algorithms = (
        (algorithm_id,) if algorithm_id is not None else TUNED_ALGORITHMS
    )
    if any(value not in TUNED_ALGORITHMS for value in selected_algorithms):
        raise ValueError(f"unsupported tuning algorithm: {algorithm_id}")
    candidates: dict[str, list[dict[str, Any]]] = {}
    for offset, selected_id in enumerate(selected_algorithms):
        base = load_algorithm_config(root / spec["algorithms"][selected_id])
        candidates[selected_id] = sample_trial_configs(
            base, selected_id, spec["search_spaces"][selected_id],
            candidate_count, int(spec["tuning_seed"]) + offset,
        )
        if smoke:
            for trial in candidates[selected_id]:
                trial["config"]["optimizer"]["population_size"] = 4
                trial["config"]["optimizer"]["max_iterations"] = 1
                trial["config"]["optimizer"]["fitness_evaluation_budget"] = 4
                fingerprint = stable_hash(trial["config"])
                trial["config_hash"] = fingerprint
                trial["trial_id"] = f"{selected_id}__{fingerprint}"
    plan = {
        "name": spec["name"], "execute": execute, "smoke": smoke,
        "algorithms": list(selected_algorithms),
        "workers": int(workers), "candidate_count_per_algorithm": candidate_count,
        "stages": [], "ranking_policy": [
            "completion_rate:max", "median_survival_score:max",
            "mean_residual_energy_auc:max", "std_residual_energy_auc:min",
            "mean_packets_delivered:max", "mean_average_e2e_delay_s:min",
            "mean_runtime_seconds:min",
        ],
        "note": "Raw objective J is not averaged across different network states.",
    }
    planned_candidates_per_algorithm = candidate_count
    for stage in stages:
        stage_cases = select_stage_cases(
            all_cases, spec["scenario_filter"], stage["seeds"]
        )
        plan["stages"].append({
            **stage,
            "candidates_per_algorithm": planned_candidates_per_algorithm,
            "cases_per_trial": len(stage_cases),
            "planned_runs": (
                len(selected_algorithms)
                * planned_candidates_per_algorithm
                * len(stage_cases)
            ),
        })
        planned_candidates_per_algorithm = min(
            planned_candidates_per_algorithm, int(stage["keep"])
        )
    if not execute:
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "trials.jsonl"
    existing: dict[tuple[str, str, str], dict[str, Any]] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[(row["stage"], row["trial_id"], row["case_id"])] = row

    all_stage_rankings: list[dict[str, Any]] = []
    overall_total = sum(int(stage["planned_runs"]) for stage in plan["stages"])
    overall_done = 0
    for stage in stages:
        stage_started = time.perf_counter()
        stage_cases = select_stage_cases(all_cases, spec["scenario_filter"], stage["seeds"])
        jobs = []
        stage_rows = []
        for algorithm_id, trial_configs in candidates.items():
            for trial in trial_configs:
                for case in stage_cases:
                    identity = {
                        "stage": stage["name"], "algorithm_id": algorithm_id,
                        "trial_id": trial["trial_id"],
                        "trial_config_hash": trial["config_hash"],
                        "case_id": case["metadata"]["case_id"],
                        "case_config_hash": case["metadata"]["config_hash"],
                        "base_seed": case["metadata"]["base_seed"],
                        "distribution": case["environment"]["distribution"],
                    }
                    key = (identity["stage"], identity["trial_id"], identity["case_id"])
                    if key in existing and existing[key]["status"] == "completed":
                        stage_rows.append(existing[key])
                    else:
                        jobs.append({
                            "identity": identity, "case": case,
                            "algorithm_config": trial["config"], "rounds": stage["rounds"],
                        })
        stage_total = len(stage_rows) + len(jobs)
        stage_done = len(stage_rows)
        overall_done += stage_done
        _publish_progress(
            output_dir, stage=stage["name"],
            stage_done=stage_done, stage_total=stage_total,
            overall_done=overall_done, overall_total=overall_total,
            latest=None, stage_started=stage_started,
        )
        if int(workers) <= 1:
            for job in jobs:
                row = _run_trial_job(job)
                stage_rows.append(row)
                _append_durable_jsonl(results_path, row)
                stage_done += 1
                overall_done += 1
                _publish_progress(
                    output_dir, stage=stage["name"],
                    stage_done=stage_done, stage_total=stage_total,
                    overall_done=overall_done, overall_total=overall_total,
                    latest=row, stage_started=stage_started,
                )
                if checkpoint_callback is not None:
                    checkpoint_callback(
                        output_dir,
                        f"{stage['name']} {stage_done}/{stage_total}",
                    )
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [executor.submit(_run_trial_job, job) for job in jobs]
                for future in as_completed(futures):
                    row = future.result()
                    stage_rows.append(row)
                    _append_durable_jsonl(results_path, row)
                    stage_done += 1
                    overall_done += 1
                    _publish_progress(
                        output_dir, stage=stage["name"],
                        stage_done=stage_done, stage_total=stage_total,
                        overall_done=overall_done, overall_total=overall_total,
                        latest=row, stage_started=stage_started,
                    )
                    if checkpoint_callback is not None:
                        checkpoint_callback(
                            output_dir,
                            f"{stage['name']} {stage_done}/{stage_total}",
                        )
        rankings = aggregate_rankings(stage_rows, len(stage_cases))
        for row in rankings:
            row["stage"] = stage["name"]
        _write_csv(output_dir / f"rankings_{stage['name']}.csv", rankings)
        all_stage_rankings.extend(rankings)
        promoted: dict[str, list[dict[str, Any]]] = {}
        for algorithm_id in selected_algorithms:
            keep_ids = {
                row["trial_id"] for row in rankings
                if row["algorithm_id"] == algorithm_id
            }
            ordered_ids = [
                row["trial_id"] for row in rankings
                if row["algorithm_id"] == algorithm_id
            ][: int(stage["keep"])]
            promoted[algorithm_id] = [
                trial for trial in candidates[algorithm_id]
                if trial["trial_id"] in set(ordered_ids) & keep_ids
            ]
        candidates = promoted

    final_stage = stages[-1]["name"]
    final_rankings = [
        row for row in all_stage_rankings if row["stage"] == final_stage and row["rank"] == 1
    ]
    best_summary = {"tuning_plan": plan, "best": {}}
    for row in final_rankings:
        algorithm_id = row["algorithm_id"]
        trial = next(
            trial for trial in candidates[algorithm_id]
            if trial["trial_id"] == row["trial_id"]
        )
        write_yaml(output_dir / f"best_config_{algorithm_id.removeprefix('eulc_')}.yaml",
                   trial["config"])
        best_summary["best"][algorithm_id] = row
    (output_dir / "best_summary.json").write_text(
        json.dumps(best_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "rankings_all_stages.csv", all_stage_rankings)
    if checkpoint_callback is not None:
        checkpoint_callback(output_dir, "tuning completed")
    return best_summary
