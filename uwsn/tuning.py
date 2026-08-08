from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Iterable

from .algorithms.config import load_algorithm_config, validate_algorithm_config
from .experiment_config.generation import ExperimentGenerator
from .experiment_config.io import load_yaml, write_yaml
from .experiment_config.runner import build_simulator


TUNED_ALGORITHMS = ("eulc_pso", "eulc_ga", "eulc_ac_aco")
TERMINAL_TRIAL_STATUSES = frozenset(("completed", "infeasible"))
AC_ACO_NO_FEASIBLE_SOLUTION = "AC-ACO did not evaluate any solution"
TUNING_CHECKPOINT_SCHEMA = 2


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
    except RuntimeError as exc:
        if str(exc) == AC_ACO_NO_FEASIBLE_SOLUTION:
            return {
                **job["identity"], "status": "infeasible",
                "rounds_completed": 0, "fnd_round": None,
                "fnd_censored": True, "survival_score": 0,
                "residual_energy_auc": 0.0,
                "final_residual_energy_fraction": 0.0,
                "packets_delivered": 0,
                "average_e2e_delay_s": float("inf"),
                "runtime_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
                "invalid_reason": "no_feasible_solution",
            }
        return {
            **job["identity"], "status": "failed",
            "rounds_completed": 0, "fnd_round": None, "fnd_censored": True,
            "survival_score": 0, "residual_energy_auc": 0.0,
            "final_residual_energy_fraction": 0.0, "packets_delivered": 0,
            "average_e2e_delay_s": float("inf"),
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
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


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _repository_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _trial_key(identity: dict[str, Any]) -> str:
    return "|".join(
        (
            str(identity["algorithm_id"]),
            str(identity["stage"]),
            str(identity["trial_config_hash"]),
            str(identity.get("scenario", identity["case_id"])),
            str(identity["base_seed"]),
        )
    )


def _read_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("trial_key")
        if key is None:
            key = _trial_key(row)
            row["trial_key"] = key
        previous = results.get(key)
        if (
            previous is not None
            and previous.get("status") == "completed"
            and row.get("status") == "completed"
            and previous != row
        ):
            raise ValueError(
                f"conflicting completed trial at {path}:{line_number}: {key}"
            )
        results[key] = row
    return results


def _read_rankings(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"required prior-stage ranking is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _promote_candidates(
    candidates: dict[str, list[dict[str, Any]]],
    rankings: list[dict[str, Any]],
    selected_algorithms: tuple[str, ...],
    keep: int,
) -> dict[str, list[dict[str, Any]]]:
    promoted: dict[str, list[dict[str, Any]]] = {}
    for selected_id in selected_algorithms:
        ordered_ids = [
            row["trial_id"]
            for row in rankings
            if row["algorithm_id"] == selected_id
        ][: int(keep)]
        wanted = set(ordered_ids)
        promoted[selected_id] = [
            trial for trial in candidates[selected_id]
            if trial["trial_id"] in wanted
        ]
        if len(promoted[selected_id]) != len(ordered_ids):
            raise ValueError(
                f"prior-stage ranking does not match generated configs for {selected_id}"
            )
    return promoted


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
    stage_name: str | None = None,
    config_start: int = 0,
    config_end: int | None = None,
    max_wall_time_seconds: int | None = None,
    resume_from: Path | None = None,
    allow_checkpoint_commit: str | None = None,
    trial_runner: Callable[[dict[str, Any]], dict[str, Any]] = _run_trial_job,
) -> dict[str, Any]:
    root = spec_path.resolve().parents[2]
    spec = load_yaml(spec_path)
    spec_hash = stable_hash(spec, length=64)
    git_commit = _repository_commit(root)
    all_cases = ExperimentGenerator(root / spec["experiment_set"]).build_cases()
    candidate_count = 1 if smoke else int(spec["candidates_per_algorithm"])
    stages = copy.deepcopy(spec["stages"])
    if smoke:
        stages = [{"name": "smoke", "rounds": 2, "seeds": [0], "keep": 1}]
    stage_names = [str(stage["name"]) for stage in stages]
    if stage_name is not None and stage_name not in stage_names:
        raise ValueError(f"unknown tuning stage: {stage_name}")
    if config_start < 0:
        raise ValueError("config_start must be non-negative")
    if config_end is not None and config_end < config_start:
        raise ValueError("config_end must be greater than or equal to config_start")
    if max_wall_time_seconds is not None and max_wall_time_seconds <= 0:
        raise ValueError("max_wall_time_seconds must be positive")
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
    if resume_from is not None:
        source = resume_from.resolve()
        if not source.is_dir():
            raise ValueError("resume_from must be a checkpoint directory")
        if source != output_dir.resolve():
            for name in (
                "trials.jsonl",
                "manifest.json",
                "progress.json",
                "progress.log",
                "rankings_screen.csv",
                "rankings_refine.csv",
                "rankings_validate.csv",
                "rankings_all_stages.csv",
                "best_summary.json",
            ):
                candidate = source / name
                if candidate.exists():
                    shutil.copy2(candidate, output_dir / name)
            for candidate in source.glob("best_config_*.yaml"):
                shutil.copy2(candidate, output_dir / candidate.name)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("schema_version") != TUNING_CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema version does not match")
        if previous_manifest.get("spec_hash") != spec_hash:
            raise ValueError("checkpoint tuning config does not match")
        previous_commit = previous_manifest.get("git_commit")
        if (
            previous_commit != git_commit
            and previous_commit != allow_checkpoint_commit
        ):
            raise ValueError("checkpoint git commit does not match")
        if previous_manifest.get("algorithms") != list(selected_algorithms):
            raise ValueError("checkpoint algorithm selection does not match")

    results_path = output_dir / "trials.jsonl"
    existing = _read_results(results_path)
    campaign_started = time.perf_counter()
    manifest: dict[str, Any] = {
        **plan,
        "schema_version": TUNING_CHECKPOINT_SCHEMA,
        "spec_hash": spec_hash,
        "git_commit": git_commit,
        "resumed_from_git_commit": (
            previous_manifest.get("git_commit")
            if manifest_path.exists()
            and previous_manifest.get("git_commit") != git_commit
            else None
        ),
        "requested_stage": stage_name,
        "config_start": int(config_start),
        "config_end": config_end,
        "max_wall_time_seconds": max_wall_time_seconds,
        "completed": False,
        "stop_reason": None,
        "completed_trial_keys": sum(
            row.get("status") in TERMINAL_TRIAL_STATUSES
            for row in existing.values()
        ),
    }
    _atomic_write_json(manifest_path, manifest)

    all_stage_rankings: list[dict[str, Any]] = []
    overall_total = sum(int(stage["planned_runs"]) for stage in plan["stages"])
    overall_done = sum(
        row.get("status") in TERMINAL_TRIAL_STATUSES
        for row in existing.values()
    )
    stopped_for_time = False
    ran_requested_stage = False
    for stage_index, stage in enumerate(stages):
        current_stage = str(stage["name"])
        if stage_name is not None and current_stage != stage_name:
            if stage_names.index(stage_name) > stage_index:
                prior_rankings = _read_rankings(
                    output_dir / f"rankings_{current_stage}.csv"
                )
                candidates = _promote_candidates(
                    candidates,
                    prior_rankings,
                    selected_algorithms,
                    int(stage["keep"]),
                )
                for row in prior_rankings:
                    row["stage"] = current_stage
                all_stage_rankings.extend(prior_rankings)
            continue
        ran_requested_stage = True
        stage_started = time.perf_counter()
        stage_cases = select_stage_cases(all_cases, spec["scenario_filter"], stage["seeds"])
        all_jobs: list[dict[str, Any]] = []
        selected_job_keys: set[str] = set()
        for selected_id, trial_configs in candidates.items():
            sliced_configs = trial_configs[config_start:config_end]
            selected_hashes = {trial["config_hash"] for trial in sliced_configs}
            for trial in trial_configs:
                for case in stage_cases:
                    identity = {
                        "stage": current_stage, "algorithm_id": selected_id,
                        "trial_id": trial["trial_id"],
                        "trial_config_hash": trial["config_hash"],
                        "case_id": case["metadata"]["case_id"],
                        "scenario": case["metadata"]["case_id"],
                        "case_config_hash": case["metadata"]["config_hash"],
                        "base_seed": case["metadata"]["base_seed"],
                        "distribution": case["environment"]["distribution"],
                    }
                    identity["trial_key"] = _trial_key(identity)
                    job = {
                        "identity": identity,
                        "case": case,
                        "algorithm_config": trial["config"],
                        "rounds": stage["rounds"],
                    }
                    all_jobs.append(job)
                    if trial["config_hash"] in selected_hashes:
                        selected_job_keys.add(identity["trial_key"])
        stage_key_order = [job["identity"]["trial_key"] for job in all_jobs]
        stage_keys = set(stage_key_order)
        stage_rows = [
            existing[key]
            for key in stage_key_order
            if key in existing
            and existing[key].get("status") in TERMINAL_TRIAL_STATUSES
        ]
        jobs = [
            job
            for job in all_jobs
            if job["identity"]["trial_key"] in selected_job_keys
            and (
                job["identity"]["trial_key"] not in existing
                or existing[job["identity"]["trial_key"]].get("status")
                not in TERMINAL_TRIAL_STATUSES
            )
        ]
        stage_total = len(stage_keys)
        stage_done = len(stage_rows)
        _publish_progress(
            output_dir, stage=current_stage,
            stage_done=stage_done, stage_total=stage_total,
            overall_done=overall_done, overall_total=overall_total,
            latest=None, stage_started=stage_started,
        )

        completed_runtimes = [
            float(row.get("runtime_seconds", 0.0))
            for row in stage_rows
            if float(row.get("runtime_seconds", 0.0)) > 0.0
        ]

        def can_start_trial() -> bool:
            if max_wall_time_seconds is None:
                return True
            elapsed = time.perf_counter() - campaign_started
            estimate = max(completed_runtimes, default=30.0)
            return elapsed + estimate < float(max_wall_time_seconds)

        def record(row: dict[str, Any]) -> None:
            nonlocal stage_done, overall_done
            key = str(row["trial_key"])
            existing[key] = row
            _append_durable_jsonl(results_path, row)
            completed_runtimes.append(float(row.get("runtime_seconds", 0.0)))
            if row.get("status") in TERMINAL_TRIAL_STATUSES:
                stage_done += 1
                overall_done += 1
                stage_rows.append(row)
            manifest["completed_trial_keys"] = sum(
                value.get("status") in TERMINAL_TRIAL_STATUSES
                for value in existing.values()
            )
            manifest["active_stage"] = current_stage
            manifest["stage_completed"] = stage_done
            manifest["stage_total"] = stage_total
            _atomic_write_json(manifest_path, manifest)
            _publish_progress(
                output_dir,
                stage=current_stage,
                stage_done=stage_done,
                stage_total=stage_total,
                overall_done=overall_done,
                overall_total=overall_total,
                latest=row,
                stage_started=stage_started,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    output_dir, f"{current_stage} {stage_done}/{stage_total}"
                )

        if int(workers) <= 1:
            for job in jobs:
                if not can_start_trial():
                    stopped_for_time = True
                    break
                record(trial_runner(job))
        else:
            job_iterator = iter(jobs)
            with ProcessPoolExecutor(max_workers=int(workers)) as executor:
                pending: dict[Any, dict[str, Any]] = {}
                while len(pending) < int(workers) and can_start_trial():
                    try:
                        job = next(job_iterator)
                    except StopIteration:
                        break
                    pending[executor.submit(trial_runner, job)] = job
                while pending:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        record(future.result())
                    while len(pending) < int(workers):
                        if not can_start_trial():
                            stopped_for_time = True
                            break
                        try:
                            job = next(job_iterator)
                        except StopIteration:
                            break
                        pending[executor.submit(trial_runner, job)] = job
                    if stopped_for_time and not pending:
                        break

        stage_complete = all(
            key in existing
            and existing[key].get("status") in TERMINAL_TRIAL_STATUSES
            for key in stage_keys
        )
        if not stage_complete:
            manifest["completed"] = False
            manifest["stop_reason"] = (
                "time_budget" if stopped_for_time else "batch_complete"
            )
            manifest["active_stage"] = current_stage
            manifest["stage_completed"] = sum(
                key in existing
                and existing[key].get("status") in TERMINAL_TRIAL_STATUSES
                for key in stage_keys
            )
            manifest["stage_total"] = stage_total
            _atomic_write_json(manifest_path, manifest)
            if checkpoint_callback is not None:
                checkpoint_callback(
                    output_dir,
                    f"{current_stage} stopped: {manifest['stop_reason']}",
                )
            return manifest

        rankings = aggregate_rankings(
            [existing[key] for key in stage_key_order], len(stage_cases)
        )
        for row in rankings:
            row["stage"] = current_stage
        _write_csv(output_dir / f"rankings_{current_stage}.csv", rankings)
        all_stage_rankings.extend(rankings)
        candidates = _promote_candidates(
            candidates, rankings, selected_algorithms, int(stage["keep"])
        )
        manifest["completed_stages"] = [
            *manifest.get("completed_stages", []),
            current_stage,
        ]
        _atomic_write_json(manifest_path, manifest)
        if stage_name is not None:
            break

    if not ran_requested_stage:
        raise ValueError(f"stage was not run: {stage_name}")
    final_stage = stages[-1]["name"]
    final_rankings = [
        row for row in all_stage_rankings if row["stage"] == final_stage and row["rank"] == 1
    ]
    best_summary = {"tuning_plan": plan, "best": {}}
    if final_rankings:
        for row in final_rankings:
            algorithm_id = row["algorithm_id"]
            trial = next(
                trial for trial in candidates[algorithm_id]
                if trial["trial_id"] == row["trial_id"]
            )
            write_yaml(
                output_dir
                / f"best_config_{algorithm_id.removeprefix('eulc_')}.yaml",
                trial["config"],
            )
            best_summary["best"][algorithm_id] = row
        _atomic_write_json(output_dir / "best_summary.json", best_summary)
    _write_csv(output_dir / "rankings_all_stages.csv", all_stage_rankings)
    manifest["completed"] = True
    manifest["stop_reason"] = None
    manifest["active_stage"] = stage_name or final_stage
    _atomic_write_json(manifest_path, manifest)
    if checkpoint_callback is not None:
        checkpoint_callback(output_dir, "tuning completed")
    return best_summary if final_rankings else manifest
