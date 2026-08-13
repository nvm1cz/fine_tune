from __future__ import annotations

import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from .algorithms.config import load_algorithm_config
from .experiment_config.generation import ExperimentGenerator
from .experiment_config.io import load_yaml, write_yaml
from .tuning import (
    TERMINAL_TRIAL_STATUSES,
    _atomic_write_json,
    _append_durable_jsonl,
    _repository_commit,
    _run_trial_job,
    aggregate_rankings,
    build_trial_config,
    stable_hash,
)

OFAT_SCHEMA = 1


def apply_coordinate(config: dict[str, Any], algorithm_id: str, name: str, value: Any) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    if name == "budget_pair":
        sample["budget_pair"] = value
    else:
        sample["budget_pair"] = [
            config["optimizer"]["population_size"],
            config["optimizer"]["max_iterations"],
        ]
        if name == "beta_schedule":
            sample.update(beta_start=value[0], beta_end=value[1])
        elif name == "rho_schedule":
            sample.update(rho_start=value[0], rho_end=value[1])
        elif name == "heuristic_weights":
            sample.update(
                heuristic_energy_weight=value[0],
                heuristic_eulc_weight=value[1],
                heuristic_centrality_weight=value[2],
            )
        else:
            sample[name] = value
    return build_trial_config(config, algorithm_id, sample)


def select_cases(cases: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    wanted = set(map(int, seeds))
    return [copy.deepcopy(case) for case in cases if int(case["metadata"]["base_seed"]) in wanted]


def _case_identity(case: dict[str, Any]) -> str:
    # case_id intentionally omits energy/packet sweeps; the config hash does not.
    return f'{case["metadata"]["case_id"]}__{case["metadata"]["config_hash"]}'


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["trial_key"]] = row
    return rows


def _winner(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rankings if float(row["completion_rate"]) == 1.0]
    if not complete:
        raise RuntimeError("no OFAT value completed every required scenario")
    return min(complete, key=lambda row: int(row["rank"]))


def run_ofat(
    spec_path: Path,
    output_dir: Path,
    *,
    algorithm_id: str,
    stage_name: str,
    execute: bool = False,
    workers: int = 1,
    max_wall_time_seconds: int | None = None,
    checkpoint_every: int = 10,
    checkpoint_callback: Callable[[Path, str], None] | None = None,
    trial_runner: Callable[[dict[str, Any]], dict[str, Any]] = _run_trial_job,
) -> dict[str, Any]:
    # workers is reserved for CLI compatibility; deterministic OFAT ordering is serial.
    del workers
    root = spec_path.resolve().parents[2]
    spec = load_yaml(spec_path)
    if algorithm_id not in spec["algorithms"]:
        raise ValueError(f"unsupported OFAT algorithm: {algorithm_id}")
    stages = {row["name"]: row for row in spec["stages"]}
    if stage_name not in stages:
        raise ValueError(f"unknown OFAT stage: {stage_name}")
    stage = stages[stage_name]
    all_cases = ExperimentGenerator(root / spec["experiment_set"]).build_cases()
    cases = select_cases(all_cases, stage["seeds"])
    coordinates = spec["coordinates"][algorithm_id]
    plan = {
        "schema_version": OFAT_SCHEMA,
        "name": spec["name"],
        "algorithm_id": algorithm_id,
        "stage": stage_name,
        "passes": int(spec["passes"]) if stage["tune_coordinates"] else 0,
        "coordinates": coordinates if stage["tune_coordinates"] else [],
        "cases_per_value": len(cases),
        "environment_combinations_per_seed": len(cases) // len(stage["seeds"]),
        "seeds": stage["seeds"],
        "rounds": stage["rounds"],
        "execute": execute,
    }
    if not execute:
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ofat_manifest.json"
    trials_path = output_dir / "ofat_trials.jsonl"
    spec_hash = stable_hash(spec, 64)
    commit = _repository_commit(root)
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if prior and (prior["spec_hash"] != spec_hash or prior["algorithm_id"] != algorithm_id):
        raise ValueError("OFAT checkpoint does not match config or algorithm")
    stage_order = [row["name"] for row in spec["stages"]]
    current_path = output_dir / f"best_config_{stage_name}.yaml"
    if not current_path.exists() and stage_order.index(stage_name) > 0:
        current_path = output_dir / f"best_config_{stage_order[stage_order.index(stage_name) - 1]}.yaml"
    current = (
        load_algorithm_config(current_path)
        if current_path.exists()
        else load_algorithm_config(root / spec["algorithms"][algorithm_id])
    )
    results = _load_rows(trials_path)
    started = time.perf_counter()
    manifest = {
        **plan, "spec_hash": spec_hash, "git_commit": commit,
        "completed": False, "stop_reason": None,
        "completed_trials": sum(r.get("status") in TERMINAL_TRIAL_STATUSES for r in results.values()),
    }
    _atomic_write_json(manifest_path, manifest)

    def time_available() -> bool:
        return max_wall_time_seconds is None or time.perf_counter() - started < max_wall_time_seconds - 60

    steps = []
    if stage["tune_coordinates"]:
        steps = [(p, coordinate) for p in range(int(spec["passes"])) for coordinate in coordinates]
    else:
        steps = [(0, {"name": "final_config", "values": [None]})]

    for pass_index, coordinate in steps:
        coordinate_name = coordinate["name"]
        candidates = []
        for value_index, value in enumerate(coordinate["values"]):
            config = current if coordinate_name == "final_config" else apply_coordinate(
                current, algorithm_id, coordinate_name, value
            )
            config_hash = stable_hash(config)
            trial_id = f"{algorithm_id}__p{pass_index}__{coordinate_name}__v{value_index}__{config_hash}"
            candidates.append((value_index, value, config, config_hash, trial_id))

        step_rows: list[dict[str, Any]] = []
        for value_index, value, config, config_hash, trial_id in candidates:
            for case in cases:
                scenario = _case_identity(case)
                trial_key = "|".join((algorithm_id, stage_name, str(pass_index), coordinate_name, config_hash, scenario))
                if trial_key in results and results[trial_key].get("status") in TERMINAL_TRIAL_STATUSES:
                    step_rows.append(results[trial_key])
                    continue
                if not time_available():
                    manifest.update(stop_reason="time_budget", active_pass=pass_index, active_coordinate=coordinate_name)
                    _atomic_write_json(manifest_path, manifest)
                    if checkpoint_callback:
                        checkpoint_callback(output_dir, f"{stage_name} stopped: time_budget")
                    return manifest
                identity = {
                    "trial_key": trial_key, "stage": stage_name, "algorithm_id": algorithm_id,
                    "trial_id": trial_id, "trial_config_hash": config_hash,
                    "case_id": case["metadata"]["case_id"], "scenario": scenario,
                    "case_config_hash": case["metadata"]["config_hash"],
                    "base_seed": case["metadata"]["base_seed"],
                    "distribution": case["environment"]["distribution"],
                    "ofat_pass": pass_index, "coordinate": coordinate_name,
                    "coordinate_value_index": value_index, "coordinate_value": value,
                    "initial_energy_j": case["environment"]["initial_energy_j"],
                    "packet_size_bits": case["environment"]["packet_size_bits"],
                    "node_count": case["environment"]["node_count"],
                    "deployment_size_m": case["environment"]["deployment_size_m"],
                }
                row = trial_runner({"identity": identity, "case": case, "algorithm_config": config, "rounds": stage["rounds"]})
                results[trial_key] = row
                step_rows.append(row)
                _append_durable_jsonl(trials_path, row)
                manifest["completed_trials"] += int(row.get("status") in TERMINAL_TRIAL_STATUSES)
                manifest.update(active_pass=pass_index, active_coordinate=coordinate_name, latest_trial_key=trial_key)
                _atomic_write_json(manifest_path, manifest)
                print(f"[{stage_name}] pass={pass_index+1} coordinate={coordinate_name} value={value_index+1}/{len(candidates)} case={len(step_rows)} status={row['status']}", flush=True)
                if checkpoint_callback and manifest["completed_trials"] % max(1, int(checkpoint_every)) == 0:
                    checkpoint_callback(output_dir, f"{stage_name} p{pass_index+1} {coordinate_name} {manifest['completed_trials']}")

        rankings = aggregate_rankings(step_rows, len(cases))
        winner = _winner(rankings)
        winning = next(item for item in candidates if item[4] == winner["trial_id"])
        current = winning[2]
        write_yaml(output_dir / f"best_config_{stage_name}.yaml", current)
        history_path = output_dir / "ofat_history.jsonl"
        _append_durable_jsonl(history_path, {
            "stage": stage_name, "pass": pass_index, "coordinate": coordinate_name,
            "selected_value": winning[1], "selected_config_hash": winning[3],
            "ranking": rankings,
        })
        manifest.update(selected_config_hash=winning[3], selected_coordinate=coordinate_name, selected_value=winning[1])
        _atomic_write_json(manifest_path, manifest)
        if checkpoint_callback:
            checkpoint_callback(output_dir, f"{stage_name} selected {coordinate_name}")

    manifest.update(completed=True, stop_reason=None)
    _atomic_write_json(manifest_path, manifest)
    if checkpoint_callback:
        checkpoint_callback(output_dir, f"{stage_name} completed")
    return manifest
