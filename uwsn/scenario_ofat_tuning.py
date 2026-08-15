from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .algorithms.config import load_algorithm_config
from .experiment_config.generation import ExperimentGenerator
from .experiment_config.io import load_yaml
from .ofat_tuning import apply_coordinate, select_cases
from .tuning import (
    TERMINAL_TRIAL_STATUSES,
    _append_durable_jsonl,
    _atomic_write_json,
    _repository_commit,
    _run_trial_job,
    aggregate_rankings,
    stable_hash,
)


SCENARIO_OFAT_SCHEMA = 1


def scenario_key(case: dict[str, Any]) -> str:
    """Return the deployment condition identity, deliberately excluding seed."""
    env, meta, channel = case["environment"], case["metadata"], case["channel"]
    dims = "x".join(str(int(value)) for value in env["deployment_size_m"])
    energy = str(float(env["initial_energy_j"])).replace(".", "p")
    return (
        f"{dims}__{meta['density']}__n{int(env['node_count'])}"
        f"__{env['distribution']}__p{int(env['packet_size_bits'])}"
        f"__e{energy}__r{int(channel['transmission_range_m'])}"
    )


def group_cases_by_scenario(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[scenario_key(case)].append(case)
    return {key: sorted(value, key=lambda row: int(row["metadata"]["base_seed"]))
            for key, value in sorted(grouped.items())}


def _load_jsonl(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[str(row[key_field])] = row
    return rows


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value.get("configs", {}))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_scenario_ofat(
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
    # Serial traversal is intentional: it makes resume order and OFAT state exact.
    del workers
    root = spec_path.resolve().parents[2]
    spec = load_yaml(spec_path)
    if algorithm_id not in spec["algorithms"]:
        raise ValueError(f"unsupported scenario OFAT algorithm: {algorithm_id}")
    stages = {row["name"]: row for row in spec["stages"]}
    if stage_name not in stages:
        raise ValueError(f"unknown scenario OFAT stage: {stage_name}")
    stage = stages[stage_name]
    cases = select_cases(
        ExperimentGenerator(root / spec["experiment_set"]).build_cases(), stage["seeds"]
    )
    groups = group_cases_by_scenario(cases)
    plan = {
        "schema_version": SCENARIO_OFAT_SCHEMA,
        "name": spec["name"],
        "mode": "per_scenario",
        "algorithm_id": algorithm_id,
        "stage": stage_name,
        "scenario_count": len(groups),
        "seeds": list(stage["seeds"]),
        "rounds": int(stage["rounds"]),
        "passes": int(spec["passes"]) if stage["tune_coordinates"] else 0,
        "coordinates": spec["coordinates"][algorithm_id] if stage["tune_coordinates"] else [],
        "execute": execute,
    }
    if not execute:
        return plan

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "scenario_ofat_manifest.json"
    trials_path = output_dir / "scenario_ofat_trials.jsonl"
    history_path = output_dir / "scenario_ofat_history.jsonl"
    configs_path = output_dir / "scenario_best_configs.json"
    registry_json_path = output_dir / f"scenario_registry_{algorithm_id}.json"
    registry_csv_path = output_dir / f"scenario_registry_{algorithm_id}.csv"
    spec_hash = stable_hash(spec, 64)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and (previous["spec_hash"] != spec_hash or previous["algorithm_id"] != algorithm_id):
        raise ValueError("scenario OFAT checkpoint does not match config or algorithm")

    results = _load_jsonl(trials_path, "trial_key")
    history = _load_jsonl(history_path, "history_key")
    configs = _load_state(configs_path)
    stage_order = [row["name"] for row in spec["stages"]]
    previous_stage = stage_order[stage_order.index(stage_name) - 1] if stage_order.index(stage_name) else None
    started = time.perf_counter()
    manifest = {
        **plan,
        "spec_hash": spec_hash,
        "git_commit": _repository_commit(root),
        "completed": False,
        "completed_scenarios": 0,
        "completed_trials": sum(
            row.get("status") in TERMINAL_TRIAL_STATUSES for row in results.values()
        ),
        "stop_reason": None,
        "completed_stages": list(previous.get("completed_stages", [])) if previous else [],
    }
    _atomic_write_json(manifest_path, manifest)

    def time_available() -> bool:
        return max_wall_time_seconds is None or time.perf_counter() - started < max_wall_time_seconds - 60

    registry_rows: list[dict[str, Any]] = []
    for scenario_index, (key, scenario_cases) in enumerate(groups.items()):
        config_state_key = f"{stage_name}|{key}"
        previous_key = f"{previous_stage}|{key}" if previous_stage else ""
        # Reconstruct this stage only from its predecessor plus durable step
        # history.  Starting from a partially updated current-stage config would
        # change earlier OFAT coordinates after a resume.
        current = configs.get(previous_key)
        if current is None:
            current = load_algorithm_config(root / spec["algorithms"][algorithm_id])
        coordinates = spec["coordinates"][algorithm_id]
        steps = (
            [(pass_index, coordinate) for pass_index in range(int(spec["passes"]))
             for coordinate in coordinates]
            if stage["tune_coordinates"]
            else [(0, {"name": "final_config", "values": [None]})]
        )
        last_ranking: dict[str, Any] | None = None
        for pass_index, coordinate in steps:
            coordinate_name = coordinate["name"]
            history_key = f"{stage_name}|{key}|{pass_index}|{coordinate_name}"
            if history_key in history:
                current = history[history_key]["selected_config"]
                last_ranking = history[history_key]["winner"]
                continue
            candidates = []
            for value_index, value in enumerate(coordinate["values"]):
                candidate = (
                    current
                    if coordinate_name == "final_config"
                    else apply_coordinate(current, algorithm_id, coordinate_name, value)
                )
                config_hash = stable_hash(candidate)
                trial_id = (
                    f"{algorithm_id}__{key}__p{pass_index}__{coordinate_name}"
                    f"__v{value_index}__{config_hash}"
                )
                candidates.append((value_index, value, candidate, config_hash, trial_id))

            step_rows: list[dict[str, Any]] = []
            for value_index, value, candidate, config_hash, trial_id in candidates:
                for case in scenario_cases:
                    seed = int(case["metadata"]["base_seed"])
                    trial_key = "|".join(
                        (algorithm_id, stage_name, key, str(pass_index), coordinate_name,
                         config_hash, str(seed))
                    )
                    if trial_key in results and results[trial_key].get("status") in TERMINAL_TRIAL_STATUSES:
                        step_rows.append(results[trial_key])
                        continue
                    if not time_available():
                        manifest.update(
                            stop_reason="time_budget", active_scenario=key,
                            active_pass=pass_index, active_coordinate=coordinate_name,
                        )
                        _atomic_write_json(manifest_path, manifest)
                        if checkpoint_callback:
                            checkpoint_callback(output_dir, f"{stage_name} stopped: time_budget")
                        return manifest
                    env = case["environment"]
                    identity = {
                        "trial_key": trial_key,
                        "stage": stage_name,
                        "algorithm_id": algorithm_id,
                        "trial_id": trial_id,
                        "trial_config_hash": config_hash,
                        "scenario_key": key,
                        "case_id": case["metadata"]["case_id"],
                        "case_config_hash": case["metadata"]["config_hash"],
                        "base_seed": seed,
                        "distribution": env["distribution"],
                        "density": case["metadata"]["density"],
                        "ofat_pass": pass_index,
                        "coordinate": coordinate_name,
                        "coordinate_value_index": value_index,
                        "coordinate_value": value,
                        "initial_energy_j": env["initial_energy_j"],
                        "packet_size_bits": env["packet_size_bits"],
                        "node_count": env["node_count"],
                        "deployment_size_m": env["deployment_size_m"],
                        "transmission_range_m": case["channel"]["transmission_range_m"],
                    }
                    row = trial_runner({
                        "identity": identity,
                        "case": case,
                        "algorithm_config": candidate,
                        "rounds": stage["rounds"],
                    })
                    results[trial_key] = row
                    step_rows.append(row)
                    _append_durable_jsonl(trials_path, row)
                    manifest["completed_trials"] += int(
                        row.get("status") in TERMINAL_TRIAL_STATUSES
                    )
                    manifest.update(
                        active_scenario=key, active_pass=pass_index,
                        active_coordinate=coordinate_name, latest_trial_key=trial_key,
                    )
                    _atomic_write_json(manifest_path, manifest)
                    print(
                        f"[{stage_name}] scenario={scenario_index + 1}/{len(groups)} {key} "
                        f"pass={pass_index + 1} coordinate={coordinate_name} "
                        f"value={value_index + 1}/{len(candidates)} seed={seed} status={row['status']}",
                        flush=True,
                    )
                    if checkpoint_callback and manifest["completed_trials"] % max(1, checkpoint_every) == 0:
                        checkpoint_callback(output_dir, f"{stage_name} {key} {manifest['completed_trials']}")

            rankings = aggregate_rankings(step_rows, len(scenario_cases))
            complete = [row for row in rankings if float(row["completion_rate"]) == 1.0]
            if not complete:
                raise RuntimeError(f"no OFAT value completed every seed for scenario {key}")
            winner = min(complete, key=lambda row: int(row["rank"]))
            winning = next(item for item in candidates if item[4] == winner["trial_id"])
            current = winning[2]
            configs[config_state_key] = current
            _atomic_write_json(configs_path, {
                "schema_version": SCENARIO_OFAT_SCHEMA,
                "algorithm_id": algorithm_id,
                "spec_hash": spec_hash,
                "configs": configs,
            })
            history_row = {
                "history_key": history_key,
                "stage": stage_name,
                "scenario_key": key,
                "pass": pass_index,
                "coordinate": coordinate_name,
                "selected_value": winning[1],
                "selected_config_hash": winning[3],
                "selected_config": current,
                "winner": winner,
                "ranking": rankings,
            }
            history[history_key] = history_row
            _append_durable_jsonl(history_path, history_row)
            last_ranking = winner
            if checkpoint_callback:
                checkpoint_callback(output_dir, f"{stage_name} {key} selected {coordinate_name}")

        manifest["completed_scenarios"] = scenario_index + 1
        if last_ranking is not None:
            env = scenario_cases[0]["environment"]
            registry_rows.append({
                "scenario_key": key,
                "algorithm_id": algorithm_id,
                "stage": stage_name,
                "deployment_size_m": "x".join(map(lambda v: str(int(v)), env["deployment_size_m"])),
                "density": scenario_cases[0]["metadata"]["density"],
                "node_count": int(env["node_count"]),
                "distribution": env["distribution"],
                "packet_size_bits": int(env["packet_size_bits"]),
                "initial_energy_j": float(env["initial_energy_j"]),
                "transmission_range_m": float(scenario_cases[0]["channel"]["transmission_range_m"]),
                "config_hash": stable_hash(current),
                "completed_runs": int(last_ranking["completed_runs"]),
                "median_survival_score": last_ranking["median_survival_score"],
                "mean_residual_energy_auc": last_ranking["mean_residual_energy_auc"],
                "std_residual_energy_auc": last_ranking["std_residual_energy_auc"],
                "mean_packets_delivered": last_ranking["mean_packets_delivered"],
                "mean_average_e2e_delay_s": last_ranking["mean_average_e2e_delay_s"],
            })
        _atomic_write_json(manifest_path, manifest)

    _atomic_write_json(registry_json_path, {
        "schema_version": SCENARIO_OFAT_SCHEMA,
        "algorithm_id": algorithm_id,
        "stage": stage_name,
        "scenarios": registry_rows,
    })
    _write_csv(registry_csv_path, registry_rows)
    manifest.update(completed=True, stop_reason=None, completed_scenarios=len(groups))
    _atomic_write_json(manifest_path, manifest)
    if checkpoint_callback:
        checkpoint_callback(output_dir, f"{stage_name} completed")
    return manifest
