from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_local_pso_uniform_matrix_tuning import _scenario_config
from scripts.run_pso_exact_sequential_tuning import METHOD_VERSION, run_exact
from uwsn.experiment_config.io import load_yaml
from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "sparse": [
        "sparse_n50_p6400_e1p0", "sparse_n50_p6400_e0p5",
        "sparse_n50_p4000_e1p0", "sparse_n50_p4000_e0p5",
    ],
    "medium": [
        "medium_n100_p6400_e1p0", "medium_n100_p6400_e0p5",
        "medium_n100_p4000_e1p0", "medium_n100_p4000_e0p5",
    ],
    "dense": [
        "dense_n150_p6400_e1p0", "dense_n150_p6400_e0p5",
        "dense_n150_p4000_e1p0", "dense_n150_p4000_e0p5",
    ],
}


def _is_exact_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == 3
        and result.get("method_version") in {
            "exact_sequential_pso_initial_state_v1", METHOD_VERSION,
        }
        and result.get("scope") == "optimizer tuning at initial network state only"
    )


def _write_manifest(
    output: Path,
    density: str,
    completed: list[str],
    stop_reason: str | None,
    active_progress: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "density": density,
        "scenario_order": SCENARIOS[density],
        "completed_scenarios": completed,
        "completed_count": len(completed),
        "total_scenarios": len(SCENARIOS[density]),
        "completed": len(completed) == len(SCENARIOS[density]),
        "stop_reason": stop_reason,
        "active_progress": active_progress,
    }
    temporary = output / "pso_matrix_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "pso_matrix_manifest.json")


def _archive(output: Path) -> None:
    destination = output / "pso_matrix_results.zip"
    temporary = output / "pso_matrix_results.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file() or path in {destination, temporary}:
                continue
            if path.name == "pso_matrix_manifest.json.tmp":
                continue
            archive.write(path, path.relative_to(output))
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed Kaggle PSO per-case tuning.")
    parser.add_argument("--density", required=True, choices=tuple(SCENARIOS))
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/tuning/local_pso_uniform_matrix.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset", required=True)
    parser.add_argument(
        "--restore-version", type=int,
        help="One-time recovery: restore this historical Dataset version, then publish to the base Dataset.",
    )
    parser.add_argument("--max-wall-time-seconds", type=int, default=39600)
    parser.add_argument("--checkpoint-every-trials", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({"density": args.density, "scenarios": SCENARIOS[args.density]}, indent=2))
        return

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = KaggleDatasetCheckpoint(args.checkpoint_dataset)
    checkpoint.restore(output, version=args.restore_version)
    archive_path = output / "pso_matrix_results.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output)

    started = time.perf_counter()
    matrix = load_yaml(args.config)
    completed = [
        scenario for scenario in SCENARIOS[args.density]
        if _is_exact_result(output / scenario / "result.json")
    ]
    _write_manifest(output, args.density, completed, None)

    for scenario in SCENARIOS[args.density]:
        if scenario in completed:
            print(f"[resume] {scenario} already complete", flush=True)
            continue
        if time.perf_counter() - started >= args.max_wall_time_seconds - 900:
            _write_manifest(output, args.density, completed, "time_budget")
            _archive(output)
            checkpoint.publish(output, f"{args.density} stopped: time_budget")
            return
        print(f"[start] {scenario}", flush=True)
        specs = [
            _scenario_config(matrix, density, case)
            for density in matrix["densities"]
            for case in matrix["packet_energy_cases"]
        ]
        spec = next(row for row in specs if row["name"] == scenario)
        generated_dir = output / "generated_configs"
        generated_dir.mkdir(parents=True, exist_ok=True)
        config_path = generated_dir / f"{scenario}.yaml"
        config_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        callback_count = 0

        class TimeBudgetReached(Exception):
            pass

        def publish_trial_progress(progress: dict[str, Any]) -> None:
            nonlocal callback_count
            callback_count += 1
            near_limit = time.perf_counter() - started >= args.max_wall_time_seconds - 900
            should_publish = (
                progress.get("completed")
                or near_limit
                or callback_count % max(1, args.checkpoint_every_trials) == 0
            )
            if should_publish:
                _write_manifest(output, args.density, completed, None, progress)
                _archive(output)
                active = progress.get("active_progress") or {}
                note = (
                    f"{args.density} {scenario}: "
                    f"{active.get('phase', 'complete')} / "
                    f"{active.get('label', 'complete')} / "
                    f"seed {active.get('seed', '-')}"
                )
                checkpoint.publish(output, note)
            if near_limit:
                raise TimeBudgetReached

        try:
            run_exact(
                config_path,
                output / scenario,
                checkpoint_callback=publish_trial_progress,
            )
        except TimeBudgetReached:
            progress_path = output / scenario / "resume_state.json"
            active = json.loads(progress_path.read_text(encoding="utf-8"))
            _write_manifest(output, args.density, completed, "time_budget", active)
            _archive(output)
            checkpoint.publish(output, f"{args.density} stopped: time_budget")
            print(f"[stop] {scenario} checkpointed at trial level", flush=True)
            return
        completed.append(scenario)
        _write_manifest(output, args.density, completed, None)
        _archive(output)
        checkpoint.publish(output, f"{args.density} completed {len(completed)}/4: {scenario}")

    _write_manifest(output, args.density, completed, None)
    _archive(output)
    checkpoint.publish(output, f"{args.density} complete")


if __name__ == "__main__":
    main()
