from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _write_manifest(output: Path, density: str, completed: list[str], stop_reason: str | None) -> None:
    payload = {
        "schema_version": 1,
        "density": density,
        "scenario_order": SCENARIOS[density],
        "completed_scenarios": completed,
        "completed_count": len(completed),
        "total_scenarios": len(SCENARIOS[density]),
        "completed": len(completed) == len(SCENARIOS[density]),
        "stop_reason": stop_reason,
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
    parser.add_argument("--max-wall-time-seconds", type=int, default=39600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({"density": args.density, "scenarios": SCENARIOS[args.density]}, indent=2))
        return

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = KaggleDatasetCheckpoint(args.checkpoint_dataset)
    checkpoint.restore(output)
    archive_path = output / "pso_matrix_results.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output)

    started = time.perf_counter()
    completed = [
        scenario for scenario in SCENARIOS[args.density]
        if (output / scenario / "result.json").exists()
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
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/run_local_pso_uniform_matrix_tuning.py"),
            "--config", str(args.config),
            "--output-dir", str(output),
            "--scenario-id", scenario,
        ], cwd=ROOT, check=True)
        completed.append(scenario)
        _write_manifest(output, args.density, completed, None)
        _archive(output)
        checkpoint.publish(output, f"{args.density} completed {len(completed)}/4: {scenario}")

    _write_manifest(output, args.density, completed, None)
    _archive(output)
    checkpoint.publish(output, f"{args.density} complete")


if __name__ == "__main__":
    main()
