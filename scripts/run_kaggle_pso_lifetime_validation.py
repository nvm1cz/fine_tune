from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.benchmark import run_benchmark
from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint


ROOT = Path(__file__).resolve().parents[1]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _archive(output: Path) -> None:
    destination = output / "pso_lifetime_results.zip"
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path not in {destination, temporary}:
                archive.write(path, path.relative_to(output))
    temporary.replace(destination)


def _job_completed(output: Path, job: dict[str, Any]) -> bool:
    manifest = output / job["scenario_id"] / job["variant"] / "benchmark_manifest.json"
    if not manifest.exists():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return bool(value.get("completed")) and int(value.get("completed_cases", 0)) == int(job["seeds"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset", required=True)
    parser.add_argument("--density", choices=("sparse", "medium", "dense"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-wall-time-seconds", type=int, default=39600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    jobs = [job for job in campaign["jobs"] if job["density"] == args.density]
    if not args.execute:
        print(json.dumps({"density": args.density, "jobs": jobs}, indent=2))
        return

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = KaggleDatasetCheckpoint(args.checkpoint_dataset)
    checkpoint.restore(output)
    archive_path = output / "pso_lifetime_results.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(output)

    started = time.perf_counter()

    def publish(note: str) -> None:
        completed = [f"{job['scenario_id']}::{job['variant']}" for job in jobs if _job_completed(output, job)]
        active = next(
            (job for job in jobs if not _job_completed(output, job)), None
        )
        state = {
            "schema_version": 1,
            "density": args.density,
            "completed_jobs": completed,
            "completed_count": len(completed),
            "total_jobs": len(jobs),
            "completed": len(completed) == len(jobs),
            "active_job": None if active is None else f"{active['scenario_id']}::{active['variant']}",
        }
        _atomic_json(output / "pso_lifetime_campaign_manifest.json", state)
        _archive(output)
        checkpoint.publish(output, note)

    for index, job in enumerate(jobs, 1):
        if _job_completed(output, job):
            print(f"[resume] {job['scenario_id']}::{job['variant']} complete", flush=True)
            continue
        elapsed = time.perf_counter() - started
        remaining = args.max_wall_time_seconds - elapsed
        if remaining < 900:
            publish(f"{args.density} stopped: time_budget")
            return
        job_output = output / job["scenario_id"] / job["variant"]
        print(f"[lifetime] {index}/{len(jobs)} {job['scenario_id']}::{job['variant']}", flush=True)
        manifest = run_benchmark(
            ROOT,
            Path(job["experiment_config"]),
            Path(job["algorithm_config"]),
            job_output,
            workers=args.workers,
            max_wall_time_seconds=int(remaining),
            checkpoint_callback=lambda _, note: publish(
                f"{args.density} {job['scenario_id']}::{job['variant']} {note}"
            ),
        )
        if not manifest.get("completed"):
            publish(f"{args.density} stopped: current job incomplete")
            return
    publish(f"{args.density} lifetime validation complete")


if __name__ == "__main__":
    main()
