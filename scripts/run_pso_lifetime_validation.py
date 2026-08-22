from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--density", choices=("sparse", "medium", "dense", "all"), default="all")
    parser.add_argument("--scenario-id")
    parser.add_argument("--variant", choices=("baseline", "selected", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-wall-time-seconds", type=int, default=39600)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    campaign = json.loads(args.campaign.read_text(encoding="utf-8"))
    jobs = [
        job for job in campaign["jobs"]
        if (args.density == "all" or job["density"] == args.density)
        and (args.scenario_id is None or job["scenario_id"] == args.scenario_id)
        and (args.variant == "all" or job["variant"] == args.variant)
    ]
    if not jobs:
        raise ValueError("no lifetime-validation jobs match the requested filters")
    print(json.dumps({
        "job_count": len(jobs),
        "jobs": [f"{job['scenario_id']}::{job['variant']}" for job in jobs],
    }, indent=2), flush=True)
    if args.plan:
        return

    for index, job in enumerate(jobs, 1):
        output = args.output_dir / job["scenario_id"] / job["variant"]
        print(
            f"[lifetime] {index}/{len(jobs)} {job['scenario_id']}::{job['variant']}",
            flush=True,
        )
        manifest = run_benchmark(
            root,
            Path(job["experiment_config"]),
            Path(job["algorithm_config"]),
            output,
            workers=args.workers,
            max_wall_time_seconds=args.max_wall_time_seconds,
        )
        print(json.dumps(manifest, indent=2), flush=True)
        if not manifest.get("completed"):
            print("[lifetime] current job incomplete; stop for safe resume", flush=True)
            return


if __name__ == "__main__":
    main()
