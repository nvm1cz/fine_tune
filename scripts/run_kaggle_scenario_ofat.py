from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint
from uwsn.scenario_ofat_tuning import run_scenario_ofat


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable per-scenario OFAT tuning.")
    parser.add_argument("--config", type=Path, default=Path("configs/tuning/kaggle_scenario_ofat.yaml"))
    parser.add_argument("--algorithm", required=True, choices=("eulc_pso", "eulc_ga", "eulc_ac_aco"))
    parser.add_argument("--stage", required=True, choices=("auto", "screen", "refine", "validate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset")
    parser.add_argument("--max-wall-time-seconds", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    checkpoint = KaggleDatasetCheckpoint(args.checkpoint_dataset) if args.checkpoint_dataset else None
    if checkpoint is not None and args.execute:
        checkpoint.restore(args.output_dir)

    stage = args.stage
    if stage == "auto":
        manifest_path = args.output_dir / "scenario_ofat_manifest.json"
        completed_stages: list[str] = []
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            completed_stages = list(manifest.get("completed_stages", []))
            if manifest.get("completed") and manifest.get("stage") not in completed_stages:
                completed_stages.append(manifest["stage"])
        stage = next((name for name in ("screen", "refine", "validate") if name not in completed_stages), "complete")
        if stage == "complete":
            result = {"completed": True, "stop_reason": "all_stages_complete"}
            print(json.dumps(result, indent=2))
            return

    result = run_scenario_ofat(
        args.config,
        args.output_dir,
        algorithm_id=args.algorithm,
        stage_name=stage,
        execute=args.execute,
        workers=args.workers,
        max_wall_time_seconds=args.max_wall_time_seconds,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=checkpoint.publish if checkpoint else None,
    )
    if args.execute and result.get("completed"):
        completed_stages = list(result.get("completed_stages", []))
        if stage not in completed_stages:
            completed_stages.append(stage)
        result["completed_stages"] = completed_stages
        manifest_path = args.output_dir / "scenario_ofat_manifest.json"
        manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if checkpoint:
            checkpoint.publish(args.output_dir, f"{stage} completed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
