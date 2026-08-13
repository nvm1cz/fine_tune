from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint
from uwsn.ofat_tuning import run_ofat


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable full-scenario OFAT optimizer tuning.")
    parser.add_argument("--config", type=Path, default=Path("configs/tuning/kaggle_ofat_full.yaml"))
    parser.add_argument("--algorithm", required=True, choices=("eulc_pso", "eulc_ga", "eulc_ac_aco"))
    parser.add_argument("--stage", required=True, choices=("screen", "refine", "validate"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset")
    parser.add_argument("--max-wall-time-seconds", type=int, default=39600)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    checkpoint = KaggleDatasetCheckpoint(args.checkpoint_dataset) if args.checkpoint_dataset else None
    if checkpoint is not None and args.execute:
        checkpoint.restore(args.output_dir)
    result = run_ofat(
        args.config, args.output_dir, algorithm_id=args.algorithm,
        stage_name=args.stage, execute=args.execute, workers=args.workers,
        max_wall_time_seconds=args.max_wall_time_seconds,
        checkpoint_every=args.checkpoint_every,
        checkpoint_callback=checkpoint.publish if checkpoint else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
