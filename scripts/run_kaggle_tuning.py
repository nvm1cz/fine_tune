from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.tuning import run_tuning
from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable successive-halving tuning for PSO, GA and AC-ACO."
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/tuning/kaggle_large.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/kaggle/working/uwsn_tuning"),
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--algorithm",
        choices=("eulc_pso", "eulc_ga", "eulc_ac_aco"),
        help="Tune only one optimizer; omit to tune all three.",
    )
    parser.add_argument(
        "--checkpoint-dataset",
        help="Kaggle Dataset handle used for cross-session checkpointing.",
    )
    parser.add_argument(
        "--stage",
        choices=("screen", "refine", "validate"),
        help="Run only one successive-halving stage.",
    )
    parser.add_argument("--config-start", type=int, default=0)
    parser.add_argument("--config-end", type=int)
    parser.add_argument("--max-wall-time-seconds", type=int)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Directory containing a compatible manifest and trials.jsonl.",
    )
    parser.add_argument(
        "--allow-checkpoint-commit",
        help="Explicitly allow one prior checkpoint commit for a controlled migration.",
    )
    args = parser.parse_args()
    checkpoint = (
        KaggleDatasetCheckpoint(args.checkpoint_dataset)
        if args.checkpoint_dataset else None
    )
    if checkpoint is not None and args.execute:
        checkpoint.restore(args.output_dir)
    result = run_tuning(
        args.config, args.output_dir, workers=args.workers,
        execute=args.execute, smoke=args.smoke, algorithm_id=args.algorithm,
        checkpoint_callback=checkpoint.publish if checkpoint is not None else None,
        stage_name=args.stage,
        config_start=args.config_start,
        config_end=args.config_end,
        max_wall_time_seconds=args.max_wall_time_seconds,
        resume_from=args.resume_from,
        allow_checkpoint_commit=args.allow_checkpoint_commit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
