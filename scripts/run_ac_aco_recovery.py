from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint
from uwsn.recovery import run_ac_aco_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description="AC-ACO held-out recovery ablation")
    parser.add_argument("--source-benchmark-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--chaos-states", type=float, nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--max-wall-time-seconds", type=int, default=39600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = KaggleDatasetCheckpoint(args.checkpoint_dataset)
    checkpoint.restore(args.output_dir)
    result = run_ac_aco_recovery(
        root,
        args.source_benchmark_dir,
        args.output_dir,
        seeds=args.seeds,
        distribution=args.distribution,
        chaos_states=args.chaos_states,
        budget=args.budget,
        max_wall_time_seconds=args.max_wall_time_seconds,
        checkpoint_callback=checkpoint.publish,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
