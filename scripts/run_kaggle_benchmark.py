from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.benchmark import run_benchmark
from uwsn.kaggle_checkpoint import KaggleDatasetCheckpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable held-out UWSN benchmark")
    parser.add_argument("--algorithm", choices=("eulc_pso", "eulc_ga"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dataset", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-wall-time-seconds", type=int, default=39600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = KaggleDatasetCheckpoint(args.checkpoint_dataset)
    checkpoint.restore(args.output_dir)
    algorithm_config = root / "configs/benchmarks" / f"best_{args.algorithm}.yaml"
    result = run_benchmark(
        root,
        root / "configs/experiment_sets/benchmark_heldout.yaml",
        algorithm_config,
        args.output_dir,
        workers=args.workers,
        max_wall_time_seconds=args.max_wall_time_seconds,
        checkpoint_callback=checkpoint.publish,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
