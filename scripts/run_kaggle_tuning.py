from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.tuning import run_tuning


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
    args = parser.parse_args()
    result = run_tuning(
        args.config, args.output_dir, workers=args.workers,
        execute=args.execute, smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
