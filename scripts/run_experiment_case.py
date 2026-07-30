from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.experiment_config.runner import run_case_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one generated UWSN experiment case.")
    parser.add_argument("--config", "--case", dest="config", required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--algorithm", type=Path)
    group.add_argument("--algorithm-id")
    args = parser.parse_args()
    print(run_case_file(
        args.config, algorithm_path=args.algorithm, algorithm_id=args.algorithm_id
    ))


if __name__ == "__main__":
    main()
