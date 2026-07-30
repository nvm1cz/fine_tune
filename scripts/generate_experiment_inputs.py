from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.experiment_config import ExperimentGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible UWSN experiment cases.")
    parser.add_argument("--experiment-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = ExperimentGenerator(args.experiment_set)
    output_dir = args.output_dir or Path("configs/generated") / generator.spec["experiment_set"]
    cases = generator.build_cases()
    if not args.dry_run:
        generator.write(output_dir, clean=args.clean)
    print(json.dumps(generator.summary(cases, output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
