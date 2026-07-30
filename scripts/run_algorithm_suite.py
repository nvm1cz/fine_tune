from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.algorithms.registry import CANONICAL_ALGORITHM_IDS, canonical_algorithm_id
from uwsn.experiment_config.io import load_yaml
from uwsn.experiment_config.runner import run_case_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated, resumable algorithm suite.")
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--algorithms", nargs="+", default=list(CANONICAL_ALGORITHM_IDS))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    case = load_yaml(args.case)
    root = (
        Path(case["output"]["root_dir"])
        / case["metadata"]["experiment_set"]
        / case["metadata"]["case_id"]
    )
    failures = 0
    for requested_id in args.algorithms:
        algorithm_id = canonical_algorithm_id(requested_id)
        output_dir = root / algorithm_id
        summary_path = output_dir / "summary.json"
        if not args.no_resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("status") == "completed"
                and summary.get("config_hash") == case["metadata"]["config_hash"]
            ):
                print(f"SKIP {algorithm_id}: completed with matching config hash")
                continue
        try:
            result = run_case_file(args.case, algorithm_id=algorithm_id)
            print(f"OK {algorithm_id}: {result}")
        except Exception as exc:  # isolate one algorithm failure from the suite
            failures += 1
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "run.log").write_text(
                f"status=failed\nerror={type(exc).__name__}: {exc}\n"
                + traceback.format_exc(),
                encoding="utf-8",
            )
            print(f"FAILED {algorithm_id}: {exc}")
    if failures:
        print(f"Suite completed with {failures} failed algorithm(s)")


if __name__ == "__main__":
    main()
