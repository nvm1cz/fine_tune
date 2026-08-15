from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ALGORITHMS = ("eulc_pso", "eulc_ga", "eulc_ac_aco")


def _load_algorithm(directory: Path, algorithm_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = json.loads(
        (directory / f"scenario_registry_{algorithm_id}.json").read_text(encoding="utf-8")
    )
    configs = json.loads(
        (directory / "scenario_best_configs.json").read_text(encoding="utf-8")
    )["configs"]
    if registry["stage"] != "validate":
        raise ValueError(f"{algorithm_id} registry is not from validate stage")
    return list(registry["scenarios"]), configs


def _score(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["completed_runs"]),
        -float(row["median_survival_score"]),
        -float(row["mean_residual_energy_auc"]),
        float(row["std_residual_energy_auc"]),
        -float(row["mean_packets_delivered"]),
        float(row["mean_average_e2e_delay_s"]),
        str(row["algorithm_id"]),
    )


def build_registry(inputs: dict[str, Path], output_dir: Path) -> list[dict[str, Any]]:
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    configs_by_algorithm: dict[str, dict[str, Any]] = {}
    for algorithm_id in ALGORITHMS:
        rows, configs = _load_algorithm(inputs[algorithm_id], algorithm_id)
        configs_by_algorithm[algorithm_id] = configs
        for row in rows:
            by_scenario.setdefault(str(row["scenario_key"]), []).append(row)

    if len(by_scenario) != 144:
        raise ValueError(f"expected 144 scenarios, found {len(by_scenario)}")
    winners = []
    for key, rows in sorted(by_scenario.items()):
        if {row["algorithm_id"] for row in rows} != set(ALGORITHMS):
            raise ValueError(f"scenario {key} does not contain all three algorithms")
        winner = min(rows, key=_score)
        algorithm_id = str(winner["algorithm_id"])
        winners.append({
            **winner,
            "best_algorithm": algorithm_id,
            "best_config": configs_by_algorithm[algorithm_id][f"validate|{key}"],
            "algorithm_comparison": sorted(rows, key=_score),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scenario_algorithm_registry.json").write_text(
        json.dumps({"schema_version": 1, "scenario_count": 144, "scenarios": winners},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_rows = [
        {key: value for key, value in row.items()
         if key not in {"best_config", "algorithm_comparison"}}
        for row in winners
    ]
    with (output_dir / "scenario_algorithm_registry.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return winners


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the validated optimizer for each UWSN scenario.")
    parser.add_argument("--pso", type=Path, required=True)
    parser.add_argument("--ga", type=Path, required=True)
    parser.add_argument("--ac-aco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    winners = build_registry(
        {"eulc_pso": args.pso, "eulc_ga": args.ga, "eulc_ac_aco": args.ac_aco},
        args.output_dir,
    )
    print(f"Wrote {len(winners)} scenario decisions to {args.output_dir}")


if __name__ == "__main__":
    main()
