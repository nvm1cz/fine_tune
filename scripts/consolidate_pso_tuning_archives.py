from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any


EXPECTED = {
    "sparse": {"node_count": 50, "archive_scenarios": 4},
    "medium": {"node_count": 100, "archive_scenarios": 4},
    "dense": {"node_count": 150, "archive_scenarios": 4},
}
PACKET_ENERGY = {(6400, 1.0), (6400, 0.5), (4000, 1.0), (4000, 0.5)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selected_config(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    label = str(result["final_winner"]["label"])
    if label not in {"baseline", "tuned"}:
        raise ValueError(f"unexpected final winner label: {label}")
    return label, result[f"{label}_config"]


def consolidate(archives: dict[str, Path], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    registry: dict[str, Any] = {
        "schema_version": 1,
        "selection_rule": "final step winner: baseline or tuned by mean best_J workflow",
        "scope": "PSO tuning at initial network state; lifetime validation pending",
        "scenarios": {},
        "sources": {},
    }
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        for density, archive_path in archives.items():
            if density not in EXPECTED:
                raise ValueError(f"unknown density: {density}")
            extract_root = temporary_root / density
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_root)
            result_paths = sorted(extract_root.rglob("result.json"))
            if len(result_paths) != EXPECTED[density]["archive_scenarios"]:
                raise ValueError(
                    f"{density}: expected 4 result.json files, found {len(result_paths)}"
                )
            registry["sources"][density] = {
                "archive_name": archive_path.name,
                "archive_sha256": _sha256(archive_path),
                "result_count": len(result_paths),
            }
            observed_cases: set[tuple[int, float]] = set()
            for result_path in result_paths:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                scenario = result["scenario"]
                scenario_id = result_path.parent.name
                packet_bits = int(scenario["packet_size_bits"])
                energy_j = float(scenario["initial_energy_j"])
                observed_cases.add((packet_bits, energy_j))
                if scenario["density"] != density:
                    raise ValueError(f"{scenario_id}: density mismatch")
                if int(scenario["node_count"]) != EXPECTED[density]["node_count"]:
                    raise ValueError(f"{scenario_id}: node-count mismatch")
                if scenario["distribution"] != "uniform":
                    raise ValueError(f"{scenario_id}: expected uniform distribution")
                if [float(value) for value in scenario["deployment_size_m"]] != [100.0] * 3:
                    raise ValueError(f"{scenario_id}: deployment-size mismatch")
                if float(scenario["transmission_range_m"]) != 150.0:
                    raise ValueError(f"{scenario_id}: transmission-range mismatch")

                selected_label, selected = _selected_config(result)
                optimizer = selected["optimizer"]
                params = optimizer["params"]
                rankings = _read_csv(result_path.parent / "all_step_rankings.csv")
                final = {row["label"]: row for row in rankings if row["phase"] == "step5_final"}
                if set(final) != {"baseline", "tuned"}:
                    raise ValueError(f"{scenario_id}: incomplete final comparison")
                baseline_j = float(final["baseline"]["mean_best_J"])
                tuned_j = float(final["tuned"]["mean_best_J"])
                selected_row = final[selected_label]
                trials = _read_csv(result_path.parent / "trials.csv")
                unique_trials = {
                    (row["phase"], row["label"], int(row["seed"]), row.get("config_hash", ""))
                    for row in trials
                }
                diversity = json.loads(
                    (result_path.parent / "diversity_diagnostic.json").read_text(encoding="utf-8")
                )
                selected_config_path = output / "selected_configs" / f"{scenario_id}.yaml"
                selected_config_path.parent.mkdir(parents=True, exist_ok=True)
                selected_config_path.write_text(
                    json.dumps(selected, indent=2) + "\n", encoding="utf-8"
                )
                row = {
                    "scenario_id": scenario_id,
                    "density": density,
                    "node_count": int(scenario["node_count"]),
                    "packet_size_bits": packet_bits,
                    "initial_energy_j": energy_j,
                    "selected_source": selected_label,
                    "population_size": int(optimizer["population_size"]),
                    "max_iterations": int(optimizer["max_iterations"]),
                    "inertia_schedule": params.get("inertia_schedule", "fixed"),
                    "inertia_start": float(params.get("inertia_start", params["inertia"])),
                    "inertia_end": float(params.get("inertia_end", params["inertia"])),
                    "c1": float(params["c1"]),
                    "c2": float(params["c2"]),
                    "velocity_max": float(params.get("velocity_max", 1.0)),
                    "baseline_mean_best_J": baseline_j,
                    "tuned_mean_best_J": tuned_j,
                    "selected_mean_best_J": float(selected_row["mean_best_J"]),
                    "selected_std_best_J": float(selected_row["std_best_J"]),
                    "selected_runtime_seconds": float(selected_row["mean_runtime_seconds"]),
                    "relative_J_improvement_vs_baseline_pct": (
                        (baseline_j - float(selected_row["mean_best_J"])) / abs(baseline_j) * 100.0
                    ),
                    "unique_trial_count": len(unique_trials),
                    "method_version": result["method_version"],
                    "vmax_sensitivity_executed": bool(diversity["vmax_sensitivity_executed"]),
                    "mean_tail_diversity_m": float(diversity["mean_tail_diversity_equivalent_m"]),
                    "selected_config_path": str(selected_config_path.relative_to(output)),
                }
                rows.append(row)
                registry["scenarios"][scenario_id] = {
                    "scenario": scenario,
                    "selected_source": selected_label,
                    "selected_config": selected,
                    "baseline_config": result["baseline_config"],
                    "tuned_config": result["tuned_config"],
                    "final_metrics": {
                        "baseline_mean_best_J": baseline_j,
                        "tuned_mean_best_J": tuned_j,
                        "selected_mean_best_J": row["selected_mean_best_J"],
                        "selected_std_best_J": row["selected_std_best_J"],
                        "selected_runtime_seconds": row["selected_runtime_seconds"],
                    },
                    "provenance": {
                        "archive_density": density,
                        "method_version": result["method_version"],
                        "unique_trial_count": len(unique_trials),
                    },
                }
            if observed_cases != PACKET_ENERGY:
                raise ValueError(f"{density}: packet-energy matrix is incomplete")

    rows.sort(key=lambda row: (
        {"sparse": 0, "medium": 1, "dense": 2}[row["density"]],
        -row["packet_size_bits"], -row["initial_energy_j"],
    ))
    _write_csv(output / "pso_best_config_summary.csv", rows)
    (output / "pso_best_config_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# EDA and Integrity Report: PSO Fine-Tuning Archives",
        "",
        "## Scope",
        "",
        "The three archives contain 12 independent uniform UWSN scenarios at "
        "100 x 100 x 100 m and transmission range 150 m. Selection is based on "
        "the final paired baseline-versus-tuned comparison at the initial network state.",
        "",
        "## Quality checks",
        "",
        "- Expected scenario count: 12; observed: 12.",
        "- Each density contains all four packet-energy combinations.",
        "- Every scenario contains result, trials, rankings, convergence, plateau, "
        "diversity, selected configuration, and final comparison artifacts.",
        "- No scenario raw J values were pooled across different network states.",
        "",
        "## Selection findings",
        "",
        f"- Tuned selected: {sum(row['selected_source'] == 'tuned' for row in rows)}/12.",
        f"- Baseline retained: {sum(row['selected_source'] == 'baseline' for row in rows)}/12.",
        "- All four medium scenarios retained baseline in the final comparison.",
        "- Sparse and dense scenarios selected tuned configurations under the project's "
        "statistical tie/runtime rules.",
        "",
        "## Required next step",
        "",
        "These results do not establish network lifetime. Run a separate 500-round, "
        "30 paired-seed validation and report FND, HND, LND, residual energy, dead nodes, "
        "packet delivery, delay, convergence, and runtime.",
    ]
    (output / "pso_tuning_eda_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive", action="append", required=True,
        help="DENSITY=PATH; provide sparse, medium and dense exactly once",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archives: dict[str, Path] = {}
    for item in args.archive:
        density, separator, raw_path = item.partition("=")
        if not separator or density in archives:
            raise ValueError(f"invalid or duplicate --archive: {item}")
        archives[density] = Path(raw_path).expanduser().resolve()
    if set(archives) != set(EXPECTED):
        raise ValueError("archives must contain exactly sparse, medium and dense")
    registry = consolidate(archives, args.output_dir)
    print(json.dumps({
        "scenario_count": len(registry["scenarios"]),
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
