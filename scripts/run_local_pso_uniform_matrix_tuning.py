from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.experiment_config.io import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def _is_exact_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == 3
        and result.get("method_version") == "exact_sequential_pso_initial_state_v1"
        and result.get("scope") == "optimizer tuning at initial network state only"
    )


def _scenario_id(density: str, nodes: int, packet_bits: int, energy_j: float) -> str:
    energy = str(float(energy_j)).replace(".", "p")
    return f"{density}_n{nodes}_p{packet_bits}_e{energy}"


def _scenario_config(matrix: dict[str, Any], density: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    optimizer = matrix["optimizer"]
    return {
        "name": _scenario_id(
            density["name"], int(density["node_count"]),
            int(case["packet_size_bits"]), float(case["initial_energy_j"]),
        ),
        "scenario": {
            "deployment_size_m": matrix["deployment_size_m"],
            "density": density["name"],
            "node_count": int(density["node_count"]),
            "distribution": matrix["distribution"],
            "packet_size_bits": int(case["packet_size_bits"]),
            "initial_energy_j": float(case["initial_energy_j"]),
            "transmission_range_m": float(matrix["transmission_range_m"]),
        },
        "optimizer": {
            "particle_candidates": optimizer["particle_candidates"],
            "initial_max_iterations": int(optimizer["initial_max_iterations"]),
            "screen_seeds": int(optimizer["screen_seeds"]),
            "screen_seed_start": int(optimizer.get("screen_seed_start", 0)),
            "final_seeds": int(optimizer["final_seeds"]),
            "final_seed_start": int(optimizer.get("final_seed_start", 10)),
            "workers": int(optimizer["workers"]),
            "simulation_rounds_per_run": int(matrix["max_simulation_rounds"]),
            "w_schedules": optimizer["w_schedules"],
            "c_pairs": optimizer["c_pairs"],
            "velocity_max_values": optimizer["velocity_max_values"],
            "plateau_sensitivity": optimizer["plateau_sensitivity"],
            "plateau_buffer_fraction": optimizer["plateau_buffer_fraction"],
            "diversity_tail_iterations": optimizer["diversity_tail_iterations"],
            "diversity_diagonal_fraction": optimizer["diversity_diagonal_fraction"],
            "population_candidates_step3": optimizer["population_candidates_step3"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe local PSO tuning for 12 uniform scenarios.")
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/tuning/local_pso_uniform_matrix.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/local_pso_uniform_matrix",
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--density", choices=("sparse", "medium", "dense"))
    parser.add_argument("--scenario-id")
    args = parser.parse_args()

    matrix = load_yaml(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = args.output_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        _scenario_config(matrix, density, case)
        for density in matrix["densities"]
        for case in matrix["packet_energy_cases"]
    ]
    if args.density:
        scenarios = [row for row in scenarios if row["scenario"]["density"] == args.density]
    if args.scenario_id:
        scenarios = [row for row in scenarios if row["name"] == args.scenario_id]
        if not scenarios:
            raise ValueError(f"unknown scenario id: {args.scenario_id}")
    print(json.dumps({
        "scenario_count": len(scenarios),
        "scenario_ids": [scenario["name"] for scenario in scenarios],
        "tuning_scope": "initial_network_state_only",
        "max_simulation_rounds": matrix["max_simulation_rounds"],
    }, indent=2), flush=True)
    if args.plan:
        return

    for index, scenario in enumerate(scenarios, 1):
        scenario_id = scenario["name"]
        scenario_output = args.output_dir / scenario_id
        result_path = scenario_output / "result.json"
        if _is_exact_result(result_path):
            print(f"[matrix] {index}/{len(scenarios)} {scenario_id} already complete; skipping", flush=True)
            continue
        config_path = generated_dir / f"{scenario_id}.yaml"
        config_path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
        print(f"[matrix] {index}/{len(scenarios)} starting {scenario_id}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_pso_exact_sequential_tuning.py"),
                "--config", str(config_path),
                "--output-dir", str(scenario_output),
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"[matrix] {index}/{len(scenarios)} completed {scenario_id}", flush=True)


if __name__ == "__main__":
    main()
