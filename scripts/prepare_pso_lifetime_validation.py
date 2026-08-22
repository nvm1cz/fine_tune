from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _current_algorithm_config(value: dict[str, Any]) -> dict[str, Any]:
    """Adapt archived configs to the current shared-selection schema."""
    config = copy.deepcopy(value)
    protocol = config.get("protocol", {})
    protocol.pop("ch_ratio", None)
    protocol.pop("candidate_ratio", None)
    return config


def prepare(registry_path: Path, output: Path, rounds: int, seeds: int) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    jobs = []
    for scenario_id, record in sorted(registry["scenarios"].items()):
        scenario = record["scenario"]
        density = str(scenario["density"])
        node_count = int(scenario["node_count"])
        experiment = {
            "experiment_set": f"lifetime_validation_{scenario_id}_{rounds}r",
            "base_configs": [
                "configs/base/environment.yaml",
                "configs/base/channel.yaml",
                "configs/base/protocol.yaml",
                "configs/base/execution.yaml",
                "configs/base/objective.yaml",
            ],
            "algorithms": [],
            "mode": "cartesian",
            "deployment_sizes": [scenario["deployment_size_m"]],
            "densities": [density],
            "distributions": [scenario["distribution"]],
            "seeds": list(range(seeds)),
            "node_count_mapping": {
                "x".join(str(int(value)) for value in scenario["deployment_size_m"]): {
                    density: node_count,
                }
            },
            "overrides": {
                "environment": {
                    "initial_energy_j": float(scenario["initial_energy_j"]),
                    "packet_size_bits": int(scenario["packet_size_bits"]),
                },
                "channel": {
                    "transmission_range_m": float(scenario["transmission_range_m"]),
                    "sound_speed_model": "constant",
                    "sound_speed_mps": 1500.0,
                },
                "execution": {
                    "rounds": int(rounds),
                    "stop_on_first_dead": False,
                },
                "output": {
                    "save_round_metrics": True,
                    "save_convergence": False,
                },
            },
            "sweep": {},
        }
        experiment_path = output / f"pso_lifetime_validation__{scenario_id}.yaml"
        _write_json(experiment_path, experiment)

        variants: list[tuple[str, dict[str, Any]]] = []
        if record["selected_source"] == "tuned":
            variants.append(("baseline", _current_algorithm_config(record["baseline_config"])))
        variants.append(("selected", _current_algorithm_config(record["selected_config"])))
        for variant, config in variants:
            algorithm_path = (
                output.parent / "algorithms"
                / f"pso_lifetime_validation__{scenario_id}__{variant}.yaml"
            )
            _write_json(algorithm_path, config)
            jobs.append({
                "scenario_id": scenario_id,
                "density": density,
                "packet_size_bits": int(scenario["packet_size_bits"]),
                "initial_energy_j": float(scenario["initial_energy_j"]),
                "variant": variant,
                "selected_source": record["selected_source"],
                "experiment_config": str(experiment_path),
                "algorithm_config": str(algorithm_path),
                "rounds": int(rounds),
                "seeds": int(seeds),
            })
    campaign = {
        "schema_version": 1,
        "registry": str(registry_path),
        "rounds": int(rounds),
        "paired_seeds": list(range(seeds)),
        "scenario_count": len(registry["scenarios"]),
        "job_count": len(jobs),
        "jobs": jobs,
        "notes": [
            "Medium selected baseline at tuning step 5, so duplicate baseline jobs are omitted.",
            "Lifetime metrics must not be compared using raw initial-state J across scenarios.",
        ],
    }
    _write_json(output / "pso_lifetime_validation_campaign.json", campaign)
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    if args.rounds <= 0 or args.seeds <= 0:
        raise ValueError("rounds and seeds must be positive")
    campaign = prepare(args.registry, args.output_dir, args.rounds, args.seeds)
    print(json.dumps({
        "scenario_count": campaign["scenario_count"],
        "job_count": campaign["job_count"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
