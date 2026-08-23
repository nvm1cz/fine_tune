from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-campaign", type=Path, required=True)
    parser.add_argument("--density", choices=("sparse", "medium", "dense"), required=True)
    parser.add_argument("--output-campaign", type=Path, required=True)
    parser.add_argument("--algorithm-dir", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_campaign.read_text(encoding="utf-8"))
    selected_jobs = [
        job for job in source["jobs"]
        if job["density"] == args.density and job["variant"] == "selected"
    ]
    if len(selected_jobs) != 4:
        raise ValueError(f"expected four selected jobs, found {len(selected_jobs)}")

    jobs = []
    for job in selected_jobs:
        config = json.loads(Path(job["algorithm_config"]).read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["protocol"]["recluster_trigger_mode"] = "cluster_energy_mean"
        destination = (
            args.algorithm_dir
            / f"pso_energy_maintenance__{job['scenario_id']}.yaml"
        )
        _write(destination, config)
        updated = dict(job)
        updated["variant"] = "energy_maintenance"
        updated["algorithm_config"] = str(destination)
        updated["recluster_trigger_mode"] = "cluster_energy_mean"
        jobs.append(updated)

    campaign = {
        "schema_version": 1,
        "source_campaign": str(args.source_campaign),
        "density": args.density,
        "rounds": source["rounds"],
        "paired_seeds": source["paired_seeds"],
        "scenario_count": 4,
        "job_count": 4,
        "recluster_trigger_mode": "cluster_energy_mean",
        "reference": {
            "citation": (
                "Yi et al., Non-Uniform Clustering Algorithm for UWSNs Based "
                "on Energy Equalization, Sensors 23(12), 5466, 2023"
            ),
            "doi": "10.3390/s23125466",
            "adaptation": (
                "Per-cluster energy trigger with global joint EULC+PSO refresh "
                "to preserve assignment and routing consistency."
            ),
        },
        "jobs": jobs,
    }
    _write(args.output_campaign, campaign)
    print(json.dumps({
        "campaign": str(args.output_campaign),
        "jobs": [f"{job['scenario_id']}::{job['variant']}" for job in jobs],
    }, indent=2))


if __name__ == "__main__":
    main()
