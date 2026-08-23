from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.experiment_config.generation import ExperimentGenerator
from uwsn.experiment_config.io import load_yaml
from uwsn.experiment_config.runner import build_simulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--algorithm", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--modes", nargs="+",
        choices=("periodic", "cluster_energy_mean"),
        default=["periodic", "cluster_energy_mean"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = ExperimentGenerator(args.experiment).build_cases()
    by_seed = {int(case["metadata"]["base_seed"]): case for case in cases}
    raw_algorithm = load_yaml(args.algorithm)
    rows = []
    for seed in args.seeds:
        if seed not in by_seed:
            raise ValueError(f"seed {seed} not present in experiment")
        for mode in args.modes:
            algorithm = copy.deepcopy(raw_algorithm)
            algorithm["protocol"]["recluster_trigger_mode"] = mode
            simulator, execution = build_simulator(copy.deepcopy(by_seed[seed]), algorithm)
            started = time.perf_counter()
            metrics = simulator.run(
                stop_on_first_dead=False,
                max_rounds=min(args.rounds, int(execution["rounds"])),
            )
            runtime = time.perf_counter() - started
            reasons = Counter(
                str(event["reoptimization_reason"])
                for event in simulator.optimization_events
            )
            route_diagnostics = (
                dict(simulator.current_route_plan.diagnostics)
                if simulator.current_route_plan is not None else {}
            )
            rows.append({
                "seed": seed,
                "mode": mode,
                "rounds_executed": len(metrics.residual_energy_by_round),
                "optimizer_refreshes": len(simulator.optimization_events),
                "runtime_seconds": runtime,
                "fnd_round": metrics.fnd_round,
                "hnd_round": metrics.hnd_round,
                "lnd_round": metrics.lnd_round,
                "final_residual_energy_j": metrics.residual_energy,
                "objective_J": (
                    simulator.current_objective_result.objective_J
                    if simulator.current_objective_result is not None else None
                ),
                "route_plans_evaluated_for_selected_solution": (
                    route_diagnostics.get("plans_evaluated")
                ),
                "routing_search_mode": route_diagnostics.get("routing_search_mode"),
                "periodic_refreshes": reasons.get("periodic", 0),
                "energy_mean_refreshes": reasons.get(
                    "cluster_head_below_cluster_mean", 0
                ),
                "other_refreshes": sum(reasons.values())
                - reasons.get("periodic", 0)
                - reasons.get("cluster_head_below_cluster_mean", 0),
            })
            print(json.dumps(rows[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
