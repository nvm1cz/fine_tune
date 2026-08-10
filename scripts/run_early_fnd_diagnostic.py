from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uwsn.algorithms.config import load_algorithm_config
from uwsn.diagnostic import capture_round_energy, selected_ch_json
from uwsn.experiment_config.generation import ExperimentGenerator
from uwsn.experiment_config.runner import build_simulator


ROUND_FIELDS = [
    "scenario_id", "initial_energy_j", "packet_size_bits", "algorithm", "seed",
    "distribution", "round", "alive_nodes", "dead_nodes",
    "selected_ch_ids", "min_residual_energy_j", "mean_residual_energy_j",
    "max_residual_energy_j", "std_residual_energy_j", "energy_range_j", "energy_cv",
    "lowest_5_nodes_mean_energy_j", "highest_5_nodes_mean_energy_j",
    "member_tx_energy_j", "ch_rx_energy_j", "aggregation_energy_j",
    "routing_tx_energy_j", "routing_rx_energy_j", "retransmission_energy_j",
    "control_energy_j", "total_round_energy_j", "cumulative_energy_consumed_j",
    "packet_delivery_ratio", "average_e2e_delay_s", "objective_J", "objective_e",
    "objective_d", "fnd_reached", "hnd_reached", "lnd_reached",
]
NODE_FIELDS = [
    "scenario_id", "initial_energy_j", "packet_size_bits", "algorithm", "seed",
    "round", "node_id", "is_alive", "is_ch",
    "is_relay_if_available", "residual_energy_j", "energy_consumed_this_round_j",
    "member_tx_energy_j", "rx_energy_j", "aggregation_energy_j",
    "forwarding_energy_j", "retransmission_energy_j",
]


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _flush(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only early-FND diagnostic")
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic_early_fnd"))
    args = parser.parse_args()
    cases = ExperimentGenerator(ROOT / "configs/experiment_sets/diagnostic_early_fnd.yaml").build_cases()
    algorithms = [
        ROOT / "configs/algorithms/eulc.yaml",
        ROOT / "configs/benchmarks/best_eulc_pso.yaml",
        ROOT / "configs/benchmarks/best_eulc_ga.yaml",
        ROOT / "configs/benchmarks/best_eulc_ac_aco.yaml",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    round_path = args.output_dir / "round_diagnostics.csv"
    node_path = args.output_dir / "node_residual_energy.csv"
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema_version": 2, "status": "running", "git_commit": _commit(),
        "experiment": "early_fnd_energy_balance", "distribution": "gaussian",
        "seeds": [0, 1, 2], "rounds": 30,
        "paper_cases": [
            {"scenario_id": "e0p5_p6400", "initial_energy_j": 0.5, "packet_size_bits": 6400},
            {"scenario_id": "e0p5_p4000", "initial_energy_j": 0.5, "packet_size_bits": 4000},
            {"scenario_id": "e1p0_p6400", "initial_energy_j": 1.0, "packet_size_bits": 6400},
            {"scenario_id": "e1p0_p4000", "initial_energy_j": 1.0, "packet_size_bits": 4000},
        ],
        "algorithms": ["eulc", "eulc_pso", "eulc_ga", "eulc_ac_aco"],
        "round_rows": 0, "node_rows": 0,
        "blank_fields": {
            "objective_J/objective_e/objective_d": "Blank for pure EULC because it does not call the shared objective evaluator.",
            "node component energy fields": "Blank because the simulator does not persist direct per-node component counters.",
            "is_relay_if_available": "Read from the selected RoutePlan forwarding load; blank when no RoutePlan is available.",
        },
        "accounting": {
            "control_energy_j": "Existing broadcast control-packet debit.",
            "retransmission_energy_j": "Actual TX+RX debit after the first attempt.",
            "aggregation_energy_j": "Remaining observed debit at the CH aggregation step; no new energy formula is introduced.",
        },
    }
    _atomic_json(manifest_path, manifest)

    with round_path.open("w", newline="", encoding="utf-8") as round_handle, node_path.open(
        "w", newline="", encoding="utf-8"
    ) as node_handle:
        round_writer = csv.DictWriter(round_handle, fieldnames=ROUND_FIELDS)
        node_writer = csv.DictWriter(node_handle, fieldnames=NODE_FIELDS)
        round_writer.writeheader()
        node_writer.writeheader()

        for algorithm_path in algorithms:
            algorithm_config = load_algorithm_config(algorithm_path)
            for case in cases:
                simulator, execution = build_simulator(case, algorithm_config)
                algorithm = simulator.algorithm_id
                seed = int(case["metadata"]["base_seed"])
                distribution = str(case["environment"]["distribution"])
                initial_energy = float(case["environment"]["initial_energy_j"])
                packet_bits = int(case["environment"]["packet_size_bits"])
                scenario_id = f"e{initial_energy:.1f}_p{packet_bits}".replace(".", "p")
                cumulative_energy = 0.0
                fnd_seen = hnd_seen = lnd_seen = False
                print(f"[{scenario_id}][{algorithm.upper()}][seed={seed}] started", flush=True)

                with capture_round_energy(simulator) as captured:
                    def collect(sim, round_idx: int, _packets_received: int) -> None:
                        nonlocal cumulative_energy, fnd_seen, hnd_seen, lnd_seen
                        energy = captured[-1]
                        residual = np.asarray(sim.energies, dtype=float)
                        before = np.asarray(energy.residual_before_j, dtype=float)
                        alive = residual > sim.params.dead_energy_threshold_j
                        alive_count = int(np.count_nonzero(alive))
                        dead_count = int(sim.case.node_count - alive_count)
                        fnd_seen = fnd_seen or dead_count >= 1
                        hnd_seen = hnd_seen or dead_count >= (sim.case.node_count + 1) // 2
                        lnd_seen = lnd_seen or alive_count == 0
                        cumulative_energy += energy.total_round_energy
                        objective = sim.current_objective_result
                        mean = float(np.mean(residual))
                        std = float(np.std(residual))
                        ordered = np.sort(residual)
                        round_writer.writerow({
                            "scenario_id": scenario_id, "initial_energy_j": initial_energy,
                            "packet_size_bits": packet_bits,
                            "algorithm": algorithm, "seed": seed, "distribution": distribution,
                            "round": round_idx, "alive_nodes": alive_count, "dead_nodes": dead_count,
                            "selected_ch_ids": selected_ch_json(sim.current_chs),
                            "min_residual_energy_j": float(ordered[0]),
                            "mean_residual_energy_j": mean, "max_residual_energy_j": float(ordered[-1]),
                            "std_residual_energy_j": std,
                            "energy_range_j": float(ordered[-1] - ordered[0]),
                            "energy_cv": std / mean if mean > 0.0 else "",
                            "lowest_5_nodes_mean_energy_j": float(np.mean(ordered[:5])),
                            "highest_5_nodes_mean_energy_j": float(np.mean(ordered[-5:])),
                            "member_tx_energy_j": energy.member_tx_energy,
                            "ch_rx_energy_j": energy.ch_rx_energy,
                            "aggregation_energy_j": energy.aggregation_energy,
                            "routing_tx_energy_j": energy.routing_tx_energy,
                            "routing_rx_energy_j": energy.routing_rx_energy,
                            "retransmission_energy_j": energy.retransmission_energy,
                            "control_energy_j": energy.control_energy,
                            "total_round_energy_j": energy.total_round_energy,
                            "cumulative_energy_consumed_j": cumulative_energy,
                            "packet_delivery_ratio": (energy.packets_delivered / energy.packets_generated
                                if energy.packets_generated else ""),
                            "average_e2e_delay_s": (energy.delivered_delay_s / energy.packets_delivered
                                if energy.packets_delivered else ""),
                            "objective_J": objective.objective_J if objective is not None else "",
                            "objective_e": objective.energy_term_e if objective is not None else "",
                            "objective_d": objective.delay_term_d if objective is not None else "",
                            "fnd_reached": fnd_seen, "hnd_reached": hnd_seen, "lnd_reached": lnd_seen,
                        })
                        chs = set(int(value) for value in sim.current_chs)
                        relays = (set(int(ch) for ch, load in sim.current_route_plan.forwarding_load_by_ch.items()
                            if load > 0) if sim.current_route_plan is not None else None)
                        consumed = before - residual
                        for node_id, value in enumerate(residual):
                            node_writer.writerow({
                                "scenario_id": scenario_id, "initial_energy_j": initial_energy,
                                "packet_size_bits": packet_bits,
                                "algorithm": algorithm, "seed": seed, "round": round_idx,
                                "node_id": node_id, "is_alive": bool(alive[node_id]),
                                "is_ch": node_id in chs,
                                "is_relay_if_available": (node_id in relays) if relays is not None else "",
                                "residual_energy_j": float(value),
                                "energy_consumed_this_round_j": float(consumed[node_id]),
                                "member_tx_energy_j": "", "rx_energy_j": "",
                                "aggregation_energy_j": "", "forwarding_energy_j": "",
                                "retransmission_energy_j": "",
                            })
                        _flush(round_handle)
                        _flush(node_handle)
                        manifest["round_rows"] += 1
                        manifest["node_rows"] += sim.case.node_count
                        manifest["last_completed"] = {"scenario_id": scenario_id,
                            "algorithm": algorithm, "seed": seed, "round": round_idx}
                        _atomic_json(manifest_path, manifest)
                        print(f"[{scenario_id}][{algorithm.upper()}][seed={seed}] round {round_idx}/30", flush=True)

                    simulator.run(stop_on_first_dead=False, max_rounds=int(execution["rounds"]),
                        round_callback=collect)

    manifest["status"] = "completed"
    manifest["expected_round_rows"] = 1440
    manifest["expected_node_rows"] = 144000
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
