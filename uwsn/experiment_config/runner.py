from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..cases import SimulationCase
from ..algorithms.config import apply_algorithm_config, load_algorithm_config
from ..algorithms.registry import canonical_algorithm_id, create_algorithm
from ..run_config import SIMULATION_PARAMS
from ..simulator import UWSNSimulator
from .io import load_yaml, write_yaml
from .validation import ConfigError, validate_case


def build_simulator(
    case_config: dict[str, Any],
    algorithm_config: dict[str, Any] | None = None,
) -> tuple[UWSNSimulator, dict[str, Any]]:
    """Create a simulator from one scenario plus one independent algorithm config."""
    if algorithm_config is None:
        raise ConfigError("algorithm_config is required; scenario inputs are algorithm-independent")
    validate_case(case_config)
    case_config = apply_algorithm_config(case_config, algorithm_config)
    embedded_algorithm = case_config.get("algorithm")
    env, channel, protocol = (
        case_config["environment"], case_config["channel"], case_config["protocol"]
    )
    execution, meta = case_config["execution"], case_config["metadata"]
    if embedded_algorithm is not None:
        algorithm_id = canonical_algorithm_id(embedded_algorithm["algorithm"]["id"])
        if "optimizer" in embedded_algorithm:
            algorithm_optimizer = embedded_algorithm["optimizer"]
            optimizer = {
                "name": algorithm_optimizer["name"],
                "population_size": algorithm_optimizer["population_size"],
                "max_iterations": algorithm_optimizer["max_iterations"],
                "fitness_evaluation_budget": algorithm_optimizer["fitness_evaluation_budget"],
                "stopping_criterion": algorithm_optimizer["stopping_criterion"],
                "parameters": algorithm_optimizer["params"],
                "seed": meta["optimizer_seed"],
            }
        else:
            optimizer = {
                "name": algorithm_id,
                "population_size": 1,
                "max_iterations": 1,
                "fitness_evaluation_budget": 1,
                "stopping_criterion": "protocol_selection",
                "parameters": {},
                "seed": meta["protocol_seed"],
            }
    else:
        raise ConfigError("resolved algorithm configuration is missing")
    if env["data_generation_rate_packets_per_alive_node_per_round"] != 1:
        raise ConfigError("Current simulator supports exactly one packet per alive node per round")
    aggregation = protocol["aggregation"]
    if aggregation != {"output_packet_size_mode": "fixed_original_size", "relay_reaggregation": False}:
        raise ConfigError("Generated aggregation config does not match the implemented model")
    if channel["sound_speed_model"] != "constant":
        raise ConfigError(
            "Mackenzie/link-average propagation is reserved for environmental sensitivity "
            "but is not wired into per-link routing yet"
        )
    dims = env["deployment_size_m"]
    simulation_case = SimulationCase(
        name=meta["case_id"],
        initial_energy=float(env["initial_energy_j"]),
        packet_size_bits=int(env["packet_size_bits"]),
        node_count=int(env["node_count"]),
        rounds=int(execution["rounds"]),
    )
    optimizer_params = optimizer["parameters"]
    objective = case_config["objective"]
    updates: dict[str, Any] = {
        "width_m": float(dims[0]), "height_m": float(dims[1]), "depth_m": float(dims[2]),
        "sink_position": tuple(float(v) for v in env["sink_position"]),
        "deployment_distribution": env["distribution"],
        "gaussian_std_fraction": float(env["gaussian_std_fraction"]),
        "exponential_scale_fraction": float(env["exponential_scale_fraction"]),
        "transmission_range_m": float(channel["transmission_range_m"]),
        "acoustic_frequency_khz": float(channel["frequency_khz"]),
        "spreading_factor": float(channel["spreading_factor"]),
        "sound_speed_model": channel["sound_speed_model"],
        "sound_speed_water_mps": float(channel["sound_speed_mps"]),
        "node_bit_rate_bps": float(channel["node_bit_rate_bps"]),
        "enable_packet_errors": bool(channel["enable_packet_errors"]),
        "enable_retransmission": bool(channel["enable_retransmissions"]),
        "max_retries": int(channel["max_retries"]),
        "shipping_noise_factor": float(channel["shipping_noise_factor"]),
        "wind_speed_mps": float(channel["wind_speed_mps"]),
        "channel_bandwidth_hz": float(channel["bandwidth_hz"]),
        "source_level_db": channel["source_level_db"],
        "directivity_index_db": float(channel["directivity_index_db"]),
        "minimum_snr_db": channel["minimum_snr_db"],
        "minimum_success_probability": channel["minimum_success_probability"],
        "acoustic_model_version": channel.get(
            "acoustic_model_version", "paper_geethu_2017_v1"
        ),
        "acoustic_distance_unit": channel.get("acoustic_distance_unit", "m"),
        "acoustic_reference_distance_m": float(
            channel.get("reference_distance_m", 1.0e-3)
        ),
        "acoustic_reference_attenuation_linear": float(
            channel.get("reference_attenuation_linear", 1.0)
        ),
        "transmission_anomaly_db": float(channel.get("transmission_anomaly_db", 0.0)),
        "noise_model": channel.get("noise_model", "component_sum"),
        "noise_formula_mode": channel.get("noise_formula_mode", "standard_wenz"),
        "fading_model": channel.get("fading_model", "none"),
        "bitrate_model": channel.get("bitrate_model", "fixed"),
        "modem_max_bitrate_bps": float(
            channel.get("modem_max_bitrate_bps", channel["node_bit_rate_bps"])
        ),
        "capacity_efficiency": float(channel.get("capacity_efficiency", 1.0)),
        "modem_transition_delay_s": float(
            channel.get("modem_transition_delay_s", 0.0)
        ),
        "charge_rx_energy_on_corrupted_packet": bool(
            channel.get("charge_rx_energy_on_corrupted_packet", True)
        ),
        "cluster_head_ratio": float(protocol["ch_ratio"]),
        "eulc_candidate_ratio": float(protocol["candidate_ratio"]),
        "alpha_weight": float(protocol["eulc_alpha"]),
        "beta_weight": float(protocol["eulc_beta"]),
        "gamma_weight": float(protocol["eulc_gamma"]),
        "layer_r0_m": float(protocol["competition_radius_m"]),
        "layer_spacing_m": float(protocol["layer_spacing_m"]),
        "competition_radius_mode": protocol.get(
            "competition_radius_mode", "dynamic_eulc"
        ),
        "competition_adjustment_factor": float(
            protocol.get("competition_adjustment_factor", 0.5)
        ),
        "recluster_interval": int(protocol["recluster_interval_rounds"]),
        "recluster_trigger_mode": protocol.get(
            "recluster_trigger_mode", "periodic"
        ),
        "assignment_mode": protocol.get("assignment_mode", "strongest_rssi"),
        "fallback_assignment_mode": protocol.get(
            "fallback_assignment_mode", "nearest_feasible"
        ),
        "require_all_alive_nodes_assigned": bool(
            protocol.get("require_all_alive_nodes_assigned", True)
        ),
        "routing_mode": protocol.get("routing_mode", "joint_optimized"),
        "reuse_optimized_route_plan": bool(
            protocol.get("reuse_optimized_route_plan", True)
        ),
        "validate_route_every_round": bool(
            protocol.get("validate_route_every_round", True)
        ),
        "routing_max_hops": int(protocol.get("routing_max_hops", 100)),
        "routing_plan_search_limit": int(
            protocol.get("routing_plan_search_limit", 128)
        ),
        "reoptimize_on_invalid_assignment": bool(
            protocol.get("reoptimize_on_invalid_assignment", True)
        ),
        "reoptimize_on_broken_route": bool(
            protocol.get("reoptimize_on_broken_route", True)
        ),
        "reoptimize_on_dead_ch_or_relay": bool(
            protocol.get("reoptimize_on_dead_ch_or_relay", True)
        ),
        "dead_energy_threshold_j": float(protocol["node_death_threshold_j"]),
        "broadcast_packet_size_bits": int(protocol["broadcast_packet_size_bits"]),
        "transmit_power_p0": float(protocol["transmit_power_p0"]),
        "electronic_energy_e1_nj": float(protocol["electronic_energy_e1_nj"]),
        "processing_energy_nj": float(protocol["processing_energy_nj"]),
        "energy_calibration_factor": float(protocol["energy_calibration_factor"]),
        "contention_penalty_factor": float(protocol["contention_penalty_factor"]),
        "contention_reference_degree": float(protocol["contention_reference_degree"]),
        "contention_max_multiplier": float(protocol["contention_max_multiplier"]),
        "optimizer": optimizer["name"],
        "pso_particles": int(optimizer["population_size"]),
        "pso_iterations": int(optimizer["max_iterations"]),
        "current_model_enabled": bool(execution["mobility_enabled"]),
        "current_model": (
            "rbf"
            if execution.get("mobility_model") == "tidal_rbf"
            else execution.get("mobility_model", "uniform")
        ),
        "current_round_duration_s": float(
            execution.get("round_duration_seconds", 1.0)
        ),
        "current_update_interval_rounds": int(
            execution.get("mobility_update_interval_rounds", 1)
        ),
        "current_boundary_policy": execution.get("boundary_policy", "clip"),
        "current_move_dead_nodes": bool(execution.get("move_dead_nodes", False)),
        "objective_mode": "normalized_multiplicative",
        "objective_lambda_energy": float(objective["lambda"]),
        "objective_lambda_delay": 1.0 - float(objective["lambda"]),
        "objective_energy_epsilon_j": float(objective["energy_epsilon_j"]),
        "objective_delay_epsilon_s": float(objective["delay_epsilon_s"]),
        "objective_dimensionless_epsilon": float(objective["dimensionless_epsilon"]),
    }
    name = optimizer["name"]
    if name == "pso":
        updates.update({
            "pso_inertia": optimizer_params["inertia"], "pso_c1": optimizer_params["c1"],
            "pso_c2": optimizer_params["c2"], "pso_omega": optimizer_params["omega"],
            "pso_inertia_schedule": optimizer_params.get("inertia_schedule", "fixed"),
            "pso_inertia_start": optimizer_params.get(
                "inertia_start", optimizer_params["inertia"]
            ),
            "pso_inertia_end": optimizer_params.get(
                "inertia_end", optimizer_params["inertia"]
            ),
            "pso_velocity_max": optimizer_params.get("velocity_max", 1.0),
            "pso_stagnation_restart_iterations": optimizer_params["stagnation_restart_iterations"],
            "pso_stagnation_restart_fraction": optimizer_params["stagnation_restart_fraction"],
        })
    elif name == "ga":
        updates.update({"ga_crossover_rate": optimizer_params["crossover_rate"],
                        "ga_mutation_sigma": optimizer_params["mutation_sigma"]})
    elif name == "ac_aco":
        updates.update({f"ac_aco_{key}": value for key, value in optimizer_params.items()})
        updates["ac_aco_ants"] = int(optimizer["population_size"])
        updates["ac_aco_iterations"] = int(optimizer["max_iterations"])
    params = replace(SIMULATION_PARAMS, **updates)
    streams = {
        name: int(meta[f"{name}_seed"])
        for name in ("topology", "protocol", "optimizer", "channel", "mobility")
    }
    simulator = UWSNSimulator(
        simulation_case, params, seed=int(meta["base_seed"]), verbose=False,
        track_pso_convergence=bool(case_config["output"]["save_convergence"]),
        seed_streams=streams,
        algorithm=create_algorithm(algorithm_id),
    )
    return simulator, execution


def run_case_file(
    config_path: Path,
    *,
    algorithm_path: Path | None = None,
    algorithm_id: str | None = None,
) -> Path:
    config = load_yaml(config_path)
    if algorithm_path is not None and algorithm_id is not None:
        raise ConfigError("Use either algorithm_path or algorithm_id, not both")
    algorithm_config = load_algorithm_config(algorithm_path) if algorithm_path else None
    if algorithm_id is not None:
        default_path = Path("configs/algorithms") / f"{canonical_algorithm_id(algorithm_id)}.yaml"
        algorithm_config = load_algorithm_config(default_path)
    if algorithm_config is None:
        raise ConfigError("Specify algorithm_path or algorithm_id for the shared scenario")
    simulator, execution = build_simulator(config, algorithm_config)
    resolved = apply_algorithm_config(config, algorithm_config) if algorithm_config else config
    canonical_id = simulator.algorithm_id
    started = time.perf_counter()
    metrics = simulator.run(
        stop_on_first_dead=bool(execution["stop_on_first_dead"]),
        stop_at_ft5=bool(execution["stop_at_first_5pct_dead"]),
        min_alive_ratio=execution["min_alive_ratio"],
        max_rounds=int(execution["rounds"]),
    )
    runtime = time.perf_counter() - started
    output_dir = (
        Path(config["output"]["root_dir"])
        / config["metadata"]["experiment_set"]
        / config["metadata"]["case_id"]
        / canonical_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "input_case.yaml", config)
    if algorithm_config is not None:
        write_yaml(output_dir / "algorithm_config.yaml", algorithm_config)
    write_yaml(output_dir / "resolved_config.yaml", resolved)
    rounds_completed = len(metrics.residual_energy_by_round)
    summary = {
        **asdict(metrics),
        "case_id": config["metadata"]["case_id"],
        "config_hash": config["metadata"]["config_hash"],
        "algorithm_id": canonical_id,
        "base_protocol": simulator.algorithm.base_protocol,
        "optimizer_name": simulator.algorithm.optimizer_name,
        "rounds_completed": rounds_completed,
        "FND": metrics.fnd_round,
        "HND": metrics.hnd_round,
        "LND": metrics.lnd_round,
        "total_energy_consumed": (
            simulator.case.initial_energy * simulator.case.node_count - metrics.residual_energy
        ),
        "packets_generated": None,
        "packets_received_by_sink": metrics.packets_received,
        "PDR": None,
        "average_delay": (
            sum(metrics.delay_by_round) / metrics.packets_received
            if metrics.packets_received else None
        ),
        "average_hop_count": None,
        "route_failures": None,
        "runtime_seconds": runtime,
        "status": "completed",
        "acoustic_model_version": simulator.params.acoustic_model_version,
        **simulator.algorithm_diagnostics,
    }
    summary["fnd_censored"] = metrics.fnd_round is None
    summary["hnd_censored"] = metrics.hnd_round is None
    summary["lnd_censored"] = metrics.lnd_round is None
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "optimization_events.json").write_text(
        json.dumps(simulator.optimization_events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "round_configuration.json").write_text(
        json.dumps(simulator.round_configuration_history, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    if config["output"]["save_round_metrics"]:
        with (output_dir / "round_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "round", "residual_energy_j", "dead_nodes", "packets_received", "delay_s",
                "configuration_valid", "assignment_valid", "route_valid",
                "reused_route_plan", "reoptimization_occurred",
                "execution_route_plan_id", "invalid_reason",
            ])
            writer.writeheader()
            for index, values in enumerate(zip(
                metrics.residual_energy_by_round, metrics.dead_nodes_by_round,
                metrics.packets_received_by_round, metrics.delay_by_round,
            ), 1):
                configuration = simulator.round_configuration_history[index - 1]
                writer.writerow({
                    "round": index,
                    "residual_energy_j": values[0],
                    "dead_nodes": values[1],
                    "packets_received": values[2],
                    "delay_s": values[3],
                    **{
                        key: configuration.get(key)
                        for key in writer.fieldnames[5:]
                    },
                })
    if config["output"]["save_convergence"]:
        fields = [
            "round", "refresh_index", "iteration", "best_score", "candidate_count",
            "evaluations", "best_J", "best_energy_term", "best_delay_term",
            "feasible_count", "invalid_count",
        ]
        with (output_dir / "convergence.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metrics.pso_convergence)
    (output_dir / "run.log").write_text(
        f"case_id={config['metadata']['case_id']}\nalgorithm_id={canonical_id}\nstatus=completed\n",
        encoding="utf-8",
    )
    return output_dir
