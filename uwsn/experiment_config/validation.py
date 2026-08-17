from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class ConfigError(ValueError):
    """Raised when an experiment config is incomplete or inconsistent."""


TOP_LEVEL_KEYS = {
    "metadata", "environment", "channel", "protocol", "execution", "output",
    "objective",
}
SECTION_KEYS = {
    "metadata": {
        "case_index", "case_id", "experiment_set", "density", "base_seed",
        "topology_seed", "protocol_seed", "optimizer_seed", "channel_seed", "mobility_seed",
        "config_hash", "generated_at", "code_version", "status",
    },
    "environment": {
        "deployment_size_m", "node_count", "distribution", "gaussian_std_fraction",
        "exponential_scale_fraction", "sink_count", "sink_position",
        "initial_energy_j", "packet_size_bits", "data_generation_rate_packets_per_alive_node_per_round",
    },
    "channel": {
        "frequency_khz", "spreading_factor", "transmission_range_m",
        "communication_range_applies_to", "sound_speed_model", "sound_speed_mps",
        "temperature_c", "salinity_ppt", "depth_mode", "enable_packet_errors",
        "enable_retransmissions", "max_retries", "shipping_noise_factor",
        "wind_speed_mps", "bandwidth_hz", "source_level_db", "directivity_index_db",
        "minimum_snr_db", "minimum_success_probability", "node_bit_rate_bps",
        "acoustic_model_version", "acoustic_distance_unit", "reference_distance_m",
        "reference_attenuation_linear", "transmission_anomaly_db", "noise_model",
        "noise_formula_mode", "fading_model", "rician_k_linear", "bitrate_model",
        "modem_max_bitrate_bps", "capacity_efficiency", "modem_transition_delay_s",
        "charge_rx_energy_on_corrupted_packet",
    },
    "protocol": {
        "aggregation", "node_death_threshold_j", "broadcast_packet_size_bits",
        "transmit_power_p0", "electronic_energy_e1_nj", "processing_energy_nj",
        "energy_calibration_factor", "contention_penalty_factor",
        "contention_reference_degree", "contention_max_multiplier",
        "assignment_mode", "fallback_assignment_mode",
        "require_all_alive_nodes_assigned", "routing_mode",
        "reuse_optimized_route_plan", "validate_route_every_round",
        "routing_max_hops", "routing_plan_search_limit",
        "reoptimize_on_invalid_assignment", "reoptimize_on_broken_route",
        "reoptimize_on_dead_ch_or_relay",
        "competition_radius_mode", "competition_adjustment_factor",
    },
    "execution": {
        "rounds", "stop_on_first_dead", "stop_at_first_5pct_dead",
        "min_alive_ratio", "mobility_enabled", "mobility_model",
        "round_duration_seconds", "mobility_update_interval_rounds",
        "boundary_policy", "move_dead_nodes",
    },
    "output": {"root_dir", "save_round_metrics", "save_convergence"},
    "objective": {
        "name", "direction", "lambda", "energy_epsilon_j", "delay_epsilon_s",
        "dimensionless_epsilon",
    },
}
OPTIMIZER_PARAMETER_KEYS = {
    "pso": {"inertia", "inertia_schedule", "inertia_start", "inertia_end",
            "velocity_max", "c1", "c2", "omega", "stagnation_restart_iterations",
            "stagnation_restart_fraction"},
    "ga": {"crossover_rate", "mutation_sigma"},
    "ac_aco": {
        "alpha", "adaptive_beta", "beta_start", "beta_end", "adaptive_rho",
        "rho_start", "rho_end", "initial_pheromone", "tau_min", "tau_max",
        "deposit_q", "deposit_mode", "elite_weight", "chaos_enabled", "chaos_r",
        "chaos_strength", "chaos_initial_state", "stagnation_limit",
        "stagnation_mode", "heuristic_energy_weight", "heuristic_eulc_weight",
        "heuristic_centrality_weight", "objective_epsilon",
    },
    "leach": set(), "eulc": set(), "eeumc": set(), "ebrec": set(),
}


def _require_keys(section: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required.difference(section)
    if missing:
        raise ConfigError(f"{label} missing keys: {sorted(missing)}")


def validate_case(case: Mapping[str, Any]) -> None:
    unknown_top = set(case).difference(TOP_LEVEL_KEYS)
    missing_top = TOP_LEVEL_KEYS.difference(case)
    if unknown_top or missing_top:
        raise ConfigError(f"Invalid top-level keys; unknown={sorted(unknown_top)}, missing={sorted(missing_top)}")
    for name, allowed in SECTION_KEYS.items():
        section = case[name]
        if not isinstance(section, Mapping):
            raise ConfigError(f"{name} must be a mapping")
        unknown = set(section).difference(allowed)
        if unknown:
            raise ConfigError(f"Unknown {name} keys: {sorted(unknown)}")

    env, channel, protocol = case["environment"], case["channel"], case["protocol"]
    execution, metadata = case["execution"], case["metadata"]
    objective = case["objective"]
    _require_keys(objective, {
        "name", "direction", "lambda", "energy_epsilon_j", "delay_epsilon_s",
        "dimensionless_epsilon",
    }, "objective")
    if objective["name"] != "normalized_multiplicative_energy_delay_objective":
        raise ConfigError("unsupported objective name")
    if objective["direction"] != "minimize":
        raise ConfigError("objective direction must be minimize")
    if not 0.0 <= float(objective["lambda"]) <= 1.0:
        raise ConfigError("objective lambda must be in [0, 1]")
    if any(float(objective[key]) <= 0.0 for key in (
        "energy_epsilon_j", "delay_epsilon_s", "dimensionless_epsilon"
    )):
        raise ConfigError("objective epsilons must be positive")
    _require_keys(env, {
        "deployment_size_m", "node_count", "distribution", "sink_position",
        "initial_energy_j", "packet_size_bits",
    }, "environment")
    size = env["deployment_size_m"]
    if not isinstance(size, list) or len(size) != 3 or any(float(v) <= 0 for v in size):
        raise ConfigError("deployment_size_m must contain three positive values")
    if int(env["node_count"]) <= 0:
        raise ConfigError("node_count must be positive")
    if env["distribution"] not in {"uniform", "gaussian", "exponential"}:
        raise ConfigError("Unsupported deployment distribution")
    sink = env["sink_position"]
    if not isinstance(sink, list) or len(sink) != 3 or any(
        float(sink[i]) < 0 or float(sink[i]) > float(size[i]) for i in range(3)
    ):
        raise ConfigError("sink_position lies outside deployment")
    if float(env["initial_energy_j"]) <= 0 or int(env["packet_size_bits"]) <= 0:
        raise ConfigError("initial energy and packet size must be positive")
    if float(channel["transmission_range_m"]) <= 0:
        raise ConfigError("transmission_range_m must be positive")
    model = channel["sound_speed_model"]
    if model not in {"constant", "mackenzie"}:
        raise ConfigError("sound_speed_model must be constant or mackenzie")
    if model == "constant" and float(channel.get("sound_speed_mps", 0)) <= 0:
        raise ConfigError("constant sound speed must be positive")
    if model == "mackenzie":
        _require_keys(channel, {"temperature_c", "salinity_ppt", "depth_mode"}, "channel")
        if channel["depth_mode"] != "link_average":
            raise ConfigError("Mackenzie depth_mode must be link_average")
    if int(channel["max_retries"]) < 0:
        raise ConfigError("max_retries cannot be negative")
    if channel.get("acoustic_model_version", "paper_geethu_2017_v1") not in {
        "paper_geethu_2017_v1", "shared_stojanovic_thorp_v2",
    }:
        raise ConfigError("unsupported acoustic_model_version")
    if channel.get("acoustic_distance_unit", "m") != "m":
        raise ConfigError("acoustic_distance_unit must be m")
    if float(channel.get("reference_distance_m", 1.0e-3)) <= 0.0:
        raise ConfigError("reference_distance_m must be positive")
    if float(channel.get("reference_attenuation_linear", 1.0)) <= 0.0:
        raise ConfigError("reference_attenuation_linear must be positive")
    if float(channel["frequency_khz"]) <= 0.0:
        raise ConfigError("frequency_khz must be positive")
    if float(channel["spreading_factor"]) <= 0.0:
        raise ConfigError("spreading_factor must be positive")
    if float(channel["node_bit_rate_bps"]) <= 0.0:
        raise ConfigError("node_bit_rate_bps must be positive")
    if float(channel["bandwidth_hz"]) <= 0.0:
        raise ConfigError("bandwidth_hz must be positive")
    if channel.get("noise_model", "component_sum") not in {
        "component_sum", "paper_approximation",
    }:
        raise ConfigError("unsupported noise_model")
    if channel.get("noise_formula_mode", "standard_wenz") not in {
        "standard_wenz", "paper_exact",
    }:
        raise ConfigError("unsupported noise_formula_mode")
    if channel.get("fading_model", "none") not in {"none", "rayleigh", "rician"}:
        raise ConfigError("unsupported fading_model")
    if channel.get("fading_model") == "rician" and float(
        channel.get("rician_k_linear", 1.0)
    ) < 0.0:
        raise ConfigError("rician_k_linear cannot be negative")
    if channel.get("bitrate_model", "fixed") not in {"fixed", "shannon_limited"}:
        raise ConfigError("unsupported bitrate_model")
    if float(channel.get("modem_max_bitrate_bps", channel["node_bit_rate_bps"])) <= 0.0:
        raise ConfigError("modem_max_bitrate_bps must be positive")
    if not 0.0 < float(channel.get("capacity_efficiency", 1.0)) <= 1.0:
        raise ConfigError("capacity_efficiency must be in (0, 1]")
    if float(channel.get("modem_transition_delay_s", 0.0)) < 0.0:
        raise ConfigError("modem_transition_delay_s cannot be negative")
    if execution.get("mobility_model", "uniform") not in {
        "uniform", "rbf", "tidal_rbf"
    }:
        raise ConfigError("unsupported mobility_model")
    if float(execution.get("round_duration_seconds", 1.0)) <= 0.0:
        raise ConfigError("round_duration_seconds must be positive")
    if int(execution.get("mobility_update_interval_rounds", 1)) <= 0:
        raise ConfigError("mobility_update_interval_rounds must be positive")
    if execution.get("boundary_policy", "clip") not in {
        "clip", "reflect", "wrap", "reject_update",
    }:
        raise ConfigError("unsupported boundary_policy")
    if channel["source_level_db"] is not None and not np.isfinite(
        float(channel["source_level_db"])
    ):
        raise ConfigError("source_level_db must be finite when provided")
    if bool(channel["enable_packet_errors"]) and channel["source_level_db"] is None:
        raise ConfigError("source_level_db is required when packet errors are enabled")
    for key in (
        "transmit_power_p0", "electronic_energy_e1_nj", "processing_energy_nj",
        "energy_calibration_factor",
    ):
        if float(protocol[key]) <= 0.0:
            raise ConfigError(f"{key} must be positive")
    if protocol.get("assignment_mode", "strongest_rssi") != "strongest_rssi":
        raise ConfigError("assignment_mode must be strongest_rssi")
    if protocol.get("routing_mode", "joint_optimized") not in {
        "joint_optimized", "legacy_greedy"
    }:
        raise ConfigError("unsupported routing_mode")
    if int(protocol.get("routing_max_hops", 100)) <= 0:
        raise ConfigError("routing_max_hops must be positive")
    if int(protocol.get("routing_plan_search_limit", 128)) <= 0:
        raise ConfigError("routing_plan_search_limit must be positive")
    if protocol.get("competition_radius_mode", "dynamic_eulc") != "dynamic_eulc":
        raise ConfigError("competition_radius_mode must be dynamic_eulc")
    if not 0.0 <= float(protocol.get("competition_adjustment_factor", 0.5)) <= 1.0:
        raise ConfigError("competition_adjustment_factor must be in [0, 1]")
    if float(protocol["node_death_threshold_j"]) < 0:
        raise ConfigError("node death threshold cannot be negative")
    if int(execution["rounds"]) <= 0:
        raise ConfigError("rounds must be positive")
    for key in (
        "base_seed", "topology_seed", "protocol_seed", "optimizer_seed",
        "channel_seed", "mobility_seed",
    ):
        if int(metadata[key]) < 0:
            raise ConfigError(f"{key} cannot be negative")
