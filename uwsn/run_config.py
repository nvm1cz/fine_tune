from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .cases import SimulationCase


@dataclass
class TunableParams:
    width_m: float
    height_m: float
    depth_m: float
    # R0: ngÆ°á»¡ng chiá»u sÃ¢u cho lá»›p nÃ´ng nháº¥t + cÆ¡ sá»Ÿ bÃ¡n kÃ­nh cáº¡nh tranh trong EULC (khÃ¡c pháº¡m vi truyá»n).
    layer_r0_m: float
    # Pháº¡m vi truyá»n thÃ´ng tá»‘i Ä‘a giá»¯a cÃ¡c node (theo bÃ i: 150 m hoáº·c 200 m).
    transmission_range_m: float
    layer_spacing_m: float
    cluster_head_ratio: float
    alpha_weight: float
    beta_weight: float
    gamma_weight: float
    balance_factor: float
    recluster_interval: int
    pso_particles: int
    pso_iterations: int
    pso_inertia: float
    pso_c1: float
    pso_c2: float
    pso_omega: float
    optimizer: str
    acoustic_frequency_khz: float
    spreading_factor: float
    transmit_power_p0: float
    electronic_energy_e1_nj: float
    processing_energy_nj: float
    broadcast_packet_size_bits: int
    energy_calibration_factor: float
    contention_penalty_factor: float
    contention_reference_degree: float
    contention_max_multiplier: float
    dead_energy_threshold_j: float
    sink_position: Tuple[float, float, float] | None
    deployment_distribution: str
    gaussian_std_fraction: float
    exponential_scale_fraction: float

    # EULC
    eulc_candidate_ratio: float
    competition_radius_mode: str = "dynamic_eulc"
    competition_adjustment_factor: float = 0.5
    assignment_mode: str = "strongest_rssi"
    fallback_assignment_mode: str = "nearest_feasible"
    require_all_alive_nodes_assigned: bool = True
    routing_mode: str = "joint_optimized"
    reuse_optimized_route_plan: bool = True
    validate_route_every_round: bool = True
    routing_max_hops: int = 100
    routing_plan_search_limit: int = 128
    reoptimize_on_invalid_assignment: bool = True
    reoptimize_on_broken_route: bool = True
    reoptimize_on_dead_ch_or_relay: bool = True
    connectivity_penalty_enabled: bool = True
    pso_stagnation_restart_iterations: int = 5
    pso_stagnation_restart_fraction: float = 0.25
    ga_crossover_rate: float = 0.85
    ga_mutation_sigma: float = 0.10

    # Current / mobility model. Disabled by default to preserve historical runs.
    current_model_enabled: bool = False
    current_model: str = "uniform"
    current_time_step_s: float = 1.0
    current_mean_velocity_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_cos_amplitude_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_sin_amplitude_mps: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_angular_frequency_rad_s: float = 0.0
    current_horizontal_only: bool = True
    current_rbf_centers_xy_m: Tuple[Tuple[float, float], ...] = ()
    current_rbf_centers_m: Tuple[Tuple[float, float, float], ...] = ()
    current_rbf_sigma_m: float = 1.0
    current_rbf_mean_coefficients_mps: Tuple[Tuple[float, float, float], ...] = ()
    current_rbf_cos_coefficients_mps: Tuple[Tuple[Tuple[float, float, float], ...], ...] = ()
    current_rbf_sin_coefficients_mps: Tuple[Tuple[Tuple[float, float, float], ...], ...] = ()
    current_rbf_angular_frequencies_rad_s: Tuple[float, ...] = ()
    current_include_rope_tension_velocity: bool = False
    node_mass_kg: float = 0.0
    node_buoyancy_force_n: float = 0.0
    node_gravity_force_n: float = 0.0
    node_rope_tension_n: float = 0.0
    node_rope_horizontal_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    current_round_duration_s: float = 1.0
    current_update_interval_rounds: int = 1
    current_boundary_policy: str = "clip"
    current_move_dead_nodes: bool = False
    kalman_coefficient_update_enabled: bool = False

    # Delay model.
    node_bit_rate_bps: float = 10_000.0
    sound_speed_model: str = "constant"
    sound_speed_water_mps: float = 1500.0
    water_temperature_c: float = 10.0
    water_salinity_ppt: float = 35.0
    sound_speed_depth_mode: str = "link_average"
    byte_alignment_delay_s: float = 0.0
    holding_time_s: float = 0.0
    query_timer_s: float = 0.0

    # Channel noise model.
    shipping_noise_factor: float = 0.5
    wind_speed_mps: float = 0.0
    channel_bandwidth_hz: float = 1.0
    source_level_db: float | None = None
    directivity_index_db: float = 0.0
    enable_packet_errors: bool = False
    enable_retransmission: bool = True
    max_retries: int = 3
    minimum_snr_db: float | None = None
    minimum_success_probability: float | None = None
    transmission_anomaly_db: float = 0.0
    noise_model: str = "component_sum"
    noise_formula_mode: str = "standard_wenz"
    fading_model: str = "rayleigh"
    rician_k_linear: float = 1.0
    bitrate_model: str = "fixed"
    modem_max_bitrate_bps: float = 10_000.0
    capacity_efficiency: float = 1.0
    modem_transition_delay_s: float = 0.0
    charge_rx_energy_on_corrupted_packet: bool = True
    acoustic_model_version: str = "paper_geethu_2017_v1"
    acoustic_distance_unit: str = "km"
    acoustic_reference_distance_m: float = 1.0e-3
    acoustic_reference_attenuation_linear: float = 1.0

    # Shared optimizer objective. Legacy/weighted-sum modes are explicit compatibility only.
    objective_mode: str = "normalized_multiplicative"
    objective_lambda_energy: float = 0.5
    objective_lambda_delay: float = 0.5
    objective_energy_reference_j: float | None = None
    objective_delay_reference_s: float | None = None
    objective_minimum_route_success_probability: float = 0.0
    objective_infeasible_penalty: float = 1.0e12
    objective_normalization_epsilon: float = 1.0e-12
    objective_energy_epsilon_j: float = 1.0e-12
    objective_delay_epsilon_s: float = 1.0e-12
    objective_dimensionless_epsilon: float = 1.0e-12

    # Adaptive Chaotic Ant Colony Optimization for CH selection.
    ac_aco_ants: int = 30
    ac_aco_iterations: int = 50
    ac_aco_alpha: float = 1.0
    ac_aco_adaptive_beta: bool = True
    ac_aco_beta_start: float = 1.0
    ac_aco_beta_end: float = 3.0
    ac_aco_adaptive_rho: bool = True
    ac_aco_rho_start: float = 0.5
    ac_aco_rho_end: float = 0.1
    ac_aco_initial_pheromone: float = 1.0
    ac_aco_tau_min: float = 1.0e-6
    ac_aco_tau_max: float = 1.0e6
    ac_aco_deposit_q: float = 1.0
    ac_aco_deposit_mode: str = "global_best"
    ac_aco_elite_weight: float = 0.0
    ac_aco_chaos_enabled: bool = True
    ac_aco_chaos_r: float = 4.0
    ac_aco_chaos_strength: float = 0.1
    ac_aco_chaos_initial_state: float | None = None
    ac_aco_stagnation_limit: int = 10
    ac_aco_stagnation_mode: str = "reset"
    ac_aco_heuristic_energy_weight: float = 0.5
    ac_aco_heuristic_eulc_weight: float = 0.25
    ac_aco_heuristic_centrality_weight: float = 0.25
    ac_aco_objective_epsilon: float = 1.0e-12

    # ðŸ”¥ AUTO SINK
    def get_sink_position(self) -> Tuple[float, float, float]:
        if self.sink_position is not None:
            return self.sink_position
        return (
            self.width_m / 2,
            self.height_m / 2,
            0.0
        )


# =========================
# SIMULATION CONFIG
# =========================

width = 100.0
height = 100.0
depth = 100.0

SIMULATION_CASE = SimulationCase(
    name="custom",
    initial_energy=0.5,
    packet_size_bits=6400,
    node_count=100,
    rounds=1000,
)

SIMULATION_PARAMS = TunableParams(
    width_m=width,
    height_m=height,
    depth_m=depth,

    layer_r0_m=20.0,
    transmission_range_m=200.0,
    layer_spacing_m=10.0,

    cluster_head_ratio=0.05,

    alpha_weight=0.6,
    beta_weight=0.25,
    gamma_weight=0.15,

    balance_factor=0.5,
    recluster_interval=20,

    pso_particles=20,
    pso_iterations=50,
    pso_inertia=0.7,
    pso_c1=1.5,
    pso_c2=1.5,
    pso_omega=0.65,
    optimizer="pso",

    acoustic_frequency_khz=10.0,
    spreading_factor=1.5,

    transmit_power_p0=3.0,
    electronic_energy_e1_nj=5.0,
    processing_energy_nj=50.0,

    broadcast_packet_size_bits=200,
    energy_calibration_factor=1.0,
    contention_penalty_factor=0.0,
    contention_reference_degree=10.0,
    contention_max_multiplier=4.0,
    dead_energy_threshold_j=0.075,

    
    sink_position=None,
    deployment_distribution="uniform",
    gaussian_std_fraction=0.18,
    exponential_scale_fraction=0.35,

    # EULC
    eulc_candidate_ratio=0.2
)

RUNS = 10
BASE_SEED = 42
OUTPUT_DIR = "outputs"
