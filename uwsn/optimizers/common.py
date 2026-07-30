from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..routing.joint import optimize_route_plan
from ..objective import CandidateEvaluation, EvaluationContext, ObjectiveResult
from ..routing.planning import assign_members_strongest_rssi
from ..run_config import TunableParams


COVERAGE_INFEASIBLE_BASE_PENALTY = 1_000_000_000_000_000.0
COVERAGE_MISSING_NODE_PENALTY = 1_000_000.0
COVERAGE_EXCESS_DISTANCE_PENALTY = 10_000.0
CONNECTIVITY_ISOLATED_CH_PENALTY = 1_000_000.0
CONNECTIVITY_EXCESS_DISTANCE_PENALTY = 10_000.0


@dataclass(frozen=True)
class PaperCostNormalization:
    max_path_cost: float
    max_candidate_energy: float
    candidate_path_costs: Dict[int, float]


def _target_cluster_head_count(
    active_node_count: int,
    candidate_count: int,
    cluster_head_ratio: float,
) -> int:
    """Return K = max(1, round(Pc * N_active)), capped by M candidates."""
    if active_node_count <= 0 or candidate_count <= 0:
        return 0
    requested_count = int(np.floor(float(cluster_head_ratio) * active_node_count + 0.5))
    return min(candidate_count, max(1, requested_count))


def _decode_priority_vector(
    priority_vector: np.ndarray,
    candidates_array: np.ndarray,
    target_ch_count: int,
) -> List[int]:
    """Decode a real-valued priority vector into exactly K candidate CHs."""
    if priority_vector.ndim != 1 or priority_vector.size != candidates_array.size:
        raise ValueError("Priority vector length must equal the EULC candidate count M")
    if target_ch_count <= 0:
        return []
    selected_offsets = np.argsort(-priority_vector, kind="stable")[:target_ch_count]
    return candidates_array[selected_offsets].astype(int).tolist()


def _build_paper_cost_normalization(
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    dist_to_sink: np.ndarray,
    all_candidates: Sequence[int],
    dead_energy_threshold_j: float,
) -> PaperCostNormalization:
    """Build fixed Eq. (8) candidate-level terms shared by all solutions."""
    active_candidates = np.asarray(
        [
            int(candidate)
            for candidate in all_candidates
            if energies[int(candidate)] > dead_energy_threshold_j
        ],
        dtype=int,
    )
    if active_candidates.size == 0:
        return PaperCostNormalization(1e-12, 1e-12, {})

    if active_candidates.size == 1:
        candidate_path_values = np.square(dist_to_sink[active_candidates])
    else:
        candidate_distances = distance_matrix[np.ix_(active_candidates, active_candidates)]
        candidate_distances_squared = np.square(candidate_distances)
        np.fill_diagonal(candidate_distances_squared, np.inf)
        nearest_other_candidate_squared = np.min(candidate_distances_squared, axis=1)
        candidate_path_values = (
            nearest_other_candidate_squared + np.square(dist_to_sink[active_candidates])
        )

    candidate_path_costs = {
        int(candidate): float(path_cost)
        for candidate, path_cost in zip(active_candidates, candidate_path_values)
        if np.isfinite(path_cost)
    }
    max_path_cost = max(candidate_path_costs.values(), default=1e-12)

    return PaperCostNormalization(
        max(float(max_path_cost), 1e-12),
        max(float(np.max(energies[active_candidates])), 1e-12),
        candidate_path_costs,
    )


def _normalize_solution(
    energies: np.ndarray,
    solution: Sequence[int],
    all_candidates: Sequence[int],
    dead_energy_threshold_j: float,
) -> List[int]:
    selected = sorted(
        set(int(s) for s in solution if energies[int(s)] > dead_energy_threshold_j)
    )
    if selected:
        return selected
    active_candidates = [
        int(candidate)
        for candidate in all_candidates
        if energies[int(candidate)] > dead_energy_threshold_j
    ]
    if not active_candidates:
        return []
    fallback = max(active_candidates, key=lambda idx: float(energies[idx]))
    return [int(fallback)]


def _ensure_candidate_coverage(
    energies: np.ndarray,
    distance_matrix: np.ndarray,
    candidates: Sequence[int],
    transmission_range_m: float,
    dead_energy_threshold_j: float,
) -> List[int]:
    covered_candidates = list(
        dict.fromkeys(
            int(candidate)
            for candidate in candidates
            if energies[int(candidate)] > dead_energy_threshold_j
        )
    )
    for node_idx in np.flatnonzero(energies > dead_energy_threshold_j):
        node = int(node_idx)
        if not covered_candidates or np.all(distance_matrix[node, covered_candidates] > transmission_range_m):
            covered_candidates.append(node)
    return covered_candidates


def _assign_solution(
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    solution: Sequence[int],
    transmission_range_m: float,
    dead_energy_threshold_j: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    alive_indices = np.flatnonzero(energies > dead_energy_threshold_j)
    if alive_indices.size == 0:
        return None
    solution_array = np.asarray(solution, dtype=int)
    if solution_array.size == 0:
        return None

    chosen = distance_matrix[np.ix_(alive_indices, solution_array)]
    if chosen.size == 0 or chosen.shape[1] == 0:
        return None

    in_range = chosen <= transmission_range_m
    if not np.all(np.any(in_range, axis=1)):
        return None

    ranged_distances = np.where(in_range, chosen, np.inf)
    closest_idx = np.argmin(ranged_distances, axis=1)
    assigned_chs = solution_array[closest_idx]
    return alive_indices, solution_array, closest_idx, assigned_chs


def _coverage_penalty(
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    solution: Sequence[int],
    transmission_range_m: float,
    dead_energy_threshold_j: float,
) -> float:
    alive_indices = np.flatnonzero(energies > dead_energy_threshold_j)
    if alive_indices.size == 0:
        return 0.0

    solution_array = np.asarray(
        [int(s) for s in solution if energies[int(s)] > dead_energy_threshold_j],
        dtype=int,
    )
    if solution_array.size == 0:
        return (
            COVERAGE_INFEASIBLE_BASE_PENALTY
            + COVERAGE_MISSING_NODE_PENALTY * float(alive_indices.size)
        )

    chosen = distance_matrix[np.ix_(alive_indices, solution_array)]
    nearest = np.min(chosen, axis=1)
    uncovered = nearest > transmission_range_m
    if not np.any(uncovered):
        return 0.0

    missing_count = int(np.count_nonzero(uncovered))
    range_scale = max(float(transmission_range_m), 1e-9)
    excess_distance = float(np.sum((nearest[uncovered] - transmission_range_m) / range_scale))
    return (
        COVERAGE_INFEASIBLE_BASE_PENALTY
        + COVERAGE_MISSING_NODE_PENALTY * missing_count
        + COVERAGE_EXCESS_DISTANCE_PENALTY * excess_distance
    )


def _inter_cluster_connectivity_penalty(
    params: TunableParams,
    distance_matrix: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution_array: np.ndarray,
) -> float:
    if solution_array.size == 0:
        return CONNECTIVITY_ISOLATED_CH_PENALTY

    transmission_range = max(float(params.transmission_range_m), 1e-9)
    total_penalty = 0.0
    for ch_idx in solution_array.astype(int):
        if float(dist_to_sink[ch_idx]) <= transmission_range:
            continue

        other_chs = solution_array[solution_array != ch_idx]
        if other_chs.size == 0:
            total_penalty += CONNECTIVITY_ISOLATED_CH_PENALTY
            continue

        shallower_chs = other_chs[layers[other_chs] < layers[ch_idx]]
        if shallower_chs.size == 0:
            total_penalty += CONNECTIVITY_ISOLATED_CH_PENALTY
            continue

        nearest = float(np.min(distance_matrix[ch_idx, shallower_chs]))
        if nearest > transmission_range:
            excess = (nearest - transmission_range) / transmission_range
            total_penalty += (
                CONNECTIVITY_ISOLATED_CH_PENALTY
                + CONNECTIVITY_EXCESS_DISTANCE_PENALTY * excess
            )

    return total_penalty


def _build_assignments(
    alive_indices: np.ndarray,
    solution_array: np.ndarray,
    assigned_chs: np.ndarray,
) -> Dict[int, List[int]]:
    assignments: Dict[int, List[int]] = {int(ch): [] for ch in solution_array}
    for ch in solution_array:
        members = alive_indices[assigned_chs == ch]
        if members.size:
            assignments[int(ch)] = members.astype(int).tolist()
    return assignments


def assign_nodes_to_cluster_heads(
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    selected_cluster_heads: Sequence[int],
    transmission_range_m: float,
    dead_energy_threshold_j: float,
) -> Dict[int, List[int]]:
    """Pure shared assignment adapter used by every CH-selection optimizer."""
    assigned = _assign_solution(
        distance_matrix,
        energies,
        selected_cluster_heads,
        transmission_range_m,
        dead_energy_threshold_j,
    )
    if assigned is None:
        return {}
    alive_indices, solution_array, _, assigned_chs = assigned
    return _build_assignments(alive_indices, solution_array, assigned_chs)


def _paper_solution_cost(
    params: TunableParams,
    energies: np.ndarray,
    solution_array: np.ndarray,
    normalization: PaperCostNormalization,
) -> float:
    """Eq. (8) candidate-level cost averaged over decoded CHs in a solution."""
    if solution_array.size == 0:
        return float("inf")

    omega = float(params.pso_omega)
    if not 0.0 <= omega <= 1.0:
        raise ValueError("pso_omega must be in [0, 1]")

    candidate_path_values = np.asarray(
        [normalization.candidate_path_costs.get(int(ch), np.inf) for ch in solution_array],
        dtype=float,
    )
    if not np.all(np.isfinite(candidate_path_values)):
        return float("inf")
    mean_distance_component = float(np.mean(candidate_path_values / normalization.max_path_cost))

    energy_penalties = (
        normalization.max_candidate_energy - energies[solution_array]
    ) / normalization.max_candidate_energy
    mean_energy_component = float(np.mean(energy_penalties))

    cost = omega * mean_distance_component + (1.0 - omega) * mean_energy_component
    if not np.isfinite(cost):
        return float("inf")
    return cost


def solution_cost(
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution_array: np.ndarray,
    normalization: PaperCostNormalization,
) -> float:
    base_cost = _paper_solution_cost(params, energies, solution_array, normalization)
    if not np.isfinite(base_cost):
        return float("inf")
    if not getattr(params, "connectivity_penalty_enabled", True):
        return base_cost
    return base_cost + _inter_cluster_connectivity_penalty(
        params, distance_matrix, layers, dist_to_sink, solution_array
    )


def evaluate_solution_result(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution: Sequence[int],
    all_candidates: Sequence[int],
    cost_normalization: PaperCostNormalization | None = None,
    positions: np.ndarray | None = None,
) -> ObjectiveResult:
    return evaluate_complete_solution(
        case, params, distance_matrix, energies, layers, dist_to_sink,
        solution, all_candidates, cost_normalization, positions
    ).objective_result


def evaluate_complete_solution(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution: Sequence[int],
    all_candidates: Sequence[int],
    cost_normalization: PaperCostNormalization | None = None,
    positions: np.ndarray | None = None,
) -> CandidateEvaluation:
    del layers, dist_to_sink, cost_normalization
    solution = _normalize_solution(
        energies, solution, all_candidates, params.dead_energy_threshold_j
    )
    if positions is None:
        raise ValueError("positions are required for the shared energy-delay objective")
    selected = list(solution)
    for node in np.flatnonzero(energies > params.dead_energy_threshold_j):
        node_id = int(node)
        if not selected or np.all(
            distance_matrix[node_id, selected] > params.transmission_range_m
        ):
            selected.append(node_id)
    selected = sorted(dict.fromkeys(selected))
    assignment = assign_members_strongest_rssi(
        np.asarray(positions, dtype=float), energies, selected, params
    )
    context = EvaluationContext(
        node_positions=np.asarray(positions, dtype=float),
        sink_position=np.asarray(params.get_sink_position(), dtype=float),
        residual_energy_j=np.asarray(energies, dtype=float),
        packet_bits=int(case.packet_size_bits),
        frequency_khz=float(params.acoustic_frequency_khz),
        bandwidth_hz=float(params.channel_bandwidth_hz),
        max_retries=int(params.max_retries),
        lambda_energy=float(params.objective_lambda_energy),
        lambda_delay=float(params.objective_lambda_delay),
        energy_reference_j=1.0,
        delay_reference_s=1.0,
        minimum_route_success_probability=float(
            params.objective_minimum_route_success_probability
        ),
        infeasible_penalty=float(params.objective_infeasible_penalty),
        normalization_epsilon=float(params.objective_normalization_epsilon),
        params=params,
        energy_epsilon_j=float(params.objective_energy_epsilon_j),
        delay_epsilon_s=float(params.objective_delay_epsilon_s),
        objective_epsilon=float(params.objective_dimensionless_epsilon),
    )
    return optimize_route_plan(selected, assignment, context)


def evaluate_solution_score(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution: Sequence[int],
    all_candidates: Sequence[int],
    cost_normalization: PaperCostNormalization | None = None,
    positions: np.ndarray | None = None,
) -> float:
    return evaluate_solution_result(
        case, params, distance_matrix, energies, layers, dist_to_sink,
        solution, all_candidates, cost_normalization, positions
    ).objective_J


def evaluate_legacy_solution_score(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution: Sequence[int],
    all_candidates: Sequence[int],
    cost_normalization: PaperCostNormalization | None = None,
    positions: np.ndarray | None = None,
) -> float:
    solution = _normalize_solution(
        energies, solution, all_candidates, params.dead_energy_threshold_j
    )
    if cost_normalization is None:
        cost_normalization = _build_paper_cost_normalization(
            distance_matrix, energies, dist_to_sink, all_candidates, params.dead_energy_threshold_j
        )
    return solution_cost(
        params,
        distance_matrix,
        energies,
        layers,
        dist_to_sink,
        np.asarray(solution, dtype=int),
        cost_normalization,
    )


def evaluate_solution(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution: Sequence[int],
    all_candidates: Sequence[int],
    cost_normalization: PaperCostNormalization | None = None,
) -> Tuple[float, Dict[int, List[int]]]:
    solution = _normalize_solution(
        energies, solution, all_candidates, params.dead_energy_threshold_j
    )
    assigned = _assign_solution(
        distance_matrix,
        energies,
        solution,
        params.transmission_range_m,
        params.dead_energy_threshold_j,
    )
    if assigned is None:
        return float("inf"), {}

    alive_indices, solution_array, _, assigned_chs = assigned
    if cost_normalization is None:
        cost_normalization = _build_paper_cost_normalization(
            distance_matrix, energies, dist_to_sink, all_candidates, params.dead_energy_threshold_j
        )
    score = solution_cost(
        params, distance_matrix, energies, layers, dist_to_sink, solution_array, cost_normalization
    )
    assignments = _build_assignments(alive_indices, solution_array, assigned_chs)
    return score, assignments


def _finalize_solution(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    solution: Sequence[int],
    candidates: Sequence[int],
    cost_normalization: PaperCostNormalization,
) -> Tuple[List[int], Dict[int, List[int]]]:
    best_solution = _normalize_solution(
        energies, solution, candidates, params.dead_energy_threshold_j
    )
    _, assignments = evaluate_solution(
        case,
        params,
        distance_matrix,
        energies,
        layers,
        dist_to_sink,
        best_solution,
        candidates,
        cost_normalization,
    )
    if getattr(params, "routing_mode", "joint_optimized") == "joint_optimized":
        # The optimizer adapter immediately materializes the complete shared
        # assignment + RoutePlan. Legacy nearest-assignment/connectivity checks must
        # not veto a CH set that the joint evaluator can repair and route.
        return best_solution, assignments
    if not assignments:
        raise RuntimeError(
            "No valid exact-K CH solution covers every active node. "
            "Increase cluster_head_ratio or transmission_range_m."
        )
    if getattr(params, "connectivity_penalty_enabled", True):
        connectivity_penalty = _inter_cluster_connectivity_penalty(
            params, distance_matrix, layers, dist_to_sink, np.asarray(best_solution, dtype=int)
        )
        if connectivity_penalty > 0.0:
            raise RuntimeError(
                "No valid exact-K CH solution gives every CH a shallower CH-to-CH "
                "path to the sink. Adjust the EULC candidate set, "
                "cluster_head_ratio, or transmission_range_m."
            )
    return best_solution, assignments
