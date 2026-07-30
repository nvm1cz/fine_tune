from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from .common import (
    _build_paper_cost_normalization,
    _ensure_candidate_coverage,
    _finalize_solution,
    _target_cluster_head_count,
    evaluate_legacy_solution_score,
)


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmax = float(np.max(values)) if values.size else 0.0
    if vmax <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return values / vmax


def _minmax_normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    if span <= 1e-12:
        return np.ones_like(values, dtype=float)
    return (values - vmin) / span


def _inverted_minmax_normalized(values: np.ndarray) -> np.ndarray:
    return 1.0 - _minmax_normalized(values)


def _layer_medoid_proximity_scores(
    distance_matrix: np.ndarray,
    layers: np.ndarray,
    candidate_array: np.ndarray,
) -> np.ndarray:
    """Proxy for EBRC min Distance(k, C) when only layer/candidate distances are available."""
    if candidate_array.size == 0:
        return np.asarray([], dtype=float)

    mean_distances = []
    for candidate in candidate_array.astype(int):
        same_layer = candidate_array[layers[candidate_array] == layers[candidate]]
        if same_layer.size <= 1:
            mean_distances.append(0.0)
            continue
        others = same_layer[same_layer != candidate]
        mean_distances.append(float(np.mean(distance_matrix[candidate, others])))
    return _inverted_minmax_normalized(np.asarray(mean_distances, dtype=float))


def _active_node_degree(
    distance_matrix: np.ndarray,
    active_nodes: np.ndarray,
    transmission_range_m: float,
) -> Dict[int, float]:
    if active_nodes.size == 0:
        return {}
    submatrix = distance_matrix[np.ix_(active_nodes, active_nodes)]
    degree = np.count_nonzero(submatrix <= transmission_range_m, axis=1) - 1
    degree = np.maximum(degree, 0).astype(float)
    normalized = _normalized(degree)
    return {int(node): float(score) for node, score in zip(active_nodes, normalized)}


def _eulc_weight_scores(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
) -> Dict[int, float]:
    active_nodes = np.flatnonzero(energies > params.dead_energy_threshold_j).astype(int)
    if active_nodes.size == 0:
        return {}

    degree_scores = _active_node_degree(
        distance_matrix,
        active_nodes,
        params.transmission_range_m,
    )
    dmax = max(float(np.max(dist_to_sink[active_nodes])), 1e-12)
    scores: Dict[int, float] = {}
    for layer in np.unique(layers[active_nodes]):
        layer_nodes = active_nodes[layers[active_nodes] == layer]
        if layer_nodes.size == 0:
            continue
        avg_energy = float(np.mean(energies[layer_nodes]))
        for node in layer_nodes:
            node = int(node)
            if float(energies[node]) <= avg_energy:
                scores[node] = 0.0
                continue
            scores[node] = float(
                params.alpha_weight * (energies[node] / max(case.initial_energy, 1e-12))
                + params.beta_weight * (1.0 - dist_to_sink[node] / dmax)
                + params.gamma_weight * degree_scores.get(node, 0.0)
            )
    return scores


def _select_eulc_layer_cluster_heads(
    params: TunableParams,
    energies: np.ndarray,
    layers: np.ndarray,
    priority_scores: Dict[int, float],
) -> List[int]:
    active_nodes = np.flatnonzero(energies > params.dead_energy_threshold_j).astype(int)
    selected: List[int] = []
    for layer in np.unique(layers[active_nodes]):
        layer_nodes = [
            int(node)
            for node in active_nodes[layers[active_nodes] == layer]
            if energies[int(node)] > params.dead_energy_threshold_j
        ]
        if not layer_nodes:
            continue

        eligible = [node for node in layer_nodes if priority_scores.get(node, 0.0) > 0.0]
        ranked = sorted(
            eligible or layer_nodes,
            key=lambda node: (
                priority_scores.get(node, 0.0),
                float(energies[node]),
            ),
            reverse=True,
        )
        # Algorithm 1 selects the maximum-weight node per layer. Pc is kept by
        # the hybrid PSO/GA/DE decoders, not forced into this direct EULC baseline.
        selected.append(int(ranked[0]))
    return selected


def _greedy_exact_k_solution(
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    candidates: Sequence[int],
    target_ch_count: int,
    priority_scores: Dict[int, float],
) -> List[int]:
    active_nodes = np.flatnonzero(energies > params.dead_energy_threshold_j).astype(int)
    candidate_list = [
        int(candidate)
        for candidate in candidates
        if energies[int(candidate)] > params.dead_energy_threshold_j
    ]
    if not candidate_list or target_ch_count <= 0:
        return []

    selected: list[int] = []
    uncovered = set(int(node) for node in active_nodes)
    remaining = set(candidate_list)
    while len(selected) < target_ch_count and remaining and uncovered:
        best_candidate = max(
            remaining,
            key=lambda cand: (
                sum(
                    1
                    for node in uncovered
                    if distance_matrix[int(node), int(cand)] <= params.transmission_range_m
                ),
                priority_scores.get(int(cand), 0.0),
                float(energies[int(cand)]),
            ),
        )
        selected.append(int(best_candidate))
        remaining.remove(best_candidate)
        newly_covered = {
            int(node)
            for node in uncovered
            if distance_matrix[int(node), int(best_candidate)] <= params.transmission_range_m
        }
        uncovered.difference_update(newly_covered)

    ranked_remaining = sorted(
        remaining,
        key=lambda cand: (priority_scores.get(int(cand), 0.0), float(energies[int(cand)])),
        reverse=True,
    )
    for candidate in ranked_remaining:
        if len(selected) >= target_ch_count:
            break
        selected.append(int(candidate))
    return selected[:target_ch_count]


def _ranked_multilevel_solution(
    params: TunableParams,
    energies: np.ndarray,
    layers: np.ndarray,
    candidates: Sequence[int],
    target_ch_count: int,
    priority_scores: Dict[int, float],
) -> List[int]:
    candidate_list = [
        int(candidate)
        for candidate in candidates
        if energies[int(candidate)] > params.dead_energy_threshold_j
    ]
    if not candidate_list or target_ch_count <= 0:
        return []

    selected: List[int] = []
    for layer in sorted(set(int(layers[node]) for node in candidate_list)):
        layer_candidates = [node for node in candidate_list if int(layers[node]) == layer]
        if not layer_candidates:
            continue
        best = max(
            layer_candidates,
            key=lambda node: (
                priority_scores.get(node, 0.0),
                float(energies[node]),
            ),
        )
        selected.append(int(best))
        if len(selected) >= target_ch_count:
            return selected[:target_ch_count]

    ranked_remaining = sorted(
        [node for node in candidate_list if node not in set(selected)],
        key=lambda node: (
            priority_scores.get(node, 0.0),
            float(energies[node]),
        ),
        reverse=True,
    )
    for node in ranked_remaining:
        if len(selected) >= target_ch_count:
            break
        selected.append(int(node))
    return selected[:target_ch_count]


def _best_ranked_solution(
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    candidates: Sequence[int],
    target_ch_count: int,
    priority_scores: Dict[int, float],
    multilevel: bool,
) -> List[int]:
    if multilevel:
        initial = _ranked_multilevel_solution(
            params,
            energies,
            layers,
            candidates,
            target_ch_count,
            priority_scores,
        )
    else:
        initial = _greedy_exact_k_solution(
            params,
            distance_matrix,
            energies,
            candidates,
            target_ch_count,
            priority_scores,
        )
    if initial:
        return initial
    return _greedy_exact_k_solution(
        params,
        distance_matrix,
        energies,
        candidates,
        target_ch_count,
        priority_scores,
    )


def _run_ranked_baseline(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    candidates: Sequence[int],
    priority_scores: Dict[int, float],
    convergence_callback: Callable[[int, float], None] | None,
    multilevel: bool = False,
) -> Tuple[List[int], Dict[int, List[int]]]:
    candidates = _ensure_candidate_coverage(
        energies,
        distance_matrix,
        candidates,
        params.transmission_range_m,
        params.dead_energy_threshold_j,
    )
    if not candidates:
        return [], {}

    active_node_count = int(np.count_nonzero(energies > params.dead_energy_threshold_j))
    target_ch_count = _target_cluster_head_count(
        active_node_count,
        len(candidates),
        params.cluster_head_ratio,
    )
    selected = _best_ranked_solution(
        params,
        distance_matrix,
        energies,
        layers,
        candidates,
        target_ch_count,
        priority_scores,
        multilevel,
    )
    cost_normalization = _build_paper_cost_normalization(
        distance_matrix,
        energies,
        dist_to_sink,
        candidates,
        params.dead_energy_threshold_j,
    )
    alternatives = [selected]
    greedy_selected = _greedy_exact_k_solution(
        params,
        distance_matrix,
        energies,
        candidates,
        target_ch_count,
        priority_scores,
    )
    if greedy_selected and set(greedy_selected) != set(selected):
        alternatives.append(greedy_selected)

    selected = min(
        alternatives,
        key=lambda solution: evaluate_legacy_solution_score(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            solution,
            candidates,
            cost_normalization,
        ),
    )
    if convergence_callback is not None:
        score = evaluate_legacy_solution_score(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            selected,
            candidates,
            cost_normalization,
        )
        convergence_callback(0, score)

    return _finalize_solution(
        case,
        params,
        distance_matrix,
        energies,
        layers,
        dist_to_sink,
        selected,
        candidates,
        cost_normalization,
    )
