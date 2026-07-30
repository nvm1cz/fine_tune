from __future__ import annotations

from typing import List

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams


def select_eulc_candidates(
    case: SimulationCase,
    params: TunableParams,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    neighbor_degree: np.ndarray,
    distance_matrix: np.ndarray,
) -> List[int]:
    candidates: List[int] = []
    unique_layers = np.unique(layers)
    dmin = float(np.min(dist_to_sink))
    dmax = max(float(np.max(dist_to_sink)), 1e-9)
    if params.competition_radius_mode != "dynamic_eulc":
        raise ValueError("competition_radius_mode must be dynamic_eulc")
    c = float(params.competition_adjustment_factor)
    if not 0.0 <= c <= 1.0:
        raise ValueError("competition_adjustment_factor must be in [0, 1]")
    r0 = params.layer_r0_m
    candidate_ratio = params.eulc_candidate_ratio
    candidate_weights: dict[int, float] = {}

    for layer in unique_layers:
        layer_nodes = np.where(layers == layer)[0]
        active_layer_nodes = layer_nodes[
            energies[layer_nodes] > params.dead_energy_threshold_j
        ]
        if len(active_layer_nodes) == 0:
            continue

        layer_energies = energies[active_layer_nodes]
        avg_energy = float(np.mean(layer_energies))
        weights: list[tuple[float, int]] = []
        for idx in active_layer_nodes:
            # Algorithm 1 uses Eres(i) > Eavg, not >=.
            if energies[idx] <= avg_energy:
                continue
            weight = (
                params.alpha_weight * (energies[idx] / case.initial_energy)
                + params.beta_weight * (1.0 - dist_to_sink[idx] / dmax)
                + params.gamma_weight * neighbor_degree[idx]
            )
            weights.append((float(weight), int(idx)))
            candidate_weights[int(idx)] = float(weight)

        if not weights:
            alive_nodes = [int(idx) for idx in active_layer_nodes]
            if alive_nodes:
                sorted_by_energy = sorted(alive_nodes, key=lambda i: float(energies[i]), reverse=True)
                take = max(1, int(candidate_ratio * len(active_layer_nodes)))
                candidates.extend(sorted_by_energy[:take])
                for idx in sorted_by_energy[:take]:
                    candidate_weights[idx] = float(energies[idx] / case.initial_energy)
            continue

        weights.sort(reverse=True)
        take = max(1, int(candidate_ratio * len(active_layer_nodes)))
        candidates.extend(idx for _, idx in weights[:take])

    unique_candidates = list(dict.fromkeys(candidates))
    alive_candidates = [
        idx
        for idx in unique_candidates
        if energies[idx] > params.dead_energy_threshold_j
    ]

    if alive_candidates:
        sorted_candidates = sorted(
            alive_candidates,
            key=lambda idx: candidate_weights.get(idx, 0.0),
            reverse=True,
        )
        selected: list[int] = []
        for cand in sorted_candidates:
            di = float(dist_to_sink[cand])
            denom = dmax - dmin
            ri = r0 if denom == 0 else r0 * (1 - c * (di - dmin) / denom)
            within = bool(selected and np.any(distance_matrix[cand, selected] <= ri))
            if not within:
                selected.append(cand)
        for node_idx in np.flatnonzero(energies > params.dead_energy_threshold_j):
            if not selected or np.all(distance_matrix[int(node_idx), selected] > params.transmission_range_m):
                selected.append(int(node_idx))
        return selected

    return [
        int(idx)
        for idx in range(case.node_count)
        if energies[idx] > params.dead_energy_threshold_j
    ]
