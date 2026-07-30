from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from .baselines import (
    _build_paper_cost_normalization,
    _ensure_candidate_coverage,
    _finalize_solution,
    evaluate_legacy_solution_score,
)


def run_eulc_cluster_head_selection(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    candidates: Sequence[int],
    rng: np.random.Generator,
    convergence_callback: Callable[[int, float], None] | None = None,
) -> Tuple[List[int], Dict[int, List[int]]]:
    """Direct EULC baseline: promote the EULC candidate set directly to CHs."""
    del rng
    eulc_candidates = _ensure_candidate_coverage(
        energies,
        distance_matrix,
        candidates,
        params.transmission_range_m,
        params.dead_energy_threshold_j,
    )
    if not eulc_candidates:
        return [], {}

    selected = list(eulc_candidates)
    cost_normalization = _build_paper_cost_normalization(
        distance_matrix,
        energies,
        dist_to_sink,
        eulc_candidates,
        params.dead_energy_threshold_j,
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
            eulc_candidates,
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
        eulc_candidates,
        cost_normalization,
    )
