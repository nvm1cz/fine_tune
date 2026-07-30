from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from .common import (
    _build_paper_cost_normalization,
    _finalize_solution,
    _target_cluster_head_count,
    evaluate_legacy_solution_score,
)


def run_leach_cluster_head_selection(
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
    """Random LEACH-style CH election used as a baseline."""
    del candidates
    alive_candidates = [
        int(idx)
        for idx in np.flatnonzero(energies > params.dead_energy_threshold_j)
    ]
    if not alive_candidates:
        return [], {}

    target_ch_count = _target_cluster_head_count(
        len(alive_candidates),
        len(alive_candidates),
        params.cluster_head_ratio,
    )
    selected = rng.choice(alive_candidates, size=target_ch_count, replace=False).astype(int).tolist()
    cost_normalization = _build_paper_cost_normalization(
        distance_matrix,
        energies,
        dist_to_sink,
        alive_candidates,
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
            alive_candidates,
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
        alive_candidates,
        cost_normalization,
    )
