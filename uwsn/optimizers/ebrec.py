from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from .baselines import (
    _layer_medoid_proximity_scores,
    _minmax_normalized,
    _run_ranked_baseline,
)


def run_ebrec_cluster_head_selection(
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
    """EBREC baseline from the uploaded paper: max residual energy, min Distance(k, C)."""
    del rng
    candidate_array = np.asarray(list(dict.fromkeys(int(c) for c in candidates)), dtype=int)
    if candidate_array.size == 0:
        return [], {}

    energy_score = _minmax_normalized(energies[candidate_array])
    center_proximity_score = _layer_medoid_proximity_scores(
        distance_matrix,
        layers,
        candidate_array,
    )
    priority_scores = {
        int(candidate): float(0.5 * energy + 0.5 * center)
        for candidate, energy, center in zip(
            candidate_array,
            energy_score,
            center_proximity_score,
        )
    }
    return _run_ranked_baseline(
        case,
        params,
        distance_matrix,
        energies,
        layers,
        dist_to_sink,
        candidates,
        priority_scores,
        convergence_callback,
        multilevel=True,
    )
