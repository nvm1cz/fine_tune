from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from .baselines import _minmax_normalized, _run_ranked_baseline


def run_eeumc_cluster_head_selection(
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
    """EEUMC baseline from the uploaded paper: highest M_combined per level."""
    del rng
    candidate_array = np.asarray(list(dict.fromkeys(int(c) for c in candidates)), dtype=int)
    if candidate_array.size == 0:
        return [], {}

    energy_normalized = _minmax_normalized(energies[candidate_array])
    distance_normalized = _minmax_normalized(dist_to_sink[candidate_array])
    alpha = min(1.0, max(0.0, float(params.alpha_weight)))
    combined_metric = alpha * energy_normalized + (1.0 - alpha) * distance_normalized
    priority_scores = {
        int(candidate): float(score)
        for candidate, score in zip(candidate_array, combined_metric)
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
