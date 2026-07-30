from __future__ import annotations

from .candidate_selection import select_eulc_candidates
from .base import NetworkState


class EULCComponents:
    """Shared, single implementation of the existing EULC preprocessing."""

    def generate_candidates(self, state: NetworkState) -> list[int]:
        return select_eulc_candidates(
            state.case,
            state.params,
            state.energies,
            state.layers,
            state.dist_to_sink,
            state.neighbor_degree,
            state.distance_matrix,
        )
