from __future__ import annotations

from typing import Callable

import numpy as np

from ..optimizers.base import EULCOptimizationProblem, Optimizer
from .base import ConvergenceCallback, NetworkState, ProtocolSelection, UWSNProtocol
from .eulc_components import EULCComponents


SelectionFunction = Callable[..., tuple[list[int], dict[int, list[int]]]]


class ExistingProtocolAdapter(UWSNProtocol):
    """Class wrapper around a distinct existing non-metaheuristic implementation."""

    selection_function: SelectionFunction
    uses_eulc_candidates = True

    def __init__(self, eulc: EULCComponents | None = None) -> None:
        self.eulc = eulc or EULCComponents()
        self._diagnostics: dict[str, object] = {}

    def select_cluster_heads(
        self,
        round_index: int,
        network_state: NetworkState,
        protocol_rng: np.random.Generator,
        optimizer_rng: np.random.Generator,
        convergence_callback: ConvergenceCallback | None = None,
    ) -> ProtocolSelection:
        del round_index, optimizer_rng
        state = network_state
        candidates = (
            self.eulc.generate_candidates(state)
            if self.uses_eulc_candidates
            else [int(i) for i in np.flatnonzero(
                state.energies > state.params.dead_energy_threshold_j
            )]
        )
        chs, assignments = self.selection_function(
            state.case, state.params, state.distance_matrix, state.energies,
            state.layers, state.dist_to_sink, candidates, protocol_rng,
            convergence_callback=convergence_callback,
        )
        self._diagnostics = {
            **super().get_diagnostics(),
            "selected_ch_count": len(chs),
            "candidate_count": len(candidates),
        }
        return ProtocolSelection(tuple(chs), assignments, dict(self._diagnostics))

    def get_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics) or super().get_diagnostics()


class EULCOptimizedProtocol(UWSNProtocol):
    base_protocol = "eulc"
    implementation_status = "partial"

    def __init__(self, algorithm_id: str, optimizer: Optimizer, eulc: EULCComponents | None = None) -> None:
        self.algorithm_id = algorithm_id
        self.optimizer = optimizer
        self.optimizer_name = optimizer.name
        self.eulc = eulc or EULCComponents()
        self._diagnostics: dict[str, object] = {}

    def select_cluster_heads(
        self,
        round_index: int,
        network_state: NetworkState,
        protocol_rng: np.random.Generator,
        optimizer_rng: np.random.Generator,
        convergence_callback: ConvergenceCallback | None = None,
    ) -> ProtocolSelection:
        del round_index, protocol_rng
        candidates = tuple(self.eulc.generate_candidates(network_state))
        result = self.optimizer.optimize(
            EULCOptimizationProblem(network_state, candidates, convergence_callback),
            optimizer_rng,
        )
        self._diagnostics = {
            **super().get_diagnostics(),
            "candidate_count": len(candidates),
            **result.diagnostics,
        }
        return ProtocolSelection(
            result.cluster_heads,
            result.assignments,
            dict(self._diagnostics),
            result.route_plan,
            result.objective_result,
        )

    def get_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics) or super().get_diagnostics()
