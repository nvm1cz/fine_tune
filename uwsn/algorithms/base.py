from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from ..objective import ObjectiveResult
from ..routing.planning import RoutePlan


ConvergenceCallback = Callable[[int, float], None]


@dataclass(frozen=True)
class NetworkState:
    case: SimulationCase
    params: TunableParams
    positions: np.ndarray
    energies: np.ndarray
    layers: np.ndarray
    dist_to_sink: np.ndarray
    neighbor_degree: np.ndarray
    distance_matrix: np.ndarray


@dataclass(frozen=True)
class ProtocolSelection:
    cluster_heads: tuple[int, ...]
    assignments: dict[int, list[int]]
    diagnostics: dict[str, object] = field(default_factory=dict)
    route_plan: RoutePlan | None = None
    objective_result: ObjectiveResult | None = None


class UWSNProtocol(ABC):
    algorithm_id: str
    implementation_status: str = "partial"
    base_protocol: str | None = None
    optimizer_name: str | None = None

    def initialize(self, network_state: NetworkState) -> None:
        del network_state

    def prepare_round(self, round_index: int, network_state: NetworkState) -> None:
        del round_index, network_state

    @abstractmethod
    def select_cluster_heads(
        self,
        round_index: int,
        network_state: NetworkState,
        protocol_rng: np.random.Generator,
        optimizer_rng: np.random.Generator,
        convergence_callback: ConvergenceCallback | None = None,
    ) -> ProtocolSelection:
        """Select CHs and produce member assignments for one refresh."""

    def get_diagnostics(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "implementation_status": self.implementation_status,
            "base_protocol": self.base_protocol,
            "optimizer_name": self.optimizer_name,
        }


class AlgorithmNotImplementedError(NotImplementedError):
    pass
