from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..algorithms.base import NetworkState, ProtocolSelection


@dataclass(frozen=True)
class EULCOptimizationProblem:
    state: NetworkState
    candidate_node_ids: tuple[int, ...]
    convergence_callback: Callable[[int, float], None] | None = None


class Optimizer(ABC):
    name: str

    @abstractmethod
    def optimize(
        self,
        problem: EULCOptimizationProblem,
        rng: np.random.Generator,
    ) -> ProtocolSelection:
        ...

    @abstractmethod
    def get_diagnostics(self) -> dict[str, object]:
        ...
