from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from ..objective import EvaluationContext, ObjectiveEvaluator, ObjectiveResult


AssignmentBuilder = Callable[[tuple[int, ...]], Mapping[int, tuple[int, ...]]]
RouteBuilder = Callable[
    [tuple[int, ...], Mapping[int, tuple[int, ...]]],
    Mapping[int, tuple[int | None, ...]],
]


@dataclass(frozen=True)
class CHOptimizationContext:
    node_positions: np.ndarray
    residual_energy_j: np.ndarray
    alive_node_ids: tuple[int, ...]
    objective_context: EvaluationContext
    objective_evaluator: ObjectiveEvaluator
    assignment_builder: AssignmentBuilder
    route_builder: RouteBuilder
    rng: np.random.Generator
    eulc_scores: Mapping[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CHOptimizationResult:
    selected_ch_ids: tuple[int, ...]
    objective_result: ObjectiveResult
    best_objective: float
    iterations_completed: int
    convergence_history: tuple[float, ...]
    diagnostics: dict[str, object] = field(default_factory=dict)


class CHSelectionOptimizer(Protocol):
    def optimize(
        self,
        candidate_node_ids: Sequence[int],
        number_of_cluster_heads: int,
        context: CHOptimizationContext,
    ) -> CHOptimizationResult:
        ...
