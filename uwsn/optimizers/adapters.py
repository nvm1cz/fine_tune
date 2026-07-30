from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..algorithms.base import ProtocolSelection
from .base import EULCOptimizationProblem, Optimizer
from .common import evaluate_complete_solution


OptimizerFunction = Callable[..., tuple[list[int], dict[int, list[int]]]]


class ExistingOptimizerAdapter(Optimizer):
    """Adapter preserving the current optimizer implementation and shared objective path."""

    def __init__(self, name: str, function: OptimizerFunction) -> None:
        self.name = name
        self._function = function
        self._diagnostics: dict[str, object] = {}

    def optimize(
        self,
        problem: EULCOptimizationProblem,
        rng: np.random.Generator,
    ) -> ProtocolSelection:
        state = problem.state
        history: list[float] = []
        detailed_history: list[dict[str, object]] = []

        def record(iteration: int, score: float) -> None:
            history.append(float(score))
            if problem.convergence_callback is not None:
                problem.convergence_callback(iteration, score)

        started = time.perf_counter()
        chs, assignments = self._function(
            state.case,
            state.params,
            state.distance_matrix,
            state.energies,
            state.layers,
            state.dist_to_sink,
            problem.candidate_node_ids,
            rng,
            convergence_callback=record,
            positions=state.positions,
            diagnostics_callback=detailed_history.append,
        )
        runtime = time.perf_counter() - started
        complete = evaluate_complete_solution(
            state.case,
            state.params,
            state.distance_matrix,
            state.energies,
            state.layers,
            state.dist_to_sink,
            chs,
            problem.candidate_node_ids,
            positions=state.positions,
        )
        objective = complete.objective_result
        solution = complete.solution
        assignments = {
            int(ch): list(members) for ch, members in solution.assignments.items()
        }
        chs = list(solution.selected_cluster_heads)
        self._diagnostics = {
            "optimizer_name": self.name,
            "best_fitness": min(history) if history else None,
            "final_fitness": history[-1] if history else None,
            "iterations_completed": max(0, len(history) - 1),
            "convergence_history": tuple(detailed_history),
            "optimizer_runtime_seconds": runtime,
            "selected_ch_count": len(chs),
            # Existing functions cache duplicate CH sets, so exact objective-call count
            # is not exposed yet. Do not fabricate it.
            "fitness_evaluations": (
                detailed_history[-1]["evaluations"] if detailed_history else None
            ),
            "constraint_repairs": None,
            "invalid_solutions": (
                detailed_history[-1]["invalid_count"] if detailed_history else None
            ),
            "objective_name": objective.objective_name,
            "objective_direction": objective.objective_direction,
            "lambda": objective.lambda_weight,
            "objective_J": objective.objective_J,
            "energy_term_e": objective.energy_term_e,
            "delay_term_d": objective.delay_term_d,
            "energy_consumed": objective.energy_consumed,
            "available_energy": objective.available_energy,
            "average_end_to_end_delay": objective.average_end_to_end_delay,
            "reference_delay": objective.reference_delay,
            "delivered_packet_count": objective.delivered_packet_count,
            "required_packet_count": objective.required_packet_count,
            "is_feasible": objective.is_feasible,
            "invalid_reason": objective.invalid_reason,
            "assignment_mode": "strongest_rssi",
            "routing_mode": "joint_optimized",
            "objective_route_plan_id": solution.route_plan.plan_id,
            "route_plan_id": solution.route_plan.plan_id,
        }
        return ProtocolSelection(
            tuple(chs),
            assignments,
            dict(self._diagnostics),
            solution.route_plan,
            objective,
        )

    def get_diagnostics(self) -> dict[str, object]:
        return dict(self._diagnostics)
