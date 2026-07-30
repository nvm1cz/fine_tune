from __future__ import annotations

from itertools import product
from typing import Mapping, Sequence

import numpy as np

from ..models.channel import channel_quality
from ..models.energy import aggregation_energy_j, receive_energy_j, transmit_energy_j
from ..objective import (
    CandidateEvaluation,
    CandidateSolution,
    EnergyDelayObjective,
    EvaluationContext,
    ObjectiveResult,
    expected_attempts_limited,
)
from .planning import AssignmentResult, RoutePlan
from .transmission import contention_multiplier
from ..run_config import TunableParams


def infeasible_objective(reason: str) -> ObjectiveResult:
    return ObjectiveResult(
        objective_value=float("inf"),
        raw_energy_j=0.0,
        raw_average_delay_s=0.0,
        normalized_energy=float("inf"),
        normalized_delay=float("inf"),
        feasible=False,
        infeasible_reasons=(reason,),
        route_success_probabilities={},
        expected_attempts_by_link={},
        energy_by_link={},
        delay_by_link={},
    )


def predict_post_assignment_energy(
    assignment: AssignmentResult,
    context: EvaluationContext,
) -> np.ndarray:
    """Predict energy after member collection without mutating the snapshot."""
    params = context.params
    positions = np.asarray(context.node_positions, dtype=float)
    predicted = np.asarray(context.residual_energy_j, dtype=float).copy()
    pairwise = positions[:, None, :] - positions[None, :, :]
    neighbor_count = np.count_nonzero(
        np.linalg.norm(pairwise, axis=2) <= params.transmission_range_m, axis=1
    ) - 1
    contention = contention_multiplier(params, neighbor_count)
    alive = np.flatnonzero(predicted > params.dead_energy_threshold_j)
    control = aggregation_energy_j(params.broadcast_packet_size_bits, params)
    predicted[alive] -= control * contention[alive]

    for ch, members in assignment.assignments.items():
        if predicted[ch] <= params.dead_energy_threshold_j:
            continue
        received = 0.0
        for member in members:
            if member == ch:
                continue
            distance = float(np.linalg.norm(positions[member] - positions[ch]))
            attempts = 1.0
            success = 1.0
            if params.enable_packet_errors:
                quality = channel_quality(
                    distance, context.frequency_khz, context.packet_bits, params
                )
                attempts, success = expected_attempts_limited(
                    quality["packet_success_probability"],
                    context.max_retries if params.enable_retransmission else 0,
                    context.normalization_epsilon,
                )
            predicted[member] -= (
                attempts
                * transmit_energy_j(distance, context.packet_bits, params)
                * contention[member]
            )
            predicted[ch] -= (
                attempts
                * receive_energy_j(context.packet_bits, params)
                * contention[ch]
            )
            received += success
        predicted[ch] -= (
            (1.0 + received)
            * aggregation_energy_j(context.packet_bits, params)
            * contention[ch]
        )
    return predicted


def _routes_from_parents(
    selected: tuple[int, ...],
    parents: Mapping[int, int | None],
) -> RoutePlan:
    routes: dict[int, tuple[int | None, ...]] = {}
    for source in selected:
        route: list[int | None] = []
        current = source
        visited = {source}
        for _ in range(len(selected) + 1):
            if current not in parents:
                return RoutePlan.from_routes(
                    routes, is_feasible=False, invalid_reason="dangling_parent"
                )
            hop = parents[current]
            route.append(hop)
            if hop is None:
                routes[source] = tuple(route)
                break
            if hop in visited:
                return RoutePlan.from_routes(
                    routes, is_feasible=False, invalid_reason="broken_route"
                )
            visited.add(hop)
            current = int(hop)
        else:
            return RoutePlan.from_routes(
                routes, is_feasible=False, invalid_reason="broken_route"
            )
    return RoutePlan.from_routes(routes)


def optimize_route_plan(
    selected_cluster_heads: Sequence[int],
    assignment: AssignmentResult,
    context: EvaluationContext,
    evaluator: EnergyDelayObjective | None = None,
) -> CandidateEvaluation:
    """Search network-wide parent plans and rank every plan with the unchanged J.

    J is not additive, so this intentionally does not label a Dijkstra edge score as
    equivalent to J. It enumerates deterministic acyclic parent combinations up to the
    configured search limit and evaluates every complete plan with EnergyDelayObjective.
    """
    selected = tuple(sorted(dict.fromkeys(int(ch) for ch in selected_cluster_heads)))
    if not assignment.is_feasible:
        solution = CandidateSolution(selected, assignment.assignments, {})
        result = infeasible_objective(assignment.invalid_reason or "unassigned_alive_node")
        return CandidateEvaluation(solution, result, False, result.invalid_reason)
    if not selected:
        solution = CandidateSolution((), assignment.assignments, {})
        result = infeasible_objective("broken_route")
        return CandidateEvaluation(solution, result, False, result.invalid_reason)

    evaluator = evaluator or EnergyDelayObjective()
    positions = np.asarray(context.node_positions, dtype=float)
    sink = np.asarray(context.sink_position, dtype=float)
    predicted = predict_post_assignment_energy(assignment, context)
    params = context.params
    options: dict[int, tuple[int | None, ...]] = {}
    for ch in selected:
        choices: list[int | None] = []
        if (
            float(np.linalg.norm(positions[ch] - sink))
            <= params.transmission_range_m
        ):
            choices.append(None)
        relays = [
            other
            for other in selected
            if other != ch
            and predicted[other] > params.dead_energy_threshold_j
            and float(np.linalg.norm(positions[ch] - positions[other]))
            <= params.transmission_range_m
            and float(np.linalg.norm(positions[other] - sink))
            < float(np.linalg.norm(positions[ch] - sink))
        ]
        relays.sort(
            key=lambda node: (
                float(np.linalg.norm(positions[node] - sink)),
                int(node),
            )
        )
        choices.extend(relays)
        if not choices:
            solution = CandidateSolution(selected, assignment.assignments, {})
            result = infeasible_objective("broken_route")
            return CandidateEvaluation(solution, result, False, result.invalid_reason)
        options[ch] = tuple(choices)

    limit = max(1, int(getattr(params, "routing_plan_search_limit", 128)))
    best: CandidateEvaluation | None = None
    evaluated = 0
    option_lists = [options[ch] for ch in selected]
    for combination in product(*option_lists):
        if evaluated >= limit:
            break
        parents = dict(zip(selected, combination))
        plan = _routes_from_parents(selected, parents)
        if not plan.is_feasible:
            continue
        solution = CandidateSolution(
            selected,
            assignment.assignments,
            plan.route_by_ch,
            plan,
        )
        result = evaluator.evaluate(solution, context)
        evaluated += 1
        candidate = CandidateEvaluation(
            solution, result, result.is_feasible, result.invalid_reason
        )
        if best is None or result.objective_J < best.objective_result.objective_J:
            best = candidate
    if best is None:
        solution = CandidateSolution(selected, assignment.assignments, {})
        result = infeasible_objective("broken_route")
        return CandidateEvaluation(solution, result, False, result.invalid_reason)

    plan = best.solution.route_plan
    assert plan is not None
    enriched = RoutePlan(
        parent_by_ch=plan.parent_by_ch,
        route_by_ch=plan.route_by_ch,
        forwarding_load_by_ch=plan.forwarding_load_by_ch,
        links=plan.links,
        is_feasible=plan.is_feasible,
        invalid_reason=plan.invalid_reason,
        diagnostics={
            **dict(plan.diagnostics),
            "routing_mode": "joint_optimized",
            "plans_evaluated": evaluated,
            "search_limit": limit,
        },
        plan_id=plan.plan_id,
    )
    solution = CandidateSolution(
        best.solution.selected_cluster_heads,
        best.solution.assignments,
        enriched.route_by_ch,
        enriched,
    )
    return CandidateEvaluation(
        solution,
        best.objective_result,
        best.is_feasible,
        best.invalid_reason,
    )
