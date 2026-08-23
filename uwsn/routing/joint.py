from __future__ import annotations

from itertools import product
import time
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


def beam_parent_combinations(
    option_lists: Sequence[Sequence[int | None]], beam_width: int,
) -> list[tuple[int | None, ...]]:
    """Keep deterministic low-rank partial parent combinations."""
    beam: list[tuple[int, tuple[int | None, ...]]] = [(0, ())]
    for options in option_lists:
        expanded = [
            (score + rank, combination + (option,))
            for score, combination in beam
            for rank, option in enumerate(options)
        ]
        expanded.sort(key=lambda item: (
            item[0], tuple(-1 if value is None else int(value) for value in item[1])
        ))
        beam = expanded[:beam_width]
    return [combination for _, combination in beam]


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
    evaluator: EnergyDelayObjective | None = None,
) -> np.ndarray:
    """Predict energy after member collection without mutating the snapshot."""
    params = context.params
    positions = np.asarray(context.node_positions, dtype=float)
    predicted = np.asarray(context.residual_energy_j, dtype=float).copy()
    if evaluator is None or evaluator.cache.disabled:
        pairwise = positions[:, None, :] - positions[None, :, :]
        neighbor_count = np.count_nonzero(
            np.linalg.norm(pairwise, axis=2) <= params.transmission_range_m, axis=1
        ) - 1
        contention = contention_multiplier(params, neighbor_count)
    else:
        sink = np.asarray(context.sink_position, dtype=float)
        evaluator.cache.prepare(context, params, positions, sink)
        contention = evaluator.cache.topology(
            positions, sink, params
        ).contention_multiplier
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
    evaluator = evaluator or EnergyDelayObjective()
    route_started = time.perf_counter() if evaluator.cache.profile else 0.0
    params = context.params
    search_mode = getattr(params, "routing_search_mode", "exhaustive")
    if search_mode not in {"exhaustive", "beam"}:
        raise ValueError(f"Unsupported routing_search_mode: {search_mode}")
    configured_limit = max(1, int(getattr(params, "routing_plan_search_limit", 128)))
    beam_width = max(1, int(getattr(params, "routing_beam_width", 8)))
    relay_limit = max(
        1, int(getattr(params, "routing_relay_candidates_per_ch", 2))
    )
    limit = min(configured_limit, beam_width) if search_mode == "beam" else configured_limit

    def finish(
        value: CandidateEvaluation,
        evaluated: int,
    ) -> CandidateEvaluation:
        evaluator.cache.record_route_search(evaluated, limit)
        evaluator.cache.record_route_seconds(route_started)
        return value

    selected = tuple(sorted(dict.fromkeys(int(ch) for ch in selected_cluster_heads)))
    evaluator.cache.record_outer_ch_candidate(selected)
    if not assignment.is_feasible:
        solution = CandidateSolution(selected, assignment.assignments, {})
        result = infeasible_objective(assignment.invalid_reason or "unassigned_alive_node")
        return finish(
            CandidateEvaluation(solution, result, False, result.invalid_reason), 0
        )
    if not selected:
        solution = CandidateSolution((), assignment.assignments, {})
        result = infeasible_objective("broken_route")
        return finish(
            CandidateEvaluation(solution, result, False, result.invalid_reason), 0
        )

    positions = np.asarray(context.node_positions, dtype=float)
    sink = np.asarray(context.sink_position, dtype=float)
    evaluator.cache.prepare(context, params, positions, sink)
    topology = (
        None
        if evaluator.cache.disabled
        else evaluator.cache.topology(positions, sink, params)
    )
    predicted = predict_post_assignment_energy(assignment, context, evaluator)
    fixed_member_links = evaluator.prepare_fixed_member_links(
        selected, assignment.assignments, context
    )
    options: dict[int, tuple[int | None, ...]] = {}
    for ch in selected:
        choices: list[int | None] = []
        reaches_sink = (
            float(np.linalg.norm(positions[ch] - sink))
            <= params.transmission_range_m
            if topology is None
            else bool(topology.sink_reachable[ch])
        )
        if reaches_sink:
            choices.append(None)
        relays = [
            other
            for other in selected
            if other != ch
            and predicted[other] > params.dead_energy_threshold_j
            and (
                float(np.linalg.norm(positions[ch] - positions[other]))
                <= params.transmission_range_m
                if topology is None
                else bool(topology.neighbor_mask[ch, other])
            )
            and (
                float(np.linalg.norm(positions[other] - sink))
                < float(np.linalg.norm(positions[ch] - sink))
                if topology is None
                else float(topology.distances_to_sink_m[other])
                < float(topology.distances_to_sink_m[ch])
            )
        ]
        relays.sort(
            key=lambda node: (
                (
                    float(np.linalg.norm(positions[node] - sink))
                    if topology is None
                    else float(topology.distances_to_sink_m[node])
                ),
                int(node),
            )
        )
        if search_mode == "beam":
            relays = relays[:relay_limit]
        choices.extend(relays)
        if not choices:
            solution = CandidateSolution(selected, assignment.assignments, {})
            result = infeasible_objective("broken_route")
            return finish(
                CandidateEvaluation(solution, result, False, result.invalid_reason), 0
            )
        options[ch] = tuple(choices)

    best: CandidateEvaluation | None = None
    evaluated = 0
    option_lists = [options[ch] for ch in selected]
    combinations = (
        beam_parent_combinations(option_lists, beam_width)
        if search_mode == "beam"
        else product(*option_lists)
    )
    for combination in combinations:
        if evaluated >= limit:
            break
        parents = dict(zip(selected, combination))
        plan = _routes_from_parents(selected, parents)
        evaluator.cache.record_generated_route_plan(plan.plan_id)
        if not plan.is_feasible:
            continue
        solution = CandidateSolution(
            selected,
            assignment.assignments,
            plan.route_by_ch,
            plan,
        )
        result = evaluator.evaluate(
            solution, context, fixed_member_links=fixed_member_links
        )
        evaluated += 1
        candidate = CandidateEvaluation(
            solution, result, result.is_feasible, result.invalid_reason
        )
        if best is None or result.objective_J < best.objective_result.objective_J:
            best = candidate
    if best is None:
        solution = CandidateSolution(selected, assignment.assignments, {})
        result = infeasible_objective("broken_route")
        return finish(
            CandidateEvaluation(solution, result, False, result.invalid_reason),
            evaluated,
        )

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
            "routing_mode": (
                "joint_optimized_beam" if search_mode == "beam"
                else "joint_optimized"
            ),
            "plans_evaluated": evaluated,
            "search_limit": limit,
            "routing_search_mode": search_mode,
            "beam_width": beam_width if search_mode == "beam" else None,
            "relay_candidates_per_ch": relay_limit if search_mode == "beam" else None,
        },
        plan_id=plan.plan_id,
    )
    solution = CandidateSolution(
        best.solution.selected_cluster_heads,
        best.solution.assignments,
        enriched.route_by_ch,
        enriched,
    )
    return finish(
        CandidateEvaluation(
            solution,
            best.objective_result,
            best.is_feasible,
            best.invalid_reason,
        ),
        evaluated,
    )
