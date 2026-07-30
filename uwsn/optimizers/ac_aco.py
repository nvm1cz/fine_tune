from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..cases import SimulationCase
from ..objective import (
    CandidateSolution,
    EnergyDelayObjective,
    EvaluationContext,
    ObjectiveResult,
)
from ..routing.joint import optimize_route_plan
from ..routing.planning import AssignmentResult, assign_members_strongest_rssi
from ..run_config import TunableParams
from .interfaces import CHOptimizationContext, CHOptimizationResult
from .common import _target_cluster_head_count


@dataclass(frozen=True)
class ACACOConfig:
    ants: int
    iterations: int
    alpha: float
    adaptive_beta: bool
    beta_start: float
    beta_end: float
    adaptive_rho: bool
    rho_start: float
    rho_end: float
    initial_pheromone: float
    tau_min: float
    tau_max: float
    deposit_q: float
    deposit_mode: str
    elite_weight: float
    chaos_enabled: bool
    chaos_r: float
    chaos_strength: float
    chaos_initial_state: float | None
    stagnation_limit: int
    stagnation_mode: str
    heuristic_energy_weight: float
    heuristic_eulc_weight: float
    heuristic_centrality_weight: float
    objective_epsilon: float

    @classmethod
    def from_params(cls, params: TunableParams) -> "ACACOConfig":
        return cls(
            ants=int(params.ac_aco_ants),
            iterations=int(params.ac_aco_iterations),
            alpha=float(params.ac_aco_alpha),
            adaptive_beta=bool(params.ac_aco_adaptive_beta),
            beta_start=float(params.ac_aco_beta_start),
            beta_end=float(params.ac_aco_beta_end),
            adaptive_rho=bool(params.ac_aco_adaptive_rho),
            rho_start=float(params.ac_aco_rho_start),
            rho_end=float(params.ac_aco_rho_end),
            initial_pheromone=float(params.ac_aco_initial_pheromone),
            tau_min=float(params.ac_aco_tau_min),
            tau_max=float(params.ac_aco_tau_max),
            deposit_q=float(params.ac_aco_deposit_q),
            deposit_mode=str(params.ac_aco_deposit_mode).lower(),
            elite_weight=float(params.ac_aco_elite_weight),
            chaos_enabled=bool(params.ac_aco_chaos_enabled),
            chaos_r=float(params.ac_aco_chaos_r),
            chaos_strength=float(params.ac_aco_chaos_strength),
            chaos_initial_state=params.ac_aco_chaos_initial_state,
            stagnation_limit=int(params.ac_aco_stagnation_limit),
            stagnation_mode=str(params.ac_aco_stagnation_mode).lower(),
            heuristic_energy_weight=float(params.ac_aco_heuristic_energy_weight),
            heuristic_eulc_weight=float(params.ac_aco_heuristic_eulc_weight),
            heuristic_centrality_weight=float(
                params.ac_aco_heuristic_centrality_weight
            ),
            objective_epsilon=float(params.ac_aco_objective_epsilon),
        )


@dataclass(frozen=True)
class _EvaluatedSolution:
    objective_result: ObjectiveResult
    assignments: Mapping[int, tuple[int, ...]]
    routes: Mapping[int, tuple[int | None, ...]]


def logistic_chaos_next(state: float, chaos_r: float, epsilon: float) -> float:
    value = float(chaos_r) * float(state) * (1.0 - float(state))
    return float(np.clip(value, epsilon, 1.0 - epsilon))


def adaptive_parameter(
    start: float,
    end: float,
    iteration: int,
    iterations: int,
    enabled: bool,
) -> float:
    if not enabled:
        return float(start)
    progress = float(iteration) / max(int(iterations) - 1, 1)
    return float(start) + progress * (float(end) - float(start))


def stable_selection_probabilities(
    candidate_ids: Sequence[int],
    pheromone_by_node_id: Mapping[int, float],
    adjusted_heuristics: Mapping[int, float],
    alpha: float,
    beta: float,
    epsilon: float,
) -> np.ndarray:
    ids = tuple(int(node) for node in candidate_ids)
    if not ids:
        raise ValueError("candidate_ids must not be empty")
    log_values = np.asarray(
        [
            float(alpha) * np.log(max(float(pheromone_by_node_id[node]), epsilon))
            + float(beta) * np.log(max(float(adjusted_heuristics[node]), epsilon))
            for node in ids
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(log_values)):
        return np.full(len(ids), 1.0 / len(ids), dtype=float)
    shifted = log_values - float(np.max(log_values))
    desirability = np.exp(shifted)
    total = float(np.sum(desirability))
    if not np.isfinite(total) or total <= epsilon:
        return np.full(len(ids), 1.0 / len(ids), dtype=float)
    probabilities = desirability / total
    if not np.all(np.isfinite(probabilities)):
        return np.full(len(ids), 1.0 / len(ids), dtype=float)
    return probabilities


def evaporate_pheromones(
    pheromone_by_node_id: Mapping[int, float],
    rho: float,
    tau_min: float,
    tau_max: float,
) -> dict[int, float]:
    return {
        int(node): float(np.clip((1.0 - rho) * value, tau_min, tau_max))
        for node, value in pheromone_by_node_id.items()
    }


def deposit_pheromone(
    pheromone_by_node_id: Mapping[int, float],
    selected_ch_ids: Sequence[int],
    objective_value: float,
    deposit_q: float,
    weight: float,
    objective_epsilon: float,
    tau_min: float,
    tau_max: float,
) -> dict[int, float]:
    updated = {int(node): float(value) for node, value in pheromone_by_node_id.items()}
    deposit = float(weight) * float(deposit_q) / max(
        float(objective_value), float(objective_epsilon)
    )
    for node in selected_ch_ids:
        node_id = int(node)
        updated[node_id] = float(
            np.clip(updated[node_id] + deposit, tau_min, tau_max)
        )
    return updated


def compute_candidate_heuristics(
    candidate_node_ids: Sequence[int],
    positions: np.ndarray,
    residual_energy_j: np.ndarray,
    alive_node_ids: Sequence[int],
    *,
    energy_weight: float,
    eulc_weight: float,
    centrality_weight: float,
    eulc_scores: Mapping[int, float] | None = None,
    epsilon: float = 1.0e-12,
) -> dict[int, float]:
    candidates = np.asarray(candidate_node_ids, dtype=int)
    alive = np.asarray(alive_node_ids, dtype=int)
    positions_array = np.asarray(positions, dtype=float)
    residual = np.asarray(residual_energy_j, dtype=float)

    def normalized(values: np.ndarray) -> np.ndarray:
        low = float(np.min(values))
        high = float(np.max(values))
        if high - low <= epsilon:
            return np.ones_like(values, dtype=float)
        return (values - low) / (high - low)

    energy_score = normalized(residual[candidates])
    candidate_positions = positions_array[candidates]
    alive_positions = positions_array[alive]
    distances = np.linalg.norm(
        candidate_positions[:, None, :] - alive_positions[None, :, :], axis=2
    )
    mean_distances = np.mean(distances, axis=1)
    centrality_score = 1.0 - normalized(mean_distances)
    supplied_scores = eulc_scores or {}
    eulc_values = np.asarray(
        [float(supplied_scores.get(int(node), 0.0)) for node in candidates],
        dtype=float,
    )
    eulc_score = normalized(eulc_values) if supplied_scores else np.zeros_like(energy_score)
    values = (
        float(energy_weight) * energy_score
        + float(eulc_weight) * eulc_score
        + float(centrality_weight) * centrality_score
        + epsilon
    )
    return {
        int(node): max(float(value), epsilon)
        for node, value in zip(candidates, values)
    }


class ACACOCHOptimizer:
    def __init__(self, config: ACACOConfig) -> None:
        self.config = config

    def optimize(
        self,
        candidate_node_ids: Sequence[int],
        number_of_cluster_heads: int,
        context: CHOptimizationContext,
    ) -> CHOptimizationResult:
        candidates = tuple(int(node) for node in candidate_node_ids)
        self._validate(candidates, number_of_cluster_heads, context)
        config = self.config
        heuristics = compute_candidate_heuristics(
            candidates,
            context.node_positions,
            context.residual_energy_j,
            context.alive_node_ids,
            energy_weight=config.heuristic_energy_weight,
            eulc_weight=config.heuristic_eulc_weight,
            centrality_weight=config.heuristic_centrality_weight,
            eulc_scores=context.eulc_scores,
            epsilon=config.objective_epsilon,
        )
        pheromones = {
            node: config.initial_pheromone for node in candidates
        }
        chaos_state = self._initial_chaos_state(context.rng)
        cache: dict[tuple[int, ...], _EvaluatedSolution] = {}
        cache_hits = 0
        evaluations = 0
        feasible_count = 0
        infeasible_count = 0
        stagnation_resets = 0
        no_improvement = 0
        best_key: tuple[int, ...] | None = None
        best_evaluated: _EvaluatedSolution | None = None
        best_objective = float("inf")
        convergence: list[float] = []
        convergence_diagnostics: list[dict[str, object]] = []
        final_rho = config.rho_start
        final_beta = config.beta_start

        for iteration in range(config.iterations):
            rho = adaptive_parameter(
                config.rho_start,
                config.rho_end,
                iteration,
                config.iterations,
                config.adaptive_rho,
            )
            beta = adaptive_parameter(
                config.beta_start,
                config.beta_end,
                iteration,
                config.iterations,
                config.adaptive_beta,
            )
            final_rho, final_beta = rho, beta
            iteration_best_key: tuple[int, ...] | None = None
            iteration_best: _EvaluatedSolution | None = None
            iteration_best_objective = float("inf")
            improved = False

            for _ in range(config.ants):
                solution, chaos_state = self._construct_solution(
                    candidates,
                    number_of_cluster_heads,
                    pheromones,
                    heuristics,
                    beta,
                    chaos_state,
                    context.rng,
                )
                evaluated = cache.get(solution)
                if evaluated is None:
                    assignments = {
                        int(ch): tuple(int(node) for node in members)
                        for ch, members in context.assignment_builder(solution).items()
                    }
                    routes = {
                        int(ch): tuple(route)
                        for ch, route in context.route_builder(solution, assignments).items()
                    }
                    objective_result = context.objective_evaluator.evaluate(
                        CandidateSolution(solution, assignments, routes),
                        context.objective_context,
                    )
                    evaluated = _EvaluatedSolution(
                        objective_result=objective_result,
                        assignments=assignments,
                        routes=routes,
                    )
                    cache[solution] = evaluated
                    evaluations += 1
                    if objective_result.feasible:
                        feasible_count += 1
                    else:
                        infeasible_count += 1
                else:
                    cache_hits += 1

                objective = float(evaluated.objective_result.objective_value)
                if objective < iteration_best_objective:
                    iteration_best_objective = objective
                    iteration_best_key = solution
                    iteration_best = evaluated
                if objective < best_objective:
                    best_objective = objective
                    best_key = solution
                    best_evaluated = evaluated
                    improved = True

            pheromones = evaporate_pheromones(
                pheromones, rho, config.tau_min, config.tau_max
            )
            deposit_key, deposit_evaluated = (
                (iteration_best_key, iteration_best)
                if config.deposit_mode == "iteration_best"
                else (best_key, best_evaluated)
            )
            if (
                deposit_key is not None
                and deposit_evaluated is not None
                and deposit_evaluated.objective_result.feasible
            ):
                pheromones = deposit_pheromone(
                    pheromones,
                    deposit_key,
                    deposit_evaluated.objective_result.objective_value,
                    config.deposit_q,
                    1.0,
                    config.objective_epsilon,
                    config.tau_min,
                    config.tau_max,
                )
            if (
                config.elite_weight > 0.0
                and best_key is not None
                and best_evaluated is not None
                and best_evaluated.objective_result.feasible
            ):
                pheromones = deposit_pheromone(
                    pheromones,
                    best_key,
                    best_evaluated.objective_result.objective_value,
                    config.deposit_q,
                    config.elite_weight,
                    config.objective_epsilon,
                    config.tau_min,
                    config.tau_max,
                )

            if improved:
                no_improvement = 0
            else:
                no_improvement += 1
            if (
                config.stagnation_limit > 0
                and no_improvement >= config.stagnation_limit
            ):
                if config.stagnation_mode == "reset":
                    pheromones = {
                        node: config.initial_pheromone for node in candidates
                    }
                else:
                    mean_pheromone = float(np.mean(list(pheromones.values())))
                    pheromones = {
                        node: float(
                            np.clip(
                                0.5 * value + 0.5 * mean_pheromone,
                                config.tau_min,
                                config.tau_max,
                            )
                        )
                        for node, value in pheromones.items()
                    }
                stagnation_resets += 1
                no_improvement = 0
            convergence.append(best_objective)
            if best_evaluated is not None:
                convergence_diagnostics.append({
                    "iteration": iteration,
                    "evaluations": evaluations,
                    "best_J": best_objective,
                    "best_energy_term": best_evaluated.objective_result.energy_term_e,
                    "best_delay_term": best_evaluated.objective_result.delay_term_d,
                    "feasible_count": feasible_count,
                    "invalid_count": infeasible_count,
                })

        if best_key is None or best_evaluated is None:
            raise RuntimeError("AC-ACO did not evaluate any solution")
        return CHOptimizationResult(
            selected_ch_ids=best_key,
            objective_result=best_evaluated.objective_result,
            best_objective=best_objective,
            iterations_completed=config.iterations,
            convergence_history=tuple(convergence),
            diagnostics={
                "candidate_count": len(candidates),
                "number_of_cluster_heads": number_of_cluster_heads,
                "ant_count": config.ants,
                "feasible_solution_count": feasible_count,
                "infeasible_solution_count": infeasible_count,
                "objective_cache_hits": cache_hits,
                "objective_evaluations": evaluations,
                "stagnation_resets": stagnation_resets,
                "pheromone_min": min(pheromones.values()),
                "pheromone_max": max(pheromones.values()),
                "final_pheromone_by_node_id": dict(pheromones),
                "chaos_enabled": config.chaos_enabled,
                "final_rho": final_rho,
                "final_beta": final_beta,
                "convergence_diagnostics": tuple(convergence_diagnostics),
            },
        )

    def _construct_solution(
        self,
        candidates: tuple[int, ...],
        number_of_cluster_heads: int,
        pheromones: Mapping[int, float],
        heuristics: Mapping[int, float],
        beta: float,
        chaos_state: float,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, ...], float]:
        selected: set[int] = set()
        state = chaos_state
        while len(selected) < number_of_cluster_heads:
            available = tuple(node for node in candidates if node not in selected)
            adjusted: dict[int, float] = {}
            for node in available:
                if self.config.chaos_enabled:
                    state = logistic_chaos_next(
                        state, self.config.chaos_r, self.config.objective_epsilon
                    )
                    factor = 1.0 + self.config.chaos_strength * (2.0 * state - 1.0)
                    adjusted[node] = max(
                        heuristics[node] * factor, self.config.objective_epsilon
                    )
                else:
                    adjusted[node] = heuristics[node]
            probabilities = stable_selection_probabilities(
                available,
                pheromones,
                adjusted,
                self.config.alpha,
                beta,
                self.config.objective_epsilon,
            )
            selected.add(int(rng.choice(available, p=probabilities)))
        return tuple(sorted(selected)), state

    def _initial_chaos_state(self, rng: np.random.Generator) -> float:
        state = self.config.chaos_initial_state
        epsilon = self.config.objective_epsilon
        if state is None:
            state = float(rng.uniform(epsilon, 1.0 - epsilon))
        forbidden = (0.0, 0.25, 0.5, 0.75, 1.0)
        if any(abs(float(state) - value) <= epsilon for value in forbidden):
            raise ValueError("chaos_initial_state must avoid 0, 0.25, 0.5, 0.75, and 1")
        return float(np.clip(state, epsilon, 1.0 - epsilon))

    def _validate(
        self,
        candidates: tuple[int, ...],
        number_of_cluster_heads: int,
        context: CHOptimizationContext,
    ) -> None:
        config = self.config
        if number_of_cluster_heads <= 0:
            raise ValueError("number_of_cluster_heads must be positive")
        if not candidates:
            raise ValueError("candidate_node_ids must not be empty")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate_node_ids must be unique")
        if number_of_cluster_heads > len(candidates):
            raise ValueError("number_of_cluster_heads exceeds candidate count")
        if context.rng is None or context.objective_evaluator is None:
            raise ValueError("rng and objective_evaluator are required")
        positions = np.asarray(context.node_positions, dtype=float)
        residual = np.asarray(context.residual_energy_j, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("node_positions must have shape (N, 3)")
        if residual.shape != (positions.shape[0],):
            raise ValueError("residual_energy_j must have shape (N,)")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(residual)):
            raise ValueError("network snapshot must be finite")
        alive = set(int(node) for node in context.alive_node_ids)
        for candidate in candidates:
            if candidate < 0 or candidate >= len(residual):
                raise ValueError(f"invalid candidate node ID: {candidate}")
            if candidate not in alive or residual[candidate] <= 0.0:
                raise ValueError(f"candidate node is not alive: {candidate}")
        finite_values = (
            config.alpha,
            config.beta_start,
            config.beta_end,
            config.rho_start,
            config.rho_end,
            config.initial_pheromone,
            config.tau_min,
            config.tau_max,
            config.deposit_q,
            config.elite_weight,
            config.chaos_r,
            config.chaos_strength,
            config.objective_epsilon,
        )
        if not all(np.isfinite(value) for value in finite_values):
            raise ValueError("AC-ACO configuration must be finite")
        if config.ants <= 0 or config.iterations <= 0:
            raise ValueError("AC-ACO ants and iterations must be positive")
        if config.alpha < 0.0 or config.beta_start < 0.0 or config.beta_end < 0.0:
            raise ValueError("alpha and beta must be non-negative")
        if not 0.0 < config.rho_start < 1.0 or not 0.0 < config.rho_end < 1.0:
            raise ValueError("rho values must be in (0, 1)")
        if not 0.0 < config.tau_min <= config.initial_pheromone <= config.tau_max:
            raise ValueError("pheromone bounds and initial value are invalid")
        if config.deposit_q <= 0.0 or config.elite_weight < 0.0:
            raise ValueError("deposit_q must be positive and elite_weight non-negative")
        if not 0.0 < config.chaos_r <= 4.0 or config.chaos_strength < 0.0:
            raise ValueError("chaos_r or chaos_strength is invalid")
        if config.objective_epsilon <= 0.0:
            raise ValueError("objective_epsilon must be positive")
        if config.deposit_mode not in {"iteration_best", "global_best"}:
            raise ValueError("deposit_mode must be iteration_best or global_best")
        if config.stagnation_mode not in {"reset", "smooth"}:
            raise ValueError("stagnation_mode must be reset or smooth")
        weights = (
            config.heuristic_energy_weight,
            config.heuristic_eulc_weight,
            config.heuristic_centrality_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("heuristic weights must be non-negative with positive sum")


def run_ac_aco_cluster_head_selection(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    candidates: Sequence[int],
    rng: np.random.Generator,
    convergence_callback: Callable[[int, float], None] | None = None,
    positions: np.ndarray | None = None,
    diagnostics_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[int], dict[int, list[int]]]:
    """Legacy simulator adapter around the state-pure AC-ACO optimizer core."""
    if positions is None:
        raise ValueError("positions are required for AC-ACO")
    alive_ids = tuple(
        int(node)
        for node in np.flatnonzero(energies > params.dead_energy_threshold_j)
    )
    target_count = _target_cluster_head_count(
        len(alive_ids),
        len(candidates),
        params.cluster_head_ratio,
    )

    def assignment_builder(
        selected: tuple[int, ...],
    ) -> Mapping[int, tuple[int, ...]]:
        result = assign_members_strongest_rssi(
            np.asarray(positions, dtype=float),
            energies,
            selected,
            params,
        )
        return result.assignments

    def route_builder(
        selected: tuple[int, ...],
        assignments: Mapping[int, tuple[int, ...]],
    ) -> Mapping[int, tuple[int | None, ...]]:
        if not assignments:
            return {}
        assignment = AssignmentResult(
            assignments=assignments,
            member_to_ch={
                int(node): int(ch)
                for ch, members in assignments.items()
                for node in members
            },
            mode="strongest_rssi",
            fallback_used=False,
            is_feasible=True,
        )
        complete = optimize_route_plan(
            selected, assignment, objective_context, objective_evaluator
        )
        return complete.solution.route_plan.route_by_ch

    objective_context = EvaluationContext(
        node_positions=np.asarray(positions, dtype=float).copy(),
        sink_position=np.asarray(params.get_sink_position(), dtype=float),
        residual_energy_j=np.asarray(energies, dtype=float).copy(),
        packet_bits=int(case.packet_size_bits),
        frequency_khz=float(params.acoustic_frequency_khz),
        bandwidth_hz=float(params.channel_bandwidth_hz),
        max_retries=int(params.max_retries),
        lambda_energy=float(params.objective_lambda_energy),
        lambda_delay=float(params.objective_lambda_delay),
        energy_reference_j=float(params.objective_energy_reference_j or 1.0),
        delay_reference_s=float(params.objective_delay_reference_s or 1.0),
        minimum_route_success_probability=float(
            params.objective_minimum_route_success_probability
        ),
        infeasible_penalty=float(params.objective_infeasible_penalty),
        normalization_epsilon=float(params.objective_normalization_epsilon),
        params=params,
        energy_epsilon_j=float(params.objective_energy_epsilon_j),
        delay_epsilon_s=float(params.objective_delay_epsilon_s),
        objective_epsilon=float(params.objective_dimensionless_epsilon),
    )
    objective_evaluator = EnergyDelayObjective()

    context = CHOptimizationContext(
        node_positions=np.asarray(positions, dtype=float).copy(),
        residual_energy_j=np.asarray(energies, dtype=float).copy(),
        alive_node_ids=alive_ids,
        objective_context=objective_context,
        objective_evaluator=objective_evaluator,
        assignment_builder=assignment_builder,
        route_builder=route_builder,
        rng=rng,
    )
    result = ACACOCHOptimizer(ACACOConfig.from_params(params)).optimize(
        candidates, target_count, context
    )
    if convergence_callback is not None:
        for iteration, objective in enumerate(result.convergence_history):
            convergence_callback(iteration, objective)
    if diagnostics_callback is not None:
        for event in result.diagnostics["convergence_diagnostics"]:
            diagnostics_callback(dict(event))
    if not result.objective_result.feasible:
        raise RuntimeError(
            "AC-ACO found no feasible CH solution: "
            + "; ".join(result.objective_result.infeasible_reasons)
        )
    final_assignments = assignment_builder(result.selected_ch_ids)
    return list(result.selected_ch_ids), {
        int(ch): list(members) for ch, members in final_assignments.items()
    }
