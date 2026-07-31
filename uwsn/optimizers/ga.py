from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..cases import SimulationCase
from ..run_config import TunableParams
from .common import (
    _build_paper_cost_normalization,
    _ensure_candidate_coverage,
    _finalize_solution,
    _normalize_solution,
    _target_cluster_head_count,
    evaluate_solution_result,
)
from ..objective import EnergyDelayObjective, ObjectiveResult, RoundObjectiveCache
from .solution_encoding import PriorityVectorEncoding


def run_ga_cluster_head_selection(
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
) -> Tuple[List[int], Dict[int, List[int]]]:
    candidates = _ensure_candidate_coverage(
        energies,
        distance_matrix,
        candidates,
        params.transmission_range_m,
        params.dead_energy_threshold_j,
    )
    candidate_count = len(candidates)
    if candidate_count == 0:
        return [], {}

    candidates_array = np.asarray(candidates, dtype=int)
    pop_size = max(2, int(params.pso_particles))
    dims = candidate_count
    active_node_count = int(np.count_nonzero(energies > params.dead_energy_threshold_j))
    target_ch_count = _target_cluster_head_count(
        active_node_count,
        candidate_count,
        params.cluster_head_ratio,
    )
    encoding = PriorityVectorEncoding(tuple(int(c) for c in candidates_array), target_ch_count)
    cost_normalization = _build_paper_cost_normalization(
        distance_matrix,
        energies,
        dist_to_sink,
        candidates,
        params.dead_energy_threshold_j,
    )
    score_cache: Dict[tuple[int, ...], ObjectiveResult] = {}
    objective_evaluator = EnergyDelayObjective(RoundObjectiveCache())

    def decode(chromosome: np.ndarray) -> List[int]:
        return encoding.decode(chromosome)

    def score_chromosome(chromosome: np.ndarray) -> float:
        solution = tuple(
            _normalize_solution(
                energies,
                decode(chromosome),
                candidates,
                params.dead_energy_threshold_j,
            )
        )
        cached = score_cache.get(solution)
        if cached is not None:
            return cached.objective_J
        result = evaluate_solution_result(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            solution,
            candidates,
            cost_normalization,
            positions,
            objective_evaluator,
        )
        score_cache[solution] = result
        return result.objective_J

    def record_diagnostics(iteration: int, chromosome: np.ndarray) -> None:
        if diagnostics_callback is None:
            return
        key = tuple(_normalize_solution(
            energies, decode(chromosome), candidates, params.dead_energy_threshold_j
        ))
        result = score_cache[key]
        diagnostics_callback({
            "iteration": iteration, "evaluations": len(score_cache),
            "best_J": result.objective_J,
            "best_energy_term": result.energy_term_e,
            "best_delay_term": result.delay_term_d,
            "feasible_count": sum(item.is_feasible for item in score_cache.values()),
            "invalid_count": sum(not item.is_feasible for item in score_cache.values()),
        })

    def tournament(scores: np.ndarray) -> np.ndarray:
        size = min(3, pop_size)
        picked = rng.choice(pop_size, size=size, replace=False)
        return population[int(picked[int(np.argmin(scores[picked]))])].copy()

    population = rng.uniform(0.0, 1.0, size=(pop_size, dims))
    scores = np.asarray([score_chromosome(chromosome) for chromosome in population], dtype=float)
    best_idx = int(np.argmin(scores))
    best = population[best_idx].copy()
    best_score = float(scores[best_idx])
    if convergence_callback is not None:
        convergence_callback(0, best_score)
    record_diagnostics(0, best)

    elite_count = min(2, pop_size)
    crossover_rate = float(params.ga_crossover_rate)
    mutation_rate = min(0.25, max(1.0 / max(dims, 1), 0.02))
    mutation_sigma = float(params.ga_mutation_sigma)

    for generation_idx in range(1, params.pso_iterations + 1):
        elite_indices = np.argsort(scores)[:elite_count]
        next_population = [population[int(idx)].copy() for idx in elite_indices]
        while len(next_population) < pop_size:
            parent_a = tournament(scores)
            parent_b = tournament(scores)
            if rng.random() < crossover_rate:
                mask = rng.random(dims) < 0.5
                child = np.where(mask, parent_a, parent_b)
            else:
                child = parent_a.copy()

            mutation_mask = rng.random(dims) < mutation_rate
            if np.any(mutation_mask):
                child[mutation_mask] += rng.normal(
                    0.0,
                    mutation_sigma,
                    size=int(np.count_nonzero(mutation_mask)),
                )
            child = np.clip(child, 0.0, 1.0)
            next_population.append(child)

        population = np.asarray(next_population, dtype=float)
        scores = np.asarray([score_chromosome(chromosome) for chromosome in population], dtype=float)
        generation_best_idx = int(np.argmin(scores))
        generation_best_score = float(scores[generation_best_idx])
        if generation_best_score < best_score:
            best_score = generation_best_score
            best = population[generation_best_idx].copy()
        if convergence_callback is not None:
            convergence_callback(generation_idx, best_score)
        record_diagnostics(generation_idx, best)

    return _finalize_solution(
        case,
        params,
        distance_matrix,
        energies,
        layers,
        dist_to_sink,
        decode(best),
        candidates,
        cost_normalization,
    )
