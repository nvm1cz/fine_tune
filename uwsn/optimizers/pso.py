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
from ..objective import ObjectiveResult
from .solution_encoding import PriorityVectorEncoding


PSO_POSITION_MIN = 0.0
PSO_POSITION_MAX = 1.0
PSO_VELOCITY_MAX = 1.0


def run_pso_cluster_head_selection(
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
    pop_size = params.pso_particles
    dims = candidate_count
    active_node_count = int(np.count_nonzero(energies > params.dead_energy_threshold_j))
    target_ch_count = _target_cluster_head_count(
        active_node_count,
        candidate_count,
        params.cluster_head_ratio,
    )
    encoding = PriorityVectorEncoding(tuple(int(c) for c in candidates_array), target_ch_count)

    positions_swarm = rng.uniform(PSO_POSITION_MIN, PSO_POSITION_MAX, size=(pop_size, dims))
    velocities = np.zeros((pop_size, dims), dtype=float)
    score_cache: Dict[tuple[int, ...], ObjectiveResult] = {}
    cost_normalization = _build_paper_cost_normalization(
        distance_matrix,
        energies,
        dist_to_sink,
        candidates,
        params.dead_energy_threshold_j,
    )

    def decode(particle: np.ndarray) -> List[int]:
        return encoding.decode(particle)

    def score_solution(solution: Sequence[int]) -> float:
        normalized = tuple(
            _normalize_solution(
                energies,
                solution,
                candidates,
                params.dead_energy_threshold_j,
            )
        )
        cached = score_cache.get(normalized)
        if cached is not None:
            return cached.objective_J
        result = evaluate_solution_result(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            normalized,
            candidates,
            cost_normalization,
            positions,
        )
        score_cache[normalized] = result
        return result.objective_J

    def record_diagnostics(iteration: int, best_position: np.ndarray) -> None:
        if diagnostics_callback is None:
            return
        key = tuple(_normalize_solution(
            energies, decode(best_position), candidates, params.dead_energy_threshold_j
        ))
        best = score_cache[key]
        diagnostics_callback({
            "iteration": iteration,
            "evaluations": len(score_cache),
            "best_J": best.objective_J,
            "best_energy_term": best.energy_term_e,
            "best_delay_term": best.delay_term_d,
            "feasible_count": sum(result.is_feasible for result in score_cache.values()),
            "invalid_count": sum(not result.is_feasible for result in score_cache.values()),
        })

    personal_best = positions_swarm.copy()
    personal_best_scores = np.full(pop_size, float("inf"))
    global_best = positions_swarm[0].copy()
    global_best_score = float("inf")
    stagnation_count = 0

    for i in range(pop_size):
        solution = decode(positions_swarm[i])
        score = score_solution(solution)
        personal_best_scores[i] = score
        if score < global_best_score:
            global_best = positions_swarm[i].copy()
            global_best_score = score

    if convergence_callback is not None:
        convergence_callback(0, global_best_score)
    record_diagnostics(0, global_best)

    stagnation_limit = int(params.pso_stagnation_restart_iterations)
    restart_count = int(np.ceil(pop_size * params.pso_stagnation_restart_fraction))
    restart_count = max(0, min(pop_size - 1, restart_count))

    for iteration_idx in range(1, params.pso_iterations + 1):
        improved_this_iteration = False
        for i in range(pop_size):
            r1 = rng.random(dims)
            r2 = rng.random(dims)
            velocities[i] = (
                params.pso_inertia * velocities[i]
                + params.pso_c1 * r1 * (personal_best[i] - positions_swarm[i])
                + params.pso_c2 * r2 * (global_best - positions_swarm[i])
            )
            velocities[i] = np.clip(velocities[i], -PSO_VELOCITY_MAX, PSO_VELOCITY_MAX)
            positions_swarm[i] = np.clip(
                positions_swarm[i] + velocities[i],
                PSO_POSITION_MIN,
                PSO_POSITION_MAX,
            )

            solution = decode(positions_swarm[i])
            score = score_solution(solution)
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = positions_swarm[i].copy()
            if score < global_best_score:
                global_best_score = score
                global_best = positions_swarm[i].copy()
                improved_this_iteration = True

        if improved_this_iteration:
            stagnation_count = 0
        elif stagnation_limit > 0 and restart_count > 0:
            stagnation_count += 1
            if stagnation_count >= stagnation_limit:
                worst_particles = np.argsort(personal_best_scores)[-restart_count:]
                for i in worst_particles:
                    positions_swarm[i] = rng.uniform(PSO_POSITION_MIN, PSO_POSITION_MAX, size=dims)
                    velocities[i] = 0.0
                    solution = decode(positions_swarm[i])
                    score = score_solution(solution)
                    personal_best[i] = positions_swarm[i].copy()
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best_score = score
                        global_best = positions_swarm[i].copy()
                stagnation_count = 0

        if convergence_callback is not None:
            convergence_callback(iteration_idx, global_best_score)
        record_diagnostics(iteration_idx, global_best)

    return _finalize_solution(
        case,
        params,
        distance_matrix,
        energies,
        layers,
        dist_to_sink,
        decode(global_best),
        candidates,
        cost_normalization,
    )
