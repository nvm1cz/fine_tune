import unittest
from dataclasses import replace

import numpy as np

from uwsn.cases import SimulationCase
from uwsn.algorithms.candidate_selection import select_eulc_candidates
from uwsn.objective import EvaluationContext, ObjectiveResult
from uwsn.optimizers.ac_aco import (
    ACACOCHOptimizer,
    ACACOConfig,
    compute_candidate_heuristics,
    deposit_pheromone,
    evaporate_pheromones,
    logistic_chaos_next,
    run_ac_aco_cluster_head_selection,
    stable_selection_probabilities,
)
from uwsn.optimizers.ga import run_ga_cluster_head_selection
from uwsn.optimizers.interfaces import CHOptimizationContext
from uwsn.optimizers.pso import run_pso_cluster_head_selection
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import UWSNSimulator


def objective_result(value: float, feasible: bool = True) -> ObjectiveResult:
    return ObjectiveResult(
        objective_value=value,
        raw_energy_j=value,
        raw_average_delay_s=value,
        normalized_energy=value,
        normalized_delay=value,
        feasible=feasible,
        infeasible_reasons=() if feasible else ("infeasible",),
        route_success_probabilities={},
        expected_attempts_by_link={},
        energy_by_link={},
        delay_by_link={},
        diagnostics={},
    )


class FakeEvaluator:
    def __init__(self, infeasible: bool = False) -> None:
        self.calls: list[tuple[int, ...]] = []
        self.infeasible = infeasible

    def evaluate(self, candidate, context):
        key = tuple(sorted(candidate.selected_cluster_heads))
        self.calls.append(key)
        return objective_result(
            float(sum(key)),
            feasible=not self.infeasible,
        )


class ACACOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = replace(
            SIMULATION_PARAMS,
            source_level_db=180.0,
            channel_bandwidth_hz=1000.0,
            dead_energy_threshold_j=0.0,
            ac_aco_ants=20,
            ac_aco_iterations=4,
            ac_aco_chaos_initial_state=0.123456,
            ac_aco_stagnation_limit=0,
        )
        self.positions = np.asarray(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
        )
        self.energies = np.full(3, 10.0)

    def context(
        self,
        evaluator=None,
        seed: int = 42,
        energies: np.ndarray | None = None,
    ) -> CHOptimizationContext:
        residual = self.energies.copy() if energies is None else energies
        evaluation_context = EvaluationContext(
            node_positions=self.positions.copy(),
            sink_position=np.asarray([100.0, 0.0, 0.0]),
            residual_energy_j=residual.copy(),
            packet_bits=100,
            frequency_khz=10.0,
            bandwidth_hz=1000.0,
            max_retries=3,
            lambda_energy=0.5,
            lambda_delay=0.5,
            energy_reference_j=1.0,
            delay_reference_s=1.0,
            minimum_route_success_probability=0.0,
            infeasible_penalty=1.0e12,
            normalization_epsilon=1.0e-12,
            params=self.params,
        )
        return CHOptimizationContext(
            node_positions=self.positions.copy(),
            residual_energy_j=residual,
            alive_node_ids=tuple(np.flatnonzero(residual > 0.0)),
            objective_context=evaluation_context,
            objective_evaluator=evaluator or FakeEvaluator(),
            assignment_builder=lambda selected: {
                int(selected[0]): tuple(range(len(residual)))
            },
            route_builder=lambda selected, assignments: {
                int(ch): (None,) for ch in selected
            },
            rng=np.random.default_rng(seed),
        )

    def optimizer(self, **changes) -> ACACOCHOptimizer:
        params = replace(self.params, **changes)
        return ACACOCHOptimizer(ACACOConfig.from_params(params))

    def test_selects_exactly_k_unique_global_candidate_ids(self) -> None:
        result = self.optimizer().optimize([0, 2], 2, self.context())
        self.assertEqual(result.selected_ch_ids, (0, 2))
        self.assertEqual(len(set(result.selected_ch_ids)), 2)
        self.assertTrue(set(result.selected_ch_ids).issubset({0, 2}))

    def test_rejects_too_many_chs_duplicate_candidates_and_dead_candidate(self) -> None:
        with self.assertRaises(ValueError):
            self.optimizer().optimize([0, 1], 3, self.context())
        with self.assertRaises(ValueError):
            self.optimizer().optimize([0, 0], 1, self.context())
        dead = np.asarray([10.0, 0.0, 10.0])
        with self.assertRaises(ValueError):
            self.optimizer().optimize([0, 1], 1, self.context(energies=dead))

    def test_same_seed_is_reproducible_and_lower_objective_wins(self) -> None:
        first = self.optimizer().optimize([2, 1], 1, self.context(seed=7))
        second = self.optimizer().optimize([2, 1], 1, self.context(seed=7))
        self.assertEqual(first.selected_ch_ids, (1,))
        self.assertEqual(first.selected_ch_ids, second.selected_ch_ids)
        self.assertEqual(first.convergence_history, second.convergence_history)
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(
                    first.convergence_history, first.convergence_history[1:]
                )
            )
        )

    def test_cache_uses_canonical_set_and_evaluator_is_the_only_fitness(self) -> None:
        evaluator = FakeEvaluator()
        result = self.optimizer().optimize(
            [0, 1, 2], 2, self.context(evaluator=evaluator)
        )
        self.assertLessEqual(len(evaluator.calls), 3)
        self.assertEqual(
            result.diagnostics["objective_evaluations"], len(evaluator.calls)
        )
        self.assertGreater(result.diagnostics["objective_cache_hits"], 0)
        self.assertTrue(all(key == tuple(sorted(key)) for key in evaluator.calls))

    def test_pheromone_evaporation_deposit_and_bounds(self) -> None:
        evaporated = evaporate_pheromones({2: 1.0, 9: 1.0}, 0.25, 0.1, 10.0)
        self.assertEqual(evaporated, {2: 0.75, 9: 0.75})
        deposited = deposit_pheromone(
            evaporated, (9,), 2.0, 1.0, 1.0, 1e-12, 0.1, 1.0
        )
        self.assertEqual(deposited[2], 0.75)
        self.assertEqual(deposited[9], 1.0)
        self.assertTrue(all(0.1 <= value <= 1.0 for value in deposited.values()))

    def test_infeasible_solution_gets_no_direct_deposit(self) -> None:
        result = self.optimizer(
            ac_aco_iterations=1,
            ac_aco_ants=1,
            ac_aco_adaptive_rho=False,
            ac_aco_rho_start=0.5,
        ).optimize([0, 2], 1, self.context(evaluator=FakeEvaluator(infeasible=True)))
        pheromones = result.diagnostics["final_pheromone_by_node_id"]
        self.assertEqual(pheromones, {0: 0.5, 2: 0.5})

    def test_stable_probability_fallback_and_small_heuristics(self) -> None:
        probabilities = stable_selection_probabilities(
            [2, 9],
            {2: 0.0, 9: 0.0},
            {2: 0.0, 9: 0.0},
            alpha=1.0,
            beta=1.0,
            epsilon=1e-12,
        )
        np.testing.assert_allclose(probabilities, [0.5, 0.5])
        self.assertTrue(np.all(np.isfinite(probabilities)))

    def test_chaos_is_candidate_specific_reproducible_and_can_be_disabled(self) -> None:
        first = logistic_chaos_next(0.123456, 4.0, 1e-12)
        second = logistic_chaos_next(first, 4.0, 1e-12)
        self.assertNotEqual(first, second)
        disabled = self.optimizer(
            ac_aco_chaos_enabled=False,
            ac_aco_ants=10,
        ).optimize([0, 1, 2], 1, self.context(seed=11))
        repeated = self.optimizer(
            ac_aco_chaos_enabled=False,
            ac_aco_ants=10,
        ).optimize([0, 1, 2], 1, self.context(seed=11))
        self.assertEqual(disabled, repeated)

    def test_heuristic_is_positive_and_uses_global_ids(self) -> None:
        heuristics = compute_candidate_heuristics(
            [2, 0],
            self.positions,
            self.energies,
            [0, 1, 2],
            energy_weight=0.5,
            eulc_weight=0.0,
            centrality_weight=0.5,
        )
        self.assertEqual(set(heuristics), {0, 2})
        self.assertTrue(all(value > 0.0 for value in heuristics.values()))

    def test_optimizer_does_not_mutate_snapshot(self) -> None:
        context = self.context()
        positions_before = context.node_positions.copy()
        energy_before = context.residual_energy_j.copy()
        self.optimizer().optimize([0, 1, 2], 1, context)
        np.testing.assert_array_equal(context.node_positions, positions_before)
        np.testing.assert_array_equal(context.residual_energy_j, energy_before)


class EULCACOIntegrationTests(unittest.TestCase):
    def test_eulc_ac_aco_adapter_and_simulator_registry(self) -> None:
        positions = np.asarray(
            [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]]
        )
        distance_matrix = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        sink = np.asarray([50.0, 50.0, 0.0])
        dist_to_sink = np.linalg.norm(positions - sink, axis=1)
        energies = np.full(3, 10.0)
        layers = np.ones(3, dtype=int)
        neighbor_degree = np.ones(3)
        case = SimulationCase("ac-aco", 10.0, 100, node_count=3, rounds=1)
        params = replace(
            SIMULATION_PARAMS,
            optimizer="ac_aco",
            objective_mode="energy_delay",
            objective_energy_reference_j=1.0,
            objective_delay_reference_s=1.0,
            source_level_db=180.0,
            channel_bandwidth_hz=1000.0,
            dead_energy_threshold_j=0.0,
            transmission_range_m=1000.0,
            cluster_head_ratio=0.34,
            ac_aco_ants=6,
            ac_aco_iterations=3,
            ac_aco_chaos_initial_state=0.123456,
            ac_aco_stagnation_limit=0,
            sink_position=tuple(sink),
        )
        candidates = select_eulc_candidates(
            case,
            params,
            energies,
            layers,
            dist_to_sink,
            neighbor_degree,
            distance_matrix,
        )
        positions_before = positions.copy()
        energies_before = energies.copy()
        first = run_ac_aco_cluster_head_selection(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            candidates,
            np.random.default_rng(9),
            positions=positions,
        )
        second = run_ac_aco_cluster_head_selection(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            candidates,
            np.random.default_rng(9),
            positions=positions,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 1)
        self.assertTrue(set(first[0]).issubset(set(candidates)))
        np.testing.assert_array_equal(positions, positions_before)
        np.testing.assert_array_equal(energies, energies_before)

        simulator = UWSNSimulator(
            case,
            params,
            seed=9,
            verbose=False,
            initial_positions=positions,
        )
        metrics = simulator.run(stop_on_first_dead=False, max_rounds=1)
        self.assertLess(metrics.residual_energy, 30.0)

    def test_pso_ga_and_ac_aco_share_same_snapshot_and_candidates(self) -> None:
        positions = np.asarray(
            [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]]
        )
        distance_matrix = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        sink = np.asarray([50.0, 50.0, 0.0])
        dist_to_sink = np.linalg.norm(positions - sink, axis=1)
        energies = np.full(3, 10.0)
        case = SimulationCase("comparison", 10.0, 100, node_count=3)
        params = replace(
            SIMULATION_PARAMS,
            objective_mode="energy_delay",
            objective_energy_reference_j=1.0,
            objective_delay_reference_s=1.0,
            source_level_db=180.0,
            channel_bandwidth_hz=1000.0,
            dead_energy_threshold_j=0.0,
            transmission_range_m=1000.0,
            cluster_head_ratio=0.34,
            pso_particles=4,
            pso_iterations=2,
            ac_aco_ants=4,
            ac_aco_iterations=2,
            ac_aco_chaos_initial_state=0.123456,
            ac_aco_stagnation_limit=0,
            sink_position=tuple(sink),
        )
        layers = np.ones(3, dtype=int)
        candidates = [0, 1, 2]
        before = energies.copy()
        for optimizer in (
            run_pso_cluster_head_selection,
            run_ga_cluster_head_selection,
            run_ac_aco_cluster_head_selection,
        ):
            selected, assignments = optimizer(
                case,
                params,
                distance_matrix,
                energies,
                layers,
                dist_to_sink,
                candidates,
                np.random.default_rng(21),
                positions=positions,
            )
            self.assertEqual(len(selected), 1)
            self.assertTrue(set(selected).issubset(set(candidates)))
            self.assertTrue(assignments)
            np.testing.assert_array_equal(energies, before)


if __name__ == "__main__":
    unittest.main()
