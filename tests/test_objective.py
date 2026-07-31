import unittest
from contextlib import contextmanager
from dataclasses import replace
import os
from unittest.mock import patch

import numpy as np

from uwsn.objective import (
    CandidateSolution,
    EnergyDelayObjectiveEvaluator,
    EvaluationContext,
    RoundObjectiveCache,
    expected_attempts_limited,
)
from uwsn.cases import SimulationCase
from uwsn.optimizers.ga import run_ga_cluster_head_selection
from uwsn.optimizers.pso import run_pso_cluster_head_selection
from uwsn.run_config import SIMULATION_PARAMS


class EnergyDelayObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = EnergyDelayObjectiveEvaluator()
        self.params = replace(
            SIMULATION_PARAMS,
            source_level_db=130.0,
            channel_bandwidth_hz=1000.0,
            enable_packet_errors=True,
            dead_energy_threshold_j=0.0,
        )

    def context(
        self,
        *,
        positions: np.ndarray | None = None,
        residual: np.ndarray | None = None,
        params=None,
        max_retries: int = 3,
        lambda_energy: float = 0.5,
        lambda_delay: float = 0.5,
        energy_reference: float = 1.0,
        delay_reference: float = 1.0,
        minimum_reliability: float = 0.0,
    ) -> EvaluationContext:
        node_positions = (
            np.asarray([[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]])
            if positions is None
            else np.asarray(positions, dtype=float)
        )
        residual_energy = (
            np.full(len(node_positions), 100.0)
            if residual is None
            else np.asarray(residual, dtype=float)
        )
        actual_params = self.params if params is None else params
        return EvaluationContext(
            node_positions=node_positions,
            sink_position=np.asarray([100.0, 0.0, 0.0]),
            residual_energy_j=residual_energy,
            packet_bits=100,
            frequency_khz=10.0,
            bandwidth_hz=1000.0,
            max_retries=max_retries,
            lambda_energy=lambda_energy,
            lambda_delay=lambda_delay,
            energy_reference_j=energy_reference,
            delay_reference_s=delay_reference,
            minimum_route_success_probability=minimum_reliability,
            infeasible_penalty=1.0e12,
            normalization_epsilon=1.0e-12,
            params=actual_params,
        )

    @staticmethod
    def direct_candidate() -> CandidateSolution:
        return CandidateSolution(
            selected_cluster_heads=(0,),
            assignments={0: (0, 1)},
            routes={0: (None,)},
        )

    @staticmethod
    @contextmanager
    def environment(name: str, value: str | None):
        previous = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous

    def test_round_link_cache_matches_independent_reference_path(self) -> None:
        context = self.context()
        with self.environment("UWSN_DISABLE_LINK_CACHE", "1"):
            reference = EnergyDelayObjectiveEvaluator().evaluate(
                self.direct_candidate(), context
            )
        cache = RoundObjectiveCache()
        optimized_evaluator = EnergyDelayObjectiveEvaluator(cache)
        first = optimized_evaluator.evaluate(self.direct_candidate(), context)
        second = optimized_evaluator.evaluate(self.direct_candidate(), context)
        self.assertEqual(reference.objective_J, first.objective_J)
        self.assertEqual(reference.raw_energy_j, first.raw_energy_j)
        self.assertEqual(reference.raw_average_delay_s, first.raw_average_delay_s)
        self.assertEqual(first.objective_J, second.objective_J)
        self.assertEqual(cache.stats.logical_objective_evaluations, 2)
        self.assertEqual(cache.stats.physical_objective_computations, 2)
        self.assertGreater(cache.stats.link_cache_hits, 0)
        self.assertGreater(cache.stats.link_cache_misses, 0)
        self.assertEqual(cache.stats.neighbor_cache_builds, 1)
        self.assertGreater(cache.stats.neighbor_cache_hits, 0)
        self.assertEqual(cache.stats.contention_cache_misses, 1)
        self.assertGreater(cache.stats.contention_cache_hits, 0)

    def test_shadow_mode_compares_every_cached_lookup_to_reference(self) -> None:
        with self.environment("UWSN_VERIFY_LINK_CACHE", "1"):
            cache = RoundObjectiveCache()
            result = EnergyDelayObjectiveEvaluator(cache).evaluate(
                self.direct_candidate(), self.context()
            )
        self.assertTrue(result.is_feasible)
        self.assertGreater(cache.stats.shadow_checks, 0)

    def test_disable_cache_uses_independent_scalar_reference_path(self) -> None:
        with self.environment("UWSN_DISABLE_LINK_CACHE", "1"), patch(
            "uwsn.objective.channel_quality_with_static_environment"
        ) as optimized_channel:
            result = EnergyDelayObjectiveEvaluator().evaluate(
                self.direct_candidate(), self.context()
            )
        self.assertTrue(result.is_feasible)
        optimized_channel.assert_not_called()

    def test_cache_invalidates_when_topology_or_params_change(self) -> None:
        cache = RoundObjectiveCache()
        evaluator = EnergyDelayObjectiveEvaluator(cache)
        near = evaluator.evaluate(
            self.direct_candidate(),
            self.context(positions=np.asarray([[90.0, 0.0, 0.0]])),
        )
        far = evaluator.evaluate(
            self.direct_candidate(),
            self.context(positions=np.asarray([[0.0, 0.0, 0.0]])),
        )
        high_power = evaluator.evaluate(
            self.direct_candidate(),
            self.context(params=replace(self.params, transmit_power_p0=10.0)),
        )
        self.assertGreater(far.raw_average_delay_s, near.raw_average_delay_s)
        self.assertGreater(high_power.raw_energy_j, far.raw_energy_j)
        self.assertEqual(cache.stats.neighbor_cache_builds, 3)

    def test_cache_invalidates_when_alive_mask_changes(self) -> None:
        cache = RoundObjectiveCache()
        evaluator = EnergyDelayObjectiveEvaluator(cache)
        evaluator.evaluate(self.direct_candidate(), self.context())
        changed = self.context(residual=np.asarray([100.0, 0.0]))
        evaluator.evaluate(
            CandidateSolution((0,), {0: (0,)}, {0: (None,)}),
            changed,
        )
        self.assertEqual(cache.stats.neighbor_cache_builds, 2)

    def test_objective_cache_does_not_consume_rng_state(self) -> None:
        rng = np.random.default_rng(2026)
        before = rng.bit_generator.state
        evaluator = EnergyDelayObjectiveEvaluator(RoundObjectiveCache())
        evaluator.evaluate(self.direct_candidate(), self.context())
        after = rng.bit_generator.state
        self.assertEqual(before, after)

    def test_fixed_member_link_decomposition_is_exact(self) -> None:
        candidate = self.direct_candidate()
        context = self.context()
        reference = EnergyDelayObjectiveEvaluator().evaluate(candidate, context)
        evaluator = EnergyDelayObjectiveEvaluator()
        fixed = evaluator.prepare_fixed_member_links(
            candidate.selected_cluster_heads,
            candidate.assignments,
            context,
        )
        decomposed = evaluator.evaluate(
            candidate,
            context,
            fixed_member_links=fixed,
        )
        self.assertEqual(reference, decomposed)

    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(
                self.direct_candidate(),
                self.context(lambda_energy=0.7, lambda_delay=0.7),
            )

    def test_raw_energy_increases_with_transmit_energy(self) -> None:
        low = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(params=replace(self.params, transmit_power_p0=1.0)),
        )
        high = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(params=replace(self.params, transmit_power_p0=10.0)),
        )
        self.assertGreater(high.raw_energy_j, low.raw_energy_j)

    def test_raw_delay_increases_with_distance(self) -> None:
        near = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(positions=np.asarray([[90.0, 0.0, 0.0]])),
        )
        far = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(positions=np.asarray([[0.0, 0.0, 0.0]])),
        )
        self.assertGreater(far.raw_average_delay_s, near.raw_average_delay_s)

    def test_noise_increases_attempts_energy_and_delay(self) -> None:
        quiet_params = replace(self.params, shipping_noise_factor=0.0, wind_speed_mps=0.0)
        noisy_params = replace(self.params, shipping_noise_factor=1.0, wind_speed_mps=10.0)
        quiet = self.evaluator.evaluate(
            self.direct_candidate(), self.context(params=quiet_params)
        )
        noisy = self.evaluator.evaluate(
            self.direct_candidate(), self.context(params=noisy_params)
        )
        link = (0, None)
        quiet_diag = quiet.diagnostics["link_expected_costs"][link]
        noisy_diag = noisy.diagnostics["link_expected_costs"][link]
        self.assertLess(noisy_diag.snr_db, quiet_diag.snr_db)
        self.assertLess(
            noisy_diag.packet_success_probability,
            quiet_diag.packet_success_probability,
        )
        self.assertGreaterEqual(
            noisy.expected_attempts_by_link[link],
            quiet.expected_attempts_by_link[link],
        )
        self.assertGreaterEqual(noisy.raw_energy_j, quiet.raw_energy_j)
        self.assertGreaterEqual(noisy.raw_average_delay_s, quiet.raw_average_delay_s)

    def test_expected_attempts_uses_retry_limit(self) -> None:
        attempts_one, limited_one = expected_attempts_limited(0.25, 0)
        attempts_four, limited_four = expected_attempts_limited(0.25, 3)
        self.assertEqual(attempts_one, 1.0)
        self.assertGreater(attempts_four, attempts_one)
        self.assertLessEqual(attempts_four, 4.0)
        self.assertGreater(limited_four, limited_one)

    def test_expected_attempts_boundary_probabilities(self) -> None:
        self.assertEqual(expected_attempts_limited(1.0, 3), (1.0, 1.0))
        attempts, limited = expected_attempts_limited(0.0, 3)
        self.assertEqual(attempts, 4.0)
        self.assertEqual(limited, 0.0)

    def test_aggregation_energy_is_success_weighted_not_attempt_weighted(self) -> None:
        candidate = CandidateSolution(
            selected_cluster_heads=(0,),
            assignments={0: (0, 1)},
            routes={0: (None,)},
        )
        result = self.evaluator.evaluate(candidate, self.context())
        member_link = result.diagnostics["link_expected_costs"][(1, 0)]
        base_aggregation = (
            self.params.energy_calibration_factor
            * 100
            * self.params.electronic_energy_e1_nj
            * 1e-9
        )
        self.assertAlmostEqual(
            member_link.aggregation_energy_j,
            member_link.limited_success_probability * base_aggregation,
        )
        if member_link.expected_attempts > 1.0 + 1.0e-6:
            self.assertLess(
                member_link.aggregation_energy_j,
                member_link.expected_attempts * base_aggregation,
            )

    def test_evaluation_is_pure_and_deterministic(self) -> None:
        candidate = CandidateSolution(
            selected_cluster_heads=(0,),
            assignments={0: (0, 1)},
            routes={0: (None,)},
        )
        context = self.context()
        positions_before = context.node_positions.copy()
        residual_before = context.residual_energy_j.copy()
        assignments_before = dict(candidate.assignments)
        routes_before = dict(candidate.routes)
        first = self.evaluator.evaluate(candidate, context)
        second = self.evaluator.evaluate(candidate, context)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(context.node_positions, positions_before)
        np.testing.assert_array_equal(context.residual_energy_j, residual_before)
        self.assertEqual(candidate.assignments, assignments_before)
        self.assertEqual(candidate.routes, routes_before)

    def test_missing_route_is_infeasible(self) -> None:
        candidate = CandidateSolution((0,), {0: (0,)}, {})
        result = self.evaluator.evaluate(candidate, self.context())
        self.assertFalse(result.feasible)
        self.assertEqual(result.objective_value, float("inf"))
        self.assertIsNotNone(result.invalid_reason)
        self.assertTrue(any("no route" in reason for reason in result.infeasible_reasons))

    def test_route_cycle_is_infeasible(self) -> None:
        candidate = CandidateSolution(
            selected_cluster_heads=(0,),
            assignments={0: (0,)},
            routes={0: (1, 0, None)},
        )
        result = self.evaluator.evaluate(candidate, self.context())
        self.assertFalse(result.feasible)
        self.assertTrue(any("cycle" in reason for reason in result.infeasible_reasons))

    def test_energy_above_residual_is_infeasible(self) -> None:
        expensive = replace(self.params, energy_calibration_factor=1.0e9)
        result = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(params=expensive, residual=np.asarray([1.0, 1.0])),
        )
        self.assertFalse(result.feasible)
        self.assertTrue(
            any("exceeds residual" in reason for reason in result.infeasible_reasons)
        )

    def test_route_probability_is_product_of_limited_link_probabilities(self) -> None:
        candidate = CandidateSolution(
            selected_cluster_heads=(0,),
            assignments={0: (0,)},
            routes={0: (1, None)},
        )
        result = self.evaluator.evaluate(candidate, self.context())
        links = result.diagnostics["link_expected_costs"]
        expected = (
            links[(0, 1)].limited_success_probability
            * links[(1, None)].limited_success_probability
        )
        self.assertAlmostEqual(result.route_success_probabilities[0], expected)

    def test_project_normalization_and_single_weight_modes(self) -> None:
        energy_only = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(
                lambda_energy=1.0,
                lambda_delay=0.0,
                energy_reference=2.0,
                delay_reference=4.0,
            ),
        )
        delay_only = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(
                lambda_energy=0.0,
                lambda_delay=1.0,
                energy_reference=2.0,
                delay_reference=4.0,
            ),
        )
        self.assertAlmostEqual(
            energy_only.normalized_energy,
            energy_only.raw_energy_j / (energy_only.available_energy_j + 1.0e-12),
        )
        self.assertAlmostEqual(
            delay_only.normalized_delay,
            delay_only.raw_average_delay_s / delay_only.reference_delay_s,
        )
        self.assertAlmostEqual(
            energy_only.objective_value, energy_only.normalized_energy
        )
        self.assertAlmostEqual(delay_only.objective_value, delay_only.normalized_delay)

    def test_skipped_packet_is_invalid_and_cannot_improve_objective(self) -> None:
        complete = CandidateSolution((0,), {0: (0, 1)}, {0: (None,)})
        skipped = CandidateSolution((0,), {0: (0,)}, {0: (None,)})
        complete_result = self.evaluator.evaluate(complete, self.context())
        skipped_result = self.evaluator.evaluate(skipped, self.context())
        self.assertTrue(complete_result.is_feasible)
        self.assertFalse(skipped_result.is_feasible)
        self.assertEqual(skipped_result.objective_J, float("inf"))
        self.assertGreater(skipped_result.objective_J, complete_result.objective_J)

    def test_energy_and_delay_terms_are_monotone_and_never_nan(self) -> None:
        low_energy = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(params=replace(self.params, transmit_power_p0=1.0)),
        )
        high_energy = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(params=replace(self.params, transmit_power_p0=10.0)),
        )
        self.assertGreaterEqual(high_energy.energy_term_e, low_energy.energy_term_e)
        self.assertGreaterEqual(high_energy.objective_J, low_energy.objective_J)

        near = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(positions=np.asarray([[90.0, 0.0, 0.0]])),
        )
        far = self.evaluator.evaluate(
            self.direct_candidate(),
            self.context(positions=np.asarray([[0.0, 0.0, 0.0]])),
        )
        self.assertGreaterEqual(far.delay_term_d, near.delay_term_d)
        self.assertGreaterEqual(far.objective_J, near.objective_J)
        self.assertFalse(np.isnan(far.objective_J))
        self.assertEqual(far.objective_direction, "minimize")

    def test_evaluating_another_candidate_does_not_change_previous_fitness(self) -> None:
        context = self.context()
        first = self.evaluator.evaluate(self.direct_candidate(), context)
        other = CandidateSolution((1,), {1: (1,)}, {1: (None,)})
        self.evaluator.evaluate(other, context)
        repeated = self.evaluator.evaluate(self.direct_candidate(), context)
        self.assertEqual(first, repeated)

    def test_pso_and_ga_share_energy_delay_objective_path(self) -> None:
        positions = np.asarray(
            [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]]
        )
        deltas = positions[:, None, :] - positions[None, :, :]
        distance_matrix = np.sqrt(np.sum(deltas * deltas, axis=2))
        sink = np.asarray([50.0, 50.0, 0.0])
        dist_to_sink = np.sqrt(np.sum((positions - sink) ** 2, axis=1))
        case = SimulationCase("objective-integration", 10.0, 100, node_count=3)
        params = replace(
            self.params,
            objective_mode="energy_delay",
            objective_energy_reference_j=1.0,
            objective_delay_reference_s=1.0,
            sink_position=tuple(sink),
            transmission_range_m=1000.0,
            cluster_head_ratio=0.34,
            pso_particles=4,
            pso_iterations=2,
        )
        energies = np.full(3, 10.0)
        layers = np.ones(3, dtype=int)
        candidates = [0, 1, 2]
        for optimizer in (
            run_pso_cluster_head_selection,
            run_ga_cluster_head_selection,
        ):
            selected, assignments = optimizer(
                case,
                params,
                distance_matrix,
                energies.copy(),
                layers,
                dist_to_sink,
                candidates,
                np.random.default_rng(42),
                positions=positions,
            )
            self.assertTrue(selected)
            self.assertTrue(assignments)


if __name__ == "__main__":
    unittest.main()
