import unittest
from dataclasses import replace
from unittest.mock import patch
import warnings

import numpy as np

from uwsn.algorithms import create_algorithm
from uwsn.cases import SimulationCase
from uwsn.routing.joint import optimize_route_plan
from uwsn.objective import EnergyDelayObjective, EvaluationContext
from uwsn.routing.planning import (
    AssignmentResult,
    RoutePlan,
    assign_members_strongest_rssi,
    validate_assignments,
    validate_route_plan,
)
from uwsn.routing.transmission import execute_round_transmissions
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import UWSNSimulator


class JointRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = replace(
            SIMULATION_PARAMS,
            transmission_range_m=80.0,
            sink_position=(0.0, 0.0, 0.0),
            dead_energy_threshold_j=0.0,
            broadcast_packet_size_bits=0,
            routing_plan_search_limit=64,
        )
        self.positions = np.asarray([
            [60.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 1.0, 0.0],
            [45.0, -1.0, 0.0],
        ])
        self.energies = np.full(4, 10.0)

    def context(self) -> EvaluationContext:
        return EvaluationContext(
            node_positions=self.positions.copy(),
            sink_position=np.zeros(3),
            residual_energy_j=self.energies.copy(),
            packet_bits=100,
            frequency_khz=self.params.acoustic_frequency_khz,
            bandwidth_hz=self.params.channel_bandwidth_hz,
            max_retries=self.params.max_retries,
            lambda_energy=0.5,
            lambda_delay=0.5,
            energy_reference_j=1.0,
            delay_reference_s=1.0,
            minimum_route_success_probability=0.0,
            infeasible_penalty=1e12,
            normalization_epsilon=1e-12,
            params=self.params,
        )

    def test_strongest_rssi_assignment_and_deterministic_tie_break(self) -> None:
        result = assign_members_strongest_rssi(
            self.positions, self.energies, (0, 1), self.params
        )
        self.assertTrue(result.is_feasible)
        self.assertEqual(result.mode, "strongest_rssi")
        self.assertEqual(result.member_to_ch[2], 1)
        self.assertEqual(result.member_to_ch[3], 0)
        self.assertEqual(result.member_to_ch[0], 0)
        self.assertEqual(result.member_to_ch[1], 1)

    def test_every_alive_node_is_assigned_once_to_alive_selected_ch(self) -> None:
        result = assign_members_strongest_rssi(
            self.positions, self.energies, (0, 1), self.params
        )
        flattened = [node for members in result.assignments.values() for node in members]
        self.assertEqual(sorted(flattened), list(range(4)))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(result.assignments), {0, 1})
        self.assertEqual(
            validate_assignments(
                result.assignments, (0, 1), self.positions, self.energies, self.params
            ),
            (True, None),
        )

    def test_unassigned_alive_node_is_infeasible_and_state_is_unchanged(self) -> None:
        positions = np.asarray([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]])
        energies = np.asarray([10.0, 10.0])
        before = energies.copy()
        result = assign_members_strongest_rssi(positions, energies, (0,), self.params)
        self.assertFalse(result.is_feasible)
        self.assertEqual(result.invalid_reason, "unassigned_alive_node")
        np.testing.assert_array_equal(energies, before)

    def test_nearest_fallback_is_explicit_and_warned(self) -> None:
        with patch(
            "uwsn.routing.planning.transmission_loss_db",
            side_effect=ValueError("unavailable"),
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = assign_members_strongest_rssi(
                self.positions, self.energies, (0, 1), self.params
            )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.mode, "nearest_feasible_fallback")
        self.assertTrue(any("falling back" in str(item.message) for item in caught))

    def test_route_failure_returns_infinite_objective(self) -> None:
        positions = np.asarray([[500.0, 0.0, 0.0]])
        energies = np.asarray([10.0])
        params = replace(self.params, transmission_range_m=10.0)
        assignment = AssignmentResult(
            assignments={0: (0,)},
            member_to_ch={0: 0},
            mode="strongest_rssi",
            fallback_used=False,
            is_feasible=True,
        )
        context = replace(
            self.context(),
            node_positions=positions,
            residual_energy_j=energies,
            params=params,
        )
        result = optimize_route_plan((0,), assignment, context)
        self.assertFalse(result.is_feasible)
        self.assertEqual(result.objective_result.objective_J, float("inf"))
        self.assertEqual(result.invalid_reason, "broken_route")

    def test_joint_router_returns_complete_deterministic_plan(self) -> None:
        assignment = assign_members_strongest_rssi(
            self.positions, self.energies, (0, 1), self.params
        )
        first = optimize_route_plan((0, 1), assignment, self.context())
        second = optimize_route_plan((0, 1), assignment, self.context())
        self.assertTrue(first.is_feasible)
        self.assertEqual(first.solution.route_plan, second.solution.route_plan)
        self.assertEqual(
            first.objective_result.objective_J,
            second.objective_result.objective_J,
        )
        self.assertEqual(set(first.solution.route_plan.route_by_ch), {0, 1})
        self.assertEqual(
            validate_route_plan(
                first.solution.route_plan,
                (0, 1),
                self.positions,
                np.zeros(3),
                self.energies,
                self.params,
            ),
            (True, None),
        )
        self.assertGreaterEqual(
            sum(first.solution.route_plan.forwarding_load_by_ch.values()), 0
        )

    def test_inner_router_evaluates_complete_plans_with_shared_objective(self) -> None:
        assignment = assign_members_strongest_rssi(
            self.positions, self.energies, (0, 1), self.params
        )
        with patch.object(
            EnergyDelayObjective,
            "evaluate",
            autospec=True,
            wraps=EnergyDelayObjective.evaluate,
        ) as evaluate:
            result = optimize_route_plan((0, 1), assignment, self.context())
        self.assertTrue(result.is_feasible)
        self.assertGreater(evaluate.call_count, 1)
        for call in evaluate.call_args_list:
            candidate = call.args[1]
            self.assertIsNotNone(candidate.route_plan)
            self.assertEqual(set(candidate.assignments), {0, 1})

    def test_route_search_counters_do_not_change_enumeration(self) -> None:
        assignment = assign_members_strongest_rssi(
            self.positions, self.energies, (0, 1), self.params
        )
        evaluator = EnergyDelayObjective()
        result = optimize_route_plan(
            (0, 1), assignment, self.context(), evaluator
        )
        counters = evaluator.cache.snapshot()
        self.assertTrue(result.is_feasible)
        self.assertEqual(counters["route_outer_candidates"], 1)
        self.assertGreater(counters["route_plans_evaluated"], 1)
        self.assertEqual(
            counters["route_plans_max"], counters["route_plans_evaluated"]
        )
        self.assertEqual(
            result.solution.route_plan.diagnostics["plans_evaluated"],
            counters["route_plans_evaluated"],
        )
        self.assertEqual(counters["outer_ch_candidates"], 1)
        self.assertEqual(counters["unique_ch_sets"], 1)
        self.assertGreaterEqual(
            counters["route_plans_generated"],
            counters["route_plans_evaluated"],
        )
        self.assertEqual(counters["fixed_assignment_computations"], 1)
        self.assertEqual(
            counters["route_dependent_computations"],
            counters["route_plans_evaluated"],
        )
        self.assertLess(
            counters["member_link_evaluations"],
            counters["route_plans_evaluated"] * 2,
        )

    def test_execution_uses_stored_route_without_greedy_rebuild(self) -> None:
        plan = RoutePlan.from_routes({0: (None,)})
        params = replace(self.params, transmission_range_m=100.0)
        energies = np.asarray([10.0])
        with patch(
            "uwsn.routing.build_route_to_sink",
            side_effect=AssertionError("legacy route rebuild"),
        ):
            received = execute_round_transmissions(
                SimulationCase("stored", 10.0, 100, node_count=1, rounds=1),
                params,
                np.zeros((1, 1)),
                np.zeros((1, 1)),
                np.zeros(1),
                energies,
                np.asarray([1]),
                np.asarray([10.0]),
                {0: [0]},
                [0],
                0,
                0.0,
                route_plan=plan,
            )
        self.assertEqual(received, 1)

    def test_broken_route_after_mobility_is_rejected(self) -> None:
        plan = RoutePlan.from_routes({0: (1, None), 1: (None,)})
        moved = self.positions.copy()
        moved[0] = [500.0, 0.0, 0.0]
        valid, reason = validate_route_plan(
            plan, (0, 1), moved, np.zeros(3), self.energies, self.params
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "mobility_topology_change")


class ReoptimizationTests(unittest.TestCase):
    def small_simulator(self, algorithm_id: str, *, rounds: int = 3) -> UWSNSimulator:
        params = replace(
            SIMULATION_PARAMS,
            width_m=20.0,
            height_m=20.0,
            depth_m=20.0,
            transmission_range_m=100.0,
            layer_r0_m=10.0,
            layer_spacing_m=5.0,
            cluster_head_ratio=0.25,
            pso_particles=4,
            pso_iterations=2,
            ac_aco_ants=4,
            ac_aco_iterations=2,
            recluster_interval=2,
            routing_plan_search_limit=16,
            source_level_db=180.0,
            channel_bandwidth_hz=1000.0,
            enable_packet_errors=True,
            enable_retransmission=True,
            current_model_enabled=True,
            current_mean_velocity_mps=(0.01, 0.0, 0.0),
            current_time_step_s=1.0,
            dead_energy_threshold_j=0.0,
        )
        return UWSNSimulator(
            SimulationCase("joint-smoke", 10.0, 100, node_count=12, rounds=rounds),
            params,
            seed=11,
            verbose=False,
            algorithm=create_algorithm(algorithm_id),
        )

    def test_periodic_reoptimization_reuses_plan_between_events(self) -> None:
        simulator = self.small_simulator("eulc_pso")
        simulator.run(stop_on_first_dead=False)
        self.assertEqual(
            [event["reoptimization_reason"] for event in simulator.optimization_events],
            ["initial", "periodic"],
        )
        self.assertTrue(simulator.round_configuration_history[1]["reused_route_plan"])
        for event in simulator.round_configuration_history:
            self.assertEqual(
                event["objective_route_plan_id"],
                event["execution_route_plan_id"],
            )

    def test_dead_ch_triggers_early_reoptimization_reason(self) -> None:
        simulator = self.small_simulator("eulc_pso", rounds=1)
        simulator._refresh_clusters(1)
        simulator.energies[simulator.current_chs[0]] = 0.0
        self.assertTrue(simulator._need_recluster(2))
        self.assertIn(
            simulator._pending_reoptimization_reason,
            {"dead_cluster_head", "dead_relay"},
        )

    def test_three_optimizers_joint_routing_smoke(self) -> None:
        for algorithm_id in ("eulc_pso", "eulc_ga", "eulc_ac_aco"):
            with self.subTest(algorithm=algorithm_id):
                simulator = self.small_simulator(algorithm_id)
                metrics = simulator.run(stop_on_first_dead=False)
                self.assertEqual(len(metrics.residual_energy_by_round), 3)
                self.assertIsNotNone(simulator.current_route_plan)
                self.assertTrue(simulator.current_objective_result.is_feasible)
                self.assertFalse(np.isnan(simulator.current_objective_result.objective_J))
                self.assertEqual(
                    simulator.algorithm_diagnostics["objective_route_plan_id"],
                    simulator.current_route_plan.plan_id,
                )


if __name__ == "__main__":
    unittest.main()
