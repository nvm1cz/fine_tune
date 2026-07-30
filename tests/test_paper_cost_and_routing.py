import unittest
from dataclasses import replace

import numpy as np

from uwsn.cases import SimulationCase
from uwsn.optimizers.common import (
    _build_paper_cost_normalization,
    solution_cost,
    _paper_solution_cost,
)
from uwsn.routing.transmission import (
    build_route_to_sink,
    pick_next_hop,
    routing_selection_costs,
)
from uwsn.run_config import SIMULATION_PARAMS


class PaperCostTests(unittest.TestCase):
    def test_cost_matches_equation_8_candidate_level_particle_cost(self) -> None:
        params = replace(SIMULATION_PARAMS, pso_omega=0.60)
        distance_matrix = np.asarray(
            [
                [0.0, 3.0, 8.0],
                [3.0, 0.0, 5.0],
                [8.0, 5.0, 0.0],
            ]
        )
        energies = np.asarray([0.50, 0.40, 0.20])
        dist_to_sink = np.asarray([10.0, 6.0, 2.0])
        candidates = [1, 2]
        normalization = _build_paper_cost_normalization(
            distance_matrix,
            energies,
            dist_to_sink,
            candidates,
            dead_energy_threshold_j=0.0,
        )

        solution = np.asarray([1, 2])
        actual = _paper_solution_cost(
            params,
            energies,
            solution,
            normalization,
        )

        # Candidate 1's nearest other candidate is 2: 5^2 + 6^2 = 61.
        # Candidate 2's nearest other candidate is 1: 5^2 + 2^2 = 29.
        self.assertAlmostEqual(normalization.max_path_cost, 61.0)
        self.assertAlmostEqual(normalization.max_candidate_energy, 0.40)
        self.assertAlmostEqual(normalization.candidate_path_costs[1], 61.0)
        self.assertAlmostEqual(normalization.candidate_path_costs[2], 29.0)
        mean_di = ((61.0 / 61.0) + (29.0 / 61.0)) / 2.0
        # CH penalties are averaged equally: CH 1 -> 0, CH 2 -> 0.5.
        mean_ei = (0.0 + 0.5) / 2.0
        expected = 0.60 * mean_di + 0.40 * mean_ei
        self.assertAlmostEqual(actual, expected)

    def test_connectivity_penalty_can_be_disabled_for_ablation(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            pso_omega=0.60,
            transmission_range_m=4.0,
            connectivity_penalty_enabled=False,
        )
        distance_matrix = np.asarray(
            [
                [0.0, 5.0],
                [5.0, 0.0],
            ]
        )
        energies = np.asarray([0.50, 0.40])
        layers = np.asarray([2, 2])
        dist_to_sink = np.asarray([10.0, 8.0])
        candidates = [0, 1]
        normalization = _build_paper_cost_normalization(
            distance_matrix,
            energies,
            dist_to_sink,
            candidates,
            dead_energy_threshold_j=0.0,
        )
        solution = np.asarray([0, 1])

        base = _paper_solution_cost(params, energies, solution, normalization)
        actual = solution_cost(
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            solution,
            normalization,
        )

        self.assertAlmostEqual(actual, base)


class PaperRoutingTests(unittest.TestCase):
    def test_p_i_j_matches_algorithm_1_and_selects_the_minimum(self) -> None:
        case = SimulationCase("routing-test", initial_energy=1.0, packet_size_bits=4000, node_count=3)
        params = replace(
            SIMULATION_PARAMS,
            balance_factor=0.50,
            transmission_range_m=20.0,
            dead_energy_threshold_j=0.0,
        )
        distance_matrix = np.asarray(
            [
                [0.0, 6.0, 8.0],
                [6.0, 0.0, 4.0],
                [8.0, 4.0, 0.0],
            ]
        )
        energies = np.asarray([1.0, 0.50, 1.0])
        layers = np.asarray([3, 2, 1])
        dist_to_sink = np.asarray([10.0, 7.0, 3.0])
        candidates = np.asarray([1, 2])

        actual_costs = routing_selection_costs(
            case,
            params,
            distance_matrix,
            energies,
            dist_to_sink,
            current_ch=0,
            candidate_chs=candidates,
        )
        expected_costs = np.asarray(
            [
                0.50 * (1.0 / 0.50) + 0.50 * ((6.0**2 + 7.0**2) / 10.0**2),
                0.50 * (1.0 / 1.00) + 0.50 * ((8.0**2 + 3.0**2) / 10.0**2),
            ]
        )
        np.testing.assert_allclose(actual_costs, expected_costs)
        self.assertEqual(
            pick_next_hop(
                case,
                params,
                distance_matrix,
                energies,
                layers,
                dist_to_sink,
                current_ch=0,
                cluster_heads=[0, 1, 2],
            ),
            2,
        )

    def test_route_reapplies_p_i_j_and_never_uses_a_same_layer_ch(self) -> None:
        case = SimulationCase("route-chain", initial_energy=1.0, packet_size_bits=4000, node_count=4)
        params = replace(
            SIMULATION_PARAMS,
            balance_factor=0.50,
            transmission_range_m=10.0,
            dead_energy_threshold_j=0.0,
        )
        distance_matrix = np.asarray(
            [
                [0.0, 8.0, 15.0, 2.0],
                [8.0, 0.0, 8.0, 7.0],
                [15.0, 8.0, 0.0, 10.0],
                [2.0, 7.0, 10.0, 0.0],
            ]
        )
        energies = np.ones(4)
        layers = np.asarray([3, 2, 1, 3])
        dist_to_sink = np.asarray([25.0, 16.0, 8.0, 5.0])

        route = build_route_to_sink(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            source_ch=0,
            cluster_heads=[0, 1, 2, 3],
        )

        self.assertEqual(route, [1, 2, None])


if __name__ == "__main__":
    unittest.main()
