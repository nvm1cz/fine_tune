from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from uwsn.cases import SimulationCase
from uwsn.diagnostic import capture_round_energy
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import UWSNSimulator
from uwsn.experiment_config.generation import ExperimentGenerator


class DiagnosticTests(unittest.TestCase):
    def test_capture_is_observational_and_accounts_for_total_debit(self) -> None:
        case = SimulationCase("diagnostic", 0.5, 6400, node_count=8, rounds=1)
        first = UWSNSimulator(case, SIMULATION_PARAMS, seed=9, verbose=False)
        second = UWSNSimulator(case, SIMULATION_PARAMS, seed=9, verbose=False)
        with capture_round_energy(first) as rows:
            first_metrics = first.run(stop_on_first_dead=False, max_rounds=1)
        second_metrics = second.run(stop_on_first_dead=False, max_rounds=1)

        self.assertTrue(np.array_equal(first.energies, second.energies))
        for stream in ("topology_rng", "protocol_rng", "optimizer_rng", "channel_rng", "mobility_rng"):
            self.assertEqual(
                getattr(first, stream).bit_generator.state,
                getattr(second, stream).bit_generator.state,
            )
        self.assertEqual(first_metrics, second_metrics)
        self.assertEqual(first.current_chs, second.current_chs)
        self.assertEqual(first.current_assignments, second.current_assignments)
        self.assertEqual(first.current_route_plan, second.current_route_plan)
        self.assertEqual(first.optimization_events, second.optimization_events)
        self.assertEqual(first.transmission_stats, second.transmission_stats)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        component_total = (row.member_tx_energy + row.ch_rx_energy + row.aggregation_energy
            + row.routing_tx_energy + row.routing_rx_energy
            + row.retransmission_energy + row.control_energy)
        self.assertAlmostEqual(component_total, row.total_round_energy, places=12)
        self.assertGreaterEqual(row.total_round_energy, 0.0)

    def test_diagnostic_spec_has_four_paper_cases_and_three_fixed_seeds(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = ExperimentGenerator(
            root / "configs/experiment_sets/diagnostic_early_fnd.yaml"
        ).build_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual({case["metadata"]["base_seed"] for case in cases}, {0, 1, 2})
        self.assertEqual({case["environment"]["distribution"] for case in cases}, {"gaussian"})
        self.assertEqual({case["execution"]["rounds"] for case in cases}, {30})
        combinations = {
            (case["environment"]["initial_energy_j"], case["environment"]["packet_size_bits"])
            for case in cases
        }
        self.assertEqual(combinations, {(0.5, 6400), (0.5, 4000), (1.0, 6400), (1.0, 4000)})


if __name__ == "__main__":
    unittest.main()
