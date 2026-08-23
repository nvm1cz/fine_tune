import unittest
from dataclasses import replace

import numpy as np

from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import UWSNSimulator, clusters_below_mean_energy


class ReclusterPolicyTests(unittest.TestCase):
    def test_cluster_triggers_when_ch_is_below_alive_cluster_mean(self) -> None:
        triggered = clusters_below_mean_energy(
            {0: [1, 2]}, [0], np.asarray([0.2, 0.8, 0.6]), 0.0
        )
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]["cluster_head"], 0)
        self.assertAlmostEqual(triggered[0]["cluster_mean_energy_j"], 0.5333333333)

    def test_dead_members_are_excluded_from_cluster_mean(self) -> None:
        triggered = clusters_below_mean_energy(
            {0: [1, 2]}, [0], np.asarray([0.5, 0.0, 0.4]), 0.1
        )
        self.assertEqual(triggered, [])

    def test_energy_mode_reclusters_without_periodic_trigger(self) -> None:
        simulator = UWSNSimulator.__new__(UWSNSimulator)
        simulator.current_assignments = {0: [1, 2]}
        simulator.current_chs = [0]
        simulator.energies = np.asarray([0.2, 0.8, 0.6])
        simulator.current_route_plan = None
        simulator.params = replace(
            SIMULATION_PARAMS,
            recluster_trigger_mode="cluster_energy_mean",
            recluster_interval=1,
            validate_route_every_round=False,
        )
        self.assertTrue(simulator._need_recluster(2))
        self.assertEqual(
            simulator._pending_reoptimization_reason,
            "cluster_head_below_cluster_mean",
        )


if __name__ == "__main__":
    unittest.main()
