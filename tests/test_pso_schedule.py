import unittest
from dataclasses import replace

import numpy as np

from scripts.run_pso_exact_sequential_tuning import detect_plateau
from uwsn.optimizers.pso import inertia_weight, swarm_diversity_metrics
from uwsn.run_config import SIMULATION_PARAMS


class PSOInertiaScheduleTests(unittest.TestCase):
    def test_fixed_schedule_preserves_legacy_inertia(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            pso_iterations=50,
            pso_inertia=0.73,
            pso_inertia_schedule="fixed",
        )
        self.assertEqual(inertia_weight(params, 1), 0.73)
        self.assertEqual(inertia_weight(params, 50), 0.73)

    def test_linear_schedule_hits_both_endpoints(self) -> None:
        params = replace(
            SIMULATION_PARAMS,
            pso_iterations=50,
            pso_inertia_schedule="linear",
            pso_inertia_start=0.9,
            pso_inertia_end=0.4,
        )
        self.assertAlmostEqual(inertia_weight(params, 1), 0.9)
        self.assertAlmostEqual(inertia_weight(params, 50), 0.4)
        self.assertAlmostEqual(inertia_weight(params, 25), 0.9 + 24 / 49 * (0.4 - 0.9))

    def test_unknown_schedule_is_rejected(self) -> None:
        params = replace(SIMULATION_PARAMS, pso_inertia_schedule="cosine")
        with self.assertRaisesRegex(ValueError, "Unsupported PSO inertia schedule"):
            inertia_weight(params, 1)

    def test_plateau_requires_consecutive_relative_improvements(self) -> None:
        improving = [1.0, 0.9, 0.8]
        flat = [0.8] * 20
        self.assertEqual(detect_plateau(improving + flat, 0.001, 20), 22)
        self.assertIsNone(detect_plateau([1.0, 0.9, 0.8], 0.001, 20))

    def test_diversity_is_mean_dimension_std_scaled_by_diagonal(self) -> None:
        swarm = np.asarray([[0.0, 0.0], [1.0, 1.0]])
        normalized, equivalent = swarm_diversity_metrics(swarm, 100.0)
        self.assertAlmostEqual(normalized, 0.5)
        self.assertAlmostEqual(equivalent, 50.0)


if __name__ == "__main__":
    unittest.main()
