import unittest
from dataclasses import replace

from uwsn.optimizers.pso import inertia_weight
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


if __name__ == "__main__":
    unittest.main()
