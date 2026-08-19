import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from scripts.run_pso_exact_sequential_tuning import _evaluate, detect_plateau
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

    def test_trial_level_resume_skips_completed_phase_label_seed(self) -> None:
        trials = []
        curves = []
        config = {"optimizer": {"population_size": 20, "params": {"inertia": 0.9}}}

        def fake_run(template, algorithm_config, seed, phase, label):
            return ({
                "phase": phase,
                "label": label,
                "seed": seed,
                "best_J": 1.0 + seed,
                "runtime_seconds": 0.1,
                "particle_dimension": 5,
            }, [{
                "phase": phase,
                "label": label,
                "seed": seed,
                "iteration": 0,
                "best_J": 1.0 + seed,
            }])

        with patch(
            "scripts.run_pso_exact_sequential_tuning._run_once",
            side_effect=fake_run,
        ) as run_once:
            _evaluate({}, "step", [("candidate", config)], [0, 1], 1, trials, curves)
            self.assertEqual(run_once.call_count, 2)
            _evaluate({}, "step", [("candidate", config)], [0, 1], 1, trials, curves)
            self.assertEqual(run_once.call_count, 2)

        self.assertEqual(len(trials), 2)
        self.assertEqual(len(curves), 2)


if __name__ == "__main__":
    unittest.main()
