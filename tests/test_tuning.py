import copy
import tempfile
import unittest
from pathlib import Path

from uwsn.algorithms.config import load_algorithm_config
from uwsn.experiment_config.generation import ExperimentGenerator
from uwsn.experiment_config.io import load_yaml
from uwsn.tuning import (
    aggregate_rankings,
    sample_trial_configs,
    select_stage_cases,
    stable_hash,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/tuning/kaggle_large.yaml"


class TuningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_yaml(SPEC_PATH)

    def test_sampling_is_reproducible_and_preserves_base(self) -> None:
        algorithm_id = "eulc_pso"
        base = load_algorithm_config(ROOT / self.spec["algorithms"][algorithm_id])
        original = copy.deepcopy(base)
        first = sample_trial_configs(
            base, algorithm_id, self.spec["search_spaces"][algorithm_id], 5, 10
        )
        second = sample_trial_configs(
            base, algorithm_id, self.spec["search_spaces"][algorithm_id], 5, 10
        )
        self.assertEqual(first, second)
        self.assertEqual(base, original)
        self.assertEqual(len({row["trial_id"] for row in first}), 5)
        self.assertTrue(all(row["config"]["protocol"]["ch_ratio"] == 0.05 for row in first))

    def test_budget_metadata_tracks_population_and_iterations(self) -> None:
        for algorithm_id in ("eulc_pso", "eulc_ga", "eulc_ac_aco"):
            base = load_algorithm_config(ROOT / self.spec["algorithms"][algorithm_id])
            trial = sample_trial_configs(
                base, algorithm_id, self.spec["search_spaces"][algorithm_id], 1, 3
            )[0]["config"]
            optimizer = trial["optimizer"]
            expected = optimizer["population_size"] * (
                optimizer["max_iterations"] + (0 if algorithm_id == "eulc_ac_aco" else 1)
            )
            self.assertEqual(optimizer["fitness_evaluation_budget"], expected)

    def test_stage_case_selection_is_shared_and_exact(self) -> None:
        cases = ExperimentGenerator(ROOT / self.spec["experiment_set"]).build_cases()
        selected = select_stage_cases(cases, self.spec["scenario_filter"], [0, 1])
        self.assertEqual(len(selected), 6)
        self.assertEqual({row["metadata"]["base_seed"] for row in selected}, {0, 1})
        self.assertEqual(
            {row["environment"]["distribution"] for row in selected},
            {"uniform", "gaussian", "exponential"},
        )

    def test_ranking_is_lexicographic_and_failed_trial_cannot_win(self) -> None:
        base = {
            "stage": "screen", "algorithm_id": "eulc_pso",
            "case_id": "case", "case_config_hash": "casehash", "base_seed": 0,
            "distribution": "uniform", "rounds_completed": 10,
            "fnd_round": None, "fnd_censored": True, "runtime_seconds": 1.0,
            "error": None,
        }
        good = {
            **base, "trial_id": "good", "trial_config_hash": "g",
            "status": "completed", "survival_score": 11,
            "residual_energy_auc": 0.8, "final_residual_energy_fraction": 0.7,
            "packets_delivered": 100, "average_e2e_delay_s": 1.0,
        }
        failed = {
            **base, "trial_id": "failed", "trial_config_hash": "f",
            "status": "failed", "survival_score": 0,
            "residual_energy_auc": 1.0, "final_residual_energy_fraction": 1.0,
            "packets_delivered": 1000, "average_e2e_delay_s": 0.0,
        }
        ranked = aggregate_rankings([failed, good], 1)
        self.assertEqual(ranked[0]["trial_id"], "good")
        self.assertEqual(ranked[0]["rank"], 1)

    def test_stable_hash_ignores_mapping_order(self) -> None:
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
