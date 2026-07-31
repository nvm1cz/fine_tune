import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uwsn.algorithms.config import load_algorithm_config
from uwsn.experiment_config.generation import ExperimentGenerator
from uwsn.experiment_config.io import load_yaml
from uwsn.tuning import (
    aggregate_rankings,
    sample_trial_configs,
    select_stage_cases,
    stable_hash,
    run_tuning,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/tuning/kaggle_large.yaml"


def deterministic_trial_runner(job):
    identity = job["identity"]
    return {
        **identity,
        "status": "completed",
        "rounds_completed": int(job["rounds"]),
        "fnd_round": None,
        "fnd_censored": True,
        "survival_score": int(job["rounds"]) + 1,
        "residual_energy_auc": 0.75,
        "final_residual_energy_fraction": 0.70,
        "packets_delivered": 100,
        "average_e2e_delay_s": 1.0,
        "runtime_seconds": 1.0,
        "error": None,
    }


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

    def test_plan_can_select_one_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = run_tuning(
                SPEC_PATH, Path(directory), execute=False,
                algorithm_id="eulc_ga",
            )
        self.assertEqual(plan["algorithms"], ["eulc_ga"])
        self.assertEqual(plan["stages"][0]["planned_runs"], 720)

    def test_one_shot_and_three_resumable_batches_are_identical(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["candidates_per_algorithm"] = 3
        spec["stages"] = [
            {"name": "validate", "rounds": 2, "seeds": [0], "keep": 1}
        ]
        with (
            tempfile.TemporaryDirectory() as one_shot_dir,
            tempfile.TemporaryDirectory() as first_batch_dir,
            tempfile.TemporaryDirectory() as batch_dir,
        ):
            one_shot = Path(one_shot_dir)
            first_batch = Path(first_batch_dir)
            batched = Path(batch_dir)
            checkpoint_counts = []

            def inspect_checkpoint(directory: Path, _note: str) -> None:
                checkpoint_counts.append(
                    json.loads(
                        (directory / "manifest.json").read_text(encoding="utf-8")
                    )["completed_trial_keys"]
                )

            with patch("uwsn.tuning.load_yaml", return_value=spec):
                run_tuning(
                    SPEC_PATH,
                    one_shot,
                    workers=1,
                    execute=True,
                    algorithm_id="eulc_pso",
                    stage_name="validate",
                    checkpoint_callback=inspect_checkpoint,
                    trial_runner=deterministic_trial_runner,
                )
                run_tuning(
                    SPEC_PATH,
                    first_batch,
                    workers=1,
                    execute=True,
                    algorithm_id="eulc_pso",
                    stage_name="validate",
                    config_start=0,
                    config_end=1,
                    trial_runner=deterministic_trial_runner,
                )
                run_tuning(
                    SPEC_PATH,
                    batched,
                    workers=1,
                    execute=True,
                    algorithm_id="eulc_pso",
                    stage_name="validate",
                    config_start=1,
                    config_end=2,
                    resume_from=first_batch,
                    trial_runner=deterministic_trial_runner,
                )
                run_tuning(
                    SPEC_PATH,
                    batched,
                    workers=1,
                    execute=True,
                    algorithm_id="eulc_pso",
                    stage_name="validate",
                    config_start=2,
                    config_end=3,
                    trial_runner=deterministic_trial_runner,
                )

            def canonical_jsonl(path: Path):
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                return sorted(rows, key=lambda row: row["trial_key"])

            def canonical_csv(path: Path):
                with path.open(newline="", encoding="utf-8") as handle:
                    return sorted(
                        list(csv.DictReader(handle)),
                        key=lambda row: (row["algorithm_id"], row["trial_id"]),
                    )

            self.assertEqual(
                canonical_jsonl(one_shot / "trials.jsonl"),
                canonical_jsonl(batched / "trials.jsonl"),
            )
            self.assertEqual(
                canonical_csv(one_shot / "rankings_validate.csv"),
                canonical_csv(batched / "rankings_validate.csv"),
            )
            self.assertEqual(
                (one_shot / "best_config_pso.yaml").read_text(encoding="utf-8"),
                (batched / "best_config_pso.yaml").read_text(encoding="utf-8"),
            )
            self.assertTrue(
                json.loads(
                    (batched / "manifest.json").read_text(encoding="utf-8")
                )["completed"]
            )
            self.assertEqual(checkpoint_counts[:9], list(range(1, 10)))

    def test_wall_time_stop_is_clean_and_resumable(self) -> None:
        spec = copy.deepcopy(self.spec)
        spec["candidates_per_algorithm"] = 1
        spec["stages"] = [
            {"name": "screen", "rounds": 2, "seeds": [0], "keep": 1}
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("uwsn.tuning.load_yaml", return_value=spec):
                result = run_tuning(
                    SPEC_PATH,
                    output,
                    workers=1,
                    execute=True,
                    algorithm_id="eulc_pso",
                    stage_name="screen",
                    max_wall_time_seconds=1,
                    trial_runner=deterministic_trial_runner,
                )
            self.assertFalse(result["completed"])
            self.assertEqual(result["stop_reason"], "time_budget")
            self.assertEqual(result["completed_trial_keys"], 0)
            self.assertFalse((output / "trials.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
