from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

from uwsn.algorithms.config import load_algorithm_config
from uwsn.experiment_config.generation import ExperimentGenerator
from uwsn.experiment_config.io import load_yaml
from uwsn.ofat_tuning import apply_coordinate, select_cases
from uwsn.scenario_ofat_tuning import group_cases_by_scenario, scenario_key


ROOT = Path(__file__).resolve().parents[1]


class OfatTuningTests(unittest.TestCase):
    def test_full_matrix_has_144_environment_cases_per_seed(self) -> None:
        cases = ExperimentGenerator(ROOT / "configs/experiment_sets/tuning_ofat_full.yaml").build_cases()
        self.assertEqual(len(cases), 1440)
        selected = select_cases(cases, [0, 1])
        self.assertEqual(len(selected), 288)
        signatures = {
            (tuple(c["environment"]["deployment_size_m"]), c["environment"]["node_count"],
             c["environment"]["distribution"], c["environment"]["initial_energy_j"],
             c["environment"]["packet_size_bits"])
            for c in select_cases(cases, [0])
        }
        self.assertEqual(len(signatures), 144)

    def test_per_scenario_groups_exclude_seed_and_cover_full_matrix(self) -> None:
        cases = ExperimentGenerator(ROOT / "configs/experiment_sets/tuning_ofat_full.yaml").build_cases()
        selected = select_cases(cases, [0, 1])
        grouped = group_cases_by_scenario(selected)
        self.assertEqual(len(grouped), 144)
        self.assertTrue(all(len(rows) == 2 for rows in grouped.values()))
        first = next(iter(grouped.values()))
        self.assertEqual(scenario_key(first[0]), scenario_key(first[1]))
        self.assertNotEqual(first[0]["metadata"]["base_seed"], first[1]["metadata"]["base_seed"])

    def test_scenario_stages_use_disjoint_tuning_and_validation_seeds(self) -> None:
        spec = load_yaml(ROOT / "configs/tuning/kaggle_scenario_ofat.yaml")
        seeds = {row["name"]: set(row["seeds"]) for row in spec["stages"]}
        self.assertFalse(seeds["screen"] & seeds["refine"])
        self.assertFalse((seeds["screen"] | seeds["refine"]) & seeds["validate"])

    def test_one_coordinate_changes_while_others_stay_fixed(self) -> None:
        base = load_algorithm_config(ROOT / "configs/algorithms/eulc_pso.yaml")
        changed = apply_coordinate(base, "eulc_pso", "c1", 0.5)
        self.assertEqual(changed["optimizer"]["params"]["c1"], 0.5)
        self.assertEqual(changed["optimizer"]["params"]["c2"], base["optimizer"]["params"]["c2"])
        self.assertEqual(changed["optimizer"]["population_size"], base["optimizer"]["population_size"])

    def test_spec_has_deterministic_coordinate_order_for_all_algorithms(self) -> None:
        spec = load_yaml(ROOT / "configs/tuning/kaggle_ofat_full.yaml")
        self.assertEqual(set(spec["coordinates"]), {"eulc_pso", "eulc_ga", "eulc_ac_aco"})
        self.assertEqual(spec["coordinates"]["eulc_ga"][0]["name"], "budget_pair")
        self.assertEqual(spec["coordinates"]["eulc_pso"][1]["name"], "inertia")
        self.assertEqual(spec["coordinates"]["eulc_ac_aco"][1]["name"], "alpha")

    def test_cli_accepts_auto_stage(self) -> None:
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "scripts/run_kaggle_ofat_tuning.py"),
                "--algorithm", "eulc_ga", "--stage", "auto",
                "--output-dir", "/tmp/uwsn-ofat-cli-plan",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"stage": "screen"', result.stdout)

    def test_scenario_cli_plan_has_144_independent_scenarios(self) -> None:
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "scripts/run_kaggle_scenario_ofat.py"),
                "--algorithm", "eulc_ga", "--stage", "screen",
                "--output-dir", "/tmp/uwsn-scenario-ofat-cli-plan",
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"mode": "per_scenario"', result.stdout)
        self.assertIn('"scenario_count": 144', result.stdout)


if __name__ == "__main__":
    unittest.main()
