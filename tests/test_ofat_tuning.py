from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

from uwsn.algorithms.config import load_algorithm_config
from uwsn.experiment_config.generation import ExperimentGenerator
from uwsn.experiment_config.io import load_yaml
from uwsn.ofat_tuning import apply_coordinate, select_cases


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


if __name__ == "__main__":
    unittest.main()
