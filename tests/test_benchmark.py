import csv
import json
import tempfile
import unittest
from pathlib import Path

from uwsn.algorithms.config import load_algorithm_config
from uwsn.benchmark import _write_derived
from uwsn.experiment_config.generation import ExperimentGenerator


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_heldout_campaign_has_shared_thirty_cases(self) -> None:
        cases = ExperimentGenerator(
            ROOT / "configs/experiment_sets/benchmark_heldout.yaml"
        ).build_cases()
        self.assertEqual(len(cases), 30)
        self.assertEqual(
            {case["metadata"]["base_seed"] for case in cases}, set(range(10, 20))
        )
        self.assertEqual(
            {case["environment"]["distribution"] for case in cases},
            {"uniform", "gaussian", "exponential"},
        )

    def test_frozen_best_configs_match_validated_results(self) -> None:
        pso = load_algorithm_config(ROOT / "configs/benchmarks/best_eulc_pso.yaml")
        ga = load_algorithm_config(ROOT / "configs/benchmarks/best_eulc_ga.yaml")
        self.assertEqual((pso["optimizer"]["population_size"], pso["optimizer"]["max_iterations"]), (40, 24))
        self.assertEqual((pso["optimizer"]["params"]["c1"], pso["optimizer"]["params"]["c2"]), (2.0, 1.6))
        self.assertEqual((ga["optimizer"]["population_size"], ga["optimizer"]["max_iterations"]), (50, 19))
        self.assertEqual(ga["optimizer"]["params"], {"crossover_rate": 0.8, "mutation_sigma": 0.1})

    def test_derived_outputs_are_rebuilt_from_durable_results(self) -> None:
        result = {
            "case_key": "ga|case|10", "algorithm_id": "eulc_ga",
            "case_id": "case", "distribution": "uniform", "seed": 10,
            "status": "completed", "runtime_seconds": 1.0,
            "round_metrics": [{"round": 1, "residual_energy_j": 49.0}],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_derived(output, [result])
            self.assertTrue((output / "completed_cases.jsonl").exists())
            with (output / "case_summary.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            with (output / "round_metrics.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)


if __name__ == "__main__":
    unittest.main()
