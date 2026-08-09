import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_kaggle_benchmarks import (
    create_outputs,
    load_benchmark,
    validate_paired_cases,
)


class BenchmarkAnalysisTests(unittest.TestCase):
    def _dataset(self, root: Path, algorithm_id: str) -> None:
        root.mkdir(parents=True)
        (root / "benchmark_manifest.json").write_text(json.dumps({
            "algorithm_id": algorithm_id, "completed": True,
            "completed_cases": 30, "git_commit": "commit",
            "algorithm_config_hash": "algorithm", "experiment_spec_hash": "spec",
        }))
        rows = []
        for seed in range(10):
            for distribution in ("uniform", "gaussian", "exponential"):
                rows.append({
                    "distribution": distribution, "seed": seed,
                    "status": "completed", "fnd_round": 10, "hnd_round": 20,
                    "lnd_round": 30, "final_residual_energy_j": 1.0,
                    "total_energy_consumed_j": 49.0, "packet_delivery_ratio": 0.9,
                    "average_e2e_delay_s": 1.0, "runtime_seconds": 2.0,
                })
        pd.DataFrame(rows).to_csv(root / "case_summary.csv", index=False)
        pd.DataFrame([{
            "round": 1, "residual_energy_j": 49.0,
            "distribution": "uniform", "seed": 0,
        }]).to_csv(root / "round_metrics.csv", index=False)

    def test_loads_complete_datasets_and_validates_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = []
            round_frames = []
            for algorithm_id in ("eulc_pso", "eulc_ga", "eulc_ac_aco"):
                target = root / algorithm_id
                self._dataset(target, algorithm_id)
                summary, rounds, _ = load_benchmark(target)
                frames.append(summary)
                round_frames.append(rounds)
            combined = pd.concat(frames, ignore_index=True)
            validate_paired_cases(combined)
            output = root / "analysis"
            create_outputs(
                combined,
                pd.concat(round_frames, ignore_index=True),
                output,
            )
            self.assertTrue((output / "summary_by_algorithm.csv").exists())
            self.assertTrue((output / "residual_energy_by_round.png").exists())
            self.assertTrue((output / "benchmark_metric_distributions.pdf").exists())


if __name__ == "__main__":
    unittest.main()
