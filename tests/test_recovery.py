import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uwsn.recovery import run_ac_aco_recovery


ROOT = Path(__file__).resolve().parents[1]


class RecoveryTests(unittest.TestCase):
    def test_recovery_stops_at_first_feasible_attempt_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = base / "source"
            output = base / "output"
            source.mkdir()
            (source / "benchmark_manifest.json").write_text(json.dumps({
                "algorithm_id": "eulc_ac_aco", "git_commit": "old",
                "algorithm_config_hash": "config", "experiment_spec_hash": "spec",
            }))
            with (source / "case_summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "distribution", "seed", "status",
                ])
                writer.writeheader()
                writer.writerow({
                    "distribution": "exponential", "seed": 11,
                    "status": "infeasible",
                })
            calls = []

            def runner(job):
                calls.append(job["algorithm_config"]["optimizer"]["params"]["chaos_initial_state"])
                feasible = len(calls) == 2
                return {
                    "case_key": job["case_key"], "algorithm_id": "eulc_ac_aco",
                    "case_id": "case", "distribution": "exponential", "seed": 11,
                    "config_hash": "case", "rounds_completed": 1 if feasible else 0,
                    "fnd_round": None, "hnd_round": None, "lnd_round": None,
                    "final_residual_energy_j": 1.0 if feasible else None,
                    "total_energy_consumed_j": 49.0 if feasible else None,
                    "packets_generated": 1 if feasible else 0,
                    "packets_delivered": 1 if feasible else 0,
                    "packet_delivery_ratio": 1.0 if feasible else 0.0,
                    "average_e2e_delay_s": 1.0 if feasible else None,
                    "runtime_seconds": 1.0, "round_metrics": [],
                    "status": "completed" if feasible else "infeasible",
                    "invalid_reason": None if feasible else "no_feasible_solution",
                    "error": None,
                }

            cases = [{
                "metadata": {"base_seed": 11, "case_id": "case"},
                "environment": {"distribution": "exponential"},
            }]
            with patch(
                "uwsn.recovery.ExperimentGenerator.build_cases",
                return_value=cases,
            ):
                manifest = run_ac_aco_recovery(
                    ROOT, source, output, seeds=[11], distribution="exponential",
                    chaos_states=[0.1, 0.3, 0.5], budget=1000,
                    max_wall_time_seconds=1000, case_runner=runner,
                )
                second = run_ac_aco_recovery(
                    ROOT, source, output, seeds=[11], distribution="exponential",
                    chaos_states=[0.1, 0.3, 0.5], budget=1000,
                    max_wall_time_seconds=1000, case_runner=runner,
                )
            self.assertEqual(calls, [0.1, 0.3])
            self.assertEqual(manifest["recovered_seeds"], [11])
            self.assertTrue(manifest["completed"])
            self.assertEqual(second["attempts_completed"], 2)


if __name__ == "__main__":
    unittest.main()
