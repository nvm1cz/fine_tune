import copy
import tempfile
import unittest
from pathlib import Path

from uwsn.cases import SimulationCase
from uwsn.algorithms.config import load_algorithm_config
from uwsn.experiment_config.generation import (
    ExperimentGenerator,
    canonical_hash,
    readable_case_id,
    seed_streams,
)
from uwsn.experiment_config.io import load_yaml, write_yaml
from uwsn.experiment_config.runner import build_simulator
from uwsn.experiment_config.validation import ConfigError, validate_case
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import UWSNSimulator


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "configs/experiment_sets/benchmark.yaml"
BASE_CHANNEL = ROOT / "configs/base/channel.yaml"


class ExperimentConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.benchmark_cases = ExperimentGenerator(BENCHMARK).build_cases()

    def test_node_count_mapping(self) -> None:
        case = next(c for c in self.benchmark_cases if
                    c["environment"]["deployment_size_m"] == [500, 500, 500]
                    and c["metadata"]["density"] == "dense")
        self.assertEqual(case["environment"]["node_count"], 1000)

    def test_sink_position_is_top_surface_center(self) -> None:
        case = next(c for c in self.benchmark_cases
                    if c["environment"]["deployment_size_m"] == [300, 300, 300])
        self.assertEqual(case["environment"]["sink_position"], [150.0, 150.0, 0.0])

    def test_case_id_and_hash_are_deterministic(self) -> None:
        case = self.benchmark_cases[0]
        clone = copy.deepcopy(case)
        clone["metadata"]["generated_at"] = "different"
        self.assertEqual(readable_case_id(case), readable_case_id(clone))
        self.assertEqual(canonical_hash(case), canonical_hash(clone))

    def test_seed_streams_are_reproducible_and_optimizer_independent(self) -> None:
        first = seed_streams(3)
        self.assertEqual(first, seed_streams(3))
        scenario = next(
            c for c in self.benchmark_cases if c["metadata"]["base_seed"] == 3
        )
        self.assertNotIn("optimizer", scenario)
        self.assertEqual(scenario["metadata"]["topology_seed"], first["topology"])

    def test_cartesian_expansion_and_baseline_axes(self) -> None:
        self.assertEqual(len(self.benchmark_cases), 4 * 3 * 3 * 10)
        algorithm_only_keys = {
            "ch_ratio", "candidate_ratio", "eulc_alpha", "eulc_beta", "eulc_gamma",
            "competition_radius_m", "layer_spacing_m", "recluster_interval_rounds",
        }
        self.assertTrue(
            algorithm_only_keys.isdisjoint(self.benchmark_cases[0]["protocol"])
        )
        self.assertEqual({c["environment"]["initial_energy_j"] for c in self.benchmark_cases}, {0.5})
        self.assertEqual({c["environment"]["packet_size_bits"] for c in self.benchmark_cases}, {6400})
        self.assertEqual(
            {c["channel"]["transmission_range_m"] for c in self.benchmark_cases},
            {200.0},
        )

    def test_fixed_channel_and_mobility_baseline(self) -> None:
        case = self.benchmark_cases[0]
        channel = case["channel"]
        execution = case["execution"]
        self.assertEqual(channel["frequency_khz"], 10.0)
        self.assertEqual(channel["spreading_factor"], 1.5)
        self.assertEqual(channel["bandwidth_hz"], 10000.0)
        self.assertEqual(channel["node_bit_rate_bps"], 10000.0)
        self.assertEqual(channel["sound_speed_mps"], 1500.0)
        self.assertEqual(channel["directivity_index_db"], 0.0)
        self.assertEqual(channel["acoustic_distance_unit"], "m")
        self.assertEqual(channel["reference_distance_m"], 1.0)
        self.assertEqual(channel["fading_model"], "none")
        self.assertNotIn("rician_k_linear", channel)
        self.assertTrue(channel["enable_packet_errors"])
        self.assertTrue(channel["enable_retransmissions"])
        self.assertEqual(channel["max_retries"], 3)
        self.assertEqual(channel["modem_transition_delay_s"], 1.5)
        self.assertTrue(execution["mobility_enabled"])
        self.assertEqual(execution["mobility_model"], "tidal_rbf")
        self.assertEqual(execution["mobility_update_interval_rounds"], 2)

    def test_transmission_range_has_no_base_default(self) -> None:
        self.assertNotIn("transmission_range_m", load_yaml(BASE_CHANNEL)["channel"])

    def test_invalid_and_unknown_config_are_rejected(self) -> None:
        invalid = copy.deepcopy(self.benchmark_cases[0])
        invalid["execution"]["rounds"] = 0
        with self.assertRaises(ConfigError):
            validate_case(invalid)
        unknown = copy.deepcopy(self.benchmark_cases[0])
        unknown["channel"]["typo_frequency"] = 10
        with self.assertRaises(ConfigError):
            validate_case(unknown)
        invalid_reference = copy.deepcopy(self.benchmark_cases[0])
        invalid_reference["channel"]["reference_distance_m"] = -1.0
        with self.assertRaises(ConfigError):
            validate_case(invalid_reference)
        invalid_energy = copy.deepcopy(self.benchmark_cases[0])
        invalid_energy["protocol"]["transmit_power_p0"] = -1.0
        with self.assertRaises(ConfigError):
            validate_case(invalid_energy)
        missing_source_level = copy.deepcopy(self.benchmark_cases[0])
        missing_source_level["channel"]["enable_packet_errors"] = True
        missing_source_level["channel"]["source_level_db"] = None
        with self.assertRaises(ConfigError):
            validate_case(missing_source_level)

    def test_yaml_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.yaml"
            write_yaml(path, self.benchmark_cases[0])
            self.assertEqual(load_yaml(path), self.benchmark_cases[0])

    def test_dry_run_build_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            generator = ExperimentGenerator(BENCHMARK)
            cases = generator.build_cases()
            generator.summary(cases, output)
            self.assertFalse(output.exists())

    def test_case_rounds_are_used_when_max_rounds_is_omitted(self) -> None:
        case = SimulationCase("round-limit", 100.0, 100, node_count=1, rounds=2)
        params = copy.deepcopy(SIMULATION_PARAMS)
        params.optimizer = "leach"
        simulator = UWSNSimulator(case, params, seed=1, verbose=False)
        metrics = simulator.run(stop_on_first_dead=False)
        self.assertEqual(len(metrics.residual_energy_by_round), 2)

    def test_generated_case_builds_simulator_with_exact_seed_streams(self) -> None:
        algorithm = load_yaml(ROOT / "configs/algorithms/eulc_pso.yaml")
        simulator, execution = build_simulator(self.benchmark_cases[0], algorithm)
        self.assertEqual(simulator.case.rounds, execution["rounds"])
        self.assertEqual(simulator.seed_streams["topology"],
                         self.benchmark_cases[0]["metadata"]["topology_seed"])

    def test_pure_eulc_uses_candidate_ratio_without_ch_ratio(self) -> None:
        raw_algorithm = load_yaml(ROOT / "configs/algorithms/eulc.yaml")
        self.assertNotIn("ch_ratio", raw_algorithm["protocol"])
        self.assertNotIn("candidate_ratio", raw_algorithm["protocol"])
        algorithm = load_algorithm_config(ROOT / "configs/algorithms/eulc.yaml")
        self.assertEqual(algorithm["protocol"]["ch_ratio"], 0.05)
        self.assertEqual(algorithm["protocol"]["candidate_ratio"], 0.2)
        simulator, _ = build_simulator(self.benchmark_cases[0], algorithm)
        self.assertEqual(simulator.params.eulc_candidate_ratio, 0.2)


if __name__ == "__main__":
    unittest.main()
