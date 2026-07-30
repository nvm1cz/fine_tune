import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from uwsn.algorithms import NetworkState
from uwsn.algorithms.config import (
    AlgorithmConfigError,
    load_algorithm_config,
    validate_algorithm_config,
)
from uwsn.algorithms.ebrec import EBRECProtocol
from uwsn.algorithms.eeumc import EEUMCProtocol
from uwsn.algorithms.eulc import EULCProtocol
from uwsn.algorithms.leach import LEACHProtocol
from uwsn.algorithms.protocols import EULCOptimizedProtocol
from uwsn.algorithms.registry import (
    ALGORITHM_REGISTRY,
    CANONICAL_ALGORITHM_IDS,
    create_algorithm,
)
from uwsn.cases import SimulationCase
from uwsn.experiment_config.generation import ExperimentGenerator
from uwsn.experiment_config.io import load_yaml, write_yaml
from uwsn.experiment_config.runner import run_case_file
from uwsn.run_config import SIMULATION_PARAMS
from uwsn.simulator import UWSNSimulator


ROOT = Path(__file__).resolve().parents[1]


class AlgorithmArchitectureTests(unittest.TestCase):
    def test_registry_contains_exactly_seven_ids(self) -> None:
        self.assertEqual(tuple(ALGORITHM_REGISTRY), CANONICAL_ALGORITHM_IDS)
        self.assertEqual(len(CANONICAL_ALGORITHM_IDS), 7)

    def test_factory_creates_distinct_protocol_classes(self) -> None:
        expected = {
            "eeumc": EEUMCProtocol, "ebrec": EBRECProtocol,
            "eulc": EULCProtocol, "leach": LEACHProtocol,
        }
        for algorithm_id, cls in expected.items():
            self.assertIsInstance(create_algorithm(algorithm_id), cls)
        for algorithm_id in ("eulc_pso", "eulc_ga", "eulc_ac_aco"):
            self.assertIsInstance(create_algorithm(algorithm_id), EULCOptimizedProtocol)

    def test_pure_eulc_and_leach_do_not_have_optimizer(self) -> None:
        self.assertFalse(hasattr(create_algorithm("eulc"), "optimizer"))
        leach = create_algorithm("leach")
        self.assertFalse(hasattr(leach, "optimizer"))
        self.assertFalse(leach.uses_eulc_candidates)

    def test_pure_eulc_promotes_candidates_directly_to_cluster_heads(self) -> None:
        algorithm = create_algorithm("eulc")
        case = SimulationCase("eulc-candidates", 10.0, 100, node_count=20, rounds=1)
        params = replace(
            SIMULATION_PARAMS,
            optimizer="eulc",
            eulc_candidate_ratio=0.4,
            transmission_range_m=1000.0,
        )
        simulator = UWSNSimulator(case, params, seed=9, verbose=False, algorithm=algorithm)
        simulator._refresh_clusters(1)
        self.assertEqual(
            set(simulator.current_chs),
            set(algorithm.eulc.generate_candidates(
                NetworkState(
                    simulator.case,
                    simulator.params,
                    simulator.positions,
                    simulator.energies,
                    simulator.layers,
                    simulator.dist_to_sink,
                    simulator.neighbor_degree,
                    simulator.distance_matrix,
                )
            )),
        )

    def test_hybrid_uses_eulc_output_as_optimizer_candidate_pool(self) -> None:
        algorithm = create_algorithm("eulc_pso")
        case = SimulationCase("eulc-hybrid", 10.0, 100, node_count=20, rounds=1)
        params = replace(
            SIMULATION_PARAMS,
            optimizer="eulc_pso",
            eulc_candidate_ratio=0.4,
            cluster_head_ratio=0.2,
            transmission_range_m=1000.0,
            pso_particles=4,
            pso_iterations=1,
        )
        simulator = UWSNSimulator(case, params, seed=10, verbose=False, algorithm=algorithm)
        candidate_pool = set(algorithm.eulc.generate_candidates(
            NetworkState(
                simulator.case,
                simulator.params,
                simulator.positions,
                simulator.energies,
                simulator.layers,
                simulator.dist_to_sink,
                simulator.neighbor_degree,
                simulator.distance_matrix,
            )
        ))
        simulator._refresh_clusters(1)
        self.assertTrue(set(simulator.current_chs).issubset(candidate_pool))
        self.assertEqual(simulator.algorithm_diagnostics["candidate_count"], len(candidate_pool))

    def test_optimized_eulc_uses_requested_optimizer_and_shared_components(self) -> None:
        expected = {
            "eulc_pso": "pso", "eulc_ga": "ga",
            "eulc_ac_aco": "ac_aco",
        }
        component_types = set()
        for algorithm_id, optimizer_name in expected.items():
            algorithm = create_algorithm(algorithm_id)
            self.assertEqual(algorithm.optimizer.name, optimizer_name)
            component_types.add(type(algorithm.eulc))
        self.assertEqual(len(component_types), 1)

    def test_unknown_id_and_wrong_optimizer_config_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_algorithm("unknown")
        config = load_algorithm_config(ROOT / "configs/algorithms/eulc_pso.yaml")
        invalid = copy.deepcopy(config)
        invalid["optimizer"]["name"] = "ga"
        with self.assertRaises(AlgorithmConfigError):
            validate_algorithm_config(invalid)

    def test_all_algorithm_configs_validate(self) -> None:
        for algorithm_id in CANONICAL_ALGORITHM_IDS:
            config = load_algorithm_config(ROOT / f"configs/algorithms/{algorithm_id}.yaml")
            self.assertEqual(config["algorithm"]["id"], algorithm_id)
            self.assertEqual(config["protocol"]["ch_ratio"], 0.05)
            self.assertEqual(config["protocol"]["candidate_ratio"], 0.2)

    def test_all_eulc_variants_share_requested_eulc4_parameters(self) -> None:
        for algorithm_id in (
            "eulc", "eulc_pso", "eulc_ga", "eulc_ac_aco"
        ):
            config = load_algorithm_config(
                ROOT / f"configs/algorithms/{algorithm_id}.yaml"
            )
            protocol = config["protocol"]
            self.assertEqual(protocol["initial_layer_width_m"], 20.0)
            self.assertEqual(protocol["layer_spacing_increment_m"], 10.0)
            self.assertEqual(protocol["competition_radius_mode"], "dynamic_eulc")
            self.assertEqual(protocol["competition_adjustment_factor"], 0.5)
            self.assertEqual(protocol["recluster_interval_rounds"], 1)
            self.assertEqual(
                config["eulc"],
                {"alpha": 0.4, "beta": 0.4, "gamma": 0.2},
            )

    def test_same_seed_produces_same_topology_for_all_algorithms(self) -> None:
        case = SimulationCase("topology", 10.0, 100, node_count=20, rounds=1)
        positions = []
        for algorithm_id in CANONICAL_ALGORITHM_IDS:
            params = replace(SIMULATION_PARAMS, optimizer=algorithm_id)
            simulator = UWSNSimulator(case, params, seed=123, verbose=False)
            positions.append(simulator.positions)
        for value in positions[1:]:
            np.testing.assert_array_equal(positions[0], value)

    def test_smoke_all_algorithms(self) -> None:
        case = SimulationCase("smoke", 10.0, 100, node_count=20, rounds=3)
        for algorithm_id in CANONICAL_ALGORITHM_IDS:
            updates = {
                "optimizer": algorithm_id, "pso_particles": 6, "pso_iterations": 2,
                "ac_aco_ants": 6, "ac_aco_iterations": 2,
            }
            params = replace(SIMULATION_PARAMS, **updates)
            simulator = UWSNSimulator(case, params, seed=7, verbose=False)
            metrics = simulator.run(stop_on_first_dead=False)
            self.assertEqual(len(metrics.residual_energy_by_round), 3, algorithm_id)
            self.assertEqual(simulator.algorithm_id, algorithm_id)

    def test_one_case_runner_writes_canonical_algorithm_directory(self) -> None:
        case = ExperimentGenerator(
            ROOT / "configs/experiment_sets/benchmark.yaml"
        ).build_cases()[0]
        case["environment"]["node_count"] = 20
        case["execution"]["rounds"] = 3
        case["metadata"]["case_id"] = "runner-smoke"
        with tempfile.TemporaryDirectory() as directory:
            case["output"]["root_dir"] = directory
            case_path = Path(directory) / "case.yaml"
            write_yaml(case_path, case)
            result = run_case_file(case_path, algorithm_id="eulc")
            self.assertEqual(result.name, "eulc")
            self.assertTrue((result / "resolved_config.yaml").exists())
            summary = load_yaml(result / "summary.json")
            self.assertEqual(summary["algorithm_id"], "eulc")
            self.assertEqual(summary["rounds_completed"], 3)


if __name__ == "__main__":
    unittest.main()
