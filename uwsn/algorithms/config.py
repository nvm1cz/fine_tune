from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..experiment_config.io import load_yaml
from .registry import CANONICAL_ALGORITHM_IDS, canonical_algorithm_id


class AlgorithmConfigError(ValueError):
    pass


COMMON_PROTOCOL_KEYS = {
    "ch_ratio", "candidate_ratio", "competition_radius_m",
    "layer_spacing_m", "recluster_interval_rounds", "recluster_trigger_mode",
}
EULC_PROTOCOL_KEYS = {
    "ch_ratio", "candidate_ratio", "initial_layer_width_m",
    "layer_spacing_increment_m", "competition_radius_mode",
    "competition_adjustment_factor", "recluster_interval_rounds",
    "recluster_trigger_mode",
}
OPTIMIZER_NAMES = {
    "eulc_pso": "pso", "eulc_ga": "ga",
    "eulc_ac_aco": "ac_aco",
}
COMMON_SELECTION_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "algorithms"
    / "common_selection.yaml"
)


def resolve_algorithm_config(config: dict[str, Any]) -> dict[str, Any]:
    """Inject the one shared CH/candidate ratio definition into an algorithm config."""
    resolved = copy.deepcopy(config)
    shared = load_yaml(COMMON_SELECTION_CONFIG)["protocol"]
    protocol = resolved.setdefault("protocol", {})
    duplicated = set(shared).intersection(protocol)
    if duplicated:
        raise AlgorithmConfigError(
            f"Shared selection keys must only be defined in common_selection.yaml: "
            f"{sorted(duplicated)}"
        )
    protocol.update(copy.deepcopy(shared))
    return resolved


def validate_algorithm_config(config: dict[str, Any]) -> None:
    allowed_top = {"algorithm", "protocol", "eulc", "eeumc", "optimizer"}
    unknown = set(config).difference(allowed_top)
    if unknown:
        raise AlgorithmConfigError(f"Unknown algorithm-config keys: {sorted(unknown)}")
    if "algorithm" not in config or "protocol" not in config:
        raise AlgorithmConfigError("algorithm and protocol sections are required")
    algorithm_meta = config["algorithm"]
    allowed_algorithm_keys = {"id", "family", "base_protocol", "optimizer"}
    unknown_meta = set(algorithm_meta).difference(allowed_algorithm_keys)
    if unknown_meta:
        raise AlgorithmConfigError(f"Unknown algorithm metadata: {sorted(unknown_meta)}")
    algorithm_id = canonical_algorithm_id(algorithm_meta["id"])
    if algorithm_id not in CANONICAL_ALGORITHM_IDS:
        raise AlgorithmConfigError(f"Unknown algorithm ID: {algorithm_id}")
    protocol_keys = (
        EULC_PROTOCOL_KEYS
        if algorithm_id in {"eulc", *OPTIMIZER_NAMES}
        else COMMON_PROTOCOL_KEYS
    )
    unknown_protocol = set(config["protocol"]).difference(protocol_keys)
    if unknown_protocol:
        raise AlgorithmConfigError(f"Unknown protocol keys: {sorted(unknown_protocol)}")
    required_protocol = (
        EULC_PROTOCOL_KEYS - {"ch_ratio", "recluster_trigger_mode"}
        if algorithm_id == "eulc"
        else protocol_keys - {"recluster_trigger_mode"}
    )
    missing_protocol = required_protocol.difference(config["protocol"])
    if missing_protocol:
        raise AlgorithmConfigError(f"Missing protocol keys: {sorted(missing_protocol)}")
    for key in ("candidate_ratio",):
        if not 0.0 < float(config["protocol"][key]) <= 1.0:
            raise AlgorithmConfigError(f"{key} must be in (0, 1]")
    if "ch_ratio" in config["protocol"] and not (
        0.0 < float(config["protocol"]["ch_ratio"]) <= 1.0
    ):
        raise AlgorithmConfigError("ch_ratio must be in (0, 1]")
    if algorithm_id in {"eulc", *OPTIMIZER_NAMES}:
        if float(config["protocol"]["initial_layer_width_m"]) <= 0.0:
            raise AlgorithmConfigError("initial_layer_width_m must be positive")
        if float(config["protocol"]["layer_spacing_increment_m"]) <= 0.0:
            raise AlgorithmConfigError("layer_spacing_increment_m must be positive")
        if config["protocol"]["competition_radius_mode"] != "dynamic_eulc":
            raise AlgorithmConfigError("competition_radius_mode must be dynamic_eulc")
        factor = float(config["protocol"]["competition_adjustment_factor"])
        if not 0.0 <= factor <= 1.0:
            raise AlgorithmConfigError("competition_adjustment_factor must be in [0, 1]")
    else:
        if float(config["protocol"]["competition_radius_m"]) <= 0.0:
            raise AlgorithmConfigError("competition_radius_m must be positive")
        if float(config["protocol"]["layer_spacing_m"]) <= 0.0:
            raise AlgorithmConfigError("layer_spacing_m must be positive")
    if int(config["protocol"]["recluster_interval_rounds"]) <= 0:
        raise AlgorithmConfigError("recluster_interval_rounds must be positive")
    if config["protocol"].get("recluster_trigger_mode", "periodic") not in {
        "periodic", "cluster_energy_mean", "hybrid"
    }:
        raise AlgorithmConfigError(
            "recluster_trigger_mode must be periodic, cluster_energy_mean or hybrid"
        )

    optimized = algorithm_id in OPTIMIZER_NAMES
    if optimized:
        if "eulc" not in config or "optimizer" not in config:
            raise AlgorithmConfigError(f"{algorithm_id} requires eulc and optimizer sections")
        expected = OPTIMIZER_NAMES[algorithm_id]
        if config["optimizer"].get("name") != expected:
            raise AlgorithmConfigError(
                f"{algorithm_id} requires optimizer.name={expected}"
            )
    elif "optimizer" in config:
        raise AlgorithmConfigError(f"{algorithm_id} must not contain optimizer")
    if algorithm_id == "eulc" and "optimizer" in config:
        raise AlgorithmConfigError("Pure EULC must not contain optimizer")
    if algorithm_id in {"eeumc", "ebrec", "leach"} and "eulc" in config:
        raise AlgorithmConfigError(f"{algorithm_id} must not contain EULC parameters")
    if algorithm_id == "eeumc" and set(config.get("eeumc", {})) != {"alpha"}:
        raise AlgorithmConfigError("EEUMC requires exactly eeumc.alpha")
    if algorithm_id in {"eulc", *OPTIMIZER_NAMES}:
        if set(config["eulc"]) != {"alpha", "beta", "gamma"}:
            raise AlgorithmConfigError("EULC requires exactly alpha, beta and gamma")
        if abs(sum(float(config["eulc"][key]) for key in ("alpha", "beta", "gamma")) - 1.0) > 1e-9:
            raise AlgorithmConfigError("EULC alpha + beta + gamma must equal 1")


def load_algorithm_config(path: Path) -> dict[str, Any]:
    config = resolve_algorithm_config(load_yaml(path))
    validate_algorithm_config(config)
    return config


def apply_algorithm_config(case: dict[str, Any], algorithm_config: dict[str, Any]) -> dict[str, Any]:
    """Resolve algorithm-only configuration without permitting environment overrides."""
    if not {"ch_ratio", "candidate_ratio"}.issubset(
        algorithm_config.get("protocol", {})
    ):
        algorithm_config = resolve_algorithm_config(algorithm_config)
    validate_algorithm_config(algorithm_config)
    resolved = copy.deepcopy(case)
    protocol = algorithm_config["protocol"]
    resolved["protocol"].update({
        # Pure EULC promotes its candidate set directly to CHs. This compatibility
        # value is unused by that baseline but keeps the shared runtime dataclass complete.
        "ch_ratio": protocol.get("ch_ratio", protocol["candidate_ratio"]),
        "candidate_ratio": protocol["candidate_ratio"],
        "competition_radius_m": protocol.get(
            "initial_layer_width_m", protocol.get("competition_radius_m")
        ),
        "layer_spacing_m": protocol.get(
            "layer_spacing_increment_m", protocol.get("layer_spacing_m")
        ),
        "recluster_interval_rounds": protocol["recluster_interval_rounds"],
        "recluster_trigger_mode": protocol.get("recluster_trigger_mode", "periodic"),
    })
    if "competition_radius_mode" in protocol:
        resolved["protocol"].update({
            "competition_radius_mode": protocol["competition_radius_mode"],
            "competition_adjustment_factor": protocol["competition_adjustment_factor"],
        })
    algorithm_id = canonical_algorithm_id(algorithm_config["algorithm"]["id"])
    if "eulc" in algorithm_config:
        weights = algorithm_config["eulc"]
        resolved["protocol"].update({
            "eulc_alpha": weights["alpha"], "eulc_beta": weights["beta"],
            "eulc_gamma": weights["gamma"],
        })
    elif "eeumc" in algorithm_config:
        resolved["protocol"]["eulc_alpha"] = algorithm_config["eeumc"]["alpha"]
    resolved["algorithm"] = copy.deepcopy(algorithm_config)
    resolved["metadata"]["algorithm_id"] = algorithm_id
    return resolved
