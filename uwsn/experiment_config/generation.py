from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io import load_yaml, write_yaml
from .validation import ConfigError, SECTION_KEYS, validate_case


def merge_sections(configs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge disjoint top-level sections and reject accidental overrides."""
    merged: dict[str, Any] = {}
    for config in configs:
        overlap = set(merged).intersection(config)
        if overlap:
            raise ConfigError(f"Conflicting top-level sections: {sorted(overlap)}")
        merged.update(copy.deepcopy(config))
    return merged


def seed_streams(base_seed: int) -> dict[str, int]:
    children = np.random.SeedSequence(int(base_seed)).spawn(5)
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(
            ("topology", "protocol", "optimizer", "channel", "mobility"), children
        )
    }


def canonical_hash(case: dict[str, Any]) -> str:
    value = copy.deepcopy(case)
    metadata = value.get("metadata", {})
    for key in ("case_index", "case_id", "config_hash", "generated_at", "code_version", "status"):
        metadata.pop(key, None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def readable_case_id(case: dict[str, Any]) -> str:
    meta, env, channel = case["metadata"], case["environment"], case["channel"]
    dims = "x".join(str(int(v)) for v in env["deployment_size_m"])
    return (
        f"{meta['experiment_set']}__{dims}__{meta['density']}"
        f"__{env['distribution']}__r{int(channel['transmission_range_m'])}"
        f"__s{int(meta['base_seed']):02d}"
    )


def _git_version(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class ExperimentGenerator:
    """Expand a strict experiment-set definition into reproducible case files."""

    def __init__(self, experiment_set_path: Path) -> None:
        self.path = experiment_set_path.resolve()
        self.root = self.path.parents[2]
        self.spec = load_yaml(self.path)
        allowed = {"experiment_set", "base_configs", "algorithms", "mode", "deployment_sizes",
                   "densities", "distributions", "seeds", "scenarios", "overrides",
                   "node_count_mapping", "sweep"}
        unknown = set(self.spec).difference(allowed)
        if unknown:
            raise ConfigError(f"Unknown experiment-set keys: {sorted(unknown)}")

    def _load_base(self) -> dict[str, Any]:
        paths = [self.root / path for path in self.spec["base_configs"]]
        return merge_sections(load_yaml(path) for path in paths)

    def _scenario_rows(self) -> list[tuple[list[int], str, str]]:
        if self.spec["mode"] == "cartesian":
            return [
                (size, density, distribution)
                for size in self.spec["deployment_sizes"]
                for density in self.spec["densities"]
                for distribution in self.spec["distributions"]
            ]
        if self.spec["mode"] == "scenarios":
            return [
                (row["deployment_size_m"], row["density"], row["distribution"])
                for row in self.spec["scenarios"]
            ]
        raise ConfigError("mode must be cartesian or scenarios")

    def build_cases(self) -> list[dict[str, Any]]:
        base = self._load_base()
        mapping = self.spec["node_count_mapping"]
        cases: list[dict[str, Any]] = []
        generated_at = datetime.now(timezone.utc).isoformat()
        version = _git_version(self.root)
        for size, density, distribution in self._scenario_rows():
            size_key = "x".join(str(int(v)) for v in size)
            if size_key not in mapping or density not in mapping[size_key]:
                raise ConfigError(f"No node-count mapping for {size_key}/{density}")
            sweep_items = list(self.spec.get("sweep", {}).items())
            sweep_values = [
                [(path, value) for value in values] for path, values in sweep_items
            ]
            sweep_combinations = list(itertools.product(*sweep_values)) if sweep_values else [()]
            for base_seed in self.spec["seeds"]:
                for sweep_values_for_case in sweep_combinations:
                    case = copy.deepcopy(base)
                    env, channel, execution = case["environment"], case["channel"], case["execution"]
                    env.update({
                        "deployment_size_m": list(size),
                        "node_count": int(mapping[size_key][density]),
                        "distribution": distribution,
                        "sink_position": [float(size[0]) / 2, float(size[1]) / 2, 0.0],
                    })
                    for section, values in self.spec.get("overrides", {}).items():
                        if section not in case:
                            raise ConfigError(f"Override references unknown section: {section}")
                        unknown = set(values).difference(SECTION_KEYS[section])
                        if unknown:
                            raise ConfigError(f"Override contains unknown {section} keys: {sorted(unknown)}")
                        case[section].update(copy.deepcopy(values))
                    for dotted_path, value in sweep_values_for_case:
                        section, key = dotted_path.split(".", 1)
                        if section not in case or key not in SECTION_KEYS[section]:
                            raise ConfigError(f"Sweep references unknown key: {dotted_path}")
                        case[section][key] = copy.deepcopy(value)
                    streams = seed_streams(int(base_seed))
                    case["metadata"] = {
                        "case_index": 0, "case_id": "", "experiment_set": self.spec["experiment_set"],
                        "density": density, "base_seed": int(base_seed),
                        "topology_seed": streams["topology"], "protocol_seed": streams["protocol"],
                        "optimizer_seed": streams["optimizer"],
                        "channel_seed": streams["channel"], "mobility_seed": streams["mobility"],
                        "config_hash": "", "generated_at": generated_at, "code_version": version,
                        "status": "pending",
                    }
                    case.setdefault("output", {"root_dir": "results", "save_round_metrics": True,
                                               "save_convergence": True})
                    validate_case(case)
                    case["metadata"]["case_id"] = readable_case_id(case)
                    case["metadata"]["config_hash"] = canonical_hash(case)
                    cases.append(case)
        for index, case in enumerate(cases, 1):
            case["metadata"]["case_index"] = index
        return cases

    def summary(self, cases: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
        return {
            "deployment_cases": len(self._scenario_rows()),
            "algorithms": len(self.spec["algorithms"]),
            "seeds": len(self.spec["seeds"]),
            "scenario_count": len(cases),
            "planned_runs": len(cases) * len(self.spec["algorithms"]),
            "rejected_cases": 0,
            "by_deployment": dict(Counter(
                "x".join(str(int(v)) for v in c["environment"]["deployment_size_m"]) for c in cases
            )),
            "output_dir": str(output_dir),
        }

    def write(self, output_dir: Path, clean: bool = False) -> list[dict[str, Any]]:
        cases = self.build_cases()
        cases_dir = output_dir / "cases"
        output_dir.mkdir(parents=True, exist_ok=True)
        cases_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            for path in cases_dir.glob("case_*.yaml"):
                path.unlink()
        rows = []
        for case in cases:
            index = case["metadata"]["case_index"]
            relative = Path("cases") / f"case_{index:06d}.yaml"
            write_yaml(output_dir / relative, case)
            env, meta, channel = case["environment"], case["metadata"], case["channel"]
            rows.append({
                "case_index": index, "case_id": meta["case_id"], "config_path": str(relative),
                "experiment_set": meta["experiment_set"],
                "deployment_x_m": env["deployment_size_m"][0],
                "deployment_y_m": env["deployment_size_m"][1],
                "deployment_z_m": env["deployment_size_m"][2], "density": meta["density"],
                "node_count": env["node_count"], "distribution": env["distribution"],
                "initial_energy_j": env["initial_energy_j"], "packet_size_bits": env["packet_size_bits"],
                "transmission_range_m": channel["transmission_range_m"],
                "sound_speed_model": channel["sound_speed_model"], "rounds": case["execution"]["rounds"],
                "base_seed": meta["base_seed"], "topology_seed": meta["topology_seed"],
                "protocol_seed": meta["protocol_seed"], "optimizer_seed": meta["optimizer_seed"],
                "channel_seed": meta["channel_seed"],
                "mobility_seed": meta["mobility_seed"], "config_hash": meta["config_hash"],
                "status": "pending",
            })
        if rows:
            with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output_dir / "manifest.json").write_text(
            json.dumps({"cases": rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return cases
