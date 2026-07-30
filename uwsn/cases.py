from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class SimulationCase:
    name: str
    initial_energy: float
    packet_size_bits: int
    node_count: int = 100
    rounds: int = 1000


def build_case_presets() -> Dict[str, SimulationCase]:
    return {
        "case1": SimulationCase("case1", initial_energy=1.0, packet_size_bits=6400),
        "case2": SimulationCase("case2", initial_energy=0.5, packet_size_bits=6400),
        "case3": SimulationCase("case3", initial_energy=0.5, packet_size_bits=4000),
        "case4": SimulationCase("case4", initial_energy=1.0, packet_size_bits=4000),
    }
