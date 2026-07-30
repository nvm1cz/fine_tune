from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

from .cases import SimulationCase


@dataclass
class RunMetrics:
    seed: int
    fnd_round: int | None
    hnd_round: int | None
    lnd_round: int | None
    first_5pct_round: int | None
    packets_received: int
    residual_energy: float
    alive_nodes: int
    residual_energy_by_round: List[float] = field(default_factory=list)
    dead_nodes_by_round: List[int] = field(default_factory=list)
    packets_received_by_round: List[int] = field(default_factory=list)
    delay_by_round: List[float] = field(default_factory=list)
    transmission_attempts: int = 0
    packets_attempted: int = 0
    retransmissions: int = 0
    dropped_packets: int = 0
    packets_generated: int = 0
    packets_delivered: int = 0
    packet_delivery_ratio: float | None = None
    average_ber: float | None = None
    average_per: float | None = None
    average_snr_db: float | None = None
    average_link_distance_m: float | None = None
    total_tx_energy_j: float = 0.0
    total_rx_energy_j: float = 0.0
    total_delay_s: float = 0.0
    average_e2e_delay_s: float | None = None
    mobility_displacement_m: float = 0.0
    pso_convergence: List[Dict[str, float]] = field(default_factory=list)


def summarize_runs(case: SimulationCase, runs: Sequence[RunMetrics]) -> Dict[str, object]:
    total_energy = case.initial_energy * case.node_count

    dead_nodes_mean = mean([case.node_count - r.alive_nodes for r in runs])
    dead_nodes_pct = (dead_nodes_mean / case.node_count) * 100

    fnd_values = [r.fnd_round for r in runs if r.fnd_round is not None]
    hnd_values = [r.hnd_round for r in runs if r.hnd_round is not None]
    lnd_values = [r.lnd_round for r in runs if r.lnd_round is not None]
    first5_values = [r.first_5pct_round for r in runs if r.first_5pct_round is not None]
    residual_energy_mean = mean([r.residual_energy for r in runs])
    residual_energy_pct = (residual_energy_mean / total_energy) * 100

    return {
        "case": asdict(case),
        "runs": len(runs),
        "fnd_mean": mean(fnd_values) if fnd_values else None,
        "hnd_mean": mean(hnd_values) if hnd_values else None,
        "lnd_mean": mean(lnd_values) if lnd_values else None,
        "first_5pct_mean": mean(first5_values) if first5_values else None,
        "residual_energy_pct": round(residual_energy_pct, 2),
        "dead_nodes_pct": round(dead_nodes_pct, 2),
        "lnd_std": pstdev(lnd_values) if len(lnd_values) > 1 else 0.0,
        "residual_energy_std": pstdev([r.residual_energy for r in runs]) if len(runs) > 1 else 0.0,
        "packet_delivery_ratio_mean": mean(
            [r.packet_delivery_ratio for r in runs if r.packet_delivery_ratio is not None]
        ) if any(r.packet_delivery_ratio is not None for r in runs) else None,
    }
