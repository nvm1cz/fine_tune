from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Iterator
from unittest.mock import patch

import numpy as np

from .routing import transmission


@dataclass
class RoundEnergyDiagnostic:
    member_tx_energy: float = 0.0
    ch_rx_energy: float = 0.0
    aggregation_energy: float = 0.0
    routing_tx_energy: float = 0.0
    routing_rx_energy: float = 0.0
    retransmission_energy: float = 0.0
    control_energy: float = 0.0
    total_round_energy: float = 0.0
    packets_generated: int = 0
    packets_delivered: int = 0
    delivered_delay_s: float = 0.0
    residual_before_j: np.ndarray | None = None
    residual_after_j: np.ndarray | None = None

    def row(self) -> dict[str, float | int]:
        return {
            key: value for key, value in asdict(self).items()
            if key not in {"residual_before_j", "residual_after_j"}
        }


@contextmanager
def capture_round_energy(simulator) -> Iterator[list[RoundEnergyDiagnostic]]:
    """Observe actual energy debits without changing decisions or RNG draws."""
    original_execute = transmission.execute_round_transmissions
    original_attempt = transmission._attempt_link
    captured: list[RoundEnergyDiagnostic] = []

    def execute_wrapper(*args, **kwargs):
        energies, params = args[5], args[1]
        cluster_heads = {int(value) for value in args[9]}
        neighbor_count = kwargs.get("neighbor_count")
        stats = kwargs["transmission_stats"]
        before_energy = np.asarray(energies, dtype=float).copy()
        generated_before = int(stats.packets_generated)
        delivered_before = int(stats.packets_delivered)
        delay_before = float(stats.total_delivered_delay_s)
        result = RoundEnergyDiagnostic()

        def attempt_wrapper(**call):
            sender, receiver = int(call["sender"]), call["receiver"]
            energy_array = call["energies"]
            sender_before = float(energy_array[sender])
            receiver_before = float(energy_array[int(receiver)]) if receiver is not None else 0.0
            attempts_before = int(call["stats"].attempts)
            value = original_attempt(**call)
            attempt_count = int(call["stats"].attempts) - attempts_before
            sender_used = sender_before - float(energy_array[sender])
            receiver_used = receiver_before - float(energy_array[int(receiver)]) if receiver is not None else 0.0
            first_tx = min(sender_used, float(call["tx_energy_j"]))
            first_rx = min(receiver_used, float(call["receiver_energy_j"]))
            if attempt_count > 1:
                result.retransmission_energy += max(0.0, sender_used - first_tx)
                result.retransmission_energy += max(0.0, receiver_used - first_rx)
            if sender in cluster_heads:
                result.routing_tx_energy += first_tx
                result.routing_rx_energy += first_rx
            else:
                result.member_tx_energy += first_tx
                result.ch_rx_energy += first_rx
            return value

        with patch.object(transmission, "_attempt_link", attempt_wrapper):
            returned = original_execute(*args, **kwargs)

        contention = transmission.contention_multiplier(params, neighbor_count)
        alive = np.flatnonzero(before_energy > float(args[11]))
        if alive.size:
            cost = transmission.aggregate_energy(params, params.broadcast_packet_size_bits) * contention[alive]
            result.control_energy = float(np.sum(np.minimum(before_energy[alive], cost)))
        result.total_round_energy = float(np.sum(before_energy) - np.sum(energies))
        known = (result.member_tx_energy + result.ch_rx_energy + result.routing_tx_energy
                 + result.routing_rx_energy + result.retransmission_energy + result.control_energy)
        result.aggregation_energy = max(0.0, result.total_round_energy - known)
        result.packets_generated = int(stats.packets_generated) - generated_before
        result.packets_delivered = int(stats.packets_delivered) - delivered_before
        result.delivered_delay_s = float(stats.total_delivered_delay_s) - delay_before
        result.residual_before_j = before_energy
        result.residual_after_j = np.asarray(energies, dtype=float).copy()
        captured.append(result)
        return returned

    with patch("uwsn.simulator.execute_round_transmissions", execute_wrapper):
        yield captured


def selected_ch_json(cluster_heads: list[int]) -> str:
    return json.dumps(sorted(int(value) for value in cluster_heads), separators=(",", ":"))
