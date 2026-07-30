from __future__ import annotations

from dataclasses import dataclass

from ..run_config import TunableParams


@dataclass(frozen=True)
class LinkDelay:
    transmission_s: float
    reception_s: float
    propagation_s: float
    byte_alignment_s: float
    holding_s: float
    modem_transition_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (
            self.transmission_s
            + self.reception_s
            + self.propagation_s
            + self.byte_alignment_s
            + self.holding_s
            + self.modem_transition_s
        )


def transmission_delay_s(packet_size_bits: int, bit_rate_bps: float) -> float:
    if bit_rate_bps <= 0.0:
        raise ValueError("bit_rate_bps must be positive")
    return float(packet_size_bits) / float(bit_rate_bps)


def propagation_delay_s(distance_m: float, sound_speed_mps: float) -> float:
    if sound_speed_mps <= 0.0:
        raise ValueError("sound_speed_mps must be positive")
    return float(distance_m) / float(sound_speed_mps)


def link_delay_s(
    packet_size_bits: int,
    distance_m: float,
    params: TunableParams,
    *,
    byte_alignment_s: float | None = None,
    holding_s: float | None = None,
    channel_capacity_bps: float | None = None,
) -> LinkDelay:
    if params.bitrate_model == "fixed":
        bitrate = float(params.node_bit_rate_bps)
    elif params.bitrate_model == "shannon_limited":
        if channel_capacity_bps is None or channel_capacity_bps <= 0.0:
            raise ValueError(
                "positive channel_capacity_bps is required in shannon_limited mode"
            )
        efficiency = float(params.capacity_efficiency)
        if not 0.0 < efficiency <= 1.0:
            raise ValueError("capacity_efficiency must be in (0, 1]")
        bitrate = min(
            float(params.modem_max_bitrate_bps),
            efficiency * float(channel_capacity_bps),
        )
    else:
        raise ValueError("bitrate_model must be fixed or shannon_limited")
    tx_rx = transmission_delay_s(packet_size_bits, bitrate)
    return LinkDelay(
        transmission_s=tx_rx,
        reception_s=tx_rx,
        propagation_s=propagation_delay_s(distance_m, params.sound_speed_water_mps),
        byte_alignment_s=(
            float(params.byte_alignment_delay_s)
            if byte_alignment_s is None
            else float(byte_alignment_s)
        ),
        holding_s=float(params.holding_time_s) if holding_s is None else float(holding_s),
        modem_transition_s=float(params.modem_transition_delay_s),
    )


def eegnbr_forwarding_delay_without_queue_s(
    packet_size_bits: int,
    distance_m: float,
    params: TunableParams,
) -> float:
    """Equation t1/t2 without byte-alignment delay: T_timer + 2tr_i + pr_ij."""
    tr_i = transmission_delay_s(packet_size_bits, params.node_bit_rate_bps)
    pr_ij = propagation_delay_s(distance_m, params.sound_speed_water_mps)
    return float(params.query_timer_s) + 2.0 * tr_i + pr_ij
