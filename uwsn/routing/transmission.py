from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..cases import SimulationCase
from ..models.channel import channel_quality
from ..models.delay import link_delay_s
from ..models.energy import (
    aggregation_energy_j,
    receive_energy_j,
    transmit_energy_j,
    transmit_energy_j_array,
)
from ..run_config import TunableParams
from .planning import RoutePlan


@dataclass
class TransmissionStats:
    attempts: int = 0
    retransmissions: int = 0
    dropped_packets: int = 0
    packets_generated: int = 0
    packets_delivered: int = 0
    total_tx_energy_j: float = 0.0
    total_rx_energy_j: float = 0.0
    total_attempt_delay_s: float = 0.0
    total_delivered_delay_s: float = 0.0
    quality_samples: int = 0
    snr_db_sum: float = 0.0
    ber_sum: float = 0.0
    per_sum: float = 0.0
    link_distance_m_sum: float = 0.0


def contention_multiplier(params: TunableParams, neighbor_count: np.ndarray | float | int) -> np.ndarray:
    factor = float(getattr(params, "contention_penalty_factor", 0.0))
    if factor <= 0:
        return np.ones_like(np.asarray(neighbor_count, dtype=float), dtype=float)

    reference = max(float(getattr(params, "contention_reference_degree", 10.0)), 1.0)
    max_multiplier = max(float(getattr(params, "contention_max_multiplier", 4.0)), 1.0)
    multiplier = 1.0 + factor * (np.asarray(neighbor_count, dtype=float) / reference)
    return np.minimum(max_multiplier, multiplier)


def send_energy_from_distance_cost(distance_cost: float, bits: int) -> float:
    return bits * distance_cost


def receive_energy(params: TunableParams, bits: int) -> float:
    return receive_energy_j(bits, params)


def aggregate_energy(params: TunableParams, bits: int) -> float:
    return aggregation_energy_j(bits, params)


def can_transmit_to_sink(params: TunableParams, dist_to_sink_m: float) -> bool:
    return dist_to_sink_m <= params.transmission_range_m


def routing_selection_costs(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    dist_to_sink: np.ndarray,
    current_ch: int,
    candidate_chs: np.ndarray,
) -> np.ndarray:
    """Equation P(i,j) from Algorithm 1; lower is better."""
    epsilon = float(params.balance_factor)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("balance_factor epsilon must be in [0, 1]")

    candidate_chs = np.asarray(candidate_chs, dtype=int)
    if candidate_chs.size == 0:
        return np.asarray([], dtype=float)

    residual_energy = np.maximum(energies[candidate_chs], 1e-12)
    energy_term = epsilon * (float(case.initial_energy) / residual_energy)

    d_i_sink_squared = max(float(dist_to_sink[current_ch]) ** 2, 1e-12)
    d_i_j_squared = np.square(distance_matrix[current_ch, candidate_chs])
    d_j_sink_squared = np.square(dist_to_sink[candidate_chs])
    distance_term = (1.0 - epsilon) * (
        (d_i_j_squared + d_j_sink_squared) / d_i_sink_squared
    )
    return energy_term + distance_term


def pick_next_hop(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    current_ch: int,
    cluster_heads: Sequence[int],
) -> int | None:
    current_layer = layers[current_ch]
    ch_array = np.asarray(cluster_heads, dtype=int)
    if ch_array.size == 0:
        return None

    forwarding_chs = ch_array[
        (ch_array != current_ch)
        & (energies[ch_array] > params.dead_energy_threshold_j)
        & (distance_matrix[current_ch, ch_array] <= params.transmission_range_m)
        & (layers[ch_array] < current_layer)
    ]
    if forwarding_chs.size == 0:
        return None

    scores = routing_selection_costs(
        case,
        params,
        distance_matrix,
        energies,
        dist_to_sink,
        current_ch,
        forwarding_chs,
    )
    return int(forwarding_chs[int(np.argmin(scores))])


def build_route_to_sink(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    source_ch: int,
    cluster_heads: Sequence[int],
) -> list[int | None]:
    """Apply P(i,j) at every CH-to-CH hop until a CH can reach the sink."""
    current_ch = int(source_ch)
    route: list[int | None] = []
    visited = {current_ch}

    for _ in range(len(cluster_heads) + 1):
        if can_transmit_to_sink(params, float(dist_to_sink[current_ch])):
            route.append(None)
            return route

        next_hop = pick_next_hop(
            case,
            params,
            distance_matrix,
            energies,
            layers,
            dist_to_sink,
            current_ch,
            cluster_heads,
        )
        if next_hop is None or next_hop in visited:
            return []

        route.append(next_hop)
        visited.add(next_hop)
        current_ch = next_hop

    return []


def execute_round_transmissions(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    tx_cost_matrix_per_bit: np.ndarray,
    tx_cost_to_sink_per_bit: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    assignments: Dict[int, List[int]],
    cluster_heads: Sequence[int],
    packets_received: int,
    dead_energy_threshold_j: float,
    neighbor_count: np.ndarray | None = None,
    routing_edges: List[tuple[int, int | None]] | None = None,
    delay_records: List[float] | None = None,
    rng: np.random.Generator | None = None,
    transmission_stats: TransmissionStats | None = None,
    route_plan: RoutePlan | None = None,
) -> int:
    stats = transmission_stats if transmission_stats is not None else TransmissionStats()
    if not bool(params.enable_packet_errors):
        return _execute_round_transmissions_legacy(
            case,
            params,
            distance_matrix,
            tx_cost_matrix_per_bit,
            tx_cost_to_sink_per_bit,
            energies,
            layers,
            dist_to_sink,
            assignments,
            cluster_heads,
            packets_received,
            dead_energy_threshold_j,
            neighbor_count=neighbor_count,
            routing_edges=routing_edges,
            delay_records=delay_records,
            route_plan=route_plan,
            stats=stats,
        )
    if rng is None:
        raise ValueError("rng is required when enable_packet_errors is true")
    if int(params.max_retries) < 0:
        raise ValueError("max_retries must be non-negative")
    return _execute_round_transmissions_with_channel(
        case,
        params,
        distance_matrix,
        tx_cost_matrix_per_bit,
        tx_cost_to_sink_per_bit,
        energies,
        layers,
        dist_to_sink,
        assignments,
        cluster_heads,
        packets_received,
        dead_energy_threshold_j,
        neighbor_count,
        routing_edges,
        delay_records,
        rng,
        stats,
        route_plan,
    )


def _attempt_link(
    *,
    params: TunableParams,
    packet_bits: int,
    distance_m: float,
    tx_energy_j: float,
    energies: np.ndarray,
    sender: int,
    receiver: int | None,
    receiver_energy_j: float,
    dead_energy_threshold_j: float,
    delay_records: List[float] | None,
    rng: np.random.Generator,
    stats: TransmissionStats,
) -> bool:
    quality = channel_quality(
        distance_m,
        params.acoustic_frequency_khz,
        packet_bits,
        params,
    )
    success_probability = quality["packet_success_probability"]
    max_attempts = 1 + (int(params.max_retries) if params.enable_retransmission else 0)

    for attempt_index in range(max_attempts):
        stats.attempts += 1
        if attempt_index > 0:
            stats.retransmissions += 1
        energies[sender] = max(0.0, energies[sender] - tx_energy_j)
        stats.total_tx_energy_j += float(tx_energy_j)
        if receiver is not None:
            rx_charge = (
                receiver_energy_j
                if params.charge_rx_energy_on_corrupted_packet
                else receiver_energy_j * success_probability
            )
            energies[receiver] = max(0.0, energies[receiver] - rx_charge)
            stats.total_rx_energy_j += float(rx_charge)
        attempt_delay = link_delay_s(
            packet_bits,
            distance_m,
            params,
            channel_capacity_bps=quality["channel_capacity_bps"],
        ).total_s
        stats.total_attempt_delay_s += attempt_delay
        stats.quality_samples += 1
        stats.snr_db_sum += float(quality["snr_db"])
        stats.ber_sum += float(quality["ber"])
        stats.per_sum += float(quality["per"])
        stats.link_distance_m_sum += float(distance_m)
        if delay_records is not None:
            delay_records.append(attempt_delay)

        has_energy = energies[sender] > dead_energy_threshold_j
        if receiver is not None:
            has_energy = has_energy and energies[receiver] > dead_energy_threshold_j
        if has_energy and float(rng.random()) < success_probability:
            return True
        if not has_energy:
            break

    stats.dropped_packets += 1
    return False


def _execute_round_transmissions_with_channel(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    tx_cost_matrix_per_bit: np.ndarray,
    tx_cost_to_sink_per_bit: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    assignments: Dict[int, List[int]],
    cluster_heads: Sequence[int],
    packets_received: int,
    dead_energy_threshold_j: float,
    neighbor_count: np.ndarray | None,
    routing_edges: List[tuple[int, int | None]] | None,
    delay_records: List[float] | None,
    rng: np.random.Generator,
    stats: TransmissionStats,
    route_plan: RoutePlan | None,
) -> int:
    if neighbor_count is None:
        neighbor_count = np.zeros_like(energies, dtype=float)
    contention = contention_multiplier(params, neighbor_count)

    alive_nodes = np.flatnonzero(energies > dead_energy_threshold_j)
    stats.packets_generated += int(alive_nodes.size)
    if alive_nodes.size:
        energies[alive_nodes] = np.maximum(
            0.0,
            energies[alive_nodes]
            - aggregate_energy(params, params.broadcast_packet_size_bits) * contention[alive_nodes],
        )

    packet_bits = case.packet_size_bits
    aggregate_cost = aggregate_energy(params, packet_bits)

    for ch_idx, members in assignments.items():
        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        members_array = np.asarray(members, dtype=int)
        transmitting_members = members_array[
            (members_array != ch_idx) & (energies[members_array] > dead_energy_threshold_j)
        ]
        successfully_received = 0
        successful_member_delays: list[float] = []
        for member in transmitting_members:
            tx = (
                transmit_energy_j(
                    float(distance_matrix[member, ch_idx]), packet_bits, params
                )
                * contention[member]
            )
            rx = receive_energy_j(packet_bits, params) * contention[ch_idx]
            member_delay_start = stats.total_attempt_delay_s
            if _attempt_link(
                params=params,
                packet_bits=packet_bits,
                distance_m=float(distance_matrix[member, ch_idx]),
                tx_energy_j=tx,
                energies=energies,
                sender=int(member),
                receiver=int(ch_idx),
                receiver_energy_j=rx,
                dead_energy_threshold_j=dead_energy_threshold_j,
                delay_records=delay_records,
                rng=rng,
                stats=stats,
            ):
                successfully_received += 1
                successful_member_delays.append(
                    stats.total_attempt_delay_s - member_delay_start
                )

        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        # The CH aggregates its own packet plus only packets actually received.
        data_packets = 1 + successfully_received
        energies[ch_idx] = max(0.0, energies[ch_idx] - data_packets * aggregate_cost)
        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        route = (
            list(route_plan.route_by_ch.get(int(ch_idx), ()))
            if route_plan is not None
            else build_route_to_sink(
                case,
                params,
                distance_matrix,
                energies,
                layers,
                dist_to_sink,
                ch_idx,
                cluster_heads,
            )
        )
        if not route:
            continue

        current = int(ch_idx)
        route_delay_start = stats.total_attempt_delay_s
        delivered = False
        for hop in route:
            if routing_edges is not None:
                routing_edges.append((current, None if hop is None else int(hop)))

            if hop is None:
                delivered = _attempt_link(
                    params=params,
                    packet_bits=packet_bits,
                    distance_m=float(dist_to_sink[current]),
                    tx_energy_j=transmit_energy_j(
                        float(dist_to_sink[current]), packet_bits, params
                    ) * contention[current],
                    energies=energies,
                    sender=current,
                    receiver=None,
                    receiver_energy_j=0.0,
                    dead_energy_threshold_j=dead_energy_threshold_j,
                    delay_records=delay_records,
                    rng=rng,
                    stats=stats,
                )
                break

            receiver = int(hop)
            success = _attempt_link(
                params=params,
                packet_bits=packet_bits,
                distance_m=float(distance_matrix[current, receiver]),
                tx_energy_j=transmit_energy_j(
                    float(distance_matrix[current, receiver]), packet_bits, params
                ) * contention[current],
                energies=energies,
                sender=current,
                receiver=receiver,
                receiver_energy_j=receive_energy_j(packet_bits, params) * contention[receiver],
                dead_energy_threshold_j=dead_energy_threshold_j,
                delay_records=delay_records,
                rng=rng,
                stats=stats,
            )
            if not success:
                break
            current = receiver

        if delivered:
            packets_received += 1
            stats.packets_delivered += data_packets
            route_delay = stats.total_attempt_delay_s - route_delay_start
            # CH's own source packet traverses the CH route. Each successfully
            # received member packet traverses member->CH plus the same route.
            stats.total_delivered_delay_s += (
                route_delay * data_packets + sum(successful_member_delays)
            )

    return packets_received


def _execute_round_transmissions_legacy(
    case: SimulationCase,
    params: TunableParams,
    distance_matrix: np.ndarray,
    tx_cost_matrix_per_bit: np.ndarray,
    tx_cost_to_sink_per_bit: np.ndarray,
    energies: np.ndarray,
    layers: np.ndarray,
    dist_to_sink: np.ndarray,
    assignments: Dict[int, List[int]],
    cluster_heads: Sequence[int],
    packets_received: int,
    dead_energy_threshold_j: float,
    neighbor_count: np.ndarray | None = None,
    routing_edges: List[tuple[int, int | None]] | None = None,
    delay_records: List[float] | None = None,
    route_plan: RoutePlan | None = None,
    stats: TransmissionStats | None = None,
) -> int:
    stats = stats if stats is not None else TransmissionStats()
    if neighbor_count is None:
        neighbor_count = np.zeros_like(energies, dtype=float)
    contention = contention_multiplier(params, neighbor_count)

    broadcast_bits = params.broadcast_packet_size_bits
    alive_nodes = np.flatnonzero(energies > dead_energy_threshold_j)
    stats.packets_generated += int(alive_nodes.size)
    if alive_nodes.size:
        energies[alive_nodes] = np.maximum(
            0.0,
            energies[alive_nodes] - aggregate_energy(params, broadcast_bits) * contention[alive_nodes],
        )

    packet_bits = case.packet_size_bits
    rx_cost = receive_energy_j(packet_bits, params)
    aggregate_cost_per_packet = aggregation_energy_j(packet_bits, params)

    for ch_idx, members in assignments.items():
        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        members_array = np.asarray(members, dtype=int)
        if members_array.size:
            transmitting_members = members_array[
                (members_array != ch_idx) & (energies[members_array] > dead_energy_threshold_j)
            ]
        else:
            transmitting_members = members_array

        if transmitting_members.size:
            tx_costs = (
                transmit_energy_j_array(
                    distance_matrix[transmitting_members, ch_idx], packet_bits, params
                )
                * contention[transmitting_members]
            )
            energies[transmitting_members] = np.maximum(
                0.0,
                energies[transmitting_members] - tx_costs,
            )
            stats.attempts += int(transmitting_members.size)
            stats.total_tx_energy_j += float(np.sum(tx_costs))
            energies[ch_idx] = max(
                0.0,
                energies[ch_idx] - rx_cost * transmitting_members.size * contention[ch_idx],
            )
            stats.total_rx_energy_j += float(
                rx_cost * transmitting_members.size * contention[ch_idx]
            )
            member_delays = [
                link_delay_s(
                    packet_bits, float(distance_matrix[int(member), ch_idx]), params
                ).total_s
                for member in transmitting_members
            ]
            stats.total_attempt_delay_s += float(sum(member_delays))
            if delay_records is not None:
                delay_records.extend(member_delays)
        else:
            member_delays = []

        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        if members_array.size:
            data_packets = max(1, int(np.count_nonzero(energies[members_array] > dead_energy_threshold_j)))
        else:
            data_packets = 1

        aggregated_bits = packet_bits
        energies[ch_idx] = max(0.0, energies[ch_idx] - data_packets * aggregate_cost_per_packet)
        if energies[ch_idx] <= dead_energy_threshold_j:
            continue

        route = (
            list(route_plan.route_by_ch.get(int(ch_idx), ()))
            if route_plan is not None
            else build_route_to_sink(
                case,
                params,
                distance_matrix,
                energies,
                layers,
                dist_to_sink,
                ch_idx,
                cluster_heads,
            )
        )
        if not route:
            continue

        current = int(ch_idx)
        route_delay_start = stats.total_attempt_delay_s
        delivered = False
        for hop in route:
            if routing_edges is not None:
                routing_edges.append((current, None if hop is None else int(hop)))

            if hop is None:
                tx = transmit_energy_j(
                    float(dist_to_sink[current]), aggregated_bits, params
                ) * contention[current]
                delay_value = link_delay_s(
                    aggregated_bits, float(dist_to_sink[current]), params
                ).total_s
                stats.total_attempt_delay_s += delay_value
                if delay_records is not None:
                    delay_records.append(delay_value)
                energies[current] = max(0.0, energies[current] - tx)
                stats.attempts += 1
                stats.total_tx_energy_j += float(tx)
                delivered = energies[current] > dead_energy_threshold_j
                break

            hop = int(hop)
            tx = transmit_energy_j(
                float(distance_matrix[current, hop]), aggregated_bits, params
            ) * contention[current]
            rx = receive_energy_j(aggregated_bits, params) * contention[hop]
            delay_value = link_delay_s(
                aggregated_bits, float(distance_matrix[current, hop]), params
            ).total_s
            stats.total_attempt_delay_s += delay_value
            if delay_records is not None:
                delay_records.append(delay_value)
            energies[current] = max(0.0, energies[current] - tx)
            energies[hop] = max(0.0, energies[hop] - rx)
            stats.attempts += 1
            stats.total_tx_energy_j += float(tx)
            stats.total_rx_energy_j += float(rx)
            if energies[current] <= dead_energy_threshold_j or energies[hop] <= dead_energy_threshold_j:
                break
            current = hop

        if delivered:
            packets_received += 1
            stats.packets_delivered += data_packets
            route_delay = stats.total_attempt_delay_s - route_delay_start
            delivered_member_count = max(0, data_packets - 1)
            stats.total_delivered_delay_s += (
                route_delay * data_packets
                + sum(member_delays[:delivered_member_count])
            )

    return packets_received
