from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping, Sequence
import warnings

import numpy as np

from ..models.acoustic import transmission_loss_db
from ..models.channel import channel_quality
from ..run_config import TunableParams


@dataclass(frozen=True)
class AssignmentResult:
    assignments: Mapping[int, tuple[int, ...]]
    member_to_ch: Mapping[int, int]
    mode: str
    fallback_used: bool
    is_feasible: bool
    invalid_reason: str | None = None


@dataclass(frozen=True)
class RoutePlan:
    parent_by_ch: Mapping[int, int | None]
    route_by_ch: Mapping[int, tuple[int | None, ...]]
    forwarding_load_by_ch: Mapping[int, int]
    links: tuple[tuple[int, int | None], ...]
    is_feasible: bool
    invalid_reason: str | None
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    plan_id: str = ""

    def __post_init__(self) -> None:
        if not self.plan_id:
            payload = {
                str(int(ch)): [None if hop is None else int(hop) for hop in route]
                for ch, route in sorted(self.route_by_ch.items())
            }
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            object.__setattr__(self, "plan_id", hashlib.sha256(encoded).hexdigest()[:16])

    @classmethod
    def from_routes(
        cls,
        routes: Mapping[int, Sequence[int | None]],
        *,
        is_feasible: bool = True,
        invalid_reason: str | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ) -> "RoutePlan":
        normalized = {
            int(ch): tuple(None if hop is None else int(hop) for hop in route)
            for ch, route in routes.items()
        }
        parents = {
            ch: (route[0] if route else None)
            for ch, route in normalized.items()
        }
        load: dict[int, int] = {ch: 0 for ch in normalized}
        links: list[tuple[int, int | None]] = []
        for source, route in normalized.items():
            current = source
            for hop in route:
                links.append((current, hop))
                if current != source:
                    load[current] = load.get(current, 0) + 1
                current = -1 if hop is None else int(hop)
        return cls(
            parent_by_ch=parents,
            route_by_ch=normalized,
            forwarding_load_by_ch=load,
            links=tuple(links),
            is_feasible=is_feasible,
            invalid_reason=invalid_reason,
            diagnostics={} if diagnostics is None else dict(diagnostics),
        )


def assign_members_strongest_rssi(
    positions: np.ndarray,
    energies: np.ndarray,
    selected_cluster_heads: Sequence[int],
    params: TunableParams,
) -> AssignmentResult:
    """Assign every alive node using minimum shared transmission loss.

    With a common source level, minimum transmission loss is exactly equivalent to
    maximum received signal level. Distance and node ID provide deterministic ties.
    """
    positions_array = np.asarray(positions, dtype=float)
    residual = np.asarray(energies, dtype=float)
    selected = tuple(sorted(dict.fromkeys(int(ch) for ch in selected_cluster_heads)))
    alive = tuple(
        int(node)
        for node in np.flatnonzero(residual > params.dead_energy_threshold_j)
    )
    alive_chs = tuple(ch for ch in selected if ch in alive)
    if not alive_chs:
        return AssignmentResult({}, {}, "strongest_rssi", False, False, "unassigned_alive_node")

    assignments: dict[int, list[int]] = {ch: [] for ch in alive_chs}
    member_to_ch: dict[int, int] = {}
    fallback_used = False
    for node in alive:
        feasible = [
            ch for ch in alive_chs
            if float(np.linalg.norm(positions_array[node] - positions_array[ch]))
            <= params.transmission_range_m
        ]
        if not feasible:
            return AssignmentResult(
                {ch: tuple(members) for ch, members in assignments.items()},
                member_to_ch,
                "strongest_rssi",
                fallback_used,
                False,
                "unassigned_alive_node",
            )
        ranked: list[tuple[float, float, int]] = []
        try:
            for ch in feasible:
                distance = float(np.linalg.norm(positions_array[node] - positions_array[ch]))
                loss = transmission_loss_db(
                    distance,
                    params.acoustic_frequency_khz,
                    params.spreading_factor,
                    params.acoustic_reference_attenuation_linear,
                    params.acoustic_reference_distance_m,
                    params.acoustic_model_version,
                    params.transmission_anomaly_db,
                )
                ranked.append((float(loss), distance, int(ch)))
            chosen = min(ranked)[2]
        except (ValueError, FloatingPointError):
            fallback_used = True
            warnings.warn(
                "Shared acoustic link quality unavailable; falling back to nearest feasible CH",
                RuntimeWarning,
                stacklevel=2,
            )
            chosen = min(
                feasible,
                key=lambda ch: (
                    float(np.linalg.norm(positions_array[node] - positions_array[ch])),
                    int(ch),
                ),
            )
        assignments[chosen].append(node)
        member_to_ch[node] = chosen

    mode = "nearest_feasible_fallback" if fallback_used else "strongest_rssi"
    return AssignmentResult(
        {ch: tuple(members) for ch, members in assignments.items()},
        member_to_ch,
        mode,
        fallback_used,
        True,
        None,
    )


def validate_route_plan(
    plan: RoutePlan,
    selected_cluster_heads: Sequence[int],
    positions: np.ndarray,
    sink_position: np.ndarray,
    energies: np.ndarray,
    params: TunableParams,
) -> tuple[bool, str | None]:
    selected = set(int(ch) for ch in selected_cluster_heads)
    positions_array = np.asarray(positions, dtype=float)
    sink = np.asarray(sink_position, dtype=float)
    residual = np.asarray(energies, dtype=float)
    max_hops = int(getattr(params, "routing_max_hops", max(1, len(selected) + 1)))
    if not plan.is_feasible:
        return False, plan.invalid_reason or "broken_route"
    if set(plan.route_by_ch) != selected:
        return False, "dangling_parent"
    for source in selected:
        if residual[source] <= params.dead_energy_threshold_j:
            return False, "dead_cluster_head"
        route = plan.route_by_ch.get(source, ())
        if not route or len(route) > max_hops:
            return False, "broken_route"
        current = source
        visited = {source}
        reached_sink = False
        for hop in route:
            if hop is None:
                target = sink
                reached_sink = True
            else:
                hop = int(hop)
                if hop not in selected:
                    return False, "dangling_parent"
                if hop in visited:
                    return False, "broken_route"
                if residual[hop] <= params.dead_energy_threshold_j:
                    return False, "dead_relay"
                target = positions_array[hop]
            distance = float(np.linalg.norm(positions_array[current] - target))
            if distance > params.transmission_range_m:
                return False, "mobility_topology_change"
            if params.enable_packet_errors:
                try:
                    quality = channel_quality(
                        distance,
                        params.acoustic_frequency_khz,
                        1,
                        params,
                    )
                except (ValueError, FloatingPointError):
                    return False, "broken_route"
                if (
                    params.minimum_snr_db is not None
                    and quality["snr_db"] < params.minimum_snr_db
                ):
                    return False, "link_quality_below_threshold"
                if (
                    params.minimum_success_probability is not None
                    and quality["packet_success_probability"]
                    < params.minimum_success_probability
                ):
                    return False, "link_quality_below_threshold"
            if hop is None:
                break
            visited.add(hop)
            current = hop
        if not reached_sink:
            return False, "broken_route"
    return True, None


def validate_assignments(
    assignments: Mapping[int, Sequence[int]],
    selected_cluster_heads: Sequence[int],
    positions: np.ndarray,
    energies: np.ndarray,
    params: TunableParams,
) -> tuple[bool, str | None]:
    selected = set(int(ch) for ch in selected_cluster_heads)
    residual = np.asarray(energies, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    alive = set(
        int(node)
        for node in np.flatnonzero(residual > params.dead_energy_threshold_j)
    )
    assigned: list[int] = []
    for ch, members in assignments.items():
        ch = int(ch)
        if ch not in selected or ch not in alive:
            return False, "dead_cluster_head"
        for member in members:
            member = int(member)
            if member not in alive:
                continue
            if float(np.linalg.norm(positions_array[member] - positions_array[ch])) > (
                params.transmission_range_m
            ):
                return False, "invalid_member_assignment"
            try:
                transmission_loss_db(
                    float(np.linalg.norm(positions_array[member] - positions_array[ch])),
                    params.acoustic_frequency_khz,
                    params.spreading_factor,
                    params.acoustic_reference_attenuation_linear,
                    params.acoustic_reference_distance_m,
                    params.acoustic_model_version,
                    params.transmission_anomaly_db,
                )
            except (ValueError, FloatingPointError):
                return False, "invalid_member_assignment"
            assigned.append(member)
    if len(assigned) != len(set(assigned)) or set(assigned) != alive:
        return False, "invalid_member_assignment"
    return True, None
