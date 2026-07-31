from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Mapping, Protocol

import numpy as np

from .models.channel import (
    StaticChannelEnvironment,
    build_static_channel_environment,
    channel_quality,
    channel_quality_with_static_environment,
)
from .models.delay import link_delay_s
from .models.energy import aggregation_energy_j, receive_energy_j, transmit_energy_j
from .routing.planning import RoutePlan
from .run_config import TunableParams
from .routing.transmission import contention_multiplier


LinkKey = tuple[int, int | None]


@dataclass(frozen=True)
class CandidateSolution:
    selected_cluster_heads: tuple[int, ...]
    assignments: Mapping[int, tuple[int, ...]]
    routes: Mapping[int, tuple[int | None, ...]]
    route_plan: RoutePlan | None = None

    def __post_init__(self) -> None:
        if self.route_plan is None:
            object.__setattr__(self, "route_plan", RoutePlan.from_routes(self.routes))


@dataclass(frozen=True)
class CandidateEvaluation:
    solution: CandidateSolution
    objective_result: "ObjectiveResult"
    is_feasible: bool
    invalid_reason: str | None


@dataclass(frozen=True)
class EvaluationContext:
    node_positions: np.ndarray
    sink_position: np.ndarray
    residual_energy_j: np.ndarray
    packet_bits: int
    frequency_khz: float
    bandwidth_hz: float
    max_retries: int
    lambda_energy: float
    lambda_delay: float
    energy_reference_j: float
    delay_reference_s: float
    minimum_route_success_probability: float
    infeasible_penalty: float
    normalization_epsilon: float
    params: TunableParams
    energy_epsilon_j: float = 1.0e-12
    delay_epsilon_s: float = 1.0e-12
    objective_epsilon: float = 1.0e-12


@dataclass(frozen=True)
class LinkExpectedCost:
    source_id: int
    destination_id: int | None
    distance_m: float
    packet_success_probability: float
    limited_success_probability: float
    expected_attempts: float
    transmit_energy_j: float
    receive_energy_j: float
    aggregation_energy_j: float
    expected_energy_j: float
    attempt_delay_s: float
    expected_delay_s: float
    snr_db: float
    ber: float
    per: float
    transmission_loss_db: float


@dataclass(frozen=True)
class BaseLinkExpectedCost:
    """Candidate-independent per-packet/per-attempt cost for one round link."""

    source_id: int
    destination_id: int | None
    distance_m: float
    packet_success_probability: float
    limited_success_probability: float
    expected_attempts: float
    transmit_energy_j: float
    receive_energy_j: float
    attempt_delay_s: float
    expected_delay_s: float
    snr_db: float
    ber: float
    per: float
    transmission_loss_db: float


@dataclass
class ObjectiveCacheStats:
    logical_objective_evaluations: int = 0
    physical_objective_computations: int = 0
    static_environment_builds: int = 0
    link_cache_hits: int = 0
    link_cache_misses: int = 0
    shadow_checks: int = 0


class RoundObjectiveCache:
    """Per-optimizer-refresh deterministic cache; never stores candidate totals."""

    def __init__(self) -> None:
        self.disabled = os.getenv("UWSN_DISABLE_LINK_CACHE") == "1"
        self.round_link_disabled = (
            os.getenv("UWSN_DISABLE_ROUND_LINK_CACHE") == "1"
        )
        self.verify = os.getenv("UWSN_VERIFY_LINK_CACHE") == "1"
        self.stats = ObjectiveCacheStats()
        self._static_key: tuple[object, ...] | None = None
        self._static_environment: StaticChannelEnvironment | None = None
        self._links: dict[tuple[int, int | None], BaseLinkExpectedCost] = {}
        self._bound_key: tuple[object, ...] | None = None

    def prepare(
        self,
        context: EvaluationContext,
        params: TunableParams,
        positions: np.ndarray,
        sink: np.ndarray,
    ) -> None:
        """Invalidate round data when topology or any model parameter changes."""
        key = (
            id(context.node_positions),
            id(context.sink_position),
            positions.shape,
            sink.shape,
            int(context.packet_bits),
            float(context.frequency_khz),
            float(context.bandwidth_hz),
            int(context.max_retries),
            tuple(vars(params).items()),
        )
        if self._bound_key is not None and self._bound_key != key:
            self._links.clear()
            self._static_key = None
            self._static_environment = None
        self._bound_key = key

    def static_environment(
        self, context: EvaluationContext, params: TunableParams
    ) -> StaticChannelEnvironment:
        key = (
            float(context.frequency_khz),
            float(params.channel_bandwidth_hz),
            float(params.shipping_noise_factor),
            float(params.wind_speed_mps),
            params.noise_model,
            params.noise_formula_mode,
        )
        if self._static_key != key:
            self._static_environment = build_static_channel_environment(
                context.frequency_khz, params
            )
            self._static_key = key
            self.stats.static_environment_builds += 1
        assert self._static_environment is not None
        return self._static_environment

    def get_link(self, key: LinkKey) -> BaseLinkExpectedCost | None:
        if self.round_link_disabled:
            self.stats.link_cache_misses += 1
            return None
        value = self._links.get(key)
        if value is None:
            self.stats.link_cache_misses += 1
        else:
            self.stats.link_cache_hits += 1
        return value

    def put_link(self, key: LinkKey, value: BaseLinkExpectedCost) -> None:
        if not self.round_link_disabled:
            self._links[key] = value

    def snapshot(self) -> dict[str, int]:
        return {
            name: int(value)
            for name, value in vars(self.stats).items()
        }


@dataclass(frozen=True)
class ObjectiveResult:
    objective_value: float
    raw_energy_j: float
    raw_average_delay_s: float
    normalized_energy: float
    normalized_delay: float
    feasible: bool
    infeasible_reasons: tuple[str, ...]
    route_success_probabilities: dict[int, float]
    expected_attempts_by_link: dict[LinkKey, float]
    energy_by_link: dict[LinkKey, float]
    delay_by_link: dict[LinkKey, float]
    diagnostics: dict[str, object] = field(default_factory=dict)
    objective_name: str = "normalized_multiplicative_energy_delay_objective"
    objective_direction: str = "minimize"
    lambda_weight: float = 0.5
    available_energy_j: float = 0.0
    reference_delay_s: float = 0.0
    delivered_packet_count: int = 0
    required_packet_count: int = 0
    log_objective: float = float("inf")

    @property
    def objective_J(self) -> float:
        return self.objective_value

    @property
    def energy_term_e(self) -> float:
        return self.normalized_energy

    @property
    def delay_term_d(self) -> float:
        return self.normalized_delay

    @property
    def energy_consumed(self) -> float:
        return self.raw_energy_j

    @property
    def available_energy(self) -> float:
        return self.available_energy_j

    @property
    def average_end_to_end_delay(self) -> float:
        return self.raw_average_delay_s

    @property
    def reference_delay(self) -> float:
        return self.reference_delay_s

    @property
    def is_feasible(self) -> bool:
        return self.feasible

    @property
    def invalid_reason(self) -> str | None:
        return "; ".join(self.infeasible_reasons) if self.infeasible_reasons else None


class ObjectiveEvaluator(Protocol):
    def evaluate(
        self,
        candidate: CandidateSolution,
        context: EvaluationContext,
    ) -> ObjectiveResult:
        ...


def expected_attempts_limited(
    packet_success_probability: float,
    max_retries: int,
    epsilon: float = 1.0e-12,
) -> tuple[float, float]:
    p = float(packet_success_probability)
    retries = int(max_retries)
    if not np.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("packet_success_probability must be finite and in [0, 1]")
    if retries < 0:
        raise ValueError("max_retries must be non-negative")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    maximum_attempts = 1 + retries
    q = 1.0 - p
    limited_success = float(np.clip(1.0 - q**maximum_attempts, 0.0, 1.0))
    if p > epsilon:
        attempts = limited_success / p
    else:
        attempts = float(maximum_attempts)
    attempts = float(np.clip(attempts, 1.0, float(maximum_attempts)))
    return attempts, limited_success


class EnergyDelayObjective:
    """Side-effect-free normalized multiplicative energy-delay objective."""

    def __init__(self, cache: RoundObjectiveCache | None = None) -> None:
        self.cache = cache or RoundObjectiveCache()

    def evaluate(
        self,
        candidate: CandidateSolution,
        context: EvaluationContext,
    ) -> ObjectiveResult:
        self.cache.stats.logical_objective_evaluations += 1
        self.cache.stats.physical_objective_computations += 1
        self._validate_context(context)
        positions = np.asarray(context.node_positions, dtype=float)
        sink = np.asarray(context.sink_position, dtype=float)
        residual = np.asarray(context.residual_energy_j, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("node_positions must have shape (N, 3)")
        if sink.shape != (3,):
            raise ValueError("sink_position must have shape (3,)")
        if residual.shape != (positions.shape[0],):
            raise ValueError("residual_energy_j must have shape (N,)")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(residual)):
            raise ValueError("positions and residual energy must be finite")

        params = replace(
            context.params,
            acoustic_frequency_khz=float(context.frequency_khz),
            channel_bandwidth_hz=float(context.bandwidth_hz),
            max_retries=int(context.max_retries),
        )
        self.cache.prepare(context, params, positions, sink)
        reasons: list[str] = []
        node_energy = np.zeros_like(residual, dtype=float)
        expected_attempts_by_link: dict[LinkKey, float] = {}
        energy_by_link: dict[LinkKey, float] = {}
        delay_by_link: dict[LinkKey, float] = {}
        link_diagnostics: dict[LinkKey, LinkExpectedCost] = {}
        route_success: dict[int, float] = {}
        packet_delays: dict[int, float] = {}
        raw_energy = 0.0
        pairwise = positions[:, None, :] - positions[None, :, :]
        neighbor_count = np.count_nonzero(
            np.linalg.norm(pairwise, axis=2) <= params.transmission_range_m, axis=1
        ) - 1
        contention = contention_multiplier(params, neighbor_count)

        selected = tuple(int(ch) for ch in candidate.selected_cluster_heads)
        alive = {
            int(node)
            for node in np.flatnonzero(residual > params.dead_energy_threshold_j)
        }
        if len(set(selected)) != len(selected):
            reasons.append("duplicate cluster head")
        for ch in selected:
            if not self._valid_node(ch, len(residual)):
                reasons.append(f"invalid cluster head: {ch}")
            elif residual[ch] <= params.dead_energy_threshold_j:
                reasons.append(f"dead cluster head: {ch}")
        if candidate.route_plan is None:
            reasons.append("missing route plan")
        elif not candidate.route_plan.is_feasible:
            reasons.append(candidate.route_plan.invalid_reason or "broken_route")

        assigned_sources: list[int] = []
        for ch, members in candidate.assignments.items():
            if int(ch) not in selected:
                reasons.append(f"assignment references unselected cluster head: {ch}")
            assigned_sources.extend(int(node) for node in members)
        missing = alive - set(assigned_sources)
        duplicate_sources = {
            node for node in assigned_sources if assigned_sources.count(node) > 1
        }
        if missing:
            reasons.append(f"nodes not assigned to a cluster head: {sorted(missing)}")
        if duplicate_sources:
            reasons.append(f"nodes assigned more than once: {sorted(duplicate_sources)}")
        if set(assigned_sources) - alive:
            reasons.append(
                f"assigned dead or invalid nodes: {sorted(set(assigned_sources) - alive)}"
            )

        route_cache: dict[int, tuple[float, float, list[LinkExpectedCost]]] = {}
        for ch in selected:
            if ch not in candidate.assignments:
                reasons.append(f"cluster head has no assignment: {ch}")
            route = candidate.routes.get(ch)
            if route is not None:
                for hop in route:
                    if hop is not None and int(hop) not in selected:
                        reasons.append(f"route uses unselected relay: {hop}")
            route_evaluation = self._evaluate_route(
                ch, route, positions, sink, residual, context, params, contention, reasons
            )
            if route_evaluation is None:
                continue
            route_cache[ch] = route_evaluation
            route_probability, route_delay, route_links = route_evaluation
            own_aggregation = (
                aggregation_energy_j(context.packet_bits, params) * float(contention[ch])
            )
            raw_energy += own_aggregation
            node_energy[ch] += own_aggregation
            for link in route_links:
                raw_energy += self._record_link(
                    link,
                    node_energy,
                    expected_attempts_by_link,
                    energy_by_link,
                    delay_by_link,
                    link_diagnostics,
                )

            members = tuple(int(node) for node in candidate.assignments.get(ch, ()))
            sources = tuple(dict.fromkeys(members))
            for source in sources:
                if not self._valid_node(source, len(residual)):
                    reasons.append(f"invalid source node: {source}")
                    continue
                if residual[source] <= params.dead_energy_threshold_j:
                    reasons.append(f"dead source node: {source}")
                    continue
                packet_probability = route_probability
                packet_delay = route_delay
                if source != ch:
                    member_distance = float(np.linalg.norm(positions[source] - positions[ch]))
                    if member_distance > params.transmission_range_m:
                        reasons.append(
                            f"link exceeds transmission range: {source}->{ch}"
                        )
                        continue
                    member_link = self._evaluate_link(
                        source,
                        ch,
                        positions,
                        sink,
                        context,
                        params,
                        contention,
                        include_aggregation=True,
                    )
                    raw_energy += self._record_link(
                        member_link,
                        node_energy,
                        expected_attempts_by_link,
                        energy_by_link,
                        delay_by_link,
                        link_diagnostics,
                    )
                    packet_probability *= member_link.limited_success_probability
                    packet_delay += member_link.expected_delay_s
                route_success[source] = packet_probability
                packet_delays[source] = packet_delay
                if (
                    context.minimum_route_success_probability > 0.0
                    and packet_probability < context.minimum_route_success_probability
                ):
                    reasons.append(
                        f"route reliability below minimum for source {source}: "
                        f"{packet_probability:.12g}"
                    )

        control_energy = aggregation_energy_j(params.broadcast_packet_size_bits, params)
        for node in alive:
            value = control_energy * float(contention[node])
            raw_energy += value
            node_energy[node] += value

        if not packet_delays:
            reasons.append("no valid packet route to sink")
        required_packet_count = len(alive)
        delivered_packet_count = len(packet_delays)
        if delivered_packet_count != required_packet_count:
            reasons.append(
                "delivered packet count does not equal required packet count: "
                f"{delivered_packet_count} != {required_packet_count}"
            )

        for node, required in enumerate(node_energy):
            if required > residual[node]:
                reasons.append(
                    f"expected energy exceeds residual energy for node {node}: "
                    f"{required:.12g} > {residual[node]:.12g}"
                )

        raw_delay = float(np.mean(list(packet_delays.values()))) if packet_delays else 0.0
        available_energy = float(
            sum(max(float(residual[node]) - params.dead_energy_threshold_j, 0.0) for node in alive)
        )
        energy_denominator = available_energy + context.energy_epsilon_j
        reference_delay = (
            context.packet_bits / params.node_bit_rate_bps
            + params.transmission_range_m / params.sound_speed_water_mps
            + context.delay_epsilon_s
        )
        if not np.isfinite(energy_denominator) or energy_denominator <= 0.0:
            reasons.append("invalid available-energy denominator")
        if not np.isfinite(reference_delay) or reference_delay <= 0.0:
            reasons.append("invalid reference-delay denominator")
        normalized_energy = (
            raw_energy / energy_denominator if energy_denominator > 0.0 else float("inf")
        )
        normalized_delay = (
            raw_delay / reference_delay if reference_delay > 0.0 else float("inf")
        )
        feasible = not reasons
        if feasible:
            log_objective = (
                context.lambda_energy
                * np.log(normalized_energy + context.objective_epsilon)
                + (1.0 - context.lambda_energy)
                * np.log(normalized_delay + context.objective_epsilon)
            )
            objective = float(np.exp(log_objective))
            if np.isnan(objective):
                reasons.append("objective evaluated to NaN")
                feasible = False
                objective = float("inf")
                log_objective = float("inf")
        else:
            objective = float("inf")
            log_objective = float("inf")
        return ObjectiveResult(
            objective_value=float(objective),
            raw_energy_j=float(raw_energy),
            raw_average_delay_s=raw_delay,
            normalized_energy=float(normalized_energy),
            normalized_delay=float(normalized_delay),
            feasible=feasible,
            infeasible_reasons=tuple(dict.fromkeys(reasons)),
            route_success_probabilities=route_success,
            expected_attempts_by_link=expected_attempts_by_link,
            energy_by_link=energy_by_link,
            delay_by_link=delay_by_link,
            diagnostics={
                "node_expected_energy_j": {
                    int(node): float(value) for node, value in enumerate(node_energy)
                },
                "packet_expected_delay_s": packet_delays,
                "link_expected_costs": link_diagnostics,
            },
            lambda_weight=float(context.lambda_energy),
            available_energy_j=available_energy,
            reference_delay_s=float(reference_delay),
            delivered_packet_count=delivered_packet_count,
            required_packet_count=required_packet_count,
            log_objective=float(log_objective),
        )

    @staticmethod
    def _validate_context(context: EvaluationContext) -> None:
        values = (
            context.lambda_energy,
            context.lambda_delay,
            context.normalization_epsilon,
            context.energy_epsilon_j,
            context.delay_epsilon_s,
            context.objective_epsilon,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("objective configuration must be finite")
        if context.lambda_energy < 0.0 or context.lambda_delay < 0.0:
            raise ValueError("objective weights must be non-negative")
        if abs(context.lambda_energy + context.lambda_delay - 1.0) > 1.0e-9:
            raise ValueError("objective weights must sum to 1")
        if (
            context.normalization_epsilon <= 0.0
            or context.energy_epsilon_j <= 0.0
            or context.delay_epsilon_s <= 0.0
            or context.objective_epsilon <= 0.0
        ):
            raise ValueError("all objective epsilons must be positive")
        if not 0.0 <= context.minimum_route_success_probability <= 1.0:
            raise ValueError("minimum route success probability must be in [0, 1]")
        if context.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if context.packet_bits <= 0 or context.frequency_khz <= 0.0:
            raise ValueError("packet_bits and frequency_khz must be positive")
        if context.bandwidth_hz <= 0.0:
            raise ValueError("bandwidth_hz must be positive")

    def _evaluate_route(
        self,
        source_ch: int,
        route: tuple[int | None, ...] | None,
        positions: np.ndarray,
        sink: np.ndarray,
        residual: np.ndarray,
        context: EvaluationContext,
        params: TunableParams,
        contention: np.ndarray,
        reasons: list[str],
    ) -> tuple[float, float, list[LinkExpectedCost]] | None:
        if not route:
            reasons.append(f"no route to sink for cluster head {source_ch}")
            return None
        current = source_ch
        visited = {current}
        probability = 1.0
        delay = 0.0
        links: list[LinkExpectedCost] = []
        reached_sink = False
        for destination in route:
            if destination is not None:
                destination = int(destination)
                if destination in visited:
                    reasons.append(f"route cycle for cluster head {source_ch}")
                    return None
                if not self._valid_node(destination, len(residual)):
                    reasons.append(f"invalid relay node: {destination}")
                    return None
                if residual[destination] <= params.dead_energy_threshold_j:
                    reasons.append(f"dead relay node: {destination}")
                    return None
            destination_position = sink if destination is None else positions[destination]
            if (
                float(np.linalg.norm(positions[current] - destination_position))
                > params.transmission_range_m
            ):
                reasons.append(
                    f"link exceeds transmission range: {current}->{destination}"
                )
                return None
            link = self._evaluate_link(
                current, destination, positions, sink, context, params, contention,
                include_aggregation=False
            )
            links.append(link)
            probability *= link.limited_success_probability
            delay += link.expected_delay_s
            if destination is None:
                reached_sink = True
                break
            visited.add(destination)
            current = destination
        if not reached_sink:
            reasons.append(f"route does not reach sink for cluster head {source_ch}")
            return None
        return probability, delay, links

    def _evaluate_link(
        self,
        source: int,
        destination: int | None,
        positions: np.ndarray,
        sink: np.ndarray,
        context: EvaluationContext,
        params: TunableParams,
        contention: np.ndarray,
        *,
        include_aggregation: bool,
    ) -> LinkExpectedCost:
        if self.cache.disabled:
            return self._evaluate_link_reference(
                source, destination, positions, sink, context, params, contention,
                include_aggregation=include_aggregation,
            )
        key = (source, destination)
        base = self.cache.get_link(key)
        if base is None:
            base = self._compute_base_link_cached(
                source, destination, positions, sink, context, params
            )
            self.cache.put_link(key, base)
        link = self._materialize_link(base, context, params, contention, include_aggregation)
        if self.cache.verify:
            reference = self._evaluate_link_reference(
                source, destination, positions, sink, context, params, contention,
                include_aggregation=include_aggregation,
            )
            self._assert_shadow_equal(reference, link)
            self.cache.stats.shadow_checks += 1
        return link

    def _compute_base_link_cached(
        self,
        source: int,
        destination: int | None,
        positions: np.ndarray,
        sink: np.ndarray,
        context: EvaluationContext,
        params: TunableParams,
    ) -> BaseLinkExpectedCost:
        destination_position = sink if destination is None else positions[destination]
        distance = float(np.linalg.norm(positions[source] - destination_position))
        if params.enable_packet_errors:
            environment = self.cache.static_environment(context, params)
            quality = channel_quality_with_static_environment(
                distance, context.packet_bits, params, environment
            )
            attempts, limited_success = expected_attempts_limited(
                quality["packet_success_probability"],
                context.max_retries if params.enable_retransmission else 0,
                context.normalization_epsilon,
            )
        else:
            quality = {
                "packet_success_probability": 1.0,
                "snr_db": float("inf"),
                "ber": 0.0,
                "per": 0.0,
                "transmission_loss_db": 0.0,
            }
            attempts, limited_success = 1.0, 1.0
        tx = transmit_energy_j(distance, context.packet_bits, params)
        rx = 0.0 if destination is None else receive_energy_j(context.packet_bits, params)
        delay = link_delay_s(
            context.packet_bits, distance, params,
            channel_capacity_bps=quality.get("channel_capacity_bps"),
        )
        attempt_delay = (
            delay.transmission_s + delay.reception_s + delay.propagation_s
            + delay.byte_alignment_s + delay.holding_s
        )
        return BaseLinkExpectedCost(
            source, destination, distance, quality["packet_success_probability"],
            limited_success, attempts, tx, rx, attempt_delay,
            attempts * attempt_delay, quality["snr_db"], quality["ber"],
            quality["per"], quality["transmission_loss_db"],
        )

    @staticmethod
    def _materialize_link(
        base: BaseLinkExpectedCost,
        context: EvaluationContext,
        params: TunableParams,
        contention: np.ndarray,
        include_aggregation: bool,
    ) -> LinkExpectedCost:
        tx = base.transmit_energy_j * float(contention[base.source_id])
        rx = (
            0.0 if base.destination_id is None
            else base.receive_energy_j * float(contention[base.destination_id])
        )
        aggregation = (
            base.limited_success_probability
            * aggregation_energy_j(context.packet_bits, params)
            * float(contention[base.destination_id])
            if include_aggregation
            else 0.0
        )
        expected_energy = base.expected_attempts * (tx + rx) + aggregation
        values = (
            base.distance_m,
            base.limited_success_probability,
            base.expected_attempts,
            tx,
            rx,
            aggregation,
            expected_energy,
            base.attempt_delay_s,
            base.expected_delay_s,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"non-finite link cost for {source}->{destination}")
        return LinkExpectedCost(
            source_id=base.source_id,
            destination_id=base.destination_id,
            distance_m=base.distance_m,
            packet_success_probability=base.packet_success_probability,
            limited_success_probability=base.limited_success_probability,
            expected_attempts=base.expected_attempts,
            transmit_energy_j=tx,
            receive_energy_j=rx,
            aggregation_energy_j=aggregation,
            expected_energy_j=expected_energy,
            attempt_delay_s=base.attempt_delay_s,
            expected_delay_s=base.expected_delay_s,
            snr_db=base.snr_db,
            ber=base.ber,
            per=base.per,
            transmission_loss_db=base.transmission_loss_db,
        )

    @staticmethod
    def _evaluate_link_reference(
        source: int,
        destination: int | None,
        positions: np.ndarray,
        sink: np.ndarray,
        context: EvaluationContext,
        params: TunableParams,
        contention: np.ndarray,
        *,
        include_aggregation: bool,
    ) -> LinkExpectedCost:
        destination_position = sink if destination is None else positions[destination]
        distance = float(np.linalg.norm(positions[source] - destination_position))
        if params.enable_packet_errors:
            quality = channel_quality(distance, context.frequency_khz, context.packet_bits, params)
            attempts, limited_success = expected_attempts_limited(
                quality["packet_success_probability"],
                context.max_retries if params.enable_retransmission else 0,
                context.normalization_epsilon,
            )
        else:
            quality = {
                "packet_success_probability": 1.0, "snr_db": float("inf"),
                "ber": 0.0, "per": 0.0, "transmission_loss_db": 0.0,
            }
            attempts, limited_success = 1.0, 1.0
        tx = transmit_energy_j(distance, context.packet_bits, params) * float(contention[source])
        rx = 0.0 if destination is None else (
            receive_energy_j(context.packet_bits, params) * float(contention[destination])
        )
        aggregation = (
            limited_success * aggregation_energy_j(context.packet_bits, params)
            * float(contention[destination]) if include_aggregation else 0.0
        )
        expected_energy = attempts * (tx + rx) + aggregation
        delay = link_delay_s(
            context.packet_bits, distance, params,
            channel_capacity_bps=quality.get("channel_capacity_bps"),
        )
        attempt_delay = (
            delay.transmission_s + delay.reception_s + delay.propagation_s
            + delay.byte_alignment_s + delay.holding_s
        )
        return LinkExpectedCost(
            source, destination, distance, quality["packet_success_probability"],
            limited_success, attempts, tx, rx, aggregation, expected_energy,
            attempt_delay, attempts * attempt_delay, quality["snr_db"],
            quality["ber"], quality["per"], quality["transmission_loss_db"],
        )

    @staticmethod
    def _assert_shadow_equal(reference: LinkExpectedCost, cached: LinkExpectedCost) -> None:
        for name in reference.__dataclass_fields__:
            left, right = getattr(reference, name), getattr(cached, name)
            if isinstance(left, float):
                if not np.isclose(left, right, rtol=1e-13, atol=1e-15, equal_nan=True):
                    raise AssertionError(f"link cache mismatch for {name}: {left!r} != {right!r}")
            elif left != right:
                raise AssertionError(f"link cache mismatch for {name}: {left!r} != {right!r}")

    @staticmethod
    def _record_link(
        link: LinkExpectedCost,
        node_energy: np.ndarray,
        attempts_by_link: dict[LinkKey, float],
        energy_by_link: dict[LinkKey, float],
        delay_by_link: dict[LinkKey, float],
        diagnostics: dict[LinkKey, LinkExpectedCost],
    ) -> float:
        key = (link.source_id, link.destination_id)
        attempts_by_link[key] = link.expected_attempts
        energy_by_link[key] = energy_by_link.get(key, 0.0) + link.expected_energy_j
        delay_by_link[key] = delay_by_link.get(key, 0.0) + link.expected_delay_s
        diagnostics[key] = link
        node_energy[link.source_id] += link.expected_attempts * link.transmit_energy_j
        if link.destination_id is not None:
            node_energy[link.destination_id] += (
                link.expected_attempts * link.receive_energy_j
                + link.aggregation_energy_j
            )
        return link.expected_energy_j

    @staticmethod
    def _valid_node(node: int, node_count: int) -> bool:
        return 0 <= int(node) < node_count


# Compatibility import only; all default optimizer call sites use the same implementation.
EnergyDelayObjectiveEvaluator = EnergyDelayObjective
