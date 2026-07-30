from __future__ import annotations

from typing import Callable, List

import numpy as np

from .algorithms import NetworkState, UWSNProtocol, create_algorithm
from .cases import SimulationCase
from .deployment import compute_distances_to_sink, compute_layers, compute_neighbor_degree, compute_pairwise_distances, deploy_nodes, resolve_sink
from .models.energy import transmit_energy_j_array
from .models.mobility import update_positions_with_current
from .metrics import RunMetrics
from .run_config import TunableParams
from .routing.transmission import TransmissionStats, execute_round_transmissions
from .routing.planning import RoutePlan, validate_assignments, validate_route_plan


class UWSNSimulator:
    def __init__(
        self,
        case: SimulationCase,
        params: TunableParams,
        seed: int,
        verbose: bool = True,
        track_pso_convergence: bool = False,
        initial_positions: np.ndarray | None = None,
        seed_streams: dict[str, int] | None = None,
        algorithm: UWSNProtocol | None = None,
    ) -> None:
        self.case = case
        self.params = params
        self.seed = seed
        self.verbose = verbose
        self.track_pso_convergence = track_pso_convergence
        self.pso_convergence: list[dict[str, float]] = []
        self.cluster_snapshots: list[dict[str, object]] = []
        self._optimizer_refresh_index = 0
        if seed_streams is None:
            children = np.random.SeedSequence(seed).spawn(5)
            stream_seeds = {
                name: int(child.generate_state(1, dtype=np.uint32)[0])
                for name, child in zip(
                    ("topology", "protocol", "optimizer", "channel", "mobility"), children
                )
            }
        else:
            required = {"topology", "protocol", "optimizer", "channel", "mobility"}
            missing = required.difference(seed_streams)
            if missing:
                raise ValueError(f"Missing seed streams: {sorted(missing)}")
            stream_seeds = {name: int(seed_streams[name]) for name in required}
        self.seed_streams = stream_seeds
        self.topology_rng = np.random.default_rng(stream_seeds["topology"])
        self.protocol_rng = np.random.default_rng(stream_seeds["protocol"])
        self.optimizer_rng = np.random.default_rng(stream_seeds["optimizer"])
        self.channel_rng = np.random.default_rng(stream_seeds["channel"])
        self.mobility_rng = np.random.default_rng(stream_seeds["mobility"])
        self.np_rng = self.optimizer_rng  # Backward-compatible public alias.
        self.algorithm = algorithm or create_algorithm(getattr(params, "optimizer", "pso"))
        self.algorithm_id = self.algorithm.algorithm_id
        self.algorithm_diagnostics: dict[str, object] = {}
        if initial_positions is None:
            self.positions = deploy_nodes(case, params, self.topology_rng)
        else:
            positions = np.asarray(initial_positions, dtype=float)
            expected_shape = (case.node_count, 3)
            if positions.shape != expected_shape:
                raise ValueError(
                    f"initial_positions must have shape {expected_shape}, got {positions.shape}"
                )
            self.positions = positions.copy()
        self.energies = np.full(case.node_count, case.initial_energy, dtype=float)
        self.sink = resolve_sink(params)
        self._refresh_topology()
        self.current_chs: list[int] = []
        self.current_assignments: dict[int, list[int]] = {}
        self.current_route_plan: RoutePlan | None = None
        self.current_objective_result = None
        self.optimization_events: list[dict[str, object]] = []
        self.round_configuration_history: list[dict[str, object]] = []
        self._pending_reoptimization_reason = "initial"
        self.last_routing_edges: list[tuple[int, int | None]] = []
        self.last_delay_records: list[float] = []
        self.transmission_stats = TransmissionStats()
        self.mobility_displacement_m = 0.0
        self._clusters_printed = False

    def _refresh_topology(self) -> None:
        self.layers = compute_layers(self.positions, self.params)
        self.dist_to_sink = compute_distances_to_sink(self.positions, self.sink)
        self.distance_matrix = compute_pairwise_distances(self.positions)
        self.neighbor_degree = compute_neighbor_degree(self.positions, self.params, self.distance_matrix)
        self.neighbor_count = (
            np.count_nonzero(self.distance_matrix <= self.params.transmission_range_m, axis=1) - 1
        ).astype(float)
        self.tx_cost_matrix_per_bit = transmit_energy_j_array(
            self.distance_matrix, 1, self.params
        )
        self.tx_cost_to_sink_per_bit = transmit_energy_j_array(
            self.dist_to_sink, 1, self.params
        )

    def _apply_current_model(self, round_idx: int) -> None:
        if not bool(getattr(self.params, "current_model_enabled", False)):
            return
        interval = int(getattr(self.params, "current_update_interval_rounds", 1))
        if interval <= 0:
            raise ValueError("current_update_interval_rounds must be positive")
        if round_idx % interval != 0:
            return
        round_duration_s = float(
            getattr(self.params, "current_round_duration_s", self.params.current_time_step_s)
        )
        dt_s = round_duration_s * interval
        t_s = float(round_idx) * round_duration_s
        previous_positions = self.positions.copy()
        self.positions = update_positions_with_current(
            self.positions,
            t_s,
            dt_s,
            self.params,
            alive_mask=self._alive_mask(),
        )
        self.mobility_displacement_m += float(
            np.sum(np.linalg.norm(self.positions - previous_positions, axis=1))
        )
        self._refresh_topology()

    def _alive_mask(self) -> np.ndarray:
        return self.energies > self.params.dead_energy_threshold_j

    def _need_recluster(self, round_idx: int) -> bool:
        if not self.current_assignments:
            self._pending_reoptimization_reason = "initial"
            return True
        if round_idx == 1:
            self._pending_reoptimization_reason = "initial"
            return True
        if (round_idx - 1) % max(1, self.params.recluster_interval) == 0:
            self._pending_reoptimization_reason = "periodic"
            return True
        dead_chs = [
            ch for ch in self.current_chs
            if self.energies[ch] <= self.params.dead_energy_threshold_j
        ]
        if dead_chs and self.params.reoptimize_on_dead_ch_or_relay:
            relay_nodes = (
                {
                    int(ch)
                    for ch, load in self.current_route_plan.forwarding_load_by_ch.items()
                    if load > 0
                }
                if self.current_route_plan is not None else set()
            )
            self._pending_reoptimization_reason = (
                "dead_relay" if any(ch in relay_nodes for ch in dead_chs)
                else "dead_cluster_head"
            )
            return True
        if self.current_route_plan is not None and self.params.validate_route_every_round:
            assignment_valid, assignment_reason = validate_assignments(
                self.current_assignments,
                self.current_chs,
                self.positions,
                self.energies,
                self.params,
            )
            if not assignment_valid and self.params.reoptimize_on_invalid_assignment:
                self._pending_reoptimization_reason = assignment_reason or "invalid_member_assignment"
                return True
            route_valid, route_reason = validate_route_plan(
                self.current_route_plan,
                self.current_chs,
                self.positions,
                self.sink,
                self.energies,
                self.params,
            )
            if not route_valid and self.params.reoptimize_on_broken_route:
                self._pending_reoptimization_reason = route_reason or "broken_route"
                return True
        return False

    def _refresh_clusters(self, round_idx: int) -> None:
        state = NetworkState(
            self.case,
            self.params,
            self.positions,
            self.energies,
            self.layers,
            self.dist_to_sink,
            self.neighbor_degree,
            self.distance_matrix,
        )
        self._optimizer_refresh_index += 1

        def record_convergence(iteration: int, best_score: float) -> None:
            if not self.track_pso_convergence:
                return
            self.pso_convergence.append(
                {
                    "round": float(round_idx),
                    "refresh_index": float(self._optimizer_refresh_index),
                    "iteration": float(iteration),
                    "best_score": float(best_score),
                    "candidate_count": float(
                        self.algorithm_diagnostics.get("candidate_count", 0)
                    ),
                }
            )
        selection = self.algorithm.select_cluster_heads(
            round_idx,
            state,
            self.protocol_rng,
            self.optimizer_rng,
            record_convergence if self.track_pso_convergence else None,
        )
        chs, assignments = list(selection.cluster_heads), selection.assignments
        self.algorithm_diagnostics = dict(selection.diagnostics)
        candidate_count = float(self.algorithm_diagnostics.get("candidate_count", 0))
        refresh_events = [
            event for event in self.pso_convergence
            if int(event["refresh_index"]) == self._optimizer_refresh_index
        ]
        detailed_history = self.algorithm_diagnostics.get("convergence_history", ())
        for index, event in enumerate(refresh_events):
            if index < len(detailed_history) and isinstance(detailed_history[index], dict):
                event.update(detailed_history[index])
                event["best_score"] = event["best_J"]
            if int(event["refresh_index"]) == self._optimizer_refresh_index:
                event["candidate_count"] = candidate_count
        self.current_chs = [ch for ch in chs if self.energies[ch] > self.params.dead_energy_threshold_j]
        self.current_assignments = assignments
        self.current_route_plan = selection.route_plan
        self.current_objective_result = selection.objective_result
        route_plan_id = (
            selection.route_plan.plan_id if selection.route_plan is not None else None
        )
        self.optimization_events.append(
            {
                "algorithm": self.algorithm_id,
                "optimization_round": round_idx,
                "reoptimization_reason": self._pending_reoptimization_reason,
                "selected_cluster_heads": list(self.current_chs),
                "assignment_mode": self.algorithm_diagnostics.get(
                    "assignment_mode", "legacy"
                ),
                "assigned_member_count": sum(len(v) for v in assignments.values()),
                "routing_mode": self.algorithm_diagnostics.get(
                    "routing_mode", "legacy_greedy"
                ),
                "route_plan_id": route_plan_id,
                "routes": (
                    dict(selection.route_plan.route_by_ch)
                    if selection.route_plan is not None else {}
                ),
                "forwarding_load_by_ch": (
                    dict(selection.route_plan.forwarding_load_by_ch)
                    if selection.route_plan is not None else {}
                ),
                "objective_J": (
                    selection.objective_result.objective_J
                    if selection.objective_result is not None else None
                ),
                "energy_term_e": (
                    selection.objective_result.energy_term_e
                    if selection.objective_result is not None else None
                ),
                "delay_term_d": (
                    selection.objective_result.delay_term_d
                    if selection.objective_result is not None else None
                ),
            }
        )
        self.cluster_snapshots.append(
            {
                "round": round_idx,
                "refresh_index": self._optimizer_refresh_index,
                "cluster_heads": list(self.current_chs),
                "assignments": {int(ch): list(members) for ch, members in assignments.items()},
            }
        )
        
        # Print clustering info only once after initial cluster formation
        if self.verbose and not self._clusters_printed:
            print("\n=== INITIAL CLUSTERING INFO ===")
            print(f"Number of Cluster Heads: {len(self.current_chs)}")
            print(f"Cluster Heads: {sorted(self.current_chs)}\n")
            for ch in sorted(self.current_chs):
                members = self.current_assignments.get(ch, [])
                alive_members = [m for m in members if self.energies[m] > self.params.dead_energy_threshold_j]
                ch_layer = int(self.layers[ch])
                ch_pos = self.positions[ch]
                print(f"CH {ch} (Layer {ch_layer}): Pos=({ch_pos[0]:.2f}, {ch_pos[1]:.2f}, {ch_pos[2]:.2f}), Members = {sorted(alive_members)} (Count: {len(alive_members)})")
            print("================================\n")
            self._clusters_printed = True

    def _run_round(self, round_idx: int, packets_received: int) -> int:
        self._apply_current_model(round_idx)
        reoptimized = self._need_recluster(round_idx)
        if reoptimized:
            self._refresh_clusters(round_idx)

        cluster_heads = [ch for ch in self.current_chs if self.energies[ch] > self.params.dead_energy_threshold_j]
        filtered_assignments = {
            ch: [member for member in members if self.energies[member] > self.params.dead_energy_threshold_j]
            for ch, members in self.current_assignments.items()
            if self.energies[ch] > self.params.dead_energy_threshold_j
        }
        self.last_routing_edges = []
        self.last_delay_records = []
        result = execute_round_transmissions(
            self.case,
            self.params,
            self.distance_matrix,
            self.tx_cost_matrix_per_bit,
            self.tx_cost_to_sink_per_bit,
            self.energies,
            self.layers,
            self.dist_to_sink,
            filtered_assignments,
            cluster_heads,
            packets_received,
            self.params.dead_energy_threshold_j,
            neighbor_count=self.neighbor_count,
            routing_edges=self.last_routing_edges,
            delay_records=self.last_delay_records,
            rng=self.channel_rng,
            transmission_stats=self.transmission_stats,
            route_plan=self.current_route_plan,
        )
        plan_id = self.current_route_plan.plan_id if self.current_route_plan else None
        self.round_configuration_history.append(
            {
                "round": round_idx,
                "configuration_valid": True,
                "assignment_valid": True,
                "route_valid": True,
                "reused_route_plan": not reoptimized and self.current_route_plan is not None,
                "reoptimization_occurred": reoptimized,
                "reoptimization_reason": (
                    self.optimization_events[-1]["reoptimization_reason"]
                    if reoptimized else None
                ),
                "objective_route_plan_id": self.algorithm_diagnostics.get(
                    "objective_route_plan_id"
                ),
                "execution_route_plan_id": plan_id,
                "invalid_reason": None,
            }
        )
        
        if self.verbose and round_idx == 1:
            print("================================\n")
        
        return result

    def run(
        self,
        stop_on_first_dead: bool = True,
        stop_at_ft5: bool = False,
        min_alive_ratio: float | None = None,
        max_rounds: int | None = None,
        round_callback: Callable[["UWSNSimulator", int, int], None] | None = None,
    ) -> RunMetrics:
        residual_by_round: List[float] = []
        dead_by_round: List[int] = []
        packets_by_round: List[int] = []
        delay_by_round: List[float] = []
        packets_received = 0
        fnd_round = None
        hnd_round = None
        lnd_round = None
        first_5pct_round = None

        round_idx = 0
        while True:
            round_idx += 1
            effective_max_rounds = self.case.rounds if max_rounds is None else max_rounds
            if round_idx > effective_max_rounds:
                break
            alive_before_round = int(np.count_nonzero(self._alive_mask()))
            if alive_before_round == 0:
                break

            packets_received = self._run_round(round_idx, packets_received)
            alive_after_round = int(np.count_nonzero(self._alive_mask()))
            dead_nodes = self.case.node_count - alive_after_round
            if first_5pct_round is None and dead_nodes >= 0.05 * self.case.node_count:
                first_5pct_round = round_idx
            if fnd_round is None and dead_nodes >= 1:
                fnd_round = round_idx

            if hnd_round is None and dead_nodes >= (self.case.node_count + 1) // 2:
                hnd_round = round_idx

            if lnd_round is None and alive_after_round == 0:
                lnd_round = round_idx

            residual_by_round.append(float(np.sum(self.energies)))
            dead_by_round.append(dead_nodes)
            packets_by_round.append(packets_received)
            delay_by_round.append(float(np.sum(self.last_delay_records)))

            if round_callback is not None:
                round_callback(self, round_idx, packets_received)

            # Stop simulation at the first dead node (FND condition).
            if stop_on_first_dead and dead_nodes >= 1:
                break

            if stop_at_ft5 and first_5pct_round is not None:
                break

            if min_alive_ratio is not None:
                alive_ratio = alive_after_round / max(1, self.case.node_count)
                if alive_ratio <= min_alive_ratio:
                    break

        # Print final cluster formation
        if self.verbose:
            print("\n=== FINAL CLUSTERING INFO (Last Round) ===")
            print(f"Number of Cluster Heads: {len(self.current_chs)}")
            print(f"Cluster Heads: {sorted(self.current_chs)}\n")
            for ch in sorted(self.current_chs):
                members = self.current_assignments.get(ch, [])
                alive_members = [m for m in members if self.energies[m] > self.params.dead_energy_threshold_j]
                ch_layer = int(self.layers[ch])
                ch_pos = self.positions[ch]
                ch_energy = self.energies[ch]
                print(f"CH {ch} (Layer {ch_layer}): Pos=({ch_pos[0]:.2f}, {ch_pos[1]:.2f}, {ch_pos[2]:.2f}), Energy={ch_energy:.6f}J, Members = {sorted(alive_members)} (Count: {len(alive_members)})")
            print("================================\n")

        return RunMetrics(
            seed=self.seed,
            fnd_round=fnd_round or None,
            hnd_round=hnd_round or None,
            lnd_round=lnd_round or None,
            first_5pct_round=first_5pct_round, 
            packets_received=packets_received,
            residual_energy=float(np.sum(self.energies)),
            alive_nodes=int(np.count_nonzero(self._alive_mask())),
            residual_energy_by_round=residual_by_round,
            dead_nodes_by_round=dead_by_round,
            packets_received_by_round=packets_by_round,
            delay_by_round=delay_by_round,
            transmission_attempts=self.transmission_stats.attempts,
            packets_attempted=self.transmission_stats.attempts,
            retransmissions=self.transmission_stats.retransmissions,
            dropped_packets=self.transmission_stats.dropped_packets,
            packets_generated=self.transmission_stats.packets_generated,
            packets_delivered=self.transmission_stats.packets_delivered,
            packet_delivery_ratio=(
                self.transmission_stats.packets_delivered
                / self.transmission_stats.packets_generated
                if self.transmission_stats.packets_generated else None
            ),
            average_ber=(
                self.transmission_stats.ber_sum / self.transmission_stats.quality_samples
                if self.transmission_stats.quality_samples else None
            ),
            average_per=(
                self.transmission_stats.per_sum / self.transmission_stats.quality_samples
                if self.transmission_stats.quality_samples else None
            ),
            average_snr_db=(
                self.transmission_stats.snr_db_sum / self.transmission_stats.quality_samples
                if self.transmission_stats.quality_samples else None
            ),
            average_link_distance_m=(
                self.transmission_stats.link_distance_m_sum
                / self.transmission_stats.quality_samples
                if self.transmission_stats.quality_samples else None
            ),
            total_tx_energy_j=self.transmission_stats.total_tx_energy_j,
            total_rx_energy_j=self.transmission_stats.total_rx_energy_j,
            total_delay_s=self.transmission_stats.total_attempt_delay_s,
            average_e2e_delay_s=(
                self.transmission_stats.total_delivered_delay_s
                / self.transmission_stats.packets_delivered
                if self.transmission_stats.packets_delivered else None
            ),
            mobility_displacement_m=self.mobility_displacement_m,
            pso_convergence=self.pso_convergence,
        )
