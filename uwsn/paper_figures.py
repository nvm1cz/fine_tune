"""
Paper-style figures (Figures 5–13): each scenario runs RUNS times; curves use the mean across runs.

Every figure uses exactly PAPER_ROUNDS simulation rounds (not FND / 10% alive stopping).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Sequence, Tuple

from .cases import SimulationCase
from .metrics import RunMetrics
from .run_batch import run_simulation_batch
from .run_config import BASE_SEED, OUTPUT_DIR, RUNS, SIMULATION_CASE, SIMULATION_PARAMS

# Fixed simulation length for all Figures 5–13 (vs-round plots and vs-node snapshots).
PAPER_ROUNDS = 1000

# Figures 5, 6, 10, 11: residual / dead vs round — 100 nodes in 100 m³.
ROUND_CURVE_SPACE_M = 100.0
ROUND_CURVE_NODE_COUNT = 100

# Fig 7–8: residual vs number of nodes, sweep approx. 0–500 (50-step), all in 100 m³.
NODE_SPACE_FIG78: Tuple[Tuple[int, float], ...] = tuple((n, 100.0) for n in range(50, 501, 50))

# Fig 9: dead vs nodes, approx. 0–1000 nodes (≤500 @ 100 m³, >500 @ 500 m³).
NODE_SPACE_FIG9: Tuple[Tuple[int, float], ...] = (
    (100, 100.0),
    (200, 100.0),
    (300, 100.0),
    (400, 100.0),
    (500, 100.0),
    (600, 500.0),
    (700, 500.0),
    (800, 500.0),
    (900, 500.0),
    (1000, 500.0),
)

# Fig 12–13: dead vs nodes, approx. 0–450 nodes, all 100 m³.
NODE_SPACE_FIG1213: Tuple[Tuple[int, float], ...] = tuple((n, 100.0) for n in range(50, 451, 50))


def _mean_series_across_runs(
    runs: Sequence[RunMetrics],
    attr: str,
) -> List[float]:
    series_list: List[List[float]] = []
    max_len = 0
    for run in runs:
        seq = getattr(run, attr)
        series_list.append(seq)
        max_len = max(max_len, len(seq))
    out: List[float] = []
    for i in range(max_len):
        vals = [s[i] for s in series_list if i < len(s)]
        if vals:
            out.append(sum(vals) / len(vals))
    return out


def _run_group(
    case: SimulationCase,
    params,
    *,
    stop_on_first_dead: bool = False,
    min_alive_ratio: float | None = None,
    max_rounds: int | None = None,
) -> List[RunMetrics]:
    return run_simulation_batch(
        case,
        params,
        (BASE_SEED + offset for offset in range(RUNS)),
        stop_on_first_dead=stop_on_first_dead,
        min_alive_ratio=min_alive_ratio,
        max_rounds=max_rounds,
    )


def _params_for(space_m: float, transmission_range_m: float):
    d = float(space_m)
    return replace(
        SIMULATION_PARAMS,
        width_m=d,
        height_m=d,
        depth_m=d,
        transmission_range_m=float(transmission_range_m),
    )


def _round_tick_step(max_round: int, max_ticks: int = 10, cap: int = 10000) -> int:
    if max_round <= 0:
        return 1
    ideal = max(1, (max_round + max_ticks - 1) // max_ticks)
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000):
        if max_round / step <= max_ticks + 1:
            return min(step, cap)
    magnitude = 10 ** (len(str(ideal)) - 1)
    rounded = max(magnitude, ((ideal + magnitude - 1) // magnitude) * magnitude)
    return min(rounded, cap)


def _save_line_plot_rounds(
    output_dir: Path,
    filename: str,
    title: str,
    y_series: List[float],
    y_label: str,
    *,
    max_tick_step_cap: int = 10000,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib missing; skip", filename)
        return
    if not y_series:
        return
    rounds = list(range(1, len(y_series) + 1))
    max_round = len(y_series)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, y_series, linewidth=2.0, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    tick_step = _round_tick_step(max_round, max_ticks=10, cap=max_tick_step_cap)
    x_ticks = list(range(0, max_round + 1, tick_step))
    if x_ticks[-1] != max_round:
        x_ticks.append(max_round)
    ax.set_xlim(0, max_round)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _save_vs_nodes_plot(
    output_dir: Path,
    filename: str,
    title: str,
    x_nodes: List[int],
    y_values: List[float],
    y_label: str,
    *,
    x_axis_max: float | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skip", filename)
        return
    if not x_nodes:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_nodes, y_values, marker="o", linewidth=2.0, markersize=8, color="tab:green")
    ax.set_title(title)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    if x_axis_max is not None:
        ax.set_xlim(0, x_axis_max)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def run_figure_residual_vs_rounds(
    output_dir: Path,
    *,
    figure_num: int,
    transmission_range_m: float,
    initial_energy: float,
    packet_size_bits: int,
    caption_suffix: str,
) -> None:
    params = _params_for(ROUND_CURVE_SPACE_M, transmission_range_m)
    case = replace(
        SIMULATION_CASE,
        name=f"fig{figure_num}_residual_vs_rounds",
        node_count=ROUND_CURVE_NODE_COUNT,
        initial_energy=initial_energy,
        packet_size_bits=packet_size_bits,
    )
    runs = _run_group(
        case,
        params,
        stop_on_first_dead=False,
        min_alive_ratio=None,
        max_rounds=PAPER_ROUNDS,
    )
    mean_residual = _mean_series_across_runs(runs, "residual_energy_by_round")
    title = (
        f"Figure {figure_num}: Residual energy vs round (mean of {RUNS} runs)\n"
        f"{caption_suffix}; cube side={int(ROUND_CURVE_SPACE_M)} m, N={ROUND_CURVE_NODE_COUNT}; "
        f"{PAPER_ROUNDS} rounds"
    )
    _save_line_plot_rounds(
        output_dir,
        f"figure_{figure_num:02d}_residual_energy_vs_rounds.png",
        title,
        mean_residual,
        "Residual energy (J)",
    )


def run_figure_dead_vs_rounds(
    output_dir: Path,
    *,
    figure_num: int,
    transmission_range_m: float,
    initial_energy: float,
    packet_size_bits: int,
    caption_suffix: str,
) -> None:
    params = _params_for(ROUND_CURVE_SPACE_M, transmission_range_m)
    case = replace(
        SIMULATION_CASE,
        name=f"fig{figure_num}_dead_vs_rounds",
        node_count=ROUND_CURVE_NODE_COUNT,
        initial_energy=initial_energy,
        packet_size_bits=packet_size_bits,
    )
    runs = _run_group(
        case,
        params,
        stop_on_first_dead=False,
        min_alive_ratio=None,
        max_rounds=PAPER_ROUNDS,
    )
    mean_dead = _mean_series_across_runs(runs, "dead_nodes_by_round")
    title = (
        f"Figure {figure_num}: Dead nodes vs round (mean of {RUNS} runs)\n"
        f"{caption_suffix}; cube side={int(ROUND_CURVE_SPACE_M)} m, N={ROUND_CURVE_NODE_COUNT}; "
        f"{PAPER_ROUNDS} rounds"
    )
    _save_line_plot_rounds(
        output_dir,
        f"figure_{figure_num:02d}_dead_nodes_vs_rounds.png",
        title,
        mean_dead,
        "Dead nodes",
    )


def run_figure_residual_and_dead_vs_rounds_shared(
    output_dir: Path,
    *,
    figure_num_residual: int,
    figure_num_dead: int,
    transmission_range_m: float,
    initial_energy: float,
    packet_size_bits: int,
    caption_residual: str,
    caption_dead: str,
) -> None:
    """One simulation batch for two figures when params match (e.g. Fig 5+10, Fig 6+11)."""
    params = _params_for(ROUND_CURVE_SPACE_M, transmission_range_m)
    case = replace(
        SIMULATION_CASE,
        name=f"fig{figure_num_residual}_{figure_num_dead}_shared_vs_rounds",
        node_count=ROUND_CURVE_NODE_COUNT,
        initial_energy=initial_energy,
        packet_size_bits=packet_size_bits,
    )
    runs = _run_group(
        case,
        params,
        stop_on_first_dead=False,
        min_alive_ratio=None,
        max_rounds=PAPER_ROUNDS,
    )
    mean_residual = _mean_series_across_runs(runs, "residual_energy_by_round")
    mean_dead = _mean_series_across_runs(runs, "dead_nodes_by_round")
    title_r = (
        f"Figure {figure_num_residual}: Residual energy vs round (mean of {RUNS} runs)\n"
        f"{caption_residual}; cube side={int(ROUND_CURVE_SPACE_M)} m, N={ROUND_CURVE_NODE_COUNT}; "
        f"{PAPER_ROUNDS} rounds"
    )
    title_d = (
        f"Figure {figure_num_dead}: Dead nodes vs round (mean of {RUNS} runs)\n"
        f"{caption_dead}; cube side={int(ROUND_CURVE_SPACE_M)} m, N={ROUND_CURVE_NODE_COUNT}; "
        f"{PAPER_ROUNDS} rounds"
    )
    _save_line_plot_rounds(
        output_dir,
        f"figure_{figure_num_residual:02d}_residual_energy_vs_rounds.png",
        title_r,
        mean_residual,
        "Residual energy (J)",
    )
    _save_line_plot_rounds(
        output_dir,
        f"figure_{figure_num_dead:02d}_dead_nodes_vs_rounds.png",
        title_d,
        mean_dead,
        "Dead nodes",
    )


def _snapshot_metrics(
    transmission_range_m: float,
    initial_energy: float,
    packet_size_bits: int,
    *,
    node_space_pairs: Sequence[Tuple[int, float]],
) -> Tuple[List[int], List[float], List[float]]:
    """Returns (node_counts, mean_residual_sum, mean_dead) at last round (≤ PAPER_ROUNDS)."""
    node_counts: List[int] = []
    residual_means: List[float] = []
    dead_means: List[float] = []

    for node_count, space_m in node_space_pairs:
        params = _params_for(space_m, transmission_range_m)
        case = replace(
            SIMULATION_CASE,
            name=f"snapshot_n{node_count}_tx{transmission_range_m}",
            node_count=node_count,
            initial_energy=initial_energy,
            packet_size_bits=packet_size_bits,
        )
        runs = _run_group(
            case,
            params,
            stop_on_first_dead=False,
            min_alive_ratio=None,
            max_rounds=PAPER_ROUNDS,
        )
        # Series length == PAPER_ROUNDS when network survives that long
        def last_residual(r: RunMetrics) -> float:
            if not r.residual_energy_by_round:
                return float("nan")
            return float(r.residual_energy_by_round[-1])

        def last_dead(r: RunMetrics) -> float:
            if not r.dead_nodes_by_round:
                return 0.0
            return float(r.dead_nodes_by_round[-1])

        residual_means.append(sum(last_residual(r) for r in runs) / len(runs))
        dead_means.append(sum(last_dead(r) for r in runs) / len(runs))
        node_counts.append(node_count)

    return node_counts, residual_means, dead_means


def run_all_paper_figures(output_dir: Path | None = None) -> None:
    out = output_dir or Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Paper figures 5–13 → {out.resolve()} ({RUNS} runs × scenarios, {PAPER_ROUNDS} rounds) ===\n")

    # Figures 5 + 10 — same scenario (reuse one batch: residual vs rounds + dead vs rounds)
    print("Running figures 5 (residual vs rounds) & 10 (dead vs rounds) — shared simulation …")
    run_figure_residual_and_dead_vs_rounds_shared(
        out,
        figure_num_residual=5,
        figure_num_dead=10,
        transmission_range_m=200.0,
        initial_energy=1.0,
        packet_size_bits=6400,
        caption_residual=(
            "residual energy over 1000 rounds, N=100, transmission_range=200 m, "
            "initial_energy=1.0 J, packet_size=6400 bits"
        ),
        caption_dead=(
            "dead nodes over 1000 rounds, N=100, transmission_range=200 m, "
            "initial_energy=1.0 J, packet_size=6400 bits"
        ),
    )
    print("Running figures 6 (residual vs rounds) & 11 (dead vs rounds) — shared simulation …")
    # Figures 6 + 11
    run_figure_residual_and_dead_vs_rounds_shared(
        out,
        figure_num_residual=6,
        figure_num_dead=11,
        transmission_range_m=150.0,
        initial_energy=1.0,
        packet_size_bits=4000,
        caption_residual=(
            "residual energy over 1000 rounds, N=100, transmission_range=150 m, "
            "initial_energy=1.0 J, packet_size=4000 bits"
        ),
        caption_dead=(
            "dead nodes over 1000 rounds, N=100, transmission_range=150 m, "
            "initial_energy=1.0 J, packet_size=4000 bits"
        ),
    )

    # Figure 7 — residual vs nodes (~0–500), tx 150 m (dead series reused for Fig 12 below)
    print("Running figure 7 (residual vs number of nodes, sweep) …")
    nc7, res_m, dead_for_fig12 = _snapshot_metrics(
        150.0, 0.5, 4000, node_space_pairs=NODE_SPACE_FIG78
    )
    _save_vs_nodes_plot(
        out,
        "figure_07_residual_energy_vs_num_nodes.png",
        f"Figure 7: Residual energy vs number of nodes (mean of {RUNS} runs, {PAPER_ROUNDS} rounds)\n"
        "nodes ~0–500, transmission_range=150 m, initial_energy=0.5 J, packet_size=4000 bits",
        nc7,
        res_m,
        "Residual energy (J) at snapshot",
        x_axis_max=500,
    )

    # Figure 8 — residual vs nodes (~0–500), tx 200 m (dead slice reused for Fig 13)
    print("Running figure 8 (residual vs number of nodes, sweep) …")
    nc8, res8, dead_for_fig13 = _snapshot_metrics(
        200.0, 0.5, 6400, node_space_pairs=NODE_SPACE_FIG78
    )
    _save_vs_nodes_plot(
        out,
        "figure_08_residual_energy_vs_num_nodes.png",
        f"Figure 8: Residual energy vs number of nodes (mean of {RUNS} runs, {PAPER_ROUNDS} rounds)\n"
        "nodes ~0–500, transmission_range=200 m, initial_energy=0.5 J, packet_size=6400 bits",
        nc8,
        res8,
        "Residual energy (J) at snapshot",
        x_axis_max=500,
    )

    # Figure 9 — dead vs nodes (~0–1000)
    print("Running figure 9 (dead vs number of nodes, sweep) …")
    nc9, _, dead9 = _snapshot_metrics(
        200.0, 0.5, 6400, node_space_pairs=NODE_SPACE_FIG9
    )
    _save_vs_nodes_plot(
        out,
        "figure_09_dead_nodes_vs_num_nodes.png",
        f"Figure 9: Dead nodes vs number of nodes (mean of {RUNS} runs, {PAPER_ROUNDS} rounds)\n"
        "nodes ~0–1000, transmission_range=200 m, initial_energy=0.5 J, packet_size=6400 bits",
        nc9,
        dead9,
        "Dead nodes",
        x_axis_max=1000,
    )

    # Figures 10–11: produced together with Fig 5–6 (shared runs).

    # Figure 12 — same (n, space) points as first len(NODE_SPACE_FIG1213) samples from Fig 7 sweep
    print("Running figure 12 (dead vs nodes — from figure 7 sweep data, saving plot) …")
    n_shared = len(NODE_SPACE_FIG1213)
    nc12 = nc7[:n_shared]
    dead12 = dead_for_fig12[:n_shared]
    _save_vs_nodes_plot(
        out,
        "figure_12_dead_nodes_vs_num_nodes.png",
        f"Figure 12: Dead nodes vs number of nodes (mean of {RUNS} runs, {PAPER_ROUNDS} rounds)\n"
        "nodes ~0–450, transmission_range=150 m, initial_energy=0.5 J, packet_size=4000 bits",
        nc12,
        dead12,
        "Dead nodes",
        x_axis_max=450,
    )

    # Figure 13 — dead slice from Fig 8 sweep (overlapping node range)
    print("Running figure 13 (dead vs nodes — from figure 8 sweep data, saving plot) …")
    nc13 = nc8[:n_shared]
    dead13 = dead_for_fig13[:n_shared]
    _save_vs_nodes_plot(
        out,
        "figure_13_dead_nodes_vs_num_nodes.png",
        f"Figure 13: Dead nodes vs number of nodes (mean of {RUNS} runs, {PAPER_ROUNDS} rounds)\n"
        "nodes ~0–450, transmission_range=200 m, initial_energy=0.5 J, packet_size=4000 bits",
        nc13,
        dead13,
        "Dead nodes",
        x_axis_max=450,
    )

    print("\n=== Paper figures 5–13 finished ===\n")
