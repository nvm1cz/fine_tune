from __future__ import annotations

import math
import re
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .metrics import RunMetrics, summarize_runs
from .paper_figures import run_all_paper_figures
from .run_batch import run_simulation_batch
from .run_config import BASE_SEED, OUTPUT_DIR, RUNS, SIMULATION_CASE, SIMULATION_PARAMS


def format_round_metric(value: float | None) -> str:
    if value is None:
        return "NA"
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def save_residual_energy_plot(output_dir: Path, case_name: str, runs: List[RunMetrics], step: int = 100) -> None:
    if not runs:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip residual energy plot.")
        return

    max_round = max(len(run.residual_energy_by_round) for run in runs)
    if max_round == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    color_map = plt.cm.get_cmap("tab10", max(1, len(runs)))
    for idx, run in enumerate(runs, start=1):
        run_rounds = list(range(1, len(run.residual_energy_by_round) + 1))
        if not run_rounds:
            continue
        marker_step = max(1, len(run_rounds) // 12)
        ax.plot(
            run_rounds,
            run.residual_energy_by_round,
            linewidth=1.8,
            alpha=0.9,
            color=color_map(idx - 1),
            marker="o",
            markersize=3,
            markevery=marker_step,
            label=f"Run {idx} (seed={run.seed}, FND={run.fnd_round})",
        )

    mean_residual_by_round: List[float] = []
    for idx in range(max_round):
        values = [run.residual_energy_by_round[idx] for run in runs if idx < len(run.residual_energy_by_round)]
        mean_residual_by_round.append(sum(values) / len(values))
    ax.plot(
        list(range(1, max_round + 1)),
        mean_residual_by_round,
        linewidth=2.4,
        color="black",
        linestyle="--",
        label="Mean",
    )

    ax.set_title(f" Residual Energy vs Round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Residual Energy (J)")
    ax.grid(True, linestyle="--", alpha=0.4)
    x_ticks = list(range(0, max_round + 1, step))
    if x_ticks[-1] != max_round:
        x_ticks.append(max_round)
    ax.set_xlim(0, max_round)
    ax.set_xticks(x_ticks)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    chart_path = output_dir / f"{case_name}_residual_energy_vs_round.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Saved residual energy chart: {chart_path}")


def save_space_sweep_plot(output_dir: Path, space_m: int, labels_and_series: List[tuple[str, List[float]]], step: int = 100) -> None:
    if not labels_and_series:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip sweep plot.")
        return

    max_round = max(len(series) for _, series in labels_and_series)
    if max_round == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    color_map = plt.cm.get_cmap("tab20", max(1, len(labels_and_series)))

    for idx, (label, series) in enumerate(labels_and_series):
        rounds = list(range(1, len(series) + 1))
        if not rounds:
            continue
        ax.plot(rounds, series, linewidth=1.1, alpha=0.85, color=color_map(idx), label=label)

    x_ticks = list(range(0, max_round + 1, step))
    if x_ticks[-1] != max_round:
        x_ticks.append(max_round)
    ax.set_xlim(0, max_round)
    ax.set_xticks(x_ticks)
    ax.set_title(f"Residual Energy vs Round - Space {space_m}m")
    ax.set_xlabel("Round")
    ax.set_ylabel("Residual Energy (J)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7, framealpha=0.9)
    fig.tight_layout()

    chart_path = output_dir / f"space_{space_m}m_residual_energy_sweep.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sweep chart: {chart_path}")


def _mean_residual_by_round(runs: List[RunMetrics]) -> List[float]:
    max_round = max((len(run.residual_energy_by_round) for run in runs), default=0)
    series: List[float] = []
    for idx in range(max_round):
        values = [run.residual_energy_by_round[idx] for run in runs if idx < len(run.residual_energy_by_round)]
        if values:
            series.append(sum(values) / len(values))
    return series


def _x_axis_step_for_rounds(max_round: int, max_ticks: int = 10) -> int:
    """Pick a round step so the X-axis has at most ~max_ticks readable labels."""
    if max_round <= 0:
        return 1
    ideal = max(1, (max_round + max_ticks - 1) // max_ticks)
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000):
        if max_round / step <= max_ticks + 1:
            return step
    magnitude = 10 ** (len(str(ideal)) - 1)
    return max(magnitude, ((ideal + magnitude - 1) // magnitude) * magnitude)


def save_alive_nodes_sweep_plot(
    output_dir: Path,
    labels_and_series: List[tuple[str, List[float]]],
    node_count: int,
    space_m: int = 100,
    max_tick_step_cap: int = 5000,
) -> None:
    if not labels_and_series:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib is not installed, skip alive-nodes plot.")
        return

    max_round = max(len(series) for _, series in labels_and_series)
    if max_round == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    color_map = plt.cm.get_cmap("tab20", max(1, len(labels_and_series)))
    for idx, (label, series) in enumerate(labels_and_series):
        rounds = list(range(1, len(series) + 1))
        if not rounds:
            continue
        ax.plot(
            rounds,
            series,
            linewidth=1.3,
            alpha=0.85,
            color=color_map(idx),
            label=label,
        )

    threshold = max(1, int(0.1 * node_count))
    ax.axhline(y=threshold, color="red", linestyle=":", linewidth=1.7, label=f"10% threshold ({threshold} nodes)")
    tick_step = _x_axis_step_for_rounds(max_round, max_ticks=10)
    tick_step = min(tick_step, max_tick_step_cap)
    tick_step = max(1, tick_step)
    x_ticks = list(range(0, max_round + 1, tick_step))
    if x_ticks[-1] != max_round:
        x_ticks.append(max_round)
    ax.set_xlim(0, max_round)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", rotation=28, labelsize=9)
    ax.set_ylim(0, node_count)
    ax.set_title(f"Alive Nodes vs Round - Sweep (space={space_m}m, nodes={node_count})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Alive nodes")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7, framealpha=0.9)
    fig.tight_layout()

    chart_path = output_dir / f"space_{space_m}m_nodes_{node_count}_alive_nodes_sweep.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved alive-nodes chart: {chart_path}")


FND_LOG_PATTERN = re.compile(
    r"\[space (?P<space>\d+)m, node (?P<node>\d+)(?:, runs=(?P<runs>\d+))?\]\s+"
    r"c=\((?P<c1>[-+]?\d+(?:\.\d+)?),(?P<c2>[-+]?\d+(?:\.\d+)?)\),\s+"
    r"pkt=(?P<pkt>\d+), e=(?P<energy>[-+]?\d+(?:\.\d+)?),\s+"
    r"FND=(?P<fnd>[-+]?\d+(?:\.\d+)?)"
)


def _parse_fnd_log_files(log_paths: Iterable[Path]) -> Dict[Tuple[int, int, float, float, int, float], float]:
    fnd_by_case: Dict[Tuple[int, int, float, float, int, float], float] = {}
    for log_path in log_paths:
        if not log_path.exists():
            continue
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = FND_LOG_PATTERN.search(line)
            if not match:
                continue
            key = (
                int(match.group("space")),
                int(match.group("node")),
                float(match.group("c1")),
                float(match.group("c2")),
                int(match.group("pkt")),
                float(match.group("energy")),
            )
            fnd_by_case[key] = float(match.group("fnd"))
    return fnd_by_case


def _format_plot_value(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _nice_tick_step(max_value: float, max_intervals: int = 8) -> float:
    if max_value <= 0:
        return 1.0
    raw_step = max_value / max(1, max_intervals)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        multiplier = 1
    elif normalized <= 2:
        multiplier = 2
    elif normalized <= 5:
        multiplier = 5
    else:
        multiplier = 10
    return multiplier * magnitude


def _y_ticks_from_zero(max_value: float, tick_step: float | None = None, padding: float = 1.08) -> Tuple[float, List[float]]:
    padded_max = max(0.0, max_value) * padding
    step = tick_step or _nice_tick_step(padded_max)
    upper = max(step, math.ceil(padded_max / step) * step)
    tick_count = int(round(upper / step))
    ticks = [idx * step for idx in range(tick_count + 1)]
    return upper, ticks


def save_fnd_faceted_plot_from_logs(
    output_dir: Path,
    log_paths: Iterable[Path],
    *,
    space_m: int = 500,
    node_counts: Tuple[int, ...] = (1000, 1500, 2000, 2500),
    filename_suffix: str = "",
    title_suffix: str = "",
    y_axis_upper: float | None = 4000.0,
    y_tick_step: float | None = 500.0,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip FND facet plot.")
        return

    fnd_by_case = _parse_fnd_log_files(log_paths)
    if not fnd_by_case:
        print("No FND data found in logs, skip FND facet plot.")
        return

    c_pairs = [(1.5, 1.5), (2.5, 0.5), (0.5, 2.5)]
    packet_sizes = [4000, 6400]
    initial_energies = [0.5, 1.0]
    colors = {
        (1.5, 1.5): "tab:blue",
        (2.5, 0.5): "tab:orange",
        (0.5, 2.5): "tab:green",
    }
    markers = {
        (1.5, 1.5): "o",
        (2.5, 0.5): "s",
        (0.5, 2.5): "^",
    }
    label_offsets = {
        (1.5, 1.5): (0, 8),
        (2.5, 0.5): (0, -14),
        (0.5, 2.5): (0, 18),
    }
    low_value_label_offsets = {
        (1.5, 1.5): (0, 10),
        (2.5, 0.5): (0, 24),
        (0.5, 2.5): (0, 38),
    }
    all_fnd_values = [
        value
        for (space, node_count, c1, c2, packet_size, initial_energy), value in fnd_by_case.items()
        if space == space_m
        and node_count in node_counts
        and (c1, c2) in c_pairs
        and packet_size in packet_sizes
        and initial_energy in initial_energies
    ]
    common_y_step: float | None = None
    common_y_upper: float | None = None
    common_y_ticks: List[float] = []
    if all_fnd_values:
        padded_data_upper = max(all_fnd_values) * 1.08
        target_upper = max(padded_data_upper, y_axis_upper or 0.0)
        common_y_step = y_tick_step or _nice_tick_step(target_upper)
        common_y_upper = max(common_y_step, math.ceil(target_upper / common_y_step) * common_y_step)
        tick_count = int(round(common_y_upper / common_y_step))
        common_y_ticks = [idx * common_y_step for idx in range(tick_count + 1)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, packet_size in enumerate(packet_sizes):
        for col, initial_energy in enumerate(initial_energies):
            ax = axes[row][col]
            facet_values: List[float] = []
            facet_series: Dict[Tuple[float, float], List[float | None]] = {}
            for c1, c2 in c_pairs:
                y_values = [
                    fnd_by_case.get((space_m, node_count, c1, c2, packet_size, initial_energy))
                    for node_count in node_counts
                ]
                facet_series[(c1, c2)] = y_values
                if all(value is None for value in y_values):
                    continue
                ax.plot(
                    node_counts,
                    y_values,
                    marker=markers[(c1, c2)],
                    linewidth=2.0,
                    markersize=6,
                    color=colors[(c1, c2)],
                    label=f"c=({c1},{c2})",
                )
                for node_count, fnd in zip(node_counts, y_values):
                    if fnd is None:
                        continue
                    facet_values.append(fnd)
                    xy_offset = label_offsets[(c1, c2)]
                    if common_y_step is not None and fnd < common_y_step * 0.35:
                        xy_offset = low_value_label_offsets[(c1, c2)]
                    ax.annotate(
                        _format_plot_value(fnd),
                        (node_count, fnd),
                        textcoords="offset points",
                        xytext=xy_offset,
                        ha="center",
                        fontsize=8,
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
                    )

            ax.set_title(f"pkt={packet_size}, e={initial_energy}")
            ax.set_xticks(node_counts)
            ax.set_xticklabels([str(node_count) for node_count in node_counts])
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(True, linestyle="--", alpha=0.35)
            if common_y_upper is not None:
                ax.set_ylim(0, common_y_upper)
                ax.set_yticks(common_y_ticks)
            if packet_size == 6400 and initial_energy == 0.5 and facet_values:
                inset_ax = ax.inset_axes([0.47, 0.46, 0.49, 0.46])
                inset_ax.set_facecolor("white")
                for c1, c2 in c_pairs:
                    y_values = facet_series.get((c1, c2), [])
                    if not y_values or all(value is None for value in y_values):
                        continue
                    inset_ax.plot(
                        node_counts,
                        y_values,
                        marker=markers[(c1, c2)],
                        linewidth=1.4,
                        markersize=3.5,
                        color=colors[(c1, c2)],
                    )
                zoom_upper, zoom_ticks = _y_ticks_from_zero(max(facet_values), padding=1.25)
                inset_ax.set_ylim(0, zoom_upper)
                inset_ax.set_yticks(zoom_ticks)
                inset_ax.set_xticks(node_counts)
                inset_ax.tick_params(axis="both", labelsize=6)
                inset_ax.tick_params(axis="x", rotation=25)
                inset_ax.grid(True, linestyle="--", alpha=0.25)
                inset_ax.set_title("zoom", fontsize=7, pad=2)
            if row == 1:
                ax.set_xlabel("Number of nodes")
            if col == 0:
                ax.set_ylabel("FND round")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, framealpha=0.9)
    fig.suptitle(
        f"FND vs Number of Nodes - Faceted by packet size and initial energy "
        f"(space={space_m}m{title_suffix})"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    chart_path = output_dir / f"space_{space_m}m{filename_suffix}_fnd_vs_nodes_faceted.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FND facet chart: {chart_path}")


def save_single_fnd_case_plot_from_logs(
    output_dir: Path,
    log_paths: Iterable[Path],
    *,
    space_m: int,
    packet_size: int,
    initial_energy: float,
    node_counts: Tuple[int, ...],
    filename_suffix: str = "",
    title_suffix: str = "",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip single FND case plot.")
        return

    fnd_by_case = _parse_fnd_log_files(log_paths)
    c_pairs = [(1.5, 1.5), (2.5, 0.5), (0.5, 2.5)]
    colors = {
        (1.5, 1.5): "tab:blue",
        (2.5, 0.5): "tab:orange",
        (0.5, 2.5): "tab:green",
    }
    markers = {
        (1.5, 1.5): "o",
        (2.5, 0.5): "s",
        (0.5, 2.5): "^",
    }
    label_offsets = {
        (1.5, 1.5): (0, 8),
        (2.5, 0.5): (0, 20),
        (0.5, 2.5): (0, 32),
    }

    fig, ax = plt.subplots(figsize=(9, 5.4))
    all_values: List[float] = []
    for c1, c2 in c_pairs:
        y_values = [
            fnd_by_case.get((space_m, node_count, c1, c2, packet_size, initial_energy))
            for node_count in node_counts
        ]
        if all(value is None for value in y_values):
            continue
        ax.plot(
            node_counts,
            y_values,
            marker=markers[(c1, c2)],
            linewidth=2.2,
            markersize=7,
            color=colors[(c1, c2)],
            label=f"c=({c1},{c2})",
        )
        for node_count, fnd in zip(node_counts, y_values):
            if fnd is None:
                continue
            all_values.append(fnd)
            ax.annotate(
                _format_plot_value(fnd),
                (node_count, fnd),
                textcoords="offset points",
                xytext=label_offsets[(c1, c2)],
                ha="center",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8},
            )

    if all_values:
        upper, ticks = _y_ticks_from_zero(max(all_values), padding=1.18)
        ax.set_ylim(0, max(50.0, upper))
        ax.set_yticks(ticks)
    ax.set_xticks(node_counts)
    ax.set_xticklabels([str(node_count) for node_count in node_counts])
    ax.set_title(
        f"Early FND detail - pkt={packet_size}, e={initial_energy}J, space={space_m}m{title_suffix}"
    )
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("FND round")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.text(
        0.01,
        0.02,
        "FND is the first node death, so this case is intentionally shown on its own scale.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="dimgray",
    )
    fig.tight_layout()

    chart_path = output_dir / (
        f"space_{space_m}m_pkt{packet_size}_e{str(initial_energy).replace('.', 'p')}"
        f"{filename_suffix}_fnd_detail.png"
    )
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved single FND case chart: {chart_path}")


def _mean_round_when_alive_at_or_below(runs: List[RunMetrics], node_count: int, alive_threshold: int) -> float | None:
    threshold_dead_nodes = node_count - alive_threshold
    rounds: List[int] = []
    for run in runs:
        for idx, dead_nodes in enumerate(run.dead_nodes_by_round, start=1):
            if dead_nodes >= threshold_dead_nodes:
                rounds.append(idx)
                break
    if not rounds:
        return None
    return sum(rounds) / len(rounds)


def _collect_alive_nodes_sweep(space_m: int, node_count: int) -> List[tuple[str, List[float]]]:
    c_pairs = [(1.5, 1.5), (2.5, 0.5), (0.5, 2.5)]
    packet_sizes = [4000, 6400]
    initial_energies = [0.5, 1.0]
    labels_and_series: List[tuple[str, List[float]]] = []
    dim = float(space_m)

    for (c1, c2), packet_size, initial_energy in product(
        c_pairs, packet_sizes, initial_energies
    ):
        case = replace(
            SIMULATION_CASE,
            name=f"s{space_m}_n{node_count}_c1{c1}_c2{c2}_pkt{packet_size}_e{initial_energy}",
            node_count=node_count,
            packet_size_bits=packet_size,
            initial_energy=initial_energy,
        )
        params = replace(
            SIMULATION_PARAMS,
            width_m=dim,
            height_m=dim,
            depth_m=dim,
            pso_c1=c1,
            pso_c2=c2,
        )

        print(
            f"Running space={space_m}m, nodes={node_count}, "
            f"c=({c1},{c2}), pkt={packet_size}, e={initial_energy}..."
        )
        runs = run_simulation_batch(
            case,
            params,
            (BASE_SEED + offset for offset in range(RUNS)),
            stop_on_first_dead=False,
            min_alive_ratio=0.1,
        )

        max_round = max((len(run.dead_nodes_by_round) for run in runs), default=0)
        mean_alive: List[float] = []
        for idx in range(max_round):
            alive_values = [
                case.node_count - run.dead_nodes_by_round[idx]
                for run in runs
                if idx < len(run.dead_nodes_by_round)
            ]
            if alive_values:
                mean_alive.append(sum(alive_values) / len(alive_values))

        summary = summarize_runs(case, runs)
        lnd_10pct_alive = _mean_round_when_alive_at_or_below(
            runs,
            case.node_count,
            max(1, int(0.1 * case.node_count)),
        )
        label = (
            f"c=({c1},{c2}), pkt={packet_size}, "
            f"e={initial_energy}, FND={format_round_metric(summary.get('fnd_mean'))}, "
            f"HND={format_round_metric(summary.get('hnd_mean'))}, "
            f"LND={format_round_metric(lnd_10pct_alive)}"
        )
        labels_and_series.append((label, mean_alive))
        print(
            f"[space {space_m}m, node {node_count}] {label}, "
            f"ResidualPct={summary['residual_energy_pct']:.2f}%"
        )

    return labels_and_series


def _collect_alive_nodes_sweep_100m(node_count: int) -> List[tuple[str, List[float]]]:
    return _collect_alive_nodes_sweep(100, node_count)


def run_alive_nodes_500m_1000nodes(output_dir: Path) -> None:
    labels_and_series = _collect_alive_nodes_sweep(500, 1000)
    save_alive_nodes_sweep_plot(
        output_dir,
        labels_and_series,
        node_count=1000,
        space_m=500,
        max_tick_step_cap=10000,
    )


def run_alive_nodes_500m_2000nodes(output_dir: Path) -> None:
    labels_and_series = _collect_alive_nodes_sweep(500, 2000)
    save_alive_nodes_sweep_plot(
        output_dir,
        labels_and_series,
        node_count=2000,
        space_m=500,
        max_tick_step_cap=10000,
    )


def run_alive_nodes_500m_3000nodes(output_dir: Path) -> None:
    labels_and_series = _collect_alive_nodes_sweep(500, 3000)
    save_alive_nodes_sweep_plot(
        output_dir,
        labels_and_series,
        node_count=3000,
        space_m=500,
        max_tick_step_cap=10000,
    )


def run_alive_nodes_100m_100nodes(output_dir: Path) -> None:
    labels_and_series = _collect_alive_nodes_sweep_100m(100)
    save_alive_nodes_sweep_plot(
        output_dir,
        labels_and_series,
        node_count=100,
        space_m=100,
        max_tick_step_cap=5000,
    )


def run_alive_nodes_100m_500nodes(output_dir: Path) -> None:
    labels_and_series = _collect_alive_nodes_sweep_100m(500)
    save_alive_nodes_sweep_plot(
        output_dir,
        labels_and_series,
        node_count=500,
        space_m=100,
        max_tick_step_cap=10000,
    )


def run_fnd_faceted_500m_from_logs(output_dir: Path) -> None:
    save_fnd_faceted_plot_from_logs(
        output_dir,
        [
            output_dir / "space500_to_fnd_runs10.log",
            output_dir / "space500_2000_2500_to_fnd_runs10.log",
        ],
        space_m=500,
        node_counts=(1000, 1500, 2000, 2500),
    )


def run_fnd_faceted_100m_from_logs(output_dir: Path) -> None:
    save_fnd_faceted_plot_from_logs(
        output_dir,
        [
            output_dir / "space100_to_fnd_runs10.log",
        ],
        space_m=100,
        node_counts=(100, 200, 300, 500),
    )


def run_parameter_sweep(output_dir: Path) -> None:
    node_by_space = {100: [100, 500], 500: [1000, 2000, 3000]}
    c_pairs = [(1.5, 1.5), (2.5, 0.5), (0.5, 2.5)]
    packet_sizes = [4000, 6400]
    initial_energies = [0.5, 1.0]

    for space_m, node_counts in node_by_space.items():
        labels_and_series: List[tuple[str, List[float]]] = []
        for node_count, (c1, c2), packet_size, initial_energy in product(
            node_counts, c_pairs, packet_sizes, initial_energies
        ):
            case = replace(
                SIMULATION_CASE,
                name=(
                    f"s{space_m}_n{node_count}_c1{c1}_c2{c2}"
                    f"_pkt{packet_size}_e{initial_energy}"
                ),
                node_count=node_count,
                packet_size_bits=packet_size,
                initial_energy=initial_energy,
            )
            params = replace(
                SIMULATION_PARAMS,
                width_m=float(space_m),
                height_m=float(space_m),
                depth_m=float(space_m),
                pso_c1=c1,
                pso_c2=c2,
            )

            print(
                f"Running space={space_m}m, nodes={node_count}, "
                f"c=({c1},{c2}), pkt={packet_size}, e={initial_energy}..."
            )
            runs = run_simulation_batch(
                case,
                params,
                (BASE_SEED + offset for offset in range(RUNS)),
            )

            summary = summarize_runs(case, runs)
            mean_series = _mean_residual_by_round(runs)
            label = (
                f"n={node_count}, c=({c1},{c2}), "
                f"pkt={packet_size}, e={initial_energy}, FND={format_round_metric(summary.get('fnd_mean'))}"
            )
            labels_and_series.append((label, mean_series))
            print(
                f"[space {space_m}m] {label}, "
                f"ResidualPct={summary['residual_energy_pct']:.2f}%"
            )

        save_space_sweep_plot(output_dir, space_m, labels_and_series, step=100)





def main() -> None:
    run_default_config()


def run_default_config() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    #run_all_paper_figures(output_dir)
    run_alive_nodes_100m_500nodes(output_dir)
    
