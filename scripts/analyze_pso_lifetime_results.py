from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SCENARIO_PATTERN = re.compile(
    r"(?P<density>[a-z]+)_n(?P<nodes>\d+)_p(?P<packet>\d+)_e(?P<energy_int>\d+)p(?P<energy_frac>\d+)"
)
VARIANT_LABELS = {
    "baseline": "PSO baseline",
    "selected": "PSO selected",
    "energy_maintenance": "PSO energy+beam",
}
COLORS = {
    "baseline": "#4C78A8",
    "selected": "#F58518",
    "energy_maintenance": "#54A24B",
}


def _scenario_metadata(name: str) -> dict[str, Any]:
    match = SCENARIO_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid scenario directory: {name}")
    groups = match.groupdict()
    return {
        "density": groups["density"],
        "node_count": int(groups["nodes"]),
        "packet_size_bits": int(groups["packet"]),
        "initial_energy_j": float(f"{groups['energy_int']}.{groups['energy_frac']}"),
    }


def _bootstrap_ci(values: np.ndarray, samples: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(20260823)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _paired_test(baseline: np.ndarray, selected: np.ndarray) -> tuple[float, float, float]:
    differences = selected - baseline
    if np.allclose(differences, 0.0):
        return 1.0, 0.0, 0.0
    result = wilcoxon(selected, baseline, zero_method="wilcox", alternative="two-sided")
    positive = np.count_nonzero(differences > 0)
    negative = np.count_nonzero(differences < 0)
    rank_biserial_sign = (positive - negative) / max(positive + negative, 1)
    return float(result.pvalue), float(rank_biserial_sign), float(np.median(differences))


def _load(input_dir: Path, density: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    rounds = []
    scenario_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    for scenario_dir in scenario_dirs:
        metadata = _scenario_metadata(scenario_dir.name)
        if metadata["density"] != density:
            continue
        variant_names = sorted(
            path.name for path in scenario_dir.iterdir()
            if path.is_dir() and (path / "case_summary.csv").exists()
        )
        for variant in variant_names:
            variant_dir = scenario_dir / variant
            manifest = json.loads((variant_dir / "benchmark_manifest.json").read_text())
            if not manifest.get("completed") or int(manifest["completed_cases"]) != 30:
                raise ValueError(f"incomplete job: {scenario_dir.name}/{variant}")
            summary = pd.read_csv(variant_dir / "case_summary.csv")
            round_frame = pd.read_csv(variant_dir / "round_metrics.csv")
            if len(summary) != 30 or summary["seed"].nunique() != 30:
                raise ValueError(f"expected 30 unique seeds: {scenario_dir.name}/{variant}")
            summary = summary.assign(
                scenario_id=scenario_dir.name, variant=variant, **metadata
            )
            round_frame = round_frame.assign(
                scenario_id=scenario_dir.name, variant=variant, **metadata
            )
            summaries.append(summary)
            rounds.append(round_frame)
    summary = pd.concat(summaries, ignore_index=True)
    round_frame = pd.concat(rounds, ignore_index=True)
    expected_rows = summary["scenario_id"].nunique() * summary["variant"].nunique() * 30
    if summary["scenario_id"].nunique() != 4 or len(summary) != expected_rows:
        raise ValueError(
            f"expected four scenarios and {expected_rows} summary rows"
        )
    return summary, round_frame


def _validate(summary: pd.DataFrame) -> dict[str, Any]:
    complete = summary[summary["status"] == "completed"]
    violations = complete[
        (complete["fnd_round"] > complete["hnd_round"])
        | (complete["hnd_round"] > complete["lnd_round"])
    ]
    if not violations.empty:
        raise ValueError(f"FND/HND/LND ordering violated in {len(violations)} rows")
    return {
        "rows": len(summary),
        "completed_rows": len(complete),
        "failed_rows": int(len(summary) - len(complete)),
        "event_order_violations": len(violations),
        "missing_fnd": int(summary["fnd_round"].isna().sum()),
        "missing_hnd": int(summary["hnd_round"].isna().sum()),
        "missing_lnd": int(summary["lnd_round"].isna().sum()),
    }


def _summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "fnd_round": "FND",
        "hnd_round": "HND",
        "lnd_round": "LND",
        "final_residual_energy_j": "final_residual_energy_j",
        "packet_delivery_ratio": "packet_delivery_ratio",
        "average_e2e_delay_s": "average_e2e_delay_s",
        "runtime_seconds": "runtime_seconds",
    }
    rows = []
    group_columns = [
        "scenario_id", "density", "node_count", "packet_size_bits",
        "initial_energy_j", "variant",
    ]
    for keys, group in summary.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
        row["algorithm"] = VARIANT_LABELS.get(
            row["variant"], str(row["variant"]).replace("_", " ").title()
        )
        row["max_rounds"] = 500
        row["n_seeds"] = len(group)
        for source, target in metrics.items():
            row[f"{target}_mean"] = group[source].mean()
            row[f"{target}_std"] = group[source].std(ddof=1)
            row[f"{target}_median"] = group[source].median()
            row[f"{target}_censored"] = int(group[source].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["packet_size_bits", "initial_energy_j", "variant"],
        ascending=[False, False, True],
    )


def _paired_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "fnd_round", "hnd_round", "lnd_round", "final_residual_energy_j",
        "packet_delivery_ratio", "average_e2e_delay_s", "runtime_seconds",
    ]
    rows = []
    if not {"baseline", "selected"}.issubset(set(summary["variant"])):
        return pd.DataFrame(columns=[
            "scenario_id", "metric", "n_pairs", "baseline_mean",
            "selected_mean", "mean_difference_selected_minus_baseline",
            "difference_ci95_low", "difference_ci95_high", "median_difference",
            "wilcoxon_p_value_uncorrected", "paired_direction_effect",
        ])
    for scenario_id, scenario in summary.groupby("scenario_id", sort=True):
        baseline = scenario[scenario["variant"] == "baseline"].set_index("seed")
        selected = scenario[scenario["variant"] == "selected"].set_index("seed")
        if not baseline.index.equals(selected.index):
            raise ValueError(f"unpaired seeds: {scenario_id}")
        for metric in metrics:
            paired = pd.concat(
                [baseline[metric].rename("baseline"), selected[metric].rename("selected")],
                axis=1,
            ).dropna()
            differences = (paired["selected"] - paired["baseline"]).to_numpy(float)
            ci_low, ci_high = _bootstrap_ci(differences)
            p_value, rank_sign, median_difference = _paired_test(
                paired["baseline"].to_numpy(float), paired["selected"].to_numpy(float)
            )
            rows.append({
                "scenario_id": scenario_id,
                "metric": metric,
                "n_pairs": len(paired),
                "baseline_mean": paired["baseline"].mean(),
                "selected_mean": paired["selected"].mean(),
                "mean_difference_selected_minus_baseline": differences.mean(),
                "difference_ci95_low": ci_low,
                "difference_ci95_high": ci_high,
                "median_difference": median_difference,
                "wilcoxon_p_value_uncorrected": p_value,
                "paired_direction_effect": rank_sign,
            })
    return pd.DataFrame(rows)


def _extended_rounds(rounds: pd.DataFrame, max_round: int = 500) -> pd.DataFrame:
    columns = [
        "residual_energy_j", "energy_consumed_j", "alive_nodes", "dead_nodes",
        "packets_delivered_cumulative",
    ]
    frames = []
    keys = ["scenario_id", "variant", "seed"]
    for values, group in rounds.groupby(keys, sort=False):
        group = group.sort_values("round").set_index("round")
        index = pd.Index(range(1, max_round + 1), name="round")
        extended = group[columns].reindex(index).ffill()
        for key, value in zip(keys, values):
            extended[key] = value
        frames.append(extended.reset_index())
    return pd.concat(frames, ignore_index=True)


def _scenario_title(row: pd.Series | dict[str, Any]) -> str:
    return f"{int(row['packet_size_bits'])} bits, {float(row['initial_energy_j']):.1f} J"


def _plot_round_metric(
    extended: pd.DataFrame, summary: pd.DataFrame, metric: str,
    ylabel: str, output: Path, ylim: tuple[float, float],
) -> None:
    scenarios = list(summary.sort_values(
        ["packet_size_bits", "initial_energy_j"], ascending=[False, False]
    )["scenario_id"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    variants = list(summary["variant"].drop_duplicates())
    for ax, scenario_id in zip(axes.flat, scenarios):
        scenario = extended[extended["scenario_id"] == scenario_id]
        metadata = summary[summary["scenario_id"] == scenario_id].iloc[0]
        for variant in variants:
            values = scenario[scenario["variant"] == variant]
            grouped = values.groupby("round")[metric]
            mean = grouped.mean()
            sem = grouped.sem().fillna(0.0)
            ci = 1.96 * sem
            color = COLORS.get(variant, "#777777")
            label = VARIANT_LABELS.get(variant, variant.replace("_", " ").title())
            ax.plot(mean.index, mean, color=color, lw=1.8, label=label)
            ax.fill_between(mean.index, mean - ci, mean + ci, color=color, alpha=0.16)
        ax.set_title(_scenario_title(metadata))
        ax.set_xlim(1, 500)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Round")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    density_name = str(summary["density"].iloc[0]).title()
    fig.suptitle(f"{density_name} UWSN: {ylabel} by round (mean ± 95% CI)", y=0.995)
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.967),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_lifetime(summary: pd.DataFrame, output: Path) -> None:
    scenarios = list(summary.sort_values(
        ["packet_size_bits", "initial_energy_j"], ascending=[False, False]
    )["scenario_id"].drop_duplicates())
    max_event = float(summary[["fnd_round", "hnd_round", "lnd_round"]].max().max())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    x = np.arange(3)
    variants = list(summary["variant"].drop_duplicates())
    width = min(0.7 / max(len(variants), 1), 0.5)
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * width
    for ax, scenario_id in zip(axes.flat, scenarios):
        scenario = summary[summary["scenario_id"] == scenario_id]
        metadata = scenario.iloc[0]
        for offset, variant in zip(offsets, variants):
            group = scenario[scenario["variant"] == variant]
            means = [group[column].mean() for column in ("fnd_round", "hnd_round", "lnd_round")]
            stds = [group[column].std(ddof=1) for column in ("fnd_round", "hnd_round", "lnd_round")]
            ax.bar(
                x + offset, means, width, yerr=stds, capsize=3,
                color=COLORS.get(variant, "#777777"),
                label=VARIANT_LABELS.get(variant, variant.replace("_", " ").title()),
            )
        ax.set_title(_scenario_title(metadata))
        ax.set_xticks(x, ["FND", "HND", "LND"])
        ax.set_ylim(0, math.ceil(max_event * 1.18 / 10) * 10)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Round")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    density_name = str(summary["density"].iloc[0]).title()
    fig.suptitle(f"{density_name} UWSN network lifetime events (mean ± SD)", y=0.995)
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.967),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_distributions(summary: pd.DataFrame, output: Path) -> None:
    scenarios = list(summary.sort_values(
        ["packet_size_bits", "initial_energy_j"], ascending=[False, False]
    )["scenario_id"].drop_duplicates())
    metrics = [
        ("packet_delivery_ratio", "Packet delivery ratio"),
        ("average_e2e_delay_s", "Average E2E delay (s)"),
        ("runtime_seconds", "Runtime per seed (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    positions = []
    labels = []
    data_by_metric = {metric: [] for metric, _ in metrics}
    colors = []
    position = 1
    variants = list(summary["variant"].drop_duplicates())
    for scenario_id in scenarios:
        metadata = summary[summary["scenario_id"] == scenario_id].iloc[0]
        short = f"{int(metadata['packet_size_bits'])}\n{metadata['initial_energy_j']:.1f}J"
        for variant in variants:
            group = summary[(summary["scenario_id"] == scenario_id) & (summary["variant"] == variant)]
            for metric, _ in metrics:
                data_by_metric[metric].append(group[metric].dropna().to_numpy())
            positions.append(position)
            abbreviation = {
                "baseline": "B", "selected": "S", "energy_maintenance": "E+B"
            }.get(variant, variant[:3].upper())
            labels.append(f"{short}\n{abbreviation}")
            colors.append(COLORS.get(variant, "#777777"))
            position += 1
        position += 0.6
    for ax, (metric, ylabel) in zip(axes, metrics):
        boxes = ax.boxplot(data_by_metric[metric], positions=positions, widths=0.55,
                           patch_artist=True, showmeans=True)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(positions, labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    density_name = str(summary["density"].iloc[0]).title()
    fig.suptitle(f"{density_name} UWSN outcome distributions across 30 seeds")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Algorithm | Initial energy (J) | Packet size (bits) | Max rounds | FND | HND | LND |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        event = lambda name: f"{row[f'{name}_mean']:.2f} ± {row[f'{name}_std']:.2f}"
        lines.append(
            f"| {row['algorithm']} | {row['initial_energy_j']:.1f} | "
            f"{int(row['packet_size_bits'])} | 500 | {event('FND')} | "
            f"{event('HND')} | {event('LND')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_table(table: pd.DataFrame, output: Path) -> None:
    cells = []
    for _, row in table.iterrows():
        cells.append([
            row["algorithm"], f"{row['initial_energy_j']:.1f}",
            str(int(row["packet_size_bits"])), "500",
            f"{row['FND_mean']:.2f} ± {row['FND_std']:.2f}",
            f"{row['HND_mean']:.2f} ± {row['HND_std']:.2f}",
            f"{row['LND_mean']:.2f} ± {row['LND_std']:.2f}",
        ])
    columns = ["Algorithm", "Initial energy (J)", "Packet size (bits)",
               "Max rounds", "FND", "HND", "LND"]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")
    rendered = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.5)
    for (row, _), cell in rendered.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E8EEF7")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def analyze(input_dir: Path, density: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary, rounds = _load(input_dir, density)
    quality = _validate(summary)
    table = _summary_table(summary)
    paired = _paired_comparisons(summary)
    extended = _extended_rounds(rounds)
    summary.to_csv(output / f"{density}_all_seed_results.csv", index=False)
    table.to_csv(output / f"{density}_lifetime_summary.csv", index=False)
    paired.to_csv(output / f"{density}_paired_comparisons.csv", index=False)
    _write_markdown_table(table, output / f"{density}_lifetime_table.md")
    _plot_table(table, output / f"{density}_lifetime_table")
    _plot_lifetime(summary, output / f"{density}_network_lifetime_events")
    residual_max = float(extended["residual_energy_j"].max()) * 1.03
    _plot_round_metric(
        extended, summary, "residual_energy_j", "Residual energy (J)",
        output / f"{density}_residual_energy_by_round", (0.0, residual_max),
    )
    node_count = int(summary["node_count"].max())
    _plot_round_metric(
        extended, summary, "dead_nodes", "Dead nodes",
        output / f"{density}_dead_nodes_by_round", (0.0, float(node_count)),
    )
    _plot_distributions(summary, output / f"{density}_outcome_distributions")
    report = {
        "schema_version": 1,
        "density": density,
        "quality": quality,
        "scenario_count": int(summary["scenario_id"].nunique()),
        "variant_count": int(summary["variant"].nunique()),
        "paired_seed_count": 30,
        "max_rounds": 500,
        "notes": [
            "FND/HND/LND are summarized as mean ± sample SD across paired seeds.",
            "Wilcoxon p-values are exploratory and uncorrected for multiple comparisons.",
            "Post-LND round curves carry the terminal network state forward to round 500.",
            "All plots of the same metric use shared axis limits across scenarios.",
        ],
    }
    (output / f"{density}_analysis_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        f"# {density.title()} PSO Lifetime Analysis",
        "",
        f"- Complete jobs: {summary['scenario_id'].nunique() * summary['variant'].nunique()}; "
        f"paired seeds: 30; summary rows: {quality['rows']}.",
        f"- FND/HND/LND ordering violations: {quality['event_order_violations']}.",
        f"- Missing FND/HND/LND: {quality['missing_fnd']}/"
        f"{quality['missing_hnd']}/{quality['missing_lnd']}.",
        "- Paired Wilcoxon comparisons are emitted only when both baseline and selected variants are present.",
        "- Lifetime curves retain the final state after LND through round 500 for a common x-axis.",
    ]
    (output / f"{density}_analysis_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--density", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input_dir, args.density, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
