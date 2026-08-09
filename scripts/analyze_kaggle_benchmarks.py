from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = {
    "eulc_pso": "EULC–PSO",
    "eulc_ga": "EULC–GA",
    "eulc_ac_aco": "EULC–AC-ACO",
}
COLORS = {
    "eulc_pso": "#0072B2",
    "eulc_ga": "#D55E00",
    "eulc_ac_aco": "#009E73",
}
CASE_KEYS = ["distribution", "seed"]


def _find(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = _find(root, "benchmark_manifest.json")
    summary_path = _find(root, "case_summary.csv")
    rounds_path = _find(root, "round_metrics.csv")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    algorithm_id = str(manifest["algorithm_id"])
    if algorithm_id not in ALGORITHMS:
        raise ValueError(f"unsupported benchmark algorithm: {algorithm_id}")
    if not manifest.get("completed") or int(manifest.get("completed_cases", -1)) != 30:
        raise ValueError(f"benchmark is not complete for {algorithm_id}: {manifest}")
    summary = pd.read_csv(summary_path)
    rounds = pd.read_csv(rounds_path)
    if len(summary) != 30:
        raise ValueError(f"expected 30 summary cases for {algorithm_id}, found {len(summary)}")
    summary["algorithm_id"] = algorithm_id
    summary["algorithm"] = ALGORITHMS[algorithm_id]
    rounds["algorithm_id"] = algorithm_id
    rounds["algorithm"] = ALGORITHMS[algorithm_id]
    if "status" not in summary:
        summary["status"] = "completed"
    provenance = {
        "algorithm_id": algorithm_id,
        "git_commit": manifest.get("git_commit"),
        "algorithm_config_hash": manifest.get("algorithm_config_hash"),
        "experiment_spec_hash": manifest.get("experiment_spec_hash"),
        "case_summary_sha256": _sha256(summary_path),
        "round_metrics_sha256": _sha256(rounds_path),
    }
    return summary, rounds, provenance


def validate_paired_cases(summary: pd.DataFrame) -> None:
    expected = None
    for algorithm_id in ALGORITHMS:
        rows = summary.loc[summary["algorithm_id"] == algorithm_id, CASE_KEYS]
        keys = set(map(tuple, rows.itertuples(index=False, name=None)))
        if len(keys) != 30:
            raise ValueError(f"{algorithm_id} does not contain 30 unique paired cases")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError(f"paired scenario/seed mapping differs for {algorithm_id}")


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _boxplot(ax: plt.Axes, summary: pd.DataFrame, column: str, ylabel: str) -> None:
    values, labels, colors = [], [], []
    for algorithm_id, label in ALGORITHMS.items():
        series = pd.to_numeric(
            summary.loc[
                (summary["algorithm_id"] == algorithm_id)
                & (summary["status"] == "completed"),
                column,
            ],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan).dropna()
        values.append(series.to_numpy())
        labels.append(label)
        colors.append(COLORS[algorithm_id])
    artists = ax.boxplot(values, tick_labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def create_outputs(summary: pd.DataFrame, rounds: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    feasible = summary[summary["status"] == "completed"].copy()
    metrics = [
        "fnd_round", "hnd_round", "lnd_round", "final_residual_energy_j",
        "total_energy_consumed_j", "packet_delivery_ratio",
        "average_e2e_delay_s", "runtime_seconds",
    ]
    overview_rows = []
    for algorithm_id, label in ALGORITHMS.items():
        all_rows = summary[summary["algorithm_id"] == algorithm_id]
        good = feasible[feasible["algorithm_id"] == algorithm_id]
        row = {
            "algorithm_id": algorithm_id,
            "algorithm": label,
            "terminal_cases": len(all_rows),
            "feasible_cases": len(good),
            "infeasible_cases": int((all_rows["status"] != "completed").sum()),
            "feasibility_rate": len(good) / len(all_rows),
        }
        for metric in metrics:
            values = pd.to_numeric(good[metric], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            row[f"{metric}_n"] = len(values)
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            row[f"{metric}_median"] = values.median() if len(values) else np.nan
        overview_rows.append(row)
    pd.DataFrame(overview_rows).to_csv(output / "summary_by_algorithm.csv", index=False)

    distribution_rows = []
    for (algorithm_id, distribution), group in feasible.groupby(
        ["algorithm_id", "distribution"], sort=True
    ):
        row = {
            "algorithm_id": algorithm_id,
            "algorithm": ALGORITHMS[algorithm_id],
            "distribution": distribution,
            "feasible_cases": len(group),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
        distribution_rows.append(row)
    pd.DataFrame(distribution_rows).to_csv(
        output / "summary_by_distribution.csv", index=False
    )
    summary.to_csv(output / "combined_case_summary.csv", index=False)
    rounds.to_csv(output / "combined_round_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    counts = summary.groupby(["algorithm_id", "status"]).size().unstack(fill_value=0)
    x = np.arange(len(ALGORITHMS))
    completed = np.array([counts.loc[a].get("completed", 0) for a in ALGORITHMS])
    infeasible = np.array([30 - value for value in completed])
    ax.bar(x, completed, color=[COLORS[a] for a in ALGORITHMS], label="Feasible")
    ax.bar(x, infeasible, bottom=completed, color="#999999", label="Infeasible")
    ax.set_xticks(x, ALGORITHMS.values())
    ax.set_ylabel("Number of held-out cases")
    ax.set_ylim(0, 31)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output, "feasibility_by_algorithm")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for algorithm_id, label in ALGORITHMS.items():
        group = rounds[rounds["algorithm_id"] == algorithm_id]
        stats = group.groupby("round")["residual_energy_j"].agg(["mean", "std"])
        ax.plot(stats.index, stats["mean"], label=label, color=COLORS[algorithm_id])
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["std"].fillna(0),
            stats["mean"] + stats["std"].fillna(0),
            color=COLORS[algorithm_id], alpha=0.14,
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Residual energy (J)")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, output, "residual_energy_by_round")

    distributions = sorted(summary["distribution"].unique())
    fig, axes = plt.subplots(1, len(distributions), figsize=(15, 4.2), sharey=True)
    for ax, distribution in zip(np.atleast_1d(axes), distributions):
        for algorithm_id, label in ALGORITHMS.items():
            group = rounds[
                (rounds["algorithm_id"] == algorithm_id)
                & (rounds["distribution"] == distribution)
            ]
            stats = group.groupby("round")["residual_energy_j"].agg(["mean", "std"])
            ax.plot(stats.index, stats["mean"], label=label, color=COLORS[algorithm_id])
            ax.fill_between(
                stats.index, stats["mean"] - stats["std"].fillna(0),
                stats["mean"] + stats["std"].fillna(0),
                color=COLORS[algorithm_id], alpha=0.12,
            )
        ax.set_title(distribution.capitalize())
        ax.set_xlabel("Round")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Residual energy (J)")
    axes[-1].legend()
    _save(fig, output, "residual_energy_by_distribution")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, label in zip(
        axes,
        ("fnd_round", "hnd_round", "lnd_round"),
        ("FND round", "HND round", "LND round"),
    ):
        _boxplot(ax, summary, metric, label)
    _save(fig, output, "network_lifetime_events")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, metric, label in zip(
        axes.ravel(),
        ("packet_delivery_ratio", "average_e2e_delay_s", "runtime_seconds", "final_residual_energy_j"),
        ("Packet delivery ratio", "Average E2E delay (s)", "Runtime (s)", "Final residual energy (J)"),
    ):
        _boxplot(ax, summary, metric, label)
    fig.tight_layout()
    _save(fig, output, "benchmark_metric_distributions")

    notes = """# Benchmark analysis notes

- All comparisons use the same held-out distribution/seed case mapping.
- `infeasible` is a terminal algorithm outcome and is reported separately.
- Infeasible cases are not replaced with zero in energy, delay, or lifetime summaries.
- Lines show the arithmetic mean; shaded regions show plus/minus one standard deviation across available feasible cases.
- FND/HND/LND boxplots include observed event rounds only; censored/missing events are not imputed.
- This report is descriptive. No post-hoc hypothesis test was selected after inspecting results.
"""
    (output / "analysis_notes.md").write_text(notes, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired UWSN held-out benchmarks")
    parser.add_argument("--pso", type=Path, required=True)
    parser.add_argument("--ga", type=Path, required=True)
    parser.add_argument("--ac-aco", dest="ac_aco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    loaded = [load_benchmark(path) for path in (args.pso, args.ga, args.ac_aco)]
    summary = pd.concat([item[0] for item in loaded], ignore_index=True)
    rounds = pd.concat([item[1] for item in loaded], ignore_index=True)
    validate_paired_cases(summary)
    create_outputs(summary, rounds, args.output_dir)
    manifest = {
        "schema_version": 1,
        "algorithms": list(ALGORITHMS),
        "paired_cases_per_algorithm": 30,
        "provenance": [item[2] for item in loaded],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
