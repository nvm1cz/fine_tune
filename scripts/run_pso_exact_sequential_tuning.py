from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_local_pso_sequential_tuning import (
    ROOT,
    _algorithm_config,
    _case_template,
    _run_once,
    lambda_args_run_once,
)
from uwsn.experiment_config.io import load_yaml, write_yaml


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def detect_plateau(values: list[float], threshold: float, patience: int) -> int | None:
    consecutive = 0
    for iteration in range(1, len(values)):
        previous, current = float(values[iteration - 1]), float(values[iteration])
        improvement = max(previous - current, 0.0) / max(abs(previous), 1e-12)
        consecutive = consecutive + 1 if improvement < threshold else 0
        if consecutive >= patience:
            return iteration
    return None


def _paired_bootstrap_ci(
    candidate: list[float], reference: list[float], *, samples: int = 10_000
) -> tuple[float, float]:
    differences = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    if not np.any(differences):
        return 0.0, 0.0
    rng = np.random.default_rng(20260817)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = np.mean(differences[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _evaluate(
    template: dict[str, Any],
    phase: str,
    variants: list[tuple[str, dict[str, Any]]],
    seeds: list[int],
    workers: int,
    trials: list[dict[str, Any]],
    curves: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows_by_label: dict[str, list[dict[str, Any]]] = {}
    config_by_label = dict(variants)
    for label, config in variants:
        jobs = [(template, config, seed, phase, label) for seed in seeds]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                completed = list(executor.map(lambda_args_run_once, jobs))
        else:
            completed = [_run_once(*job) for job in jobs]
        phase_rows = []
        for index, (summary, convergence) in enumerate(completed, 1):
            phase_rows.append(summary)
            trials.append(summary)
            curves.extend(convergence)
            print(
                f"[{phase}] {label} run={index}/{len(seeds)} seed={summary['seed']} "
                f"D={summary['particle_dimension']} J={summary['best_J']:.9g} "
                f"runtime={summary['runtime_seconds']:.2f}s",
                flush=True,
            )
        rows_by_label[label] = phase_rows

    rankings = []
    for candidate_index, (label, _) in enumerate(variants):
        rows = rows_by_label[label]
        values = [float(row["best_J"]) for row in rows]
        rankings.append({
            "phase": phase,
            "label": label,
            "candidate_index": candidate_index,
            "runs": len(rows),
            "mean_best_J": statistics.mean(values),
            "std_best_J": statistics.pstdev(values),
            "median_best_J": statistics.median(values),
            "mean_runtime_seconds": statistics.mean(float(row["runtime_seconds"]) for row in rows),
            "mean_particle_dimension": statistics.mean(int(row["particle_dimension"]) for row in rows),
        })
    minimum = min(rankings, key=lambda row: (row["mean_best_J"], row["candidate_index"]))
    reference = [float(row["best_J"]) for row in rows_by_label[minimum["label"]]]
    statistically_tied = []
    for row in rankings:
        values = [float(item["best_J"]) for item in rows_by_label[row["label"]]]
        low, high = _paired_bootstrap_ci(values, reference)
        row["paired_difference_ci95_low"] = low
        row["paired_difference_ci95_high"] = high
        row["statistically_indistinguishable_from_min_mean"] = low <= 0.0 <= high
        if row["statistically_indistinguishable_from_min_mean"]:
            statistically_tied.append(row)
    winner = min(
        statistically_tied,
        key=lambda row: (row["std_best_J"], row["mean_runtime_seconds"], row["candidate_index"]),
    )
    rankings.sort(key=lambda row: (
        row["mean_best_J"], row["std_best_J"], row["mean_runtime_seconds"], row["candidate_index"]
    ))
    for rank, row in enumerate(rankings, 1):
        row["rank_by_mean"] = rank
        row["selected"] = row["label"] == winner["label"]
    return config_by_label[winner["label"]], winner, rankings


def _mean_curve(curves: list[dict[str, Any]], phase: str, label: str) -> list[dict[str, float]]:
    selected = [row for row in curves if row["phase"] == phase and row["label"] == label]
    iterations = sorted({int(row["iteration"]) for row in selected})
    return [{
        "iteration": iteration,
        "mean_best_J": statistics.mean(
            float(row["best_J"]) for row in selected if int(row["iteration"]) == iteration
        ),
    } for iteration in iterations]


def _plateau_sensitivity(
    curves: list[dict[str, Any]], phase: str, label: str,
    settings: list[dict[str, Any]], buffer_fraction: float,
) -> list[dict[str, Any]]:
    selected = [row for row in curves if row["phase"] == phase and row["label"] == label]
    rows = []
    for setting in settings:
        per_seed = []
        for seed in sorted({int(row["seed"]) for row in selected}):
            values = [
                float(row["best_J"]) for row in sorted(
                    (item for item in selected if int(item["seed"]) == seed),
                    key=lambda item: int(item["iteration"]),
                )
            ]
            detected = detect_plateau(
                values, float(setting["relative_improvement"]), int(setting["patience"])
            )
            if detected is not None:
                per_seed.append(detected)
        median_plateau = statistics.median(per_seed) if per_seed else None
        recommended = (
            int(math.ceil(float(median_plateau) * (1.0 + buffer_fraction)))
            if median_plateau is not None else len(_mean_curve(curves, phase, label)) - 1
        )
        rows.append({
            "name": setting["name"],
            "relative_improvement": setting["relative_improvement"],
            "patience": setting["patience"],
            "detected_seed_count": len(per_seed),
            "total_seed_count": len({int(row["seed"]) for row in selected}),
            "median_plateau_iteration": median_plateau,
            "recommended_max_iterations": max(1, recommended),
        })
    recommendations = [int(row["recommended_max_iterations"]) for row in rows]
    stable = max(recommendations) - min(recommendations) <= max(2, round(0.1 * statistics.median(recommendations)))
    for row in rows:
        row["stable_across_thresholds"] = stable
    return rows


def _plot_final(curves: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    for label in ("baseline", "tuned"):
        mean = _mean_curve(curves, "step5_final", label)
        plt.plot([row["iteration"] for row in mean], [row["mean_best_J"] for row in mean], label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Mean best J (lower is better)")
    plt.title("PSO baseline vs tuned at the initial network state")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "baseline_vs_tuned_convergence.png", dpi=200)
    plt.savefig(output / "baseline_vs_tuned_convergence.pdf")
    plt.close()


def run_exact(spec_path: Path, output: Path) -> dict[str, Any]:
    spec = load_yaml(spec_path)
    template = _case_template(spec)
    template["execution"]["rounds"] = 1
    template["execution"]["stop_on_first_dead"] = False
    # This campaign measures optimizer behaviour at the initial network state.
    # Disable the round mobility update so every candidate sees the generated
    # deployment itself, rather than a time-evolved topology.
    template["execution"]["mobility_enabled"] = False
    optimizer = spec["optimizer"]
    screen_start = int(optimizer.get("screen_seed_start", 0))
    screen_seeds = list(range(screen_start, screen_start + int(optimizer["screen_seeds"])))
    final_start = int(optimizer.get("final_seed_start", 10))
    final_seeds = list(range(final_start, final_start + int(optimizer["final_seeds"])))
    workers = int(optimizer.get("workers", 1))
    trials: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []

    def config(n: int, iterations: int, schedule: str, start: float, end: float,
               c1: float, c2: float, vmax: float) -> dict[str, Any]:
        return _algorithm_config(
            n, iterations, inertia_schedule=schedule, inertia_start=start,
            inertia_end=end, c1=c1, c2=c2, velocity_max=vmax,
        )

    probe, probe_winner, rows = _evaluate(
        template, "step0_dimension_probe",
        [("probe_N20", config(20, 50, "fixed", 0.9, 0.9, 2.0, 2.0, 1.0))],
        screen_seeds, workers, trials, curves,
    )
    ranking_rows.extend(rows); winners.append(probe_winner)
    mean_dimension = float(probe_winner["mean_particle_dimension"])
    baseline_n = min(30, max(20, int(math.ceil(10.0 + 2.0 * math.sqrt(mean_dimension)))))

    w_variants = [("fixed_0.9", config(baseline_n, 50, "fixed", 0.9, 0.9, 2.0, 2.0, 1.0))]
    w_variants += [
        (f"linear_{start}_{end}", config(baseline_n, 50, "linear", start, end, 2.0, 2.0, 1.0))
        for start, end in optimizer["w_schedules"]
    ]
    selected, winner, rows = _evaluate(
        template, "step1_inertia", w_variants, screen_seeds, workers, trials, curves
    )
    ranking_rows.extend(rows); winners.append(winner)
    selected_params = selected["optimizer"]["params"]

    c_variants = [(
        f"c1_{c1}_c2_{c2}",
        config(
            baseline_n, 50, selected_params["inertia_schedule"],
            selected_params["inertia_start"], selected_params["inertia_end"],
            c1, c2, 1.0,
        ),
    ) for c1, c2 in optimizer["c_pairs"]]
    selected, winner, rows = _evaluate(
        template, "step2_acceleration", c_variants, screen_seeds, workers, trials, curves
    )
    ranking_rows.extend(rows); winners.append(winner)
    selected_params = selected["optimizer"]["params"]

    sensitivity = _plateau_sensitivity(
        curves, "step2_acceleration", winner["label"],
        optimizer["plateau_sensitivity"], float(optimizer["plateau_buffer_fraction"]),
    )
    primary_iterations = int(next(row for row in sensitivity if row["name"] == "primary")["recommended_max_iterations"])
    n_variants = [(
        f"N_{n}_iter_{primary_iterations}",
        config(
            n, primary_iterations, selected_params["inertia_schedule"],
            selected_params["inertia_start"], selected_params["inertia_end"],
            selected_params["c1"], selected_params["c2"], 1.0,
        ),
    ) for n in optimizer["population_candidates_step3"]]
    selected, winner, rows = _evaluate(
        template, "step3_population_iterations", n_variants,
        screen_seeds, workers, trials, curves,
    )
    # Required practical trade-off: below 0.1% J improvement, prefer lower runtime.
    best_mean = min(float(row["mean_best_J"]) for row in rows)
    practical_ties = [
        row for row in rows
        if (float(row["mean_best_J"]) - best_mean) / max(abs(best_mean), 1e-12) < 0.001
    ]
    practical_winner = min(practical_ties, key=lambda row: float(row["mean_runtime_seconds"]))
    if practical_winner["label"] != winner["label"]:
        selected = dict(n_variants)[practical_winner["label"]]
        winner = practical_winner
    for row in rows:
        row["selected"] = row["label"] == winner["label"]
    ranking_rows.extend(rows); winners.append(winner)
    selected_params = selected["optimizer"]["params"]
    selected_n = int(selected["optimizer"]["population_size"])
    selected_iterations = int(selected["optimizer"]["max_iterations"])

    selected_curve = [
        row for row in curves
        if row["phase"] == "step3_population_iterations" and row["label"] == winner["label"]
    ]
    diversity_values = [
        float(row["swarm_diversity_equivalent_m"])
        for seed in screen_seeds
        for row in sorted(
            (item for item in selected_curve if int(item["seed"]) == seed),
            key=lambda item: int(item["iteration"]),
        )[-int(optimizer["diversity_tail_iterations"]):]
        if row["swarm_diversity_equivalent_m"] is not None
    ]
    measured_diversity_m = statistics.mean(diversity_values)
    physical_diagonal_m = math.sqrt(sum(float(v) ** 2 for v in spec["scenario"]["deployment_size_m"]))
    diversity_threshold_m = physical_diagonal_m * float(optimizer["diversity_diagonal_fraction"])
    mean_step3 = _mean_curve(curves, "step3_population_iterations", winner["label"])
    plateau_iteration = detect_plateau(
        [row["mean_best_J"] for row in mean_step3], 0.001, 20
    )
    should_test_vmax = plateau_iteration is not None and measured_diversity_m > diversity_threshold_m
    if should_test_vmax:
        vmax_variants = [(
            f"Vmax_{vmax}",
            config(
                selected_n, selected_iterations, selected_params["inertia_schedule"],
                selected_params["inertia_start"], selected_params["inertia_end"],
                selected_params["c1"], selected_params["c2"], vmax,
            ),
        ) for vmax in optimizer["velocity_max_values"]]
        tuned, winner, rows = _evaluate(
            template, "step4_vmax", vmax_variants, screen_seeds, workers, trials, curves
        )
        ranking_rows.extend(rows); winners.append(winner)
    else:
        tuned = config(
            selected_n, selected_iterations, selected_params["inertia_schedule"],
            selected_params["inertia_start"], selected_params["inertia_end"],
            selected_params["c1"], selected_params["c2"], 1.0,
        )

    baseline = config(30, 50, "fixed", 0.9, 0.9, 2.0, 2.0, 1.0)
    _, final_winner, rows = _evaluate(
        template, "step5_final", [("baseline", baseline), ("tuned", tuned)],
        final_seeds, workers, trials, curves,
    )
    ranking_rows.extend(rows); winners.append(final_winner)

    final_convergence = []
    for label in ("baseline", "tuned"):
        selected_final = [row for row in curves if row["phase"] == "step5_final" and row["label"] == label]
        plateau_values = []
        for seed in final_seeds:
            values = [float(row["best_J"]) for row in sorted(
                (item for item in selected_final if int(item["seed"]) == seed),
                key=lambda item: int(item["iteration"]),
            )]
            found = detect_plateau(values, 0.001, 20)
            if found is not None:
                plateau_values.append(found)
        final_convergence.append({
            "label": label,
            "detected_runs": len(plateau_values),
            "mean_convergence_iteration": statistics.mean(plateau_values) if plateau_values else None,
            "median_convergence_iteration": statistics.median(plateau_values) if plateau_values else None,
        })

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "trials.csv", trials)
    _write_csv(output / "convergence.csv", curves)
    _write_csv(output / "all_step_rankings.csv", ranking_rows)
    _write_csv(output / "step_winners.csv", winners)
    _write_csv(output / "plateau_sensitivity.csv", sensitivity)
    _write_csv(output / "final_convergence_summary.csv", final_convergence)
    diversity = {
        "plateau_detected": plateau_iteration is not None,
        "plateau_iteration": plateau_iteration,
        "tail_iterations": int(optimizer["diversity_tail_iterations"]),
        "mean_tail_diversity_equivalent_m": measured_diversity_m,
        "physical_diagonal_m": physical_diagonal_m,
        "threshold_m": diversity_threshold_m,
        "exceeds_threshold": measured_diversity_m > diversity_threshold_m,
        "vmax_sensitivity_executed": should_test_vmax,
    }
    (output / "diversity_diagnostic.json").write_text(
        json.dumps(diversity, indent=2) + "\n", encoding="utf-8"
    )
    write_yaml(output / "best_config.yaml", tuned)
    result = {
        "schema_version": 3,
        "method_version": "exact_sequential_pso_initial_state_v1",
        "scenario": spec["scenario"],
        "scope": "optimizer tuning at initial network state only",
        "lifetime_validation_required": True,
        "selection_metric": "mean best_J (minimize)",
        "baseline_population_from_formula": baseline_n,
        "mean_observed_particle_dimension": mean_dimension,
        "plateau_sensitivity": sensitivity,
        "diversity": diversity,
        "baseline_config": baseline,
        "tuned_config": tuned,
        "final_winner": final_winner,
        "step_winners": winners,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tuned_optimizer = tuned["optimizer"]
    tuned_params = tuned_optimizer["params"]
    explanation = [
        "# PSO Fine-Tuning Selection Evidence",
        "",
        f"Scenario: `{spec['name']}`.",
        "",
        "This is optimizer tuning at the initial network state. It is not network-lifetime validation.",
        "",
        f"- Observed mean particle dimension D: {mean_dimension:.6g}.",
        f"- Selected population: {tuned_optimizer['population_size']}.",
        f"- Selected max iterations: {tuned_optimizer['max_iterations']}.",
        f"- Selected inertia: {tuned_params['inertia_schedule']} "
        f"({tuned_params['inertia_start']} -> {tuned_params['inertia_end']}).",
        f"- Selected c1/c2: {tuned_params['c1']} / {tuned_params['c2']}.",
        f"- Selected Vmax: {tuned_params['velocity_max']}.",
        f"- Measured final-20 diversity equivalent: {measured_diversity_m:.9g} m; "
        f"threshold: {diversity_threshold_m:.9g} m.",
        f"- Vmax sensitivity executed: {should_test_vmax}.",
        "",
        "## Numerical selection evidence",
        "",
    ]
    explanation.extend(
        f"- {row['phase']}: `{row['label']}`; mean best J = "
        f"{float(row['mean_best_J']):.12g}, std = {float(row['std_best_J']):.12g}, "
        f"runtime/run = {float(row['mean_runtime_seconds']):.6g} s."
        for row in winners
    )
    explanation.extend([
        "",
        "Full numerical evidence is stored in `all_step_rankings.csv`, "
        "`plateau_sensitivity.csv`, and `diversity_diagnostic.json`.",
        "FND/network lifetime must be validated in a separate simulation and is not "
        "a selection criterion in this campaign.",
    ])
    (output / "selection_explanation.md").write_text("\n".join(explanation) + "\n", encoding="utf-8")
    _plot_final(curves, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_exact(args.config, args.output_dir)


if __name__ == "__main__":
    main()
