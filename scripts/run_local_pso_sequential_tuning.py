from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uwsn.algorithms.config import load_algorithm_config
from uwsn.experiment_config.generation import (
    ExperimentGenerator,
    canonical_hash,
    seed_streams,
)
from uwsn.experiment_config.io import load_yaml
from uwsn.experiment_config.runner import build_simulator


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case_template(spec: dict[str, Any]) -> dict[str, Any]:
    wanted = spec["scenario"]
    cases = ExperimentGenerator(
        ROOT / "configs/experiment_sets/tuning_ofat_full.yaml"
    ).build_cases()
    case = next(
        copy.deepcopy(row)
        for row in cases
        if row["environment"]["deployment_size_m"] == wanted["deployment_size_m"]
        and row["metadata"]["density"] == wanted["density"]
        and row["environment"]["distribution"] == wanted["distribution"]
        and int(row["environment"]["packet_size_bits"]) == wanted["packet_size_bits"]
        and float(row["environment"]["initial_energy_j"]) == wanted["initial_energy_j"]
        and int(row["metadata"]["base_seed"]) == 0
    )
    if int(case["environment"]["node_count"]) != int(wanted["node_count"]):
        raise ValueError("scenario node_count does not match experiment mapping")
    case["channel"]["transmission_range_m"] = float(wanted["transmission_range_m"])
    case["execution"]["rounds"] = int(spec["optimizer"]["simulation_rounds_per_run"])
    # Parameter tuning observes the network only until the first node dies.
    # The configured round count remains a safety/censoring ceiling.
    case["execution"]["stop_on_first_dead"] = True
    case["output"]["save_convergence"] = True
    return case


def _seeded_case(template: dict[str, Any], seed: int) -> dict[str, Any]:
    case = copy.deepcopy(template)
    case["metadata"]["base_seed"] = int(seed)
    for name, value in seed_streams(seed).items():
        case["metadata"][f"{name}_seed"] = value
    case["metadata"]["config_hash"] = canonical_hash(case)
    case["metadata"]["case_id"] = f"local-pso-tune__s{seed:02d}"
    return case


def _algorithm_config(
    population: int,
    max_iterations: int,
    *,
    inertia_schedule: str,
    inertia_start: float,
    inertia_end: float,
    c1: float,
    c2: float,
    velocity_max: float,
) -> dict[str, Any]:
    config = load_algorithm_config(ROOT / "configs/algorithms/eulc_pso.yaml")
    optimizer = config["optimizer"]
    optimizer["population_size"] = int(population)
    optimizer["max_iterations"] = int(max_iterations)
    optimizer["fitness_evaluation_budget"] = int(population) * (int(max_iterations) + 1)
    optimizer["params"].update({
        "inertia": float(inertia_start),
        "inertia_schedule": inertia_schedule,
        "inertia_start": float(inertia_start),
        "inertia_end": float(inertia_end),
        "c1": float(c1),
        "c2": float(c2),
        "velocity_max": float(velocity_max),
    })
    return config


def _run_once(
    template: dict[str, Any],
    algorithm_config: dict[str, Any],
    seed: int,
    phase: str,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    simulator, execution = build_simulator(_seeded_case(template, seed), algorithm_config)
    started = time.perf_counter()
    metrics = simulator.run(
        stop_on_first_dead=True,
        max_rounds=int(execution["rounds"]),
    )
    runtime = time.perf_counter() - started
    first_refresh = [
        row for row in metrics.pso_convergence if int(row["refresh_index"]) == 1
    ]
    if not first_refresh:
        raise RuntimeError("PSO produced no convergence history")
    params = algorithm_config["optimizer"]["params"]
    summary = {
        "phase": phase,
        "label": label,
        "seed": seed,
        "population_size": algorithm_config["optimizer"]["population_size"],
        "max_iterations": algorithm_config["optimizer"]["max_iterations"],
        "inertia_schedule": params.get("inertia_schedule", "fixed"),
        "inertia_start": params.get("inertia_start", params["inertia"]),
        "inertia_end": params.get("inertia_end", params["inertia"]),
        "c1": params["c1"],
        "c2": params["c2"],
        "velocity_max": params.get("velocity_max", 1.0),
        "particle_dimension": int(first_refresh[-1]["candidate_count"]),
        "best_J": float(first_refresh[-1]["best_score"]),
        "initial_J": float(first_refresh[0]["best_score"]),
        "improvement": float(first_refresh[0]["best_score"] - first_refresh[-1]["best_score"]),
        "runtime_seconds": runtime,
        "fnd_round": metrics.fnd_round,
        "fnd_censored": metrics.fnd_round is None,
        "rounds_executed": len(metrics.residual_energy_by_round),
        "alive_nodes_at_stop": int(metrics.alive_nodes),
        "residual_energy_j": float(metrics.residual_energy),
        "packet_delivery_ratio": metrics.packet_delivery_ratio,
    }
    curve = [
        {
            "phase": phase,
            "label": label,
            "seed": seed,
            "iteration": int(row["iteration"]),
            "best_J": float(row["best_score"]),
        }
        for row in first_refresh
    ]
    return summary, curve


def _evaluate(
    template: dict[str, Any],
    phase: str,
    variants: list[tuple[str, dict[str, Any]]],
    seed_count: int,
    seed_start: int,
    workers: int,
    trial_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for candidate_index, (label, config) in enumerate(variants):
        rows = []
        jobs = (
            (template, config, seed, phase, label)
            for seed in range(seed_start, seed_start + seed_count)
        )
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                completed = executor.map(lambda_args_run_once, jobs)
                completed_rows = list(completed)
        else:
            completed_rows = [_run_once(*job) for job in jobs]
        for run_index, (summary, curve) in enumerate(completed_rows, 1):
            rows.append(summary)
            trial_rows.append(summary)
            curve_rows.extend(curve)
            print(
                f"[{phase}] {label} run={run_index}/{seed_count} seed={summary['seed']} "
                f"D={summary['particle_dimension']} J={summary['best_J']:.9g} "
                f"runtime={summary['runtime_seconds']:.2f}s",
                flush=True,
            )
        values = [float(row["best_J"]) for row in rows]
        ranking = {
            "phase": phase,
            "label": label,
            "candidate_index": candidate_index,
            "runs": len(rows),
            "mean_best_J": statistics.mean(values),
            "std_best_J": statistics.pstdev(values),
            "median_best_J": statistics.median(values),
            "mean_runtime_seconds": statistics.mean(row["runtime_seconds"] for row in rows),
            "mean_particle_dimension": statistics.mean(row["particle_dimension"] for row in rows),
        }
        rankings.append(ranking)
    # Preserve the user-provided order on an exact statistical tie. This avoids
    # falsely presenting a lexicographically smaller label as an improvement.
    rankings.sort(key=lambda row: (
        row["mean_best_J"], row["std_best_J"], row["candidate_index"]
    ))
    for index, row in enumerate(rankings, 1):
        row["rank"] = index
    winner_label = rankings[0]["label"]
    winner_config = next(config for label, config in variants if label == winner_label)
    return winner_config, rankings[0]


def lambda_args_run_once(args: tuple[Any, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pickle-safe argument adapter for local process workers."""
    return _run_once(*args)


def _plot(curves: list[dict[str, Any]], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    final = [row for row in curves if row["phase"] == "final_comparison"]
    labels = sorted({row["label"] for row in final})
    for label in labels:
        selected = [row for row in final if row["label"] == label]
        iterations = sorted({row["iteration"] for row in selected})
        means = [
            statistics.mean(row["best_J"] for row in selected if row["iteration"] == iteration)
            for iteration in iterations
        ]
        plt.plot(iterations, means, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Mean best objective J (minimize)")
    plt.title("Default vs tuned PSO convergence (30 paired seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "default_vs_tuned_convergence.png", dpi=180)
    plt.savefig(output / "default_vs_tuned_convergence.pdf")
    plt.close()


def _all_rankings(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trials:
        grouped.setdefault((str(row["phase"]), str(row["label"])), []).append(row)
    rankings = []
    for (phase, label), rows in sorted(grouped.items()):
        values = [float(row["best_J"]) for row in rows]
        rankings.append({
            "phase": phase,
            "label": label,
            "runs": len(rows),
            "mean_best_J": statistics.mean(values),
            "std_best_J": statistics.pstdev(values),
            "median_best_J": statistics.median(values),
            "mean_runtime_seconds": statistics.mean(
                float(row["runtime_seconds"]) for row in rows
            ),
            "mean_particle_dimension": statistics.mean(
                int(row["particle_dimension"]) for row in rows
            ),
        })
    return rankings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/tuning/local_pso_100_sparse_uniform_p6400_e1_r150.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local_pso_tuning")
    args = parser.parse_args()
    spec = load_yaml(args.config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    template = _case_template(spec)
    screen_seeds = int(spec["optimizer"]["screen_seeds"])
    screen_seed_start = int(spec["optimizer"].get("screen_seed_start", 0))
    final_seeds = int(spec["optimizer"]["final_seeds"])
    final_seed_start = int(
        spec["optimizer"].get("final_seed_start", screen_seed_start + screen_seeds)
    )
    workers = int(spec["optimizer"].get("workers", 1))
    trials: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    phase_winners: list[dict[str, Any]] = []

    initial_iterations = int(spec["optimizer"].get("initial_max_iterations", 200))

    def config(population: int, iterations: int, schedule: str, start: float, end: float,
               c1: float, c2: float, vmax: float) -> dict[str, Any]:
        return _algorithm_config(
            population, iterations, inertia_schedule=schedule, inertia_start=start,
            inertia_end=end, c1=c1, c2=c2, velocity_max=vmax,
        )

    particle_variants = [
        (f"N={n},iter={initial_iterations}", config(
            n, initial_iterations, "fixed", 0.9, 0.9, 2.0, 2.0, 1.0
        ))
        for n in spec["optimizer"]["particle_candidates"]
    ]
    selected, winner = _evaluate(
        template, "step0_particles", particle_variants, screen_seeds,
        screen_seed_start, workers, trials, curves
    )
    phase_winners.append(winner)
    population = int(selected["optimizer"]["population_size"])

    w_variants = [
        (f"w={start}->{end}", config(
            population, initial_iterations, "linear", start, end, 2.0, 2.0, 1.0
        ))
        for start, end in spec["optimizer"]["w_schedules"]
    ]
    selected, winner = _evaluate(
        template, "step1_inertia", w_variants, screen_seeds,
        screen_seed_start, workers, trials, curves
    )
    phase_winners.append(winner)
    selected_params = selected["optimizer"]["params"]

    c_variants = []
    for c1, c2 in spec["optimizer"]["c_pairs"]:
        if abs(float(c1) - 1.49445) < 1e-8 and abs(float(c2) - 1.49445) < 1e-8:
            candidate = config(
                population, initial_iterations, "fixed", 0.729, 0.729, c1, c2, 1.0
            )
        else:
            candidate = config(
                population, initial_iterations, selected_params["inertia_schedule"],
                selected_params["inertia_start"], selected_params["inertia_end"],
                c1, c2, 1.0,
            )
        c_variants.append((f"c1={c1},c2={c2}", candidate))
    selected, winner = _evaluate(
        template, "step2_acceleration", c_variants, screen_seeds,
        screen_seed_start, workers, trials, curves
    )
    phase_winners.append(winner)
    selected_params = selected["optimizer"]["params"]

    retest_variants = [
        (
            f"N={n},iter={iterations}",
            config(
                n, iterations, selected_params["inertia_schedule"], selected_params["inertia_start"],
                selected_params["inertia_end"], selected_params["c1"],
                selected_params["c2"], 1.0,
            ),
        )
        for n in spec["optimizer"]["particle_candidates"]
        for iterations in spec["optimizer"].get("iteration_candidates", [100, 150, 200])
    ]
    selected, winner = _evaluate(
        template, "step3_particles_retest", retest_variants, screen_seeds,
        screen_seed_start, workers, trials, curves
    )
    phase_winners.append(winner)
    selected_params = selected["optimizer"]["params"]
    population = int(selected["optimizer"]["population_size"])
    selected_iterations = int(selected["optimizer"]["max_iterations"])

    vmax_variants = [
        (
            f"Vmax={vmax}",
            config(
                population, selected_iterations, selected_params["inertia_schedule"], selected_params["inertia_start"],
                selected_params["inertia_end"], selected_params["c1"],
                selected_params["c2"], vmax,
            ),
        )
        for vmax in spec["optimizer"]["velocity_max_values"]
    ]
    tuned, winner = _evaluate(
        template, "step4_velocity", vmax_variants, screen_seeds,
        screen_seed_start, workers, trials, curves
    )
    phase_winners.append(winner)

    default = config(30, 200, "fixed", 0.9, 0.9, 2.0, 2.0, 1.0)
    final_variants = [("default", default), ("tuned", tuned)]
    _, final_winner = _evaluate(
        template, "final_comparison", final_variants, final_seeds,
        final_seed_start, workers, trials, curves
    )
    phase_winners.append(final_winner)

    _write_csv(output / "trials.csv", trials)
    _write_csv(output / "convergence.csv", curves)
    _write_csv(output / "phase_winners.csv", phase_winners)
    _write_csv(output / "all_config_rankings.csv", _all_rankings(trials))
    result = {
        "schema_version": 1,
        "scenario": spec["scenario"],
        "selection_rule": "minimize mean best_J, then minimize std best_J",
        "screen_seeds": screen_seeds,
        "screen_seed_start": screen_seed_start,
        "final_seeds": final_seeds,
        "final_seed_start": final_seed_start,
        "initial_max_iterations": initial_iterations,
        "iteration_candidates": spec["optimizer"].get(
            "iteration_candidates", [100, 150, 200]
        ),
        "default_config": default,
        "tuned_config": tuned,
        "phase_winners": phase_winners,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _plot(curves, output)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
