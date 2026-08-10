from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ENERGY_FIELDS = [
    "member_tx_energy_j", "ch_rx_energy_j", "aggregation_energy_j",
    "routing_tx_energy_j", "routing_rx_energy_j", "retransmission_energy_j",
    "control_energy_j",
]


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def _first_dead_rows(rounds: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_fields = ["scenario_id", "initial_energy_j", "packet_size_bits", "algorithm", "seed"]
    for keys, group in rounds.groupby(group_fields, sort=False):
        scenario_id, initial_energy, packet_bits, algorithm, seed = keys
        dead = group[_bool_series(group["fnd_reached"]).fillna(False)]
        if dead.empty:
            continue
        fnd_round = int(dead["round"].min())
        trace = nodes[(nodes.scenario_id == scenario_id) & (nodes.algorithm == algorithm)
            & (nodes.seed == seed)]
        alive_flags = _bool_series(trace["is_alive"]).fillna(False)
        at_fnd = trace[(trace["round"] == fnd_round) & (~alive_flags)]
        previous_dead = set(trace[(trace["round"] < fnd_round) & (~alive_flags)].node_id)
        newly_dead = sorted(set(at_fnd.node_id).difference(previous_dead))
        for node_id in newly_dead:
            history = trace[(trace.node_id == node_id) & (trace["round"] <= fnd_round)]
            relay = _bool_series(history["is_relay_if_available"])
            rows.append({
                "scenario_id": scenario_id, "initial_energy_j": initial_energy,
                "packet_size_bits": int(packet_bits),
                "algorithm": algorithm, "seed": int(seed), "first_dead_node_id": int(node_id),
                "first_dead_round": fnd_round,
                "ch_rounds_before_death": int(_bool_series(history["is_ch"]).fillna(False).sum()),
                "relay_rounds_before_death": int(relay.fillna(False).sum()) if relay.notna().any() else np.nan,
                "cumulative_energy_consumed_j": float(history["energy_consumed_this_round_j"].sum()),
                "residual_energy_trace": ";".join(
                    f"{int(row['round'])}:{float(row['residual_energy_j']):.12g}"
                    for _, row in history.iterrows()
                ),
            })
    return pd.DataFrame(rows)


def analyze(round_path: Path, node_path: Path, output: Path) -> None:
    rounds, nodes = pd.read_csv(round_path), pd.read_csv(node_path)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    group_fields = ["scenario_id", "initial_energy_j", "packet_size_bits", "algorithm", "seed"]
    for keys, group in rounds.groupby(group_fields, sort=False):
        scenario_id, initial_energy, packet_bits, algorithm, seed = keys
        dead = group[_bool_series(group["fnd_reached"]).fillna(False)]
        fnd = int(dead["round"].min()) if not dead.empty else np.nan
        pre = group[group["round"] <= fnd] if np.isfinite(fnd) else group
        correlation = pre[["objective_J", "energy_cv"]].dropna().corr().iloc[0, 1] if pre["objective_J"].notna().sum() > 1 else np.nan
        summaries.append({
            "scenario_id": scenario_id, "initial_energy_j": initial_energy,
            "packet_size_bits": int(packet_bits),
            "algorithm": algorithm, "seed": int(seed), "fnd_round": fnd,
            "final_alive_nodes": int(group.iloc[-1].alive_nodes),
            "min_energy_at_fnd_or_end_j": float(pre.iloc[-1].min_residual_energy_j),
            "mean_energy_at_fnd_or_end_j": float(pre.iloc[-1].mean_residual_energy_j),
            "min_drop_vs_mean_drop_j": float((pre.iloc[0].mean_residual_energy_j - pre.iloc[-1].mean_residual_energy_j)
                - (pre.iloc[0].min_residual_energy_j - pre.iloc[-1].min_residual_energy_j)),
            "energy_cv_at_fnd_or_end": float(pre.iloc[-1].energy_cv),
            "objective_J_energy_cv_correlation": correlation,
            "total_energy_consumed_j": float(group.total_round_energy_j.sum()),
            "mean_packet_delivery_ratio": float(group.packet_delivery_ratio.mean()),
            "mean_e2e_delay_s": float(group.average_e2e_delay_s.mean()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "diagnostic_summary.csv", index=False)

    breakdown = rounds.groupby(
        ["scenario_id", "initial_energy_j", "packet_size_bits", "algorithm"], sort=False
    )[ENERGY_FIELDS + ["total_round_energy_j"]].sum().reset_index()
    for field in ENERGY_FIELDS:
        breakdown[field.replace("_j", "_share")] = np.where(
            breakdown.total_round_energy_j > 0,
            breakdown[field] / breakdown.total_round_energy_j,
            np.nan,
        )
    breakdown.to_csv(output / "energy_breakdown.csv", index=False)
    first_dead = _first_dead_rows(rounds, nodes)
    first_dead.to_csv(output / "first_dead_node_analysis.csv", index=False)

    contributor_totals = rounds[ENERGY_FIELDS].sum().sort_values(ascending=False)
    top = contributor_totals.head(3)
    report = [
        "# Early-FND diagnostic report", "",
        "This report describes associations in the observed diagnostic data; it does not establish causality.", "",
        "## FND comparison", "",
        "Results are separated by the four paper cases; no energy/packet-size case is pooled.", "",
        "## Energy balance before FND", "",
    ]
    for row in summaries:
        report.append(
            f"- {row['scenario_id']} / {row['algorithm']} seed {row['seed']}: FND={row['fnd_round']}, "
            f"min={row['min_energy_at_fnd_or_end_j']:.6g} J, "
            f"mean={row['mean_energy_at_fnd_or_end_j']:.6g} J, "
            f"CV={row['energy_cv_at_fnd_or_end']:.6g}, "
            f"corr(J,CV)={row['objective_J_energy_cv_correlation']}."
        )
    report.extend(["", "## Largest observed energy contributors", ""])
    report.extend(f"{index + 1}. {name}: {value:.12g} J" for index, (name, value) in enumerate(top.items()))
    report.extend(["", "## Interpretation guide applied cautiously", "",
        "- If pure EULC and all hybrids show similarly early FND with fast mean-energy loss, the data are consistent with a shared energy/channel or simulation-assumption contributor.",
        "- If mean energy remains comparatively high while minimum energy collapses, the data are consistent with shared energy imbalance or routing concentration.",
        "- If pure EULC survives materially longer than hybrids, inspect optimizer-objective/routing integration; this remains an association.",
        "- A dominant retransmission, control, or routing share respectively supports those mechanisms as strong contributors, not proven sole causes.",
        "", "## Questions answered", "",
        "A–F are represented by diagnostic_summary.csv, energy_breakdown.csv, and first_dead_node_analysis.csv. "
        "Per-node component energy is intentionally absent because the simulator does not persist direct per-node component counters.",
        "", "## Proposed ablation plan (not executed)", "",
        "1. Full model reference.",
        "2. Ideal channel with noise/PER/retransmission disabled.",
        "3. Delay accounting disabled.",
        "4. Mobility disabled.",
        "5. Control/reclustering overhead isolated.",
    ])
    (output / "diagnostic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze early-FND diagnostic CSV files")
    parser.add_argument("--round-csv", type=Path, required=True)
    parser.add_argument("--node-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.round_csv, args.node_csv, args.output_dir)


if __name__ == "__main__":
    main()
