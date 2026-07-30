from __future__ import annotations

from .ac_aco import run_ac_aco_cluster_head_selection
from .adapters import ExistingOptimizerAdapter
from .ga import run_ga_cluster_head_selection
from .pso import run_pso_cluster_head_selection


OPTIMIZER_FUNCTIONS = {
    "pso": run_pso_cluster_head_selection,
    "ga": run_ga_cluster_head_selection,
    "ac_aco": run_ac_aco_cluster_head_selection,
}


def create_optimizer(name: str) -> ExistingOptimizerAdapter:
    canonical = name.lower().replace("-", "_")
    try:
        return ExistingOptimizerAdapter(canonical, OPTIMIZER_FUNCTIONS[canonical])
    except KeyError as exc:
        raise ValueError(f"Unknown optimizer: {name}") from exc
