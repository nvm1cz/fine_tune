from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List

from .cases import SimulationCase
from .metrics import RunMetrics
from .run_config import TunableParams
from .simulator import UWSNSimulator


def _requested_worker_count(workers: int | None, job_count: int) -> int:
    if workers is not None:
        return max(1, min(int(workers), job_count))

    env_value = os.environ.get("UWSN_PARALLEL_RUNS")
    if not env_value:
        return 1

    try:
        requested = int(env_value)
    except ValueError:
        return 1

    cpu_count = os.cpu_count() or 1
    return max(1, min(requested, job_count, cpu_count))


def _run_one_simulation(args: tuple[SimulationCase, TunableParams, int, bool, float | None, int | None]) -> RunMetrics:
    case, params, seed, stop_on_first_dead, min_alive_ratio, max_rounds = args
    sim = UWSNSimulator(case, params, seed, verbose=False)
    return sim.run(
        stop_on_first_dead=stop_on_first_dead,
        min_alive_ratio=min_alive_ratio,
        max_rounds=max_rounds,
    )


def run_simulation_batch(
    case: SimulationCase,
    params: TunableParams,
    seeds: Iterable[int],
    *,
    stop_on_first_dead: bool = True,
    min_alive_ratio: float | None = None,
    max_rounds: int | None = None,
    workers: int | None = None,
) -> List[RunMetrics]:
    seed_list = list(seeds)
    if not seed_list:
        return []

    jobs = [
        (case, params, seed, stop_on_first_dead, min_alive_ratio, max_rounds)
        for seed in seed_list
    ]
    max_workers = _requested_worker_count(workers, len(jobs))
    if max_workers <= 1 or len(jobs) == 1:
        return [_run_one_simulation(job) for job in jobs]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run_one_simulation, jobs))
