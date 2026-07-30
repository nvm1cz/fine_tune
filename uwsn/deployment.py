from __future__ import annotations

import numpy as np

from .cases import SimulationCase
from .run_config import TunableParams
from .models.energy import euclidean

# vi tri cua sink tren mat nuoc
def resolve_sink(params: TunableParams) -> np.ndarray:
    sink = params.sink_position or (params.width_m / 2.0, params.height_m / 2.0, 0.0)
    return np.asarray(sink, dtype=float)

# rải node ngẫu nhiên trong không gian 3D
def deploy_nodes(case: SimulationCase, params: TunableParams, rng: np.random.Generator) -> np.ndarray:
    distribution = str(getattr(params, "deployment_distribution", "uniform")).lower()
    dimensions = np.asarray([params.width_m, params.height_m, params.depth_m], dtype=float)
    lower = np.zeros(3, dtype=float)

    if distribution in {"uniform", "random"}:
        return rng.uniform(lower, dimensions, size=(case.node_count, 3))

    if distribution in {"gauss", "gaussian", "normal"}:
        center = dimensions / 2.0
        std_fraction = max(float(getattr(params, "gaussian_std_fraction", 0.18)), 1e-6)
        std = dimensions * std_fraction
        positions = rng.normal(loc=center, scale=std, size=(case.node_count, 3))
        return np.clip(positions, lower, dimensions)

    if distribution in {"exp", "exponential"}:
        scale_fraction = max(float(getattr(params, "exponential_scale_fraction", 0.35)), 1e-6)
        scale = dimensions * scale_fraction
        positions = rng.exponential(scale=scale, size=(case.node_count, 3))
        return np.clip(positions, lower, dimensions)

    raise ValueError(
        "Unsupported deployment_distribution. Use uniform, gaussian, or exponential."
    )

# cong thuc chia layer 
def compute_layers(positions: np.ndarray, params: TunableParams) -> np.ndarray:
    r0 = params.layer_r0_m
    space = params.layer_spacing_m
    depths = positions[:, 2]
    layers = np.ones(len(depths), dtype=int)
    mask = depths >= r0
    layers[mask] = 2 + np.floor((depths[mask] - r0) / (r0 + space)).astype(int)
    return layers

# d(j, SN) 
def compute_distances_to_sink(positions: np.ndarray, sink: np.ndarray) -> np.ndarray:
    deltas = positions - sink
    return np.sqrt(np.sum(deltas * deltas, axis=1))

#d(i, j))
def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    deltas = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))

# tinh so hang xom cua moi node
def compute_neighbor_degree(
    positions: np.ndarray,
    params: TunableParams,
    pairwise_distances: np.ndarray | None = None,
) -> np.ndarray:
    if pairwise_distances is None:
        pairwise_distances = compute_pairwise_distances(positions)
    degree = np.count_nonzero(pairwise_distances <= params.transmission_range_m, axis=1) - 1
    degree = degree.astype(float)
    max_degree = max(float(np.max(degree)), 1.0)
    return degree / max_degree
