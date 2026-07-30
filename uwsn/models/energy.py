from __future__ import annotations

import numpy as np

from .acoustic import (
    attenuation_linear,
    attenuation_linear_array,
    thorp_absorption_db_per_km,
)
from ..run_config import TunableParams


# Compatibility wrappers; propagation logic lives only in acoustic_propagation.py.
def attenuation(distance_m: float, params: TunableParams) -> float:
    return attenuation_linear(
        distance_m,
        params.acoustic_frequency_khz,
        params.spreading_factor,
        params.acoustic_reference_attenuation_linear,
        params.acoustic_reference_distance_m,
        params.acoustic_model_version,
        params.transmission_anomaly_db,
    )


def attenuation_array(distance_m: np.ndarray, params: TunableParams) -> np.ndarray:
    return attenuation_linear_array(
        distance_m,
        params.acoustic_frequency_khz,
        params.spreading_factor,
        params.acoustic_reference_attenuation_linear,
        params.acoustic_reference_distance_m,
        params.acoustic_model_version,
        params.transmission_anomaly_db,
    )


def transmit_energy_j(distance_m: float, bits: int, params: TunableParams) -> float:
    """Return the existing distance-based transmit-energy cost without mutating state."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    e1_j = params.electronic_energy_e1_nj * 1e-9
    tx_scale = params.energy_calibration_factor * params.transmit_power_p0 * e1_j
    return float(bits) * tx_scale * attenuation(float(distance_m), params)


def transmit_energy_j_array(
    distance_m: np.ndarray,
    bits: int,
    params: TunableParams,
) -> np.ndarray:
    """Vectorized empirical TX energy using the shared propagation model."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    e1_j = params.electronic_energy_e1_nj * 1e-9
    tx_scale = params.energy_calibration_factor * params.transmit_power_p0 * e1_j
    return float(bits) * tx_scale * attenuation_array(distance_m, params)


def receive_energy_j(bits: int, params: TunableParams) -> float:
    """Return the existing receive-energy cost without mutating state."""
    if bits < 0:
        raise ValueError("bits must be non-negative")
    return params.energy_calibration_factor * bits * params.processing_energy_nj * 1e-9


def aggregation_energy_j(bits: int, params: TunableParams) -> float:
    """Return the existing aggregation-energy cost without mutating state."""
    if bits < 0:
        raise ValueError("bits must be non-negative")
    return params.energy_calibration_factor * bits * params.electronic_energy_e1_nj * 1e-9


# Khoảng cách Euclidean giữa hai điểm trong không gian 3D
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))
