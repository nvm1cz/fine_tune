from __future__ import annotations

import numpy as np


SHARED_ACOUSTIC_MODEL_VERSION = "shared_stojanovic_thorp_v2"
PAPER_GEETHU_MODEL_VERSION = "paper_geethu_2017_v1"
COMPATIBILITY_REFERENCE_DISTANCE_M = 1.0e-3
COMPATIBILITY_REFERENCE_ATTENUATION_LINEAR = 1.0


def thorp_absorption_db_per_km(freq_khz: float) -> float:
    """Return the Thorp absorption coefficient in dB/km for frequency in kHz."""
    frequency = float(freq_khz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_khz must be finite and positive")
    f2 = frequency * frequency
    value = (
        0.11 * f2 / (1.0 + f2)
        + 44.0 * f2 / (4100.0 + f2)
        + 2.75e-4 * f2
        + 0.003
    )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Thorp absorption must be finite and positive")
    return float(value)


def _attenuation_core(
    distance_m: np.ndarray,
    frequency_khz: float,
    spreading_factor: float,
    reference_attenuation_linear: float,
    reference_distance_m: float,
    model_version: str,
    transmission_anomaly_db: float,
) -> np.ndarray:
    distances = np.asarray(distance_m, dtype=float)
    spreading = float(spreading_factor)
    reference_attenuation = float(reference_attenuation_linear)
    reference_distance = float(reference_distance_m)
    if np.any(~np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("distance_m must be finite and non-negative")
    if not np.isfinite(spreading) or spreading <= 0.0:
        raise ValueError("spreading_factor must be finite and positive")
    if not np.isfinite(reference_attenuation) or reference_attenuation <= 0.0:
        raise ValueError("reference_attenuation_linear must be finite and positive")
    if not np.isfinite(reference_distance) or reference_distance <= 0.0:
        raise ValueError("reference_distance_m must be finite and positive")

    effective_distance_m = np.maximum(distances, reference_distance)
    effective_distance_km = effective_distance_m / 1000.0
    alpha_db_per_km = thorp_absorption_db_per_km(frequency_khz)
    if model_version == SHARED_ACOUSTIC_MODEL_VERSION:
        spreading_term = np.power(effective_distance_km, spreading)
        absorption_term = np.power(
            10.0, alpha_db_per_km * effective_distance_km / 10.0
        )
        attenuation = reference_attenuation * spreading_term * absorption_term
    elif model_version == PAPER_GEETHU_MODEL_VERSION:
        anomaly = float(transmission_anomaly_db)
        if not np.isfinite(anomaly):
            raise ValueError("transmission_anomaly_db must be finite")
        loss_db = (
            spreading * 10.0 * np.log10(effective_distance_m)
            + alpha_db_per_km * effective_distance_km
            + anomaly
        )
        attenuation = np.power(10.0, loss_db / 10.0)
    else:
        raise ValueError(f"unsupported acoustic model version: {model_version}")
    if np.any(~np.isfinite(attenuation)) or np.any(attenuation <= 0.0):
        raise ValueError("attenuation must be finite and positive")
    return attenuation


def attenuation_linear(
    distance_m: float,
    frequency_khz: float,
    spreading_factor: float,
    reference_attenuation_linear: float = COMPATIBILITY_REFERENCE_ATTENUATION_LINEAR,
    reference_distance_m: float = COMPATIBILITY_REFERENCE_DISTANCE_M,
    model_version: str = SHARED_ACOUSTIC_MODEL_VERSION,
    transmission_anomaly_db: float = 0.0,
) -> float:
    """Return dimensionless A0*l_km^k*10^(alpha*l_km/10)."""
    value = _attenuation_core(
        np.asarray(float(distance_m)),
        frequency_khz,
        spreading_factor,
        reference_attenuation_linear,
        reference_distance_m,
        model_version,
        transmission_anomaly_db,
    )
    return float(value)


def attenuation_linear_array(
    distance_m: np.ndarray,
    frequency_khz: float,
    spreading_factor: float,
    reference_attenuation_linear: float = COMPATIBILITY_REFERENCE_ATTENUATION_LINEAR,
    reference_distance_m: float = COMPATIBILITY_REFERENCE_DISTANCE_M,
    model_version: str = SHARED_ACOUSTIC_MODEL_VERSION,
    transmission_anomaly_db: float = 0.0,
) -> np.ndarray:
    """Vectorized attenuation using the same implementation as the scalar API."""
    return _attenuation_core(
        np.asarray(distance_m, dtype=float),
        frequency_khz,
        spreading_factor,
        reference_attenuation_linear,
        reference_distance_m,
        model_version,
        transmission_anomaly_db,
    )


def transmission_loss_db(
    distance_m: float,
    frequency_khz: float,
    spreading_factor: float,
    reference_attenuation_linear: float = COMPATIBILITY_REFERENCE_ATTENUATION_LINEAR,
    reference_distance_m: float = COMPATIBILITY_REFERENCE_DISTANCE_M,
    model_version: str = SHARED_ACOUSTIC_MODEL_VERSION,
    transmission_anomaly_db: float = 0.0,
) -> float:
    """Return transmission loss in dB, exactly 10*log10(attenuation_linear)."""
    attenuation = attenuation_linear(
        distance_m,
        frequency_khz,
        spreading_factor,
        reference_attenuation_linear,
        reference_distance_m,
        model_version,
        transmission_anomaly_db,
    )
    value = 10.0 * float(np.log10(attenuation))
    if not np.isfinite(value):
        raise ValueError("transmission loss must be finite")
    return value
