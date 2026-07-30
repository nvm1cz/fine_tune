from __future__ import annotations

from dataclasses import dataclass
from math import erfc

import numpy as np

from .acoustic import (
    attenuation_linear,
    thorp_absorption_db_per_km,
    transmission_loss_db,
)
from ..run_config import TunableParams
from .units import db_to_linear, hz_to_khz, khz_to_hz, linear_to_db


@dataclass(frozen=True)
class AmbientNoise:
    turbulence_db: float
    shipping_db: float
    wind_db: float
    thermal_db: float
    total_db: float


def turbulence_noise_db(freq_khz: float) -> float:
    f = _positive_frequency(freq_khz)
    return 17.0 - 30.0 * float(np.log10(f))


def shipping_noise_db(freq_khz: float, shipping_factor: float) -> float:
    f = _positive_frequency(freq_khz)
    epsilon = min(max(float(shipping_factor), 0.0), 1.0)
    return 40.0 + 20.0 * (epsilon - 0.5) + 26.0 * np.log10(f) - 60.0 * np.log10(f + 0.03)


def wind_noise_db(
    freq_khz: float,
    wind_speed_mps: float,
    formula_mode: str = "standard_wenz",
) -> float:
    f = _positive_frequency(freq_khz)
    wind = max(float(wind_speed_mps), 0.0)
    sign = -1.0 if formula_mode == "standard_wenz" else 1.0
    if formula_mode not in {"standard_wenz", "paper_exact"}:
        raise ValueError("noise_formula_mode must be standard_wenz or paper_exact")
    return (
        50.0 + 7.5 * np.sqrt(wind) + 20.0 * np.log10(f)
        + sign * 40.0 * np.log10(f + 0.4)
    )


def thermal_noise_db(freq_khz: float) -> float:
    f = _positive_frequency(freq_khz)
    return -15.0 + 20.0 * float(np.log10(f))


def ambient_noise(
    freq_khz: float,
    shipping_factor: float,
    wind_speed_mps: float,
    noise_model: str = "component_sum",
    formula_mode: str = "standard_wenz",
) -> AmbientNoise:
    nt = turbulence_noise_db(freq_khz)
    ns = shipping_noise_db(freq_khz, shipping_factor)
    nw = wind_noise_db(freq_khz, wind_speed_mps, formula_mode)
    nth = thermal_noise_db(freq_khz)
    if noise_model == "component_sum":
        total_db = linear_to_db(
            sum(db_to_linear(value) for value in (nt, ns, nw, nth))
        )
    elif noise_model == "paper_approximation":
        total_db = 50.0 - 18.0 * float(np.log10(_positive_frequency(freq_khz)))
    else:
        raise ValueError("noise_model must be component_sum or paper_approximation")
    return AmbientNoise(
        turbulence_db=nt,
        shipping_db=ns,
        wind_db=nw,
        thermal_db=nth,
        total_db=total_db,
    )


def snr_linear(signal_power: float, noise_power: float) -> float:
    if signal_power < 0.0:
        raise ValueError("signal_power must be non-negative")
    if noise_power <= 0.0:
        raise ValueError("noise_power must be positive")
    return float(signal_power) / float(noise_power)


def shannon_capacity_bps(bandwidth_hz: float, snr: float) -> float:
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be positive")
    if snr < 0.0:
        raise ValueError("snr must be non-negative")
    return float(bandwidth_hz) * float(np.log2(1.0 + float(snr)))


def ber_from_snr_linear(snr: float) -> float:
    value = float(snr)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("snr_linear must be finite and non-negative")
    ber = 0.5 * (1.0 - float(np.sqrt(value / (1.0 + value))))
    return float(np.clip(ber, 0.0, 0.5))


def awgn_bpsk_ber_from_snr_linear(snr: float) -> float:
    """Return coherent-BPSK BER for a non-fading AWGN channel."""
    value = float(snr)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("snr_linear must be finite and non-negative")
    return float(np.clip(0.5 * erfc(float(np.sqrt(value))), 0.0, 0.5))


def rician_ber_from_snr_linear(snr: float, rician_k_linear: float) -> float:
    gamma = float(snr)
    k_factor = float(rician_k_linear)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("snr_linear must be finite and non-negative")
    if not np.isfinite(k_factor) or k_factor < 0.0:
        raise ValueError("rician_k_linear must be finite and non-negative")
    denominator = k_factor + gamma
    argument = 0.0 if denominator == 0.0 else np.sqrt(k_factor * gamma / denominator)
    return float(np.clip(0.5 * erfc(float(argument)), 0.0, 0.5))


def packet_success_probability(ber: float, packet_bits: int, epsilon: float = 1e-15) -> float:
    bit_error_rate = float(ber)
    bits = int(packet_bits)
    if not 0.0 <= bit_error_rate <= 0.5:
        raise ValueError("ber must be in [0, 0.5]")
    if bits <= 0:
        raise ValueError("packet_bits must be positive")
    log_success = bits * float(np.log1p(-bit_error_rate))
    return float(np.clip(np.exp(log_success), 0.0, 1.0))


def channel_quality(
    distance_m: float,
    freq_khz: float,
    packet_bits: int,
    params: TunableParams,
) -> dict[str, float]:
    distance = float(distance_m)
    frequency = _positive_frequency(freq_khz)
    bits = int(packet_bits)
    bandwidth = float(params.channel_bandwidth_hz)
    spreading = float(params.spreading_factor)
    shipping = float(params.shipping_noise_factor)
    wind = float(params.wind_speed_mps)
    source_level = params.source_level_db

    if distance < 0.0:
        raise ValueError("distance_m must be non-negative")
    if bits <= 0:
        raise ValueError("packet_bits must be positive")
    if bandwidth <= 0.0:
        raise ValueError("channel_bandwidth_hz must be positive")
    if spreading <= 0.0:
        raise ValueError("spreading_factor must be positive")
    if not 0.0 <= shipping <= 1.0:
        raise ValueError("shipping_noise_factor must be in [0, 1]")
    if wind < 0.0:
        raise ValueError("wind_speed_mps must be non-negative")
    if source_level is None:
        raise ValueError(
            "source_level_db is required; transmit power in Watt cannot be treated as dB"
        )

    alpha = float(thorp_absorption_db_per_km(frequency))
    attenuation = attenuation_linear(
        distance,
        frequency,
        spreading,
        params.acoustic_reference_attenuation_linear,
        params.acoustic_reference_distance_m,
        params.acoustic_model_version,
        params.transmission_anomaly_db,
    )
    loss_db = transmission_loss_db(
        distance,
        frequency,
        spreading,
        params.acoustic_reference_attenuation_linear,
        params.acoustic_reference_distance_m,
        params.acoustic_model_version,
        params.transmission_anomaly_db,
    )
    noise = ambient_noise(
        frequency,
        shipping,
        wind,
        params.noise_model,
        params.noise_formula_mode,
    )
    noise_psd_db = float(noise.total_db)
    noise_power_db = noise_psd_db + 10.0 * float(np.log10(bandwidth))
    snr_db = (
        float(source_level)
        - loss_db
        - noise_power_db
        + float(params.directivity_index_db)
    )
    snr = db_to_linear(snr_db)
    if params.fading_model == "none":
        ber = awgn_bpsk_ber_from_snr_linear(snr)
    elif params.fading_model == "rayleigh":
        ber = ber_from_snr_linear(snr)
    elif params.fading_model == "rician":
        ber = rician_ber_from_snr_linear(snr, params.rician_k_linear)
    else:
        raise ValueError("fading_model must be none, rayleigh or rician")
    success = packet_success_probability(ber, bits)

    minimum_snr = params.minimum_snr_db
    snr_threshold_passed = minimum_snr is None or snr_db >= float(minimum_snr)
    minimum_success = params.minimum_success_probability
    if minimum_success is not None:
        threshold = float(minimum_success)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_success_probability must be in [0, 1]")
    success_threshold_passed = (
        minimum_success is None or success >= float(minimum_success)
    )

    result = {
        "distance_m": distance,
        "frequency_khz": frequency,
        "frequency_hz": khz_to_hz(frequency),
        "absorption_db_per_km": alpha,
        "attenuation_linear": attenuation,
        "transmission_loss_db": loss_db,
        "noise_psd_db": noise_psd_db,
        "noise_power_db": noise_power_db,
        "snr_db": snr_db,
        "snr_linear": snr,
        "ber": ber,
        "per": 1.0 - success,
        "packet_success_probability": success,
        "channel_capacity_bps": shannon_capacity_bps(bandwidth, snr),
        "snr_threshold_passed": snr_threshold_passed,
        "success_threshold_passed": success_threshold_passed,
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("channel quality contains NaN or infinity")
    return result


def _positive_frequency(freq_khz: float) -> float:
    f = float(freq_khz)
    if f <= 0.0:
        raise ValueError("freq_khz must be positive")
    return f
