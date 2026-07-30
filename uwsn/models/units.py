from __future__ import annotations

import numpy as np


def db_to_linear(value_db: float) -> float:
    value = float(value_db)
    if not np.isfinite(value):
        raise ValueError("value_db must be finite")
    return float(np.power(10.0, value / 10.0))


def linear_to_db(value_linear: float) -> float:
    value = float(value_linear)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("value_linear must be finite and positive")
    return 10.0 * float(np.log10(value))


def hz_to_khz(value_hz: float) -> float:
    value = float(value_hz)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    return value / 1000.0


def khz_to_hz(value_khz: float) -> float:
    value = float(value_khz)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("frequency_khz must be finite and positive")
    return value * 1000.0
