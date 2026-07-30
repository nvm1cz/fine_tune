from __future__ import annotations

import warnings
from typing import Callable

from ..optimizers.registry import create_optimizer
from .base import UWSNProtocol
from .ebrec import EBRECProtocol
from .eeumc import EEUMCProtocol
from .eulc import EULCProtocol
from .leach import LEACHProtocol
from .protocols import EULCOptimizedProtocol


CANONICAL_ALGORITHM_IDS = (
    "eeumc", "ebrec", "eulc", "leach",
    "eulc_pso", "eulc_ga", "eulc_ac_aco",
)
ALIASES = {
    "pso": "eulc_pso", "ga": "eulc_ga",
    "ac_aco": "eulc_ac_aco", "ac-aco": "eulc_ac_aco",
    "eulc+pso": "eulc_pso", "eulc+ga": "eulc_ga",
    "eulc+ac-aco": "eulc_ac_aco",
}


def canonical_algorithm_id(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ALIASES:
        canonical = ALIASES[normalized]
        warnings.warn(
            f"Algorithm ID '{value}' is deprecated; use '{canonical}'",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical
    if normalized not in CANONICAL_ALGORITHM_IDS:
        raise ValueError(f"Unknown algorithm ID: {value}")
    return normalized


def _optimized(algorithm_id: str, optimizer_name: str) -> UWSNProtocol:
    return EULCOptimizedProtocol(algorithm_id, create_optimizer(optimizer_name))


ALGORITHM_REGISTRY: dict[str, Callable[[], UWSNProtocol]] = {
    "eeumc": EEUMCProtocol,
    "ebrec": EBRECProtocol,
    "eulc": EULCProtocol,
    "leach": LEACHProtocol,
    "eulc_pso": lambda: _optimized("eulc_pso", "pso"),
    "eulc_ga": lambda: _optimized("eulc_ga", "ga"),
    "eulc_ac_aco": lambda: _optimized("eulc_ac_aco", "ac_aco"),
}


def create_algorithm(algorithm_id: str) -> UWSNProtocol:
    canonical = canonical_algorithm_id(algorithm_id)
    return ALGORITHM_REGISTRY[canonical]()
