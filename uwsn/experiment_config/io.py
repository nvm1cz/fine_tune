from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML without adding a runtime dependency."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load YAML config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return value


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic JSON syntax, which is valid YAML 1.2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
