from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PriorityVectorEncoding:
    """Continuous priority-vector adapter for an exact-K discrete CH set."""

    candidate_node_ids: tuple[int, ...]
    target_cluster_head_count: int

    def decode(self, vector: np.ndarray) -> list[int]:
        candidates = np.asarray(self.candidate_node_ids, dtype=int)
        if vector.ndim != 1 or vector.size != candidates.size:
            raise ValueError("Priority vector length must equal candidate count")
        if self.target_cluster_head_count <= 0:
            return []
        offsets = np.argsort(-vector, kind="stable")[: self.target_cluster_head_count]
        return candidates[offsets].astype(int).tolist()

    def repair(self, selected: Sequence[int]) -> list[int]:
        valid = set(self.candidate_node_ids)
        unique = list(dict.fromkeys(int(node) for node in selected if int(node) in valid))
        return unique[: self.target_cluster_head_count]

    def validate(self, selected: Sequence[int]) -> bool:
        repaired = self.repair(selected)
        return (
            len(repaired) == self.target_cluster_head_count
            and len(set(repaired)) == len(repaired)
        )
