"""Oracle policies used as upper-bound references in the main table.

The catalog oracle requires per-row per-arm realised utility (i.e. the
full-matrix audit subset). It is not online observable and is reported as
an upper bound, not a baseline a system could deploy.
"""
from __future__ import annotations

import numpy as np

from .fixed_baselines import BaselineDecision


def catalog_oracle(utility_matrix: np.ndarray, *,
                   solo_index: int = 0) -> BaselineDecision:
    """Pick argmax_k Y(k) on each row of the full-matrix split."""
    decisions = np.argmax(utility_matrix, axis=1).astype(np.int64)
    return BaselineDecision(name="Catalog Oracle", decisions=decisions)


def oracle_with_abstention(utility_matrix: np.ndarray, *,
                           solo_index: int = 0) -> BaselineDecision:
    """The paper's oracle: pick the best collaborative arm only if it beats
    SOLO on that row, otherwise pick SOLO."""
    best = np.argmax(utility_matrix, axis=1)
    best_val = utility_matrix[np.arange(utility_matrix.shape[0]), best]
    solo_val = utility_matrix[:, solo_index]
    decisions = np.where(best_val > solo_val, best, solo_index).astype(np.int64)
    return BaselineDecision(name="Oracle+Abstain", decisions=decisions)
