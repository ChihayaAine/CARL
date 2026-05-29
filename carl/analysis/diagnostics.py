"""Lightweight diagnostics: per-protocol harm rate, overlap (ess / N),
selected-action mix, conformal residual histogram.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class Diagnostics:
    selected_mix: Dict[str, float]
    per_arm_harm_rate: Dict[str, float]
    effective_sample_size: float
    abstention_rate: float


def diagnostics(
    decisions: np.ndarray,
    *,
    util_matrix: np.ndarray,
    behavior_prob: np.ndarray,
    arm_names: Sequence[str],
    solo_index: int = 0,
) -> Diagnostics:
    K = len(arm_names)
    counts = np.bincount(decisions, minlength=K)
    mix = {arm_names[k]: float(counts[k] / max(decisions.size, 1)) for k in range(K)}

    solo_util = util_matrix[:, solo_index]
    harm = {}
    for k in range(K):
        if k == solo_index:
            harm[arm_names[k]] = 0.0
            continue
        harm[arm_names[k]] = float((util_matrix[:, k] < solo_util).mean())

    w = 1.0 / np.clip(behavior_prob, 1e-6, 1.0)
    ess = (w.sum() ** 2) / max((w ** 2).sum(), 1e-12)
    ess_norm = float(ess / max(behavior_prob.size, 1))

    abst = float((decisions == solo_index).mean()) if decisions.size else 1.0

    return Diagnostics(selected_mix=mix, per_arm_harm_rate=harm,
                       effective_sample_size=ess_norm,
                       abstention_rate=abst)
