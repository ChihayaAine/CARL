"""Greedy advantage policy (no abstention, no LCB)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GreedyDecision:
    decisions: np.ndarray
    advantage: np.ndarray
    solo_index: int


def greedy_policy(a_hat: np.ndarray, *, solo_index: int = 0,
                  positive_only: bool = True) -> GreedyDecision:
    best = np.argmax(a_hat, axis=1)
    best_val = a_hat[np.arange(a_hat.shape[0]), best]
    if positive_only:
        decisions = np.where(best_val > 0.0, best, solo_index)
    else:
        decisions = best
    return GreedyDecision(decisions=decisions, advantage=a_hat,
                          solo_index=solo_index)
