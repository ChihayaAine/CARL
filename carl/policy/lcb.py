"""Lower-Confidence-Bound routing policy.

For each row x_i the policy:
1. Computes per-arm LCB of one-vs-solo advantage from the calibrated
   conformal quantiles.
2. Picks the arm with the largest LCB if it exceeds zero, else falls
   back to SOLO (the abstention rule from the paper's Sec. 4.4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from carl.calibration.conformal import ConformalResult, apply_lcb


@dataclass
class LCBDecision:
    decisions: np.ndarray       # (N,) integer treatment indices in [0, K)
    lcb: np.ndarray             # (N, K) per-arm LCB
    abstained: np.ndarray       # (N,) bool, True when we routed to SOLO due to LCB <= 0
    solo_index: int


def lcb_policy(
    a_hat: np.ndarray,
    sigma_hat: np.ndarray,
    cal: ConformalResult,
    *,
    kappa: float = 1.0,
    abstain_threshold: float = 0.0,
) -> LCBDecision:
    lcb = apply_lcb(a_hat, sigma_hat, cal, kappa=kappa)
    best = np.argmax(lcb, axis=1)
    best_val = lcb[np.arange(lcb.shape[0]), best]
    decisions = np.where(best_val > abstain_threshold, best, cal.solo_index)
    abstained = ~(best_val > abstain_threshold)
    return LCBDecision(decisions=decisions, lcb=lcb,
                       abstained=abstained, solo_index=cal.solo_index)
