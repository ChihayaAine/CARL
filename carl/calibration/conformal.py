"""Conformal calibration of one-sided standardized residuals.

Given calibration rows i with DR pseudo-outcome ``psi_k(i)``, model
prediction ``A_hat_k(x_i)``, and epistemic sigma ``sigma_hat_k(x_i)``,
form the one-sided residual

    r_i = (A_hat_k(x_i) - psi_k(i)) / max(sigma_hat_k(x_i), sigma_floor)

and take the (1 - delta)-quantile ``q_delta`` of {r_i} as the conformal
slack. Then for new x the lower bound is

    LCB_k(x) = A_hat_k(x) - q_delta * sigma_hat_k(x)

This is the LCB rule the policy module consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class ConformalResult:
    q: np.ndarray              # (K,) per-arm quantiles; q[solo_index] is 0.0
    delta: float
    sigma_floor: float
    solo_index: int
    meta: Dict[str, float] = field(default_factory=dict)


def fit_conformal_quantiles(
    psi: np.ndarray,
    a_hat: np.ndarray,
    sigma_hat: np.ndarray,
    *,
    delta: float = 0.10,
    solo_index: int = 0,
    sigma_floor: float = 1e-3,
) -> ConformalResult:
    """Compute the per-arm (1 - delta)-quantile of standardized residuals."""
    N, K = psi.shape
    q = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if k == solo_index or N == 0:
            continue
        denom = np.maximum(sigma_hat[:, k], sigma_floor)
        r = (a_hat[:, k] - psi[:, k]) / denom
        q[k] = float(np.quantile(r, 1.0 - delta, method="higher"))
    return ConformalResult(q=q, delta=delta, sigma_floor=sigma_floor,
                           solo_index=solo_index,
                           meta={"n_calib": float(N)})


def apply_lcb(
    a_hat: np.ndarray,
    sigma_hat: np.ndarray,
    cal: ConformalResult,
    *,
    kappa: float = 1.0,
) -> np.ndarray:
    """Return LCB_k(x) = a_hat - kappa * q_k * sigma_hat (per arm).

    For the solo arm, LCB == 0 (the contrast with itself).
    """
    denom = np.maximum(sigma_hat, cal.sigma_floor)
    lcb = a_hat - kappa * cal.q[None, :] * denom
    lcb[:, cal.solo_index] = 0.0
    return lcb
