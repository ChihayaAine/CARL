"""Doubly Robust scoring of one-vs-solo advantage.

Per row i and arm k != SOLO we form the DR pseudo-outcome for the contrast
``A_k(x_i) = E[Y | x_i, T=k] - E[Y | x_i, T=SOLO]`` using the well-known
augmented IPW form (e.g. Robins-Rotnitzky-Zhao). For an arm k the DR signal
is

    psi_k(i) = mu_Y[k](x_i) + 1{T_i = k}/e_k(x_i) * (Y_i - mu_Y[k](x_i))
             - mu_Y[0](x_i) - 1{T_i = 0}/e_0(x_i) * (Y_i - mu_Y[0](x_i))

where ``mu_Y[k] = mu_R[k] - lambda*mu_Ct[k] - mu*mu_Cl[k]`` and the
indicator/propensity terms are clipped via the previously enforced
``e_min`` floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from carl.nuisances.crossfit import NuisancePredictions


@dataclass
class DRScores:
    psi: np.ndarray             # (N, K)  DR pseudo-outcomes (psi[:, 0] == 0)
    weight: np.ndarray          # (N, K)  importance weights
    mu_Y: np.ndarray            # (N, K)  E[Y | x, T=k] from nuisances
    sigma_R: np.ndarray         # (N, K)  epistemic sigma carried through
    solo_index: int
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def N(self) -> int:
        return int(self.psi.shape[0])

    @property
    def K(self) -> int:
        return int(self.psi.shape[1])

    def direct_contrast(self) -> np.ndarray:
        return self.mu_Y - self.mu_Y[:, [self.solo_index]]


def build_dr_scores(
    nuis: NuisancePredictions,
    *,
    T: np.ndarray,
    R: np.ndarray,
    Ct: np.ndarray,
    Cl: np.ndarray,
    lam: float,
    mu: float,
    solo_index: int = 0,
    clip_w: float = 50.0,
) -> DRScores:
    """Build DR pseudo-outcomes for one-vs-solo contrasts."""
    T = np.asarray(T, dtype=np.int64)
    R = np.asarray(R, dtype=np.float64)
    Ct = np.asarray(Ct, dtype=np.float64)
    Cl = np.asarray(Cl, dtype=np.float64)
    Y = R - lam * Ct - mu * Cl
    N, K = nuis.mu_R.shape

    mu_Y = nuis.mu_Y(lam, mu)
    e = np.clip(nuis.e, 1e-6, 1.0)

    # AIPW per arm: psi_k(i) = mu_Y[k] + 1{T==k}/e_k * (Y - mu_Y[k])
    psi_per_arm = mu_Y.copy()
    row_idx = np.arange(N)
    indicator = np.zeros((N, K), dtype=np.float64)
    indicator[row_idx, T] = 1.0
    weight = indicator / e
    weight = np.clip(weight, 0.0, clip_w)
    psi_per_arm = psi_per_arm + weight * (Y[:, None] - mu_Y)
    # contrast vs solo arm
    psi = psi_per_arm - psi_per_arm[:, [solo_index]]
    return DRScores(
        psi=psi,
        weight=weight,
        mu_Y=mu_Y,
        sigma_R=nuis.sigma_R,
        solo_index=solo_index,
        meta={"lambda": lam, "mu": mu, "clip_w": clip_w},
    )
