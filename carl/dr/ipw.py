"""Inverse-propensity (Horvitz-Thompson) and self-normalised IPW estimators
of policy value, used both as a sanity check on the DR pipeline and as
baselines in the analysis tables.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OPEResult:
    value: float
    se: float
    n: int


def _per_row_indicator(T: np.ndarray, decisions: np.ndarray) -> np.ndarray:
    return (T == decisions).astype(np.float64)


def ipw_value(
    T: np.ndarray,
    decisions: np.ndarray,
    Y: np.ndarray,
    behavior_prob: np.ndarray,
    *,
    self_normalised: bool = True,
    clip_w: float = 50.0,
) -> OPEResult:
    """IPW / SNIPW estimate of E[Y | follow decisions]."""
    ind = _per_row_indicator(T, decisions)
    w = np.clip(ind / np.clip(behavior_prob, 1e-6, 1.0), 0.0, clip_w)
    if self_normalised:
        denom = max(float(w.sum()), 1e-12)
        v = float((w * Y).sum() / denom)
    else:
        v = float((w * Y).mean())
    # crude IID SE
    if w.sum() > 0:
        se = float((w * (Y - v)).std() / max(np.sqrt(len(Y)), 1.0))
    else:
        se = 0.0
    return OPEResult(value=v, se=se, n=int(len(Y)))


def dr_value(
    T: np.ndarray,
    decisions: np.ndarray,
    Y: np.ndarray,
    mu_Y: np.ndarray,
    e: np.ndarray,
    *,
    clip_w: float = 50.0,
) -> OPEResult:
    """Doubly Robust estimate of E[Y | follow decisions]."""
    N, K = mu_Y.shape
    rows = np.arange(N)
    pred = mu_Y[rows, decisions]
    e_dec = np.clip(e[rows, decisions], 1e-6, 1.0)
    ind = (T == decisions).astype(np.float64)
    w = np.clip(ind / e_dec, 0.0, clip_w)
    psi = pred + w * (Y - pred)
    v = float(psi.mean())
    se = float(psi.std() / max(np.sqrt(len(psi)), 1.0))
    return OPEResult(value=v, se=se, n=int(N))
