"""Causal-flavored baselines that use DR but differ from CARL.

- DR-Greedy: pick argmax of DR-estimated advantage; no calibrated pessimism.
- DR-LCB (absolute): LCB on absolute DR-estimated utility, not on the
  one-vs-solo advantage. With +Solo it still abstains, but uses an
  absolute-utility lower bound rather than an advantage lower bound.
- Causal-Routing-style / +Solo: train an outcome-regression-like model with
  DR pseudo-outcomes but on absolute utility, not the contrast vs SOLO.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from carl.nuisances.crossfit import NuisancePredictions
from carl.calibration.conformal import ConformalResult, apply_lcb
from carl.dr.advantage import fit_advantage_ensemble
from .fixed_baselines import BaselineDecision


def dr_greedy(a_hat: np.ndarray, *, solo_index: int = 0) -> BaselineDecision:
    best = np.argmax(a_hat, axis=1)
    return BaselineDecision(name="DR-Greedy", decisions=best.astype(np.int64))


def dr_lcb_absolute(
    X_tr: np.ndarray,
    nuis_tr: NuisancePredictions,
    Y_tr: np.ndarray,
    T_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    lam: float,
    mu: float,
    cal_abs: ConformalResult,
    kappa: float = 1.0,
    solo_index: int = 0,
    seed: int = 0,
) -> BaselineDecision:
    """Absolute-utility variant of LCB. Trains a regressor on the DR
    pseudo-outcome of absolute utility (per arm) and applies LCB on it.
    Falls back to solo when no arm's lower bound exceeds solo's."""
    N, K = nuis_tr.mu_R.shape
    mu_Y = nuis_tr.mu_Y(lam, mu)
    e = np.clip(nuis_tr.e, 1e-6, 1.0)
    ind = np.zeros((N, K))
    ind[np.arange(N), T_tr] = 1.0
    w = np.clip(ind / e, 0.0, 50.0)
    targets = mu_Y + w * (Y_tr[:, None] - mu_Y)
    model = fit_advantage_ensemble(X_tr, targets, solo_index=-1,
                                   n_boot=5, seed=seed)
    # NOTE: solo_index=-1 keeps ensembles for every arm
    mean, sigma = model.predict(X_te)
    lcb = apply_lcb(mean, sigma, cal_abs, kappa=kappa)
    solo_lcb = lcb[:, solo_index][:, None]
    diff = lcb - solo_lcb
    best = np.argmax(diff, axis=1)
    best_val = diff[np.arange(diff.shape[0]), best]
    decisions = np.where(best_val > 0.0, best, solo_index)
    return BaselineDecision(name="DR-LCB(abs)", decisions=decisions.astype(np.int64))


def causal_routing_style(
    X_tr: np.ndarray,
    nuis_tr: NuisancePredictions,
    Y_tr: np.ndarray,
    T_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    lam: float,
    mu: float,
    with_solo: bool = False,
    solo_index: int = 0,
    seed: int = 0,
) -> BaselineDecision:
    """DR-trained regressor on absolute utility, greedy at deployment.

    +Solo exposes T_0 in the argmax."""
    N, K = nuis_tr.mu_R.shape
    mu_Y = nuis_tr.mu_Y(lam, mu)
    e = np.clip(nuis_tr.e, 1e-6, 1.0)
    ind = np.zeros((N, K))
    ind[np.arange(N), T_tr] = 1.0
    w = np.clip(ind / e, 0.0, 50.0)
    targets = mu_Y + w * (Y_tr[:, None] - mu_Y)
    model = fit_advantage_ensemble(X_tr, targets, solo_index=-1,
                                   n_boot=3, seed=seed)
    mean, _ = model.predict(X_te)
    mask = np.ones(K, dtype=bool)
    if not with_solo:
        mask[solo_index] = False
    masked = np.where(mask[None, :], mean, -np.inf)
    decisions = np.argmax(masked, axis=1)
    name = "Causal-Routing-style+Solo" if with_solo else "Causal-Routing-style"
    return BaselineDecision(name=name, decisions=decisions.astype(np.int64))
