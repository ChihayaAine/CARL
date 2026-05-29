"""Predictive baselines that don't use the DR target.

Includes:
- DM-Greedy: direct outcome regression on per-arm utility (no DR, no LCB)
- Outcome-Reg+LCB: same direct model with bootstrap sigma + LCB pessimism
- Adaptive-Orch+Abs.: classifier over logged utility, picks argmax m_k(x)
- Naive-Obs: logged-mean per arm, ignoring confounding
- MasRouter-style: cascaded controller. Stage 1: binary collaborate vs not;
  Stage 2: pick among collaborative protocols. +Solo exposes T_0 to stage 2.
- CascadeDebate-style: solo→VERIFY→DEBATE_2 ladder gated on entropy
- Confidence-Trigger: solo confidence threshold → single collab protocol
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from carl.nuisances.heads import build_nuisance_estimator, fit_head_ensemble
from carl.calibration.conformal import ConformalResult, apply_lcb
from .fixed_baselines import BaselineDecision


def _direct_per_arm_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    T_tr: np.ndarray,
    K: int,
    *,
    n_boot: int = 1,
    seed: int = 0,
):
    make = lambda s: build_nuisance_estimator("boost", seed=s)
    ensembles = []
    for k in range(K):
        mask = (T_tr == k)
        if mask.sum() < 2 or np.unique(y_tr[mask]).size < 2:
            ensembles.append(None)
            continue
        ensembles.append(fit_head_ensemble(make, X_tr[mask], y_tr[mask],
                                           n_boot=n_boot, seed=seed + k))
    return ensembles


def _predict_per_arm(ensembles, X: np.ndarray, K: int):
    N = X.shape[0]
    mean = np.zeros((N, K), dtype=np.float64)
    sigma = np.zeros((N, K), dtype=np.float64)
    for k, ens in enumerate(ensembles):
        if ens is None:
            continue
        m, s = ens.predict_mean_std(X)
        mean[:, k] = m
        sigma[:, k] = s
    return mean, sigma


def dm_greedy(X_tr, Y_tr, T_tr, X_te, K, *, solo_index=0, seed=0) -> BaselineDecision:
    ens = _direct_per_arm_model(X_tr, Y_tr, T_tr, K, n_boot=1, seed=seed)
    mean, _ = _predict_per_arm(ens, X_te, K)
    best = np.argmax(mean, axis=1)
    return BaselineDecision(name="DM-Greedy", decisions=best.astype(np.int64))


def outcome_reg_lcb(X_tr, Y_tr, T_tr, X_te, K, *,
                    cal: ConformalResult, kappa: float = 1.0,
                    solo_index: int = 0, seed: int = 0) -> BaselineDecision:
    ens = _direct_per_arm_model(X_tr, Y_tr, T_tr, K, n_boot=5, seed=seed)
    mean, sigma = _predict_per_arm(ens, X_te, K)
    contrast = mean - mean[:, [solo_index]]
    lcb = apply_lcb(contrast, sigma, cal, kappa=kappa)
    best = np.argmax(lcb, axis=1)
    best_val = lcb[np.arange(lcb.shape[0]), best]
    decisions = np.where(best_val > 0.0, best, solo_index)
    return BaselineDecision(name="Outcome-Reg+LCB", decisions=decisions.astype(np.int64))


def adaptive_orch_abs(X_tr, Y_tr, T_tr, X_te, K, *, solo_index=0, seed=0) -> BaselineDecision:
    """Classifier that picks the arm with highest predicted utility, with
    abstention to solo when no arm exceeds solo's predicted utility."""
    ens = _direct_per_arm_model(X_tr, Y_tr, T_tr, K, n_boot=1, seed=seed)
    mean, _ = _predict_per_arm(ens, X_te, K)
    best = np.argmax(mean, axis=1)
    best_val = mean[np.arange(mean.shape[0]), best]
    solo_val = mean[:, solo_index]
    decisions = np.where(best_val > solo_val, best, solo_index)
    return BaselineDecision(name="Adaptive-Orch+Abs", decisions=decisions.astype(np.int64))


def naive_obs(T_tr, Y_tr, X_te, K, *, solo_index=0) -> BaselineDecision:
    """Pick the arm with the highest *logged* (confounded) mean utility."""
    means = np.zeros(K)
    for k in range(K):
        mask = (T_tr == k)
        means[k] = float(Y_tr[mask].mean()) if mask.any() else -np.inf
    if np.all(np.isneginf(means)):
        arm = solo_index
    else:
        arm = int(np.nanargmax(np.where(np.isfinite(means), means, -np.inf)))
    return BaselineDecision(name="Naive-Obs",
                            decisions=np.full(X_te.shape[0], arm, dtype=np.int64))


def mas_router(X_tr, Y_tr, T_tr, X_te, K, *,
               with_solo: bool = False, solo_index: int = 0, seed: int = 0
               ) -> BaselineDecision:
    """Cascaded controller. Stage-1: binary collaborate vs not, trained on the
    indicator ``Y_tr > Y_tr[T==solo].mean()``. Stage-2: pick the highest-utility
    arm among collaborative protocols. +Solo exposes T_0 in stage-2."""
    from sklearn.linear_model import LogisticRegression

    name = "MasRouter-style+Solo" if with_solo else "MasRouter-style"
    if X_tr.shape[0] == 0:
        return BaselineDecision(name=name,
                                decisions=np.full(X_te.shape[0], solo_index, dtype=np.int64))
    solo_mask = (T_tr == solo_index)
    solo_mean = float(Y_tr[solo_mask].mean()) if solo_mask.any() else 0.0
    coll = (Y_tr > solo_mean).astype(int)
    if np.unique(coll).size < 2:
        coll_pred = np.zeros(X_te.shape[0], dtype=bool)
    else:
        try:
            clf = LogisticRegression(max_iter=400).fit(X_tr, coll)
            coll_pred = clf.predict(X_te).astype(bool)
        except Exception:
            coll_pred = np.zeros(X_te.shape[0], dtype=bool)
    # Stage 2 selector
    ens = _direct_per_arm_model(X_tr, Y_tr, T_tr, K, n_boot=1, seed=seed)
    mean, _ = _predict_per_arm(ens, X_te, K)
    select_mask = np.ones(K, dtype=bool)
    if not with_solo:
        select_mask[solo_index] = False
    masked = np.where(select_mask[None, :], mean, -np.inf)
    coll_arm = np.argmax(masked, axis=1)
    decisions = np.where(coll_pred, coll_arm, solo_index)
    return BaselineDecision(name=name, decisions=decisions.astype(np.int64))


def cascade_debate(entropies: np.ndarray, *,
                   solo_index: int,
                   verify_index: int,
                   debate_index: int,
                   theta_low: float,
                   theta_high: float) -> BaselineDecision:
    """Cascade: solo always runs; if entropy >= theta_low fire VERIFY; if
    >= theta_high also fire DEBATE_2 (we record the deepest tier as the
    routing decision since that determines the realised utility)."""
    entropies = np.asarray(entropies, dtype=np.float64)
    decisions = np.full(entropies.shape[0], solo_index, dtype=np.int64)
    decisions = np.where(entropies >= theta_low, verify_index, decisions)
    decisions = np.where(entropies >= theta_high, debate_index, decisions)
    return BaselineDecision(name="CascadeDebate-style", decisions=decisions)


def confidence_trigger(entropies: np.ndarray, *,
                       solo_index: int, collab_index: int,
                       threshold: float) -> BaselineDecision:
    """Single-threshold confidence trigger: fire collaborative protocol when
    solo first-token entropy exceeds ``threshold``."""
    entropies = np.asarray(entropies, dtype=np.float64)
    decisions = np.where(entropies >= threshold, collab_index, solo_index)
    return BaselineDecision(name="Confidence-Trigger",
                            decisions=decisions.astype(np.int64))
