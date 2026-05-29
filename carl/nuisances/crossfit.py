"""Cross-fitted nuisance estimation.

For every treatment k we need three regressions and one classifier:

- ``mu_R[k](x)``  -- expected reward given pulled arm k
- ``mu_Ct[k](x)`` -- expected token cost given pulled arm k
- ``mu_Cl[k](x)`` -- expected latency cost given pulled arm k
- ``e[k](x)``      -- propensity P(T=k | x)

To avoid own-fold overfit we split the rows into ``n_folds`` disjoint folds
and, for each fold, train the heads on the remaining folds before predicting
on the held-out fold. Bootstrap ensembling on top of that lets us report an
epistemic sigma for the per-arm reward predictions, which the LCB policy
later consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .heads import (
    HeadEnsemble,
    build_nuisance_estimator,
    build_propensity_estimator,
    fit_head_ensemble,
)


@dataclass
class NuisancePredictions:
    """Out-of-fold nuisance predictions over N rows for K arms."""
    mu_R: np.ndarray           # (N, K)
    sigma_R: np.ndarray        # (N, K)  epistemic from bootstrap ensemble
    mu_Ct: np.ndarray          # (N, K)
    mu_Cl: np.ndarray          # (N, K)
    e: np.ndarray              # (N, K)  propensities, rows sum to 1
    fold_ids: np.ndarray       # (N,)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.mu_R.shape[0])

    @property
    def K(self) -> int:
        return int(self.mu_R.shape[1])

    def mu_Y(self, lam: float, mu: float) -> np.ndarray:
        """E[Y | x, T=k] = mu_R - lambda*mu_Ct - mu*mu_Cl."""
        return self.mu_R - lam * self.mu_Ct - mu * self.mu_Cl


def _stratified_folds(n: int, n_folds: int, *, treatments: Optional[np.ndarray],
                      seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(n, dtype=np.int64)
    if treatments is None or np.unique(treatments).size <= 1:
        order = rng.permutation(n)
        for i, idx in enumerate(order):
            fold_ids[idx] = i % n_folds
        return fold_ids
    # Stratify by treatment so every fold sees every arm.
    for k in np.unique(treatments):
        rows = np.flatnonzero(treatments == k)
        order = rng.permutation(rows)
        for i, idx in enumerate(order):
            fold_ids[idx] = i % n_folds
    return fold_ids


def _safe_unique_y(y: np.ndarray) -> bool:
    return y.size > 0 and np.unique(y).size >= 2


def _fit_predict_regressor(
    make: Callable[[int], Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap ensemble; falls back to a constant prediction if no data."""
    if X_te.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if X_tr.shape[0] == 0 or not _safe_unique_y(y_tr):
        mean = float(y_tr.mean()) if y_tr.size else 0.0
        return (np.full(X_te.shape[0], mean), np.zeros(X_te.shape[0]))
    ens = fit_head_ensemble(make, X_tr, y_tr, n_boot=n_boot, seed=seed)
    return ens.predict_mean_std(X_te)


def crossfit_predict(
    X: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Ct: np.ndarray,
    Cl: np.ndarray,
    *,
    K: int,
    n_folds: int = 5,
    n_boot: int = 5,
    nuisance_kind: str = "boost",
    propensity_kind: str = "logistic",
    e_min: float = 0.02,
    seed: int = 0,
) -> NuisancePredictions:
    """Run cross-fitting for every arm.

    Parameters
    ----------
    X : (N, d) feature matrix.
    T : (N,) integer treatment ids in [0, K).
    R, Ct, Cl : (N,) reward, token cost, latency cost.
    K : number of arms.
    n_folds : number of disjoint folds.
    n_boot : bootstrap ensemble size for the reward heads.
    e_min : floor for predicted propensities so importance weights stay bounded.
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.int64)
    R = np.asarray(R, dtype=np.float64)
    Ct = np.asarray(Ct, dtype=np.float64)
    Cl = np.asarray(Cl, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        z = np.zeros((0, K))
        return NuisancePredictions(z.copy(), z.copy(), z.copy(), z.copy(),
                                   np.full((0, K), 1.0 / K),
                                   np.zeros(0, dtype=np.int64))
    fold_ids = _stratified_folds(n, n_folds, treatments=T, seed=seed)

    mu_R = np.zeros((n, K), dtype=np.float64)
    sigma_R = np.zeros((n, K), dtype=np.float64)
    mu_Ct = np.zeros((n, K), dtype=np.float64)
    mu_Cl = np.zeros((n, K), dtype=np.float64)
    e = np.full((n, K), 1.0 / K, dtype=np.float64)

    reg_factory = lambda s: build_nuisance_estimator(nuisance_kind, seed=s)
    prop_factory = lambda s: build_propensity_estimator(propensity_kind, seed=s)

    for f in range(n_folds):
        te = np.flatnonzero(fold_ids == f)
        tr = np.flatnonzero(fold_ids != f)
        if te.size == 0:
            continue
        X_te = X[te]
        # Propensity (single multinomial classifier)
        if tr.size > 0 and np.unique(T[tr]).size >= 2:
            try:
                clf = prop_factory(seed + f)
                clf.fit(X[tr], T[tr])
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_te)
                    classes = list(getattr(clf, "classes_", np.arange(K)))
                    full = np.full((te.size, K), 1.0 / K, dtype=np.float64)
                    for j, c in enumerate(classes):
                        if 0 <= int(c) < K:
                            full[:, int(c)] = proba[:, j]
                else:
                    full = np.full((te.size, K), 1.0 / K, dtype=np.float64)
            except Exception:
                full = np.full((te.size, K), 1.0 / K, dtype=np.float64)
        else:
            # uniform fallback
            full = np.full((te.size, K), 1.0 / K, dtype=np.float64)
        # Floor and renormalise so weights stay bounded.
        full = np.clip(full, e_min, None)
        full /= full.sum(axis=1, keepdims=True)
        e[te] = full

        # Outcome regressions, one per arm, trained on rows where T==k.
        for k in range(K):
            mask_tr = (T[tr] == k)
            X_tr_k = X[tr][mask_tr]
            mu_R[te, k], sigma_R[te, k] = _fit_predict_regressor(
                reg_factory, X_tr_k, R[tr][mask_tr], X_te,
                n_boot=n_boot, seed=seed + f * 100 + k,
            )
            mu_Ct[te, k], _ = _fit_predict_regressor(
                reg_factory, X_tr_k, Ct[tr][mask_tr], X_te,
                n_boot=1, seed=seed + f * 100 + k + 17,
            )
            mu_Cl[te, k], _ = _fit_predict_regressor(
                reg_factory, X_tr_k, Cl[tr][mask_tr], X_te,
                n_boot=1, seed=seed + f * 100 + k + 53,
            )

    return NuisancePredictions(
        mu_R=mu_R, sigma_R=sigma_R, mu_Ct=mu_Ct, mu_Cl=mu_Cl,
        e=e, fold_ids=fold_ids,
        meta={"n_folds": n_folds, "n_boot": n_boot, "e_min": e_min, "K": K},
    )


def fit_nuisances(
    features,
    episodes: Sequence,
    *,
    cfg,
) -> NuisancePredictions:
    """Convenience adapter: take a FeatureBundle and a list of EpisodeRecords
    sharing the same example order, and run cross-fitting under the cfg knobs.
    """
    # Build dense X and per-row T/R/Ct/Cl by aligning on example_id.
    from carl.features.extract import FeatureBundle

    bundle: FeatureBundle = features
    id_to_row = {eid: i for i, eid in enumerate(bundle.ids)}
    rows: List[int] = []
    T: List[int] = []
    R: List[float] = []
    Ct: List[float] = []
    Cl: List[float] = []
    for ep in episodes:
        i = id_to_row.get(ep.example_id)
        if i is None:
            continue
        rows.append(i)
        T.append(int(ep.treatment_index))
        R.append(float(ep.reward))
        Ct.append(float(ep.token_cost_norm))
        Cl.append(float(ep.latency_cost_norm))
    if not rows:
        K = len(cfg.protocols)
        z = np.zeros((0, K))
        return NuisancePredictions(z.copy(), z.copy(), z.copy(), z.copy(),
                                   np.full((0, K), 1.0 / K),
                                   np.zeros(0, dtype=np.int64))
    X = np.concatenate([bundle.enc[rows], bundle.side[rows]], axis=1)
    return crossfit_predict(
        X, np.asarray(T), np.asarray(R), np.asarray(Ct), np.asarray(Cl),
        K=len(cfg.protocols),
        n_folds=int(cfg.n_folds),
        n_boot=int(cfg.n_boot),
        nuisance_kind=str(cfg.nuisance_kind),
        propensity_kind=str(cfg.propensity_kind),
        e_min=float(cfg.e_min),
        seed=int(cfg.seed),
    )
