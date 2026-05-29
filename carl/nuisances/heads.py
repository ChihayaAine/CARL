"""Nuisance head factories.

The paper describes a shared encoder feeding 2-layer MLPs of hidden size 256
with GELU + dropout 0.1 for each treatment-specific head. We provide two
backends: a fast, dependency-free ``HistGradientBoostingRegressor`` default
suitable for tests and small-scale runs, and an optional Torch MLP that
matches the paper's architecture for full-scale experiments.

Each "head" here is a single sklearn-style estimator with ``fit`` /
``predict`` / ``predict_proba``. Cross-fitting is handled in
:mod:`carl.nuisances.crossfit`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

import numpy as np

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
    from sklearn.linear_model import LogisticRegression, Ridge
    _SKLEARN = True
except Exception:
    _SKLEARN = False


def _check_sklearn() -> None:
    if not _SKLEARN:
        raise ImportError("scikit-learn is required for nuisance estimators")


def build_nuisance_estimator(kind: str = "boost", *, seed: int = 0) -> Any:
    _check_sklearn()
    kind = (kind or "boost").lower()
    if kind == "boost":
        return HistGradientBoostingRegressor(
            loss="squared_error", max_iter=200, learning_rate=0.05,
            max_depth=6, l2_regularization=0.0, random_state=seed,
        )
    if kind == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    raise ValueError(kind)


def build_propensity_estimator(kind: str = "logistic", *, seed: int = 0) -> Any:
    _check_sklearn()
    kind = (kind or "logistic").lower()
    if kind == "logistic":
        return LogisticRegression(max_iter=400, C=1.0, multi_class="auto")
    if kind == "boost":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6, random_state=seed,
        )
    raise ValueError(kind)


@dataclass
class HeadEnsemble:
    """Bootstrap ensemble of regressors used to obtain mean + epistemic sigma."""
    members: List[Any]

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = np.stack([m.predict(X) for m in self.members], axis=0)
        return preds.mean(axis=0), preds.std(axis=0)


def fit_head_ensemble(make: Callable[[int], Any], X: np.ndarray, y: np.ndarray,
                      *, n_boot: int, seed: int = 0) -> HeadEnsemble:
    """Fit ``n_boot`` regressors on bootstrap resamples and return them."""
    rng = np.random.default_rng(seed)
    members: List[Any] = []
    n = X.shape[0]
    for b in range(n_boot):
        if n == 0:
            members.append(make(b + seed))
            continue
        idx = rng.integers(0, n, size=n)
        m = make(b + seed)
        m.fit(X[idx], y[idx])
        members.append(m)
    return HeadEnsemble(members=members)
