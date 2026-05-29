"""Headline metrics used in the main and ablation tables.

Definitions follow the paper:
- Util  = mean cost-aware utility Y under the policy's decisions
- Res.  = task-specific resolved metric (EM / F1 / resolved)
- CHR   = collaboration harm rate, computed on the full-matrix audit subset
          as the fraction of inputs where the policy chose a non-solo arm
          whose realised utility was strictly less than SOLO's utility
- Catalog-oracle recovery = (U(pi) - U(solo)) / (U(oracle) - U(solo))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class Summary:
    name: str
    util: float
    res: float
    chr: Optional[float]
    n: int


def utility(reward: np.ndarray, token_cost: np.ndarray, latency_cost: np.ndarray,
            *, lam: float, mu: float) -> np.ndarray:
    return reward - lam * token_cost - mu * latency_cost


def realised_metrics_from_logs(
    decisions: np.ndarray,
    *,
    util_matrix: np.ndarray,
    res_matrix: np.ndarray,
    solo_index: int = 0,
) -> tuple[float, float, float]:
    """Compute (Util, Res, CHR) when full-matrix data is available."""
    rows = np.arange(decisions.shape[0])
    util = float(util_matrix[rows, decisions].mean()) if decisions.size else 0.0
    res = float(res_matrix[rows, decisions].mean()) if decisions.size else 0.0
    non_solo = decisions != solo_index
    if non_solo.any():
        solo_util = util_matrix[rows, solo_index]
        sel_util = util_matrix[rows, decisions]
        chr_ = float(((sel_util < solo_util) & non_solo).sum() / non_solo.sum())
    else:
        chr_ = 0.0
    return util, res, chr_


def catalog_oracle_recovery(util_policy: float, util_solo: float,
                            util_oracle: float) -> float:
    denom = util_oracle - util_solo
    if abs(denom) < 1e-12:
        return float("nan")
    return float((util_policy - util_solo) / denom)


def summarize(name: str, decisions: np.ndarray, *, util_matrix: np.ndarray,
              res_matrix: np.ndarray, solo_index: int = 0) -> Summary:
    util, res, chr_ = realised_metrics_from_logs(
        decisions, util_matrix=util_matrix, res_matrix=res_matrix,
        solo_index=solo_index)
    return Summary(name=name, util=util, res=res, chr=chr_,
                   n=int(decisions.shape[0]))
