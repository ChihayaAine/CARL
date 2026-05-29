"""Fixed-arm baselines: SOLO, SELF_REFLECT, VERIFY, DEBATE_2, PROPOSE_VERIFY,
plus a uniform random arm picker. These are mostly used by the analysis
table generator to compare CARL against constant-policy oracles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FixedDecision:
    decisions: np.ndarray
    arm: int


def fixed_policy(N: int, arm: int) -> FixedDecision:
    return FixedDecision(decisions=np.full(N, arm, dtype=np.int64), arm=arm)


def uniform_policy(N: int, K: int, *, seed: int = 0) -> FixedDecision:
    rng = np.random.default_rng(seed)
    decisions = rng.integers(0, K, size=N).astype(np.int64)
    return FixedDecision(decisions=decisions, arm=-1)
