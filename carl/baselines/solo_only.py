"""Solo-only baselines: Self-Consistency and Rerank.

These are solo-arm variants that re-run SOLO multiple times under
different decoding seeds and then aggregate. Since solo cannot be
"routed to" with these aggregations using only logged episodes, we expose
them as decisions == solo_index and the actual aggregation lives in the
episode collection script. The analysis treats them as additional pseudo-
arms with their own utility profile when those rows are available.
"""
from __future__ import annotations

import numpy as np

from .fixed_baselines import BaselineDecision


def solo_self_consistency(N: int, *, solo_index: int = 0) -> BaselineDecision:
    return BaselineDecision(name="Solo-SelfConsist",
                            decisions=np.full(N, solo_index, dtype=np.int64))


def solo_rerank(N: int, *, solo_index: int = 0) -> BaselineDecision:
    return BaselineDecision(name="Solo-Rerank",
                            decisions=np.full(N, solo_index, dtype=np.int64))
