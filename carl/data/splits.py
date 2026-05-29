"""Disjoint per-task splits matching Table 3 of the paper."""
from __future__ import annotations

import random
from typing import Dict, List


def make_splits(examples: List[dict], sizes: Dict[str, int], seed: int = 0
                ) -> Dict[str, List[dict]]:
    """Cut a list of examples into disjoint train/val/calib/test/full_matrix
    splits using ``sizes`` (some keys may be missing in which case those
    splits stay empty)."""
    rng = random.Random(seed)
    pool = list(examples)
    rng.shuffle(pool)
    needed = sum(int(v) for v in sizes.values())
    if len(pool) < needed:
        # Scale down proportionally so the pipeline can run on small data.
        scale = len(pool) / float(max(needed, 1))
        sizes = {k: max(int(v * scale), 1) for k, v in sizes.items()}
    out: Dict[str, List[dict]] = {}
    cursor = 0
    for k, n in sizes.items():
        n = int(n)
        out[k] = pool[cursor:cursor + n]
        cursor += n
    return out
