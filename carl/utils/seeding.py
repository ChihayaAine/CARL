from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def derive_seed(base: int, *parts: str | int) -> int:
    """Deterministic 32-bit integer derived from a base seed and string parts."""
    h = hashlib.blake2b(digest_size=4)
    h.update(int(base).to_bytes(8, "little", signed=False))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False) & 0x7FFFFFFF
