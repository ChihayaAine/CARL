from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def treatment_hash(spec: Mapping[str, Any]) -> str:
    """SHA-256 over the JSON-canonical form of a protocol specification.

    The hash is the treatment-definition audit identifier required by the paper
    so that calibration / training data can be invalidated when prompts,
    decoding parameters, or checkpoints change.
    """
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
