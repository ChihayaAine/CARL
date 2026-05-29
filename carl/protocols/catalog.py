from __future__ import annotations

from typing import Dict, List, Optional

from .base import Protocol
from .debate import DebateTwoProtocol
from .propose_verify import ProposeVerifyProtocol
from .self_reflect import SelfReflectProtocol
from .solo import SoloProtocol
from .verify import VerifyProtocol

PROTOCOL_IDS: List[str] = ["SOLO", "SELF_REFLECT", "VERIFY", "DEBATE_2", "PROPOSE_VERIFY"]
SOLO_ID: str = "SOLO"
PROTOCOL_INDEX: Dict[str, int] = {pid: i for i, pid in enumerate(PROTOCOL_IDS)}

_CTOR = {
    "SOLO": SoloProtocol,
    "SELF_REFLECT": SelfReflectProtocol,
    "VERIFY": VerifyProtocol,
    "DEBATE_2": DebateTwoProtocol,
    "PROPOSE_VERIFY": ProposeVerifyProtocol,
}


def build_catalog(backend, prompt_style: str = "generic",
                  seed: Optional[int] = None) -> Dict[str, Protocol]:
    """Construct one instance per protocol id."""
    out: Dict[str, Protocol] = {}
    for pid in PROTOCOL_IDS:
        out[pid] = _CTOR[pid](backend=backend, prompt_style=prompt_style, seed=seed)
    return out


def catalog_for_task(backend, task: str, seed: Optional[int] = None) -> Dict[str, Protocol]:
    style = {"math": "math", "qa": "qa", "code": "code"}.get(task, "generic")
    return build_catalog(backend, prompt_style=style, seed=seed)


def protocol_index(pid: str) -> int:
    return PROTOCOL_INDEX[pid]


def treatment_hash_for(backend, prompt_style: str, pid: str) -> str:
    return _CTOR[pid](backend=backend, prompt_style=prompt_style).treatment_hash
