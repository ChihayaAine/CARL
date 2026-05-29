from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Execution:
    """Record of executing a protocol on one input.

    Fields are exactly what the paper requires to be logged with each episode
    (Methodology -> Training and Inference Summary, pass 1):
    final answer, token counts, latency, treatment hash, and per-turn detail.
    """
    answer: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    treatment_id: str
    treatment_hash: str
    turns: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class Protocol(abc.ABC):
    """A fully-specified protocol (model, prompts, roles, decoding params,
    topology, turn budget, stopping rule, aggregation rule).

    Subclasses implement :meth:`spec` (returns the canonical dict that gets
    SHA-256 hashed) and :meth:`run` (executes the protocol and returns an
    :class:`Execution`).
    """

    id: str = "ABSTRACT"

    def __init__(self, backend, prompt_style: str = "generic",
                 model_name: Optional[str] = None, seed: Optional[int] = None):
        self.backend = backend
        self.prompt_style = prompt_style
        self.model_name = model_name or getattr(backend, "model", "unknown")
        self.seed = seed

    # ------------------------------------------------------------------ spec
    @abc.abstractmethod
    def spec(self) -> Dict[str, Any]:
        """Canonical dict describing the protocol, suitable for SHA-256
        hashing. Must include every routing-relevant choice (prompts, decoding,
        roles, turn budget, stopping/aggregation rule)."""

    @property
    def treatment_hash(self) -> str:
        from ..utils.hashing import treatment_hash as _h
        return _h(self.spec())

    # ------------------------------------------------------------------- run
    @abc.abstractmethod
    def run(self, example: Dict[str, Any]) -> Execution:
        """Execute the protocol on a single example.

        ``example`` is expected to provide at minimum a ``question`` /
        ``problem`` / ``issue`` field. Other fields (passages, retrieved
        files, mock-answer markers, difficulty markers, etc.) are protocol-
        and task-specific.
        """

    # --------------------------------------------------------------- helpers
    def _wrap_prompt(self, prompt: str, example: Dict[str, Any]) -> str:
        """Embed mock helpers (answer, difficulty, protocol tag) so the mock
        backend can produce deterministic synthetic answers without leaking
        them to a real backend that would never see these tags."""
        parts = [f"[protocol]{self.id}[/protocol]", prompt]
        gt = example.get("answer") or example.get("answer_text") or example.get("solution")
        if gt is not None:
            parts.append(f"\n[mock_answer]{gt}[/mock_answer]")
        diff = example.get("difficulty_hint")
        if diff is not None:
            parts.append(f"\n[mock_difficulty]{diff}[/mock_difficulty]")
        return "\n".join(parts)

    def _now(self) -> float:
        return time.monotonic()
