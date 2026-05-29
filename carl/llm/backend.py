from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    # When the backend can return per-token logprobs of the *first* output token
    # we use them in the cheap solo probe to compute pre-treatment entropy.
    top_token_logprobs: Optional[List[Tuple[str, float]]] = None
    finish_reason: str = "stop"


class LLMBackend(abc.ABC):
    """Abstract LLM backend.

    Implementations only need to provide :meth:`complete`. The backend must
    return tokens consumed and end-to-end latency so the cost-aware utility
    accountancy in :mod:`carl.behavior` and :mod:`carl.rewards` can compute
    Eq. 1 of the paper without external probes.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def complete(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        request_logprobs: bool = False,
        logprobs_top_k: int = 5,
    ) -> LLMResponse:
        ...

    # Convenience wrappers; subclasses do not need to override these.
    def chat(self, prompt: str, **kwargs) -> LLMResponse:
        return self.complete([{"role": "user", "content": prompt}], **kwargs)

    def system_chat(self, system: str, prompt: str, **kwargs) -> LLMResponse:
        return self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            **kwargs,
        )


def build_backend(cfg) -> LLMBackend:
    """Construct the LLM backend selected by configuration."""
    kind = (cfg.backend_kind or "auto").lower()
    if kind == "auto":
        kind = "openai" if cfg.openai_api_key else "mock"
    if kind == "openai":
        from .openai_backend import OpenAIBackend
        return OpenAIBackend(
            api_key=cfg.openai_api_key or "",
            base_url=cfg.openai_base_url,
            model=cfg.model_name,
            rpm=cfg.llm_rpm,
        )
    if kind == "mock":
        from .mock_backend import MockBackend
        return MockBackend(model=cfg.model_name or "mock-model")
    raise ValueError(f"Unknown backend kind: {kind!r}")


class _RateLimiter:
    """Very small client-side rpm guard."""

    def __init__(self, rpm: int):
        self.rpm = max(int(rpm), 1)
        self._times: List[float] = []

    def acquire(self) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        self._times = [t for t in self._times if t >= cutoff]
        if len(self._times) >= self.rpm:
            sleep = 60.0 - (now - self._times[0])
            if sleep > 0:
                time.sleep(sleep)
        self._times.append(time.monotonic())
