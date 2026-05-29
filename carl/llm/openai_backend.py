from __future__ import annotations

import math
import time
from typing import List, Optional, Sequence, Tuple

import requests

from .backend import LLMBackend, LLMResponse, _RateLimiter


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible Chat Completions backend.

    The same code works against the real OpenAI endpoint, OpenRouter,
    vLLM's OpenAI server, together.ai, fireworks, anyscale, and similar
    endpoints. Only ``api_key``, ``base_url`` and ``model`` change.
    """

    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str, *, rpm: int = 600,
                 timeout: float = 120.0, max_retries: int = 4):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._limiter = _RateLimiter(rpm)

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
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if stop:
            body["stop"] = list(stop)
        if seed is not None:
            body["seed"] = int(seed)
        if request_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = int(logprobs_top_k)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._limiter.acquire()
            t0 = time.monotonic()
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=self.timeout)
                latency = time.monotonic() - t0
                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 10))
                    continue
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                text = msg.get("content", "") or ""
                usage = data.get("usage", {}) or {}
                top_lp = None
                if request_logprobs:
                    top_lp = _extract_first_token_logprobs(choice)
                return LLMResponse(
                    text=text,
                    input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0)),
                    latency_seconds=float(latency),
                    top_token_logprobs=top_lp,
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            except Exception as e:  # noqa: BLE001
                last_exc = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"OpenAI-compatible backend failed after {self.max_retries} retries: {last_exc}")


def _extract_first_token_logprobs(choice: dict) -> Optional[List[Tuple[str, float]]]:
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    if not content:
        return None
    first = content[0]
    top = first.get("top_logprobs") or []
    out: List[Tuple[str, float]] = []
    for entry in top:
        tok = entry.get("token", "")
        val = float(entry.get("logprob", math.log(1e-6)))
        out.append((tok, val))
    if not out:
        # Some servers return only the chosen token's logprob.
        out.append((first.get("token", ""), float(first.get("logprob", math.log(1e-6)))))
    return out
