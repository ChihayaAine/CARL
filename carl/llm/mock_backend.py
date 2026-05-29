from __future__ import annotations

import hashlib
import math
import re
import time
from typing import List, Optional, Sequence, Tuple

from .backend import LLMBackend, LLMResponse


class MockBackend(LLMBackend):
    """Deterministic synthetic backend used for tests, smoke runs, and CI.

    Behaviour summary
    -----------------
    - For ``SOLO``-style prompts: extracts the ground-truth answer that the
      protocol or reward harness has embedded in the system prompt under the
      ``[mock_answer]`` tag and returns it with a configurable success rate.
    - For ``VERIFY`` / ``SELF_REFLECT`` / ``DEBATE`` / ``PROPOSE_VERIFY``
      prompts: the success rate is a deterministic function of the protocol
      tag and a hashed difficulty signal. This lets the unit tests and
      synthetic experiments exercise the full DR / LCB pipeline with
      reproducible, heterogeneous treatment effects (some protocols help on
      some inputs, hurt on others).
    - Token and latency counts are deterministic functions of the message
      length and protocol tag.

    The synthetic-execution mode is *only* a substitute for a real LLM. Real
    runs use :class:`carl.llm.openai_backend.OpenAIBackend`.
    """

    name = "mock"

    _PROTOCOL_BIAS = {
        "SOLO":            0.55,
        "SELF_REFLECT":    0.62,
        "VERIFY":          0.66,
        "DEBATE_2":        0.60,
        "PROPOSE_VERIFY":  0.68,
    }
    _PROTOCOL_LAT = {
        "SOLO":            1.0,
        "SELF_REFLECT":    2.1,
        "VERIFY":          3.4,
        "DEBATE_2":        6.0,
        "PROPOSE_VERIFY":  7.5,
    }
    _PROTOCOL_TOK_OUT = {
        "SOLO":           240,
        "SELF_REFLECT":   500,
        "VERIFY":         780,
        "DEBATE_2":      1500,
        "PROPOSE_VERIFY":1800,
    }

    def __init__(self, model: str = "mock-model"):
        self.model = model

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _h(s: str) -> int:
        return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(),
                              "little")

    @staticmethod
    def _u01(s: str) -> float:
        return (MockBackend._h(s) % 10_000) / 10_000.0

    @staticmethod
    def _extract_tag(messages: Sequence[dict], tag: str) -> Optional[str]:
        for m in messages:
            content = m.get("content") or ""
            match = re.search(rf"\[{re.escape(tag)}\](.+?)\[/{re.escape(tag)}\]", content,
                              flags=re.DOTALL)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _extract_protocol(messages: Sequence[dict]) -> str:
        for m in messages:
            content = m.get("content") or ""
            match = re.search(r"\[protocol\](\w+)\[/protocol\]", content)
            if match:
                return match.group(1)
        return "SOLO"

    # -------------------------------------------------------------- main API
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
        t0 = time.monotonic()
        protocol = self._extract_protocol(messages)
        answer = self._extract_tag(messages, "mock_answer") or "0"
        # difficulty in [0, 1]
        diff_tag = self._extract_tag(messages, "mock_difficulty")
        difficulty = float(diff_tag) if diff_tag is not None else 0.5

        # Synthetic success probability
        bias = self._PROTOCOL_BIAS.get(protocol, 0.5)
        sig = "{}|{}|{}|{}".format(protocol, answer, difficulty,
                                   "" if seed is None else str(seed))
        noise = self._u01(sig) - 0.5
        # Harder inputs reduce success; collaborative protocols help only when
        # difficulty is in a middle band (mimics the heterogeneity that makes
        # selective collaboration useful in the paper).
        if protocol == "SOLO":
            p_success = bias - 0.45 * difficulty + 0.10 * noise
        else:
            help_band = 1.0 - abs(difficulty - 0.55) * 2.0
            p_success = bias - 0.30 * difficulty + 0.18 * help_band + 0.10 * noise

        p_success = float(min(max(p_success, 0.02), 0.98))
        success = self._u01("succ|" + sig) < p_success

        text = answer if success else self._wrong_variant(answer, sig)

        in_tok = sum(len((m.get("content") or "")) // 4 for m in messages)
        out_tok = self._PROTOCOL_TOK_OUT.get(protocol, 300)
        latency = self._PROTOCOL_LAT.get(protocol, 2.0) + 0.05 * difficulty

        # Optional logprobs over a tiny synthetic vocabulary.
        top_lp = None
        if request_logprobs:
            # entropy roughly tracks difficulty; this is the cheap solo probe
            # signal used by carl.features.solo_probe.
            ent = 0.4 + 0.9 * difficulty
            top_lp = self._make_logprobs(ent, sig, logprobs_top_k)

        # advance monotonic clock so latency_seconds is always > 0
        latency_meas = max(time.monotonic() - t0, 1e-4)
        return LLMResponse(
            text=text,
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            latency_seconds=float(latency if latency > 0 else latency_meas),
            top_token_logprobs=top_lp,
            finish_reason="stop",
        )

    @staticmethod
    def _wrong_variant(answer: str, sig: str) -> str:
        h = MockBackend._h(sig)
        alt = re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), answer)
        if alt == answer:
            alt = answer + f"_alt{h % 10}"
        return alt

    @staticmethod
    def _make_logprobs(target_entropy: float, sig: str, k: int) -> List[Tuple[str, float]]:
        # Build a distribution over k tokens with approximately the requested
        # Shannon entropy. We solve for a temperature-like parameter via a
        # closed-form approximation: low entropy -> one-hot, high entropy ->
        # uniform.
        k = max(k, 2)
        if target_entropy <= 0:
            probs = [1.0] + [0.0] * (k - 1)
        else:
            t = min(max(target_entropy / math.log(k), 1e-4), 1.0)
            base = [1.0] + [t] * (k - 1)
            s = sum(base)
            probs = [p / s for p in base]
        h = MockBackend._h("lp|" + sig)
        # rotate so different inputs get different "top" tokens
        rot = h % k
        probs = probs[rot:] + probs[:rot]
        toks = [f"tok_{i}" for i in range(k)]
        return [(toks[i], math.log(max(probs[i], 1e-6))) for i in range(k)]
