from __future__ import annotations

from typing import Any, Dict

from .base import Execution, Protocol
from .prompts import SYSTEM_BY_STYLE, render_user


class SoloProtocol(Protocol):
    """T0: one-agent direct answer."""

    id = "SOLO"

    DECODING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
    TURN_BUDGET = 1
    STOPPING_RULE = "single_turn"
    AGGREGATION = "identity"

    def spec(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model_name,
            "prompt_style": self.prompt_style,
            "system": SYSTEM_BY_STYLE[self.prompt_style],
            "decoding": self.DECODING,
            "turn_budget": self.TURN_BUDGET,
            "stopping_rule": self.STOPPING_RULE,
            "aggregation": self.AGGREGATION,
            "topology": "single",
        }

    def run(self, example: Dict[str, Any]) -> Execution:
        t0 = self._now()
        sys_prompt = SYSTEM_BY_STYLE[self.prompt_style]
        user_prompt = self._wrap_prompt(render_user(example["input"]), example)

        resp = self.backend.system_chat(sys_prompt, user_prompt,
                                        seed=self.seed, **self.DECODING)
        elapsed = max(self._now() - t0, resp.latency_seconds)

        return Execution(
            answer=resp.text,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            latency_seconds=elapsed,
            treatment_id=self.id,
            treatment_hash=self.treatment_hash,
            turns=[{"role": "solver", "text": resp.text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens}],
        )
