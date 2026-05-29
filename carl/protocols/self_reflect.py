from __future__ import annotations

from typing import Any, Dict

from .base import Execution, Protocol
from .prompts import REFLECTOR_SYSTEM, SYSTEM_BY_STYLE, render_user


class SelfReflectProtocol(Protocol):
    """T1: same solver runs a second critique-and-revision turn."""

    id = "SELF_REFLECT"

    DECODING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
    TURN_BUDGET = 2
    STOPPING_RULE = "after_revise"
    AGGREGATION = "last_turn"

    def spec(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model_name,
            "prompt_style": self.prompt_style,
            "solver_system": SYSTEM_BY_STYLE[self.prompt_style],
            "reflector_system": REFLECTOR_SYSTEM,
            "decoding": self.DECODING,
            "turn_budget": self.TURN_BUDGET,
            "stopping_rule": self.STOPPING_RULE,
            "aggregation": self.AGGREGATION,
            "topology": "single_with_self_revision",
        }

    def run(self, example: Dict[str, Any]) -> Execution:
        t0 = self._now()
        in_tok = 0
        out_tok = 0
        turns = []

        sys_prompt = SYSTEM_BY_STYLE[self.prompt_style]
        user_prompt = self._wrap_prompt(render_user(example["input"]), example)
        first = self.backend.system_chat(sys_prompt, user_prompt,
                                         seed=self.seed, **self.DECODING)
        in_tok += first.input_tokens
        out_tok += first.output_tokens
        turns.append({"role": "solver", "text": first.text,
                      "input_tokens": first.input_tokens,
                      "output_tokens": first.output_tokens})

        reflect_user = self._wrap_prompt(
            render_user(example["input"], history=first.text), example
        )
        second = self.backend.system_chat(REFLECTOR_SYSTEM, reflect_user,
                                          seed=self.seed, **self.DECODING)
        in_tok += second.input_tokens
        out_tok += second.output_tokens
        turns.append({"role": "reflector", "text": second.text,
                      "input_tokens": second.input_tokens,
                      "output_tokens": second.output_tokens})

        elapsed = max(self._now() - t0, first.latency_seconds + second.latency_seconds)

        return Execution(
            answer=second.text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_seconds=elapsed,
            treatment_id=self.id,
            treatment_hash=self.treatment_hash,
            turns=turns,
        )
