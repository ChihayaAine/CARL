from __future__ import annotations

from typing import Any, Dict

from .base import Execution, Protocol
from .prompts import SYSTEM_BY_STYLE, VERIFIER_SYSTEM, render_user


class VerifyProtocol(Protocol):
    """T2: independent verifier checks the solver's answer over two turns."""

    id = "VERIFY"

    DECODING = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
    TURN_BUDGET = 2
    STOPPING_RULE = "after_verifier"
    AGGREGATION = "verifier_overrides"

    def spec(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model_name,
            "prompt_style": self.prompt_style,
            "solver_system": SYSTEM_BY_STYLE[self.prompt_style],
            "verifier_system": VERIFIER_SYSTEM,
            "decoding": self.DECODING,
            "turn_budget": self.TURN_BUDGET,
            "stopping_rule": self.STOPPING_RULE,
            "aggregation": self.AGGREGATION,
            "topology": "solver_then_verifier",
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

        verify_user = self._wrap_prompt(
            render_user(example["input"], history=f"Candidate: {first.text}"),
            example,
        )
        verifier = self.backend.system_chat(VERIFIER_SYSTEM, verify_user,
                                            seed=self.seed, **self.DECODING)
        in_tok += verifier.input_tokens
        out_tok += verifier.output_tokens
        turns.append({"role": "verifier", "text": verifier.text,
                      "input_tokens": verifier.input_tokens,
                      "output_tokens": verifier.output_tokens})

        elapsed = max(self._now() - t0, first.latency_seconds + verifier.latency_seconds)

        return Execution(
            answer=verifier.text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_seconds=elapsed,
            treatment_id=self.id,
            treatment_hash=self.treatment_hash,
            turns=turns,
        )
