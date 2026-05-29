from __future__ import annotations

from typing import Any, Dict

from .base import Execution, Protocol
from .prompts import (DEBATER_A_SYSTEM, DEBATER_B_SYSTEM, JUDGE_SYSTEM,
                      SYSTEM_BY_STYLE, render_user)


class DebateTwoProtocol(Protocol):
    """T3: two proposers debate over 2-3 turns; a judge adjudicates."""

    id = "DEBATE_2"

    DECODING_DEBATERS = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 512}
    DECODING_JUDGE = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256}
    TURN_BUDGET = 3
    STOPPING_RULE = "fixed_turns_then_judge"
    AGGREGATION = "judge"

    def spec(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model_name,
            "prompt_style": self.prompt_style,
            "debater_a_system": DEBATER_A_SYSTEM,
            "debater_b_system": DEBATER_B_SYSTEM,
            "judge_system": JUDGE_SYSTEM,
            "decoding_debaters": self.DECODING_DEBATERS,
            "decoding_judge": self.DECODING_JUDGE,
            "turn_budget": self.TURN_BUDGET,
            "stopping_rule": self.STOPPING_RULE,
            "aggregation": self.AGGREGATION,
            "topology": "two_proposers_one_judge",
        }

    def run(self, example: Dict[str, Any]) -> Execution:
        t0 = self._now()
        in_tok = 0
        out_tok = 0
        turns = []

        sys_solver = SYSTEM_BY_STYLE[self.prompt_style]
        user_a = self._wrap_prompt(render_user(example["input"]), example)
        a1 = self.backend.system_chat(sys_solver + "\n" + DEBATER_A_SYSTEM,
                                      user_a, seed=self.seed,
                                      **self.DECODING_DEBATERS)
        in_tok += a1.input_tokens
        out_tok += a1.output_tokens
        turns.append({"role": "debater_a", "text": a1.text,
                      "input_tokens": a1.input_tokens, "output_tokens": a1.output_tokens})

        b1 = self.backend.system_chat(sys_solver + "\n" + DEBATER_B_SYSTEM,
                                      self._wrap_prompt(render_user(example["input"], a1.text), example),
                                      seed=None if self.seed is None else self.seed + 1,
                                      **self.DECODING_DEBATERS)
        in_tok += b1.input_tokens
        out_tok += b1.output_tokens
        turns.append({"role": "debater_b", "text": b1.text,
                      "input_tokens": b1.input_tokens, "output_tokens": b1.output_tokens})

        transcript = f"A: {a1.text}\nB: {b1.text}"
        judge_user = self._wrap_prompt(
            render_user(example["input"], history=transcript), example
        )
        j = self.backend.system_chat(JUDGE_SYSTEM, judge_user,
                                     seed=self.seed, **self.DECODING_JUDGE)
        in_tok += j.input_tokens
        out_tok += j.output_tokens
        turns.append({"role": "judge", "text": j.text,
                      "input_tokens": j.input_tokens, "output_tokens": j.output_tokens})

        elapsed = max(self._now() - t0,
                      a1.latency_seconds + b1.latency_seconds + j.latency_seconds)

        return Execution(
            answer=j.text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_seconds=elapsed,
            treatment_id=self.id,
            treatment_hash=self.treatment_hash,
            turns=turns,
        )
