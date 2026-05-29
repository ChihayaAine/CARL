from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .base import Execution, Protocol
from .prompts import (PROPOSER_SYSTEM, SYSTEM_BY_STYLE, VERIFIER_SYSTEM,
                      render_user)


class ProposeVerifyProtocol(Protocol):
    """T4: generate K proposals then verify them across three turns."""

    id = "PROPOSE_VERIFY"

    K_PROPOSERS = 3
    DECODING_PROPOSERS = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 512}
    DECODING_VERIFIER = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
    TURN_BUDGET = 3
    STOPPING_RULE = "propose_then_verify"
    AGGREGATION = "verifier_picks"

    def spec(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model_name,
            "prompt_style": self.prompt_style,
            "proposer_system_template": PROPOSER_SYSTEM,
            "verifier_system": VERIFIER_SYSTEM,
            "n_proposers": self.K_PROPOSERS,
            "decoding_proposers": self.DECODING_PROPOSERS,
            "decoding_verifier": self.DECODING_VERIFIER,
            "turn_budget": self.TURN_BUDGET,
            "stopping_rule": self.STOPPING_RULE,
            "aggregation": self.AGGREGATION,
            "topology": "k_proposers_one_verifier",
        }

    def run(self, example: Dict[str, Any]) -> Execution:
        t0 = self._now()
        in_tok = 0
        out_tok = 0
        turns: List[Dict[str, Any]] = []
        sys_solver = SYSTEM_BY_STYLE[self.prompt_style]

        proposals: List[str] = []
        for i in range(self.K_PROPOSERS):
            system = sys_solver + "\n" + PROPOSER_SYSTEM.format(idx=i + 1)
            seed = None if self.seed is None else (self.seed + i * 17)
            r = self.backend.system_chat(
                system,
                self._wrap_prompt(render_user(example["input"]), example),
                seed=seed, **self.DECODING_PROPOSERS,
            )
            proposals.append(r.text)
            in_tok += r.input_tokens
            out_tok += r.output_tokens
            turns.append({"role": f"proposer_{i+1}", "text": r.text,
                          "input_tokens": r.input_tokens,
                          "output_tokens": r.output_tokens})

        # Majority pre-aggregation, then verifier final pass.
        majority = Counter(p.strip() for p in proposals).most_common(1)[0][0]
        verify_input = "\nCandidates:\n" + "\n".join(f"- {p}" for p in proposals)
        verify_input += f"\nMajority: {majority}"
        v = self.backend.system_chat(
            VERIFIER_SYSTEM,
            self._wrap_prompt(render_user(example["input"], verify_input), example),
            seed=self.seed, **self.DECODING_VERIFIER,
        )
        in_tok += v.input_tokens
        out_tok += v.output_tokens
        turns.append({"role": "verifier", "text": v.text,
                      "input_tokens": v.input_tokens,
                      "output_tokens": v.output_tokens})

        elapsed = max(self._now() - t0, sum(t.get("latency", 0.0) or 0.0 for t in turns))

        return Execution(
            answer=v.text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_seconds=elapsed,
            treatment_id=self.id,
            treatment_hash=self.treatment_hash,
            turns=turns,
            meta={"majority": majority},
        )
