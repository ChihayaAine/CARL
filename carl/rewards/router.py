"""Single entry point that dispatches a (prediction, example) pair to the
correct task-specific reward function and returns a float in [0, 1]."""
from __future__ import annotations

from typing import Any, Dict

from .code import code_resolved
from .math import math_exact_match
from .qa import qa_token_f1


def score_example(prediction: str, example: Dict[str, Any], *,
                  use_real_harness: bool = False,
                  timeout_seconds: int = 900) -> float:
    task = example.get("task", "math")
    target = example.get("answer", "")
    if task == "math":
        return math_exact_match(prediction, target)
    if task == "qa":
        return qa_token_f1(prediction, target)
    if task == "code":
        return code_resolved(prediction, example,
                             use_real_harness=use_real_harness,
                             timeout_seconds=timeout_seconds)
    raise ValueError(f"Unknown task: {task}")
