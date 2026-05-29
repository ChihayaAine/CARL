"""Centralised prompt templates.

The templates are intentionally short and free of identifying information.
They cover the three task styles used by the paper (math, qa, code) and the
five protocol roles (solver, reflector, verifier, debaters, judge,
proposer-set, aggregator).
"""
from __future__ import annotations

from typing import Dict


SYSTEM_BY_STYLE: Dict[str, str] = {
    "math": (
        "You are a careful problem solver. Return only the final answer on the "
        "last line, prefixed by 'Answer:' with no extra text."
    ),
    "qa": (
        "You are a question answering assistant. Use the provided context. "
        "Return only the final answer on the last line, prefixed by 'Answer:'."
    ),
    "code": (
        "You are a software engineer. Given a repository issue and retrieved "
        "files, produce a minimal unified-diff patch that fixes the issue. "
        "Return only the patch between <patch> and </patch>."
    ),
    "generic": (
        "Solve the user's request. Return only the final answer on the last "
        "line, prefixed by 'Answer:'."
    ),
}


VERIFIER_SYSTEM = (
    "You are an independent verifier. Decide whether the candidate answer is "
    "correct given the original problem. If it is correct, return the same "
    "answer; otherwise return your best alternative. Use the same format "
    "as the original protocol (Answer:/<patch> ... </patch>)."
)

REFLECTOR_SYSTEM = (
    "You are reviewing your own first draft. Identify the most likely "
    "mistake, and revise the answer if needed. Use the same final-line "
    "format as the original protocol."
)

DEBATER_A_SYSTEM = (
    "You are debater A. Argue for an answer and refute debater B's previous "
    "turn if any. Use the same final-line format as the original protocol."
)

DEBATER_B_SYSTEM = (
    "You are debater B. Argue for an answer and refute debater A's previous "
    "turn if any. Use the same final-line format as the original protocol."
)

JUDGE_SYSTEM = (
    "You are a judge. Read the debate transcript and return the better "
    "answer. Do not introduce new content. Use the same final-line format "
    "as the original protocol."
)

PROPOSER_SYSTEM = (
    "You are proposer #{idx}. Produce one candidate answer. Use the same "
    "final-line format as the original protocol."
)


def render_user(task_input: str, history: str = "") -> str:
    if history:
        return f"{task_input.strip()}\n\nPrevious turn:\n{history.strip()}"
    return task_input.strip()
