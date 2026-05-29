"""Generic textual judge helper, used by Code's auxiliary patch-quality score
and by analysis utilities that want a quick LLM-free quality estimate."""
from __future__ import annotations

from .code import code_textual_judge

textual_judge_score = code_textual_judge
