"""Tiny end-to-end smoke test under the mock backend.

Runs the entire pipeline on a synthetic 60-example math task and verifies
that we get a non-empty summary table with CARL in the named rows.
"""
import os

import pytest

from carl.runner.pipeline import run_full_pipeline


@pytest.mark.parametrize("task", ["math"])
def test_smoke_math_pipeline(task, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")  # force mock backend
    state = run_full_pipeline(task=task, n_examples=60,
                              run_solo_probe=False, seed=0)
    summaries = state["eval_state"]["summaries"]
    names = {s.name for s in summaries}
    assert "CARL" in names
    assert any(name.startswith("Always-") for name in names)
    assert all(isinstance(s.util, float) for s in summaries)
