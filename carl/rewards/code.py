"""SWE-bench Lite resolved indicator + textual judge proxy.

The official ``swebench`` harness is wired up but disabled by default. When
``use_real_harness`` is False the textual judge proxy (Appendix C of the
paper) substitutes for the resolved indicator. Both routines return a float
in ``[0, 1]``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional


_PATCH = re.compile(r"<patch>(.+?)</patch>", flags=re.DOTALL)


def _extract_patch(pred: str) -> str:
    m = _PATCH.search(pred or "")
    return (m.group(1) if m else (pred or "")).strip()


def code_textual_judge(prediction: str, target_patch: str) -> float:
    """Lightweight overlap-based judge proxy.

    Compares the predicted patch against the gold patch on three signals
    paraphrasing the rubric in Appendix C: localization (changed-line
    overlap), edit minimality (length ratio), apparent correctness
    (line-level intersection-over-union of added lines).
    """
    pred = _extract_patch(prediction)
    gold = _extract_patch(target_patch)
    if not gold:
        return 0.0
    if not pred:
        return 0.0

    def added(t: str):
        return [l[1:].strip() for l in t.splitlines()
                if l.startswith("+") and not l.startswith("+++")]

    def removed(t: str):
        return [l[1:].strip() for l in t.splitlines()
                if l.startswith("-") and not l.startswith("---")]

    p_add, g_add = set(added(pred)), set(added(gold))
    p_rem, g_rem = set(removed(pred)), set(removed(gold))

    def iou(a, b):
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    add_score = iou(p_add, g_add)
    rem_score = iou(p_rem, g_rem)

    # edit minimality: penalise patches that are >2x larger than gold
    lp, lg = max(len(pred), 1), max(len(gold), 1)
    min_score = 1.0 - min(abs(lp - lg) / max(lg * 2.0, 1.0), 1.0)

    score = 0.5 * add_score + 0.3 * rem_score + 0.2 * min_score
    return float(max(0.0, min(score, 1.0)))


def code_resolved(prediction: str, example: Dict[str, Any], *,
                  use_real_harness: bool = False,
                  timeout_seconds: int = 900) -> float:
    """Return the SWE-bench Lite resolved indicator.

    If ``use_real_harness`` is True we delegate to the official ``swebench``
    package. Otherwise we use the textual judge proxy.
    """
    if not use_real_harness:
        return code_textual_judge(prediction, example.get("answer") or "")

    try:
        from swebench.harness.run_evaluation import main as swe_main  # type: ignore  # noqa: F401
    except Exception:
        # Fall back gracefully if swebench is not installed at runtime.
        return code_textual_judge(prediction, example.get("answer") or "")

    return _real_harness_run(prediction, example, timeout_seconds)


def _real_harness_run(prediction: str, example: Dict[str, Any],
                      timeout_seconds: int) -> float:
    """Adapter to the official SWE-bench Lite harness.

    We write the predicted patch + instance_id to a temporary file and call
    the official ``swebench.harness`` runner. Returns 1.0 if the harness
    reports the instance as resolved, 0.0 otherwise. Any execution failure
    (docker not running, image build failure, timeout) returns 0.0.
    """
    import json
    import subprocess
    import tempfile
    from pathlib import Path

    patch = _extract_patch(prediction)
    instance_id = example.get("meta", {}).get("instance_id") or example.get("id")
    if not instance_id or not patch:
        return 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pred_file = tmp / "predictions.jsonl"
        with open(pred_file, "w") as f:
            f.write(json.dumps({
                "instance_id": instance_id,
                "model_name_or_path": "carl-protocol",
                "model_patch": patch,
            }) + "\n")
        report_dir = tmp / "report"
        report_dir.mkdir()
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--predictions_path", str(pred_file),
            "--dataset_name", "princeton-nlp/SWE-bench_Lite",
            "--max_workers", "1",
            "--run_id", "carl",
            "--report_dir", str(report_dir),
            "--instance_ids", str(instance_id),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=timeout_seconds)
        except Exception:
            return 0.0
        if res.returncode != 0:
            return 0.0
        # Inspect the report file
        for path in report_dir.glob("**/*.json"):
            try:
                with open(path, "r") as f:
                    report = json.load(f)
                resolved = report.get("resolved", [])
                if isinstance(resolved, list):
                    return 1.0 if instance_id in resolved else 0.0
                if isinstance(resolved, dict):
                    return 1.0 if resolved.get(instance_id) else 0.0
            except Exception:
                continue
        return 0.0
