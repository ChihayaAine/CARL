"""Dataset loaders.

Real benchmark data are loaded from HuggingFace `datasets` when available; if
that is unavailable (for example in offline tests), a deterministic synthetic
dataset of the requested size is returned instead.

Each example produced by :func:`normalize_example` carries:
    id              : stable identifier
    input           : the prompt body fed to a protocol's solver
    answer          : the gold answer (used by the rewards module)
    task            : math|qa|code
    benchmark_id    : data source identifier (math500|musique|2wiki|swe_lite|synth)
    difficulty_hint : float in [0, 1] used only by the mock backend
    meta            : task-specific extras (passages, repo, instance_id, ...)
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger

log = get_logger(__name__)


def _stable_difficulty(text: str) -> float:
    h = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(),
                       "little")
    return (h % 1000) / 1000.0


def normalize_example(raw: Dict[str, Any], *, task: str,
                      benchmark_id: str, idx: int) -> Dict[str, Any]:
    eid = raw.get("id") or raw.get("instance_id") or f"{benchmark_id}-{idx}"
    if task == "math":
        question = raw.get("problem") or raw.get("question") or raw.get("input") or ""
        answer = raw.get("answer") or raw.get("solution") or raw.get("final_answer") or ""
        return {
            "id": str(eid), "input": str(question), "answer": str(answer),
            "task": task, "benchmark_id": benchmark_id,
            "difficulty_hint": _stable_difficulty(question),
            "meta": {"problem_class": raw.get("type") or raw.get("level") or ""},
        }
    if task == "qa":
        question = raw.get("question") or raw.get("input") or ""
        answer = (raw.get("answer") or raw.get("answers")
                  or raw.get("final_answer") or "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        # gather passages
        ctx = raw.get("paragraphs") or raw.get("supports") or raw.get("context") or []
        if isinstance(ctx, dict):
            ctx = list(ctx.values())
        passages = []
        for p in ctx if isinstance(ctx, list) else []:
            if isinstance(p, dict):
                passages.append(p.get("paragraph_text") or p.get("text") or "")
            else:
                passages.append(str(p))
        return {
            "id": str(eid),
            "input": _qa_render(question, passages),
            "answer": str(answer),
            "task": task, "benchmark_id": benchmark_id,
            "difficulty_hint": _stable_difficulty(question),
            "meta": {"passages": passages, "hop_count": raw.get("hop_count") or 1},
        }
    if task == "code":
        problem = raw.get("problem_statement") or raw.get("issue") or raw.get("text") or ""
        repo = raw.get("repo") or "unknown"
        instance_id = raw.get("instance_id") or raw.get("id") or f"{benchmark_id}-{idx}"
        gold_patch = raw.get("patch") or raw.get("gold_patch") or ""
        return {
            "id": str(instance_id), "input": str(problem),
            "answer": str(gold_patch),
            "task": task, "benchmark_id": benchmark_id,
            "difficulty_hint": _stable_difficulty(problem),
            "meta": {
                "repo": repo,
                "base_commit": raw.get("base_commit") or "",
                "instance_id": instance_id,
                "files": raw.get("files") or [],
            },
        }
    raise ValueError(f"Unknown task: {task!r}")


def _qa_render(question: str, passages: List[str]) -> str:
    body = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages[:5]))
    if body:
        return f"Context:\n{body}\n\nQuestion: {question}"
    return f"Question: {question}"


def load_examples(task: str, split: str = "test", n: Optional[int] = None,
                  *, cfg=None, seed: int = 0) -> List[Dict[str, Any]]:
    """Load examples for ``task``/``split``.

    Tries HuggingFace `datasets` first. Falls back to a deterministic
    synthetic dataset if `datasets` is missing, the network is unavailable,
    or the named split is not found.
    """
    try:
        if task == "math":
            return _load_math(split, n, seed)
        if task == "qa":
            return _load_qa(split, n, seed)
        if task == "code":
            return _load_code(split, n, seed)
    except Exception as e:  # noqa: BLE001
        log.warning("Falling back to synthetic %s data (cause: %s)", task, e)
    return load_synthetic(task, n or 200, seed=seed, split=split)


# --------------------------------------------------------------- benchmarks
def _load_math(split: str, n: Optional[int], seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    if split == "test":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    else:
        ds = load_dataset("hendrycks/competition_math",
                          split="train" if split in {"train", "val", "calib"} else split)
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None:
        rows = rows[:n]
    return [normalize_example(r, task="math", benchmark_id="math500", idx=i)
            for i, r in enumerate(rows)]


def _load_qa(split: str, n: Optional[int], seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    out: List[Dict[str, Any]] = []
    try:
        ds1 = load_dataset("dgslibisey/MuSiQue", split="validation")
        out.extend(normalize_example(r, task="qa", benchmark_id="musique", idx=i)
                   for i, r in enumerate(ds1))
    except Exception as e:  # noqa: BLE001
        log.warning("MuSiQue load failed: %s", e)
    try:
        ds2 = load_dataset("voidful/2WikiMultihopQA", split="validation")
        out.extend(normalize_example(r, task="qa", benchmark_id="2wiki", idx=i)
                   for i, r in enumerate(ds2))
    except Exception as e:  # noqa: BLE001
        log.warning("2Wiki load failed: %s", e)
    rng = random.Random(seed)
    rng.shuffle(out)
    if n is not None:
        out = out[:n]
    return out


def _load_code(split: str, n: Optional[int], seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    rows = list(ds)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None:
        rows = rows[:n]
    return [normalize_example(r, task="code", benchmark_id="swe_lite", idx=i)
            for i, r in enumerate(rows)]


# ---------------------------------------------------------------- synthetic
def load_synthetic(task: str, n: int, *, seed: int = 0,
                   split: str = "train") -> List[Dict[str, Any]]:
    """Deterministic synthetic data used when the real dataset is unavailable.

    The mock backend uses ``answer`` / ``difficulty_hint`` to produce
    heterogeneous-but-reproducible reward signals so the DR / LCB pipeline can
    be smoke-tested without network access.
    """
    rng = random.Random(hash((task, split, seed)) & 0x7FFFFFFF)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        diff = rng.random()
        gid = f"{task}-{split}-{i}"
        if task == "math":
            x = rng.randint(2, 99)
            y = rng.randint(2, 99)
            inp = f"Compute {x} + {y}."
            ans = str(x + y)
        elif task == "qa":
            colors = ["red", "blue", "green", "yellow"]
            c = rng.choice(colors)
            inp = f"Context:\n[1] The sky is {c} in this story.\n\nQuestion: What colour is the sky?"
            ans = c
        elif task == "code":
            issue = f"Function should return {i + 1}; it currently returns {i}."
            inp = f"Issue: {issue}\n\nFiles:\n- src/util.py (1 file)"
            ans = f"<patch>diff --git a/src/util.py +++ b/src/util.py\n-return {i}\n+return {i+1}\n</patch>"
        else:
            raise ValueError(task)
        out.append({
            "id": gid, "input": inp, "answer": ans,
            "task": task, "benchmark_id": "synth",
            "difficulty_hint": diff,
            "meta": {},
        })
    return out
