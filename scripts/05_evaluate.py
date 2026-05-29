#!/usr/bin/env python3
"""Evaluate CARL and all baselines on the test split."""
from __future__ import annotations

import argparse
import pickle

from carl.config import load_config, task_dir
from carl.runner.pipeline import run_evaluate, run_features, _split_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.task)
    splits = _split_pool(cfg, seed=args.seed)
    test_feats = run_features(cfg, splits.get("test", []), seed=args.seed)
    with open(task_dir(cfg.task, "checkpoints") / "train_state.pkl", "rb") as f:
        train_state = pickle.load(f)["state"]
    with open(task_dir(cfg.task, "checkpoints") / "calib_state.pkl", "rb") as f:
        calib_state = pickle.load(f)

    logs_dir = task_dir(cfg.task, "logs")
    fm_path = logs_dir / "full_matrix.jsonl"
    state = run_evaluate(cfg, features_test=test_feats,
                         train_state=train_state, calib_state=calib_state,
                         full_matrix_episodes_path=fm_path if fm_path.exists() else None,
                         seed=args.seed)
    out = task_dir(cfg.task, "tables") / "main.csv"
    print(f"wrote {out}")
    for s in state["summaries"]:
        chr_s = "N/A" if s.chr is None else f"{s.chr:.3f}"
        print(f"  {s.name:<28s} util={s.util:+.3f} res={s.res:.3f} chr={chr_s} n={s.n}")


if __name__ == "__main__":
    main()
