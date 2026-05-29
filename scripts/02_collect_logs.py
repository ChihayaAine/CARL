#!/usr/bin/env python3
"""Collect log episodes under the mixed behavior policy.

Writes:
    runs/<task>/logs/episodes.jsonl
    runs/<task>/logs/calib.jsonl
    runs/<task>/logs/full_matrix.jsonl   (if cfg.splits includes 'full_matrix')
"""
from __future__ import annotations

import argparse

from carl.config import load_config, task_dir
from carl.runner.pipeline import run_collect, _split_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None,
                    help="override total dataset size for fast smoke tests")
    args = ap.parse_args()

    cfg = load_config(args.task)
    if args.n is not None:
        total = sum(int(v) for v in cfg.splits.values())
        if total > 0:
            scale = args.n / total
            cfg.splits = {k: max(int(v * scale), 1) for k, v in cfg.splits.items()}
    splits = _split_pool(cfg, seed=args.seed)
    logs_dir = task_dir(cfg.task, "logs")

    train_path = run_collect(cfg, behavior_split=splits.get("train", []),
                             full_matrix_split=splits.get("full_matrix", []),
                             out_dir=logs_dir, seed=args.seed)
    print(f"wrote {train_path}")

    cal_dir = task_dir(cfg.task, "logs_calib")
    cal_path = run_collect(cfg, behavior_split=splits.get("calib", []),
                           out_dir=cal_dir, seed=args.seed + 1)
    final = logs_dir / "calib.jsonl"
    cal_path.rename(final)
    print(f"wrote {final}")


if __name__ == "__main__":
    main()
