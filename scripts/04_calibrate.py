#!/usr/bin/env python3
"""Fit the conformal quantiles on the calibration split."""
from __future__ import annotations

import argparse
import pickle

from carl.config import load_config, task_dir
from carl.runner.pipeline import run_calibrate, run_features, _split_pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.task)
    splits = _split_pool(cfg, seed=args.seed)
    calib_feats = run_features(cfg, splits.get("calib", []), seed=args.seed)
    with open(task_dir(cfg.task, "checkpoints") / "train_state.pkl", "rb") as f:
        train_pkl = pickle.load(f)
    train_state = train_pkl["state"]
    logs_dir = task_dir(cfg.task, "logs")
    cal_state = run_calibrate(cfg, features=calib_feats,
                              train_state=train_state,
                              calib_episodes_path=logs_dir / "calib.jsonl",
                              seed=args.seed)
    out = task_dir(cfg.task, "checkpoints") / "calib_state.pkl"
    with open(out, "wb") as f:
        pickle.dump(cal_state, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
