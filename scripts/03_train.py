#!/usr/bin/env python3
"""Train nuisances + DR + shrinkage + advantage ensemble on a task's logs."""
from __future__ import annotations

import argparse
import pickle

from carl.config import load_config, task_dir
from carl.runner.pipeline import (
    run_features, run_train, _split_pool,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the solo-probe entropy feature")
    args = ap.parse_args()

    cfg = load_config(args.task)
    splits = _split_pool(cfg, seed=args.seed)
    train_feats = run_features(cfg, splits.get("train", []),
                               run_solo_probe=not args.no_probe, seed=args.seed)
    logs_dir = task_dir(cfg.task, "logs")
    state = run_train(cfg, features=train_feats,
                      episodes_path=logs_dir / "episodes.jsonl", seed=args.seed)
    out = task_dir(cfg.task, "checkpoints") / "train_state.pkl"
    with open(out, "wb") as f:
        pickle.dump({"cfg_raw": cfg.raw, "state": state}, f)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
