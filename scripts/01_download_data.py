#!/usr/bin/env python3
"""Eagerly load benchmark datasets so subsequent stages run offline.

Usage:
    python scripts/01_download_data.py --task math --n 500
    python scripts/01_download_data.py --task qa --n 1000
    python scripts/01_download_data.py --task code --n 300

Falls back gracefully to synthetic data when the real dataset cannot be
fetched (the loader logs a warning).
"""
from __future__ import annotations

import argparse

from carl.config import load_config
from carl.data.loaders import load_examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math", choices=["math", "qa", "code"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.task)
    ex = load_examples(args.task, split="train", n=args.n, cfg=cfg, seed=args.seed)
    print(f"loaded {len(ex)} {args.task} examples; benchmark_id={ex[0]['benchmark_id'] if ex else 'n/a'}")


if __name__ == "__main__":
    main()
