#!/usr/bin/env python3
"""Run the entire pipeline end-to-end on a task with a single command."""
from __future__ import annotations

import argparse

from carl.runner.pipeline import run_full_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="math", choices=["math", "qa", "code"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=None,
                    help="override total dataset size for a fast smoke run")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the solo-probe entropy feature (saves API calls)")
    args = ap.parse_args()

    state = run_full_pipeline(task=args.task, seed=args.seed,
                              n_examples=args.n,
                              run_solo_probe=not args.no_probe)
    for s in state["eval_state"]["summaries"]:
        chr_s = "N/A" if s.chr is None else f"{s.chr:.3f}"
        print(f"{s.name:<28s} util={s.util:+.3f} res={s.res:.3f} chr={chr_s} n={s.n}")


if __name__ == "__main__":
    main()
