"""End-to-end pipeline orchestration.

Each module here corresponds to one stage of the paper's training and
deployment pipeline. They are designed to be callable both as Python
functions (for tests / smoke runs) and via the ``scripts/`` CLIs.
"""
from .pipeline import (
    run_collect,
    run_features,
    run_train,
    run_calibrate,
    run_evaluate,
    run_full_pipeline,
)

__all__ = [
    "run_collect", "run_features", "run_train",
    "run_calibrate", "run_evaluate", "run_full_pipeline",
]
