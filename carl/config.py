"""Configuration loader: merges default.yaml with task-specific YAML and .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)
    task: str = "math"

    # cost coefficients
    lambda_: float = 0.05
    mu: float = 0.02

    # policy / propensity
    eps_probe: float = 0.15
    b_min: float = 0.02
    e_min: float = 0.02

    # nuisance / dr
    n_folds: int = 5
    n_boot: int = 5
    K_NN: int = 50
    gamma: float = 0.5
    nuisance_kind: str = "boost"
    propensity_kind: str = "logistic"

    # rng
    seed: int = 0

    # calibration / LCB
    delta: float = 0.10
    kappa: float = 1.0

    # optimisation
    lr: float = 2e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 5
    mlp_hidden: int = 256
    shared_hidden: int = 512
    dropout: float = 0.1

    # encoder
    encoder_kind: str = "sbert"
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_dim: int = 384

    # protocols and reward
    protocols: tuple = ("SOLO", "SELF_REFLECT", "VERIFY", "DEBATE_2", "PROPOSE_VERIFY")
    reward: str = "em"
    retrieval: Optional[str] = None
    prompt_style: str = "generic"

    # reference budgets for normalizing token / latency cost
    ref_tokens: float = 2000.0
    ref_latency_seconds: float = 30.0

    # dataset-specific
    dataset: Dict[str, Any] = field(default_factory=dict)
    splits: Dict[str, int] = field(default_factory=dict)

    # backend (loaded from env)
    backend_kind: str = "auto"  # auto | openai | mock
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "mock-model"
    llm_rpm: int = 600

    # code-task only
    bm25_top_k: int = 5
    use_real_harness: bool = False
    harness_timeout_seconds: int = 900


def load_config(task: str = "math", overrides: Optional[Dict[str, Any]] = None) -> Config:
    default = _load_yaml(CONFIGS_DIR / "default.yaml")
    task_cfg = _load_yaml(CONFIGS_DIR / f"{task}.yaml")

    merged: Dict[str, Any] = {**default, **task_cfg}
    if overrides:
        merged.update(overrides)

    cfg = Config(raw=merged, task=task)
    cfg.lambda_ = float(merged.get("lambda", cfg.lambda_))
    cfg.mu = float(merged.get("mu", cfg.mu))
    cfg.eps_probe = float(merged.get("eps_probe", cfg.eps_probe))
    cfg.b_min = float(merged.get("b_min", cfg.b_min))
    cfg.e_min = float(merged.get("e_min", cfg.e_min))
    cfg.n_folds = int(merged.get("n_folds", cfg.n_folds))
    cfg.n_boot = int(merged.get("n_boot", cfg.n_boot))
    cfg.K_NN = int(merged.get("K_NN", cfg.K_NN))
    cfg.gamma = float(merged.get("gamma", cfg.gamma))
    cfg.nuisance_kind = merged.get("nuisance_kind", cfg.nuisance_kind)
    cfg.propensity_kind = merged.get("propensity_kind", cfg.propensity_kind)
    cfg.seed = int(merged.get("seed", cfg.seed))
    cfg.delta = float(merged.get("delta", cfg.delta))
    cfg.kappa = float(merged.get("kappa", cfg.kappa))

    cfg.lr = float(merged.get("lr", cfg.lr))
    cfg.weight_decay = float(merged.get("weight_decay", cfg.weight_decay))
    cfg.batch_size = int(merged.get("batch_size", cfg.batch_size))
    cfg.max_epochs = int(merged.get("max_epochs", cfg.max_epochs))
    cfg.patience = int(merged.get("patience", cfg.patience))
    cfg.mlp_hidden = int(merged.get("mlp_hidden", cfg.mlp_hidden))
    cfg.shared_hidden = int(merged.get("shared_hidden", cfg.shared_hidden))
    cfg.dropout = float(merged.get("dropout", cfg.dropout))

    enc = merged.get("encoder", {})
    cfg.encoder_kind = enc.get("kind", cfg.encoder_kind)
    cfg.encoder_name = enc.get("name", cfg.encoder_name)
    cfg.encoder_dim = int(enc.get("dim", cfg.encoder_dim))

    cfg.protocols = tuple(merged.get("protocols", cfg.protocols))
    cfg.reward = merged.get("reward", cfg.reward)
    cfg.retrieval = merged.get("retrieval", cfg.retrieval)
    cfg.prompt_style = merged.get("prompt_style", cfg.prompt_style)

    refs = merged.get("reference_budgets", {}).get(task, {})
    cfg.ref_tokens = float(refs.get("tokens", cfg.ref_tokens))
    cfg.ref_latency_seconds = float(refs.get("latency_seconds", cfg.ref_latency_seconds))

    cfg.dataset = merged.get("dataset", {})
    cfg.splits = merged.get("splits", {}).get(task, {})

    cfg.openai_api_key = os.environ.get("OPENAI_API_KEY") or None
    cfg.openai_base_url = os.environ.get("OPENAI_BASE_URL", cfg.openai_base_url)
    cfg.model_name = os.environ.get("MODEL_NAME", cfg.model_name)
    cfg.llm_rpm = int(os.environ.get("LLM_RPM", cfg.llm_rpm))

    if cfg.backend_kind == "auto":
        cfg.backend_kind = "openai" if cfg.openai_api_key else "mock"

    cfg.bm25_top_k = int(merged.get("bm25_top_k", cfg.bm25_top_k))
    cfg.use_real_harness = bool(merged.get("use_real_harness", cfg.use_real_harness))
    cfg.harness_timeout_seconds = int(merged.get("harness_timeout_seconds", cfg.harness_timeout_seconds))

    return cfg


def task_dir(task: str, subdir: str = "") -> Path:
    d = RUNS_DIR / task
    if subdir:
        d = d / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d
