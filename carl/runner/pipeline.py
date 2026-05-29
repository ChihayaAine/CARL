"""High-level orchestration.

The pipeline stages:

1. ``run_collect``    : execute every example with a behavior policy and
                        write per-episode logs (also writes the full-matrix
                        audit episodes when configured).
2. ``run_features``   : build the pre-treatment feature bundle for a split
                        and cache it on disk.
3. ``run_train``      : cross-fit nuisances, build DR scores, shrink, and
                        fit the advantage ensemble. Cached on disk.
4. ``run_calibrate``  : fit conformal quantiles on the calibration split.
5. ``run_evaluate``   : score CARL and every baseline on the test split and
                        write ``runs/<task>/tables/main.csv``.

``run_full_pipeline`` calls them in sequence. All artefacts live under
``runs/<task>/``.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from carl.analysis import (
    Diagnostics, Summary, diagnostics, evaluate_off_policy,
    realised_metrics_from_logs, summarize, write_main_table,
    write_summary_jsonl,
)
from carl.baselines import (
    BaselineDecision, always_arm, best_fixed_from_dr,
    cascade_debate, confidence_trigger, dm_greedy,
    naive_obs, outcome_reg_lcb, mas_router, adaptive_orch_abs,
    dr_greedy, dr_lcb_absolute, causal_routing_style,
    catalog_oracle, oracle_with_abstention,
    solo_self_consistency, solo_rerank,
)
from carl.behavior.exploit import exploit_distribution
from carl.behavior.logger import EpisodeLogger, EpisodeRecord, read_episodes
from carl.behavior.policy import behavior_policy, sample_treatment
from carl.behavior.probe import probe_distribution
from carl.calibration import fit_conformal_quantiles, apply_lcb
from carl.config import Config, load_config, task_dir
from carl.data.loaders import load_examples
from carl.data.splits import make_splits
from carl.dr import build_dr_scores, shrink_to_direct, fit_advantage_ensemble
from carl.features.encoder import build_encoder
from carl.features.extract import featurize_examples, FeatureBundle, SIDE_DIM
from carl.llm.backend import build_backend
from carl.nuisances import crossfit_predict, NuisancePredictions
from carl.policy import lcb_policy
from carl.protocols.catalog import (
    PROTOCOL_INDEX, SOLO_ID, catalog_for_task, protocol_index,
)
from carl.rewards.router import score_example
from carl.utils.logging import get_logger
from carl.utils.seeding import set_seed

log = get_logger(__name__)


# --------------------------------------------------------------------- helpers

def _split_pool(cfg: Config, seed: int) -> Dict[str, List[dict]]:
    """Load examples for the task and cut them into splits per cfg.splits."""
    n_total = sum(int(v) for v in cfg.splits.values()) if cfg.splits else 200
    pool = load_examples(cfg.task, split="train", n=n_total, cfg=cfg, seed=seed)
    if not pool:
        pool = load_examples(cfg.task, split="test", n=n_total, cfg=cfg, seed=seed)
    sizes = dict(cfg.splits) if cfg.splits else {
        "train": int(n_total * 0.5), "val": int(n_total * 0.1),
        "calib": int(n_total * 0.1), "test": int(n_total * 0.2),
        "full_matrix": int(n_total * 0.1),
    }
    return make_splits(pool, sizes, seed=seed)


def _arm_indices(cfg: Config) -> Dict[str, int]:
    return {pid: protocol_index(pid) for pid in cfg.protocols}


def _execute_one(protocol, example, *, cfg, backend) -> EpisodeRecord:
    """Run a single protocol on a single example and return an EpisodeRecord."""
    t0 = time.monotonic()
    execn = protocol.run(example)
    latency = execn.latency_seconds or (time.monotonic() - t0)
    reward = score_example(
        execn.answer, example,
        use_real_harness=cfg.use_real_harness,
        timeout_seconds=cfg.harness_timeout_seconds,
    )
    tok_cost = float(execn.input_tokens + execn.output_tokens) / max(cfg.ref_tokens, 1.0)
    lat_cost = float(latency) / max(cfg.ref_latency_seconds, 1.0)
    utility = reward - cfg.lambda_ * tok_cost - cfg.mu * lat_cost
    return EpisodeRecord(
        example_id=example["id"], task=cfg.task,
        benchmark_id=example.get("benchmark_id", ""),
        treatment_id=protocol.id,
        treatment_index=protocol_index(protocol.id),
        treatment_hash=execn.treatment_hash,
        propensity=0.0, propensity_vector=[], behavior_source="",
        reward=float(reward),
        token_cost_norm=float(tok_cost),
        latency_cost_norm=float(lat_cost),
        utility=float(utility),
        input_tokens=int(execn.input_tokens),
        output_tokens=int(execn.output_tokens),
        latency_seconds=float(latency),
        prediction=execn.answer,
        difficulty_hint=float(example.get("difficulty_hint", 0.0)),
        meta={"turns": execn.turns, "exec_meta": execn.meta},
    )


# ---------------------------------------------------------------- collect logs

def run_collect(
    cfg: Config,
    *,
    behavior_split: Sequence[dict],
    full_matrix_split: Sequence[dict] = (),
    out_dir: Optional[Path] = None,
    seed: int = 0,
) -> Path:
    """Execute protocols under the behavior policy.

    Episodes are written to ``runs/<task>/logs/episodes.jsonl``. If
    ``full_matrix_split`` is non-empty we additionally execute every protocol
    on every example and write to ``logs/full_matrix.jsonl`` so the analysis
    layer can compute realized CHR / Catalog Oracle metrics.
    """
    set_seed(seed)
    out_dir = Path(out_dir or task_dir(cfg.task, "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = build_backend(cfg)
    catalog = catalog_for_task(backend, cfg.task, seed=seed)

    arm_ids = list(cfg.protocols)
    K = len(arm_ids)
    solo_idx = arm_ids.index(SOLO_ID) if SOLO_ID in arm_ids else 0
    # cost prior for the exploit policy (later arms are more expensive)
    cost_prior = np.linspace(0.0, 1.0, K)

    episodes_path = out_dir / "episodes.jsonl"
    counts = np.ones(K, dtype=np.float64) * 0.1  # smoothing
    with EpisodeLogger(episodes_path) as logger:
        rng = np.random.default_rng(seed)
        for i, ex in enumerate(behavior_split):
            diff = float(ex.get("difficulty_hint") or 0.0)
            pi_phi = exploit_distribution(K, difficulty=diff,
                                          protocol_cost=cost_prior)
            q_psi = probe_distribution(
                K, cost=cost_prior, counts=counts,
                uncertainty=np.ones(K) * 0.1,
                cost_weight=0.2, coverage_weight=1.0, uncertainty_weight=1.0,
            )
            b = behavior_policy(pi_phi, q_psi, eps=cfg.eps_probe, b_min=cfg.b_min)
            # decide whether this row is a probe row or an exploit row
            is_probe = rng.random() < cfg.eps_probe
            if is_probe:
                k = int(rng.choice(K, p=q_psi))
                source = "probe"
            else:
                k, _, _ = sample_treatment(b, rng)
                source = "exploit"
            counts[k] += 1.0
            protocol = catalog[arm_ids[k]]
            rec = _execute_one(protocol, ex, cfg=cfg, backend=backend)
            rec.propensity = float(b[k])
            rec.propensity_vector = b.tolist()
            rec.behavior_source = source
            logger.write(rec)

    if full_matrix_split:
        fm_path = out_dir / "full_matrix.jsonl"
        with EpisodeLogger(fm_path) as logger:
            for ex in full_matrix_split:
                for pid in arm_ids:
                    rec = _execute_one(catalog[pid], ex, cfg=cfg, backend=backend)
                    rec.propensity = 1.0
                    rec.propensity_vector = [0.0] * K
                    rec.propensity_vector[protocol_index(pid)] = 1.0
                    rec.behavior_source = "full_matrix"
                    logger.write(rec)
    return episodes_path


# ----------------------------------------------------------------- featurize

def run_features(cfg: Config, examples: Sequence[dict], *,
                 out_dir: Optional[Path] = None, seed: int = 0,
                 run_solo_probe: bool = True) -> FeatureBundle:
    out_dir = Path(out_dir or task_dir(cfg.task, "features"))
    enc = build_encoder(cfg)
    backend = build_backend(cfg) if run_solo_probe else None
    bundle = featurize_examples(list(examples), enc, backend=backend,
                                run_solo_probe=run_solo_probe,
                                bm25_top_k=cfg.bm25_top_k)
    np.savez_compressed(out_dir / "features.npz",
                        X=bundle.X, enc=bundle.enc, side=bundle.side,
                        ids=np.asarray(bundle.ids))
    return bundle


# ------------------------------------------------------------------- training

def _align_episodes_to_features(bundle: FeatureBundle, episodes
                                ) -> tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]:
    """Return (X, T, R, Ct, Cl, propensity) aligned on example ids in episodes order."""
    id_to_row = {eid: i for i, eid in enumerate(bundle.ids)}
    rows: List[int] = []
    T: List[int] = []
    R: List[float] = []
    Ct: List[float] = []
    Cl: List[float] = []
    P: List[float] = []
    for ep in episodes:
        i = id_to_row.get(ep.example_id)
        if i is None:
            continue
        rows.append(i)
        T.append(int(ep.treatment_index))
        R.append(float(ep.reward))
        Ct.append(float(ep.token_cost_norm))
        Cl.append(float(ep.latency_cost_norm))
        P.append(float(ep.propensity))
    X = bundle.X[rows]
    return (X, np.asarray(T), np.asarray(R), np.asarray(Ct),
            np.asarray(Cl), np.asarray(P))


def run_train(cfg: Config, *, features: FeatureBundle,
              episodes_path: Path, seed: int = 0) -> Dict[str, Any]:
    """Cross-fit nuisances + DR + shrinkage + advantage ensemble."""
    set_seed(seed)
    episodes = read_episodes(episodes_path)
    X, T, R, Ct, Cl, _ = _align_episodes_to_features(features, episodes)
    K = len(cfg.protocols)
    nuis = crossfit_predict(
        X, T, R, Ct, Cl, K=K,
        n_folds=cfg.n_folds, n_boot=cfg.n_boot,
        nuisance_kind=cfg.nuisance_kind,
        propensity_kind=cfg.propensity_kind,
        e_min=cfg.e_min, seed=seed,
    )
    dr = build_dr_scores(nuis, T=T, R=R, Ct=Ct, Cl=Cl,
                         lam=cfg.lambda_, mu=cfg.mu,
                         solo_index=cfg.protocols.index(SOLO_ID))
    shrunk = shrink_to_direct(dr.psi, dr.direct_contrast(), X,
                              K_NN=cfg.K_NN, gamma=cfg.gamma)
    adv = fit_advantage_ensemble(X, shrunk.advantage,
                                 solo_index=cfg.protocols.index(SOLO_ID),
                                 n_boot=cfg.n_boot, seed=seed)
    return {
        "nuisances": nuis, "dr": dr, "shrunk": shrunk, "advantage": adv,
        "X_train": X, "T_train": T, "R_train": R, "Ct_train": Ct, "Cl_train": Cl,
    }


# --------------------------------------------------------------- calibration

def run_calibrate(cfg: Config, *, features: FeatureBundle, train_state,
                  calib_episodes_path: Path, seed: int = 0):
    set_seed(seed)
    eps = read_episodes(calib_episodes_path)
    X, T, R, Ct, Cl, _ = _align_episodes_to_features(features, eps)
    K = len(cfg.protocols)
    nuis = crossfit_predict(
        X, T, R, Ct, Cl, K=K,
        n_folds=cfg.n_folds, n_boot=cfg.n_boot,
        nuisance_kind=cfg.nuisance_kind,
        propensity_kind=cfg.propensity_kind,
        e_min=cfg.e_min, seed=seed,
    )
    dr_cal = build_dr_scores(nuis, T=T, R=R, Ct=Ct, Cl=Cl,
                             lam=cfg.lambda_, mu=cfg.mu,
                             solo_index=cfg.protocols.index(SOLO_ID))
    adv = train_state["advantage"]
    a_hat, sigma_hat = adv.predict(X)
    cal = fit_conformal_quantiles(
        dr_cal.psi, a_hat, sigma_hat,
        delta=cfg.delta, solo_index=cfg.protocols.index(SOLO_ID),
    )
    # Also fit a calibrator on absolute utility for DR-LCB(abs) baseline
    abs_target = nuis.mu_Y(cfg.lambda_, cfg.mu)
    abs_hat = abs_target  # use the same nuisance mean as the model
    abs_sigma = nuis.sigma_R
    cal_abs = fit_conformal_quantiles(
        abs_target, abs_hat, abs_sigma, delta=cfg.delta,
        solo_index=cfg.protocols.index(SOLO_ID),
    )
    return {"cal": cal, "cal_abs": cal_abs, "X_cal": X, "T_cal": T,
            "R_cal": R, "Ct_cal": Ct, "Cl_cal": Cl, "nuis_cal": nuis}


# ----------------------------------------------------------------- evaluation

def _utility(R, Ct, Cl, lam, mu):
    return R - lam * Ct - mu * Cl


def _expand_full_matrix(episodes, ids: Sequence[str], K: int, lam: float, mu: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Build (N_test, K) utility and reward matrices from full-matrix episodes."""
    id_to_row = {eid: i for i, eid in enumerate(ids)}
    util = np.full((len(ids), K), np.nan, dtype=np.float64)
    res = np.full((len(ids), K), np.nan, dtype=np.float64)
    for ep in episodes:
        i = id_to_row.get(ep.example_id)
        if i is None:
            continue
        k = int(ep.treatment_index)
        if 0 <= k < K:
            util[i, k] = float(ep.utility)
            res[i, k] = float(ep.reward)
    # rows with missing entries fall back to row mean to keep tables computable
    for j in range(K):
        col = util[:, j]
        mask = np.isnan(col)
        if mask.any():
            fill = np.nanmean(col) if np.isfinite(np.nanmean(col)) else 0.0
            util[mask, j] = fill
        colr = res[:, j]
        mask = np.isnan(colr)
        if mask.any():
            fill = float(np.nanmean(colr)) if np.isfinite(np.nanmean(colr)) else 0.0
            res[mask, j] = fill
    return util, res


def run_evaluate(
    cfg: Config,
    *,
    features_test: FeatureBundle,
    train_state,
    calib_state,
    full_matrix_episodes_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    set_seed(seed)
    out_dir = Path(out_dir or task_dir(cfg.task, "tables"))
    X_te = features_test.X
    K = len(cfg.protocols)
    solo_idx = cfg.protocols.index(SOLO_ID)

    # ---- CARL: advantage + LCB
    adv = train_state["advantage"]
    a_hat, sigma_hat = adv.predict(X_te)
    cal = calib_state["cal"]
    cal_abs = calib_state["cal_abs"]
    carl = lcb_policy(a_hat, sigma_hat, cal,
                      kappa=cfg.kappa, abstain_threshold=0.0)

    # ---- baselines
    X_tr = train_state["X_train"]
    T_tr = train_state["T_train"]
    R_tr = train_state["R_train"]
    Ct_tr = train_state["Ct_train"]
    Cl_tr = train_state["Cl_train"]
    Y_tr = _utility(R_tr, Ct_tr, Cl_tr, cfg.lambda_, cfg.mu)

    decisions: List[BaselineDecision] = []
    decisions.append(BaselineDecision(name="CARL", decisions=carl.decisions.astype(np.int64)))

    # Fixed arms
    for name in cfg.protocols:
        decisions.append(always_arm(X_te.shape[0], protocol_index(name),
                                    name=f"Always-{name}"))
    decisions.append(best_fixed_from_dr(train_state["dr"].psi,
                                        N_test=X_te.shape[0],
                                        solo_index=solo_idx,
                                        arm_names=cfg.protocols))

    # Predictive
    decisions.append(dm_greedy(X_tr, Y_tr, T_tr, X_te, K,
                               solo_index=solo_idx, seed=seed))
    decisions.append(outcome_reg_lcb(X_tr, Y_tr, T_tr, X_te, K,
                                     cal=cal, kappa=cfg.kappa,
                                     solo_index=solo_idx, seed=seed))
    decisions.append(adaptive_orch_abs(X_tr, Y_tr, T_tr, X_te, K,
                                       solo_index=solo_idx, seed=seed))
    decisions.append(naive_obs(T_tr, Y_tr, X_te, K, solo_index=solo_idx))
    decisions.append(mas_router(X_tr, Y_tr, T_tr, X_te, K,
                                with_solo=False, solo_index=solo_idx, seed=seed))
    decisions.append(mas_router(X_tr, Y_tr, T_tr, X_te, K,
                                with_solo=True, solo_index=solo_idx, seed=seed))
    # Confidence trigger / cascade-debate use the solo entropy feature
    ent = X_te[:, -SIDE_DIM + 10]  # index 10 within side block
    if PROTOCOL_INDEX.get("VERIFY") is not None:
        decisions.append(confidence_trigger(
            ent, solo_index=solo_idx,
            collab_index=PROTOCOL_INDEX["VERIFY"],
            threshold=float(np.median(ent)),
        ))
    if PROTOCOL_INDEX.get("DEBATE_2") is not None and PROTOCOL_INDEX.get("VERIFY") is not None:
        q1, q2 = np.quantile(ent, [0.5, 0.8]) if ent.size else (0.0, 1.0)
        decisions.append(cascade_debate(
            ent, solo_index=solo_idx,
            verify_index=PROTOCOL_INDEX["VERIFY"],
            debate_index=PROTOCOL_INDEX["DEBATE_2"],
            theta_low=float(q1), theta_high=float(q2),
        ))

    # Causal-flavored baselines
    decisions.append(dr_greedy(a_hat, solo_index=solo_idx))
    decisions.append(dr_lcb_absolute(
        X_tr, train_state["nuisances"], Y_tr, T_tr, X_te,
        lam=cfg.lambda_, mu=cfg.mu,
        cal_abs=cal_abs, kappa=cfg.kappa, solo_index=solo_idx, seed=seed,
    ))
    decisions.append(causal_routing_style(
        X_tr, train_state["nuisances"], Y_tr, T_tr, X_te,
        lam=cfg.lambda_, mu=cfg.mu,
        with_solo=False, solo_index=solo_idx, seed=seed,
    ))
    decisions.append(causal_routing_style(
        X_tr, train_state["nuisances"], Y_tr, T_tr, X_te,
        lam=cfg.lambda_, mu=cfg.mu,
        with_solo=True, solo_index=solo_idx, seed=seed,
    ))
    decisions.append(solo_self_consistency(X_te.shape[0], solo_index=solo_idx))
    decisions.append(solo_rerank(X_te.shape[0], solo_index=solo_idx))

    # ---- Compute realized metrics if a full-matrix split was provided.
    summaries: List[Summary] = []
    if full_matrix_episodes_path and Path(full_matrix_episodes_path).exists():
        fm_eps = read_episodes(full_matrix_episodes_path)
        util_mat, res_mat = _expand_full_matrix(fm_eps, features_test.ids, K,
                                                cfg.lambda_, cfg.mu)
        decisions.append(catalog_oracle(util_mat, solo_index=solo_idx))
        decisions.append(oracle_with_abstention(util_mat, solo_index=solo_idx))
        for d in decisions:
            summaries.append(summarize(d.name, d.decisions,
                                       util_matrix=util_mat,
                                       res_matrix=res_mat,
                                       solo_index=solo_idx))
    else:
        # No full-matrix split: fall back to direct contrast estimates from
        # the advantage model (descriptive only, NOT a causal evaluation).
        for d in decisions:
            mean_adv = float(a_hat[np.arange(a_hat.shape[0]),
                                   d.decisions].mean()) if d.decisions.size else 0.0
            summaries.append(Summary(name=d.name, util=mean_adv, res=0.0,
                                     chr=None, n=int(d.decisions.shape[0])))

    write_main_table(summaries, out_dir, name="main")
    write_summary_jsonl([asdict(s) for s in summaries], out_dir / "summaries.jsonl")

    return {"decisions": decisions, "summaries": summaries,
            "carl": carl, "advantage_pred": (a_hat, sigma_hat)}


# ----------------------------------------------------------------- full run

def run_full_pipeline(
    task: str = "math",
    *,
    overrides: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    n_examples: Optional[int] = None,
    skip_collect: bool = False,
    run_solo_probe: bool = True,
) -> Dict[str, Any]:
    """Run the entire pipeline end-to-end.

    Parameters
    ----------
    n_examples : override the total dataset size (for fast smoke tests).
    skip_collect : reuse cached logs if True.
    """
    cfg = load_config(task, overrides=overrides)
    set_seed(seed)
    if n_examples is not None:
        # rescale split sizes proportionally
        total = sum(int(v) for v in cfg.splits.values())
        if total > 0:
            scale = n_examples / total
            cfg.splits = {k: max(int(v * scale), 1) for k, v in cfg.splits.items()}

    splits = _split_pool(cfg, seed=seed)
    train = splits.get("train", [])
    val = splits.get("val", [])
    calib = splits.get("calib", [])
    test = splits.get("test", [])
    # Audit subset is drawn from the test set so per-row CHR / Catalog Oracle
    # metrics can be reported against the same example ids the policies decide on.
    audit_n = min(int(cfg.splits.get("full_matrix", 0)) or len(test), len(test))
    full_matrix = list(test[:audit_n])
    splits["full_matrix"] = full_matrix

    logs_dir = task_dir(cfg.task, "logs")
    train_log = logs_dir / "episodes.jsonl"
    calib_log = logs_dir / "calib.jsonl"
    fm_log = logs_dir / "full_matrix.jsonl"

    if not skip_collect or not train_log.exists():
        log.info("collecting train logs (%d examples)", len(train))
        run_collect(cfg, behavior_split=train, full_matrix_split=full_matrix,
                    out_dir=logs_dir, seed=seed)
    if not skip_collect or not calib_log.exists():
        log.info("collecting calib logs (%d examples)", len(calib))
        # rename episodes from this sub-run so we don't append into train log
        cal_dir = task_dir(cfg.task, "logs_calib")
        path = run_collect(cfg, behavior_split=calib, out_dir=cal_dir, seed=seed + 1)
        # move into expected name
        path.rename(calib_log)

    train_feats = run_features(cfg, train, run_solo_probe=run_solo_probe, seed=seed)
    calib_feats = run_features(cfg, calib, run_solo_probe=run_solo_probe, seed=seed)
    test_feats = run_features(cfg, test, run_solo_probe=run_solo_probe, seed=seed)

    train_state = run_train(cfg, features=train_feats,
                            episodes_path=train_log, seed=seed)
    calib_state = run_calibrate(cfg, features=calib_feats,
                                train_state=train_state,
                                calib_episodes_path=calib_log, seed=seed)
    eval_state = run_evaluate(cfg, features_test=test_feats,
                              train_state=train_state, calib_state=calib_state,
                              full_matrix_episodes_path=fm_log if fm_log.exists() else None,
                              seed=seed)
    return {"cfg": cfg, "splits": splits,
            "train_state": train_state, "calib_state": calib_state,
            "eval_state": eval_state}
