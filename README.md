# CARL: Causal Advantage Routing for Language agents

Reference implementation of the **CARL** framework for selective multi-agent
collaboration. CARL frames protocol selection as estimation of the
*conditional collaboration advantage* of each protocol over solo reasoning,
combines confounded execution logs with budgeted randomized probes, forms
one-vs-solo doubly robust targets, and deploys a calibrated lower-confidence-bound
(LCB) policy that invokes a protocol only when its estimated advantage remains
positive under uncertainty, otherwise abstaining to solo execution.

The repository is structured as a research artifact: every piece of the paper's
methodology (Eq. 1-9, Proposition 1-2) is implemented as an independently
testable module, baselines and ablations sit alongside the main method, and
the pipeline can be exercised end-to-end with either a real OpenAI-compatible
backend or a deterministic mock backend.

## Repository layout

```
carl/
  llm/           pluggable LLM backends (OpenAI-compatible, mock)
  protocols/     SOLO + 4 collaborative protocols, SHA-256 treatment hashes
  features/      pre-treatment features (length, task type, BM25, solo probe)
  rewards/       MATH EM, QA F1, SWE-bench resolved (+ textual judge proxy)
  data/          MATH-500 / MuSiQue / 2Wiki / SWE-Lite loaders + splits
  behavior/      probe q_psi, exploit pi_phi, mixed policy with b_min floor
  nuisances/     shared encoder + reward/token/latency/propensity heads + cross-fitting
  dr/            one-vs-solo DR scores, kNN-shrinkage, advantage ensemble
  calibration/   paired execution, conformal a_k quantile
  policy/        CARL LCB policy + greedy + fixed
  baselines/     fourteen baselines (fixed, predictive routers, causal ablations)
  analysis/      Util/CHR/PTC/Cov/ECE, DM/IPW/SNIPW/DR OPE, table generators
  runner/        log collection / nuisance training / calibration / evaluation
  utils/         seeding, hashing, logging
configs/         hyperparameters per task
scripts/         01_download_data -> 06_make_tables
tests/           unit tests + end-to-end smoke test under the mock backend
```

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Configure backend (optional). Without a key, the mock backend is used.
cp .env.example .env
# edit .env to set OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME

# 3. End-to-end smoke test on the mock backend (~ 30 s, no network)
bash scripts/smoke_test.sh

# 4. Real run (requires an LLM backend). Defaults are kept small.
python scripts/01_download_data.py --task math
python scripts/02_collect_logs.py --task math --n-train 200
python scripts/03_train_carl.py --task math
python scripts/04_calibrate.py --task math --n-cal 100
python scripts/05_evaluate.py --task math --policy carl
python scripts/06_make_tables.py --out tables/
```

## Hyperparameters

All hyperparameters are fixed to the values reported in the paper and
loaded from `configs/default.yaml`:

| Symbol | Value | Meaning |
| ------ | ----- | ------- |
| eps_probe | 0.15 | randomized probe fraction |
| b_min | 0.02 | behavior-policy floor |
| e_min | 0.02 | propensity clip |
| delta | 0.10 | calibration miscoverage |
| kappa | 1.0  | LCB risk knob |
| K_NN  | 50   | shrinkage neighbourhood size |
| gamma | 0.5  | shrinkage scale |
| n_folds | 5  | cross-fitting folds |
| n_boot  | 5  | advantage ensemble bootstrap size |
| lambda  | 0.05 | token cost weight |
| mu      | 0.02 | latency cost weight |

## Notes on backends and harnesses

- **LLM backend.** `OPENAI_API_KEY` + optional `OPENAI_BASE_URL` selects an
  OpenAI-compatible endpoint (use it with a real OpenAI account, OpenRouter,
  vLLM, together, etc.). Without a key the mock backend produces deterministic
  synthetic answers and costs so that tests, unit benchmarks, and the smoke
  pipeline still exercise every component.
- **SWE-bench Lite harness.** The official `swebench` package is supported
  but disabled by default. Set `--use-real-harness` on
  `02_collect_logs.py`/`05_evaluate.py` to enable; otherwise a fast textual
  judge proxy stands in for the resolved indicator (the paper documents this
  proxy as an auxiliary diagnostic in Appendix C).

## Reproducing the paper

The repository implements the algorithm exactly; the numerical values in the
paper (e.g. `.61 / .35 / .55` cost-aware utility for CARL on Math/Code/QA)
require real LLM executions on the full data splits in
`configs/{math,code,qa}.yaml` x five seeds. The runner scripts accept
`--n-train`, `--n-cal`, `--n-eval`, and `--seed` flags so that any sub-sampled
study can be reproduced from a single command.
