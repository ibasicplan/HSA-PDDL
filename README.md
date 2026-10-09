# GPT-OSS-20B × HSA: 2×2 Ablation for Reliable PDDL Generation

Two factors over GPT-OSS-20B — SFT (on/off) and hierarchical semantic abstraction HSA (on/off) — yield four generation paths.

| Experiment | SFT | HSA | Path |
|---|---:|---:|---|
| `base_direct` | × | × | NL → PDDL |
| `sft_direct` | ✓ | × | NL → PDDL |
| `base_hsa` | × | ✓ | NL → EIR → AIR → GIR → PDDL |
| `sft_hsa` | ✓ | ✓ | NL → EIR → AIR → GIR → PDDL |

## Setup

```bash
cd HSA_PDDL_GPTOSS_2x2

export HSA_BASE_MODEL=/path/to/base/checkpoint
export HSA_SFT_MODEL=/path/to/sft/checkpoint      # different dir from HSA_BASE_MODEL
export HSA_FD_ENTRY=/path/to/fast-downward.py     # Fast Downward planner
export HSA_VAL_BIN=/path/to/Validate              # VAL validator

source ./run/setup_llmplan_env.sh
"$PYTHON_BIN" -m pip install -r requirements.txt
```

`PYTHON_BIN` defaults to `python3`.

## Self-tests (no model loading)

```bash
bash run/run_tests.sh
```

## Run

Dataset views (mystery/quantum/rovers) are prepared automatically.

```bash
bash run/run_2x2_all_domains.sh                 # all five domains
bash run/run_2x2_domain.sh zenotravel           # one domain: zenotravel|logistics|mystery|quantum|rovers
HSA_DOMAINS="zenotravel logistics" bash run/run_2x2_all_domains.sh
```

Results land under `results/` (override with `HSA_RESULTS_ROOT`). Resume is on by default.

## Common controls

```bash
export HSA_MAX_SAMPLES=20                    # smoke: limit samples per domain
export HSA_BEGIN_IDX=1 HSA_END_IDX=100
export HSA_MAX_ATTEMPTS=3 HSA_SEED=123
export HSA_BOOTSTRAP_ITERATIONS=5000
export CUDA_VISIBLE_DEVICES=0 AUTO_SELECT_GPU=0
```

## Outputs

Per domain: `<results>/<domain>/all_experiments_summary.{md,csv,json}`, `paired_effects.csv`, `factorial_interactions.csv`. All domains: `<results>/cross_domain_summary.{md,csv}`.
