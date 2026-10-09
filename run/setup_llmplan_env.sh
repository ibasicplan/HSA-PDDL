#!/usr/bin/env bash
# Source this file before manual runs. All values may be overridden beforehand.

HSA_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HSA_PROJECT_ROOT

# Default to the system python; override with PYTHON_BIN before sourcing.
export PYTHON_BIN="${PYTHON_BIN:-python3}"

# Model and planner/validator paths have no bundled default. Set them before
# sourcing, or the preflight will report the missing location.
export HSA_BASE_MODEL="${HSA_BASE_MODEL:-}"
export HSA_SFT_MODEL="${HSA_SFT_MODEL:-}"
export HSA_FD_ENTRY="${HSA_FD_ENTRY:-}"
export HSA_VAL_BIN="${HSA_VAL_BIN:-}"

# Bundled datasets under data/.
export HSA_ZENOTRAVEL_DATA="${HSA_ZENOTRAVEL_DATA:-$HSA_PROJECT_ROOT/data/zenotravel}"
export HSA_LOGISTICS_DATA="${HSA_LOGISTICS_DATA:-$HSA_PROJECT_ROOT/data/logistics}"
export HSA_MYSTERY_SOURCE="${HSA_MYSTERY_SOURCE:-$HSA_PROJECT_ROOT/data/blocksworld_mystery}"
export HSA_QUANTUM_SOURCE="${HSA_QUANTUM_SOURCE:-$HSA_PROJECT_ROOT/data/quantum}"
export HSA_ROVERS_DATA="${HSA_ROVERS_DATA:-$HSA_PROJECT_ROOT/data/Rovers}"

# Prepared dataset views and generated artifacts.
export HSA_PREPARED_DATA_ROOT="${HSA_PREPARED_DATA_ROOT:-$HSA_PROJECT_ROOT/data/prepared}"
export HSA_MYSTERY_DATA="${HSA_MYSTERY_DATA:-$HSA_PREPARED_DATA_ROOT/mystery}"
export HSA_QUANTUM_DATA="${HSA_QUANTUM_DATA:-$HSA_PREPARED_DATA_ROOT/quantum}"
export HSA_ROVERS_CATALOG="${HSA_ROVERS_CATALOG:-$HSA_PREPARED_DATA_ROOT/rovers_relation_catalog.json}"
export HSA_RESULTS_ROOT="${HSA_RESULTS_ROOT:-$HSA_PROJECT_ROOT/results}"

export HSA_SEED="${HSA_SEED:-123}"
export HSA_MAX_ATTEMPTS="${HSA_MAX_ATTEMPTS:-3}"
export HSA_RESUME="${HSA_RESUME:-1}"
export HSA_GENERATOR_BACKEND="${HSA_GENERATOR_BACKEND:-transformers}"
export HSA_PLANNER_BACKEND="${HSA_PLANNER_BACKEND:-real}"
export HSA_VAL_BACKEND="${HSA_VAL_BACKEND:-real}"
export HSA_DEVICE="${HSA_DEVICE:-auto}"
export HSA_EXPERIMENTS="${HSA_EXPERIMENTS:-base_direct sft_direct base_hsa sft_hsa}"
