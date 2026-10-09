#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=run/setup_llmplan_env.sh
source "$ROOT/run/setup_llmplan_env.sh"

MASTER_ROOT="${HSA_DOMAIN_RESULTS_ROOT:-${HSA_RESULTS_ROOT:?set HSA_RESULTS_ROOT}}"
DATA_ROOT="${HSA_DATA_ROOT:?set HSA_DATA_ROOT}"
CATALOG="${HSA_CATALOG:?set HSA_CATALOG}"
SPLIT="${HSA_SPLIT:-all}"
STATUS_FILE="$MASTER_ROOT/experiment_status.tsv"
ANY_FAILURE=0

mkdir -p "$MASTER_ROOT"
[[ -f "$STATUS_FILE" ]] || printf 'experiment\tstatus\texit_code\toutput\n' > "$STATUS_FILE"
if [[ "${HSA_SKIP_PREFLIGHT:-0}" != 1 ]]; then
    DATA_ROOT="$DATA_ROOT" CATALOG="$CATALOG" SPLIT="$SPLIT" bash "$ROOT/run/preflight.sh"
fi

COMMON_ARGS=(
    --data_root "$DATA_ROOT" --base_model "$HSA_BASE_MODEL" --sft_model "$HSA_SFT_MODEL"
    --catalog "$CATALOG" --fd_entry "$HSA_FD_ENTRY" --val_bin "$HSA_VAL_BIN"
    --split "$SPLIT" --seed "$HSA_SEED" --max_attempts "$HSA_MAX_ATTEMPTS"
    --fd_search "${HSA_FD_SEARCH:-astar(lmcut())}"
    --fd_timeout_s "${HSA_FD_TIMEOUT_S:-1800}" --val_timeout_s "${HSA_VAL_TIMEOUT_S:-360}"
    --generator_backend "$HSA_GENERATOR_BACKEND" --device "$HSA_DEVICE"
    --planner_backend "$HSA_PLANNER_BACKEND" --val_backend "$HSA_VAL_BACKEND"
    --max_context_tokens "${HSA_MAX_CONTEXT_TOKENS:-16384}"
    --max_new_tokens "${HSA_MAX_NEW_TOKENS:-4096}" --eir_max_new_tokens "${HSA_EIR_MAX_NEW_TOKENS:-4096}"
    --relation_max_new_tokens "${HSA_RELATION_MAX_NEW_TOKENS:-512}"
    --grounding_max_new_tokens "${HSA_GROUNDING_MAX_NEW_TOKENS:-512}"
)
[[ "$HSA_RESUME" == 1 ]] && COMMON_ARGS+=(--resume)
[[ -z "${HSA_MAX_SAMPLES:-}" ]] || COMMON_ARGS+=(--max_samples "$HSA_MAX_SAMPLES")
[[ -z "${HSA_BEGIN_IDX:-}" ]] || COMMON_ARGS+=(--begin_idx "$HSA_BEGIN_IDX")
[[ -z "${HSA_END_IDX:-}" ]] || COMMON_ARGS+=(--end_idx "$HSA_END_IDX")
[[ -z "${HSA_MOCK_RESPONSE_DIR:-}" ]] || COMMON_ARGS+=(--mock_response_dir "$HSA_MOCK_RESPONSE_DIR")
[[ -z "${HSA_ANNOTATIONS_ROOT:-}" ]] || COMMON_ARGS+=(--annotations_root "$HSA_ANNOTATIONS_ROOT")
[[ "${HSA_REQUIRE_GOLD_ANNOTATIONS:-0}" != 1 ]] || COMMON_ARGS+=(--require_gold_annotations)

for experiment in $HSA_EXPERIMENTS; do
    output="$MASTER_ROOT/$experiment"
    echo "[EXPERIMENT] $experiment -> $output"
    set +e
    bash "$ROOT/run/run_single.sh" --experiment "$experiment" --output "$output" "${COMMON_ARGS[@]}" 2>&1 | tee "$MASTER_ROOT/${experiment}.console.log"
    rc=${PIPESTATUS[0]}
    set -e
    if [[ "$rc" == 0 ]]; then
        printf '%s\tSUCCESS\t0\t%s\n' "$experiment" "$output" >> "$STATUS_FILE"
    else
        printf '%s\tFAILED\t%s\t%s\n' "$experiment" "$rc" "$output" >> "$STATUS_FILE"
        ANY_FAILURE=1
    fi
done

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/evaluation/summarize.py" --master_root "$MASTER_ROOT" \
    --bootstrap_iterations "${HSA_BOOTSTRAP_ITERATIONS:-5000}" --seed "$HSA_SEED"
[[ "$ANY_FAILURE" == 0 ]] || { echo "[FATAL] One or more factorial cells failed; inspect $STATUS_FILE" >&2; exit 1; }
echo "[DONE] $MASTER_ROOT/all_experiments_summary.md"
