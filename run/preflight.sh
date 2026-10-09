#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=run/setup_llmplan_env.sh
source "$ROOT/run/setup_llmplan_env.sh"

DATA_ROOT="${DATA_ROOT:-${HSA_DATA_ROOT:-}}"
CATALOG="${CATALOG:-${HSA_CATALOG:-}}"
SPLIT="${SPLIT:-${HSA_SPLIT:-all}}"

[[ -n "$DATA_ROOT" && -d "$DATA_ROOT" ]] || { echo "[FATAL] Dataset root not found: ${DATA_ROOT:-<unset>}" >&2; exit 2; }
[[ -n "$CATALOG" && -f "$CATALOG" ]] || { echo "[FATAL] Catalog not found: ${CATALOG:-<unset>}" >&2; exit 2; }
[[ -n "${HSA_BASE_MODEL:-}" ]] || {
    echo "[FATAL] Base GPT-OSS-20B was not auto-detected. Export HSA_BASE_MODEL=/absolute/path/to/base/checkpoint" >&2
    exit 2
}
[[ -d "$HSA_BASE_MODEL" ]] || { echo "[FATAL] Base model directory not found: $HSA_BASE_MODEL" >&2; exit 2; }
[[ -d "$HSA_SFT_MODEL" ]] || { echo "[FATAL] SFT model directory not found: $HSA_SFT_MODEL" >&2; exit 2; }
[[ "$(realpath "$HSA_BASE_MODEL")" != "$(realpath "$HSA_SFT_MODEL")" ]] || {
    echo "[FATAL] Base and SFT checkpoints resolve to the same directory; this invalidates the SFT factor." >&2
    exit 2
}

if [[ "$HSA_PLANNER_BACKEND" == real ]]; then
    [[ -f "$HSA_FD_ENTRY" ]] || { echo "[FATAL] Fast Downward entry not found: $HSA_FD_ENTRY" >&2; exit 2; }
fi
if [[ "$HSA_VAL_BACKEND" == real ]]; then
    [[ -x "$HSA_VAL_BIN" ]] || { echo "[FATAL] VAL executable not found/executable: $HSA_VAL_BIN" >&2; exit 2; }
fi

count=0
sample_roots=()
if [[ -d "$DATA_ROOT/easy" || -d "$DATA_ROOT/hard" ]]; then
    for chosen_split in easy hard; do
        [[ "$SPLIT" == all || "$SPLIT" == "$chosen_split" ]] || continue
        [[ -d "$DATA_ROOT/$chosen_split" ]] || { echo "[FATAL] Missing split: $DATA_ROOT/$chosen_split" >&2; exit 2; }
        sample_roots+=("$DATA_ROOT/$chosen_split")
    done
else
    [[ "$SPLIT" == all ]] || {
        echo "[FATAL] Flat dataset $DATA_ROOT has no '$SPLIT' split; use HSA_SPLIT=all" >&2
        exit 2
    }
    sample_roots+=("$DATA_ROOT")
fi
for sample_root in "${sample_roots[@]}"; do
    while IFS= read -r sample; do
        [[ -f "$sample/domain.pddl" && -f "$sample/problem.nl" && -f "$sample/problem.pddl" ]] || continue
        count=$((count + 1))
    done < <(find -L "$sample_root" -mindepth 1 -maxdepth 1 -type d | sort -V)
done
(( count > 0 )) || { echo "[FATAL] No complete samples found under $DATA_ROOT" >&2; exit 2; }

if [[ "$HSA_GENERATOR_BACKEND" == transformers ]]; then
    "$PYTHON_BIN" -c "import torch, transformers; print('[OK] torch', torch.__version__, 'transformers', transformers.__version__)"
else
    "$PYTHON_BIN" -c "import sys; print('[OK] mock generator with Python', sys.version.split()[0])"
fi
echo "[OK] Preflight: samples=$count split=$SPLIT base=$HSA_BASE_MODEL sft=$HSA_SFT_MODEL"
