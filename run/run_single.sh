#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=run/setup_llmplan_env.sh
source "$ROOT/run/setup_llmplan_env.sh"

GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-55000}"
GPU_POLL_INTERVAL_S="${GPU_POLL_INTERVAL_S:-30}"
GPU_CANDIDATES="${GPU_CANDIDATES:-}"
AUTO_SELECT_GPU="${AUTO_SELECT_GPU:-1}"

pick_gpu() {
    local idx free best_idx="" best_free=-1
    command -v nvidia-smi >/dev/null 2>&1 || return 2
    while IFS=',' read -r idx free; do
        idx="$(echo "$idx" | xargs)"; free="$(echo "$free" | xargs)"
        [[ "$idx" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || continue
        [[ -z "$GPU_CANDIDATES" || ",$GPU_CANDIDATES," == *",$idx,"* ]] || continue
        if (( free >= GPU_MIN_FREE_MB && free > best_free )); then best_idx="$idx"; best_free="$free"; fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    [[ -n "$best_idx" ]] || return 1
    printf '%s\n' "$best_idx"
}

ARG_STRING=" $* "
NEEDS_GPU=1
if [[ "$HSA_GENERATOR_BACKEND" == mock || "$ARG_STRING" == *" --generator_backend mock "* || "$HSA_DEVICE" == cpu || "$ARG_STRING" == *" --device cpu "* ]]; then NEEDS_GPU=0; fi
if [[ "$NEEDS_GPU" == 1 && ( "$AUTO_SELECT_GPU" == 1 || -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES:-}" == auto ) ]]; then
    while true; do
        if selected="$(pick_gpu)"; then export CUDA_VISIBLE_DEVICES="$selected"; echo "[GPU] Selected physical GPU $selected"; break; fi
        rc=$?
        [[ "$rc" != 2 ]] || { echo "[FATAL] nvidia-smi unavailable; set HSA_DEVICE=cpu only for an intentional CPU run." >&2; exit 2; }
        echo "[GPU] Waiting for >= ${GPU_MIN_FREE_MB} MiB free memory; retry in ${GPU_POLL_INTERVAL_S}s"
        sleep "$GPU_POLL_INTERVAL_S"
    done
fi

export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$PYTHON_BIN" "$ROOT/main.py" "$@"

