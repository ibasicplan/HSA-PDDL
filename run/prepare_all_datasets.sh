#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=run/setup_llmplan_env.sh
source "$ROOT/run/setup_llmplan_env.sh"

DOMAINS=" ${HSA_DOMAINS:-zenotravel logistics mystery quantum rovers} "

if [[ "$DOMAINS" == *" mystery "* ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_mystery_dataset.py" \
        --source "$HSA_MYSTERY_SOURCE" --output "$HSA_MYSTERY_DATA" \
        --split easy --mode symlink --all
fi

if [[ "$DOMAINS" == *" quantum "* ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/prepare_quantum_dataset.py" \
        --source "$HSA_QUANTUM_SOURCE" --output "$HSA_QUANTUM_DATA" \
        --split easy --mode symlink --all
fi

if [[ "$DOMAINS" == *" rovers "* ]]; then
    bash "$ROOT/run/prepare_rovers_dataset.sh"
fi

echo "[OK] Dataset views are ready under $HSA_PREPARED_DATA_ROOT"
