#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=run/setup_llmplan_env.sh
source "$ROOT/run/setup_llmplan_env.sh"

mkdir -p "$HSA_PREPARED_DATA_ROOT"
PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/build_rovers_catalog.py" \
    --source "$HSA_ROVERS_DATA" \
    --output "$HSA_ROVERS_CATALOG" \
    --coverage-policy "${HSA_ROVERS_COVERAGE_POLICY:-auto}"

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/audit_rovers_dataset.py" \
    --source "$HSA_ROVERS_DATA" \
    --catalog "$HSA_ROVERS_CATALOG" \
    --json-output "$HSA_PREPARED_DATA_ROOT/rovers_dataset_audit.json"

echo "[OK] Rovers source was not modified. Catalog: $HSA_ROVERS_CATALOG"
