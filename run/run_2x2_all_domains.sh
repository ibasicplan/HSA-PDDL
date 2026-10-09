#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=run/setup_llmplan_env.sh
source "$ROOT/run/setup_llmplan_env.sh"

bash "$ROOT/run/prepare_all_datasets.sh"
ANY_FAILURE=0
read -r -a DOMAINS <<< "${HSA_DOMAINS:-zenotravel logistics mystery quantum rovers}"
for domain in "${DOMAINS[@]}"; do
    echo "================ DOMAIN: $domain ================"
    if ! HSA_DOMAIN_RESULTS_ROOT="$HSA_RESULTS_ROOT/$domain" bash "$ROOT/run/run_2x2_domain.sh" "$domain"; then
        ANY_FAILURE=1
    fi
done

PYTHONPATH="$ROOT" "$PYTHON_BIN" "$ROOT/scripts/summarize_cross_domain.py" \
    --results_root "$HSA_RESULTS_ROOT" --domains "${DOMAINS[@]}" \
    --bootstrap_iterations "${HSA_BOOTSTRAP_ITERATIONS:-5000}" --seed "$HSA_SEED"

[[ "$ANY_FAILURE" == 0 ]] || { echo "[FATAL] At least one domain failed; inspect per-domain status files." >&2; exit 1; }
echo "[DONE] $HSA_RESULTS_ROOT/cross_domain_summary.md"
