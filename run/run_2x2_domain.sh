#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=run/setup_llmplan_env.sh
source "$ROOT/run/setup_llmplan_env.sh"

DOMAIN="${1:-}"
case "$DOMAIN" in
    zenotravel)
        export HSA_DATA_ROOT="$HSA_ZENOTRAVEL_DATA"
        export HSA_CATALOG="$ROOT/catalog/zenotravel_relation_catalog.json"
        export HSA_MAX_NEW_TOKENS="${HSA_MAX_NEW_TOKENS:-4096}"
        export HSA_EIR_MAX_NEW_TOKENS="${HSA_EIR_MAX_NEW_TOKENS:-4096}"
        ;;
    logistics)
        export HSA_DATA_ROOT="$HSA_LOGISTICS_DATA"
        export HSA_CATALOG="$ROOT/catalog/logistics_relation_catalog.json"
        export HSA_MAX_NEW_TOKENS="${HSA_MAX_NEW_TOKENS:-8192}"
        export HSA_EIR_MAX_NEW_TOKENS="${HSA_EIR_MAX_NEW_TOKENS:-8192}"
        ;;
    mystery)
        export HSA_DATA_ROOT="$HSA_MYSTERY_DATA"
        export HSA_CATALOG="$ROOT/catalog/mystery_4ops_relation_catalog.json"
        export HSA_MAX_NEW_TOKENS="${HSA_MAX_NEW_TOKENS:-4096}"
        export HSA_EIR_MAX_NEW_TOKENS="${HSA_EIR_MAX_NEW_TOKENS:-4096}"
        ;;
    quantum)
        export HSA_DATA_ROOT="$HSA_QUANTUM_DATA"
        export HSA_CATALOG="$ROOT/catalog/quantum_relation_catalog.json"
        export HSA_MAX_NEW_TOKENS="${HSA_MAX_NEW_TOKENS:-8192}"
        export HSA_EIR_MAX_NEW_TOKENS="${HSA_EIR_MAX_NEW_TOKENS:-8192}"
        ;;
    rovers)
        if [[ -n "${HSA_SPLIT:-}" && "$HSA_SPLIT" != all ]]; then
            echo "[FATAL] Rovers uses a flat layout without easy/hard labels; set HSA_SPLIT=all" >&2
            exit 2
        fi
        export HSA_SPLIT=all
        bash "$ROOT/run/prepare_rovers_dataset.sh"
        export HSA_DATA_ROOT="$HSA_ROVERS_DATA"
        export HSA_CATALOG="$HSA_ROVERS_CATALOG"
        export HSA_MAX_NEW_TOKENS="${HSA_MAX_NEW_TOKENS:-8192}"
        export HSA_EIR_MAX_NEW_TOKENS="${HSA_EIR_MAX_NEW_TOKENS:-8192}"
        ;;
    *)
        echo "Usage: bash run/run_2x2_domain.sh {zenotravel|logistics|mystery|quantum|rovers}" >&2
        exit 2
        ;;
esac

export HSA_SPLIT="${HSA_SPLIT:-all}"
export HSA_DOMAIN_RESULTS_ROOT="${HSA_DOMAIN_RESULTS_ROOT:-$HSA_RESULTS_ROOT/$DOMAIN}"
exec bash "$ROOT/run/run_all_experiments.sh"
