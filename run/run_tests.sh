#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHONPATH="$ROOT" "$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -v

