#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m pytest tests/test_kr3_hand_result_interface.py

printf '\nKR3 interface checks passed.\n'
