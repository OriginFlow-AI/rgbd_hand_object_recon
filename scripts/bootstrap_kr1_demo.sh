#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e '.[dev]'

PYTHONPATH=src "$VENV_DIR/bin/python" -m hand_recon demo --config configs/mock_rgbd.json
"$VENV_DIR/bin/python" -m pytest -q
PYTHONPATH=src "$VENV_DIR/bin/python" -m hand_recon verify

printf '\nKR1 bootstrap and checks passed.\n'
