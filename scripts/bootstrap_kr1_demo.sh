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
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

"$VENV_DIR/bin/python" demo/run_mock_rgbd_pipeline.py --output-dir outputs/mock_rgbd_demo
"$VENV_DIR/bin/python" -m pytest tests/test_mock_rgbd_pipeline.py
"$VENV_DIR/bin/python" scripts/run_icp_registration.py --selftest

printf '\nKR1 bootstrap and checks passed.\n'
