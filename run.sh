#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m hand_recon demo --config configs/mock_rgbd.json
"$PYTHON_BIN" scripts/generate_hand_visual_report.py \
  --output-html outputs/reports/hand_reconstruction_visual_report.html

printf '\nResult: %s\n' "$ROOT/outputs/reports/hand_reconstruction_visual_report.html"
