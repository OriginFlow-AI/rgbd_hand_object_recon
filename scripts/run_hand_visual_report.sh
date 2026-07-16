#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/generate_hand_visual_report.py \
  --output-html outputs/mock_rgbd_demo/report/index.html

printf '\nVisual report written to outputs/mock_rgbd_demo/report/index.html\n'
