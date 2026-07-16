#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m hand_recon demo --config configs/mock_rgbd.json
"$PYTHON_BIN" scripts/evaluate_normalized_npz_accuracy.py \
  --prediction-npz outputs/mock_rgbd_demo/scale/root_translation_optimized_hands.npz \
  --output-json outputs/mock_rgbd_demo/scale/accuracy_report.json
"$PYTHON_BIN" -m pytest tests/test_mock_rgbd_pipeline.py
"$PYTHON_BIN" -m hand_recon verify

printf '\nKR1 checks passed.\n'
