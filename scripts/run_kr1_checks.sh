#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 demo/run_mock_rgbd_pipeline.py --output-dir outputs/mock_rgbd_demo
python3 -m pytest tests/test_mock_rgbd_pipeline.py
python3 scripts/run_icp_registration.py --selftest

printf '\nKR1 checks passed.\n'
