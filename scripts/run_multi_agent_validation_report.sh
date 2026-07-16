#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/generate_multi_agent_validation_report.py \
  --output-html outputs/reports/multi_agent_validation_report.html

printf '\nMulti-agent validation report written to outputs/reports/multi_agent_validation_report.html\n'
