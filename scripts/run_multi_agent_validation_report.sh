#!/usr/bin/env bash
set -euo pipefail

python3 scripts/generate_multi_agent_validation_report.py \
  --output-html outputs/reports/multi_agent_validation_report.html

printf '\nMulti-agent validation report written to outputs/reports/multi_agent_validation_report.html\n'
