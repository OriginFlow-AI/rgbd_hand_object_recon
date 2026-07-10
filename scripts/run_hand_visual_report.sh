#!/usr/bin/env bash
set -euo pipefail

python3 scripts/generate_hand_visual_report.py \
  --output-html outputs/reports/hand_reconstruction_visual_report.html

printf '\nVisual report written to outputs/reports/hand_reconstruction_visual_report.html\n'
