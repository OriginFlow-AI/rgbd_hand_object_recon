#!/usr/bin/env bash
set -euo pipefail

python3 -m pytest tests/test_kr3_hand_result_interface.py

printf '\nKR3 interface checks passed.\n'
