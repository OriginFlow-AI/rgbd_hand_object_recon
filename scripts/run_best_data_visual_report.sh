#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/run_icp_registration.py \
  --inputs \
  data/reinterhand/m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands/mano_fits/meshes/100001_right.ply \
  data/reinterhand/m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands/mano_fits/meshes/100004_right.ply \
  data/reinterhand/m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands/mano_fits/meshes/100007_right.ply \
  data/reinterhand/m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands/mano_fits/meshes/100010_right.ply \
  data/reinterhand/m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands/mano_fits/meshes/100013_right.ply \
  --output-dir outputs/reinterhand_best_right_sequence_icp \
  --input-scale 0.001 \
  --voxel-size-m 0.001 \
  --max-points 50000 \
  --max-iterations 80 \
  --distance-threshold-m 0.03 \
  --trim-fraction 0.9 \
  --min-pairs 100

"$PYTHON_BIN" scripts/generate_best_data_visual_report.py \
  --icp-dir outputs/reinterhand_best_right_sequence_icp \
  --output-html outputs/reports/best_data_reinterhand_visual_report.html

printf '\nBest-data visual report written to outputs/reports/best_data_reinterhand_visual_report.html\n'
