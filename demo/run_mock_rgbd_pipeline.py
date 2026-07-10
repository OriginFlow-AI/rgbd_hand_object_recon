#!/usr/bin/env python3
"""Run the KR1 mock multi-view RGB-D reconstruction demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon.pipelines.mock_rgbd import run_mock_rgbd_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, default=ROOT / "mock_data" / "rgbd_scene_001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "mock_rgbd_demo")
    parser.add_argument("--voxel-size-m", type=float, default=0.003)
    parser.add_argument("--hand-side", choices=["left", "right"], default="right")
    parser.add_argument("--overwrite-mock-data", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_mock_rgbd_pipeline(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        voxel_size_m=args.voxel_size_m,
        hand_side=args.hand_side,
        overwrite_mock_data=args.overwrite_mock_data,
    )
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
