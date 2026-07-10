#!/usr/bin/env python3
"""Minimal software-integration example for hand_recon."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon import generate_mock_visual_report, load_hand_result_npz, run_mock_reconstruction, validate_hand_result


def main() -> int:
    output_dir = ROOT / "outputs" / "api_demo_mock_rgbd"
    result = run_mock_reconstruction(
        scene_dir=ROOT / "mock_data" / "rgbd_scene_001",
        output_dir=output_dir,
        hand_side="right",
    )
    hand_result = load_hand_result_npz(result.output_paths["kr3_hand_result"])
    errors = validate_hand_result(hand_result)
    if errors:
        raise SystemExit("invalid hand result: " + "; ".join(errors))

    report_path = generate_mock_visual_report(
        demo_dir=output_dir,
        output_html=ROOT / "outputs" / "reports" / "api_demo_hand_visual.html",
    )
    print(f"status={result.status}")
    print(f"hand_result={result.output_paths['kr3_hand_result']}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
