#!/usr/bin/env python3
"""Run the KR1 mock multi-view RGB-D reconstruction demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon.evaluation import evaluate_quality  # noqa: E402
from hand_recon.icp import write_ascii_ply  # noqa: E402
from hand_recon.mock_data import LABEL_HAND, LABEL_OBJECT, generate_mock_rgbd_scene  # noqa: E402
from hand_recon.normalized_output import build_normalized_hand_npz_payload, write_normalized_hand_npz  # noqa: E402
from hand_recon.pose import generate_pose_output  # noqa: E402
from hand_recon.reconstruction import reconstruct_multiview_pointcloud  # noqa: E402
from hand_recon.rgbd import load_mock_rgbd_scene  # noqa: E402


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite_mock_data or not (args.scene_dir / "cameras.json").exists():
        generate_mock_rgbd_scene(args.scene_dir, overwrite=args.overwrite_mock_data)

    scene = load_mock_rgbd_scene(args.scene_dir)
    metadata = {
        "scene_id": scene.scene_id,
        "coordinate_frame": scene.coordinate_frame,
        "timestamp": scene.views[0].camera.timestamp if scene.views else "",
    }

    fused_result = reconstruct_multiview_pointcloud(
        scene,
        labels=[LABEL_HAND, LABEL_OBJECT],
        voxel_size_m=args.voxel_size_m,
    )
    hand_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_HAND], voxel_size_m=args.voxel_size_m)
    object_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_OBJECT], voxel_size_m=args.voxel_size_m)

    output_paths = {
        "fused_pointcloud": args.output_dir / "fused_pointcloud.ply",
        "hand_pointcloud": args.output_dir / "hand_pointcloud.ply",
        "object_pointcloud": args.output_dir / "object_pointcloud.ply",
        "pose_output": args.output_dir / "pose_output.json",
        "quality_report": args.output_dir / "quality_report.json",
        "summary": args.output_dir / "summary.json",
        "root_translation_optimized_hands": args.output_dir / "scale" / "root_translation_optimized_hands.npz",
    }

    write_ascii_ply(output_paths["fused_pointcloud"], fused_result.fused_points)
    write_ascii_ply(output_paths["hand_pointcloud"], hand_result.fused_points)
    write_ascii_ply(output_paths["object_pointcloud"], object_result.fused_points)

    pose_output = generate_pose_output(hand_result.fused_points, object_result.fused_points, metadata=metadata)
    quality_report = evaluate_quality(
        scene=scene,
        fused_result=fused_result,
        hand_points=hand_result.fused_points,
        object_points=object_result.fused_points,
        pose_output=pose_output,
    )

    _write_json(output_paths["pose_output"], pose_output)
    _write_json(output_paths["quality_report"], quality_report)
    normalized_payload = build_normalized_hand_npz_payload(
        scene,
        hand_result.fused_points,
        frame_index=0,
        is_right=args.hand_side == "right",
    )
    write_normalized_hand_npz(output_paths["root_translation_optimized_hands"], normalized_payload)

    summary = {
        "status": "ok" if quality_report["passed"] else "failed",
        "scene_dir": str(args.scene_dir),
        "output_dir": str(args.output_dir),
        "parameters": {"voxel_size_m": args.voxel_size_m, "hand_side": args.hand_side},
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "per_view_stats": fused_result.per_view_stats,
        "quality_passed": bool(quality_report["passed"]),
    }
    _write_json(output_paths["summary"], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if quality_report["passed"] else 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
