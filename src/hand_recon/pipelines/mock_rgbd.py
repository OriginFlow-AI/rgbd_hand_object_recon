"""Mock RGB-D reconstruction pipeline.

This module owns the importable pipeline. CLI/demo scripts should stay thin and
delegate here so application code can call the same behavior without shelling
out to scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hand_recon.evaluation import evaluate_quality
from hand_recon.icp import write_ascii_ply
from hand_recon.interfaces.hand_result import build_kr3_hand_result_from_normalized, write_kr3_hand_result_npz
from hand_recon.mock_data import LABEL_HAND, LABEL_OBJECT, generate_mock_rgbd_scene
from hand_recon.normalized_output import build_normalized_hand_npz_payload, write_normalized_hand_npz
from hand_recon.pose import generate_pose_output
from hand_recon.reconstruction import ReconstructionResult, reconstruct_multiview_pointcloud
from hand_recon.rgbd import load_mock_rgbd_scene


@dataclass(frozen=True)
class MockRgbdPipelineResult:
    status: str
    scene_dir: Path
    output_dir: Path
    output_paths: dict[str, Path]
    summary: dict[str, Any]
    pose_output: dict[str, Any]
    quality_report: dict[str, Any]
    fused_result: ReconstructionResult
    hand_result: ReconstructionResult
    object_result: ReconstructionResult

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def run_mock_rgbd_pipeline(
    *,
    scene_dir: Path,
    output_dir: Path,
    voxel_size_m: float = 0.003,
    hand_side: str = "right",
    overwrite_mock_data: bool = False,
) -> MockRgbdPipelineResult:
    """Run the full mock RGB-D hand/object reconstruction pipeline."""

    if hand_side not in {"left", "right"}:
        raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")

    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite_mock_data or not (scene_dir / "cameras.json").exists():
        generate_mock_rgbd_scene(scene_dir, overwrite=overwrite_mock_data)

    scene = load_mock_rgbd_scene(scene_dir)
    metadata = {
        "scene_id": scene.scene_id,
        "coordinate_frame": scene.coordinate_frame,
        "timestamp": scene.views[0].camera.timestamp if scene.views else "",
    }

    fused_result = reconstruct_multiview_pointcloud(
        scene,
        labels=[LABEL_HAND, LABEL_OBJECT],
        voxel_size_m=voxel_size_m,
    )
    hand_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_HAND], voxel_size_m=voxel_size_m)
    object_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_OBJECT], voxel_size_m=voxel_size_m)

    output_paths = build_mock_output_paths(output_dir)
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

    write_json(output_paths["pose_output"], pose_output)
    write_json(output_paths["quality_report"], quality_report)

    normalized_payload = build_normalized_hand_npz_payload(
        scene,
        hand_result.fused_points,
        frame_index=0,
        is_right=hand_side == "right",
    )
    write_normalized_hand_npz(output_paths["root_translation_optimized_hands"], normalized_payload)
    kr3_payload = build_kr3_hand_result_from_normalized(
        normalized_payload,
        source_system="super_labelator",
        track_id=f"{hand_side}_hand_0",
    )
    write_kr3_hand_result_npz(output_paths["kr3_hand_result"], kr3_payload)

    summary = {
        "status": "ok" if quality_report["passed"] else "failed",
        "scene_dir": str(scene_dir),
        "output_dir": str(output_dir),
        "parameters": {"voxel_size_m": voxel_size_m, "hand_side": hand_side},
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "per_view_stats": fused_result.per_view_stats,
        "quality_passed": bool(quality_report["passed"]),
    }
    write_json(output_paths["summary"], summary)

    return MockRgbdPipelineResult(
        status=summary["status"],
        scene_dir=scene_dir,
        output_dir=output_dir,
        output_paths=output_paths,
        summary=summary,
        pose_output=pose_output,
        quality_report=quality_report,
        fused_result=fused_result,
        hand_result=hand_result,
        object_result=object_result,
    )


def build_mock_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "fused_pointcloud": output_dir / "fused_pointcloud.ply",
        "hand_pointcloud": output_dir / "hand_pointcloud.ply",
        "object_pointcloud": output_dir / "object_pointcloud.ply",
        "pose_output": output_dir / "pose_output.json",
        "quality_report": output_dir / "quality_report.json",
        "summary": output_dir / "summary.json",
        "root_translation_optimized_hands": output_dir / "scale" / "root_translation_optimized_hands.npz",
        "kr3_hand_result": output_dir / "kr3" / "hand_result.npz",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
