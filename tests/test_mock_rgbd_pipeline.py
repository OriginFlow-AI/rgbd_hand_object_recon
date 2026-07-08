from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon.evaluation import evaluate_quality
from hand_recon.mock_data import LABEL_HAND, LABEL_OBJECT, generate_mock_rgbd_scene
from hand_recon.pose import generate_pose_output
from hand_recon.reconstruction import reconstruct_multiview_pointcloud
from hand_recon.rgbd import backproject_depth_to_points, load_mock_rgbd_scene


def test_mock_rgbd_pipeline(tmp_path: Path) -> None:
    scene_dir = tmp_path / "rgbd_scene_001"
    metadata = generate_mock_rgbd_scene(scene_dir, view_count=4)

    assert (scene_dir / "cameras.json").exists()
    assert metadata["scene_id"] == "rgbd_scene_001"
    assert len(metadata["views"]) == 4

    camera_metadata = json.loads((scene_dir / "cameras.json").read_text(encoding="utf-8"))
    assert camera_metadata["mask_labels"]["hand"] == LABEL_HAND
    assert camera_metadata["mask_labels"]["object"] == LABEL_OBJECT

    scene = load_mock_rgbd_scene(scene_dir)
    assert len(scene.views) == 4
    assert scene.views[0].camera.camera_to_world.shape == (4, 4)

    view = scene.views[0]
    hand_camera_points = backproject_depth_to_points(
        view.depth,
        view.camera,
        mask=view.mask,
        valid_labels=[LABEL_HAND],
    )
    assert hand_camera_points.ndim == 2
    assert hand_camera_points.shape[1] == 3
    assert hand_camera_points.shape[0] > 0

    fused_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_HAND, LABEL_OBJECT], voxel_size_m=0.003)
    hand_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_HAND], voxel_size_m=0.003)
    object_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_OBJECT], voxel_size_m=0.003)
    assert fused_result.fused_points.shape[0] > 0
    assert hand_result.fused_points.shape[0] > 0
    assert object_result.fused_points.shape[0] > 0

    pose_output = generate_pose_output(
        hand_result.fused_points,
        object_result.fused_points,
        metadata={"scene_id": scene.scene_id, "coordinate_frame": scene.coordinate_frame, "timestamp": "test"},
    )
    assert pose_output["hands"][0]["status"] == "ok"
    assert pose_output["objects"][0]["status"] == "ok"
    assert pose_output["hands"][0]["translation_m"]
    assert pose_output["objects"][0]["translation_m"]

    quality_report = evaluate_quality(
        scene,
        fused_result,
        hand_result.fused_points,
        object_result.fused_points,
        pose_output,
    )
    assert quality_report["passed"] is True
    assert quality_report["metrics"]["fused_point_count"] > 100
    assert quality_report["metrics"]["hand_point_count"] > 0
    assert quality_report["metrics"]["object_point_count"] > 0
