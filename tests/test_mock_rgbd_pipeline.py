from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_recon.evaluation import evaluate_quality
from hand_recon.mock_data import LABEL_HAND, LABEL_OBJECT, generate_mock_rgbd_scene
from hand_recon.normalized_output import (
    HAND_ANGLE_NAMES_20DOF,
    JOINT_NAMES,
    build_normalized_hand_npz_payload,
    write_normalized_hand_npz,
)
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

    normalized = build_normalized_hand_npz_payload(scene, hand_result.fused_points, frame_index=0, is_right=True)
    assert normalized["K"].shape == (3, 3)
    assert normalized["baseline_m"].shape == ()
    assert normalized["frame_index"].shape == (1,)
    assert normalized["root_translation_m"].shape == (1, 3)
    assert normalized["wrist_pose_6d_left_m_rad"].shape == (1, 6)
    assert normalized["joints_3d_left_m"].shape == (1, 21, 3)
    assert normalized["valid_joint_mask"].shape == (1, 21)
    assert normalized["hand_angles_20dof_rad"].shape == (1, 20)
    assert normalized["hand_angles_20dof_deg"].shape == (1, 20)
    assert normalized["joint_names"].tolist() == JOINT_NAMES.tolist()
    assert normalized["hand_angle_names_20dof"].tolist() == HAND_ANGLE_NAMES_20DOF.tolist()
    assert normalized["hand_side"].tolist() == [1]
    assert normalized["is_right"].tolist() == [True]
    assert normalized["frame_status"].tolist() == ["ok"]

    left_normalized = build_normalized_hand_npz_payload(scene, hand_result.fused_points, frame_index=0, is_right=False)
    assert left_normalized["hand_side"].tolist() == [0]
    assert left_normalized["is_right"].tolist() == [False]

    normalized_path = tmp_path / "scale" / "root_translation_optimized_hands.npz"
    write_normalized_hand_npz(normalized_path, normalized)
    with np.load(normalized_path, allow_pickle=False) as loaded:
        assert set(normalized).issubset(set(loaded.files))
        assert loaded["joints_3d_left_m"].shape == (1, 21, 3)
