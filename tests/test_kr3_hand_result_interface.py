from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon.interfaces.hand_result import (  # noqa: E402
    HAND_RESULT_SCHEMA_VERSION,
    SOURCE_SYSTEMS,
    UMETRACK_ANGLE_NAMES_22DOF,
    build_kr3_hand_result_from_normalized,
    validate_kr3_hand_result,
    write_kr3_hand_result_npz,
)
from hand_recon.mock_data import LABEL_HAND, generate_mock_rgbd_scene  # noqa: E402
from hand_recon.normalized_output import build_normalized_hand_npz_payload  # noqa: E402
from hand_recon.reconstruction import reconstruct_multiview_pointcloud  # noqa: E402
from hand_recon.rgbd import load_mock_rgbd_scene  # noqa: E402


def test_kr3_hand_result_interface_from_mock_pipeline(tmp_path: Path) -> None:
    scene_dir = tmp_path / "rgbd_scene_001"
    generate_mock_rgbd_scene(scene_dir, view_count=4)
    scene = load_mock_rgbd_scene(scene_dir)
    hand_result = reconstruct_multiview_pointcloud(scene, labels=[LABEL_HAND], voxel_size_m=0.003)
    normalized = build_normalized_hand_npz_payload(scene, hand_result.fused_points, frame_index=7, is_right=True)

    kr3_payload = build_kr3_hand_result_from_normalized(
        normalized,
        source_system="super_labelator",
        timestamp_ns=123456789,
        track_id="right_hand_track_0",
    )

    assert validate_kr3_hand_result(kr3_payload) == []
    assert kr3_payload["schema_version"].item() == HAND_RESULT_SCHEMA_VERSION
    assert kr3_payload["source_system"].tolist() == ["super_labelator"]
    assert kr3_payload["source_system"].tolist()[0] in SOURCE_SYSTEMS
    assert kr3_payload["frame_index"].tolist() == [7]
    assert kr3_payload["timestamp_ns"].tolist() == [123456789]
    assert kr3_payload["track_id"].tolist() == ["right_hand_track_0"]
    assert kr3_payload["hand_angles_22dof_rad"].shape == (1, 22)
    assert kr3_payload["hand_angles_22dof_deg"].shape == (1, 22)
    assert kr3_payload["hand_angle_names_22dof"].tolist() == UMETRACK_ANGLE_NAMES_22DOF.tolist()
    assert kr3_payload["joints_3d_m"].shape == (1, 21, 3)
    assert kr3_payload["mesh_vertices_m"].ndim == 3
    assert kr3_payload["mesh_vertices_m"].shape[2] == 3
    assert kr3_payload["mesh_faces"].ndim == 2
    assert kr3_payload["mesh_faces"].shape[1] == 3
    assert kr3_payload["mano_pose_axis_angle"].shape == (1, 48)
    assert kr3_payload["mano_shape_betas"].shape == (1, 10)
    assert kr3_payload["umetrack_joint_angles_rad"].shape == (1, 22)

    output_path = tmp_path / "kr3" / "hand_result.npz"
    write_kr3_hand_result_npz(output_path, kr3_payload)
    loaded = np.load(output_path, allow_pickle=True)
    assert loaded["hand_angles_22dof_rad"].shape == (1, 22)
    assert loaded["joints_3d_m"].shape == (1, 21, 3)
    assert loaded["mesh_vertices_m"].shape == kr3_payload["mesh_vertices_m"].shape


def test_kr3_schema_lists_weekly_report_fields() -> None:
    schema_path = ROOT / "schemas" / "kr3" / "hand_result_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = set(schema["required"])

    assert "hand_angles_22dof_rad" in required
    assert "joints_3d_m" in required
    assert "mesh_vertices_m" in required
    assert "mesh_faces" in required
    assert schema["properties"]["hand_angles_22dof_rad"]["x-npz-shape"] == ["N", 22]
    assert schema["properties"]["joints_3d_m"]["x-npz-shape"] == ["N", 21, 3]
    assert schema["properties"]["mesh_vertices_m"]["x-npz-shape"] == ["N", "V", 3]
