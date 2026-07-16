from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hand_recon import HandSurfaceConfig, generate_mock_visual_report, run_mock_reconstruction
from hand_recon.domain import validate_triangle_mesh
from hand_recon.fusion.tsdf import build_masked_tsdf
from hand_recon.mock_data import LABEL_HAND, generate_mock_rgbd_scene
from hand_recon.pipelines.hand_surface import reconstruct_hand_surface
from hand_recon.reconstruction import reconstruct_multiview_pointcloud
from hand_recon.rgbd import load_mock_rgbd_scene


def test_joint_independent_tsdf_surface_contract(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    generate_mock_rgbd_scene(scene_dir, view_count=4)
    scene = load_mock_rgbd_scene(scene_dir)
    cloud = reconstruct_multiview_pointcloud(scene, labels=LABEL_HAND, voxel_size_m=0.003)

    result = reconstruct_hand_surface(scene, cloud, hand_label=LABEL_HAND)

    validate_triangle_mesh(result.mesh)
    assert result.ok
    assert result.parameters["uses_joint_localization"] is False
    assert result.parameters["surface_semantics"] == "observed_not_completed"
    assert result.mesh.vertex_count > 1000
    assert result.mesh.face_count > 1000
    assert result.quality["metrics"]["component_count"] == 1
    assert result.quality["metrics"]["non_manifold_edge_count"] == 0
    assert result.quality["metrics"]["source_to_surface_p95_m"] < 0.006
    assert np.allclose(np.linalg.norm(result.mesh.vertex_normals, axis=1), 1.0, atol=1e-6)


def test_surface_artifact_bundle_and_report_do_not_require_joint_files(tmp_path: Path) -> None:
    result = run_mock_reconstruction(scene_dir=tmp_path / "scene", output_dir=tmp_path / "output")
    manifest = json.loads(result.output_paths["surface_manifest"].read_text(encoding="utf-8"))

    assert manifest["uses_joint_localization"] is False
    assert manifest["surface_semantics"] == "observed_not_completed"
    assert manifest["counts"]["mesh_face_count"] > 1000
    assert all(entry["exists"] for entry in manifest["artifacts"].values())
    assert all(entry["sha256"] for entry in manifest["artifacts"].values())
    assert len([key for key in manifest["artifacts"] if key.startswith("view_")]) == 4
    for entry in manifest["artifacts"].values():
        artifact_path = result.output_dir / entry["path"]
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == entry["sha256"]

    ply_header = result.output_paths["hand_surface"].read_text(encoding="ascii").split("end_header", 1)[0]
    assert "element face" in ply_header
    assert "property float nx" in ply_header
    assert "property uchar red" in ply_header

    result.output_paths["kr3_hand_result"].unlink()
    result.output_paths["root_translation_optimized_hands"].unlink()
    report_path = generate_mock_visual_report(
        demo_dir=result.output_dir,
        output_html=tmp_path / "standalone.html",
        max_points=200,
    )
    report = report_path.read_text(encoding="utf-8")
    assert "多视角 RGB-D 手部表面重建" in report
    assert "关节点定位：</strong>未使用" in report
    assert "22DOF" not in report


def test_tsdf_rejects_empty_observation_and_resource_overflow(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    generate_mock_rgbd_scene(scene_dir, view_count=2)
    scene = load_mock_rgbd_scene(scene_dir)

    with pytest.raises(ValueError, match="at least three XYZ points"):
        build_masked_tsdf(scene, np.empty((0, 3)), label=LABEL_HAND)

    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 10.0]])
    with pytest.raises(ValueError, match="max_voxel_count"):
        build_masked_tsdf(
            scene,
            points,
            label=LABEL_HAND,
            voxel_size_m=0.001,
            truncation_m=0.003,
            max_voxel_count=1000,
        )

    with pytest.raises(ValueError, match="at least twice"):
        HandSurfaceConfig(voxel_size_m=0.004, truncation_m=0.006)
