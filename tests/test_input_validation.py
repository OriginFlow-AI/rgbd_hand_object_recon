from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hand_recon.exceptions import DataValidationError
from hand_recon.icp import (
    PointCloud,
    concatenate_point_clouds,
    icp_point_to_point,
    load_ascii_ply,
    load_npz_cloud,
    scale_point_cloud,
    voxel_downsample,
    write_ascii_ply,
)
from hand_recon.mock_data import generate_mock_rgbd_scene
from hand_recon.rgbd import backproject_depth_to_points, load_mock_rgbd_scene


def _metadata(scene_dir: Path) -> dict[str, object]:
    return json.loads((scene_dir / "cameras.json").read_text(encoding="utf-8"))


def test_scene_loader_rejects_paths_outside_scene(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    generate_mock_rgbd_scene(scene_dir, view_count=1)
    metadata = _metadata(scene_dir)
    metadata["views"][0]["files"]["depth"] = "../outside.npy"  # type: ignore[index]
    (scene_dir / "cameras.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DataValidationError, match="escapes the scene directory"):
        load_mock_rgbd_scene(scene_dir)


def test_scene_loader_rejects_invalid_intrinsics(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    generate_mock_rgbd_scene(scene_dir, view_count=1)
    metadata = _metadata(scene_dir)
    metadata["views"][0]["intrinsics"]["fx"] = 0  # type: ignore[index]
    (scene_dir / "cameras.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DataValidationError, match="fx/fy > 0"):
        load_mock_rgbd_scene(scene_dir)


def test_scene_loader_rejects_mismatched_mask_contract(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    generate_mock_rgbd_scene(scene_dir, view_count=1)
    metadata = _metadata(scene_dir)
    metadata["mask_labels"]["hand"] = 5  # type: ignore[index]
    (scene_dir / "cameras.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DataValidationError, match="mask_labels"):
        load_mock_rgbd_scene(scene_dir)


def test_mock_overwrite_preserves_unrelated_files(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    generate_mock_rgbd_scene(scene_dir, view_count=1)
    sentinel = scene_dir / "keep-me.txt"
    sentinel.write_text("user data", encoding="utf-8")

    generate_mock_rgbd_scene(scene_dir, view_count=2, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "user data"
    assert len(load_mock_rgbd_scene(scene_dir).views) == 2


def test_backprojection_rejects_zero_focal_length() -> None:
    with pytest.raises(DataValidationError, match="fx/fy must be positive"):
        backproject_depth_to_points(np.ones((2, 2)), {"fx": 0, "fy": 1, "cx": 0, "cy": 0})


def test_point_filter_keeps_colors_aligned() -> None:
    points = np.array([[0.0, 0.0, 0.0], [np.nan, 2.0, 3.0], [1.0, 1.0, 1.0]])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

    filtered_points, filtered_colors = voxel_downsample(points, colors, voxel_size=None)

    assert filtered_points.tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert filtered_colors is not None
    assert filtered_colors.tolist() == [[255, 0, 0], [0, 0, 255]]

    combined_points, combined_colors = concatenate_point_clouds([PointCloud(points, colors, "dirty")])
    assert combined_points.tolist() == filtered_points.tolist()
    assert combined_colors is not None
    assert combined_colors.tolist() == filtered_colors.tolist()

    scaled = scale_point_cloud(PointCloud(points, colors, "dirty"), 1.0)
    assert scaled.points.tolist() == filtered_points.tolist()
    assert scaled.colors is not None
    assert scaled.colors.tolist() == filtered_colors.tolist()


def test_empty_ascii_ply_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "empty.ply"
    write_ascii_ply(path, np.empty((0, 3)))

    loaded = load_ascii_ply(path)

    assert loaded.points.shape == (0, 3)


def test_explicit_npz_key_must_exist(tmp_path: Path) -> None:
    path = tmp_path / "cloud.npz"
    np.savez(path, points=np.zeros((3, 3)))

    with pytest.raises(ValueError, match="was not found"):
        load_npz_cloud(path, npz_points_key="misspelled_points")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_pairs": 2}, "min_pairs"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"distance_threshold": 0.0}, "distance_threshold"),
        ({"init_transform": np.diag([1.0, 1.0, 1.0, 2.0])}, "homogeneous"),
    ],
)
def test_icp_rejects_invalid_parameters(kwargs: dict[str, float | int], message: str) -> None:
    points = np.arange(90, dtype=np.float64).reshape(30, 3)
    with pytest.raises(ValueError, match=message):
        icp_point_to_point(points, points, **kwargs)
