"""RGB-D scene loading and depth backprojection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    camera_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_to_world: np.ndarray
    timestamp: str


@dataclass(frozen=True)
class RgbdView:
    camera: CameraModel
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class RgbdScene:
    scene_id: str
    scene_dir: Path
    coordinate_frame: str
    depth_unit: str
    mask_labels: dict[str, int]
    views: list[RgbdView]


def load_mock_rgbd_scene(scene_dir: Path) -> RgbdScene:
    scene_dir = Path(scene_dir)
    metadata_path = scene_dir / "cameras.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    views: list[RgbdView] = []
    for item in metadata["views"]:
        intrinsics = item["intrinsics"]
        camera_to_world = np.asarray(item["extrinsics"]["camera_to_world"], dtype=np.float64)
        if camera_to_world.shape != (4, 4):
            raise ValueError(f"camera_to_world for {item['camera_id']} must be 4x4")
        camera = CameraModel(
            camera_id=item["camera_id"],
            width=int(item["width"]),
            height=int(item["height"]),
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
            camera_to_world=camera_to_world,
            timestamp=str(item.get("timestamp", "")),
        )
        files = item["files"]
        rgb = np.load(scene_dir / files["rgb"])
        depth = np.load(scene_dir / files["depth"])
        mask = np.load(scene_dir / files["mask"])
        _validate_view_arrays(camera, rgb, depth, mask)
        views.append(RgbdView(camera=camera, rgb=rgb, depth=depth, mask=mask))

    return RgbdScene(
        scene_id=str(metadata["scene_id"]),
        scene_dir=scene_dir,
        coordinate_frame=str(metadata.get("coordinate_frame", "world")),
        depth_unit=str(metadata.get("depth_unit", "meter")),
        mask_labels={str(key): int(value) for key, value in metadata.get("mask_labels", {}).items()},
        views=views,
    )


def backproject_depth_to_points(
    depth: np.ndarray,
    intrinsics: CameraModel | dict[str, float],
    mask: np.ndarray | None = None,
    valid_labels: Iterable[int] | None = None,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape HxW, got {depth.shape}")

    if isinstance(intrinsics, CameraModel):
        fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    else:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])

    valid = np.isfinite(depth) & (depth > 0.0)
    if mask is not None and valid_labels is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.shape != depth.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match depth shape {depth.shape}")
        labels = np.asarray(list(valid_labels), dtype=mask_arr.dtype)
        valid &= np.isin(mask_arr, labels)

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float64)


def depth_valid_ratio(depth: np.ndarray) -> float:
    depth = np.asarray(depth)
    if depth.size == 0:
        return 0.0
    valid = np.isfinite(depth) & (depth > 0)
    return float(np.count_nonzero(valid) / depth.size)


def _validate_view_arrays(camera: CameraModel, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> None:
    expected_hw = (camera.height, camera.width)
    if depth.shape != expected_hw:
        raise ValueError(f"depth for {camera.camera_id} must be {expected_hw}, got {depth.shape}")
    if mask.shape != expected_hw:
        raise ValueError(f"mask for {camera.camera_id} must be {expected_hw}, got {mask.shape}")
    if rgb.shape[:2] != expected_hw or rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb for {camera.camera_id} must be HxWx3, got {rgb.shape}")
