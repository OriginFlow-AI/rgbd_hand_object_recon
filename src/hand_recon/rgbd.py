"""RGB-D scene loading and depth backprojection helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hand_recon.exceptions import DataValidationError, UnsafeDataError
from hand_recon.io.json_io import read_json_object
from hand_recon.time_utils import parse_iso8601_ns

EXPECTED_MASK_LABELS = {"background": 0, "hand": 1, "object": 2}


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
    metadata = read_json_object(metadata_path)
    depth_unit = str(metadata.get("depth_unit", "meter"))
    if depth_unit != "meter":
        raise DataValidationError(f"unsupported depth_unit {depth_unit!r} in {metadata_path}; expected 'meter'")
    coordinate_frame = str(metadata.get("coordinate_frame", "world"))
    if coordinate_frame != "world":
        raise DataValidationError(
            f"unsupported coordinate_frame {coordinate_frame!r} in {metadata_path}; expected 'world'"
        )
    scene_id = str(metadata.get("scene_id", scene_dir.name)).strip()
    if not scene_id:
        raise DataValidationError(f"scene_id must not be empty in {metadata_path}")

    raw_views = metadata.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise DataValidationError(f"views must be a non-empty array in {metadata_path}")

    views: list[RgbdView] = []
    camera_ids: set[str] = set()
    for index, raw_item in enumerate(raw_views):
        item = _require_mapping(raw_item, f"views[{index}]")
        camera_id = str(item.get("camera_id", "")).strip()
        if not camera_id:
            raise DataValidationError(f"views[{index}].camera_id must not be empty")
        if camera_id in camera_ids:
            raise DataValidationError(f"duplicate camera_id in {metadata_path}: {camera_id!r}")
        camera_ids.add(camera_id)

        intrinsics = _require_mapping(item.get("intrinsics"), f"intrinsics for {camera_id}")
        extrinsics = _require_mapping(item.get("extrinsics"), f"extrinsics for {camera_id}")
        try:
            width = int(item["width"])
            height = int(item["height"])
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
            camera_to_world = np.asarray(extrinsics["camera_to_world"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"invalid camera parameters for {camera_id}: {exc}") from exc
        _validate_camera_parameters(camera_id, width, height, fx, fy, cx, cy, camera_to_world)
        timestamp = str(item.get("timestamp", "")).strip()
        try:
            parse_iso8601_ns(timestamp)
        except DataValidationError as exc:
            raise DataValidationError(f"invalid timestamp for {camera_id}: {exc}") from exc
        camera = CameraModel(
            camera_id=camera_id,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_to_world=camera_to_world,
            timestamp=timestamp,
        )
        files = _require_mapping(item.get("files"), f"files for {camera_id}")
        rgb = _load_scene_array(scene_dir, files, "rgb", camera_id)
        depth = _load_scene_array(scene_dir, files, "depth", camera_id)
        mask = _load_scene_array(scene_dir, files, "mask", camera_id)
        _validate_view_arrays(camera, rgb, depth, mask)
        views.append(RgbdView(camera=camera, rgb=rgb, depth=depth, mask=mask))

    mask_labels_raw = metadata.get("mask_labels", {})
    if not isinstance(mask_labels_raw, dict):
        raise DataValidationError(f"mask_labels must be an object in {metadata_path}")
    try:
        mask_labels = {str(key): int(value) for key, value in mask_labels_raw.items()}
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"mask_labels values must be integers in {metadata_path}") from exc
    if mask_labels != EXPECTED_MASK_LABELS:
        raise DataValidationError(
            f"mask_labels in {metadata_path} must equal {EXPECTED_MASK_LABELS}, got {mask_labels}"
        )

    return RgbdScene(
        scene_id=scene_id,
        scene_dir=scene_dir,
        coordinate_frame=coordinate_frame,
        depth_unit=depth_unit,
        mask_labels=mask_labels,
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
        try:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"invalid intrinsics: {exc}") from exc
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise DataValidationError("fx/fy must be positive finite values and cx/cy must be finite")

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
    if rgb.ndim != 3 or rgb.shape != (*expected_hw, 3):
        raise ValueError(f"rgb for {camera.camera_id} must be HxWx3, got {rgb.shape}")
    if not np.issubdtype(depth.dtype, np.number):
        raise DataValidationError(f"depth for {camera.camera_id} must be numeric, got {depth.dtype}")
    if not np.issubdtype(mask.dtype, np.integer):
        raise DataValidationError(f"mask for {camera.camera_id} must use an integer dtype, got {mask.dtype}")
    if not np.issubdtype(rgb.dtype, np.number):
        raise DataValidationError(f"rgb for {camera.camera_id} must be numeric, got {rgb.dtype}")


def _validate_camera_parameters(
    camera_id: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_to_world: np.ndarray,
) -> None:
    if width <= 0 or height <= 0:
        raise DataValidationError(f"camera dimensions for {camera_id} must be positive, got {width}x{height}")
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
        raise DataValidationError(f"camera intrinsics for {camera_id} must be finite with fx/fy > 0")
    if camera_to_world.shape != (4, 4):
        raise DataValidationError(f"camera_to_world for {camera_id} must be 4x4, got {camera_to_world.shape}")
    if not np.all(np.isfinite(camera_to_world)):
        raise DataValidationError(f"camera_to_world for {camera_id} contains non-finite values")
    if not np.allclose(camera_to_world[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise DataValidationError(f"camera_to_world for {camera_id} has an invalid homogeneous last row")
    rotation = camera_to_world[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise DataValidationError(f"camera_to_world rotation for {camera_id} must be a proper orthonormal matrix")


def _load_scene_array(scene_dir: Path, files: dict[str, Any], field: str, camera_id: str) -> np.ndarray:
    try:
        relative_value = files[field]
    except KeyError as exc:
        raise DataValidationError(f"files for {camera_id} is missing {field!r}") from exc
    if not isinstance(relative_value, str) or not relative_value:
        raise DataValidationError(f"{field} path for {camera_id} must be a non-empty relative string")
    relative_path = Path(relative_value)
    if relative_path.is_absolute():
        raise DataValidationError(f"{field} path for {camera_id} must be relative: {relative_path}")
    scene_root = scene_dir.resolve()
    input_path = (scene_root / relative_path).resolve()
    if not input_path.is_relative_to(scene_root):
        raise DataValidationError(f"{field} path for {camera_id} escapes the scene directory: {relative_path}")
    try:
        return np.load(input_path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise UnsafeDataError(f"refusing object array for {field} of {camera_id}: {input_path}") from exc
        raise DataValidationError(f"invalid NumPy array for {field} of {camera_id}: {input_path}: {exc}") from exc


def _require_mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataValidationError(f"{description} must be a JSON object")
    return value
