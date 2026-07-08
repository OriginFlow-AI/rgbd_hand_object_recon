"""Multi-view RGB-D point-cloud reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from hand_recon.icp import apply_transform, validate_points, voxel_downsample
from hand_recon.rgbd import RgbdScene, backproject_depth_to_points, depth_valid_ratio


@dataclass(frozen=True)
class ReconstructionResult:
    raw_points: np.ndarray
    fused_points: np.ndarray
    per_view_stats: list[dict[str, float | int | str]]


def transform_points(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return apply_transform(points, camera_to_world)


def reconstruct_multiview_pointcloud(
    scene: RgbdScene,
    labels: Iterable[int] | int | None = None,
    voxel_size_m: float = 0.003,
) -> ReconstructionResult:
    valid_labels = _normalize_labels(labels)
    world_clouds = []
    per_view_stats: list[dict[str, float | int | str]] = []

    for view in scene.views:
        all_camera_points = backproject_depth_to_points(view.depth, view.camera)
        selected_camera_points = backproject_depth_to_points(
            view.depth,
            view.camera,
            mask=view.mask,
            valid_labels=valid_labels,
        )
        selected_world_points = transform_points(selected_camera_points, view.camera.camera_to_world)
        world_clouds.append(selected_world_points)
        per_view_stats.append(
            {
                "camera_id": view.camera.camera_id,
                "depth_valid_ratio": depth_valid_ratio(view.depth),
                "valid_depth_point_count": int(all_camera_points.shape[0]),
                "selected_point_count": int(selected_world_points.shape[0]),
            }
        )

    raw_points = np.vstack(world_clouds) if world_clouds else np.zeros((0, 3), dtype=np.float64)
    raw_points = validate_points(raw_points) if raw_points.size else np.zeros((0, 3), dtype=np.float64)
    fused_points, _ = voxel_downsample(raw_points, voxel_size=voxel_size_m)
    return ReconstructionResult(raw_points=raw_points, fused_points=fused_points, per_view_stats=per_view_stats)


def _normalize_labels(labels: Iterable[int] | int | None) -> list[int] | None:
    if labels is None:
        return None
    if isinstance(labels, int):
        return [labels]
    return [int(label) for label in labels]
