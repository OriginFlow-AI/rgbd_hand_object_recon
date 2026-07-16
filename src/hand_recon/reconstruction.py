"""Multi-view RGB-D point-cloud reconstruction utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from hand_recon.icp import apply_transform, validate_points, voxel_downsample
from hand_recon.rgbd import RgbdScene, backproject_depth_to_points, depth_valid_ratio


@dataclass(frozen=True)
class ReconstructionResult:
    raw_points: np.ndarray
    raw_colors_rgb: np.ndarray
    raw_view_indices: np.ndarray
    fused_points: np.ndarray
    fused_colors_rgb: np.ndarray
    per_view_points: list[np.ndarray]
    per_view_colors_rgb: list[np.ndarray]
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
    color_clouds = []
    raw_view_indices = []
    per_view_points: list[np.ndarray] = []
    per_view_colors: list[np.ndarray] = []
    per_view_stats: list[dict[str, float | int | str]] = []

    for view_index, view in enumerate(scene.views):
        all_camera_points = backproject_depth_to_points(view.depth, view.camera)
        selected_camera_points = backproject_depth_to_points(
            view.depth,
            view.camera,
            mask=view.mask,
            valid_labels=valid_labels,
        )
        selected_world_points = transform_points(selected_camera_points, view.camera.camera_to_world)
        selected = np.isfinite(view.depth) & (view.depth > 0.0)
        if valid_labels is not None:
            selected &= np.isin(view.mask, np.asarray(valid_labels, dtype=view.mask.dtype))
        selected_colors = np.clip(np.asarray(view.rgb)[selected, :3], 0, 255).astype(np.uint8)
        world_clouds.append(selected_world_points)
        color_clouds.append(selected_colors)
        raw_view_indices.append(np.full(selected_world_points.shape[0], view_index, dtype=np.int32))
        per_view_points.append(selected_world_points)
        per_view_colors.append(selected_colors)
        per_view_stats.append(
            {
                "camera_id": view.camera.camera_id,
                "depth_valid_ratio": depth_valid_ratio(view.depth),
                "valid_depth_point_count": int(all_camera_points.shape[0]),
                "selected_point_count": int(selected_world_points.shape[0]),
            }
        )

    raw_points = np.vstack(world_clouds) if world_clouds else np.zeros((0, 3), dtype=np.float64)
    raw_colors = np.vstack(color_clouds) if color_clouds else np.zeros((0, 3), dtype=np.uint8)
    view_indices = np.concatenate(raw_view_indices) if raw_view_indices else np.zeros((0,), dtype=np.int32)
    raw_points = validate_points(raw_points) if raw_points.size else np.zeros((0, 3), dtype=np.float64)
    fused_points, fused_colors = voxel_downsample(raw_points, colors=raw_colors, voxel_size=voxel_size_m)
    if fused_colors is None:
        fused_colors = np.zeros((fused_points.shape[0], 3), dtype=np.uint8)
    return ReconstructionResult(
        raw_points=raw_points,
        raw_colors_rgb=raw_colors,
        raw_view_indices=view_indices,
        fused_points=fused_points,
        fused_colors_rgb=fused_colors,
        per_view_points=per_view_points,
        per_view_colors_rgb=per_view_colors,
        per_view_stats=per_view_stats,
    )


def _normalize_labels(labels: Iterable[int] | int | None) -> list[int] | None:
    if labels is None:
        return None
    if isinstance(labels, int):
        return [labels]
    return [int(label) for label in labels]
