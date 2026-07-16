"""Dependency-light projective TSDF fusion for calibrated RGB-D views."""

from __future__ import annotations

import math

import numpy as np

from hand_recon.domain import TsdfVolume
from hand_recon.rgbd import RgbdScene


def build_masked_tsdf(
    scene: RgbdScene,
    observed_points_m: np.ndarray,
    *,
    label: int,
    voxel_size_m: float = 0.003,
    truncation_m: float = 0.009,
    padding_m: float = 0.012,
    max_voxel_count: int = 2_000_000,
) -> TsdfVolume:
    """Fuse masked depth evidence into a bounded world-space TSDF.

    Positive values are in front of the measured surface and negative values
    are behind it. Voxels without any valid masked observation retain zero
    weight and are never used for surface extraction.
    """

    points = np.asarray(observed_points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 3:
        raise ValueError("observed_points_m must contain at least three XYZ points")
    if not np.all(np.isfinite(points)):
        raise ValueError("observed_points_m contains non-finite values")
    for name, value in (
        ("voxel_size_m", voxel_size_m),
        ("truncation_m", truncation_m),
        ("padding_m", padding_m),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite value")
    if truncation_m < 2.0 * voxel_size_m:
        raise ValueError("truncation_m must be at least twice voxel_size_m")
    if max_voxel_count <= 0:
        raise ValueError("max_voxel_count must be greater than zero")

    origin = points.min(axis=0) - padding_m
    upper = points.max(axis=0) + padding_m
    shape = np.ceil((upper - origin) / voxel_size_m).astype(np.int64) + 1
    voxel_count = int(np.prod(shape, dtype=np.int64))
    if voxel_count > max_voxel_count:
        raise ValueError(f"TSDF grid would contain {voxel_count} voxels, exceeding max_voxel_count={max_voxel_count}")

    axes = [origin[axis] + np.arange(int(shape[axis]), dtype=np.float64) * voxel_size_m for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    world_points = grid.reshape(-1, 3)
    weighted_sum = np.zeros((voxel_count,), dtype=np.float64)
    weights = np.zeros((voxel_count,), dtype=np.float64)
    view_count = np.zeros((voxel_count,), dtype=np.uint16)

    for view in scene.views:
        world_to_camera = np.linalg.inv(view.camera.camera_to_world)
        camera_points = world_points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z = camera_points[:, 2]
        in_front = np.isfinite(z) & (z > 0.0)
        safe_z = np.where(in_front, z, 1.0)
        cols = np.rint(view.camera.fx * camera_points[:, 0] / safe_z + view.camera.cx).astype(np.int64)
        rows = np.rint(view.camera.fy * camera_points[:, 1] / safe_z + view.camera.cy).astype(np.int64)
        inside = in_front & (rows >= 0) & (rows < view.camera.height) & (cols >= 0) & (cols < view.camera.width)
        candidate = np.flatnonzero(inside)
        if candidate.size == 0:
            continue
        sampled_depth = np.asarray(view.depth, dtype=np.float64)[rows[candidate], cols[candidate]]
        sampled_mask = np.asarray(view.mask)[rows[candidate], cols[candidate]]
        signed_distance = sampled_depth - z[candidate]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & (sampled_mask == int(label))
            & (signed_distance >= -truncation_m)
        )
        selected = candidate[valid]
        if selected.size == 0:
            continue
        values = np.clip(signed_distance[valid] / truncation_m, -1.0, 1.0)
        weighted_sum[selected] += values
        weights[selected] += 1.0
        view_count[selected] += 1

    tsdf = np.ones((voxel_count,), dtype=np.float64)
    observed = weights > 0
    tsdf[observed] = weighted_sum[observed] / weights[observed]
    volume_shape = tuple(int(value) for value in shape)
    return TsdfVolume(
        origin_m=origin,
        voxel_size_m=float(voxel_size_m),
        values=tsdf.reshape(volume_shape),
        weights=weights.reshape(volume_shape),
        observed_view_count=view_count.reshape(volume_shape),
    )
