"""Synthetic multi-view RGB-D data for the KR1 demo pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

LABEL_BACKGROUND = 0
LABEL_HAND = 1
LABEL_OBJECT = 2


def generate_mock_rgbd_scene(
    scene_dir: Path,
    scene_id: str = "rgbd_scene_001",
    view_count: int = 4,
    width: int = 128,
    height: int = 96,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a small deterministic RGB-D scene with hand/object labels.

    The data is synthetic by design: a simple point-renderer projects known
    world-space hand and object surfaces into several RGB-D cameras.  The demo
    then reconstructs those points only from RGB-D files and camera metadata.
    """

    scene_dir = Path(scene_dir)
    if view_count <= 0:
        raise ValueError("view_count must be greater than zero")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    if scene_dir.exists() and overwrite:
        # Only remove files owned by this generator. Deleting the caller's
        # entire directory could destroy unrelated data placed alongside it.
        (scene_dir / "cameras.json").unlink(missing_ok=True)
        frames_dir = scene_dir / "frames"
        if frames_dir.is_symlink() or frames_dir.is_file():
            frames_dir.unlink()
        elif frames_dir.exists():
            shutil.rmtree(frames_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)

    cameras_path = scene_dir / "cameras.json"
    if cameras_path.exists() and not overwrite:
        return json.loads(cameras_path.read_text(encoding="utf-8"))

    hand_points = _make_hand_points()
    object_points = _make_object_points()
    points = np.vstack([hand_points, object_points])
    labels = np.concatenate(
        [
            np.full(hand_points.shape[0], LABEL_HAND, dtype=np.uint8),
            np.full(object_points.shape[0], LABEL_OBJECT, dtype=np.uint8),
        ]
    )
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors[labels == LABEL_HAND] = np.array([230, 172, 128], dtype=np.uint8)
    colors[labels == LABEL_OBJECT] = np.array([70, 135, 230], dtype=np.uint8)

    timestamp = "2026-07-07T00:00:00+08:00"
    target = np.array([0.0, 0.0, 0.43], dtype=np.float64)
    radius = 0.64
    camera_height = 0.46
    fx = fy = 118.0
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    views: list[dict[str, Any]] = []
    for index in range(view_count):
        angle = 2.0 * np.pi * index / view_count + np.pi / 4.0
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), camera_height], dtype=np.float64)
        camera_to_world = _look_at_camera_to_world(eye, target)
        depth, mask, rgb = _render_rgbd(
            points=points,
            labels=labels,
            colors=colors,
            camera_to_world=camera_to_world,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

        view_dir = scene_dir / "frames" / f"view_{index:02d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        np.save(view_dir / "rgb.npy", rgb)
        np.save(view_dir / "depth.npy", depth)
        np.save(view_dir / "mask.npy", mask)

        views.append(
            {
                "camera_id": f"view_{index:02d}",
                "width": width,
                "height": height,
                "timestamp": timestamp,
                "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "extrinsics": {"camera_to_world": camera_to_world.tolist()},
                "files": {
                    "rgb": f"frames/view_{index:02d}/rgb.npy",
                    "depth": f"frames/view_{index:02d}/depth.npy",
                    "mask": f"frames/view_{index:02d}/mask.npy",
                },
            }
        )

    metadata = {
        "scene_id": scene_id,
        "coordinate_frame": "world",
        "depth_unit": "meter",
        "mask_labels": {"background": LABEL_BACKGROUND, "hand": LABEL_HAND, "object": LABEL_OBJECT},
        "views": views,
        "source_geometry": {
            "hand_point_count": int(hand_points.shape[0]),
            "object_point_count": int(object_points.shape[0]),
        },
    }
    cameras_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _make_hand_points() -> np.ndarray:
    rng = np.random.default_rng(20260707)
    parts = []

    palm = _ellipsoid_surface(center=(-0.055, -0.015, 0.405), radii=(0.038, 0.022, 0.052), u_count=42, v_count=22)
    parts.append(palm)

    finger_x = [-0.085, -0.067, -0.049, -0.031, -0.016]
    lengths = [0.052, 0.068, 0.076, 0.066, 0.052]
    radii = [0.0065, 0.0070, 0.0072, 0.0067, 0.0058]
    for idx, (x_base, length, radius) in enumerate(zip(finger_x, lengths, radii, strict=True)):
        base = np.array([x_base, -0.012 + 0.002 * idx, 0.445], dtype=np.float64)
        axis = np.array([0.006 * (idx - 2), 0.002, length], dtype=np.float64)
        parts.append(_cylinder_surface(base=base, axis=axis, radius=radius, height_steps=38, angle_steps=20))

    thumb_base = np.array([-0.086, -0.016, 0.395], dtype=np.float64)
    thumb_axis = np.array([-0.050, 0.014, 0.038], dtype=np.float64)
    parts.append(_cylinder_surface(base=thumb_base, axis=thumb_axis, radius=0.008, height_steps=32, angle_steps=20))

    points = np.vstack(parts)
    points += rng.normal(0.0, 0.0005, size=points.shape)
    return points.astype(np.float64)


def _make_object_points() -> np.ndarray:
    cube = _box_surface(center=(0.075, 0.025, 0.405), size=(0.070, 0.054, 0.060), steps=22)
    top = _ellipsoid_surface(center=(0.075, 0.025, 0.458), radii=(0.028, 0.028, 0.016), u_count=30, v_count=16)
    return np.vstack([cube, top]).astype(np.float64)


def _ellipsoid_surface(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    u_count: int,
    v_count: int,
) -> np.ndarray:
    u = np.linspace(0.0, 2.0 * np.pi, u_count, endpoint=False)
    v = np.linspace(0.08 * np.pi, 0.92 * np.pi, v_count)
    uu, vv = np.meshgrid(u, v)
    center_arr = np.asarray(center, dtype=np.float64)
    radii_arr = np.asarray(radii, dtype=np.float64)
    points = np.column_stack(
        [
            radii_arr[0] * np.cos(uu).ravel() * np.sin(vv).ravel(),
            radii_arr[1] * np.sin(uu).ravel() * np.sin(vv).ravel(),
            radii_arr[2] * np.cos(vv).ravel(),
        ]
    )
    return points + center_arr


def _cylinder_surface(
    base: np.ndarray,
    axis: np.ndarray,
    radius: float,
    height_steps: int,
    angle_steps: int,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    direction = axis / np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, helper))) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    normal_a = np.cross(direction, helper)
    normal_a /= np.linalg.norm(normal_a)
    normal_b = np.cross(direction, normal_a)
    t = np.linspace(0.0, 1.0, height_steps)
    angles = np.linspace(0.0, 2.0 * np.pi, angle_steps, endpoint=False)
    tt, aa = np.meshgrid(t, angles)
    ring = radius * np.cos(aa)[..., None] * normal_a + radius * np.sin(aa)[..., None] * normal_b
    points = base + tt[..., None] * axis + ring
    return points.reshape(-1, 3)


def _box_surface(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    steps: int,
) -> np.ndarray:
    center_arr = np.asarray(center, dtype=np.float64)
    half = np.asarray(size, dtype=np.float64) / 2.0
    xs = np.linspace(-half[0], half[0], steps)
    ys = np.linspace(-half[1], half[1], steps)
    zs = np.linspace(-half[2], half[2], steps)
    faces = []
    for z in (-half[2], half[2]):
        xx, yy = np.meshgrid(xs, ys)
        faces.append(np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)]))
    for y in (-half[1], half[1]):
        xx, zz = np.meshgrid(xs, zs)
        faces.append(np.column_stack([xx.ravel(), np.full(xx.size, y), zz.ravel()]))
    for x in (-half[0], half[0]):
        yy, zz = np.meshgrid(ys, zs)
        faces.append(np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()]))
    return np.vstack(faces) + center_arr


def _look_at_camera_to_world(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = right
    transform[:3, 1] = down
    transform[:3, 2] = forward
    transform[:3, 3] = eye
    return transform


def _render_rgbd(
    points: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2]
    valid_z = z > 0.05
    projected = camera_points[valid_z]
    projected_labels = labels[valid_z]
    projected_colors = colors[valid_z]
    z = projected[:, 2]
    u = np.rint(fx * projected[:, 0] / z + cx).astype(np.int64)
    v = np.rint(fy * projected[:, 1] / z + cy).astype(np.int64)
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_uv]
    v = v[valid_uv]
    z = z[valid_uv]
    projected_labels = projected_labels[valid_uv]
    projected_colors = projected_colors[valid_uv]

    depth = np.full((height, width), np.inf, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    order = np.argsort(z)
    for idx in order:
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                uu = int(u[idx] + du)
                vv = int(v[idx] + dv)
                if 0 <= uu < width and 0 <= vv < height and z[idx] < depth[vv, uu]:
                    depth[vv, uu] = z[idx]
                    mask[vv, uu] = projected_labels[idx]
                    rgb[vv, uu] = projected_colors[idx]

    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32), mask, rgb
