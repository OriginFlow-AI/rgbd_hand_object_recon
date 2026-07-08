"""Mock pose output helpers for the KR1 RGB-D demo."""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_mock_pose(points: np.ndarray, label_name: str, item_id: str) -> dict[str, Any]:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return {
            "id": item_id,
            "label": label_name,
            "pose_type": "mock_bbox_centroid",
            "status": "no_points",
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_matrix": np.eye(3).tolist(),
            "bbox_3d_m": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
            "point_count": 0,
            "confidence": 0.0,
        }

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = points.mean(axis=0)
    confidence = min(0.95, max(0.05, float(points.shape[0]) / 900.0))
    return {
        "id": item_id,
        "label": label_name,
        "pose_type": "mock_bbox_centroid",
        "status": "ok",
        "translation_m": _round_vector(center),
        "rotation_matrix": np.eye(3).tolist(),
        "bbox_3d_m": {"min": _round_vector(mins), "max": _round_vector(maxs)},
        "point_count": int(points.shape[0]),
        "confidence": round(confidence, 6),
    }


def generate_pose_output(
    hand_points: np.ndarray,
    object_points: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    hand_pose = estimate_mock_pose(hand_points, "hand", "hand_0")
    object_pose = estimate_mock_pose(object_points, "object", "object_0")
    return {
        "status": "ok" if hand_pose["status"] == "ok" and object_pose["status"] == "ok" else "partial",
        "scene_id": metadata.get("scene_id", ""),
        "coordinate_frame": metadata.get("coordinate_frame", "world"),
        "timestamp": metadata.get("timestamp", ""),
        "hands": [hand_pose],
        "objects": [object_pose],
    }


def pose_confidences(pose_output: dict[str, Any]) -> list[float]:
    values = []
    for group_name in ("hands", "objects"):
        for item in pose_output.get(group_name, []):
            values.append(float(item.get("confidence", 0.0)))
    return values


def _round_vector(values: np.ndarray) -> list[float]:
    return [round(float(value), 8) for value in np.asarray(values, dtype=np.float64).reshape(3)]
