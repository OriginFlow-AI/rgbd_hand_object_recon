"""Quality metrics for the mock multi-view RGB-D reconstruction pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from hand_recon.pose import pose_confidences
from hand_recon.reconstruction import ReconstructionResult
from hand_recon.rgbd import RgbdScene, depth_valid_ratio


def evaluate_quality(
    scene: RgbdScene,
    fused_result: ReconstructionResult,
    hand_points: np.ndarray,
    object_points: np.ndarray,
    pose_output: dict[str, Any],
    min_valid_depth_ratio: float = 0.01,
    min_fused_points: int = 100,
) -> dict[str, Any]:
    ratios = {view.camera.camera_id: depth_valid_ratio(view.depth) for view in scene.views}
    views_with_valid_depth = sum(1 for value in ratios.values() if value >= min_valid_depth_ratio)
    fused_points = np.asarray(fused_result.fused_points, dtype=np.float64)
    hand_points = np.asarray(hand_points, dtype=np.float64)
    object_points = np.asarray(object_points, dtype=np.float64)
    confidences = pose_confidences(pose_output)
    warnings = []

    if views_with_valid_depth < len(scene.views):
        warnings.append("one_or_more_views_have_low_valid_depth_ratio")
    if fused_points.shape[0] < min_fused_points:
        warnings.append("fused_point_count_below_threshold")
    if hand_points.shape[0] == 0:
        warnings.append("no_hand_points")
    if object_points.shape[0] == 0:
        warnings.append("no_object_points")

    bbox_extent = _bbox_extent(fused_points)
    coverage_score = views_with_valid_depth / max(1, len(scene.views))
    pose_confidence_mean = float(np.mean(confidences)) if confidences else 0.0
    passed = len(warnings) == 0

    return {
        "status": "ok" if passed else "failed",
        "scene_id": scene.scene_id,
        "view_count": len(scene.views),
        "metrics": {
            "depth_valid_ratio_mean": round(float(np.mean(list(ratios.values()))) if ratios else 0.0, 6),
            "depth_valid_ratio_per_view": {key: round(float(value), 6) for key, value in ratios.items()},
            "hand_point_count": int(hand_points.shape[0]),
            "object_point_count": int(object_points.shape[0]),
            "raw_point_count": int(fused_result.raw_points.shape[0]),
            "fused_point_count": int(fused_points.shape[0]),
            "views_with_valid_depth": int(views_with_valid_depth),
            "bbox_extent_m": [round(float(value), 8) for value in bbox_extent],
            "coverage_score": round(float(coverage_score), 6),
            "pose_confidence_mean": round(float(pose_confidence_mean), 6),
        },
        "thresholds": {
            "min_valid_depth_ratio": float(min_valid_depth_ratio),
            "min_fused_points": int(min_fused_points),
        },
        "passed": passed,
        "warnings": warnings,
    }


def _bbox_extent(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(3, dtype=np.float64)
    return points.max(axis=0) - points.min(axis=0)
