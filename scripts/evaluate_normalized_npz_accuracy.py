#!/usr/bin/env python3
"""Evaluate normalized hand NPZ reconstruction quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FIELDS = [
    "K",
    "baseline_m",
    "frame_index",
    "left_row",
    "right_row",
    "root_translation_m",
    "global_orient_residual_rotvec",
    "global_orient_residual_deg",
    "global_scale",
    "scale_group",
    "hand_side",
    "is_right",
    "wrist_pose_6d_left_m_rad",
    "hand_angles_20dof_rad",
    "hand_angles_20dof_deg",
    "hand_angle_names_20dof",
    "joint_names",
    "joints_3d_left_m",
    "valid_joint_mask",
    "anchor_mask",
    "visible_anchor_mask",
    "anchor_confidence",
    "anchor_confidence_source",
    "frame_confidence",
    "frame_status",
    "scale_meta_json",
    "summary_json",
]

BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prediction = _load_npz(args.prediction_npz)
    reference = _load_npz(args.reference_npz) if args.reference_npz else None
    report = evaluate_prediction(prediction, reference=reference)

    output_json = args.output_json or args.prediction_npz.with_name("accuracy_report.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


def evaluate_prediction(
    prediction: dict[str, np.ndarray],
    *,
    reference: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_FIELDS if field not in prediction]
    shape_errors = _validate_shapes(prediction)
    metrics = _self_consistency_metrics(prediction)
    reference_metrics = _reference_metrics(prediction, reference) if reference is not None else {}
    warnings = []

    if missing_fields:
        warnings.append("missing_required_fields")
    if shape_errors:
        warnings.append("invalid_field_shapes")
    if metrics.get("valid_joint_ratio", 0.0) < 0.95:
        warnings.append("low_valid_joint_ratio")
    if metrics.get("ok_frame_ratio", 0.0) < 0.95:
        warnings.append("low_ok_frame_ratio")
    if reference is not None and reference_metrics.get("matched_rows", 0) == 0:
        warnings.append("no_reference_rows_matched")

    return {
        "status": "ok" if not warnings else "failed",
        "passed": not warnings,
        "missing_fields": missing_fields,
        "shape_errors": shape_errors,
        "metrics": metrics,
        "reference_metrics": reference_metrics,
        "warnings": warnings,
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _validate_shapes(data: dict[str, np.ndarray]) -> list[str]:
    if "frame_index" not in data:
        return ["frame_index missing"]
    n = int(np.asarray(data["frame_index"]).shape[0])
    expected = {
        "K": (3, 3),
        "frame_index": (n,),
        "left_row": (n,),
        "right_row": (n,),
        "root_translation_m": (n, 3),
        "global_orient_residual_rotvec": (n, 3),
        "global_orient_residual_deg": (n,),
        "global_scale": (n,),
        "scale_group": (n,),
        "hand_side": (n,),
        "is_right": (n,),
        "wrist_pose_6d_left_m_rad": (n, 6),
        "hand_angles_20dof_rad": (n, 20),
        "hand_angles_20dof_deg": (n, 20),
        "hand_angle_names_20dof": (20,),
        "joint_names": (21,),
        "joints_3d_left_m": (n, 21, 3),
        "valid_joint_mask": (n, 21),
        "anchor_mask": (n, 21),
        "visible_anchor_mask": (n, 21),
        "anchor_confidence": (n, 21),
        "frame_confidence": (n,),
        "frame_status": (n,),
        "summary_json": (n,),
    }
    errors = []
    for key, shape in expected.items():
        if key in data and tuple(np.asarray(data[key]).shape) != shape:
            errors.append(f"{key} expected {shape}, got {tuple(np.asarray(data[key]).shape)}")
    return errors


def _self_consistency_metrics(data: dict[str, np.ndarray]) -> dict[str, Any]:
    if "frame_index" not in data:
        return {}
    n = int(np.asarray(data["frame_index"]).shape[0])
    joints = np.asarray(data.get("joints_3d_left_m", np.zeros((n, 21, 3))), dtype=np.float64)
    valid_mask = np.asarray(data.get("valid_joint_mask", np.zeros((n, 21))), dtype=bool)
    frame_status = np.asarray(data.get("frame_status", np.full((n,), "invalid")))
    frame_confidence = np.asarray(data.get("frame_confidence", np.zeros((n,))), dtype=np.float64)
    root = np.asarray(data.get("root_translation_m", np.zeros((n, 3))), dtype=np.float64)

    bone_lengths = _bone_lengths(joints)
    root_step = _root_step_metrics(data, root)
    return {
        "row_count": n,
        "ok_frame_ratio": _safe_mean(frame_status == "ok"),
        "valid_joint_ratio": _safe_mean(valid_mask),
        "frame_confidence_mean": _safe_mean(frame_confidence),
        "frame_confidence_min": _safe_min(frame_confidence),
        "bone_length_mean_m": _safe_mean(bone_lengths),
        "bone_length_p95_m": _safe_percentile(bone_lengths, 95),
        "root_step_median_m": root_step["median_m"],
        "root_step_p95_m": root_step["p95_m"],
    }


def _reference_metrics(data: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, Any]:
    pred_rows, ref_rows = _match_rows(data, reference)
    if not pred_rows:
        return {"matched_rows": 0}

    pred_joints = np.asarray(data["joints_3d_left_m"], dtype=np.float64)[pred_rows]
    ref_joints = np.asarray(reference["joints_3d_left_m"], dtype=np.float64)[ref_rows]
    pred_mask = np.asarray(data.get("valid_joint_mask", np.ones(pred_joints.shape[:2])), dtype=bool)[pred_rows]
    ref_mask = np.asarray(reference.get("valid_joint_mask", np.ones(ref_joints.shape[:2])), dtype=bool)[ref_rows]
    valid = pred_mask & ref_mask
    joint_error = np.linalg.norm(pred_joints - ref_joints, axis=2)
    valid_error = joint_error[valid]

    pred_root = np.asarray(data["root_translation_m"], dtype=np.float64)[pred_rows]
    ref_root = np.asarray(reference["root_translation_m"], dtype=np.float64)[ref_rows]
    root_error = np.linalg.norm(pred_root - ref_root, axis=1)

    metrics = {
        "matched_rows": len(pred_rows),
        "mpjpe_m": _safe_mean(valid_error),
        "mpjpe_mm": _safe_mean(valid_error) * 1000.0,
        "joint_error_p95_m": _safe_percentile(valid_error, 95),
        "root_translation_rmse_m": float(np.sqrt(np.mean(root_error**2))) if root_error.size else 0.0,
    }
    if "hand_angles_20dof_deg" in data and "hand_angles_20dof_deg" in reference:
        pred_angles = np.asarray(data["hand_angles_20dof_deg"], dtype=np.float64)[pred_rows]
        ref_angles = np.asarray(reference["hand_angles_20dof_deg"], dtype=np.float64)[ref_rows]
        metrics["hand_angle_mae_deg"] = _safe_mean(np.abs(pred_angles - ref_angles))
    return metrics


def _match_rows(data: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> tuple[list[int], list[int]]:
    if (
        "frame_index" not in data
        or "hand_side" not in data
        or "frame_index" not in reference
        or "hand_side" not in reference
    ):
        n = min(
            int(np.asarray(data.get("joints_3d_left_m", [])).shape[0]),
            int(np.asarray(reference.get("joints_3d_left_m", [])).shape[0]),
        )
        return list(range(n)), list(range(n))

    lookup = {}
    for idx, (frame_index, hand_side) in enumerate(zip(reference["frame_index"], reference["hand_side"], strict=False)):
        lookup[(int(frame_index), int(hand_side))] = idx
    pred_rows = []
    ref_rows = []
    for idx, (frame_index, hand_side) in enumerate(zip(data["frame_index"], data["hand_side"], strict=False)):
        key = (int(frame_index), int(hand_side))
        if key in lookup:
            pred_rows.append(idx)
            ref_rows.append(lookup[key])
    return pred_rows, ref_rows


def _bone_lengths(joints: np.ndarray) -> np.ndarray:
    if joints.ndim != 3 or joints.shape[1:] != (21, 3):
        return np.zeros((0,), dtype=np.float64)
    values = [np.linalg.norm(joints[:, child] - joints[:, parent], axis=1) for parent, child in BONES]
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float64)


def _root_step_metrics(data: dict[str, np.ndarray], root: np.ndarray) -> dict[str, float]:
    if root.shape[0] < 2:
        return {"median_m": 0.0, "p95_m": 0.0}
    frame_index = np.asarray(data.get("frame_index", np.arange(root.shape[0])), dtype=np.int64)
    hand_side = np.asarray(data.get("hand_side", np.zeros(root.shape[0])), dtype=np.int64)
    steps = []
    for side in sorted(set(int(value) for value in hand_side.tolist())):
        indices = np.where(hand_side == side)[0]
        indices = indices[np.argsort(frame_index[indices])]
        if indices.size >= 2:
            steps.append(np.linalg.norm(np.diff(root[indices], axis=0), axis=1))
    all_steps = np.concatenate(steps) if steps else np.zeros((0,), dtype=np.float64)
    return {"median_m": _safe_percentile(all_steps, 50), "p95_m": _safe_percentile(all_steps, 95)}


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(np.mean(values)) if values.size else 0.0


def _safe_min(values: np.ndarray) -> float:
    values = np.asarray(values)
    return float(np.min(values)) if values.size else 0.0


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values)
    return float(np.percentile(values, percentile)) if values.size else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
