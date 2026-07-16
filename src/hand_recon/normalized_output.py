"""Normalized hand NPZ output helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hand_recon.icp import apply_transform
from hand_recon.io.npz import write_npz_arrays
from hand_recon.rgbd import RgbdScene

JOINT_NAMES = np.array(
    [
        "wrist",
        "thumb_cmc",
        "thumb_mcp",
        "thumb_ip",
        "thumb_tip",
        "index_mcp",
        "index_pip",
        "index_dip",
        "index_tip",
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "middle_tip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "ring_tip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
        "pinky_tip",
    ],
    dtype="<U16",
)

HAND_ANGLE_NAMES_20DOF = np.array(
    [
        "thumb_cmc_flexion",
        "thumb_cmc_abduction",
        "thumb_mcp_flexion",
        "thumb_ip_flexion",
        "index_mcp_abduction",
        "index_mcp_flexion",
        "index_pip_flexion",
        "index_dip_flexion",
        "middle_mcp_abduction",
        "middle_mcp_flexion",
        "middle_pip_flexion",
        "middle_dip_flexion",
        "ring_mcp_abduction",
        "ring_mcp_flexion",
        "ring_pip_flexion",
        "ring_dip_flexion",
        "pinky_mcp_abduction",
        "pinky_mcp_flexion",
        "pinky_pip_flexion",
        "pinky_dip_flexion",
    ],
    dtype="<U24",
)


def build_normalized_hand_npz_payload(
    scene: RgbdScene,
    hand_points_world: np.ndarray,
    *,
    frame_index: int = 0,
    is_right: bool = True,
    left_row: int = -1,
    right_row: int = -1,
    global_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build a schema-compatible normalized hand result payload.

    The KR1 demo does not run WiLoR or stereo optimization, so row indices and
    orientation residuals are placeholders.  The 21 joints are deterministic
    geometric landmarks estimated from the reconstructed hand point cloud.
    """

    hand_points_world = np.asarray(hand_points_world, dtype=np.float64)
    camera = scene.views[0].camera if scene.views else None
    valid_points = hand_points_world.ndim == 2 and hand_points_world.shape[0] > 0 and hand_points_world.shape[1] == 3
    status = "ok" if valid_points and camera is not None else "invalid"
    hand_side = 1 if is_right else 0

    if camera is None:
        k_matrix = np.eye(3, dtype=np.float64)
        baseline_m = 0.0
        world_to_left_camera = np.eye(4, dtype=np.float64)
    else:
        k_matrix = np.array(
            [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        baseline_m = _estimate_baseline_m(scene)
        world_to_left_camera = np.linalg.inv(camera.camera_to_world)

    if status == "ok":
        joints_world = estimate_mock_21_joints(hand_points_world)
        joints_left = apply_transform(joints_world, world_to_left_camera)
        valid_joint_mask = np.ones((1, len(JOINT_NAMES)), dtype=bool)
    else:
        joints_left = np.zeros((len(JOINT_NAMES), 3), dtype=np.float64)
        valid_joint_mask = np.zeros((1, len(JOINT_NAMES)), dtype=bool)

    joints_3d_left_m = joints_left.reshape(1, len(JOINT_NAMES), 3)
    global_orient_residual_rotvec = np.zeros((1, 3), dtype=np.float64)
    if status == "ok":
        hand_angles_rad = compute_hand_angles_20dof(joints_3d_left_m[0]).reshape(1, 20)
    else:
        hand_angles_rad = np.zeros((1, 20), dtype=np.float64)
    wrist_pose = np.hstack([joints_3d_left_m[:, 0, :], global_orient_residual_rotvec])
    anchor_mask = valid_joint_mask.copy()
    anchor_confidence = anchor_mask.astype(np.float64)
    frame_confidence = np.array([float(anchor_confidence.mean())], dtype=np.float64)
    summary = {
        "scene_id": scene.scene_id,
        "frame_index": int(frame_index),
        "hand_side": int(hand_side),
        "is_right": bool(is_right),
        "status": status,
        "source": "mock_rgbd_pointcloud",
        "point_count": int(hand_points_world.shape[0]) if hand_points_world.ndim == 2 else 0,
        "coordinate_frame": "rectified_left_camera",
        "position_unit": "m",
        "rotation_unit": "rad",
    }
    scale_meta = {
        "source": "mock_placeholder",
        "global_scale": float(global_scale),
        "scale_group": int(hand_side),
        "baseline_m": float(baseline_m),
        "note": "KR1 mock demo has no WiLoR/stereo optimizer rows; row fields use -1 placeholders.",
    }

    return {
        "K": k_matrix,
        "baseline_m": np.array(float(baseline_m), dtype=np.float64),
        "frame_index": np.array([int(frame_index)], dtype=np.int64),
        "left_row": np.array([int(left_row)], dtype=np.int64),
        "right_row": np.array([int(right_row)], dtype=np.int64),
        "root_translation_m": joints_3d_left_m[:, 0, :].copy(),
        "global_orient_residual_rotvec": global_orient_residual_rotvec,
        "global_orient_residual_deg": np.zeros((1,), dtype=np.float64),
        "global_scale": np.array([float(global_scale)], dtype=np.float64),
        "scale_group": np.array([int(hand_side)], dtype=np.int64),
        "hand_side": np.array([int(hand_side)], dtype=np.int64),
        "is_right": np.array([bool(is_right)], dtype=bool),
        "wrist_pose_6d_left_m_rad": wrist_pose,
        "hand_angles_20dof_rad": hand_angles_rad,
        "hand_angles_20dof_deg": np.rad2deg(hand_angles_rad),
        "hand_angle_names_20dof": HAND_ANGLE_NAMES_20DOF.copy(),
        "joint_names": JOINT_NAMES.copy(),
        "joints_3d_left_m": joints_3d_left_m,
        "valid_joint_mask": valid_joint_mask,
        "anchor_mask": anchor_mask,
        "visible_anchor_mask": anchor_mask.copy(),
        "anchor_confidence": anchor_confidence,
        "anchor_confidence_source": np.array("anchor_mask_placeholder", dtype=np.str_),
        "frame_confidence": frame_confidence,
        "frame_status": np.array([status], dtype="<U16"),
        "scale_meta_json": np.array(json.dumps(scale_meta, ensure_ascii=False), dtype=np.str_),
        "summary_json": np.array([json.dumps(summary, ensure_ascii=False)], dtype=np.str_),
    }


def write_normalized_hand_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Atomically write a normalized payload without pickle-backed arrays."""

    write_npz_arrays(path, payload)


def estimate_mock_21_joints(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        return np.zeros((len(JOINT_NAMES), 3), dtype=np.float64)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = np.maximum(maxs - mins, 1e-6)
    order = np.argsort(extents)
    long_axis_idx = int(order[-1])
    cross_axis_idx = int(order[-2])
    thick_axis_idx = int(order[0])
    center = points.mean(axis=0)

    joints = np.zeros((len(JOINT_NAMES), 3), dtype=np.float64)

    def point(cross_frac: float, long_frac: float, thick_frac: float = 0.0) -> np.ndarray:
        value = center.copy()
        value[cross_axis_idx] = mins[cross_axis_idx] + cross_frac * extents[cross_axis_idx]
        value[long_axis_idx] = mins[long_axis_idx] + long_frac * extents[long_axis_idx]
        value[thick_axis_idx] = center[thick_axis_idx] + thick_frac * extents[thick_axis_idx]
        return value

    joints[0] = point(0.52, 0.13)
    joints[1] = point(0.12, 0.34)
    joints[2] = point(0.04, 0.44, 0.04)
    joints[3] = point(0.00, 0.56, 0.08)
    joints[4] = point(-0.04, 0.68, 0.10)

    finger_specs = {
        5: (0.25, (0.55, 0.72, 0.86, 0.98)),
        9: (0.45, (0.57, 0.76, 0.90, 1.02)),
        13: (0.65, (0.55, 0.72, 0.86, 0.98)),
        17: (0.84, (0.52, 0.67, 0.80, 0.92)),
    }
    for start, (cross_frac, long_fracs) in finger_specs.items():
        for offset, long_frac in enumerate(long_fracs):
            joints[start + offset] = point(cross_frac, long_frac)

    return joints


def compute_hand_angles_20dof(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.shape != (len(JOINT_NAMES), 3):
        raise ValueError(f"joints must have shape (21, 3), got {joints.shape}")

    frame = _palm_frame(joints)
    angles = np.zeros(20, dtype=np.float64)

    thumb_abduction, thumb_flexion = _base_flexion_abduction(joints[2] - joints[1], frame)
    angles[0] = thumb_flexion
    angles[1] = thumb_abduction
    angles[2] = _joint_flexion(joints[1], joints[2], joints[3])
    angles[3] = _joint_flexion(joints[2], joints[3], joints[4])

    cursor = 4
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        abduction, flexion = _base_flexion_abduction(joints[pip] - joints[mcp], frame)
        angles[cursor] = abduction
        angles[cursor + 1] = flexion
        angles[cursor + 2] = _joint_flexion(joints[mcp], joints[pip], joints[dip])
        angles[cursor + 3] = _joint_flexion(joints[pip], joints[dip], joints[tip])
        cursor += 4

    return angles


def _estimate_baseline_m(scene: RgbdScene) -> float:
    if len(scene.views) < 2:
        return 0.0
    left = scene.views[0].camera.camera_to_world[:3, 3]
    right = scene.views[1].camera.camera_to_world[:3, 3]
    return float(np.linalg.norm(right - left))


def _palm_frame(joints: np.ndarray) -> np.ndarray:
    wrist = joints[0]
    index_mcp = joints[5]
    middle_mcp = joints[9]
    pinky_mcp = joints[17]

    x_axis = _safe_normalize(pinky_mcp - index_mcp, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_axis = _safe_normalize(middle_mcp - wrist, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    z_axis = _safe_normalize(np.cross(x_axis, y_axis), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    y_axis = _safe_normalize(np.cross(z_axis, x_axis), y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _base_flexion_abduction(vector: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
    local = frame.T @ np.asarray(vector, dtype=np.float64)
    abduction = float(np.arctan2(local[0], local[1]))
    flexion = float(np.arctan2(abs(local[2]), local[1]))
    return abduction, flexion


def _joint_flexion(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    angle = _angle_between(np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b))
    return float(np.pi - angle)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = _safe_normalize(np.asarray(a, dtype=np.float64), np.zeros(3, dtype=np.float64))
    b_unit = _safe_normalize(np.asarray(b, dtype=np.float64), np.zeros(3, dtype=np.float64))
    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return float(np.arccos(dot))


def _safe_normalize(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm
