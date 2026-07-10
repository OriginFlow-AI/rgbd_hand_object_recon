"""KR3 unified hand-result interface helpers.

The interface is intentionally adapter-friendly: ground-truth capture, DMA
vision, and a multi-modal super-labelator can all emit the same payload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hand_recon.normalized_output import HAND_ANGLE_NAMES_20DOF, JOINT_NAMES

HAND_RESULT_SCHEMA_VERSION = "kr3_hand_result_v0.1"

SOURCE_SYSTEMS = (
    "ground_truth_system",
    "dma_vision",
    "super_labelator",
)

HAND_MODEL_TYPES = (
    "umetrack",
    "mano",
    "hybrid",
    "mock",
    "unknown",
)

COORDINATE_FRAMES = (
    "rectified_left_camera",
    "world",
    "camera",
    "wrist_local",
)

UMETRACK_ANGLE_NAMES_22DOF = np.array(
    list(HAND_ANGLE_NAMES_20DOF)
    + [
        "wrist_flexion",
        "wrist_abduction",
    ],
    dtype="<U32",
)

KR3_REQUIRED_FIELDS = (
    "schema_version",
    "source_system",
    "frame_index",
    "timestamp_ns",
    "track_id",
    "hand_side",
    "is_right",
    "coordinate_frame",
    "wrist_pose_6d_m_rad",
    "hand_angles_22dof_rad",
    "hand_angles_22dof_deg",
    "hand_angle_names_22dof",
    "hand_angle_convention",
    "joint_names",
    "joints_3d_m",
    "valid_joint_mask",
    "mesh_vertices_m",
    "mesh_faces",
    "mesh_vertex_valid_mask",
    "mesh_model",
    "mesh_topology_id",
    "frame_confidence",
    "frame_status",
    "provenance_json",
)


@dataclass(frozen=True)
class AdapterContract:
    """Expected KR3 adapter surface for one upstream system."""

    source_system: str
    input_summary: tuple[str, ...]
    output_fields: tuple[str, ...] = KR3_REQUIRED_FIELDS


GROUND_TRUTH_SYSTEM_CONTRACT = AdapterContract(
    source_system="ground_truth_system",
    input_summary=(
        "synchronized camera/depth frames",
        "calibration and timestamp records",
        "high-confidence hand labels or model fits",
    ),
)

DMA_VISION_CONTRACT = AdapterContract(
    source_system="dma_vision",
    input_summary=(
        "pure RGB or multi-view RGB frames",
        "camera metadata",
        "DMA hand predictions",
    ),
)

SUPER_LABELATOR_CONTRACT = AdapterContract(
    source_system="super_labelator",
    input_summary=(
        "vision predictions",
        "depth or geometry observations",
        "human review or multi-modal corrections",
    ),
)


def build_kr3_hand_result_from_normalized(
    normalized: dict[str, np.ndarray],
    *,
    source_system: str = "super_labelator",
    timestamp_ns: int = 0,
    track_id: str = "hand_0",
    mesh_model: str = "mock",
    mesh_topology_id: str = "mock_joint_fan_v0",
) -> dict[str, np.ndarray]:
    """Adapt the current normalized NPZ payload into the KR3 hand interface."""

    if source_system not in SOURCE_SYSTEMS:
        raise ValueError(f"source_system must be one of {SOURCE_SYSTEMS}, got {source_system!r}")
    if mesh_model not in HAND_MODEL_TYPES:
        raise ValueError(f"mesh_model must be one of {HAND_MODEL_TYPES}, got {mesh_model!r}")

    frame_index = np.asarray(normalized["frame_index"], dtype=np.int64)
    n = int(frame_index.shape[0])
    hand_side = np.asarray(normalized["hand_side"], dtype=np.int64).reshape(n)
    is_right = np.asarray(normalized["is_right"], dtype=bool).reshape(n)
    joints = np.asarray(normalized["joints_3d_left_m"], dtype=np.float64).reshape(n, 21, 3)
    wrist_pose = np.asarray(normalized["wrist_pose_6d_left_m_rad"], dtype=np.float64).reshape(n, 6)
    valid_joint_mask = np.asarray(normalized["valid_joint_mask"], dtype=bool).reshape(n, 21)
    frame_confidence = np.asarray(normalized["frame_confidence"], dtype=np.float64).reshape(n)
    frame_status = np.asarray(normalized["frame_status"]).astype("<U16").reshape(n)

    angles_20 = np.asarray(normalized.get("hand_angles_20dof_rad", np.zeros((n, 20))), dtype=np.float64).reshape(n, 20)
    hand_angles_22 = np.concatenate([angles_20, np.zeros((n, 2), dtype=np.float64)], axis=1)
    mesh_vertices = _mock_mesh_vertices_from_joints(joints)
    mesh_faces = _mock_mesh_faces()

    provenance = [
        json.dumps(
            {
                "adapter": "build_kr3_hand_result_from_normalized",
                "source_schema": "root_translation_optimized_hands.npz",
                "source_system": source_system,
                "mesh_topology_id": mesh_topology_id,
            },
            ensure_ascii=False,
        )
        for _ in range(n)
    ]

    payload = {
        "schema_version": np.array(HAND_RESULT_SCHEMA_VERSION, dtype=object),
        "source_system": np.full((n,), source_system, dtype="<U32"),
        "frame_index": frame_index,
        "timestamp_ns": np.full((n,), int(timestamp_ns), dtype=np.int64),
        "track_id": np.full((n,), track_id, dtype="<U64"),
        "hand_side": hand_side,
        "is_right": is_right,
        "coordinate_frame": np.full((n,), "rectified_left_camera", dtype="<U32"),
        "wrist_pose_6d_m_rad": wrist_pose,
        "hand_angles_22dof_rad": hand_angles_22,
        "hand_angles_22dof_deg": np.rad2deg(hand_angles_22),
        "hand_angle_names_22dof": UMETRACK_ANGLE_NAMES_22DOF.copy(),
        "hand_angle_convention": np.array("umetrack_compatible_v0", dtype=object),
        "joint_names": JOINT_NAMES.copy(),
        "joints_3d_m": joints,
        "valid_joint_mask": valid_joint_mask,
        "mesh_vertices_m": mesh_vertices,
        "mesh_faces": mesh_faces,
        "mesh_vertex_valid_mask": np.ones(mesh_vertices.shape[:2], dtype=bool),
        "mesh_model": np.full((n,), mesh_model, dtype="<U16"),
        "mesh_topology_id": np.array(mesh_topology_id, dtype=object),
        "frame_confidence": frame_confidence,
        "frame_status": frame_status,
        "provenance_json": np.asarray(provenance, dtype=object),
        "mano_pose_axis_angle": np.zeros((n, 48), dtype=np.float64),
        "mano_shape_betas": np.zeros((n, 10), dtype=np.float64),
        "mano_global_orient": wrist_pose[:, 3:6].copy(),
        "mano_transl_m": wrist_pose[:, 0:3].copy(),
        "umetrack_joint_angles_rad": hand_angles_22.copy(),
    }

    errors = validate_kr3_hand_result(payload)
    if errors:
        raise ValueError("invalid KR3 hand result payload: " + "; ".join(errors))
    return payload


def validate_kr3_hand_result(payload: dict[str, np.ndarray]) -> list[str]:
    """Return shape/field validation errors for a KR3 hand result payload."""

    errors: list[str] = []
    missing = [field for field in KR3_REQUIRED_FIELDS if field not in payload]
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
        return errors

    frame_index = np.asarray(payload["frame_index"])
    if frame_index.ndim != 1:
        return ["frame_index must have shape (N,)"]
    n = int(frame_index.shape[0])
    vertices = np.asarray(payload["mesh_vertices_m"])
    faces = np.asarray(payload["mesh_faces"])
    vertex_count = int(vertices.shape[1]) if vertices.ndim == 3 else -1

    expected_shapes: dict[str, tuple[int, ...]] = {
        "source_system": (n,),
        "timestamp_ns": (n,),
        "track_id": (n,),
        "hand_side": (n,),
        "is_right": (n,),
        "coordinate_frame": (n,),
        "wrist_pose_6d_m_rad": (n, 6),
        "hand_angles_22dof_rad": (n, 22),
        "hand_angles_22dof_deg": (n, 22),
        "hand_angle_names_22dof": (22,),
        "joint_names": (21,),
        "joints_3d_m": (n, 21, 3),
        "valid_joint_mask": (n, 21),
        "mesh_vertices_m": (n, vertex_count, 3),
        "mesh_vertex_valid_mask": (n, vertex_count),
        "mesh_model": (n,),
        "frame_confidence": (n,),
        "frame_status": (n,),
        "provenance_json": (n,),
    }
    for field, expected in expected_shapes.items():
        actual = tuple(np.asarray(payload[field]).shape)
        if actual != expected:
            errors.append(f"{field} expected {expected}, got {actual}")

    if faces.ndim != 2 or faces.shape[1] != 3:
        errors.append(f"mesh_faces expected (F, 3), got {tuple(faces.shape)}")
    if not set(np.asarray(payload["source_system"]).astype(str)).issubset(SOURCE_SYSTEMS):
        errors.append("source_system contains unsupported values")
    if not set(np.asarray(payload["coordinate_frame"]).astype(str)).issubset(COORDINATE_FRAMES):
        errors.append("coordinate_frame contains unsupported values")
    if not set(np.asarray(payload["mesh_model"]).astype(str)).issubset(HAND_MODEL_TYPES):
        errors.append("mesh_model contains unsupported values")
    return errors


def write_kr3_hand_result_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Write a validated KR3 hand-result payload as compressed NPZ."""

    errors = validate_kr3_hand_result(payload)
    if errors:
        raise ValueError("invalid KR3 hand result payload: " + "; ".join(errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def adapter_contracts_json() -> dict[str, Any]:
    """Expose adapter contracts in a JSON-serializable form for docs/tools."""

    contracts = [
        GROUND_TRUTH_SYSTEM_CONTRACT,
        DMA_VISION_CONTRACT,
        SUPER_LABELATOR_CONTRACT,
    ]
    return {
        contract.source_system: {
            "input_summary": list(contract.input_summary),
            "output_fields": list(contract.output_fields),
        }
        for contract in contracts
    }


def _mock_mesh_vertices_from_joints(joints: np.ndarray) -> np.ndarray:
    """Create a small deterministic mesh placeholder from 21 joints."""

    wrist = joints[:, 0:1, :]
    palm_center = joints[:, [0, 5, 9, 13, 17], :].mean(axis=1, keepdims=True)
    return np.concatenate([joints, palm_center, wrist], axis=1)


def _mock_mesh_faces() -> np.ndarray:
    faces = []
    for finger in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)):
        for a, b, c in zip(finger[:-2], finger[1:-1], finger[2:]):
            faces.append((a, b, c))
    faces.extend([(0, 5, 9), (0, 9, 13), (0, 13, 17), (5, 9, 21), (9, 13, 21), (13, 17, 21)])
    return np.asarray(faces, dtype=np.int64)
