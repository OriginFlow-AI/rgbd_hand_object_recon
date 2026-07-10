"""Compatibility adapter for unified hand result payloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_recon.interfaces.hand_result import (
    HAND_RESULT_SCHEMA_VERSION,
    KR3_REQUIRED_FIELDS,
    SOURCE_SYSTEMS,
    UMETRACK_ANGLE_NAMES_22DOF,
    build_kr3_hand_result_from_normalized,
    validate_kr3_hand_result,
    write_kr3_hand_result_npz,
)


def build_hand_result_from_normalized(
    normalized: dict[str, np.ndarray],
    *,
    source_system: str = "super_labelator",
    timestamp_ns: int = 0,
    track_id: str = "hand_0",
    mesh_model: str = "mock",
    mesh_topology_id: str = "mock_joint_fan_v0",
) -> dict[str, np.ndarray]:
    """Build the unified hand result payload from a normalized hand NPZ payload."""

    return build_kr3_hand_result_from_normalized(
        normalized,
        source_system=source_system,
        timestamp_ns=timestamp_ns,
        track_id=track_id,
        mesh_model=mesh_model,
        mesh_topology_id=mesh_topology_id,
    )


def validate_hand_result(payload: dict[str, np.ndarray]) -> list[str]:
    """Validate the unified hand result payload."""

    return validate_kr3_hand_result(payload)


def write_hand_result_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Write the unified hand result payload as a compressed NPZ file."""

    write_kr3_hand_result_npz(path, payload)
