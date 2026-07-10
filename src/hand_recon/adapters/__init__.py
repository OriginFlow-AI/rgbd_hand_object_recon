"""Adapter layer for external systems and normalized hand result formats."""

from hand_recon.adapters.hand_result import (
    HAND_RESULT_SCHEMA_VERSION,
    KR3_REQUIRED_FIELDS,
    SOURCE_SYSTEMS,
    UMETRACK_ANGLE_NAMES_22DOF,
    build_hand_result_from_normalized,
    validate_hand_result,
    write_hand_result_npz,
)

__all__ = [
    "HAND_RESULT_SCHEMA_VERSION",
    "KR3_REQUIRED_FIELDS",
    "SOURCE_SYSTEMS",
    "UMETRACK_ANGLE_NAMES_22DOF",
    "build_hand_result_from_normalized",
    "validate_hand_result",
    "write_hand_result_npz",
]
