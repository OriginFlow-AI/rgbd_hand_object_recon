"""Interface contracts reserved for downstream hand result integrations."""

from hand_recon.interfaces.hand_result import (
    HAND_RESULT_SCHEMA_VERSION,
    KR3_REQUIRED_FIELDS,
    SOURCE_SYSTEMS,
    UMETRACK_ANGLE_NAMES_22DOF,
    build_kr3_hand_result_from_normalized,
    validate_kr3_hand_result,
    write_kr3_hand_result_npz,
)

__all__ = [
    "HAND_RESULT_SCHEMA_VERSION",
    "KR3_REQUIRED_FIELDS",
    "SOURCE_SYSTEMS",
    "UMETRACK_ANGLE_NAMES_22DOF",
    "build_kr3_hand_result_from_normalized",
    "validate_kr3_hand_result",
    "write_kr3_hand_result_npz",
]
