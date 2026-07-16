"""Software integration API for RGB-D hand/object reconstruction."""

__version__ = "0.1.0"

from hand_recon.api import (
    MockRgbdPipelineResult,
    ReinterHandIcpResult,
    build_hand_result_from_normalized,
    generate_mock_visual_report,
    load_hand_result_npz,
    run_mock_reconstruction,
    run_reinterhand_best_data_visualization,
    run_reinterhand_best_right_icp,
    validate_hand_result,
    write_hand_result_npz,
)

__all__ = [
    "__version__",
    "MockRgbdPipelineResult",
    "ReinterHandIcpResult",
    "build_hand_result_from_normalized",
    "generate_mock_visual_report",
    "load_hand_result_npz",
    "run_reinterhand_best_data_visualization",
    "run_reinterhand_best_right_icp",
    "run_mock_reconstruction",
    "validate_hand_result",
    "write_hand_result_npz",
]
