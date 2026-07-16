"""Stable public API for software integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hand_recon.adapters.hand_result import (
    build_hand_result_from_normalized,
    validate_hand_result,
    write_hand_result_npz,
)
from hand_recon.io.npz import load_npz_arrays
from hand_recon.pipelines.mock_rgbd import MockRgbdPipelineResult, run_mock_rgbd_pipeline
from hand_recon.pipelines.reinterhand import ReinterHandIcpResult, run_reinterhand_best_right_icp
from hand_recon.reports.best_data_visual import generate_best_data_visual_report
from hand_recon.reports.hand_visual import generate_hand_visual_report


def run_mock_reconstruction(
    *,
    scene_dir: Path,
    output_dir: Path,
    voxel_size_m: float = 0.003,
    hand_side: str = "right",
    overwrite_mock_data: bool = False,
) -> MockRgbdPipelineResult:
    """Run the mock RGB-D reconstruction workflow and return structured results."""

    return run_mock_rgbd_pipeline(
        scene_dir=scene_dir,
        output_dir=output_dir,
        voxel_size_m=voxel_size_m,
        hand_side=hand_side,
        overwrite_mock_data=overwrite_mock_data,
    )


def load_hand_result_npz(path: Path) -> dict[str, np.ndarray]:
    """Safely load a unified hand result NPZ into memory.

    Python object arrays are intentionally rejected because they require pickle
    deserialization. Hand-result files emitted by this package use only numeric,
    boolean, and Unicode arrays.
    """

    return load_npz_arrays(path)


def generate_mock_visual_report(
    *,
    demo_dir: Path,
    output_html: Path,
    max_points: int = 2800,
) -> Path:
    """Generate the visual HTML report for a mock reconstruction output directory."""

    return generate_hand_visual_report(demo_dir=demo_dir, output_html=output_html, max_points=max_points)


def run_reinterhand_best_data_visualization(
    *,
    data_root: Path,
    icp_output_dir: Path,
    output_html: Path,
    refresh_icp: bool = True,
) -> Path:
    """Generate the best-data Re:InterHand visual report, optionally refreshing ICP first."""

    if refresh_icp:
        run_reinterhand_best_right_icp(data_root=data_root, output_dir=icp_output_dir)
    return generate_best_data_visual_report(icp_dir=icp_output_dir, output_html=output_html)


__all__ = [
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
