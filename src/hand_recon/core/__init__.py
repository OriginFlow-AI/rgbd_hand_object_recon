"""Core geometry and reconstruction primitives.

These exports are stable enough for internal application integration. Higher
level applications should prefer `hand_recon.api` unless they need low-level
geometry control.
"""

from hand_recon.icp import (
    IcpResult,
    PointCloud,
    apply_transform,
    icp_point_to_point,
    load_point_cloud,
    random_downsample,
    validate_points,
    voxel_downsample,
    write_ascii_ply,
)
from hand_recon.reconstruction import ReconstructionResult, reconstruct_multiview_pointcloud
from hand_recon.rgbd import CameraModel, RgbdScene, RgbdView, backproject_depth_to_points, load_mock_rgbd_scene

__all__ = [
    "IcpResult",
    "PointCloud",
    "ReconstructionResult",
    "CameraModel",
    "RgbdScene",
    "RgbdView",
    "apply_transform",
    "backproject_depth_to_points",
    "icp_point_to_point",
    "load_mock_rgbd_scene",
    "load_point_cloud",
    "random_downsample",
    "reconstruct_multiview_pointcloud",
    "validate_points",
    "voxel_downsample",
    "write_ascii_ply",
]
