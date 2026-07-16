"""Application use case for joint-independent RGB-D hand-surface recovery."""

from __future__ import annotations

from hand_recon.config import HandSurfaceConfig
from hand_recon.domain import SurfaceRunResult
from hand_recon.fusion.tsdf import build_masked_tsdf
from hand_recon.reconstruction import ReconstructionResult
from hand_recon.rgbd import RgbdScene
from hand_recon.surface.mesh import extract_surface_mesh
from hand_recon.surface.quality import evaluate_surface_quality


def reconstruct_hand_surface(
    scene: RgbdScene,
    hand_cloud: ReconstructionResult,
    *,
    hand_label: int,
    config: HandSurfaceConfig | None = None,
) -> SurfaceRunResult:
    """Recover an observed surface directly from masked multi-view depth."""

    config = config or HandSurfaceConfig()
    volume = build_masked_tsdf(
        scene,
        hand_cloud.fused_points,
        label=hand_label,
        voxel_size_m=config.voxel_size_m,
        truncation_m=config.truncation_m,
        padding_m=config.padding_m,
        max_voxel_count=config.max_voxel_count,
    )
    mesh = extract_surface_mesh(
        volume,
        color_points_m=hand_cloud.fused_points,
        color_values_rgb=hand_cloud.fused_colors_rgb,
        min_weight=config.min_weight,
    )
    quality = evaluate_surface_quality(
        mesh,
        volume,
        hand_cloud.fused_points,
        input_view_count=len(scene.views),
    )
    parameters = {
        "backend": "projective_tsdf_marching_tetrahedra",
        "voxel_size_m": config.voxel_size_m,
        "truncation_m": config.truncation_m,
        "padding_m": config.padding_m,
        "min_weight": config.min_weight,
        "max_voxel_count": config.max_voxel_count,
        "uses_joint_localization": False,
        "surface_semantics": "observed_not_completed",
    }
    return SurfaceRunResult(status=str(quality["status"]), mesh=mesh, quality=quality, parameters=parameters)
