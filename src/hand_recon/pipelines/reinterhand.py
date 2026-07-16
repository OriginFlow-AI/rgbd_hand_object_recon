"""Re:InterHand pilot pipelines for best-data visualization."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hand_recon.icp import (
    PointCloud,
    apply_transform,
    concatenate_point_clouds,
    icp_point_to_point,
    load_point_cloud,
    prepare_working_points,
    scale_point_cloud,
    voxel_downsample,
    write_ascii_ply,
)
from hand_recon.io.json_io import write_json

DEFAULT_REINTERHAND_CAPTURE = "m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands"
DEFAULT_BEST_RIGHT_FRAMES = ("100001", "100004", "100007", "100010", "100013")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReinterHandIcpResult:
    status: str
    summary_path: Path
    output_dir: Path
    summary: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def run_reinterhand_best_right_icp(
    *,
    data_root: Path,
    output_dir: Path,
    capture: str = DEFAULT_REINTERHAND_CAPTURE,
    frame_ids: tuple[str, ...] = DEFAULT_BEST_RIGHT_FRAMES,
    hand_side: str = "right",
    input_scale: float = 0.001,
    voxel_size_m: float = 0.001,
    max_points: int = 50000,
    max_iterations: int = 80,
    tolerance: float = 1e-7,
    distance_threshold_m: float = 0.03,
    trim_fraction: float = 0.9,
    min_pairs: int = 100,
) -> ReinterHandIcpResult:
    """Align a short Re:InterHand MANO mesh sequence into one shared frame."""

    if len(frame_ids) < 2:
        raise ValueError("frame_ids must include one target frame and at least one source frame")
    if hand_side not in {"left", "right"}:
        raise ValueError("hand_side must be 'left' or 'right'")
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be a positive finite value")
    if max_points <= 0:
        raise ValueError("max_points must be greater than zero")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = Path(data_root).resolve() / capture / "mano_fits" / "meshes"
    input_paths = [mesh_dir / f"{frame_id}_{hand_side}.ply" for frame_id in frame_ids]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Re:InterHand mesh files: " + ", ".join(str(path) for path in missing))
    LOGGER.info("starting Re:InterHand ICP: target=%s sources=%d", input_paths[0], len(input_paths) - 1)

    target_path = input_paths[0]
    source_paths = input_paths[1:]
    target_cloud = scale_point_cloud(load_point_cloud(target_path), input_scale)
    target_work = prepare_working_points(target_cloud, voxel_size_m, max_points, seed=11)
    aligned_clouds = [target_cloud]
    per_source = []

    target_out = output_dir / f"target_{target_path.stem}.ply"
    write_ascii_ply(target_out, target_cloud.points, target_cloud.colors)

    for index, source_path in enumerate(source_paths, start=1):
        source_cloud = scale_point_cloud(load_point_cloud(source_path), input_scale)
        source_work = prepare_working_points(source_cloud, voxel_size_m, max_points, seed=100 + index)
        result = icp_point_to_point(
            source_work,
            target_work,
            max_iterations=max_iterations,
            tolerance=tolerance,
            distance_threshold=distance_threshold_m,
            trim_fraction=trim_fraction,
            min_pairs=min_pairs,
        )
        aligned_points = apply_transform(source_cloud.points, result.transform)
        aligned_cloud = PointCloud(points=aligned_points, colors=source_cloud.colors, name=source_cloud.name)
        aligned_clouds.append(aligned_cloud)

        out_path = output_dir / f"aligned_{index:02d}_{source_path.stem}.ply"
        write_ascii_ply(out_path, aligned_points, source_cloud.colors)
        per_source.append(
            {
                "source": str(source_path),
                "output_ply": str(out_path),
                "used_initial_transform": False,
                **result.to_json(),
            }
        )

    merged_points, merged_colors = concatenate_point_clouds(aligned_clouds)
    merged_points, merged_colors = voxel_downsample(merged_points, merged_colors, voxel_size=voxel_size_m)
    merged_path = output_dir / "merged_aligned_voxel.ply"
    write_ascii_ply(merged_path, merged_points, merged_colors)

    failed_sources = [item for item in per_source if item["status"] not in {"converged", "max_iterations"}]
    status = "failed" if failed_sources else "ok"
    summary = {
        "status": status,
        "method": "point_to_point_icp",
        "target": str(target_path),
        "target_output_ply": str(target_out),
        "parameters": {
            "voxel_size_m": voxel_size_m,
            "input_scale": input_scale,
            "max_points": max_points,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "distance_threshold_m": distance_threshold_m,
            "trim_fraction": trim_fraction,
            "min_pairs": min_pairs,
        },
        "sources": per_source,
        "merged_output_ply": str(merged_path),
    }
    summary_path = output_dir / "icp_summary.json"
    write_json(summary_path, summary)
    LOGGER.info("Re:InterHand ICP finished: status=%s merged=%s", status, merged_path)
    return ReinterHandIcpResult(status=status, summary_path=summary_path, output_dir=output_dir, summary=summary)
