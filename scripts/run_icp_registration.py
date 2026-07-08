#!/usr/bin/env python3
"""Run first-stage rigid ICP registration for hand point clouds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon.icp import (  # noqa: E402
    PointCloud,
    apply_transform,
    icp_point_to_point,
    load_point_cloud,
    random_downsample,
    run_synthetic_selftest,
    voxel_downsample,
    write_ascii_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="Run a synthetic ICP registration self-test.")
    parser.add_argument("--target", type=Path, help="Anchor point cloud (.ply/.npz/.npy).")
    parser.add_argument("--source", action="append", type=Path, default=[], help="Source cloud to align; may repeat.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        help="Alternative to --target/--source: first input is target, remaining inputs are sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "icp_registration")
    parser.add_argument("--npz-points-key", default=None, help="Specific point array key when reading .npz files.")
    parser.add_argument("--voxel-size-m", type=float, default=0.002, help="Voxel downsample size for ICP working clouds.")
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Scale input coordinates before registration, e.g. 0.001 for Re:InterHand MANO meshes in millimeters.",
    )
    parser.add_argument("--max-points", type=int, default=50000, help="Random cap after voxel downsampling for ICP.")
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--distance-threshold-m", type=float, default=0.035)
    parser.add_argument("--trim-fraction", type=float, default=0.9)
    parser.add_argument("--min-pairs", type=int, default=80)
    parser.add_argument(
        "--init-transforms-json",
        type=Path,
        help="Optional JSON mapping source path/name/stem to a 4x4 initial transform.",
    )
    return parser.parse_args()


def load_initial_transforms(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for key, value in data.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError(f"initial transform for {key!r} must be a 4x4 matrix")
        out[key] = arr
    return out


def find_init_transform(source_path: Path, source_cloud: PointCloud, mapping: dict[str, np.ndarray]) -> np.ndarray | None:
    candidates = [
        str(source_path),
        source_path.as_posix(),
        source_path.name,
        source_path.stem,
        source_cloud.name,
    ]
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def prepare_working_points(cloud: PointCloud, voxel_size_m: float, max_points: int, seed: int) -> np.ndarray:
    points, _ = voxel_downsample(cloud.points, voxel_size=voxel_size_m)
    return random_downsample(points, max_points=max_points, seed=seed)


def scale_cloud(cloud: PointCloud, scale: float) -> PointCloud:
    if scale == 1.0:
        return cloud
    return PointCloud(points=cloud.points * scale, colors=cloud.colors, name=cloud.name)


def concatenate_clouds(clouds: list[PointCloud]) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.vstack([cloud.points for cloud in clouds]).astype(np.float64)
    if all(cloud.colors is not None for cloud in clouds):
        colors = np.vstack([cloud.colors for cloud in clouds if cloud.colors is not None]).astype(np.uint8)
        return points, colors
    return points, None


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_registration(args: argparse.Namespace) -> dict[str, Any]:
    if args.inputs:
        if len(args.inputs) < 2:
            raise SystemExit("--inputs needs at least two point clouds")
        target_path = args.inputs[0]
        source_paths = args.inputs[1:]
    else:
        if args.target is None or not args.source:
            raise SystemExit("pass --selftest, or pass --target plus at least one --source")
        target_path = args.target
        source_paths = args.source

    args.output_dir.mkdir(parents=True, exist_ok=True)
    init_transforms = load_initial_transforms(args.init_transforms_json)

    target_cloud = scale_cloud(load_point_cloud(target_path, npz_points_key=args.npz_points_key), args.input_scale)
    target_work = prepare_working_points(target_cloud, args.voxel_size_m, args.max_points, seed=11)
    aligned_clouds = [target_cloud]
    per_source = []

    target_out = args.output_dir / f"target_{target_path.stem}.ply"
    write_ascii_ply(target_out, target_cloud.points, target_cloud.colors)

    for index, source_path in enumerate(source_paths, start=1):
        source_cloud = scale_cloud(load_point_cloud(source_path, npz_points_key=args.npz_points_key), args.input_scale)
        source_work = prepare_working_points(source_cloud, args.voxel_size_m, args.max_points, seed=100 + index)
        init_transform = find_init_transform(source_path, source_cloud, init_transforms)
        result = icp_point_to_point(
            source_work,
            target_work,
            init_transform=init_transform,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            distance_threshold=args.distance_threshold_m,
            trim_fraction=args.trim_fraction,
            min_pairs=args.min_pairs,
        )
        aligned_points = apply_transform(source_cloud.points, result.transform)
        aligned_cloud = PointCloud(points=aligned_points, colors=source_cloud.colors, name=source_cloud.name)
        aligned_clouds.append(aligned_cloud)

        out_path = args.output_dir / f"aligned_{index:02d}_{source_path.stem}.ply"
        write_ascii_ply(out_path, aligned_points, source_cloud.colors)
        per_source.append(
            {
                "source": str(source_path),
                "output_ply": str(out_path),
                "used_initial_transform": init_transform is not None,
                **result.to_json(),
            }
        )

    merged_points, merged_colors = concatenate_clouds(aligned_clouds)
    merged_points, merged_colors = voxel_downsample(merged_points, merged_colors, voxel_size=args.voxel_size_m)
    merged_path = args.output_dir / "merged_aligned_voxel.ply"
    write_ascii_ply(merged_path, merged_points, merged_colors)

    summary = {
        "status": "ok",
        "method": "point_to_point_icp",
        "target": str(target_path),
        "target_output_ply": str(target_out),
        "parameters": {
            "voxel_size_m": args.voxel_size_m,
            "input_scale": args.input_scale,
            "max_points": args.max_points,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
            "distance_threshold_m": args.distance_threshold_m,
            "trim_fraction": args.trim_fraction,
            "min_pairs": args.min_pairs,
        },
        "sources": per_source,
        "merged_output_ply": str(merged_path),
    }
    write_summary(args.output_dir / "icp_summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    if args.selftest:
        result = run_synthetic_selftest()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ok" else 1
    summary = run_registration(args)
    print(json.dumps({"status": "ok", "summary_json": str(args.output_dir / "icp_summary.json")}, indent=2, ensure_ascii=False))
    failed = [item for item in summary["sources"] if item["status"] not in ("converged", "max_iterations")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
