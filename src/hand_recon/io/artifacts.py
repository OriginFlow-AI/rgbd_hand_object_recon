"""Versioned artifact bundle for surface reconstruction runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from hand_recon.domain import SurfaceRunResult
from hand_recon.icp import write_ascii_ply
from hand_recon.io.geometry import write_surface_geometry_npz, write_triangle_mesh_ply
from hand_recon.io.json_io import read_json_object, write_json
from hand_recon.reconstruction import ReconstructionResult
from hand_recon.rgbd import RgbdScene


def write_surface_artifacts(
    *,
    output_dir: Path,
    scene: RgbdScene,
    cloud: ReconstructionResult,
    result: SurfaceRunResult,
) -> dict[str, Path]:
    """Write the complete geometry, diagnostics, and manifest bundle."""

    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    paths = {
        "surface_quality": root / "quality" / "surface_quality.json",
        "hand_geometry": root / "geometry" / "hand_geometry.npz",
        "hand_fused_colored": root / "geometry" / "hand_fused.ply",
        "hand_surface": root / "geometry" / "hand_surface.ply",
        "surface_report": root / "report" / "index.html",
    }
    paths["surface_report"].unlink(missing_ok=True)
    view_paths: dict[str, Path] = {}
    for stats, points, colors in zip(
        cloud.per_view_stats,
        cloud.per_view_points,
        cloud.per_view_colors_rgb,
        strict=True,
    ):
        camera_id = str(stats["camera_id"])
        path = root / "views" / f"{camera_id}_hand.ply"
        write_ascii_ply(path, points, colors)
        view_paths[f"{camera_id}_hand"] = path

    write_ascii_ply(paths["hand_fused_colored"], cloud.fused_points, cloud.fused_colors_rgb)
    write_triangle_mesh_ply(paths["hand_surface"], result.mesh)
    write_surface_geometry_npz(paths["hand_geometry"], cloud, result.mesh)
    write_json(paths["surface_quality"], result.quality)
    all_paths = {**paths, **view_paths}
    manifest = _build_manifest(root, scene, cloud, result, all_paths)
    write_json(manifest_path, manifest)
    return {"surface_manifest": manifest_path, **all_paths}


def refresh_manifest_checksums(manifest_path: Path) -> None:
    """Refresh hashes after late artifacts such as the HTML report are written."""

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = read_json_object(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"manifest artifacts must be an object: {manifest_path}")
    for artifact in artifacts.values():
        if not isinstance(artifact, dict) or "path" not in artifact:
            raise ValueError(f"invalid artifact entry in {manifest_path}")
        path = _resolve_inside(root, str(artifact["path"]))
        artifact["exists"] = path.is_file()
        artifact["sha256"] = _sha256(path) if path.is_file() else None
        artifact["bytes"] = path.stat().st_size if path.is_file() else 0
    write_json(manifest_path, manifest)


def _build_manifest(
    root: Path,
    scene: RgbdScene,
    cloud: ReconstructionResult,
    result: SurfaceRunResult,
    paths: dict[str, Path],
) -> dict[str, Any]:
    artifacts = {}
    for name, path in paths.items():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes output directory: {resolved}") from exc
        artifacts[name] = {
            "path": relative.as_posix(),
            "exists": resolved.is_file(),
            "sha256": _sha256(resolved) if resolved.is_file() else None,
            "bytes": resolved.stat().st_size if resolved.is_file() else 0,
        }
    return {
        "schema_version": "hand_surface_artifacts_v1",
        "status": result.status,
        "scene_id": scene.scene_id,
        "source_scene_dir": str(scene.scene_dir.resolve()),
        "coordinate_frame": scene.coordinate_frame,
        "position_unit": "meter",
        "purpose": "joint-independent multi-view RGB-D observed hand-surface reconstruction",
        "surface_semantics": "observed_not_completed",
        "uses_joint_localization": False,
        "parameters": result.parameters,
        "counts": {
            "view_count": len(scene.views),
            "raw_point_count": int(cloud.raw_points.shape[0]),
            "fused_point_count": int(cloud.fused_points.shape[0]),
            "mesh_vertex_count": result.mesh.vertex_count,
            "mesh_face_count": result.mesh.face_count,
        },
        "per_view_stats": cloud.per_view_stats,
        "artifacts": artifacts,
        "compatibility_note": "Legacy KR3 joint/angle files are optional side outputs and are not consumed by this bundle.",
    }


def _resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes output directory: {relative}") from exc
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
