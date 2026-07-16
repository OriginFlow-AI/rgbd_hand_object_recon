"""Geometry artifact serialization with explicit metric contracts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from hand_recon.domain import TriangleMesh
from hand_recon.io.npz import write_npz_arrays
from hand_recon.reconstruction import ReconstructionResult


def write_triangle_mesh_ply(path: Path, mesh: TriangleMesh) -> None:
    """Atomically write an ASCII PLY with vertices, normals, colors, and faces."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write("comment coordinate_frame world\ncomment position_unit meter\n")
            stream.write(f"element vertex {mesh.vertex_count}\n")
            for name in ("x", "y", "z", "nx", "ny", "nz"):
                stream.write(f"property float {name}\n")
            for name in ("red", "green", "blue"):
                stream.write(f"property uchar {name}\n")
            stream.write(f"element face {mesh.face_count}\n")
            stream.write("property list uchar int vertex_indices\nend_header\n")
            for vertex, normal, color in zip(
                mesh.vertices_m,
                mesh.vertex_normals,
                mesh.vertex_colors_rgb,
                strict=True,
            ):
                stream.write(
                    f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                    f"{normal[0]:.8f} {normal[1]:.8f} {normal[2]:.8f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )
            for face in mesh.faces:
                stream.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_surface_geometry_npz(
    path: Path,
    cloud: ReconstructionResult,
    mesh: TriangleMesh,
) -> None:
    """Write the complete joint-independent geometry bundle."""

    write_npz_arrays(
        path,
        {
            "schema_version": np.array("hand_surface_geometry_v1", dtype=np.str_),
            "coordinate_frame": np.array(mesh.coordinate_frame, dtype=np.str_),
            "position_unit": np.array("meter", dtype=np.str_),
            "raw_points_m": np.asarray(cloud.raw_points, dtype=np.float64),
            "raw_colors_rgb": np.asarray(cloud.raw_colors_rgb, dtype=np.uint8),
            "raw_view_indices": np.asarray(cloud.raw_view_indices, dtype=np.int32),
            "fused_points_m": np.asarray(cloud.fused_points, dtype=np.float64),
            "fused_colors_rgb": np.asarray(cloud.fused_colors_rgb, dtype=np.uint8),
            "mesh_vertices_m": np.asarray(mesh.vertices_m, dtype=np.float64),
            "mesh_faces": np.asarray(mesh.faces, dtype=np.int64),
            "mesh_vertex_normals": np.asarray(mesh.vertex_normals, dtype=np.float64),
            "mesh_vertex_colors_rgb": np.asarray(mesh.vertex_colors_rgb, dtype=np.uint8),
        },
    )
