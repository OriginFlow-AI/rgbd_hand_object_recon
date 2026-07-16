"""Stable data contracts shared by reconstruction application layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TriangleMesh:
    """Observed triangle surface in a declared metric coordinate frame."""

    vertices_m: np.ndarray
    faces: np.ndarray
    vertex_normals: np.ndarray
    vertex_colors_rgb: np.ndarray
    coordinate_frame: str = "world"

    def __post_init__(self) -> None:
        validate_triangle_mesh(self)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_m.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True)
class TsdfVolume:
    """Dense, bounded TSDF grid used to extract an observed surface."""

    origin_m: np.ndarray
    voxel_size_m: float
    values: np.ndarray
    weights: np.ndarray
    observed_view_count: np.ndarray


@dataclass(frozen=True)
class SurfaceRunResult:
    """Primary output for joint-independent hand-surface reconstruction."""

    status: str
    mesh: TriangleMesh
    quality: dict[str, Any]
    parameters: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def validate_triangle_mesh(mesh: TriangleMesh) -> None:
    """Raise ``ValueError`` when a triangle-mesh contract is malformed."""

    vertices = np.asarray(mesh.vertices_m)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.vertex_normals)
    colors = np.asarray(mesh.vertex_colors_rgb)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError(f"vertices_m must have shape (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if normals.shape != vertices.shape:
        raise ValueError(f"vertex_normals must match vertices_m, got {normals.shape} and {vertices.shape}")
    if colors.shape != vertices.shape:
        raise ValueError(f"vertex_colors_rgb must match vertices_m, got {colors.shape} and {vertices.shape}")
    if not np.issubdtype(vertices.dtype, np.number) or not np.all(np.isfinite(vertices)):
        raise ValueError("vertices_m must contain finite numeric values")
    if not np.issubdtype(normals.dtype, np.number) or not np.all(np.isfinite(normals)):
        raise ValueError("vertex_normals must contain finite numeric values")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must use an integer dtype")
    if not np.issubdtype(colors.dtype, np.integer):
        raise ValueError("vertex_colors_rgb must use an integer dtype")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertices.shape[0]):
        raise ValueError("faces contain vertex indices outside the mesh")
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError("vertex_colors_rgb values must be in [0, 255]")
    if not str(mesh.coordinate_frame).strip():
        raise ValueError("coordinate_frame must not be empty")
