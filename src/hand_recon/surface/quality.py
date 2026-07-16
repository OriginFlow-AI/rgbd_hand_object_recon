"""Geometry-based quality gates for observed hand surfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from hand_recon.domain import TriangleMesh, TsdfVolume


def evaluate_surface_quality(
    mesh: TriangleMesh,
    volume: TsdfVolume,
    source_points_m: np.ndarray,
    *,
    input_view_count: int,
) -> dict[str, Any]:
    """Evaluate whether a mesh is supported by the RGB-D observations."""

    vertices = np.asarray(mesh.vertices_m, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    source = np.asarray(source_points_m, dtype=np.float64)
    triangle_cross = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    areas = 0.5 * np.linalg.norm(triangle_cross, axis=1)

    edges = np.sort(
        np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    non_manifold_edges = int(np.count_nonzero(edge_counts > 2))
    unique_edge_count = int(edge_counts.shape[0])

    component_count, labels = _mesh_components(vertices.shape[0], edges)
    face_component = labels[faces[:, 0]]
    component_face_counts = np.bincount(face_component, minlength=component_count)
    largest_component_ratio = float(component_face_counts.max() / faces.shape[0]) if faces.shape[0] else 0.0

    source_distances = cKDTree(vertices).query(source, k=1, workers=-1)[0]
    mesh_distances = cKDTree(source).query(vertices, k=1, workers=-1)[0]
    support = _sample_grid_nearest(volume.observed_view_count, volume, vertices)
    weights = _sample_grid_nearest(volume.weights, volume, vertices)
    multi_view_ratio = float(np.mean(support >= 2)) if support.size else 0.0
    supported_ratio = float(np.mean(weights >= 1)) if weights.size else 0.0

    thresholds = {
        "min_vertex_count": 100,
        "min_face_count": 100,
        "max_source_to_surface_p95_m": float(3.0 * volume.voxel_size_m),
        "min_largest_component_face_ratio": 0.80,
        "min_supported_vertex_ratio": 0.95,
        "max_non_manifold_edge_count": 0,
    }
    metrics = {
        "input_view_count": int(input_view_count),
        "tsdf_grid_shape": list(volume.values.shape),
        "tsdf_observed_voxel_count": int(np.count_nonzero(volume.weights > 0)),
        "mesh_vertex_count": mesh.vertex_count,
        "mesh_face_count": mesh.face_count,
        "surface_area_m2": round(float(areas.sum()), 8),
        "triangle_area_m2_mean": round(float(areas.mean()), 12),
        "component_count": int(component_count),
        "largest_component_face_ratio": round(largest_component_ratio, 6),
        "boundary_edge_count": boundary_edges,
        "boundary_edge_ratio": round(boundary_edges / max(1, unique_edge_count), 6),
        "non_manifold_edge_count": non_manifold_edges,
        "supported_vertex_ratio": round(supported_ratio, 6),
        "multi_view_vertex_ratio": round(multi_view_ratio, 6),
        "source_to_surface_mean_m": round(float(np.mean(source_distances)), 8),
        "source_to_surface_p95_m": round(float(np.percentile(source_distances, 95)), 8),
        "surface_to_source_mean_m": round(float(np.mean(mesh_distances)), 8),
        "surface_to_source_p95_m": round(float(np.percentile(mesh_distances, 95)), 8),
        "bbox_extent_m": [round(float(value), 8) for value in np.ptp(vertices, axis=0)],
    }
    warnings = []
    if mesh.vertex_count < thresholds["min_vertex_count"]:
        warnings.append("mesh_vertex_count_below_threshold")
    if mesh.face_count < thresholds["min_face_count"]:
        warnings.append("mesh_face_count_below_threshold")
    if metrics["source_to_surface_p95_m"] > thresholds["max_source_to_surface_p95_m"]:
        warnings.append("source_to_surface_distance_above_threshold")
    if largest_component_ratio < thresholds["min_largest_component_face_ratio"]:
        warnings.append("surface_is_fragmented")
    if supported_ratio < thresholds["min_supported_vertex_ratio"]:
        warnings.append("surface_has_low_observation_support")
    if non_manifold_edges > thresholds["max_non_manifold_edge_count"]:
        warnings.append("surface_has_non_manifold_edges")
    status = "ok" if not warnings else "partial"
    return {
        "schema_version": "hand_surface_quality_v1",
        "status": status,
        "definition": "observed RGB-D surface; unobserved regions are not inferred",
        "metrics": metrics,
        "thresholds": thresholds,
        "warnings": warnings,
        "passed": status == "ok",
    }


def _mesh_components(vertex_count: int, edges: np.ndarray) -> tuple[int, np.ndarray]:
    data = np.ones(edges.shape[0] * 2, dtype=np.uint8)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count)).tocsr()
    return connected_components(graph, directed=False)


def _sample_grid_nearest(grid: np.ndarray, volume: TsdfVolume, points: np.ndarray) -> np.ndarray:
    indices = np.rint((points - volume.origin_m) / volume.voxel_size_m).astype(np.int64)
    upper = np.asarray(grid.shape, dtype=np.int64) - 1
    indices = np.clip(indices, 0, upper)
    return np.asarray(grid)[tuple(indices[:, axis] for axis in range(3))]
