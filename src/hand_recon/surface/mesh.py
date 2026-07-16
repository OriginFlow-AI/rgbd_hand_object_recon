"""Triangle-surface extraction and cleanup for TSDF volumes."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from hand_recon.domain import TriangleMesh, TsdfVolume

_CUBE_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.int64,
)
_TETRAHEDRA = np.array(
    [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]],
    dtype=np.int64,
)
_TETRA_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def extract_surface_mesh(
    volume: TsdfVolume,
    *,
    color_points_m: np.ndarray,
    color_values_rgb: np.ndarray,
    min_weight: float = 1.0,
) -> TriangleMesh:
    """Extract the zero level set using marching tetrahedra."""

    values = np.asarray(volume.values, dtype=np.float64)
    weights = np.asarray(volume.weights, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 2 or weights.shape != values.shape:
        raise ValueError("TSDF values and weights must be matching 3-D grids with every dimension >= 2")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("TSDF values and weights must be finite")
    if min_weight <= 0:
        raise ValueError("min_weight must be greater than zero")

    cells = _cell_origins(values.shape)
    corner_indices = cells[:, None, :] + _CUBE_CORNERS[None, :, :]
    corner_values = values[tuple(corner_indices[..., axis] for axis in range(3))]
    corner_weights = weights[tuple(corner_indices[..., axis] for axis in range(3))]
    corner_points = volume.origin_m + corner_indices.astype(np.float64) * volume.voxel_size_m

    triangles: list[np.ndarray] = []
    for tetra in _TETRAHEDRA:
        tetra_values = corner_values[:, tetra]
        tetra_weights = corner_weights[:, tetra]
        has_crossing = (tetra_values.min(axis=1) < 0.0) & (tetra_values.max(axis=1) >= 0.0)
        supported = np.count_nonzero(tetra_weights >= min_weight, axis=1) >= 3
        for cell_index in np.flatnonzero(has_crossing & supported):
            polygon = _tetra_intersections(corner_points[cell_index, tetra], tetra_values[cell_index])
            triangles.extend(_triangulate_polygon(polygon))

    if not triangles:
        raise ValueError("TSDF contains no supported zero crossing; cannot extract a hand surface")
    raw_faces_as_vertices = np.asarray(triangles, dtype=np.float64)
    vertices, faces = _merge_triangle_vertices(raw_faces_as_vertices, tolerance=volume.voxel_size_m * 1e-5)
    vertices, faces = _remove_degenerate_and_unreferenced(vertices, faces)
    vertices, faces = _keep_largest_face_component(vertices, faces)
    faces = _orient_faces_from_tsdf(vertices, faces, volume)
    normals = _vertex_normals(vertices, faces)
    colors = _nearest_colors(vertices, color_points_m, color_values_rgb)
    return TriangleMesh(
        vertices_m=vertices,
        faces=faces,
        vertex_normals=normals,
        vertex_colors_rgb=colors,
        coordinate_frame="world",
    )


def _cell_origins(shape: tuple[int, ...]) -> np.ndarray:
    axes = [np.arange(size - 1, dtype=np.int64) for size in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _tetra_intersections(points: np.ndarray, values: np.ndarray) -> np.ndarray:
    intersections = []
    for first, second in _TETRA_EDGES:
        value_a = float(values[first])
        value_b = float(values[second])
        if (value_a < 0.0) == (value_b < 0.0):
            continue
        denominator = value_a - value_b
        fraction = 0.5 if abs(denominator) < 1e-12 else value_a / denominator
        intersections.append(points[first] + fraction * (points[second] - points[first]))
    return np.asarray(intersections, dtype=np.float64)


def _triangulate_polygon(polygon: np.ndarray) -> list[np.ndarray]:
    if polygon.shape[0] < 3:
        return []
    if polygon.shape[0] == 3:
        return [polygon]
    center = polygon.mean(axis=0)
    centered = polygon - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_x = vh[0]
    axis_y = vh[1]
    angles = np.arctan2(centered @ axis_y, centered @ axis_x)
    ordered = polygon[np.argsort(angles)]
    return [ordered[[0, index, index + 1]] for index in range(1, ordered.shape[0] - 1)]


def _merge_triangle_vertices(triangles: np.ndarray, *, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    flat = triangles.reshape(-1, 3)
    keys = np.rint(flat / tolerance).astype(np.int64)
    _, unique_indices, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    vertices = flat[unique_indices]
    faces = inverse.reshape(-1, 3).astype(np.int64)
    return vertices, faces


def _remove_degenerate_and_unreferenced(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    faces = faces[distinct]
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    faces = faces[np.linalg.norm(cross, axis=1) > 1e-14]
    canonical = np.sort(faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    faces = faces[np.sort(first)]
    if faces.shape[0] == 0:
        raise ValueError("surface extraction produced only degenerate faces")
    used, remap = np.unique(faces, return_inverse=True)
    return vertices[used], remap.reshape(-1, 3).astype(np.int64)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-15
    normals[valid] /= lengths[valid, None]
    return normals


def _keep_largest_face_component(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)),
        shape=(vertices.shape[0], vertices.shape[0]),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    if component_count <= 1:
        return vertices, faces
    face_labels = labels[faces[:, 0]]
    largest = int(np.argmax(np.bincount(face_labels, minlength=component_count)))
    return _remove_degenerate_and_unreferenced(vertices, faces[face_labels == largest])


def _orient_faces_from_tsdf(vertices: np.ndarray, faces: np.ndarray, volume: TsdfVolume) -> np.ndarray:
    gradients = np.gradient(np.asarray(volume.values, dtype=np.float64), volume.voxel_size_m)
    centers = vertices[faces].mean(axis=1)
    indices = np.rint((centers - volume.origin_m) / volume.voxel_size_m).astype(np.int64)
    indices = np.clip(indices, 0, np.asarray(volume.values.shape, dtype=np.int64) - 1)
    sampled_gradient = np.column_stack(
        [gradient[tuple(indices[:, axis] for axis in range(3))] for gradient in gradients]
    )
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    flip = np.einsum("ij,ij->i", face_normals, sampled_gradient) < 0.0
    oriented = faces.copy()
    oriented[flip, 1], oriented[flip, 2] = faces[flip, 2], faces[flip, 1]
    return oriented


def _nearest_colors(vertices: np.ndarray, points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] == 0:
        return np.full(vertices.shape, 210, dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[0] != points.shape[0] or colors.shape[1] < 3:
        raise ValueError("color_values_rgb must align with color_points_m")
    _, indices = cKDTree(points).query(vertices, k=1, workers=-1)
    return np.clip(colors[indices, :3], 0, 255).astype(np.uint8)
