"""Small, dependency-light point-to-point ICP utilities.

The implementation is intentionally generic: it can refine calibrated multi-view
point clouds, align mesh-sampled clouds, or align frame-level fused clouds for a
dynamic sequence.  SciPy's KD-tree is used when available; a chunked NumPy
nearest-neighbor fallback keeps the self-test runnable in minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray | None = None
    name: str = ""


@dataclass
class IcpResult:
    transform: np.ndarray
    iterations: int
    mean_error: float
    rmse: float
    fitness: float
    pair_count: int
    status: str
    history: list[dict[str, float | int]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "iterations": self.iterations,
            "mean_error": self.mean_error,
            "rmse": self.rmse,
            "fitness": self.fitness,
            "pair_count": self.pair_count,
            "transform": self.transform.tolist(),
            "history": self.history,
        }


def validate_points(points: np.ndarray) -> np.ndarray:
    """Return finite XYZ rows from an ``(N, >=3)`` numeric array."""

    valid_points, _ = _coerce_points_with_mask(points)
    return valid_points


def _coerce_points_with_mask(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        point_array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("point array must contain numeric values") from exc
    if point_array.ndim != 2 or point_array.shape[1] < 3:
        raise ValueError(f"point array must have shape (N, >=3), got {point_array.shape}")
    point_array = point_array[:, :3]
    keep = np.all(np.isfinite(point_array), axis=1)
    return point_array[keep], keep


def load_point_cloud(path: Path, npz_points_key: str | None = None) -> PointCloud:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return load_ascii_ply(path)
    if suffix == ".npz":
        return load_npz_cloud(path, npz_points_key=npz_points_key)
    if suffix == ".npy":
        points = validate_points(np.load(path))
        return PointCloud(points=points, name=path.stem)
    raise ValueError(f"unsupported point-cloud format: {path}")


def load_npz_cloud(path: Path, npz_points_key: str | None = None) -> PointCloud:
    with np.load(path, allow_pickle=False) as data:
        if npz_points_key is not None and npz_points_key not in data:
            raise ValueError(f"point array key {npz_points_key!r} was not found in {path}")
        point_keys = (
            [npz_points_key]
            if npz_points_key
            else ["points", "voxel_points", "raw_points", "tsdf_points", "observed_points"]
        )
        points = None
        point_keep = None
        original_point_count = 0
        chosen_key = None
        for key in point_keys:
            if key and key in data:
                raw_points = np.asarray(data[key])
                points, point_keep = _coerce_points_with_mask(raw_points)
                original_point_count = raw_points.shape[0]
                chosen_key = key
                break
        if points is None:
            for key in data.files:
                value = np.asarray(data[key])
                if value.ndim == 2 and value.shape[1] >= 3 and np.issubdtype(value.dtype, np.number):
                    points, point_keep = _coerce_points_with_mask(value)
                    original_point_count = value.shape[0]
                    chosen_key = key
                    break
        if points is None or point_keep is None:
            requested = f" key {npz_points_key!r}" if npz_points_key else ""
            raise ValueError(f"no Nx3 point array found in {path}{requested}")

        colors = None
        color_candidates = []
        if chosen_key:
            color_candidates.extend(
                [
                    chosen_key.replace("points", "colors"),
                    f"{chosen_key}_colors",
                ]
            )
        color_candidates.extend(["colors", "voxel_colors", "raw_colors", "tsdf_colors", "observed_colors"])
        for key in dict.fromkeys(color_candidates):
            if key in data:
                candidate = np.asarray(data[key])
                if candidate.ndim == 2 and candidate.shape[0] >= original_point_count and candidate.shape[1] >= 3:
                    colors = np.clip(candidate[:original_point_count, :3][point_keep], 0, 255).astype(np.uint8)
                    break
    return PointCloud(points=points, colors=colors, name=path.stem)


def load_ascii_ply(path: Path) -> PointCloud:
    path = Path(path)
    with path.open("rb") as f:
        header: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY file has no end_header: {path}")
            text = line.decode("ascii", errors="ignore").rstrip("\n")
            header.append(text)
            if text.strip() == "end_header":
                break
        body_offset = f.tell()

    format_line = next((line for line in header if line.startswith("format ")), None)
    if format_line is None:
        raise ValueError(f"PLY file has no format line: {path}")
    file_format = format_line.split()[1]

    vertex_count = None
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header:
        parts = line.split()
        if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and len(parts) >= 3 and parts[0] == "property" and parts[1] != "list":
            vertex_properties.append((parts[1], parts[-1]))
    if vertex_count is None:
        raise ValueError(f"PLY file has no vertex element: {path}")

    property_names = [name for _, name in vertex_properties]
    if file_format == "ascii":
        with path.open("r", encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            rows = []
            for _ in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                rows.append(line.split())
        if len(rows) != vertex_count:
            raise ValueError(f"PLY vertex count mismatch in {path}: expected {vertex_count}, got {len(rows)}")
        arr = np.asarray(rows, dtype=np.float64) if rows else np.empty((0, len(vertex_properties)), dtype=np.float64)
    elif file_format == "binary_little_endian":
        dtype = np.dtype([(name, _ply_numpy_dtype(prop_type)) for prop_type, name in vertex_properties])
        with path.open("rb") as f:
            f.seek(body_offset)
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
        arr = np.column_stack([data[name].astype(np.float64) for name in property_names])
    else:
        raise ValueError(f"unsupported PLY format {file_format!r}: {path}")

    prop_to_idx = {name: idx for idx, name in enumerate(property_names)}
    try:
        points = np.column_stack([arr[:, prop_to_idx["x"]], arr[:, prop_to_idx["y"]], arr[:, prop_to_idx["z"]]])
    except KeyError as exc:
        raise ValueError(f"PLY file lacks x/y/z properties: {path}") from exc

    points, point_keep = _coerce_points_with_mask(points)
    colors = None
    if {"red", "green", "blue"}.issubset(prop_to_idx):
        colors = np.column_stack(
            [arr[:, prop_to_idx["red"]], arr[:, prop_to_idx["green"]], arr[:, prop_to_idx["blue"]]]
        )
        colors = np.clip(colors, 0, 255).astype(np.uint8)[point_keep]
    return PointCloud(points=points, colors=colors, name=Path(path).stem)


def _ply_numpy_dtype(prop_type: str) -> str:
    mapping = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    try:
        return mapping[prop_type]
    except KeyError as exc:
        raise ValueError(f"unsupported PLY vertex property type: {prop_type}") from exc


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_count = np.asarray(points).shape[0] if np.asarray(points).ndim >= 1 else 0
    points, point_keep = _coerce_points_with_mask(points)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] < 3:
            raise ValueError(f"colors must have shape (N, >=3), got {colors.shape}")
        if colors.shape[0] == original_count:
            colors = colors[point_keep]
        elif colors.shape[0] != points.shape[0]:
            raise ValueError("colors must have the same row count as points")
        colors = np.clip(colors[:, :3], 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for point in points:
                f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")
        else:
            for point, color in zip(points, colors, strict=True):
                f.write(
                    f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {int(color[0])} {int(color[1])} {int(color[2])}\n"
                )


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = validate_points(points)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError("transform must contain only finite values")
    return points @ transform[:3, :3].T + transform[:3, 3]


def best_fit_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = validate_points(source)
    target = validate_points(target)
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same number of correspondences")
    if source.shape[0] < 3:
        raise ValueError("at least 3 point pairs are required")
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    cov = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(cov)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = tgt_centroid - rotation @ src_centroid
    return make_transform(rotation, translation)


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    voxel_size: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    original_count = np.asarray(points).shape[0] if np.asarray(points).ndim >= 1 else 0
    points, point_keep = _coerce_points_with_mask(points)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] < 3:
            raise ValueError(f"colors must have shape (N, >=3), got {colors.shape}")
        if colors.shape[0] == original_count:
            colors = colors[point_keep]
        elif colors.shape[0] != points.shape[0]:
            raise ValueError("colors must have the same row count as points")
    if voxel_size is None or voxel_size <= 0 or points.shape[0] == 0:
        return points, colors
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    point_sum = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(point_sum, inverse, points)
    out_points = point_sum / counts[:, None]
    out_colors = None
    if colors is not None:
        color_sum = np.zeros((counts.shape[0], 3), dtype=np.float64)
        np.add.at(color_sum, inverse, np.asarray(colors)[:, :3].astype(np.float64))
        out_colors = np.clip(color_sum / counts[:, None], 0, 255).astype(np.uint8)
    return out_points, out_colors


def random_downsample(points: np.ndarray, max_points: int, seed: int = 20260702) -> np.ndarray:
    points = validate_points(points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def prepare_working_points(cloud: PointCloud, voxel_size: float, max_points: int, seed: int) -> np.ndarray:
    """Create the deterministic reduced point set used by ICP."""

    points, _ = voxel_downsample(cloud.points, voxel_size=voxel_size)
    return random_downsample(points, max_points=max_points, seed=seed)


def scale_point_cloud(cloud: PointCloud, scale: float) -> PointCloud:
    """Return a point cloud converted by a positive unit scale."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"point-cloud scale must be a positive finite value, got {scale}")
    points, keep = _coerce_points_with_mask(cloud.points)
    colors = None
    if cloud.colors is not None:
        color_array = np.asarray(cloud.colors)
        if color_array.ndim != 2 or color_array.shape[0] != keep.shape[0] or color_array.shape[1] < 3:
            raise ValueError(f"colors for cloud {cloud.name!r} do not match its points")
        colors = color_array[:, :3][keep]
    return PointCloud(points=points * scale, colors=colors, name=cloud.name)


def concatenate_point_clouds(clouds: list[PointCloud]) -> tuple[np.ndarray, np.ndarray | None]:
    """Concatenate clouds and preserve colors only when every cloud has them."""

    if not clouds:
        return np.zeros((0, 3), dtype=np.float64), None
    point_arrays: list[np.ndarray] = []
    keep_masks: list[np.ndarray] = []
    for cloud in clouds:
        points, keep = _coerce_points_with_mask(cloud.points)
        point_arrays.append(points)
        keep_masks.append(keep)
    points = np.vstack(point_arrays).astype(np.float64)
    if all(cloud.colors is not None for cloud in clouds):
        color_arrays = []
        for cloud, keep in zip(clouds, keep_masks, strict=True):
            colors = np.asarray(cloud.colors)
            if colors.ndim != 2 or colors.shape[0] != cloud.points.shape[0] or colors.shape[1] < 3:
                raise ValueError(f"colors for cloud {cloud.name!r} do not match its points")
            color_arrays.append(colors[:, :3][keep])
        return points, np.clip(np.vstack(color_arrays), 0, 255).astype(np.uint8)
    return points, None


def nearest_neighbors(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = validate_points(source)
    target = validate_points(target)
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except ImportError:
        chunk = 4096
        all_distances = np.empty((source.shape[0],), dtype=np.float64)
        all_indices = np.empty((source.shape[0],), dtype=np.int64)
        target_sq = np.sum(target * target, axis=1)
        for start in range(0, source.shape[0], chunk):
            stop = min(start + chunk, source.shape[0])
            src = source[start:stop]
            dist_sq = np.sum(src * src, axis=1, keepdims=True) + target_sq[None, :] - 2.0 * src @ target.T
            indices = np.argmin(dist_sq, axis=1)
            all_indices[start:stop] = indices
            all_distances[start:stop] = np.sqrt(np.maximum(dist_sq[np.arange(stop - start), indices], 0.0))
        return all_distances, all_indices
    distances, indices = cKDTree(target).query(source, k=1, workers=-1)
    return distances.astype(np.float64), indices.astype(np.int64)


def icp_point_to_point(
    source_points: np.ndarray,
    target_points: np.ndarray,
    init_transform: np.ndarray | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    distance_threshold: float | None = None,
    trim_fraction: float = 0.9,
    min_pairs: int = 30,
) -> IcpResult:
    source_points = validate_points(source_points)
    target_points = validate_points(target_points)
    if min_pairs < 3:
        raise ValueError("min_pairs must be at least 3")
    if source_points.shape[0] < min_pairs or target_points.shape[0] < min_pairs:
        raise ValueError("source and target need enough points for ICP")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if distance_threshold is not None and distance_threshold <= 0:
        raise ValueError("distance_threshold must be greater than zero when provided")
    if not (0.0 < trim_fraction <= 1.0):
        raise ValueError("trim_fraction must be in (0, 1]")

    transform = np.eye(4, dtype=np.float64) if init_transform is None else np.asarray(init_transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("init_transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("init_transform must have a homogeneous [0, 0, 0, 1] last row")
    transformed = apply_transform(source_points, transform)
    previous_error = np.inf
    history: list[dict[str, float | int]] = []
    status = "max_iterations"
    pair_count = 0

    for iteration in range(1, max_iterations + 1):
        distances, indices = nearest_neighbors(transformed, target_points)
        keep = np.ones((distances.shape[0],), dtype=bool)
        if distance_threshold is not None and distance_threshold > 0:
            keep &= distances <= float(distance_threshold)
        if trim_fraction < 1.0 and np.any(keep):
            kept_distances = distances[keep]
            cutoff_index = max(min_pairs, int(np.ceil(kept_distances.shape[0] * trim_fraction)))
            cutoff_index = min(cutoff_index, kept_distances.shape[0])
            cutoff = np.partition(kept_distances, cutoff_index - 1)[cutoff_index - 1]
            keep &= distances <= cutoff

        pair_count = int(np.count_nonzero(keep))
        if pair_count < min_pairs:
            status = "not_enough_pairs"
            break

        mean_error = float(np.mean(distances[keep]))
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        history.append({"iteration": iteration, "pair_count": pair_count, "mean_error": mean_error, "rmse": rmse})
        delta = best_fit_transform(transformed[keep], target_points[indices[keep]])
        transform = delta @ transform
        transformed = apply_transform(source_points, transform)

        if abs(previous_error - mean_error) < tolerance:
            status = "converged"
            break
        previous_error = mean_error

    final_distances, _ = nearest_neighbors(transformed, target_points)
    if distance_threshold is not None and distance_threshold > 0:
        final_keep = final_distances <= float(distance_threshold)
    else:
        final_keep = np.ones((final_distances.shape[0],), dtype=bool)
    if trim_fraction < 1.0 and np.any(final_keep):
        kept = final_distances[final_keep]
        cutoff_index = max(min_pairs, int(np.ceil(kept.shape[0] * trim_fraction)))
        cutoff_index = min(cutoff_index, kept.shape[0])
        cutoff = np.partition(kept, cutoff_index - 1)[cutoff_index - 1]
        final_keep &= final_distances <= cutoff
    final_pair_count = int(np.count_nonzero(final_keep))
    if final_pair_count:
        mean_error = float(np.mean(final_distances[final_keep]))
        rmse = float(np.sqrt(np.mean(final_distances[final_keep] ** 2)))
    else:
        mean_error = float("inf")
        rmse = float("inf")
    fitness = final_pair_count / max(1, source_points.shape[0])
    return IcpResult(
        transform=transform,
        iterations=len(history),
        mean_error=mean_error,
        rmse=rmse,
        fitness=float(fitness),
        pair_count=final_pair_count,
        status=status,
        history=history,
    )


def rotation_matrix_from_euler(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx = np.sin(rx), np.cos(rx)
    sy, cy = np.sin(ry), np.cos(ry)
    sz, cz = np.sin(rz), np.cos(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def rotation_angle(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    value = (np.trace(rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(value, -1.0, 1.0)))


def make_synthetic_hand_points(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = []

    palm_count = 900
    phi = rng.uniform(0, 2 * np.pi, palm_count)
    costheta = rng.uniform(-1, 1, palm_count)
    theta = np.arccos(costheta)
    palm = np.column_stack(
        [
            0.035 * np.sin(theta) * np.cos(phi),
            0.043 * np.sin(theta) * np.sin(phi),
            0.016 * np.cos(theta),
        ]
    )
    palm[:, 1] -= 0.012
    palm[:, 0] += 0.004 * np.sin(3 * phi)
    parts.append(palm)

    bases = np.array([-0.028, -0.012, 0.004, 0.019, 0.032], dtype=np.float64)
    lengths = np.array([0.052, 0.074, 0.082, 0.073, 0.058], dtype=np.float64)
    radii = np.array([0.0065, 0.0068, 0.0072, 0.0065, 0.0058], dtype=np.float64)
    for finger_id, (base_x, length, radius) in enumerate(zip(bases, lengths, radii, strict=True)):
        n = 420
        t = rng.uniform(0, 1, n)
        a = rng.uniform(0, 2 * np.pi, n)
        bend = (finger_id - 2) * 0.003 * t
        x = base_x + radius * np.cos(a) + bend
        y = 0.018 + length * t
        z = radius * np.sin(a) + 0.004 * np.sin(np.pi * t) - 0.003 * max(finger_id - 2, 0)
        parts.append(np.column_stack([x, y, z]))

    thumb_count = 420
    t = rng.uniform(0, 1, thumb_count)
    a = rng.uniform(0, 2 * np.pi, thumb_count)
    x = -0.034 - 0.045 * t + 0.007 * np.cos(a)
    y = -0.006 + 0.035 * t + 0.007 * np.sin(a)
    z = -0.002 + 0.004 * np.sin(np.pi * t)
    parts.append(np.column_stack([x, y, z]))

    points = np.vstack(parts)
    points += rng.normal(0, 0.0006, points.shape)
    return points.astype(np.float64)


def run_synthetic_selftest() -> dict[str, Any]:
    base = make_synthetic_hand_points()
    rng = np.random.default_rng(20260702)
    reports = []
    for idx in range(3):
        angles = rng.uniform(-0.16, 0.16, size=3)
        translation = rng.uniform(-0.018, 0.018, size=3)
        true_transform = make_transform(rotation_matrix_from_euler(*angles), translation)
        source = apply_transform(base, true_transform)
        source += rng.normal(0, 0.0004, source.shape)
        source = source[rng.choice(source.shape[0], size=int(source.shape[0] * 0.88), replace=False)]
        result = icp_point_to_point(
            source,
            base,
            max_iterations=80,
            tolerance=1e-8,
            distance_threshold=0.04,
            trim_fraction=0.92,
            min_pairs=200,
        )
        residual_transform = result.transform @ true_transform
        rot_err = rotation_angle(residual_transform[:3, :3])
        trans_err = float(np.linalg.norm(residual_transform[:3, 3]))
        ok = bool(result.mean_error < 0.0045 and rot_err < 0.035 and trans_err < 0.004)
        reports.append(
            {
                "case": idx,
                "ok": ok,
                "mean_error_m": result.mean_error,
                "rmse_m": result.rmse,
                "rotation_error_rad": rot_err,
                "translation_error_m": trans_err,
                "iterations": result.iterations,
                "status": result.status,
            }
        )
    return {"status": "ok" if all(item["ok"] for item in reports) else "failed", "cases": reports}
