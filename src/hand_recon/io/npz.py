"""Safe NPZ loading and atomic writing utilities."""

from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from hand_recon.exceptions import DataValidationError, UnsafeDataError


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load NPZ arrays without permitting Python object deserialization."""

    input_path = Path(path)
    try:
        with np.load(input_path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (ValueError, zipfile.BadZipFile, EOFError) as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise UnsafeDataError(
                f"refusing unsafe NPZ object arrays in {input_path}; regenerate the file with Unicode/numeric dtypes"
            ) from exc
        raise DataValidationError(f"invalid NPZ file {input_path}: {exc}") from exc


def write_npz_arrays(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Atomically write a compressed NPZ containing no object arrays."""

    output_path = Path(path)
    unsafe_fields = sorted(key for key, value in payload.items() if np.asarray(value).dtype.hasobject)
    if unsafe_fields:
        raise UnsafeDataError("NPZ payload contains object arrays: " + ", ".join(unsafe_fields))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(temporary_path, **payload)
        with temporary_path.open("rb") as temporary:
            os.fsync(temporary.fileno())
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
