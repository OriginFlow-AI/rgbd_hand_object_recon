"""Safety wrappers for the vendored Re:InterHand download scripts.

The original scripts built shell commands with string concatenation and piped
multipart archives directly to ``tar``. These helpers keep the upstream file
lists while avoiding a shell and validating every archive member before write.
"""

from __future__ import annotations

import subprocess
import tarfile
from shutil import copyfileobj
from pathlib import Path
from typing import BinaryIO

BASE_URL = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand"


def download_file(url: str, output_dir: Path) -> None:
    """Download one file with wget resume support and no shell interpolation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "--continue", url], cwd=output_dir, check=True)


def extract_multipart(output_dir: Path, filename_prefix: str) -> None:
    """Safely stream-concatenated ``tar.gz*`` parts into ``output_dir``."""

    output_dir = output_dir.resolve()
    parts = sorted(path for path in output_dir.glob(f"{filename_prefix}*") if path.is_file())
    if not parts:
        raise FileNotFoundError(f"no archive parts found for {filename_prefix!r} in {output_dir}")
    reader = _MultipartReader(parts)
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:
            for member in archive:
                _validate_member(member, output_dir)
                destination = output_dir / member.name
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"archive member has no file body: {member.name!r}")
                with source, destination.open("wb") as output:
                    copyfileobj(source, output, length=1024 * 1024)
    finally:
        reader.close()


def _validate_member(member: tarfile.TarInfo, output_dir: Path) -> None:
    member_path = Path(member.name)
    destination = (output_dir / member_path).resolve()
    if member_path.is_absolute() or not destination.is_relative_to(output_dir):
        raise ValueError(f"archive member escapes the output directory: {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"archive member type is not allowed: {member.name!r}")


class _MultipartReader:
    """Present a sorted set of archive parts as one binary stream."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = iter(paths)
        self._current: BinaryIO | None = None

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while remaining != 0:
            if self._current is None:
                try:
                    self._current = next(self._paths).open("rb")
                except StopIteration:
                    break
            chunk = self._current.read(-1 if size < 0 else remaining)
            if chunk:
                chunks.append(chunk)
                if size > 0:
                    remaining -= len(chunk)
            else:
                self._current.close()
                self._current = None
        return b"".join(chunks)

    def close(self) -> None:
        if self._current is not None:
            self._current.close()
            self._current = None
