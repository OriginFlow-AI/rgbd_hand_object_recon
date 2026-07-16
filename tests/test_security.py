from __future__ import annotations

import importlib.util
import io
import tarfile
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from hand_recon import load_hand_result_npz, run_mock_reconstruction
from hand_recon.exceptions import DataValidationError, UnsafeDataError

ROOT = Path(__file__).resolve().parents[1]


def _load_prepare_script() -> ModuleType:
    path = ROOT / "scripts" / "prepare_reinterhand_pilot.py"
    spec = importlib.util.spec_from_file_location("prepare_reinterhand_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_download_utils() -> ModuleType:
    path = ROOT / "third_party" / "reinterhand_download" / "download_utils.py"
    spec = importlib.util.spec_from_file_location("reinterhand_download_utils", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_loader_rejects_pickle_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, payload=np.array({"unexpected": "object"}, dtype=object))

    with pytest.raises(UnsafeDataError, match="object arrays"):
        load_hand_result_npz(path)


def test_public_loader_reports_corrupt_npz(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a zip archive")

    with pytest.raises(DataValidationError, match="invalid NPZ file"):
        load_hand_result_npz(path)


def test_mock_outputs_are_pickle_free_and_have_honest_provenance(tmp_path: Path) -> None:
    result = run_mock_reconstruction(scene_dir=tmp_path / "scene", output_dir=tmp_path / "output")

    hand_result = load_hand_result_npz(result.output_paths["kr3_hand_result"])
    assert hand_result["source_system"].tolist() == ["synthetic_mock"]
    assert hand_result["timestamp_ns"].tolist()[0] > 0
    assert all(not value.dtype.hasobject for value in hand_result.values())

    with np.load(result.output_paths["root_translation_optimized_hands"], allow_pickle=False) as normalized:
        assert all(not normalized[key].dtype.hasobject for key in normalized.files)


def test_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        content = b"not allowed"
        member = tarfile.TarInfo("../escaped.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))

    module = _load_prepare_script()
    with pytest.raises(ValueError, match="escapes the output directory"):
        module.extract_tar_gz(archive_path, tmp_path / "extract")
    assert not (tmp_path / "escaped.txt").exists()


def test_vendored_multipart_extraction_streams_safe_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "payload.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        content = b"safe"
        member = tarfile.TarInfo("nested/result.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    archive_bytes = archive_path.read_bytes()
    archive_path.unlink()
    midpoint = len(archive_bytes) // 2
    (tmp_path / "payload.tar.gzaa").write_bytes(archive_bytes[:midpoint])
    (tmp_path / "payload.tar.gzab").write_bytes(archive_bytes[midpoint:])

    module = _load_download_utils()
    module.extract_multipart(tmp_path, "payload.tar.gz")

    assert (tmp_path / "nested" / "result.txt").read_bytes() == b"safe"


def test_download_restart_when_server_ignores_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_prepare_script()
    output_path = tmp_path / "archive.bin"
    output_path.write_bytes(b"old")

    class Response(io.BytesIO):
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

    monkeypatch.setattr(module, "remote_size", lambda _url: 6)
    monkeypatch.setattr(module, "request_with_retries", lambda _url, headers: Response(b"new123"))

    report = module.download_file("https://example.invalid/archive", output_path)

    assert report["status"] == "downloaded"
    assert output_path.read_bytes() == b"new123"
