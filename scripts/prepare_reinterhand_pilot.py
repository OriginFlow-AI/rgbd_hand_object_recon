#!/usr/bin/env python3
"""Prepare a small Re:InterHand pilot capture.

The official Re:InterHand scripts download full capture-level archives.  This
helper keeps the first pilot controlled: it downloads metadata, selects the
smallest available frame-list capture by default, then downloads only MANO fits
and optional Mugsy camera parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import tarfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
S3_BASE = "https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/ReInterHand"
CAPTURES = [
    "m--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands",
    "m--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands",
    "m--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands",
    "m--20221110--1033--TQH976--pilot--ProjectGoliathScript--Hands--two-hands",
    "m--20221111--0944--JFQ550--pilot--ProjectGoliathScript--Hands--two-hands",
    "m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands",
    "m--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands",
    "m--20230313--1433--TXB805--pilot--ProjectGoliath--Hands--two-hands",
    "m--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands",
    "m--20230317--1433--TRO760--pilot--ProjectGoliath--Hands--two-hands",
]

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "reinterhand")
    parser.add_argument("--capture", default="auto-smallest", choices=["auto-smallest", *CAPTURES])
    parser.add_argument(
        "--skip-metadata", action="store_true", help="Use existing metadata files instead of scanning all captures."
    )
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--download-mano", action="store_true")
    parser.add_argument("--download-mugsy-cam-params", action="store_true")
    parser.add_argument("--extract-mano", action="store_true")
    parser.add_argument(
        "--allow-unverified-extract",
        action="store_true",
        help="Allow extraction when CHECKSUM has no entry. A checksum mismatch is always rejected.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=ROOT / "outputs" / "reinterhand_pilot_summary.json")
    return parser.parse_args()


def url_for(capture: str, relpath: str) -> str:
    return f"{S3_BASE}/{capture}/{relpath}"


def request_with_retries(url: str, headers: dict[str, str] | None = None, timeout: int = 45):
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            req = Request(url, headers=headers or {})
            return urlopen(req, timeout=timeout)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    if last_error is None:  # Defensive: the retry loop always runs at least once.
        raise RuntimeError(f"request failed without a reported error: {url}")
    raise last_error


def remote_size(url: str) -> int | None:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=30) as response:
            value = response.headers.get("Content-Length")
        return int(value) if value else None
    except (HTTPError, URLError, TimeoutError, TypeError, ValueError):
        return None


def download_file(
    url: str,
    output_path: Path,
    force: bool = False,
    trust_existing: bool = False,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if trust_existing and output_path.exists() and output_path.stat().st_size > 0 and not force:
        return {"path": str(output_path), "bytes": output_path.stat().st_size, "status": "already_exists"}
    expected_size = remote_size(url)
    start = output_path.stat().st_size if output_path.exists() and not force else 0
    if force and output_path.exists():
        output_path.unlink()
        start = 0
    if expected_size is not None and output_path.exists() and output_path.stat().st_size == expected_size:
        return {"path": str(output_path), "bytes": expected_size, "status": "already_complete"}
    if expected_size is not None and start > expected_size:
        LOGGER.warning("discarding oversized partial download: %s", output_path)
        start = 0

    headers = {}
    mode = "wb"
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        mode = "ab"

    with request_with_retries(url, headers=headers) as response:
        response_status = getattr(response, "status", None)
        if start > 0 and response_status != 206:
            # Some object stores ignore Range and return the whole body. Appending
            # that response would silently corrupt the archive.
            LOGGER.warning("server ignored Range; restarting download: %s", output_path)
            start = 0
            mode = "wb"
        with output_path.open(mode) as out:
            copied = start
            last_print = time.time()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                copied += len(chunk)
                now = time.time()
                if now - last_print > 5:
                    if expected_size:
                        pct = copied / expected_size * 100.0
                        print(
                            f"  {output_path.name}: {copied / 1024**2:.1f}/{expected_size / 1024**2:.1f} MiB ({pct:.1f}%)"
                        )
                    else:
                        print(f"  {output_path.name}: {copied / 1024**2:.1f} MiB")
                    last_print = now

    final_size = output_path.stat().st_size
    if expected_size is not None and final_size != expected_size:
        raise RuntimeError(f"download size mismatch for {output_path}: expected {expected_size}, got {final_size}")
    return {"path": str(output_path), "bytes": final_size, "status": "downloaded"}


def md5sum(path: Path) -> str:
    # Re:InterHand publishes MD5 checksums, so this is an upstream compatibility
    # check rather than a cryptographic authenticity guarantee.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksum(capture_dir: Path) -> dict[str, str]:
    checksum_path = capture_dir / "CHECKSUM"
    if not checksum_path.exists():
        return {}
    out = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def verify_file(path: Path, expected_md5: str | None) -> dict[str, object]:
    if not expected_md5:
        return {"path": str(path), "verified": None, "reason": "no_checksum"}
    actual = md5sum(path)
    return {"path": str(path), "verified": actual == expected_md5, "expected_md5": expected_md5, "actual_md5": actual}


def download_metadata(data_root: Path, force: bool = False) -> list[dict[str, object]]:
    reports = []
    for capture in CAPTURES:
        capture_dir = data_root / capture
        capture_dir.mkdir(parents=True, exist_ok=True)
        item_report = {"capture": capture, "files": []}
        for relpath in ["CHECKSUM", "frame_list.txt", "frame_list_orig.txt"]:
            output_path = capture_dir / relpath
            try:
                result = download_file(url_for(capture, relpath), output_path, force=force, trust_existing=True)
                item_report["files"].append(result)
            except Exception as exc:
                item_report["files"].append({"path": str(output_path), "status": "error", "error": repr(exc)})
        reports.append(item_report)
    return reports


def frame_count(capture_dir: Path) -> int | None:
    path = capture_dir / "frame_list.txt"
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def choose_capture(data_root: Path, requested: str) -> str:
    if requested != "auto-smallest":
        return requested
    candidates = []
    for capture in CAPTURES:
        count = frame_count(data_root / capture)
        if count is not None:
            candidates.append((count, capture))
    if not candidates:
        raise RuntimeError("no frame_list.txt files are available; run metadata download first")
    return min(candidates)[1]


def extract_tar_gz(path: Path, output_dir: Path) -> dict[str, object]:
    """Extract an archive after rejecting links, devices, and path traversal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.rglob("*"))
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        _validate_tar_members(members, output_dir)
        for member in members:
            destination = output_dir / member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"archive member has no file body: {member.name!r}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    after = set(output_dir.rglob("*"))
    new_files = [p for p in after - before if p.is_file()]
    return {"path": str(path), "output_dir": str(output_dir), "new_file_count": len(new_files)}


def _validate_tar_members(members: list[tarfile.TarInfo], output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in members:
        member_path = Path(member.name)
        if member_path.is_absolute():
            raise ValueError(f"archive contains an absolute path: {member.name!r}")
        destination = (output_root / member_path).resolve()
        if not destination.is_relative_to(output_root):
            raise ValueError(f"archive member escapes the output directory: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"archive member type is not allowed: {member.name!r}")


def summarize_capture(capture_dir: Path) -> dict[str, object]:
    meshes_dir = capture_dir / "mano_fits" / "meshes"
    params_dir = capture_dir / "mano_fits" / "params"
    return {
        "capture": capture_dir.name,
        "frame_count": frame_count(capture_dir),
        "mano_mesh_count": len(list(meshes_dir.glob("*.ply"))) if meshes_dir.exists() else 0,
        "mano_param_count": len(list(params_dir.glob("*.json"))) if params_dir.exists() else 0,
        "has_mugsy_cam_params": (capture_dir / "Mugsy_cameras" / "cam_params.json").exists(),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.extract_mano and not args.download_mano:
        raise SystemExit("--extract-mano requires --download-mano")
    args.data_root.mkdir(parents=True, exist_ok=True)
    metadata_reports = [] if args.skip_metadata else download_metadata(args.data_root, force=args.force)
    capture = choose_capture(args.data_root, args.capture)
    capture_dir = args.data_root / capture
    checksums = read_checksum(capture_dir)
    selected_reports: list[dict[str, object]] = []

    if args.download_mano and not args.metadata_only:
        relpath = "mano_fits/mano_fits.tar.gzaa"
        output_path = capture_dir / relpath
        result = download_file(url_for(capture, relpath), output_path, force=args.force)
        verification = verify_file(output_path, checksums.get(relpath))
        result["verify"] = verification
        selected_reports.append(result)
        if args.extract_mano:
            if verification["verified"] is False:
                raise RuntimeError(f"refusing to extract {output_path}: checksum mismatch")
            if verification["verified"] is None and not args.allow_unverified_extract:
                raise RuntimeError(
                    f"refusing to extract {output_path} without a checksum; pass --allow-unverified-extract to override"
                )
            selected_reports.append(extract_tar_gz(output_path, capture_dir / "mano_fits"))

    if args.download_mugsy_cam_params and not args.metadata_only:
        relpath = "Mugsy_cameras/cam_params.json"
        output_path = capture_dir / relpath
        result = download_file(url_for(capture, relpath), output_path, force=args.force)
        result["verify"] = verify_file(output_path, checksums.get(relpath))
        selected_reports.append(result)

    disk = shutil.disk_usage(args.data_root)
    summary = {
        "status": "ok",
        "data_root": str(args.data_root),
        "selected_capture": capture,
        "selected_capture_summary": summarize_capture(capture_dir),
        "metadata_reports": metadata_reports,
        "selected_reports": selected_reports,
        "disk_free_bytes": disk.free,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "selected_capture": capture, "summary_json": str(args.summary_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
