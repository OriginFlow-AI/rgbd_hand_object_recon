"""Unified command-line interface for supported project workflows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from hand_recon.config import MockRgbdConfig, load_mock_rgbd_config
from hand_recon.exceptions import HandReconError
from hand_recon.icp import run_synthetic_selftest
from hand_recon.pipelines.mock_rgbd import run_mock_rgbd_pipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rgbd-hand-recon", description="RGB-D hand/object reconstruction tools")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the deterministic mock RGB-D reconstruction pipeline")
    demo.add_argument("--config", type=Path, help="optional JSON configuration file")
    demo.add_argument("--scene-dir", type=Path)
    demo.add_argument("--output-dir", type=Path)
    demo.add_argument("--voxel-size-m", type=float)
    demo.add_argument("--hand-side", choices=["left", "right"])
    demo.add_argument("--overwrite-mock-data", action="store_true", default=None)
    demo.set_defaults(handler=_run_demo)

    verify = subparsers.add_parser("verify", help="run the dependency-light synthetic ICP verification")
    verify.set_defaults(handler=_run_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except (HandReconError, OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


def _run_demo(args: argparse.Namespace) -> int:
    config = load_mock_rgbd_config(args.config) if args.config else MockRgbdConfig()
    overrides = {
        "scene_dir": args.scene_dir,
        "output_dir": args.output_dir,
        "voxel_size_m": args.voxel_size_m,
        "hand_side": args.hand_side,
        "overwrite_mock_data": args.overwrite_mock_data,
    }
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})
    result = run_mock_rgbd_pipeline(
        scene_dir=config.scene_dir,
        output_dir=config.output_dir,
        voxel_size_m=config.voxel_size_m,
        hand_side=config.hand_side,
        overwrite_mock_data=config.overwrite_mock_data,
    )
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


def _run_verify(_args: argparse.Namespace) -> int:
    result = run_synthetic_selftest()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":  # pragma: no cover - console-script convenience
    sys.exit(main())
