#!/usr/bin/env python3
"""Compatibility CLI for the joint-independent surface visual report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hand_recon.visualization.surface_report import generate_surface_visual_report

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generate the RGB-D observed hand-surface report")
    parser.add_argument(
        "--output-html",
        type=Path,
        default=ROOT / "outputs" / "mock_rgbd_demo" / "report" / "index.html",
    )
    parser.add_argument("--demo-dir", type=Path, default=ROOT / "outputs" / "mock_rgbd_demo")
    parser.add_argument("--max-points", type=int, default=3500)
    parser.add_argument("--max-faces", type=int, default=4500)
    return parser.parse_args()


def generate_hand_visual_report(
    *,
    demo_dir: Path = ROOT / "outputs" / "mock_rgbd_demo",
    output_html: Path = ROOT / "outputs" / "mock_rgbd_demo" / "report" / "index.html",
    max_points: int = 3500,
) -> Path:
    """Preserve the public report function while using surface artifacts."""

    return generate_surface_visual_report(
        demo_dir=demo_dir,
        output_html=output_html,
        max_points=max_points,
    )


def main() -> int:
    args = parse_args()
    output_html = generate_surface_visual_report(
        demo_dir=args.demo_dir,
        output_html=args.output_html,
        max_points=args.max_points,
        max_faces=args.max_faces,
    )
    print(json.dumps({"status": "ok", "html": str(output_html)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
