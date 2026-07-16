#!/usr/bin/env python3
"""Minimal software-integration example for hand_recon."""

from __future__ import annotations

from pathlib import Path

from hand_recon import load_surface_geometry_npz, run_mock_reconstruction

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    output_dir = ROOT / "outputs" / "api_demo_mock_rgbd"
    result = run_mock_reconstruction(
        scene_dir=ROOT / "mock_data" / "rgbd_scene_001",
        output_dir=output_dir,
        hand_side="right",
    )
    geometry = load_surface_geometry_npz(result.output_paths["hand_geometry"])
    if geometry["mesh_faces"].shape[0] == 0:
        raise SystemExit("surface reconstruction returned no mesh faces")

    print(f"status={result.status}")
    print(f"mesh={result.output_paths['hand_surface']}")
    print(f"report={result.output_paths['surface_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
