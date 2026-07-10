from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon import generate_mock_visual_report, load_hand_result_npz, run_mock_reconstruction, validate_hand_result


def test_public_api_runs_mock_pipeline_and_report(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    output_dir = tmp_path / "demo"
    result = run_mock_reconstruction(scene_dir=scene_dir, output_dir=output_dir, hand_side="right")

    assert result.ok
    assert result.output_paths["summary"].exists()
    assert result.output_paths["kr3_hand_result"].exists()
    assert result.quality_report["passed"] is True

    hand_result = load_hand_result_npz(result.output_paths["kr3_hand_result"])
    assert validate_hand_result(hand_result) == []
    assert hand_result["hand_angles_22dof_rad"].shape == (1, 22)
    assert hand_result["joints_3d_m"].shape == (1, 21, 3)
    assert hand_result["mesh_vertices_m"].shape[2] == 3

    report_path = generate_mock_visual_report(
        demo_dir=output_dir,
        output_html=tmp_path / "reports" / "hand_visual.html",
        max_points=300,
    )
    assert report_path.exists()
    assert "<svg" in report_path.read_text(encoding="utf-8")
