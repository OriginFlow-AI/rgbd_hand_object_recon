from __future__ import annotations

from pathlib import Path

from hand_recon import (
    generate_mock_visual_report,
    load_hand_result_npz,
    load_surface_geometry_npz,
    run_mock_reconstruction,
    validate_hand_result,
)


def test_public_api_runs_mock_pipeline_and_report(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    output_dir = tmp_path / "demo"
    result = run_mock_reconstruction(scene_dir=scene_dir, output_dir=output_dir, hand_side="right")

    assert result.ok
    assert result.output_paths["summary"].exists()
    assert result.output_paths["kr3_hand_result"].exists()
    assert result.quality_report["passed"] is True
    assert result.surface_result.ok
    assert result.surface_result.mesh.vertex_count > 100
    assert result.surface_result.mesh.face_count > 100
    assert result.output_paths["hand_surface"].exists()
    assert result.output_paths["surface_report"].exists()

    surface = load_surface_geometry_npz(result.output_paths["hand_geometry"])
    assert surface["mesh_vertices_m"].shape[0] > 100
    assert surface["mesh_faces"].shape[0] > 100
    assert all(not value.dtype.hasobject for value in surface.values())

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
    report = report_path.read_text(encoding="utf-8")
    assert 'data-role="surface-viewer"' in report
    assert 'data-role="view-selector"' in report
    assert 'data-role="point-view-filter"' in report
    assert 'data-role="color-mode"' in report
    assert "joints_3d_m" not in report
    assert "http://" not in report and "https://" not in report
