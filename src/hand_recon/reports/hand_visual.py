#!/usr/bin/env python3
"""Generate a self-contained visual HTML report for the mock hand pipeline."""

from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-html", type=Path, default=ROOT / "outputs" / "reports" / "hand_reconstruction_visual_report.html")
    parser.add_argument("--demo-dir", type=Path, default=ROOT / "outputs" / "mock_rgbd_demo")
    parser.add_argument("--max-points", type=int, default=2800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_html = generate_hand_visual_report(
        demo_dir=args.demo_dir,
        output_html=args.output_html,
        max_points=args.max_points,
    )
    print(json.dumps({"status": "ok", "html": str(output_html)}, ensure_ascii=False))
    return 0


def generate_hand_visual_report(
    *,
    demo_dir: Path = ROOT / "outputs" / "mock_rgbd_demo",
    output_html: Path = ROOT / "outputs" / "reports" / "hand_reconstruction_visual_report.html",
    max_points: int = 2800,
) -> Path:
    """Generate the hand reconstruction visual HTML report."""

    demo_dir = Path(demo_dir)
    output_html = Path(output_html)
    quality = load_json(demo_dir / "quality_report.json")
    pose = load_json(demo_dir / "pose_output.json")
    hand = read_ascii_ply(demo_dir / "hand_pointcloud.ply")
    obj = read_ascii_ply(demo_dir / "object_pointcloud.ply")
    fused = read_ascii_ply(demo_dir / "fused_pointcloud.ply")
    hand_result_path = demo_dir / "kr3" / "hand_result.npz"
    with np.load(hand_result_path, allow_pickle=True) as data:
        joints = np.asarray(data["joints_3d_m"], dtype=np.float64)[0]
        mesh_vertices = np.asarray(data["mesh_vertices_m"], dtype=np.float64)[0]
        mesh_faces = np.asarray(data["mesh_faces"], dtype=np.int64)
        angles_deg = np.asarray(data["hand_angles_22dof_deg"], dtype=np.float64)[0]
        angle_names = np.asarray(data["hand_angle_names_22dof"]).astype(str).tolist()

    rng = np.random.default_rng(20260709)
    report = render_report(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        quality=quality,
        pose=pose,
        fused=sample_points(fused, max_points, rng),
        hand=sample_points(hand, max_points, rng),
        obj=sample_points(obj, max_points, rng),
        joints=joints,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        angles_deg=angles_deg,
        angle_names=angle_names,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(report, encoding="utf-8")
    return output_html


def render_report(
    *,
    generated_at: str,
    quality: dict[str, Any],
    pose: dict[str, Any],
    fused: np.ndarray,
    hand: np.ndarray,
    obj: np.ndarray,
    joints: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    angles_deg: np.ndarray,
    angle_names: list[str],
) -> str:
    metrics = quality.get("metrics", {})
    hand_pose = pose.get("hands", [{}])[0]
    object_pose = pose.get("objects", [{}])[0]
    metric_cards = [
        ("质量状态", "通过" if quality.get("passed") else "需检查", "整体质量门槛"),
        ("视角数量", str(quality.get("view_count", 0)), "参与融合的 RGB-D 视角"),
        ("融合点数", str(metrics.get("fused_point_count", 0)), "体素降采样后的点数"),
        ("覆盖率", f"{float(metrics.get('coverage_score', 0.0)):.2f}", "有效深度视角覆盖"),
        ("手部点数", str(metrics.get("hand_point_count", 0)), "hand mask 点云"),
        ("物体点数", str(metrics.get("object_point_count", 0)), "object mask 点云"),
    ]
    bbox = metrics.get("bbox_extent_m", [0.0, 0.0, 0.0])
    metric_rows = [
        ("depth valid ratio mean", f"{float(metrics.get('depth_valid_ratio_mean', 0.0)):.6f}", "有效深度像素占比均值。"),
        ("bbox extent", ", ".join(f"{float(v):.4f} m" for v in bbox), "融合点云三维包围盒尺寸。"),
        ("pose confidence mean", f"{float(metrics.get('pose_confidence_mean', 0.0)):.3f}", "手部与物体 mock 位姿平均置信度。"),
        ("hand centroid", vector_text(hand_pose.get("translation_m", [])), "手部点云中心估计。"),
        ("object centroid", vector_text(object_pose.get("translation_m", [])), "物体点云中心估计。"),
    ]
    per_view = metrics.get("depth_valid_ratio_per_view", {})
    source_note = (
        "当前报告基于 mock RGB-D 闭环生成，用于接口、可视化和质量诊断；"
        "其中手部网格为轻量占位拓扑，不等同于真实 MANO 或 UmeTrack 网格精度。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多视角 RGB-D 手部/物体重建可视化报告</title>
  <style>
    :root {{
      --ink: #1c2733;
      --muted: #5b697a;
      --line: #d7dee8;
      --soft: #f4f7fa;
      --panel: #ffffff;
      --blue: #4a90e2;
      --green: #4d9b53;
      --orange: #f0a202;
      --red: #e85d4f;
    }}
    body {{ margin: 0; background: #eef2f5; color: var(--ink); font-family: Arial, "Microsoft YaHei", sans-serif; }}
    .page {{ max-width: 1480px; margin: 0 auto; background: white; min-height: 100vh; box-shadow: 0 0 32px rgba(20, 32, 44, .12); }}
    header {{ padding: 28px 44px 22px; border-bottom: 1px solid var(--line); background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }}
    h1 {{ margin: 0 0 10px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ line-height: 1.58; }}
    main {{ padding: 24px 44px 44px; }}
    .muted {{ color: var(--muted); }}
    .note {{ padding: 14px 16px; background: #fff8df; border: 1px solid #eedb98; border-radius: 8px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
    .metric {{ padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }}
    .metric .value {{ font-size: 25px; font-weight: 700; margin: 8px 0 2px; }}
    .section {{ margin-top: 24px; }}
    .grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); overflow: hidden; }}
    .card-body {{ padding: 16px; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); background: white; }}
    th, td {{ padding: 11px 14px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #f1f5f9; }}
    code {{ font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; }}
    .viz svg {{ display: block; width: 100%; height: auto; background: white; }}
    .caption {{ color: var(--muted); font-size: 14px; }}
    .pill {{ display: inline-block; padding: 3px 9px; border-radius: 999px; background: #e8f1fc; color: #1f5f9f; font-weight: 700; font-size: 12px; }}
    @media (max-width: 1100px) {{ .metric-grid {{ grid-template-columns: repeat(3, 1fr); }} .grid2 {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 720px) {{ main, header {{ padding-left: 18px; padding-right: 18px; }} .metric-grid, .grid3 {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>多视角 RGB-D 手部/物体重建可视化报告</h1>
      <div class="muted">生成时间：{escape(generated_at)} · 数据集：mock RGB-D scene · 报告类型：可视化诊断</div>
      <p class="note">{escape(source_note)}</p>
      <div class="metric-grid">
        {''.join(render_metric_card(*item) for item in metric_cards)}
      </div>
    </header>
    <main>
      <section class="section">
        <h2>指标摘要</h2>
        <table>
          <thead><tr><th>指标</th><th>当前值</th><th>说明</th></tr></thead>
          <tbody>{''.join(f'<tr><td><code>{escape(k)}</code></td><td>{escape(v)}</td><td>{escape(d)}</td></tr>' for k, v, d in metric_rows)}</tbody>
        </table>
      </section>

      <section class="section grid2">
        <article class="card">
          <div class="viz">{projection_triptych([("hand", hand, "#4a90e2"), ("object", obj, "#f0a202")], "Step 1  多视角点云融合与类别叠加")}</div>
          <div class="card-body">
            <h3>Step 1：多视角点云融合</h3>
            <p>蓝色为手部点云，黄色为物体点云。三视图用于检查点云是否处在同一共享坐标系，以及手/物体空间关系是否合理。</p>
          </div>
        </article>
        <article class="card">
          <div class="viz">{projection_triptych([("fused", fused, "#4a90e2")], "Step 2  体素融合后的整体点云")}</div>
          <div class="card-body">
            <h3>Step 2：体素融合后的整体形状</h3>
            <p>融合点云经过体素降采样后保留主要几何形状，当前融合点数为 {escape(str(metrics.get("fused_point_count", 0)))}。</p>
          </div>
        </article>
      </section>

      <section class="section grid2">
        <article class="card">
          <div class="viz">{joint_projection(hand, joints)}</div>
          <div class="card-body">
            <h3>Step 3：21 个手部关键点</h3>
            <p>绿色点和连线为 21 个手部关键点骨架，灰色背景为手部点云。用于检查 wrist、各手指 MCP/PIP/DIP/tip 的空间分布。</p>
          </div>
        </article>
        <article class="card">
          <div class="viz">{mesh_projection(hand, mesh_vertices, mesh_faces)}</div>
          <div class="card-body">
            <h3>Step 4：手部网格拓扑占位</h3>
            <p>浅蓝面片展示当前接口中的 mesh 顶点/三角面字段已经可视化连通。该网格用于接口诊断，真实精度仍需接入完整手模型。</p>
          </div>
        </article>
      </section>

      <section class="section grid2">
        <article class="card">
          <div class="viz">{bar_chart(list(per_view.keys()), [float(v) for v in per_view.values()], "Step 5  各视角有效深度比例", value_format="{:.3f}")}</div>
          <div class="card-body">
            <h3>Step 5：多视角覆盖质量</h3>
            <p>各视角有效深度比例均高于当前阈值，覆盖率为 {float(metrics.get("coverage_score", 0.0)):.2f}。</p>
          </div>
        </article>
        <article class="card">
          <div class="viz">{angle_chart(angle_names, angles_deg)}</div>
          <div class="card-body">
            <h3>Step 6：22 维手部关节角</h3>
            <p>柱状图展示当前 22 维关节角，单位为 degree。最后两个 wrist 维度为当前接口预留位，mock 数据中为 0。</p>
          </div>
        </article>
      </section>

      <section class="section">
        <h2>接口字段快照</h2>
        <table>
          <thead><tr><th>字段</th><th>shape</th><th>用途</th></tr></thead>
          <tbody>
            <tr><td><code>hand_angles_22dof_rad</code></td><td>(1, 22)</td><td>手部 22 维关节角，单位 rad。</td></tr>
            <tr><td><code>joints_3d_m</code></td><td>(1, 21, 3)</td><td>21 个 3D 手部关键点，单位 m。</td></tr>
            <tr><td><code>mesh_vertices_m</code></td><td>{escape(str(tuple(mesh_vertices.reshape(1, *mesh_vertices.shape).shape)))}</td><td>手部 mesh 顶点，单位 m。</td></tr>
            <tr><td><code>mesh_faces</code></td><td>{escape(str(tuple(mesh_faces.shape)))}</td><td>mesh 三角面片。</td></tr>
          </tbody>
        </table>
      </section>
    </main>
  </div>
</body>
</html>
"""


def render_metric_card(label: str, value: str, detail: str) -> str:
    return f"""<div class="metric">
      <div class="muted">{escape(label)}</div>
      <div class="value">{escape(value)}</div>
      <div class="caption">{escape(detail)}</div>
    </div>"""


def projection_triptych(series: list[tuple[str, np.ndarray, str]], title: str) -> str:
    panels = [
        ("XY top", (0, 1)),
        ("XZ front", (0, 2)),
        ("YZ side", (1, 2)),
    ]
    width = 900
    height = 440
    panel_w = width / 3
    all_points = np.vstack([points[:, [axis[0], axis[1]]] for _, points, _ in series for _, axis in panels if points.size])
    bounds = point_bounds(all_points)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    pieces.append('<rect x="0" y="0" width="900" height="440" fill="#ffffff"/>')
    for idx, (label, axes) in enumerate(panels):
        x0 = idx * panel_w
        pieces.append(f'<line x1="{x0:.1f}" y1="0" x2="{x0:.1f}" y2="{height}" stroke="#d7dee8"/>')
        pieces.append(f'<text x="{x0 + 12:.1f}" y="20" font-size="13" font-weight="700">{escape(label)}</text>')
        for _, points, color in series:
            if points.size == 0:
                continue
            projected = points[:, [axes[0], axes[1]]]
            dots = scatter_points(projected, bounds, x0 + 18, 34, panel_w - 36, height - 74, color, radius=1.15, opacity=0.78)
            pieces.append(dots)
    pieces.append(f'<text x="14" y="{height - 14}" font-size="12" font-weight="700">{escape(title)}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def joint_projection(hand: np.ndarray, joints: np.ndarray) -> str:
    width = 760
    height = 400
    axes = (0, 2)
    projected_hand = hand[:, [axes[0], axes[1]]]
    projected_joints = joints[:, [axes[0], axes[1]]]
    bounds = point_bounds(np.vstack([projected_hand, projected_joints]))
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="21 hand landmarks">']
    pieces.append('<rect x="0" y="0" width="760" height="400" fill="#ffffff"/>')
    pieces.append(scatter_points(projected_hand, bounds, 28, 30, width - 56, height - 66, "#b9c0c8", radius=1.1, opacity=0.42))
    bone_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    mapped = map_points(projected_joints, bounds, 28, 30, width - 56, height - 66)
    for a, b in bone_edges:
        ax, ay = mapped[a]
        bx, by = mapped[b]
        pieces.append(f'<line x1="{ax:.2f}" y1="{ay:.2f}" x2="{bx:.2f}" y2="{by:.2f}" stroke="#4d9b53" stroke-width="2.2" opacity="0.86"/>')
    for idx, (x, y) in enumerate(mapped):
        color = "#4d9b53" if idx else "#5967d8"
        pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.0" fill="{color}" stroke="white" stroke-width="1.5"/>')
    pieces.append(f'<text x="14" y="{height - 12}" font-size="12" font-weight="700">Step 3  21 hand landmarks over point cloud</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def mesh_projection(hand: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> str:
    width = 760
    height = 400
    axes = (0, 2)
    hand_2d = hand[:, [axes[0], axes[1]]]
    vert_2d = vertices[:, [axes[0], axes[1]]]
    bounds = point_bounds(np.vstack([hand_2d, vert_2d]))
    mapped_hand = map_points(hand_2d, bounds, 28, 30, width - 56, height - 66)
    mapped_vertices = map_points(vert_2d, bounds, 28, 30, width - 56, height - 66)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="hand mesh topology">']
    pieces.append('<rect x="0" y="0" width="760" height="400" fill="#ffffff"/>')
    for x, y in mapped_hand:
        pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.0" fill="#b9c0c8" opacity=".32"/>')
    for face in faces:
        pts = " ".join(f"{mapped_vertices[int(i)][0]:.2f},{mapped_vertices[int(i)][1]:.2f}" for i in face)
        pieces.append(f'<polygon points="{pts}" fill="#cdeeff" stroke="#4a90e2" stroke-width="1.4" opacity=".62"/>')
    for x, y in mapped_vertices:
        pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.6" fill="#4a90e2" stroke="white" stroke-width="1.0"/>')
    pieces.append(f'<text x="14" y="{height - 12}" font-size="12" font-weight="700">Step 4  mesh vertices and faces</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def bar_chart(labels: list[str], values: list[float], title: str, value_format: str = "{:.2f}") -> str:
    width = 760
    height = 400
    left = 54
    right = 24
    top = 38
    bottom = 68
    max_value = max(values + [1e-9]) * 1.15
    bar_area_w = width - left - right
    bar_w = bar_area_w / max(1, len(values)) * 0.62
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    pieces.append('<rect x="0" y="0" width="760" height="400" fill="#ffffff"/>')
    pieces.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9aa7b3"/>')
    pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9aa7b3"/>')
    for idx, (label, value) in enumerate(zip(labels, values)):
        cx = left + (idx + 0.5) * bar_area_w / max(1, len(values))
        bar_h = (height - bottom - top) * value / max_value
        x = cx - bar_w / 2
        y = height - bottom - bar_h
        pieces.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#4a90e2" opacity=".86"/>')
        pieces.append(f'<text x="{cx:.2f}" y="{y - 8:.2f}" font-size="12" text-anchor="middle">{escape(value_format.format(value))}</text>')
        pieces.append(f'<text x="{cx:.2f}" y="{height - 38}" font-size="12" text-anchor="middle">{escape(label)}</text>')
    pieces.append(f'<text x="14" y="20" font-size="13" font-weight="700">{escape(title)}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def angle_chart(names: list[str], angles_deg: np.ndarray) -> str:
    width = 760
    height = 400
    left = 46
    right = 20
    top = 28
    bottom = 108
    values = np.asarray(angles_deg, dtype=np.float64)
    limit = max(15.0, float(np.max(np.abs(values))) * 1.2)
    zero_y = top + (height - top - bottom) / 2
    area_h = height - top - bottom
    bar_area_w = width - left - right
    bar_w = bar_area_w / len(values) * 0.72
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="22 hand joint angles">']
    pieces.append('<rect x="0" y="0" width="760" height="400" fill="#ffffff"/>')
    pieces.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#9aa7b3"/>')
    pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#9aa7b3"/>')
    for idx, value in enumerate(values):
        cx = left + (idx + 0.5) * bar_area_w / len(values)
        h = abs(value) / limit * area_h / 2
        y = zero_y - h if value >= 0 else zero_y
        color = "#4d9b53" if value >= 0 else "#e85d4f"
        pieces.append(f'<rect x="{cx - bar_w/2:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" opacity=".86"/>')
        short = names[idx].replace("_", " ")
        pieces.append(f'<text transform="translate({cx:.2f},{height - 96}) rotate(62)" font-size="10">{escape(short)}</text>')
    pieces.append(f'<text x="14" y="20" font-size="13" font-weight="700">Step 6  22 joint angles in degree</text>')
    pieces.append(f'<text x="{width - right - 72}" y="{top + 14}" font-size="12" fill="#5b697a">±{limit:.1f}°</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def read_ascii_ply(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = 0
    start = 0
    for idx, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line == "end_header":
            start = idx + 1
            break
    rows = []
    for line in lines[start : start + vertex_count]:
        parts = line.split()
        if len(parts) >= 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(rows, dtype=np.float64)


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(indices)]


def point_bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    if points.size == 0:
        return (-1.0, 1.0, -1.0, 1.0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.08, 1e-6)
    return float(mins[0] - pad[0]), float(maxs[0] + pad[0]), float(mins[1] - pad[1]), float(maxs[1] + pad[1])


def map_points(points: np.ndarray, bounds: tuple[float, float, float, float], x: float, y: float, w: float, h: float) -> np.ndarray:
    min_x, max_x, min_y, max_y = bounds
    scale_x = w / max(max_x - min_x, 1e-9)
    scale_y = h / max(max_y - min_y, 1e-9)
    scale = min(scale_x, scale_y)
    used_w = (max_x - min_x) * scale
    used_h = (max_y - min_y) * scale
    off_x = x + (w - used_w) / 2
    off_y = y + (h - used_h) / 2
    mapped_x = off_x + (points[:, 0] - min_x) * scale
    mapped_y = off_y + used_h - (points[:, 1] - min_y) * scale
    return np.column_stack([mapped_x, mapped_y])


def scatter_points(
    points: np.ndarray,
    bounds: tuple[float, float, float, float],
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    *,
    radius: float,
    opacity: float,
) -> str:
    mapped = map_points(points, bounds, x, y, w, h)
    return "".join(
        f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{color}" opacity="{opacity:.2f}"/>'
        for px, py in mapped
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def vector_text(values: list[float]) -> str:
    if not values:
        return "-"
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "] m"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
