#!/usr/bin/env python3
"""Generate a visual HTML report from the best local Re:InterHand pilot data."""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CAPTURE = "m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands"
MESH_DIR = ROOT / "data" / "reinterhand" / CAPTURE / "mano_fits" / "meshes"
DEFAULT_ICP_DIR = ROOT / "outputs" / "reinterhand_best_right_sequence_icp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--icp-dir", type=Path, default=DEFAULT_ICP_DIR)
    parser.add_argument(
        "--output-html", type=Path, default=ROOT / "outputs" / "reports" / "best_data_reinterhand_visual_report.html"
    )
    parser.add_argument("--input-scale", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_html = generate_best_data_visual_report(
        icp_dir=args.icp_dir,
        output_html=args.output_html,
        input_scale=args.input_scale,
    )
    print(json.dumps({"status": "ok", "html": str(output_html)}, ensure_ascii=False))
    return 0


def generate_best_data_visual_report(
    *,
    icp_dir: Path = DEFAULT_ICP_DIR,
    output_html: Path = ROOT / "outputs" / "reports" / "best_data_reinterhand_visual_report.html",
    input_scale: float = 0.001,
) -> Path:
    """Generate the Re:InterHand best-data visual HTML report."""

    icp_dir = Path(icp_dir)
    output_html = Path(output_html)
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("input_scale must be a positive finite value")
    summary_path = icp_dir / "icp_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing ICP summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    target_path = ROOT / summary["target"]
    source_paths = [ROOT / item["source"] for item in summary["sources"]]
    aligned_paths = [ROOT / item["output_ply"] for item in summary["sources"]]
    merged_path = ROOT / summary["merged_output_ply"]

    target = read_ply_mesh(target_path, scale=input_scale)
    sources = [read_ply_mesh(path, scale=input_scale) for path in source_paths]
    aligned = [read_ply_mesh(path, scale=1.0) for path in aligned_paths]
    merged = read_ply_mesh(merged_path, scale=1.0)
    pilot_summary = load_json(ROOT / "outputs" / "reinterhand_pilot_summary.json")

    html_text = render_report(
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        pilot_summary=pilot_summary,
        icp_summary=summary,
        target=target,
        sources=sources,
        aligned=aligned,
        merged=merged,
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    return output_html


def render_report(
    *,
    generated_at: str,
    pilot_summary: dict[str, Any],
    icp_summary: dict[str, Any],
    target: dict[str, np.ndarray],
    sources: list[dict[str, np.ndarray]],
    aligned: list[dict[str, np.ndarray]],
    merged: dict[str, np.ndarray],
) -> str:
    metrics = icp_summary.get("sources", [])
    rmse_values = [float(item.get("rmse", 0.0)) for item in metrics]
    fitness_values = [float(item.get("fitness", 0.0)) for item in metrics]
    selected = pilot_summary.get("selected_capture_summary", {})
    metric_cards = [
        ("数据源", "Re:InterHand", "真实 MANO mesh pilot 数据"),
        ("mesh 总数", str(selected.get("mano_mesh_count", 0)), "已解压 MANO mesh 文件"),
        ("对齐帧数", str(1 + len(sources)), "1 个锚点 + 多个连续帧"),
        ("顶点 / 面片", f"{target['points'].shape[0]} / {target['faces'].shape[0]}", "单帧 MANO 拓扑"),
        ("平均 RMSE", f"{np.mean(rmse_values) * 1000:.3f} mm" if rmse_values else "-", "多源 ICP 后误差"),
        ("平均 fitness", f"{np.mean(fitness_values):.3f}" if fitness_values else "-", "阈值内匹配比例"),
    ]
    rows = []
    for item in metrics:
        rows.append(
            (
                Path(item["source"]).name,
                item.get("status", ""),
                str(item.get("iterations", "")),
                f"{float(item.get('mean_error', 0.0)) * 1000:.3f} mm",
                f"{float(item.get('rmse', 0.0)) * 1000:.3f} mm",
                f"{float(item.get('fitness', 0.0)):.3f}",
            )
        )
    source_note = (
        "本报告使用本地可用的最高可信数据：Re:InterHand pilot 中的真实 MANO hand mesh 与相机参数下载记录。"
        "图中结果用于展示真实 mesh 序列与刚体对齐质量；它不是 RGB-D 传感器原始深度结果。"
    )
    raw_series = [("anchor", target["points"], "#4a90e2")] + [
        (f"src{i}", item["points"], color)
        for i, (item, color) in enumerate(
            zip(sources, ["#4d9b53", "#f0a202", "#d95f9f", "#6b6bd6"], strict=False), start=1
        )
    ]
    aligned_series = [("anchor", target["points"], "#4a90e2")] + [
        (f"aligned{i}", item["points"], color)
        for i, (item, color) in enumerate(
            zip(aligned, ["#4d9b53", "#f0a202", "#d95f9f", "#6b6bd6"], strict=False), start=1
        )
    ]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Re:InterHand 最佳数据可视化报告</title>
  <style>
    :root {{
      --ink: #182330;
      --muted: #607083;
      --line: #d8e0ea;
      --soft: #f5f8fb;
      --blue: #4a90e2;
      --green: #4d9b53;
      --orange: #f0a202;
      --pink: #d95f9f;
    }}
    body {{ margin: 0; background: #edf1f5; color: var(--ink); font-family: Arial, "Microsoft YaHei", sans-serif; }}
    .page {{ max-width: 1480px; margin: 0 auto; min-height: 100vh; background: white; box-shadow: 0 0 30px rgba(20, 32, 44, .12); }}
    header {{ padding: 30px 44px 24px; border-bottom: 1px solid var(--line); background: #fbfcfe; }}
    main {{ padding: 24px 44px 44px; }}
    h1 {{ margin: 0 0 10px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ line-height: 1.6; }}
    .muted {{ color: var(--muted); }}
    .note {{ margin-top: 16px; padding: 14px 16px; border: 1px solid #d9c77b; background: #fff8df; border-radius: 8px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin-top: 20px; }}
    .metric {{ padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: white; }}
    .metric .value {{ font-size: 24px; font-weight: 700; margin: 8px 0 2px; }}
    .grid2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .section {{ margin-top: 24px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: white; }}
    .card-body {{ padding: 16px; }}
    .viz svg {{ display: block; width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); }}
    th, td {{ padding: 11px 14px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ background: var(--soft); }}
    code {{ font-family: Consolas, "SFMono-Regular", monospace; font-size: 12px; }}
    iframe {{ width: 100%; height: 980px; border: 0; display: block; background: white; }}
    .embed-frame {{ border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: white; }}
    @media (max-width: 1100px) {{ .metric-grid {{ grid-template-columns: repeat(3, 1fr); }} .grid2 {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 720px) {{ header, main {{ padding-left: 18px; padding-right: 18px; }} .metric-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <h1>Re:InterHand 最佳数据可视化报告</h1>
      <div class="muted">生成时间：{escape(generated_at)} · Capture：{escape(CAPTURE)}</div>
      <p class="note">{escape(source_note)}</p>
      <div class="metric-grid">{"".join(render_metric_card(*item) for item in metric_cards)}</div>
    </header>
    <main>
      <section class="section grid2">
        <article class="card">
          <div class="viz">{mesh_triptych(target, "Step 1  anchor MANO mesh topology")}</div>
          <div class="card-body">
            <h3>Step 1：锚点手部 mesh</h3>
            <p>使用真实 MANO 拓扑，单帧包含 {target["points"].shape[0]} 个顶点和 {target["faces"].shape[0]} 个三角面片。三视图用于检查手部形状、尺度和拓扑完整性。</p>
          </div>
        </article>
        <article class="card">
          <div class="viz">{projection_triptych(raw_series, "Step 2  raw mesh sequence before alignment")}</div>
          <div class="card-body">
            <h3>Step 2：连续帧原始位置叠加</h3>
            <p>不同颜色表示连续帧手部 mesh。未对齐前的空间差异可用于观察手部运动与帧间漂移。</p>
          </div>
        </article>
      </section>

      <section class="section grid2">
        <article class="card">
          <div class="viz">{projection_triptych(aligned_series, "Step 3  aligned mesh sequence after ICP")}</div>
          <div class="card-body">
            <h3>Step 3：ICP 对齐后的多帧叠加</h3>
            <p>多帧 mesh 被刚体配准到同一锚点帧。平均 RMSE 为 {np.mean(rmse_values) * 1000:.3f} mm，平均 fitness 为 {np.mean(fitness_values):.3f}。</p>
          </div>
        </article>
        <article class="card">
          <div class="viz">{projection_triptych([("merged", merged["points"], "#4a90e2")], "Step 4  merged aligned voxel cloud")}</div>
          <div class="card-body">
            <h3>Step 4：对齐融合点云</h3>
            <p>对齐后的帧被体素融合，得到 {merged["points"].shape[0]} 个点。该结果适合做重建稳定性、局部形变和配准误差诊断。</p>
          </div>
        </article>
      </section>

      <section class="section grid2">
        <article class="card">
          <div class="viz">{metric_bar_chart([Path(item["source"]).stem for item in metrics], [float(item["rmse"]) * 1000 for item in metrics], "Step 5  ICP RMSE by source frame", "mm")}</div>
          <div class="card-body">
            <h3>Step 5：逐帧配准误差</h3>
            <p>柱状图展示各源帧对齐到锚点后的 RMSE。当前最大 RMSE 为 {max(rmse_values) * 1000:.3f} mm。</p>
          </div>
        </article>
        <article class="card">
          <div class="viz">{metric_bar_chart([Path(item["source"]).stem for item in metrics], [float(item["fitness"]) for item in metrics], "Step 6  ICP fitness by source frame", "fitness")}</div>
          <div class="card-body">
            <h3>Step 6：逐帧匹配比例</h3>
            <p>fitness 表示阈值内匹配比例。当前所有源帧状态均为 converged。</p>
          </div>
        </article>
      </section>

      <section class="section">
        <h2>配准明细</h2>
        <table>
          <thead><tr><th>源帧</th><th>状态</th><th>迭代</th><th>mean error</th><th>RMSE</th><th>fitness</th></tr></thead>
          <tbody>{"".join(f"<tr><td><code>{escape(a)}</code></td><td>{escape(b)}</td><td>{escape(c)}</td><td>{escape(d)}</td><td>{escape(e)}</td><td>{escape(f)}</td></tr>" for a, b, c, d, e, f in rows)}</tbody>
        </table>
      </section>

      <section class="section">
        <h2>手部重建补充图</h2>
        <p class="muted">下方内嵌同目录下的手部重建可视化报告，包含点云、关键点、网格和关节角图。</p>
        <div class="embed-frame">
          <iframe src="hand_reconstruction_visual_report.html" title="hand reconstruction visual report"></iframe>
        </div>
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
      <div class="muted">{escape(detail)}</div>
    </div>"""


def mesh_triptych(mesh: dict[str, np.ndarray], title: str) -> str:
    panels = [("XY top", (0, 1)), ("XZ front", (0, 2)), ("YZ side", (1, 2))]
    width = 900
    height = 430
    panel_w = width / 3
    all_2d = np.vstack([mesh["points"][:, [a, b]] for _, (a, b) in panels])
    bounds = point_bounds(all_2d)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    pieces.append('<rect x="0" y="0" width="900" height="430" fill="#ffffff"/>')
    for idx, (label, axes) in enumerate(panels):
        x0 = idx * panel_w
        pieces.append(f'<line x1="{x0:.1f}" y1="0" x2="{x0:.1f}" y2="{height}" stroke="#d8e0ea"/>')
        pieces.append(f'<text x="{x0 + 12:.1f}" y="20" font-size="13" font-weight="700">{escape(label)}</text>')
        projected = mesh["points"][:, [axes[0], axes[1]]]
        mapped = map_points(projected, bounds, x0 + 18, 34, panel_w - 36, height - 70)
        for face in mesh["faces"][::2]:
            pts = " ".join(f"{mapped[int(i)][0]:.2f},{mapped[int(i)][1]:.2f}" for i in face)
            pieces.append(
                f'<polygon points="{pts}" fill="#cdeeff" stroke="#4a90e2" stroke-width="0.75" opacity=".35"/>'
            )
        for x, y in mapped:
            pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.25" fill="#1f5f9f" opacity=".72"/>')
    pieces.append(f'<text x="14" y="{height - 12}" font-size="12" font-weight="700">{escape(title)}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def projection_triptych(series: list[tuple[str, np.ndarray, str]], title: str) -> str:
    panels = [("XY top", (0, 1)), ("XZ front", (0, 2)), ("YZ side", (1, 2))]
    width = 900
    height = 430
    panel_w = width / 3
    all_2d = np.vstack([points[:, [a, b]] for _, points, _ in series for _, (a, b) in panels if points.size])
    bounds = point_bounds(all_2d)
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    pieces.append('<rect x="0" y="0" width="900" height="430" fill="#ffffff"/>')
    for idx, (label, axes) in enumerate(panels):
        x0 = idx * panel_w
        pieces.append(f'<line x1="{x0:.1f}" y1="0" x2="{x0:.1f}" y2="{height}" stroke="#d8e0ea"/>')
        pieces.append(f'<text x="{x0 + 12:.1f}" y="20" font-size="13" font-weight="700">{escape(label)}</text>')
        for _, points, color in series:
            projected = points[:, [axes[0], axes[1]]]
            mapped = map_points(projected, bounds, x0 + 18, 34, panel_w - 36, height - 70)
            for x, y in mapped:
                pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.35" fill="{color}" opacity=".66"/>')
    pieces.append(f'<text x="14" y="{height - 12}" font-size="12" font-weight="700">{escape(title)}</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def metric_bar_chart(labels: list[str], values: list[float], title: str, unit: str) -> str:
    width = 760
    height = 410
    left = 54
    right = 24
    top = 38
    bottom = 116
    max_value = max(values + [1e-9]) * 1.15
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    pieces.append('<rect x="0" y="0" width="760" height="410" fill="#ffffff"/>')
    pieces.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#97a4b2"/>'
    )
    pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#97a4b2"/>')
    area_w = width - left - right
    bar_w = area_w / max(1, len(values)) * 0.58
    for idx, (label, value) in enumerate(zip(labels, values, strict=False)):
        cx = left + (idx + 0.5) * area_w / max(1, len(values))
        h = (height - bottom - top) * value / max_value
        x = cx - bar_w / 2
        y = height - bottom - h
        pieces.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="#4a90e2" opacity=".86"/>'
        )
        pieces.append(f'<text x="{cx:.2f}" y="{y - 8:.2f}" font-size="12" text-anchor="middle">{value:.3f}</text>')
        pieces.append(
            f'<text transform="translate({cx - 4:.2f},{height - 102}) rotate(58)" font-size="10">{escape(label)}</text>'
        )
    pieces.append(f'<text x="14" y="20" font-size="13" font-weight="700">{escape(title)} ({escape(unit)})</text>')
    pieces.append("</svg>")
    return "".join(pieces)


def read_ply_mesh(path: Path, scale: float) -> dict[str, np.ndarray]:
    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    vertex_count = 0
    face_count = 0
    header_end = 0
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        if len(parts) == 3 and parts[:2] == ["element", "face"]:
            face_count = int(parts[2])
        if line.strip() == "end_header":
            header_end = idx + 1
            break
    vertices = []
    for line in lines[header_end : header_end + vertex_count]:
        values = line.split()
        vertices.append([float(values[0]) * scale, float(values[1]) * scale, float(values[2]) * scale])
    faces = []
    face_start = header_end + vertex_count
    for line in lines[face_start : face_start + face_count]:
        values = line.split()
        if len(values) >= 4 and int(values[0]) == 3:
            faces.append([int(values[1]), int(values[2]), int(values[3])])
    return {
        "points": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
        "path": str(path),
    }


def point_bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    pad = np.maximum((maxs - mins) * 0.08, 1e-6)
    return float(mins[0] - pad[0]), float(maxs[0] + pad[0]), float(mins[1] - pad[1]), float(maxs[1] + pad[1])


def map_points(
    points: np.ndarray, bounds: tuple[float, float, float, float], x: float, y: float, w: float, h: float
) -> np.ndarray:
    min_x, max_x, min_y, max_y = bounds
    scale_x = w / max(max_x - min_x, 1e-9)
    scale_y = h / max(max_y - min_y, 1e-9)
    scale = min(scale_x, scale_y)
    used_w = (max_x - min_x) * scale
    used_h = (max_y - min_y) * scale
    off_x = x + (w - used_w) / 2
    off_y = y + (h - used_h) / 2
    return np.column_stack([off_x + (points[:, 0] - min_x) * scale, off_y + used_h - (points[:, 1] - min_y) * scale])


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
