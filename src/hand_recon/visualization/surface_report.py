"""Self-contained interactive report for joint-independent hand surfaces."""

from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from hand_recon.io.json_io import read_json_object
from hand_recon.io.npz import load_npz_arrays
from hand_recon.rgbd import RgbdView, load_mock_rgbd_scene


def generate_surface_visual_report(
    *,
    demo_dir: Path,
    output_html: Path,
    max_points: int = 3500,
    max_faces: int = 4500,
) -> Path:
    """Render a portable report from the surface artifact contract only."""

    if max_points <= 0 or max_faces <= 0:
        raise ValueError("max_points and max_faces must be greater than zero")
    demo_dir = Path(demo_dir).resolve()
    output_html = Path(output_html).resolve()
    manifest = read_json_object(demo_dir / "manifest.json")
    quality = read_json_object(_artifact_path(demo_dir, manifest, "surface_quality"))
    geometry = load_npz_arrays(_artifact_path(demo_dir, manifest, "hand_geometry"))
    scene_dir = Path(str(manifest["source_scene_dir"])).resolve()
    scene = load_mock_rgbd_scene(scene_dir)

    points, point_colors, point_views = _sample_points(
        np.asarray(geometry["raw_points_m"], dtype=np.float64),
        np.asarray(geometry["raw_colors_rgb"], dtype=np.uint8),
        np.asarray(geometry["raw_view_indices"], dtype=np.int32),
        max_points,
    )
    mesh_vertices, mesh_colors, mesh_faces = _sample_mesh(
        np.asarray(geometry["mesh_vertices_m"], dtype=np.float64),
        np.asarray(geometry["mesh_vertex_colors_rgb"], dtype=np.uint8),
        np.asarray(geometry["mesh_faces"], dtype=np.int64),
        max_faces,
    )
    payload = {
        "points": np.column_stack([points, point_colors, point_views]).round(7).tolist(),
        "meshVertices": np.column_stack([mesh_vertices, mesh_colors]).round(7).tolist(),
        "meshFaces": mesh_faces.tolist(),
        "views": [_preview(view) for view in scene.views],
    }
    artifact_links = _artifact_links(demo_dir, output_html, manifest)
    report = _render_html(manifest, quality, payload, artifact_links)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(output_html, report)
    return output_html


def _render_html(
    manifest: dict[str, Any],
    quality: dict[str, Any],
    payload: dict[str, Any],
    artifact_links: list[tuple[str, str]],
) -> str:
    metrics = quality.get("metrics", {})
    warnings = quality.get("warnings", [])
    cards = [
        ("表面状态", str(quality.get("status", "unknown")), "观测几何质量门禁"),
        ("输入视角", metrics.get("input_view_count", 0), "同步标定 RGB-D 视角"),
        ("融合点数", manifest.get("counts", {}).get("fused_point_count", 0), "彩色体素融合点云"),
        ("网格顶点", metrics.get("mesh_vertex_count", 0), "TSDF 零等值面顶点"),
        ("三角面", metrics.get("mesh_face_count", 0), "清理后的主连通表面"),
        ("多视角支持", f"{100 * float(metrics.get('multi_view_vertex_ratio', 0)):.1f}%", "至少两个视角支持的顶点"),
    ]
    rows = [
        ("source_to_surface_p95_m", _number(metrics.get("source_to_surface_p95_m"), 6), "观测点到网格顶点 P95"),
        ("surface_to_source_p95_m", _number(metrics.get("surface_to_source_p95_m"), 6), "网格顶点到观测点 P95"),
        ("surface_area_m2", _number(metrics.get("surface_area_m2"), 6), "观测表面积"),
        ("largest_component_face_ratio", _number(metrics.get("largest_component_face_ratio"), 4), "最大连通面占比"),
        ("boundary_edge_ratio", _number(metrics.get("boundary_edge_ratio"), 4), "观测边界占比；非推断补洞"),
        ("non_manifold_edge_count", str(metrics.get("non_manifold_edge_count", 0)), "非流形边数量"),
    ]
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RGB-D 手部表面重建报告</title>
  <style>
    :root {{ --ink:#17212b; --muted:#607080; --line:#d9e1e8; --panel:#fff; --bg:#eef3f6; --blue:#2d78c4; --green:#2f8f5b; --amber:#b87800; }}
    * {{ box-sizing:border-box }} body {{ margin:0; color:var(--ink); background:var(--bg); font-family:Inter,Arial,"Microsoft YaHei",sans-serif }}
    .page {{ max-width:1500px; margin:auto; min-height:100vh; background:white; box-shadow:0 0 30px #18304420 }}
    header,main {{ padding:26px 38px }} header {{ background:linear-gradient(135deg,#f9fcff,#edf6f2); border-bottom:1px solid var(--line) }}
    h1 {{ margin:0 0 8px; font-size:30px }} h2 {{ margin:0 0 14px }} h3 {{ margin:0 0 8px }} p {{ line-height:1.55 }}
    .note {{ padding:12px 14px; background:#fff8df; border:1px solid #ead48c; border-radius:9px }} .muted {{ color:var(--muted) }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-top:18px }}
    .metric,.card {{ border:1px solid var(--line); background:var(--panel); border-radius:10px }} .metric {{ padding:13px }}
    .metric strong {{ display:block; font-size:24px; margin:7px 0 3px }} .section {{ margin-top:24px }}
    .viewer-grid {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(320px,1fr); gap:16px }}
    .viewer-wrap {{ overflow:hidden; background:#101820; border-radius:10px }} #viewer {{ width:100%; height:auto; display:block; cursor:grab }}
    .controls {{ display:flex; flex-wrap:wrap; gap:8px; padding:12px; background:#17242f; color:#eef7ff }}
    button,select,label.control {{ border:1px solid #9db1c2; border-radius:7px; background:white; color:#17212b; padding:7px 10px }}
    .previews {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px }} .preview canvas {{ width:100%; image-rendering:pixelated; background:#111 }}
    .card-body {{ padding:15px }} table {{ width:100%; border-collapse:collapse }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left }} th {{ background:#f3f7fa }}
    .links {{ display:flex; flex-wrap:wrap; gap:8px }} .links a {{ color:#155f9f; border:1px solid #b9d1e5; border-radius:7px; padding:7px 10px; text-decoration:none }}
    .warning {{ color:#815900 }} code {{ font-family:Consolas,monospace; font-size:12px }}
    @media(max-width:1100px) {{ .metrics {{ grid-template-columns:repeat(3,1fr) }} .viewer-grid {{ grid-template-columns:1fr }} }}
    @media(max-width:700px) {{ header,main {{ padding:18px }} .metrics,.previews {{ grid-template-columns:1fr }} }}
  </style>
</head>
<body><div class="page">
  <header>
    <h1>多视角 RGB-D 手部表面重建</h1>
    <div class="muted">场景：{_escape(manifest.get("scene_id", "-"))} · 坐标系：{_escape(manifest.get("coordinate_frame", "-"))} · 单位：meter</div>
    <p class="note">本报告展示的是 RGB-D 实际观测支持的手部外表面。未观测区域没有使用关节点、骨架或参数手模型进行臆测补全；边界和低支持区域会作为质量风险保留。</p>
    <div class="metrics">{"".join(_card(*item) for item in cards)}</div>
  </header>
  <main>
    <section class="section viewer-grid">
      <article class="card">
        <div class="viewer-wrap">
          <canvas id="viewer" width="1000" height="620" data-role="surface-viewer"></canvas>
          <div class="controls" data-role="layer-controls">
            <label class="control"><input id="meshLayer" type="checkbox" checked> 表面网格</label>
            <label class="control"><input id="pointLayer" type="checkbox" checked> 观测点云</label>
            <select id="pointView" data-role="point-view-filter"><option value="-1">全部视角</option></select>
            <select id="colorMode" data-role="color-mode"><option value="rgb">RGB 着色</option><option value="view">视角着色</option></select>
            <button data-view="front">正视</button><button data-view="side">侧视</button><button data-view="top">顶视</button><button data-view="reset">复位</button>
          </div>
        </div>
      </article>
      <article class="card"><div class="card-body">
        <h2>三维检查</h2>
        <p>拖动旋转，滚轮缩放；可分别开关 TSDF 表面与按 RGB 着色的多视角观测点。报告仅嵌入确定性采样，全量结果保存在 PLY/NPZ。</p>
        <p><strong>后端：</strong><code>{_escape(manifest.get("parameters", {}).get("backend", "-"))}</code></p>
        <p><strong>语义：</strong><code>observed_not_completed</code></p>
        <p><strong>关节点定位：</strong>未使用</p>
        <p class="warning"><strong>质量提示：</strong>{_escape(", ".join(map(str, warnings)) if warnings else "无")}</p>
      </div></article>
    </section>

    <section class="section card"><div class="card-body">
      <h2>输入证据</h2>
      <p>选择相机后同时核对 RGB、深度伪彩和分割掩码；任何表面都必须能够追溯到这些观测。</p>
      <select id="viewSelect" data-role="view-selector"></select>
      <div class="previews">
        <div class="preview"><h3>RGB</h3><canvas id="rgbPreview" width="64" height="48"></canvas></div>
        <div class="preview"><h3>深度</h3><canvas id="depthPreview" width="64" height="48"></canvas></div>
        <div class="preview"><h3>手部分割</h3><canvas id="maskPreview" width="64" height="48"></canvas></div>
      </div>
    </div></section>

    <section class="section"><h2>表面质量</h2><table><thead><tr><th>指标</th><th>数值</th><th>解释</th></tr></thead><tbody>
      {"".join(f"<tr><td><code>{_escape(name)}</code></td><td>{_escape(value)}</td><td>{_escape(detail)}</td></tr>" for name, value, detail in rows)}
    </tbody></table></section>

    <section class="section card"><div class="card-body"><h2>完整产物</h2><div class="links">
      {"".join(f'<a href="{_escape(href)}">{_escape(label)}</a>' for label, href in artifact_links)}
    </div></div></section>
  </main>
</div>
<script id="surface-data" type="application/json">{payload_json}</script>
<script>
(() => {{
  'use strict';
  const data=JSON.parse(document.getElementById('surface-data').textContent);
  const canvas=document.getElementById('viewer'),ctx=canvas.getContext('2d');
  let yaw=-0.65,pitch=0.35,zoom=1.0,drag=false,lastX=0,lastY=0;
  const all=data.meshVertices.length?data.meshVertices:data.points;
  const center=[0,1,2].map(a=>(Math.min(...all.map(p=>p[a]))+Math.max(...all.map(p=>p[a])))/2);
  const radius=Math.max(...all.map(p=>Math.hypot(p[0]-center[0],p[1]-center[1],p[2]-center[2])))||1;
  function project(p) {{ const x=(p[0]-center[0])/radius,y=(p[1]-center[1])/radius,z=(p[2]-center[2])/radius;
    const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch); const x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;
    return [canvas.width/2+x1*canvas.height*.38*zoom,canvas.height/2-y1*canvas.height*.38*zoom,z2]; }}
  function draw() {{ ctx.fillStyle='#101820';ctx.fillRect(0,0,canvas.width,canvas.height);
    const pv=data.meshVertices.map(project);
    if(document.getElementById('meshLayer').checked) {{ const faces=data.meshFaces.map(f=>[f,(pv[f[0]][2]+pv[f[1]][2]+pv[f[2]][2])/3]).sort((a,b)=>a[1]-b[1]);
      for(const [f] of faces) {{ const c=f.map(i=>data.meshVertices[i].slice(3,6)); const rgb=[0,1,2].map(k=>Math.round((c[0][k]+c[1][k]+c[2][k])/3));
        ctx.beginPath();ctx.moveTo(pv[f[0]][0],pv[f[0]][1]);ctx.lineTo(pv[f[1]][0],pv[f[1]][1]);ctx.lineTo(pv[f[2]][0],pv[f[2]][1]);ctx.closePath();ctx.fillStyle=`rgba(${{rgb[0]}},${{rgb[1]}},${{rgb[2]}},.58)`;ctx.fill(); }} }}
    if(document.getElementById('pointLayer').checked) {{ const selected=Number(document.getElementById('pointView').value),mode=document.getElementById('colorMode').value,palette=['#50a5f1','#ef8354','#65b96e','#b47aea','#f1c453','#52b9b0'];
      for(const p of data.points) {{if(selected>=0&&p[6]!==selected)continue;const q=project(p);ctx.fillStyle=mode==='view'?palette[p[6]%palette.length]:`rgb(${{p[3]}},${{p[4]}},${{p[5]}})`;ctx.fillRect(q[0]-1,q[1]-1,2.2,2.2); }} }}
    ctx.fillStyle='#dceaf5';ctx.font='14px sans-serif';ctx.fillText('拖动旋转 · 滚轮缩放 · 表面为显示采样',16,24);
  }}
  canvas.addEventListener('pointerdown',e=>{{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)}});
  canvas.addEventListener('pointermove',e=>{{if(!drag)return;yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY;draw()}});
  canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('wheel',e=>{{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()}},{{passive:false}});
  document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{{const v=b.dataset.view;if(v==='front'){{yaw=0;pitch=0}}else if(v==='side'){{yaw=Math.PI/2;pitch=0}}else if(v==='top'){{yaw=0;pitch=Math.PI/2}}else{{yaw=-.65;pitch=.35;zoom=1}}draw()}}));
  document.getElementById('meshLayer').addEventListener('change',draw);document.getElementById('pointLayer').addEventListener('change',draw);document.getElementById('pointView').addEventListener('change',draw);document.getElementById('colorMode').addEventListener('change',draw);
  const select=document.getElementById('viewSelect'),pointView=document.getElementById('pointView');data.views.forEach((v,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=v.cameraId;select.appendChild(o);const p=o.cloneNode(true);pointView.appendChild(p)}});
  function paint(id,pixels) {{ const c=document.getElementById(id),x=c.getContext('2d'),im=x.createImageData(c.width,c.height);for(let i=0;i<pixels.length;i++){{im.data[i*4]=pixels[i][0];im.data[i*4+1]=pixels[i][1];im.data[i*4+2]=pixels[i][2];im.data[i*4+3]=255}}x.putImageData(im,0,0) }}
  function previews() {{const v=data.views[Number(select.value)||0];paint('rgbPreview',v.rgb);paint('depthPreview',v.depth);paint('maskPreview',v.mask)}} select.addEventListener('change',previews);previews();draw();
}})();
</script></body></html>"""


def _preview(view: RgbdView, width: int = 64, height: int = 48) -> dict[str, Any]:
    rows = np.rint(np.linspace(0, view.camera.height - 1, height)).astype(np.int64)
    cols = np.rint(np.linspace(0, view.camera.width - 1, width)).astype(np.int64)
    rgb = np.clip(view.rgb[np.ix_(rows, cols)], 0, 255).astype(np.uint8)
    depth = np.asarray(view.depth, dtype=np.float64)[np.ix_(rows, cols)]
    valid = np.isfinite(depth) & (depth > 0)
    colored_depth = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2, 98])
        scaled = np.clip((depth - low) / max(float(high - low), 1e-9), 0.0, 1.0)
        colored_depth[..., 0] = (255 * scaled).astype(np.uint8)
        colored_depth[..., 1] = (255 * (1.0 - np.abs(2.0 * scaled - 1.0))).astype(np.uint8)
        colored_depth[..., 2] = (255 * (1.0 - scaled)).astype(np.uint8)
        colored_depth[~valid] = 0
    labels = np.asarray(view.mask)[np.ix_(rows, cols)]
    mask = np.zeros((*labels.shape, 3), dtype=np.uint8)
    mask[labels == 1] = [235, 172, 126]
    mask[labels == 2] = [65, 135, 230]
    return {
        "cameraId": view.camera.camera_id,
        "rgb": rgb.reshape(-1, 3).tolist(),
        "depth": colored_depth.reshape(-1, 3).tolist(),
        "mask": mask.reshape(-1, 3).tolist(),
    }


def _sample_points(
    points: np.ndarray,
    colors: np.ndarray,
    views: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] <= limit:
        return points, colors, views
    indices = np.linspace(0, points.shape[0] - 1, limit, dtype=np.int64)
    return points[indices], colors[indices], views[indices]


def _sample_mesh(
    vertices: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if faces.shape[0] > limit:
        extent = float(np.max(np.ptp(vertices, axis=0)))
        cluster_size = max(extent / 120.0, 1e-6)
        for _ in range(16):
            clustered = _cluster_mesh(vertices, colors, faces, cluster_size)
            if 0 < clustered[2].shape[0] <= limit:
                return clustered
            cluster_size *= 1.3
    used = np.unique(faces)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    return vertices[used], colors[used], remap[faces]


def _cluster_mesh(
    vertices: np.ndarray,
    colors: np.ndarray,
    faces: np.ndarray,
    cluster_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor((vertices - vertices.min(axis=0)) / cluster_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    out_vertices = np.zeros((counts.shape[0], 3), dtype=np.float64)
    out_colors = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(out_vertices, inverse, vertices)
    np.add.at(out_colors, inverse, colors.astype(np.float64))
    out_vertices /= counts[:, None]
    out_colors = np.clip(out_colors / counts[:, None], 0, 255).astype(np.uint8)
    out_faces = inverse[faces]
    distinct = (
        (out_faces[:, 0] != out_faces[:, 1])
        & (out_faces[:, 1] != out_faces[:, 2])
        & (out_faces[:, 0] != out_faces[:, 2])
    )
    out_faces = out_faces[distinct]
    canonical = np.sort(out_faces, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    out_faces = out_faces[np.sort(first)]
    if out_faces.shape[0] == 0:
        return out_vertices[:0], out_colors[:0], out_faces
    used, remap = np.unique(out_faces, return_inverse=True)
    return out_vertices[used], out_colors[used], remap.reshape(-1, 3)


def _artifact_path(root: Path, manifest: dict[str, Any], name: str) -> Path:
    artifacts = manifest.get("artifacts", {})
    entry = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise ValueError(f"manifest does not declare artifact {name!r}")
    path = (root / entry["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest artifact escapes demo directory: {entry['path']}") from exc
    return path


def _artifact_links(root: Path, output_html: Path, manifest: dict[str, Any]) -> list[tuple[str, str]]:
    labels = {
        "hand_surface": "表面 PLY",
        "hand_fused_colored": "融合点云 PLY",
        "hand_geometry": "完整几何 NPZ",
        "surface_quality": "表面质量 JSON",
        "surface_manifest": "产物清单 JSON",
    }
    links = []
    for key, label in labels.items():
        path = root / "manifest.json" if key == "surface_manifest" else _artifact_path(root, manifest, key)
        links.append((label, Path(os.path.relpath(path, output_html.parent)).as_posix()))
    return links


def _write_text_atomic(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _card(label: Any, value: Any, detail: Any) -> str:
    return f'<div class="metric"><span class="muted">{_escape(label)}</span><strong>{_escape(value)}</strong><small>{_escape(detail)}</small></div>'


def _number(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
