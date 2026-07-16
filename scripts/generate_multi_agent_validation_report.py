#!/usr/bin/env python3
"""Generate a multi-agent style validation report as standalone HTML."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckItem:
    name: str
    status: str
    evidence: str
    detail: str = ""


@dataclass(frozen=True)
class AgentReport:
    role: str
    status: str
    summary: str
    checks: list[CheckItem]
    risks: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-html", type=Path, default=ROOT / "outputs" / "reports" / "multi_agent_validation_report.html"
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--skip-commands", action="store_true")
    parser.add_argument("--command-timeout-sec", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command_timeout_sec <= 0:
        raise SystemExit("--command-timeout-sec must be greater than zero")
    command_results = [] if args.skip_commands else run_validation_commands(args.command_timeout_sec)
    context = collect_context(command_results)
    agents = build_agent_reports(context)
    overall_status = summarize_status(agents, command_results)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "overall_status": overall_status,
        "command_results": command_results,
        "agents": [agent_to_dict(agent) for agent in agents],
        "artifacts": context["artifacts"],
        "npz_fields": context["npz_fields"],
    }

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render_html(payload), encoding="utf-8")
    output_json = args.output_json or args.output_html.with_suffix(".json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {"status": overall_status, "html": str(args.output_html), "json": str(output_json)}, ensure_ascii=False
        )
    )
    return 0 if overall_status in {"ok", "warning"} else 1


def run_validation_commands(timeout_sec: int) -> list[dict[str, Any]]:
    commands = [
        ("KR1 mock RGB-D checks", ["bash", "scripts/run_kr1_checks.sh"]),
        ("KR3 interface checks", ["bash", "scripts/run_kr3_checks.sh"]),
        ("Full pytest suite", [sys.executable, "-m", "pytest"]),
    ]
    return [run_command(name, cmd, timeout_sec) for name, cmd in commands]


def run_command(name: str, cmd: list[str], timeout_sec: int) -> dict[str, Any]:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env={**os.environ, "PYTHON_BIN": sys.executable},
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "name": name,
            "cmd": " ".join(cmd),
            "started_at": started_at,
            "returncode": completed.returncode,
            "status": "pass" if completed.returncode == 0 else "fail",
            "stdout_tail": tail(completed.stdout),
            "stderr_tail": tail(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "cmd": " ".join(cmd),
            "started_at": started_at,
            "returncode": None,
            "status": "fail",
            "stdout_tail": tail(exc.stdout or ""),
            "stderr_tail": f"timeout after {timeout_sec}s\n{tail(exc.stderr or '')}",
        }


def collect_context(command_results: list[dict[str, Any]]) -> dict[str, Any]:
    artifact_specs = [
        ("KR3 architecture doc", ROOT / "docs" / "kr3_hand_result_interface.md"),
        ("KR1 delivery report", ROOT / "docs" / "reports" / "kr1_delivery_report.md"),
        ("KR3 delivery report", ROOT / "docs" / "reports" / "kr3_delivery_report.md"),
        ("KR3 schema", ROOT / "schemas" / "kr3" / "hand_result_schema.json"),
        ("KR3 interface code", ROOT / "src" / "hand_recon" / "interfaces" / "hand_result.py"),
        ("KR3 interface test", ROOT / "tests" / "test_kr3_hand_result_interface.py"),
        ("KR3 check script", ROOT / "scripts" / "run_kr3_checks.sh"),
        ("KR3 mock output", ROOT / "outputs" / "mock_rgbd_demo" / "kr3" / "hand_result.npz"),
        ("KR1 quality report", ROOT / "outputs" / "mock_rgbd_demo" / "quality_report.json"),
        (
            "KR1 normalized output",
            ROOT / "outputs" / "mock_rgbd_demo" / "scale" / "root_translation_optimized_hands.npz",
        ),
    ]
    artifacts = [
        {
            "name": name,
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        }
        for name, path in artifact_specs
    ]
    return {
        "command_results": command_results,
        "artifacts": artifacts,
        "schema": load_schema(),
        "npz_fields": inspect_kr3_npz(ROOT / "outputs" / "mock_rgbd_demo" / "kr3" / "hand_result.npz"),
        "doc_text": read_text(ROOT / "docs" / "kr3_hand_result_interface.md"),
        "readme_text": read_text(ROOT / "README.md"),
    }


def load_schema() -> dict[str, Any]:
    schema_path = ROOT / "schemas" / "kr3" / "hand_result_schema.json"
    if not schema_path.exists():
        return {}
    return json.loads(schema_path.read_text(encoding="utf-8"))


def inspect_kr3_npz(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    fields: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            value = data[key]
            fields[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return fields


def build_agent_reports(context: dict[str, Any]) -> list[AgentReport]:
    return [
        code_agent_report(context),
        schema_agent_report(context),
        test_agent_report(context),
        doc_agent_report(context),
        acceptance_agent_report(context),
    ]


def code_agent_report(context: dict[str, Any]) -> AgentReport:
    required_paths = [
        "src/hand_recon/interfaces/hand_result.py",
        "src/hand_recon/interfaces/__init__.py",
        "src/hand_recon/pipelines/mock_rgbd.py",
    ]
    checks = [file_check(path, f"{path} exists") for path in required_paths]
    pipeline_text = read_text(ROOT / "src" / "hand_recon" / "pipelines" / "mock_rgbd.py")
    checks.append(
        CheckItem(
            "Pipeline writes KR3 NPZ",
            "pass" if "write_kr3_hand_result_npz" in pipeline_text and "kr3_hand_result" in pipeline_text else "fail",
            "src/hand_recon/pipelines/mock_rgbd.py",
            "Pipeline should emit outputs/mock_rgbd_demo/kr3/hand_result.npz.",
        )
    )
    status = status_from_checks(checks)
    return AgentReport(
        role="代码与接口智能体",
        status=status,
        summary="检查 KR3 adapter、pipeline 接入和接口代码是否存在。",
        checks=checks,
        risks=[] if status == "pass" else ["KR3 接口代码或 demo 接入不完整。"],
    )


def schema_agent_report(context: dict[str, Any]) -> AgentReport:
    schema = context["schema"]
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})
    core_fields = {
        "hand_angles_22dof_rad": ["N", 22],
        "joints_3d_m": ["N", 21, 3],
        "mesh_vertices_m": ["N", "V", 3],
        "mesh_faces": ["F", 3],
        "mano_pose_axis_angle": ["N", 48],
        "umetrack_joint_angles_rad": ["N", 22],
    }
    checks = []
    for field, expected_shape in core_fields.items():
        actual_shape = properties.get(field, {}).get("x-npz-shape")
        checks.append(
            CheckItem(
                field,
                "pass" if field in required or field.startswith(("mano_", "umetrack_")) else "fail",
                "schemas/kr3/hand_result_schema.json",
                f"shape={actual_shape}, expected={expected_shape}",
            )
        )
        if actual_shape != expected_shape:
            checks[-1] = CheckItem(field, "fail", checks[-1].evidence, checks[-1].detail)
    checks.append(
        CheckItem(
            "source_system enum",
            "pass"
            if {"ground_truth_system", "dma_vision", "super_labelator"}.issubset(
                set(properties.get("source_system", {}).get("items", {}).get("enum", []))
            )
            else "fail",
            "schemas/kr3/hand_result_schema.json",
            "Covers all three upstream system adapters.",
        )
    )
    status = status_from_checks(checks)
    return AgentReport(
        role="Schema 智能体",
        status=status,
        summary="检查 KR3 机器可读 schema 是否覆盖 22DOF、21 joints、mesh 和来源系统。",
        checks=checks,
        risks=[] if status == "pass" else ["schema 中关键字段缺失或 shape 约定不一致。"],
    )


def test_agent_report(context: dict[str, Any]) -> AgentReport:
    checks = [
        CheckItem(
            result["name"],
            "pass" if result["status"] == "pass" else "fail",
            result["cmd"],
            f"returncode={result['returncode']}",
        )
        for result in context["command_results"]
    ]
    npz_fields = context["npz_fields"]
    expected_npz_shapes = {
        "hand_angles_22dof_rad": [1, 22],
        "joints_3d_m": [1, 21, 3],
        "mesh_faces": [21, 3],
        "mano_pose_axis_angle": [1, 48],
        "umetrack_joint_angles_rad": [1, 22],
    }
    for field, expected in expected_npz_shapes.items():
        actual = npz_fields.get(field, {}).get("shape")
        checks.append(
            CheckItem(
                f"NPZ {field}",
                "pass" if actual == expected else "fail",
                "outputs/mock_rgbd_demo/kr3/hand_result.npz",
                f"shape={actual}, expected={expected}",
            )
        )
    status = status_from_checks(checks)
    return AgentReport(
        role="测试与产物智能体",
        status=status,
        summary="执行验收命令并检查 KR3 mock NPZ 的核心字段 shape。",
        checks=checks,
        risks=[] if status == "pass" else ["测试命令失败或 KR3 输出产物字段 shape 不符合预期。"],
    )


def doc_agent_report(context: dict[str, Any]) -> AgentReport:
    doc_text = context["doc_text"]
    readme_text = context["readme_text"]
    terms = [
        ("22DOF", "22DOF" in doc_text),
        ("21 joints", "21" in doc_text and "joints_3d_m" in doc_text),
        ("mesh", "mesh_vertices_m" in doc_text and "mesh_faces" in doc_text),
        ("MANO/UmeTrack", "MANO" in doc_text and "UmeTrack" in doc_text),
        ("README entry", "KR3" in readme_text and "hand_result_schema.json" in readme_text),
    ]
    checks = [
        CheckItem(name, "pass" if ok else "fail", "docs/kr3_hand_result_interface.md / README.md") for name, ok in terms
    ]
    status = status_from_checks(checks)
    return AgentReport(
        role="文档与周报智能体",
        status=status,
        summary="检查周报可展示说明是否覆盖目标、接口字段和入口链接。",
        checks=checks,
        risks=[] if status == "pass" else ["周报说明中核心关键词或入口链接不完整。"],
    )


def acceptance_agent_report(context: dict[str, Any]) -> AgentReport:
    artifacts = context["artifacts"]
    checks = [
        CheckItem(item["name"], "pass" if item["exists"] else "fail", item["path"], f"size={item['size_bytes']} bytes")
        for item in artifacts
    ]
    warnings = []
    mesh_model = context["npz_fields"].get("mesh_model")
    if mesh_model:
        warnings.append("当前 mesh 为 mock adapter 产物，真实 MANO/UmeTrack SDK 接入后需要替换 mesh 顶点和拓扑。")
    if not context["command_results"]:
        warnings.append("本次报告使用 --skip-commands 生成，命令结果未刷新。")
    status = "warning" if warnings and status_from_checks(checks) == "pass" else status_from_checks(checks)
    return AgentReport(
        role="验收视角智能体",
        status=status,
        summary="从交付物、可复现命令和剩余风险角度判断是否可用于周报展示。",
        checks=checks,
        risks=warnings,
    )


def file_check(relative_path: str, name: str) -> CheckItem:
    path = ROOT / relative_path
    return CheckItem(name, "pass" if path.exists() else "fail", relative_path)


def status_from_checks(checks: list[CheckItem]) -> str:
    return "fail" if any(item.status == "fail" for item in checks) else "pass"


def summarize_status(agents: list[AgentReport], command_results: list[dict[str, Any]]) -> str:
    if any(agent.status == "fail" for agent in agents):
        return "fail"
    if any(result["status"] == "fail" for result in command_results):
        return "fail"
    if any(agent.status == "warning" for agent in agents):
        return "warning"
    return "ok"


def render_html(payload: dict[str, Any]) -> str:
    status = payload["overall_status"]
    title = "多智能体协同校验报告"
    agent_cards = "\n".join(render_agent(agent) for agent in payload["agents"])
    command_cards = (
        "\n".join(render_command(result) for result in payload["command_results"]) or "<p>本次跳过命令执行。</p>"
    )
    artifact_rows = "\n".join(
        f"<tr><td>{escape(item['name'])}</td><td><code>{escape(item['path'])}</code></td><td>{badge('pass' if item['exists'] else 'fail')}</td><td>{item['size_bytes']}</td></tr>"
        for item in payload["artifacts"]
    )
    npz_rows = "\n".join(
        f"<tr><td><code>{escape(key)}</code></td><td>{escape(str(value['shape']))}</td><td>{escape(value['dtype'])}</td></tr>"
        for key, value in sorted(payload["npz_fields"].items())
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #17202a; background: #f6f8fb; }}
    header {{ padding: 28px 36px; background: #102033; color: white; }}
    main {{ padding: 24px 36px 40px; max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
    .card {{ background: white; border: 1px solid #d9e0ea; border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(16, 32, 51, 0.05); }}
    .agent-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .muted {{ color: #607087; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9e0ea; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e7ecf2; text-align: left; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; }}
    pre {{ white-space: pre-wrap; background: #0f1720; color: #dce7f3; padding: 12px; border-radius: 6px; max-height: 320px; overflow: auto; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
    .pass {{ background: #def7e5; color: #176331; }}
    .fail {{ background: #ffe2e2; color: #9c1c1c; }}
    .warning {{ background: #fff3cd; color: #795300; }}
    .status {{ font-size: 32px; font-weight: 700; }}
    ul {{ padding-left: 20px; }}
    @media (max-width: 860px) {{ .summary, .agent-grid {{ grid-template-columns: 1fr; }} main, header {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div>生成时间：{escape(payload["generated_at"])}</div>
  </header>
  <main>
    <section class="summary">
      <div class="card"><div class="muted">总体状态</div><div class="status">{badge(status)}</div></div>
      <div class="card"><div class="muted">覆盖范围</div><strong>KR1 + KR3 + HTML 报告</strong></div>
      <div class="card"><div class="muted">核心 KR3 字段</div><strong>22DOF / 21 joints / mesh</strong></div>
      <div class="card"><div class="muted">主要产物</div><code>outputs/mock_rgbd_demo/kr3/hand_result.npz</code></div>
    </section>

    <h2>智能体结论</h2>
    <section class="agent-grid">{agent_cards}</section>

    <h2>验收命令</h2>
    {command_cards}

    <h2>交付物</h2>
    <table><thead><tr><th>名称</th><th>路径</th><th>状态</th><th>大小</th></tr></thead><tbody>{artifact_rows}</tbody></table>

    <h2>KR3 NPZ 字段快照</h2>
    <table><thead><tr><th>字段</th><th>shape</th><th>dtype</th></tr></thead><tbody>{npz_rows}</tbody></table>
  </main>
</body>
</html>
"""


def render_agent(agent: dict[str, Any]) -> str:
    checks = "\n".join(
        f'<li>{badge(item["status"])} <strong>{escape(item["name"])}</strong> <span class="muted">{escape(item["evidence"])}</span><br><span>{escape(item.get("detail", ""))}</span></li>'
        for item in agent["checks"]
    )
    risks = "\n".join(f"<li>{escape(risk)}</li>" for risk in agent["risks"]) or "<li>暂无新增风险。</li>"
    return f"""<div class="card">
  <h3>{escape(agent["role"])} {badge(agent["status"])}</h3>
  <p>{escape(agent["summary"])}</p>
  <ul>{checks}</ul>
  <p class="muted">风险/备注</p>
  <ul>{risks}</ul>
</div>"""


def render_command(result: dict[str, Any]) -> str:
    output = "\n".join(part for part in [result.get("stdout_tail", ""), result.get("stderr_tail", "")] if part)
    return f"""<div class="card">
  <h3>{escape(result["name"])} {badge(result["status"])}</h3>
  <p><code>{escape(result["cmd"])}</code> returncode={escape(str(result["returncode"]))}</p>
  <pre>{escape(output)}</pre>
</div>"""


def badge(status: str) -> str:
    label = {"pass": "PASS", "fail": "FAIL", "warning": "WARN", "ok": "OK"}.get(status, status.upper())
    class_name = "pass" if status in {"pass", "ok"} else "warning" if status == "warning" else "fail"
    return f'<span class="badge {class_name}">{label}</span>'


def agent_to_dict(agent: AgentReport) -> dict[str, Any]:
    return {
        "role": agent.role,
        "status": agent.status,
        "summary": agent.summary,
        "checks": [item.__dict__ for item in agent.checks],
        "risks": agent.risks,
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def tail(text: str, line_count: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-line_count:])


def escape(value: str) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
