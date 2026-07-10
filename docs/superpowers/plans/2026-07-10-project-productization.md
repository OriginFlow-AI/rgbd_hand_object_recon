# RGB-D 手物重建工程产品化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有演示工程升级为可安装、可运行、可集成、可量化且能输出时间对齐与 pose 检出率可视化结果的标准 Python 项目。

**Architecture:** 保留现有 `hand_recon` 公共 API 和已验证几何能力，在包内新增独立 `analysis` 子域，并以统一 CLI 编排 demo、分析和验收。分析核心使用纯函数，数据适配、报告渲染和命令行保持单向依赖；mock、真实标注可用率和真实模型检测率使用不同来源标签。

**Tech Stack:** Python 3.10+、NumPy、SciPy、pytest、setuptools、标准库 argparse/csv/json/html、内联 SVG、GitHub Actions。

---

## 文件结构

本计划新增或修改：

```text
pyproject.toml
Makefile
run.sh
START_HERE.md
.github/workflows/ci.yml
src/hand_recon/
  __main__.py
  cli.py
  time_utils.py
  analysis/
    __init__.py
    models.py
    metrics.py
    sources.py
    report.py
    service.py
tests/
  test_cli.py
  test_analysis_metrics.py
  test_analysis_sources.py
  test_analysis_report.py
  test_hand_result_security.py
docs/
  architecture.md
  data-analysis.md
  data-collection.md
```

现有 `api.py`、`interfaces/hand_result.py`、schema、mock pipeline、README 和启动脚本只做兼容式修改，不迁移真实数据，不删除 `data/`、`outputs/` 或 `dist/`。

### Task 1: 标准打包和统一 CLI

**Files:**
- Create: `pyproject.toml`
- Create: `src/hand_recon/cli.py`
- Create: `src/hand_recon/__main__.py`
- Create: `tests/test_cli.py`
- Modify: `src/hand_recon/__init__.py`

- [ ] **Step 1: 写 CLI 失败测试**

```python
def test_module_cli_exposes_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hand_recon", "--help"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "demo" in result.stdout
    assert "analyze" in result.stdout
```

- [ ] **Step 2: 运行测试并确认缺少 `hand_recon.__main__` 而失败**

Run: `python3 -m pytest tests/test_cli.py -q`

Expected: FAIL，stderr 包含 `No module named hand_recon.__main__`。

- [ ] **Step 3: 写最小打包和 CLI 实现**

`pyproject.toml` 必须声明 setuptools 的 `src` 布局、运行依赖、`dev` extra、pytest 配置和 console script：

```toml
[project.scripts]
rgbd-hand-recon = "hand_recon.cli:main"
```

`cli.py` 使用 `build_parser()` 创建 `demo`、`analyze`、`verify` 三个子命令；尚未接线的命令返回清晰错误，不静默成功。`__main__.py` 只调用 `raise SystemExit(main())`。

- [ ] **Step 4: 验证模块 CLI 与可编辑安装**

Run: `python3 -m pytest tests/test_cli.py -q && python3 -m pip install -e . --no-deps && python3 -c 'import hand_recon; print(hand_recon.__version__)'`

Expected: 测试通过，导入输出 `0.1.0`。

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml src/hand_recon/__init__.py src/hand_recon/cli.py src/hand_recon/__main__.py tests/test_cli.py
git commit -m "feat: 建立标准包与统一命令行"
```

### Task 2: 时间对齐和 pose 检出率纯函数

**Files:**
- Create: `src/hand_recon/analysis/__init__.py`
- Create: `src/hand_recon/analysis/models.py`
- Create: `src/hand_recon/analysis/metrics.py`
- Create: `tests/test_analysis_metrics.py`

- [ ] **Step 1: 写时间对齐失败测试**

```python
def test_alignment_separates_aligned_misaligned_and_not_evaluable() -> None:
    records = [
        FrameRecord("s", "1", "a", 1_000_000, "hardware", "test"),
        FrameRecord("s", "1", "b", 2_000_000, "hardware", "test"),
        FrameRecord("s", "2", "a", 10_000_000, "hardware", "test"),
        FrameRecord("s", "2", "b", 20_000_000, "hardware", "test"),
        FrameRecord("s", "3", "a", None, "hardware", "test"),
        FrameRecord("s", "3", "b", 30_000_000, "hardware", "test"),
    ]
    result = analyze_alignment(records, tolerance_ns=5_000_000, expected_camera_ids={"a", "b"})
    assert result.summary["aligned_groups"] == 1
    assert result.summary["misaligned_groups"] == 1
    assert result.summary["not_evaluable_groups"] == 1
    assert result.summary["alignment_rate"] == 0.5
```

- [ ] **Step 2: 写 pose 检出率失败测试**

```python
def test_pose_detection_uses_explicit_denominator_and_reasons() -> None:
    records = [
        PoseRecord("s", "1", "fused", True, "ok", 0.9, 21, "detector"),
        PoseRecord("s", "2", "fused", True, "ok", 0.2, 21, "detector"),
        PoseRecord("s", "3", "fused", True, "missing", None, 0, "detector"),
        PoseRecord("s", "4", "fused", False, "not_run", None, None, "detector"),
    ]
    result = analyze_pose_detection(records, confidence_threshold=0.5, min_valid_joints=21)
    assert result.summary["eligible_records"] == 3
    assert result.summary["detected_records"] == 1
    assert result.summary["detection_rate"] == pytest.approx(1 / 3)
    assert result.summary["failure_reasons"] == {"low_confidence": 1, "missing": 1}
```

- [ ] **Step 3: 运行两个测试并确认模块不存在而失败**

Run: `python3 -m pytest tests/test_analysis_metrics.py -q`

Expected: FAIL，无法导入 `hand_recon.analysis`。

- [ ] **Step 4: 实现数据模型、校验和指标**

`FrameRecord` 拒绝空序列/帧/相机；时间戳只接受非负整数或 `None`。`PoseRecord` 校验置信度范围和关节点数量。`analyze_alignment` 按 `(sequence_id, frame_id)` 分组，缺相机、缺时间戳、时钟域不一致均标记 `not_evaluable`。`analyze_pose_detection` 仅把 `expected=True` 放入分母，并输出逐行判定原因。

- [ ] **Step 5: 运行测试和边界检查**

Run: `python3 -m pytest tests/test_analysis_metrics.py -q`

Expected: 所有指标测试通过。

- [ ] **Step 6: 提交**

```bash
git add src/hand_recon/analysis tests/test_analysis_metrics.py
git commit -m "feat: 实现时间对齐与姿态检出率指标"
```

### Task 3: Mock、JSONL 和 Re:InterHand 数据适配

**Files:**
- Create: `src/hand_recon/time_utils.py`
- Create: `src/hand_recon/analysis/sources.py`
- Create: `tests/test_analysis_sources.py`

- [ ] **Step 1: 写来源适配失败测试**

测试必须覆盖：ISO-8601 转纳秒、四个 mock 相机记录、一个 fused mock pose、JSONL `record_type` 分流、curated/original Re:InterHand 覆盖率和缺失区间。

```python
assert parse_iso8601_ns("1970-01-01T00:00:01+00:00") == 1_000_000_000
assert len(load_mock_frame_records(scene_dir)) == 4
assert load_mock_pose_records(pose_path, hand_result_path)[0].view_id == "fused"
assert coverage["curated"]["slot_availability_rate"] == 1.0
assert coverage["original"]["missing_frame_count_per_side"] == 1
```

- [ ] **Step 2: 运行测试并确认缺少 source adapter 而失败**

Run: `python3 -m pytest tests/test_analysis_sources.py -q`

Expected: FAIL，无法导入 `hand_recon.analysis.sources`。

- [ ] **Step 3: 实现只读适配器**

适配器不得写入输入目录。`load_mock_pose_records` 只产生一个 `view_id="fused"` 记录，source 写为 `synthetic_mock_bbox_centroid`。Re:InterHand 适配器从 frame list、params 和 mesh 文件名做集合运算，结果命名为 `annotation_availability`，不得写成 detection rate。连续缺失 frame id 按现有采样序列合并成区间。

- [ ] **Step 4: 验证小型 fixture 和本地实际数据**

Run: `python3 -m pytest tests/test_analysis_sources.py -q`

Run: `python3 -c 'from pathlib import Path; from hand_recon.analysis.sources import inspect_reinterhand_capture; print(inspect_reinterhand_capture(Path("data/reinterhand/m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands"))["curated"])'`

Expected: curated 期望帧 10,531、左右手 slot 可用率 1.0。

- [ ] **Step 5: 提交**

```bash
git add src/hand_recon/time_utils.py src/hand_recon/analysis/sources.py tests/test_analysis_sources.py
git commit -m "feat: 接入本地数据质量分析来源"
```

### Task 4: 自包含分析报告和 CLI 接线

**Files:**
- Create: `src/hand_recon/analysis/report.py`
- Create: `src/hand_recon/analysis/service.py`
- Create: `tests/test_analysis_report.py`
- Modify: `src/hand_recon/analysis/__init__.py`
- Modify: `src/hand_recon/cli.py`
- Modify: `src/hand_recon/api.py`
- Modify: `src/hand_recon/__init__.py`

- [ ] **Step 1: 写报告失败测试**

```python
def test_project_analysis_writes_machine_and_human_outputs(tmp_path: Path) -> None:
    result = analyze_project_data(
        scene_dir=fixture_scene,
        demo_output_dir=fixture_output,
        reinterhand_capture_dir=fixture_capture,
        output_dir=tmp_path,
        tolerance_ms=5.0,
        confidence_threshold=0.5,
        min_valid_joints=21,
    )
    assert result.summary_path.exists()
    assert result.alignment_csv_path.exists()
    assert result.pose_csv_path.exists()
    html = result.report_path.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "not_evaluable" in html
    assert "文件可用率不等于模型检出率" in html
```

- [ ] **Step 2: 运行测试并确认 service/report 不存在而失败**

Run: `python3 -m pytest tests/test_analysis_report.py -q`

Expected: FAIL，无法导入分析服务。

- [ ] **Step 3: 实现 CSV、JSON、HTML 与图形**

HTML 只使用本地 CSS 和内联 SVG，至少绘制时间偏差点图、skew 柱图、pose 状态柱图、Re:InterHand 覆盖条和缺失区间。无真实时间戳时显示原因，不能显示 0 ms 代替未知。

- [ ] **Step 4: 接入 API 与 CLI**

`python -m hand_recon analyze` 支持 `--scene-dir`、`--demo-output-dir`、`--reinterhand-capture-dir`、`--output-dir`、`--tolerance-ms`、`--confidence-threshold` 和 `--min-valid-joints`，成功后打印四个产物绝对路径。

- [ ] **Step 5: 运行报告和 CLI 测试**

Run: `python3 -m pytest tests/test_analysis_report.py tests/test_cli.py -q`

Expected: 测试通过，HTML 包含 SVG 和能力边界说明。

- [ ] **Step 6: 提交**

```bash
git add src/hand_recon/analysis src/hand_recon/cli.py src/hand_recon/api.py src/hand_recon/__init__.py tests/test_analysis_report.py tests/test_cli.py
git commit -m "feat: 生成数据质量可视化报告"
```

### Task 5: 修正 mock 来源、时间戳与 NPZ 安全语义

**Files:**
- Create: `tests/test_hand_result_security.py`
- Modify: `src/hand_recon/interfaces/hand_result.py`
- Modify: `src/hand_recon/adapters/hand_result.py`
- Modify: `src/hand_recon/pipelines/mock_rgbd.py`
- Modify: `src/hand_recon/api.py`
- Modify: `schemas/kr3/hand_result_schema.json`
- Modify: `tests/test_kr3_hand_result_interface.py`
- Modify: `tests/test_mock_rgbd_pipeline.py`

- [ ] **Step 1: 写语义和安全失败测试**

```python
def test_mock_pipeline_marks_synthetic_source_and_real_timestamp(tmp_path: Path) -> None:
    result = run_mock_reconstruction(scene_dir=tmp_path / "scene", output_dir=tmp_path / "out")
    payload = load_hand_result_npz(result.output_paths["kr3_hand_result"])
    assert payload["source_system"].tolist() == ["synthetic_mock"]
    assert payload["timestamp_ns"].tolist()[0] > 0
    assert all(value.dtype != object for value in payload.values())

def test_public_loader_rejects_pickle_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, payload=np.array({"secret": 1}, dtype=object))
    with pytest.raises(ValueError, match="object arrays"):
        load_hand_result_npz(path)
```

- [ ] **Step 2: 运行测试并确认旧行为失败**

Run: `python3 -m pytest tests/test_hand_result_security.py -q`

Expected: FAIL，旧 source 为 super_labelator、timestamp 为 0 或 loader 允许 pickle。

- [ ] **Step 3: 实现最小修正**

把 `synthetic_mock` 加入 Python 和 JSON schema source 枚举；mock pipeline 传入 scene ISO 时间戳解析后的 ns；schema/version/convention/topology/provenance 使用 Unicode dtype；公共 loader 固定 `allow_pickle=False` 并把 object-array 错误改成可行动异常。

- [ ] **Step 4: 运行契约与回归测试**

Run: `python3 -m pytest tests/test_hand_result_security.py tests/test_kr3_hand_result_interface.py tests/test_mock_rgbd_pipeline.py tests/test_public_api.py -q`

Expected: 所有测试通过，生成 NPZ 可在禁止 pickle 时读取。

- [ ] **Step 5: 提交**

```bash
git add src/hand_recon schemas/kr3/hand_result_schema.json tests
git commit -m "fix: 明确合成数据来源并收紧结果加载"
```

### Task 6: 一键运行、开发任务和 CI

**Files:**
- Create: `run.sh`
- Create: `Makefile`
- Create: `.github/workflows/ci.yml`
- Modify: `scripts/bootstrap_kr1_demo.sh`
- Modify: `scripts/run_kr1_checks.sh`
- Modify: `.gitignore`

- [ ] **Step 1: 写 shell 入口验收测试**

在 `tests/test_cli.py` 增加 `bash -n run.sh`、`make help` 和 `python -m hand_recon verify` 的 subprocess 断言，并先运行确认文件缺失而失败。

- [ ] **Step 2: 实现入口**

`./run.sh` 使用 `.venv/bin/python`（存在时）或 `python3`，设置 `PYTHONPATH=src`，依次运行 demo 和 analyze，最后只打印 HTML 路径。`Makefile` 提供 `help/setup/demo/analyze/test/lint/check`；`setup` 执行 `pip install -e '.[dev]'`。CI 在 Python 3.10 和 3.12 运行 `make check`。

- [ ] **Step 3: 统一旧验收入口**

bootstrap 安装本包而非只装 requirements；KR1 检查改为调用统一 CLI，并保留精度脚本和 ICP selftest。

- [ ] **Step 4: 验证**

Run: `bash -n run.sh scripts/*.sh && make help && python3 -m pytest tests/test_cli.py -q`

Expected: shell 语法和入口测试通过。

- [ ] **Step 5: 提交**

```bash
git add run.sh Makefile .github/workflows/ci.yml scripts/bootstrap_kr1_demo.sh scripts/run_kr1_checks.sh .gitignore tests/test_cli.py
git commit -m "build: 统一启动验收与持续集成"
```

### Task 7: 中文文档体系与采集方案

**Files:**
- Create: `START_HERE.md`
- Create: `docs/architecture.md`
- Create: `docs/data-analysis.md`
- Create: `docs/data-collection.md`
- Modify: `README.md`
- Modify: `docs/project_structure.md`
- Modify: `docs/integration/software_integration.md`
- Modify: `docs/mock_rgbd_io_schema.md`

- [ ] **Step 1: 重写 README 和开始文档**

README 按“项目是什么→三分钟运行→输出→能力边界→架构→软件集成→文档索引”排版。`START_HERE.md` 只保留 setup、run、analyze、check、常见问题和每日三行收口。

- [ ] **Step 2: 写架构与分析文档**

架构文档解释输入→融合→pose→契约→分析→报告的数据流和各目录依赖。分析文档给出分母、公式、阈值、mock 0 ms 的限制、curated 100% 与 original 98.929947% 的差异和复现命令。

- [ ] **Step 3: 写采集方案与公开先例**

采集文档必须包含硬件触发/时钟域、内外参与时间标定、不可变 raw manifest、在线质量门禁、标注/真值、数据划分、验收阈值和安全解包。引用 Re:InterHand、DexYCB、HOT3D 和 Intel RealSense 官方资料，并解释它们可借鉴的部分。

- [ ] **Step 4: 文档一致性检查**

Run: `rg -n '3 passed|source_system=.super_labelator.|真实检出率 100%|Gitee 初始化' README.md START_HERE.md docs`

Expected: 当前使用文档没有过期测试数或把 mock/标注可用率冒充真实检测率的表述；历史报告中的旧描述明确标记为历史。

- [ ] **Step 5: 提交**

```bash
git add README.md START_HERE.md docs
git commit -m "docs: 重构中文使用学习与采集文档"
```

### Task 8: 生成实际结果并完成多角色复核

**Files:**
- Generated, ignored: `outputs/analysis/summary.json`
- Generated, ignored: `outputs/analysis/alignment_groups.csv`
- Generated, ignored: `outputs/analysis/pose_records.csv`
- Generated, ignored: `outputs/analysis/report.html`

- [ ] **Step 1: 运行最简单效果入口**

Run: `./run.sh`

Expected: demo 状态 ok，报告写入 `outputs/analysis/report.html`。

- [ ] **Step 2: 运行全量质量门禁**

Run: `python3 -m pytest -q`

Run: `python3 -m compileall -q src tests`

Run: `python3 -m hand_recon --help && python3 -m hand_recon verify`

Run: `git diff --check`

Expected: 全部退出码为 0。

- [ ] **Step 3: 核对实际指标**

使用 `jq` 检查 mock 名义 skew、pose source/分母、Re:InterHand curated/original 可用率和 `limitations`。真实时间戳缺失必须是 `not_evaluable`，不能输出伪造真实对齐率。

- [ ] **Step 4: 两阶段独立审查**

先由需求符合性审查员逐条比对设计与计划，再由代码质量审查员检查 API、异常、安全、可维护性和测试；修复所有 Critical/Important 问题后重复全量门禁。

### Task 9: 重新初始化 Git、首次提交与安全上传判断

**Files:**
- Delete only: `.git/`
- Recreate: `.git/`

- [ ] **Step 1: 保存远程并审计首次提交清单**

Run: `git remote get-url origin`

Run: `git status --short && git ls-files --others --exclude-standard`

Expected: 远程为 `git@github.com:OriginFlow-AI/rgbd_hand_object_recon.git`，清单不含 data/outputs/dist/cache/venv。

- [ ] **Step 2: 删除旧历史并初始化 main**

```bash
rm -rf .git
git init -b main
git remote add origin git@github.com:OriginFlow-AI/rgbd_hand_object_recon.git
```

- [ ] **Step 3: 暂存后复核大文件与忽略边界**

Run: `git add . && git status --short && git diff --cached --check && git diff --cached --stat`

Expected: 不包含真实数据、输出、dist、缓存、凭据或大于 1 MiB 的意外文件。

- [ ] **Step 4: 再次验证并做中文首次提交**

Run: `python3 -m pytest -q && python3 -m compileall -q src tests`

```bash
git commit -m "初始化：规范化 RGB-D 手物重建与数据质量分析工程"
```

- [ ] **Step 5: 判断上传方式**

Run: `git ls-remote --heads origin main`

如果远程 `main` 已有旧历史，新初始提交与其无共同祖先；本轮不静默 force-push。优先推送新分支 `project-reinitialized-20260710` 供核对，或在用户明确授权覆盖后使用 `--force-with-lease`。如果远程为空，则正常 `git push -u origin main`。
