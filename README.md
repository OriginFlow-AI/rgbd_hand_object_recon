# RGB-D Hand/Object Reconstruction

这是一个多视角 RGB-D 手部/物体重建工程。它把标定后的 RGB、深度和语义 mask
融合到统一世界坐标系，输出点云、mock 位姿、质量指标、规范化手部结果以及 KR3
统一接口文件。核心能力位于 `src/hand_recon/`；`demo/` 和 `scripts/` 只保留兼容的
命令行包装。

当前可稳定复现的是 deterministic mock RGB-D 闭环和点到点 ICP。mock pose、21
关节点和 mesh 是几何占位结果，不等同于真实 MANO/UmeTrack 模型精度；
Re:InterHand pilot 提供真实 MANO mesh，但不是原生 RGB-D。

## 三分钟开始

首次运行（创建 `.venv`、安装包与开发依赖、执行完整验收）：

```bash
bash scripts/bootstrap_kr1_demo.sh
```

生成 mock 重建和自包含 HTML 报告：

```bash
./run.sh
```

报告位置：

```text
outputs/reports/hand_reconstruction_visual_report.html
```

如果依赖已经安装，也可以不安装包、直接从源码运行：

```bash
PYTHONPATH=src python3 -m hand_recon demo --config configs/mock_rgbd.json
PYTHONPATH=src python3 -m hand_recon verify
python3 -m pytest -q
```

标准开发入口：

```bash
make setup
make demo
make check
```

## 核心执行流程

```text
cameras.json + rgb/depth/mask .npy
  -> 校验场景、相机参数、数组 shape 和文件边界
  -> depth 反投影到各 camera frame
  -> camera_to_world 变换到 world frame
  -> 多视角拼接和体素降采样
  -> hand/object mask 分流
  -> mock pose + 质量门禁
  -> normalized hand NPZ
  -> KR3 adapter + 安全 NPZ
  -> HTML 报告或下游软件
```

主要模块职责：

| 层 | 路径 | 职责 |
|---|---|---|
| public API | `src/hand_recon/api.py` | 软件集成的稳定入口 |
| configuration | `src/hand_recon/config.py` | JSON 配置加载和严格校验 |
| core geometry | `rgbd.py`、`reconstruction.py`、`icp.py` | 反投影、坐标变换、融合与 ICP |
| pipelines | `src/hand_recon/pipelines/` | mock 和 Re:InterHand 流程编排 |
| interfaces/adapters | `interfaces/`、`adapters/` | KR3 契约和兼容适配 |
| I/O | `src/hand_recon/io/` | 原子 JSON/NPZ 写入和安全 NPZ 加载 |
| reports | `src/hand_recon/reports/` | 自包含 HTML 报告 |

详细依赖方向与设计理由见 [docs/architecture.md](docs/architecture.md)。

## 主要输出

默认 demo 写入 `outputs/mock_rgbd_demo/`：

```text
fused_pointcloud.ply
hand_pointcloud.ply
object_pointcloud.ply
pose_output.json
quality_report.json
summary.json
scale/root_translation_optimized_hands.npz
kr3/hand_result.npz
```

质量门禁关注有效深度视角、融合点数、hand/object 点数、覆盖率、包围盒范围和
mock pose 置信度。默认配置下业务输出仍保持 4 个视角和 3335 个融合点。

NPZ 字符串字段使用普通 Unicode 数组，公共 loader 固定 `allow_pickle=False`，不会
反序列化 Python object。mock KR3 结果明确标记为 `source_system=synthetic_mock`，并
写入场景时间戳；真实系统可使用 `ground_truth_system`、`dma_vision` 或
`super_labelator`。

## 软件集成

```python
from pathlib import Path

from hand_recon import (
    generate_mock_visual_report,
    load_hand_result_npz,
    run_mock_reconstruction,
    validate_hand_result,
)

result = run_mock_reconstruction(
    scene_dir=Path("mock_data/rgbd_scene_001"),
    output_dir=Path("outputs/mock_rgbd_demo"),
    hand_side="right",
)
payload = load_hand_result_npz(result.output_paths["kr3_hand_result"])
assert validate_hand_result(payload) == []

generate_mock_visual_report(
    demo_dir=result.output_dir,
    output_html=Path("outputs/reports/hand_reconstruction_visual_report.html"),
)
```

不要让业务代码直接 import `scripts/*.py`，也不要把 `outputs/` 或 HTML 展示文本当作
稳定机器接口。更完整的集成说明见
[docs/integration/software_integration.md](docs/integration/software_integration.md)。

## 配置、日志与错误

`configs/mock_rgbd.json` 是可运行配置，而不只是示例。支持字段：

- `scene_dir`、`output_dir`
- `voxel_size_m`
- `hand_side`
- `overwrite_mock_data`
- `depth_unit`（当前仅支持 `meter`）
- `mask_labels`（当前契约为 background=0、hand=1、object=2）

未知字段、非法单位、非正体素、路径越界、坏 JSON/NPY/NPZ 和不一致的 KR3 字段会
返回可定位错误。CLI 日志级别可用全局参数控制：

```bash
PYTHONPATH=src python3 -m hand_recon --log-level DEBUG demo --config configs/mock_rgbd.json
```

## Re:InterHand 与 ICP

选择性下载/校验 pilot：

```bash
python3 scripts/prepare_reinterhand_pilot.py \
  --download-mano \
  --download-mugsy-cam-params \
  --extract-mano
```

解压前会检查上游 MD5（兼容上游清单，不代表密码学来源认证），并拒绝路径穿越、
链接和特殊文件。完整的上游下载清单位于 `third_party/reinterhand_download/`，其本地
安全包装不再使用 shell 拼接。

对任意点云运行 ICP：

```bash
python3 scripts/run_icp_registration.py \
  --inputs target_cloud.ply source_view_01.ply source_view_02.ply \
  --output-dir outputs/icp_registration \
  --voxel-size-m 0.002 \
  --distance-threshold-m 0.035
```

支持 `.ply`、`.npy` 和 `.npz`，也可用 `--init-transforms-json` 提供 4×4 初值。

## 验收与 CI

```bash
python3 -m pytest -q
bash scripts/run_kr1_checks.sh
bash scripts/run_kr3_checks.sh
PYTHONPATH=src python3 -m hand_recon verify
```

`make check` 额外运行 Ruff 和源码编译。GitHub Actions 在 Python 3.10 与 3.12 上执行
相同质量门禁。

## 数据与能力边界

- `data/`、`outputs/`、`dist/`、虚拟环境和缓存不会提交。
- 相机输入必须提供 meter 深度和 `camera_to_world`；当前没有自动单位换算。
- mock pose 是点云包围盒质心，mock joints/mesh 是确定性几何占位。
- KR3 MANO/UmeTrack optional 字段目前为接口预留，不代表 SDK 已接入。
- 点到点 ICP 依赖合理初值，低重叠、对称形状和强遮挡仍可能落入局部最优。
- Open3D/TSDF 尚未成为运行依赖。

## 文档索引

- [首次运行](START_HERE.md)
- [架构与核心流程](docs/architecture.md)
- [目录职责](docs/project_structure.md)
- [mock RGB-D I/O](docs/mock_rgbd_io_schema.md)
- [normalized hand NPZ](docs/root_translation_optimized_hands_npz_schema.md)
- [KR3 hand result](docs/kr3_hand_result_interface.md)
- [KR3 machine-readable schema](schemas/kr3/hand_result_schema.json)
- [软件集成](docs/integration/software_integration.md)
- [重建精度闭环](docs/reconstruction_accuracy_closed_loop.md)
- [手部重建 SOP](docs/hand_reconstruction_sop.md)

历史 KR 报告保留在 `docs/reports/`，用于追溯当时状态；当前命令、字段和测试结果以
本 README、schema 与自动化测试为准。
