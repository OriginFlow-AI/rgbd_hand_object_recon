# 软件集成说明

本文档说明当前工程重构后的集成方式。后续软件系统应优先调用 `hand_recon` 包内 API，不直接依赖 `scripts/` 和 `demo/`。

## 分层原则

从第一性原理看，软件集成只需要稳定契约：

```text
输入数据/配置
  -> pipeline
  -> 结构化结果
  -> adapter/schema
  -> report 或下游软件消费
```

因此目录按职责划分：

| 层 | 路径 | 职责 |
|---|---|---|
| public API | `src/hand_recon/api.py`、`src/hand_recon/__init__.py` | 软件集成入口。 |
| core library | `src/hand_recon/core/` | 点云、RGB-D、ICP、几何基础能力导出。 |
| pipelines | `src/hand_recon/pipelines/` | 可组合流程，例如 mock RGB-D、Re:InterHand ICP。 |
| adapters | `src/hand_recon/adapters/` | 外部系统/统一 hand result 的格式适配。 |
| reports | `src/hand_recon/reports/` | HTML 报告生成。 |
| io | `src/hand_recon/io/` | PLY/NPZ/JSON/scene 读写入口。 |
| scripts | `scripts/` | 薄 CLI 包装，供人工验收和批处理使用。 |
| docs | `docs/` | 说明、schema 文档、迁移文档、周报报告。 |
| tests | `tests/` | 自动化测试。 |
| outputs | `outputs/` | 运行产物，不作为源码接口。 |

## 推荐 API

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

hand_result = load_hand_result_npz(result.output_paths["kr3_hand_result"])
errors = validate_hand_result(hand_result)
assert not errors

generate_mock_visual_report(
    demo_dir=result.output_dir,
    output_html=Path("outputs/reports/hand_reconstruction_visual_report.html"),
)
```

## Re:InterHand 最佳数据报告

```python
from pathlib import Path

from hand_recon import run_reinterhand_best_data_visualization

run_reinterhand_best_data_visualization(
    data_root=Path("data/reinterhand"),
    icp_output_dir=Path("outputs/reinterhand_best_right_sequence_icp"),
    output_html=Path("outputs/reports/best_data_reinterhand_visual_report.html"),
    refresh_icp=True,
)
```

## 面向外部系统的 hand result

统一 hand result payload 的核心字段：

| 字段 | shape | 用途 |
|---|---:|---|
| `hand_angles_22dof_rad` | `(N, 22)` | 22DOF 手部关节角。 |
| `joints_3d_m` | `(N, 21, 3)` | 21 个 3D 手部关键点。 |
| `mesh_vertices_m` | `(N, V, 3)` | 手部 mesh 顶点。 |
| `mesh_faces` | `(F, 3)` | mesh 面片拓扑。 |
| `source_system` | `(N,)` | 真实系统三类值，或仅用于测试的 `synthetic_mock`。 |

机器可读 schema：

```text
schemas/kr3/hand_result_schema.json
```

## 不建议软件直接依赖的内容

- 不直接 import `scripts/*.py`。
- 不把 `outputs/` 当成源码或稳定 API。
- 不在业务代码里硬编码 `mock_data/rgbd_scene_001`。
- 不依赖 HTML 页面里的展示文字作为机器接口。

## 验收命令

```bash
python3 -m pytest
PYTHONPATH=src python3 -m hand_recon demo --config configs/mock_rgbd.json
bash scripts/run_best_data_visual_report.sh
```
