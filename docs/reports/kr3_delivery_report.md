# KR3 交付报告：真值/DMA/super-labelator 接口架构预留（历史快照）

> 本文记录接口初次预留时的三来源、object dtype 和测试数量。当前接口已额外区分
> `synthetic_mock`，并全部使用非 pickle Unicode dtype；以 schema 和自动化测试为准。

报告日期：2026-07-09

## 目标

KR3：给出真值系统与 DMA 纯视觉、多模态 super-labelator 的数据接口-架构设计预留。

本阶段聚焦手部结果接口，核心关注：

- 手部 22DOF 关节角。
- 手部 21 个 3D 关键点。
- 手部 mesh 网格建模。
- MANO 与 UmeTrack 双格式预留。

## 交付结论

KR3 已完成接口预留层面的交付。当前仓库新增统一 `KR3 HandResult` 接口，三类上游系统 `ground_truth_system`、`dma_vision`、`super_labelator` 后续都可以通过 adapter 写入同一套输出字段。mock RGB-D demo 已能生成 `outputs/mock_rgbd_demo/kr3/hand_result.npz`，并通过自动化测试校验 22DOF、21 joints、mesh 和 MANO/UmeTrack placeholder 字段。

注意：当前交付是接口/架构预留和 mock adapter，不表示真实真值系统、DMA 纯视觉系统或 super-labelator 已完成接入。

## 核心代码与文档

| 类型 | 文件 |
|---|---|
| KR3 架构说明 | `docs/kr3_hand_result_interface.md` |
| 机器可读 schema | `schemas/kr3/hand_result_schema.json` |
| Python adapter/interface | `src/hand_recon/interfaces/hand_result.py` |
| 接口导出 | `src/hand_recon/interfaces/__init__.py` |
| KR3 测试 | `tests/test_kr3_hand_result_interface.py` |
| KR3 验收脚本 | `scripts/run_kr3_checks.sh` |
| demo 接入 | `demo/run_mock_rgbd_pipeline.py` |
| mock I/O 文档接入 | `docs/mock_rgbd_io_schema.md` |
| 旧 NPZ 到 KR3 扩展说明 | `docs/root_translation_optimized_hands_npz_schema.md` |

## 架构设计

```text
真值系统 ground_truth_system
  -> ground-truth adapter
  -> KR3 HandResult

DMA 纯视觉 dma_vision
  -> DMA vision adapter
  -> KR3 HandResult

多模态 super-labelator
  -> fusion/review adapter
  -> KR3 HandResult

KR3 HandResult
  -> 质量评估 / 模型训练 / 回归测试 / 可视化 / 下游控制接口
```

统一接口的好处是：真实系统、DMA 模型和人工/多模态标注系统可以独立替换 adapter，下游评测、训练和展示代码仍然消费同一个 `hand_result.npz`。

## 运行与验收

KR3 一键验收：

```bash
bash scripts/run_kr3_checks.sh
```

当前验收结果：

```text
KR3 interface checks passed.
tests/test_kr3_hand_result_interface.py: 2 passed
```

完整回归：

```bash
python3 -m pytest
```

当前结果：

```text
3 passed
```

## 产物路径

| 产物 | 路径 |
|---|---|
| KR3 mock 输出 | `outputs/mock_rgbd_demo/kr3/hand_result.npz` |
| 多智能体协同 HTML 报告 | `outputs/reports/multi_agent_validation_report.html` |
| 多智能体协同 JSON 报告 | `outputs/reports/multi_agent_validation_report.json` |
| KR3 架构文档 | `docs/kr3_hand_result_interface.md` |
| KR3 schema | `schemas/kr3/hand_result_schema.json` |

## KR3 NPZ 字段快照

来自 `outputs/mock_rgbd_demo/kr3/hand_result.npz`：

| 字段 | shape | dtype |
|---|---:|---|
| `schema_version` | `()` | `object` |
| `source_system` | `(1,)` | `<U32` |
| `hand_angles_22dof_rad` | `(1, 22)` | `float64` |
| `joints_3d_m` | `(1, 21, 3)` | `float64` |
| `mesh_vertices_m` | `(1, 23, 3)` | `float64` |
| `mesh_faces` | `(21, 3)` | `int64` |
| `mano_pose_axis_angle` | `(1, 48)` | `float64` |
| `mano_shape_betas` | `(1, 10)` | `float64` |
| `umetrack_joint_angles_rad` | `(1, 22)` | `float64` |

## 22DOF / 21 joints / mesh 覆盖

| 能力 | 当前接口字段 | 状态 |
|---|---|---|
| 22DOF 关节角 | `hand_angles_22dof_rad`、`hand_angles_22dof_deg`、`hand_angle_names_22dof` | 已预留并测试 |
| 21 个 3D 关键点 | `joints_3d_m`、`joint_names`、`valid_joint_mask` | 已预留并测试 |
| 手部 mesh | `mesh_vertices_m`、`mesh_faces`、`mesh_vertex_valid_mask`、`mesh_topology_id` | 已预留并测试 |
| MANO | `mano_pose_axis_angle`、`mano_shape_betas`、`mano_global_orient`、`mano_transl_m` | optional 字段已预留 |
| UmeTrack | `umetrack_joint_angles_rad`、`hand_angle_convention=umetrack_compatible_v0` | optional 字段已预留 |
| 三类来源系统 | `source_system=ground_truth_system/dma_vision/super_labelator` | schema enum 已预留 |

## 多智能体协同校验证据

生成命令：

```bash
bash scripts/run_multi_agent_validation_report.sh
```

当前报告：

```text
outputs/reports/multi_agent_validation_report.html
outputs/reports/multi_agent_validation_report.json
```

报告状态为 `warning`，原因是当前 mesh 和三类系统接入仍为 mock/interface 预留，属于已知风险，不影响本阶段“接口-架构设计预留”的展示。

## 周报展示口径

可以表述为：

```text
KR3 已完成手部结果接口和架构预留：新增统一 `KR3 HandResult`，面向真值系统、DMA 纯视觉和多模态 super-labelator 三类来源。接口覆盖 22DOF 关节角、21 个 3D 关键点、mesh vertices/faces，并预留 MANO 与 UmeTrack optional 字段。当前 mock RGB-D demo 已可生成 `outputs/mock_rgbd_demo/kr3/hand_result.npz`，自动化测试通过。
```

## 剩余风险

- 当前是接口预留和 mock adapter，不是真实真值系统/DMA/super-labelator 已接入。
- 当前 mock mesh 由 21 joints 构造，真实 MANO/UmeTrack mesh 需要后续 SDK 或真实模型 adapter 写入。
- `hand_result_report.json` 仍待真实系统接入后补充，用于记录人工复核率、真实误差、mesh 质量等指标。
- 真实精度仍需补 GT/reference、MPJPE、2D 重投影误差、depth residual 和 mesh Chamfer/ICP 指标。
