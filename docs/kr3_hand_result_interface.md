# KR3 手部结果数据接口与架构预留

本文档对应 KR3：

```text
给出真值系统与 DMA 纯视觉、多模态 super-labelator 的数据接口-架构设计预留。
```

当前 KR3 聚焦手部结果，不扩展到完整物体/场景语义。统一接口优先保证三类核心结果可对齐：

- 手部 22DOF 关节角。
- 手部 21 个 3D 关键点。
- 手部 mesh 网格建模结果。

## 周报可展示文件

```text
docs/kr3_hand_result_interface.md
schemas/kr3/hand_result_schema.json
src/hand_recon/interfaces/hand_result.py
tests/test_kr3_hand_result_interface.py
```

这些文件证明 KR3 已经从口头预留变成仓库内可检查的接口预留：有架构说明、有字段 schema、有 Python adapter 出口、有自动化测试。

## 数据流架构

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

三类上游系统都只需要实现各自 adapter，adapter 的共同输出是同一个 `KR3 HandResult`。这样后续替换真实相机、DMA 模型或人工/多模态标注系统时，下游评测与训练代码不需要改字段。

## 上游系统边界

| source_system | 输入预期 | 输出责任 |
|---|---|---|
| `ground_truth_system` | 同步相机/depth/标定/时间戳、高可信手部标签或模型拟合 | 输出高置信 22DOF、21 joints、mesh、坐标系和标定来源。 |
| `dma_vision` | 纯 RGB 或多视角 RGB、相机参数、DMA 手部预测 | 输出视觉模型预测结果，并标记置信度和是否需要复核。 |
| `super_labelator` | 视觉预测、depth/几何观测、人工修正或多模态融合结果 | 输出融合后的最终标签或候选标签，保留 provenance。 |
| `synthetic_mock` | 工程内确定性 mock RGB-D 数据 | 仅用于闭环、接口与回归测试，不能冒充真实系统结果。 |

## 统一输出路径

建议每次完整流程保留：

```text
$OUT_DIR/kr3/hand_result.npz
$OUT_DIR/kr3/hand_result_report.json
```

当前仓库先实现 `hand_result.npz` 的接口预留和 mock adapter。真实系统接入后再补 `hand_result_report.json`，用于记录覆盖率、误差、人工复核比例和 mesh 质量指标。

## 坐标系和单位

默认兼容当前稳定流程：

```text
coordinate_frame = rectified_left_camera
position unit = m
rotation unit = rad
```

后续真值系统可输出 `world`，DMA 可输出 `camera`，手腕局部动作分析可输出 `wrist_local`。每行结果必须写 `coordinate_frame`，不要依赖文件名或隐式约定。

## 核心字段

| 字段 | shape | 含义 |
|---|---:|---|
| `schema_version` | scalar | 当前为 `kr3_hand_result_v0.1`。 |
| `source_system` | `(N,)` | `ground_truth_system` / `dma_vision` / `super_labelator` / `synthetic_mock`。 |
| `frame_index` | `(N,)` | 原始帧号。 |
| `timestamp_ns` | `(N,)` | 同步时间戳，单位 ns；未知时为 0。 |
| `track_id` | `(N,)` | 手实例 track id。 |
| `hand_side` | `(N,)` | `0=left hand`，`1=right hand`。 |
| `is_right` | `(N,)` | bool 形式左右手标识。 |
| `coordinate_frame` | `(N,)` | 该行 3D 输出所在坐标系。 |
| `wrist_pose_6d_m_rad` | `(N, 6)` | `[x, y, z, rx, ry, rz]`。 |
| `hand_angles_22dof_rad` | `(N, 22)` | 主接口 22DOF，单位 rad。 |
| `hand_angles_22dof_deg` | `(N, 22)` | 同上，单位 degree，便于周报/可视化。 |
| `hand_angle_names_22dof` | `(22,)` | 22 个角的固定顺序。 |
| `hand_angle_convention` | scalar | 当前为 `umetrack_compatible_v0`。 |
| `joint_names` | `(21,)` | 21 个关键点名称。 |
| `joints_3d_m` | `(N, 21, 3)` | 21 个 3D 关键点，单位 m。 |
| `valid_joint_mask` | `(N, 21)` | 每个 joint 是否有效。 |
| `mesh_vertices_m` | `(N, V, 3)` | 手部 mesh 顶点，单位 m。 |
| `mesh_faces` | `(F, 3)` | mesh 三角面片拓扑。 |
| `mesh_vertex_valid_mask` | `(N, V)` | 顶点有效 mask。 |
| `mesh_model` | `(N,)` | `umetrack` / `mano` / `hybrid` / `mock` / `unknown`。 |
| `mesh_topology_id` | scalar | mesh 拓扑版本，如 `mano_v1_778`。 |
| `frame_confidence` | `(N,)` | 当前行整体置信度。 |
| `frame_status` | `(N,)` | `ok` / `invalid` / `review_needed`。 |
| `provenance_json` | `(N,)` | 每行来源、adapter、模型版本、复核记录摘要。 |

完整 schema 见：

```text
schemas/kr3/hand_result_schema.json
```

## 22DOF 约定

主接口采用 `umetrack_compatible_v0`。公开 UmeTrack 工程里定义了每只手 21 个 landmarks 和 22 个 joint angle slots；其模型头预测 20 个 finger angles，并在 decoder 中追加 2 个 wrist angle slots。当前工程据此固定 22 维主字段，前 20 维兼容已有 `hand_angles_20dof_*`，最后 2 维预留 wrist：

```text
0  thumb_cmc_flexion
1  thumb_cmc_abduction
2  thumb_mcp_flexion
3  thumb_ip_flexion
4  index_mcp_abduction
5  index_mcp_flexion
6  index_pip_flexion
7  index_dip_flexion
8  middle_mcp_abduction
9  middle_mcp_flexion
10 middle_pip_flexion
11 middle_dip_flexion
12 ring_mcp_abduction
13 ring_mcp_flexion
14 ring_pip_flexion
15 ring_dip_flexion
16 pinky_mcp_abduction
17 pinky_mcp_flexion
18 pinky_pip_flexion
19 pinky_dip_flexion
20 wrist_flexion
21 wrist_abduction
```

参考来源：

- UmeTrack 官方仓库：https://github.com/facebookresearch/UmeTrack
- UmeTrack 官方仓库中 `lib/common/hand.py` 定义了 `NUM_LANDMARKS_PER_HAND = 21`，`NUM_JOINTS_PER_HAND = 22`。
- UmeTrack regressor：模型输出 20 个 `joint_angles`，`decode_joint_angles` 追加 2 个 wrist angle slots。

## 21 joints 约定

21 个关键点沿用当前稳定输出：

```text
0  wrist
1  thumb_cmc
2  thumb_mcp
3  thumb_ip
4  thumb_tip
5  index_mcp
6  index_pip
7  index_dip
8  index_tip
9  middle_mcp
10 middle_pip
11 middle_dip
12 middle_tip
13 ring_mcp
14 ring_pip
15 ring_dip
16 ring_tip
17 pinky_mcp
18 pinky_pip
19 pinky_dip
20 pinky_tip
```

对应字段：

```text
joints_3d_m[row, joint_id, xyz]
```

## MANO 与 UmeTrack 预留

主接口不强绑定单一手模型，而是同时预留 MANO 和 UmeTrack：

| 字段 | shape | 用途 |
|---|---:|---|
| `mano_pose_axis_angle` | `(N, 48)` | optional MANO pose。 |
| `mano_shape_betas` | `(N, 10)` | optional MANO shape。 |
| `mano_global_orient` | `(N, 3)` | optional MANO global orient。 |
| `mano_transl_m` | `(N, 3)` | optional MANO translation。 |
| `umetrack_joint_angles_rad` | `(N, 22)` | optional UmeTrack 原生 22DOF 或主字段镜像。 |

MANO mesh 常见拓扑为 778 vertices；UmeTrack 或重建网格可以使用不同顶点数。因此统一接口使用变量 `V`：

```text
mesh_vertices_m: (N, V, 3)
mesh_faces:      (F, 3)
mesh_topology_id: scalar
```

只要 `mesh_topology_id` 固定，下游可区分 `mano_v1_778`、`umetrack_runtime_model`、`mock_joint_fan_v0` 等拓扑。

## 与现有 KR1 输出的关系

当前已有：

```text
$OUT_DIR/scale/root_translation_optimized_hands.npz
```

KR3 新接口可以由它适配得到：

```text
root_translation_optimized_hands.npz
  -> build_kr3_hand_result_from_normalized(...)
  -> $OUT_DIR/kr3/hand_result.npz
```

当前 mock adapter 做的事情：

- 复用 `joints_3d_left_m` 作为 `joints_3d_m`。
- 复用 `wrist_pose_6d_left_m_rad` 作为 `wrist_pose_6d_m_rad`。
- 将现有 20DOF 角扩展为 22DOF，最后 2 个 wrist slots 先置 0。
- 从 21 joints 构造一个轻量 mock mesh，占位证明 mesh 字段链路可跑通。
- 预留 MANO/UmeTrack optional 字段，真实模型接入后由 adapter 写入。

## 验收方式

本阶段 KR3 是“接口-架构设计预留”，建议验收标准：

- 文档可说明三类上游系统如何接入统一接口。
- schema 明确 22DOF、21 joints、mesh、MANO/UmeTrack optional 字段。
- Python adapter 能从当前 mock NPZ 构造 KR3 payload。
- 测试能验证字段存在、shape 正确、NPZ 可写可读。

运行：

```bash
python3 -m pytest tests/test_kr3_hand_result_interface.py
```
