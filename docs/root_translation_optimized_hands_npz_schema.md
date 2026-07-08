# root_translation_optimized_hands.npz 字段说明

本文档说明稳定流程最终输出文件 `root_translation_optimized_hands.npz` 的字段含义。

输出路径通常为：

```text
$OUT_DIR/scale/root_translation_optimized_hands.npz
```

该文件由以下脚本生成：

```text
tools/stereo_shared_scale_optimizer.py
```

当前流程坐标系约定：

```text
所有 3D 坐标默认位于 rectified left camera 坐标系下。
位置单位为 m。
旋转向量单位为 rad。
```

## 字段总表

| 字段 | shape | 含义 |
|---|---:|---|
| `K` | `(3, 3)` | rectified left camera 内参矩阵。 |
| `baseline_m` | scalar | 双目 baseline，单位 m。 |
| `frame_index` | `(N,)` | 原始视频帧号 / source frame index。 |
| `left_row` | `(N,)` | 对应 `wilor_left_raw_results.npz` 中的行号。 |
| `right_row` | `(N,)` | 对应 `wilor_right_raw_results.npz` 中的行号。 |
| `root_translation_m` | `(N, 3)` | 优化后的整手 root translation，left camera 坐标系，单位 m。 |
| `global_orient_residual_rotvec` | `(N, 3)` | global orientation residual，旋转向量，单位 rad。 |
| `global_orient_residual_deg` | `(N,)` | `global_orient_residual_rotvec` 的模长，单位 degree。 |
| `global_scale` | `(N,)` | 当前 hand side 对应的 shared scale。 |
| `scale_group` | `(N,)` | scale 分组，目前按 side 分组。通常和 `hand_side` 一致。 |
| `hand_side` | `(N,)` | 左右手标识。`0=left hand`，`1=right hand`。 |
| `is_right` | `(N,)` | bool。`True=right hand`，`False=left hand`。 |
| `wrist_pose_6d_left_m_rad` | `(N, 6)` | wrist 6DoF，格式 `[x, y, z, rx, ry, rz]`；位置单位 m，旋转向量单位 rad；坐标系为 left camera。 |
| `hand_angles_20dof_rad` | `(N, 20)` | 基于最终 21 joints 派生的 20DoF-like 几何角，单位 rad。 |
| `hand_angles_20dof_deg` | `(N, 20)` | 同上，单位 degree。 |
| `hand_angle_names_20dof` | `(20,)` | `hand_angles_20dof_*` 的角度名称。 |
| `joint_names` | `(21,)` | 21 个关节点名称。 |
| `joints_3d_left_m` | `(N, 21, 3)` | 最终恢复后的 21 个 3D joints，left camera 坐标系，单位 m。每行通过 `hand_side/is_right` 区分左右手。 |
| `valid_joint_mask` | `(N, 21)` | 每个 joint 是否有效。 |
| `anchor_mask` | `(N, 21)` | 当前优化实际使用的 anchor joint mask。 |
| `visible_anchor_mask` | `(N, 21)` | 当前无遮挡版本中等同于 `anchor_mask`，占位兼容字段。 |
| `anchor_confidence` | `(N, 21)` | 当前无遮挡版本中 anchor 有效为 1，否则 0。 |
| `anchor_confidence_source` | scalar object | 当前为 `anchor_mask_placeholder`，表示未接遮挡置信度。 |
| `frame_confidence` | `(N,)` | 当前帧/当前手的 anchor 平均置信度。无遮挡版本中较简单。 |
| `frame_status` | `(N,)` | `ok` 或 `invalid`。 |
| `scale_meta_json` | scalar object | shared scale 估计的元信息 JSON。 |
| `summary_json` | `(N,)` | 每一行结果的摘要 JSON。 |

## 左右手区分

最终结果是逐行结构：

```text
row i = 某一帧中的某一只手
```

不要仅依赖行顺序判断左右手，应使用：

```text
hand_side
is_right
```

约定：

```text
hand_side = 0 -> left hand
hand_side = 1 -> right hand
is_right = False -> left hand
is_right = True  -> right hand
```

`joints_3d_left_m[i]`、`wrist_pose_6d_left_m_rad[i]`、`hand_angles_20dof_rad[i]` 都对应同一行的同一只手。

## 21 joints 定义

`joint_names` 当前为：

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

对应 `joints_3d_left_m` 的第二维：

```text
joints_3d_left_m[row, joint_id, xyz]
```

## 20 个几何关节角定义

`hand_angles_20dof_rad` 和 `hand_angles_20dof_deg` 基于 21 joints 几何计算。

当前 `hand_angle_names_20dof` 为：

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
```

计算方式：

```text
PIP/DIP/IP flexion:
  flexion = pi - angle_between(a - b, c - b)

MCP/CMC flexion + abduction:
  先由 wrist、index/middle/pinky MCP 构建 palm frame
  abduction = atan2(local_x, local_y)
  flexion = atan2(abs(local_z), local_y)
```

注意：这些角度是 21joint 几何派生角，不等同于完整 MANO pose、20DoF IK 或 MuJoCo hand DoF。MCP/CMC abduction 的符号遵循 palm x 轴，即 index MCP -> pinky MCP。

## wrist 6DoF 定义

字段：

```text
wrist_pose_6d_left_m_rad
```

格式：

```text
[x, y, z, rx, ry, rz]
```

其中：

```text
x, y, z = joints_3d_left_m[:, wrist, :]
rx, ry, rz = global_orient_residual_rotvec
```

解释：

```text
位置是最终恢复后的 wrist joint 坐标。
旋转是当前优化器估计的 global orientation residual，不是 WiLoR/MANO 原始 global orient。
```

## 当前没有默认输出 smooth 字段

当前稳定输出不包含：

```text
joints_3d_left_m_smooth
wrist_pose_6d_left_m_rad_smooth
hand_angles_20dof_rad_smooth
hand_angles_20dof_deg_smooth
```

原因：安全时序平滑需要可靠的 `track_id`。当前最终 NPZ 尚未稳定输出 hand instance track，因此不默认写入 smooth 字段，避免不同手之间被错误平滑。

## 与中间文件的关系

最终 NPZ 依赖两个中间结果：

```text
wilor_left_raw_results.npz
wilor_right_raw_results.npz
stereo_sparse_triangulation.npz
```

来源关系：

```text
WiLoR raw result
-> left_row / right_row / 2D joints / root-relative 3D joints

stereo_sparse_triangulation
-> sparse stereo 3D observations / valid masks / hand side

stereo_shared_scale_optimizer
-> root translation / global orient residual / final 3D joints / angle fields
```
