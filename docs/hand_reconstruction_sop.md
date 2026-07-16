# 无关节点手部表面重建 SOP

## 目标

从同一时刻的标定 RGB-D 与 hand mask 恢复**观测手部表面**。本 SOP 不使用关节点、
骨架、MANO/UmeTrack 参数，也不承诺补全完全遮挡区域。

## 1. 输入验收

每个视角必须提供：

- 唯一 `camera_id` 和可解析时间戳；
- 正数 `fx/fy`、有限 `cx/cy`；
- 4×4 `camera_to_world`；
- `H×W×3` RGB、米制 `H×W` depth、整数 `H×W` mask；
- mask 约定：background=0、hand=1、object=2。

真实系统还应在进入本工程前检查最大曝光时间差。不同步的运动手部不是刚体多视角重建
问题，不能靠 ICP 静默修正。

## 2. 运行

```bash
PYTHONPATH=src .venv/bin/python -m hand_recon demo --config configs/mock_rgbd.json
```

或：

```bash
./run.sh
```

## 3. 几何链路

1. 在每视角用 hand mask 选择有效 depth。
2. pinhole 反投影，并把对应 RGB 与 view index 一起保留。
3. 用 `camera_to_world` 转到世界坐标。
4. 按 `voxel_size_m` 融合点和颜色。
5. 用融合点 bbox 加 padding 建立有上限的 TSDF ROI。
6. 将 voxel 投回各相机，仅积分 hand mask 内有效深度证据。
7. 用 marching tetrahedra 提取零等值面。
8. 删除退化/重复面，保留最大主连通面，以 TSDF 梯度定向法线。
9. 写出全量几何、质量、manifest 和离线报告。

## 4. 必查产物

```text
outputs/mock_rgbd_demo/manifest.json
outputs/mock_rgbd_demo/geometry/hand_geometry.npz
outputs/mock_rgbd_demo/geometry/hand_fused.ply
outputs/mock_rgbd_demo/geometry/hand_surface.ply
outputs/mock_rgbd_demo/quality/surface_quality.json
outputs/mock_rgbd_demo/report/index.html
```

首先检查 manifest 中所有 `exists=true` 且 `sha256` 非空；然后检查质量报告：

- `status=ok` 或明确理解每个 `warnings`；
- `source_to_surface_p95_m` 与 depth/voxel 噪声同量级；
- `component_count=1`；
- `non_manifold_edge_count=0`；
- `supported_vertex_ratio` 接近 1；
- `multi_view_vertex_ratio` 足够高。

## 5. 可视化验收

HTML 中依次检查：

1. 各相机 RGB、深度伪彩、hand mask 是否对齐；
2. 彩色观测点是否出现外参重影；
3. 网格是否贴合观测点；
4. 指缝和轮廓处是否有跨深度飞面；
5. 边界是否来自真实遮挡，而不是大块数据缺失。

报告只嵌入采样数据；最终数值分析使用 PLY/NPZ 全量数据。

## 6. 失败处理

- `at least three XYZ points`：hand mask 内没有足够有效 depth。
- `max_voxel_count`：通常是单位错误或离群点扩大 bbox，先修输入。
- `no supported zero crossing`：检查相机方向、mask、depth 与 truncation。
- `surface_is_fragmented`：优先检查标定、同步、mask 小碎片和深度断层。
- `source_to_surface...above_threshold`：检查 voxel/truncation 是否匹配传感器噪声。
- `surface_has_non_manifold_edges`：不得作为正常结果下发，应检查提取/清理或坏输入。

## 7. 真实数据接入顺序

1. 提交一个许可允许的小型、同步、多视角 RGB-D fixture。
2. 用同一 manifest 适配器读取，不修改几何核心。
3. 根据真实 depth 噪声重新标定 voxel、truncation 和质量阈值。
4. 增加逐视角 mesh 深度回投影与 silhouette IoU。
5. 只有标定残差证明确有必要时，增加受限 pose-graph refinement。
6. 若后续需要形状补全，独立输出 `completed_surface`，不得覆盖 `observed_surface`。
