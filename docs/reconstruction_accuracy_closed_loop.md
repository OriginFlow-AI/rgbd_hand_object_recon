# 手部表面精度闭环

本阶段的验收对象是 RGB-D 观测表面，不是关节点或关节角。

## 无 GT 闭环

稳定文件：

```text
outputs/mock_rgbd_demo/quality/surface_quality.json
```

指标分四类：

| 类别 | 指标 | 含义 |
|---|---|---|
| 观测一致性 | `source_to_surface_mean/p95_m` | 融合点是否被网格解释 |
| 表面支持 | `supported_vertex_ratio`、`multi_view_vertex_ratio` | 顶点是否有 TSDF/多视角证据 |
| 拓扑 | `component_count`、`boundary_edge_ratio`、`non_manifold_edge_count` | 表面是否碎裂或非法 |
| 规模 | vertices/faces、area、bbox、grid shape | 尺度与资源回归 |

当前 nearest-vertex 距离是轻量门禁。真实系统下一步应实现 point-to-triangle 距离和逐视角
mesh depth 回投影；后者同时报告 depth residual mean/P95 与 mask silhouette IoU。

## 有 GT 闭环

若存在高可信扫描或 mesh，应先确认坐标/单位一致，再报告：

- accuracy：predicted surface → GT distance；
- completeness：GT → predicted surface distance；
- symmetric Chamfer；
- F-score@1/2/5 mm；
- normal consistency；
- observed-only 与 completed-surface 分开统计。

不得只做 ICP 后给一个 RMSE。ICP 会消除全局位姿误差，应同时保留对齐前结果和修正量。

## 阈值来源

阈值必须来自传感器噪声、voxel size 和业务容差。mock 当前使用：

- source→surface P95 小于 `3 × voxel_size`；
- 最大连通面占比至少 0.80；
- 观测支持顶点比例至少 0.95；
- 非流形边为 0。

这些是工程回归线，不是对真实设备精度的承诺。

## 版本比较

比较版本时固定输入和配置，至少保存：

```text
manifest.json
quality/surface_quality.json
geometry/hand_geometry.npz
```

manifest 的参数、数量和 SHA-256 用于确认比较的是同一输入语义和完整产物；数值指标使用
容差比较，不对 ASCII PLY 或整份 HTML 做字节级 golden。
