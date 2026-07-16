# RGB-D 手部表面重建

本工程的核心目的只有一个：把同一时刻、已经标定并带手部分割的多视角 RGB-D
观测，恢复成可追溯、可量化、可查看的手部三维表面。

当前默认主链不依赖关节点、骨架、MANO 或 UmeTrack。它直接融合深度证据，输出彩色
点云、TSDF 三角网格、质量报告、版本化产物清单和离线交互式 HTML。旧 KR3
关节点/角度文件继续生成只是为了兼容既有消费者，不参与表面重建和主报告。

## 快速运行

```bash
make setup
./run.sh
```

完成后打开：

```text
outputs/mock_rgbd_demo/report/index.html
```

只运行代码和数值验收：

```bash
make check PYTHON=.venv/bin/python
```

## 第一性原理

手部表面重建不是“生成一个看起来像手的模型”，而是让输出几何能够解释输入观测：

```text
标定 RGB / meter depth / hand mask
  → 输入、单位、路径和相机矩阵校验
  → 各视角反投影为 world-frame 彩色手部点云
  → 体素融合形成观测点云
  → 在手部 ROI 中投影式 TSDF 融合
  → marching tetrahedra 提取零等值面
  → 退化面、重复面和小碎片清理，TSDF 梯度定向法线
  → 观测支持、距离、连通性、边界和非流形质量门禁
  → PLY / NPZ / JSON / HTML
```

没有被相机看到的区域无法仅凭 RGB-D 确定。本工程把输出明确标记为
`observed_not_completed`：它宁可保留真实边界和质量警告，也不使用关节点扇形面或
隐式形状先验伪装成“完整扫描”。这里的“更完整”指更多视角支持、更连续的实测表面
以及更小的观测残差。

## 主要产物

默认目录 `outputs/mock_rgbd_demo/`：

```text
manifest.json                         # 唯一产物交接契约、相对路径和 SHA-256
geometry/
  hand_geometry.npz                  # raw/fused 点色、view id、mesh、法线
  hand_fused.ply                     # 全量彩色融合点云
  hand_surface.ply                   # 顶点、法线、颜色和三角面
views/
  view_00_hand.ply ...               # 每视角 world-frame 彩色手部点云
quality/
  surface_quality.json               # 表面质量门禁
report/
  index.html                         # 无 CDN、可旋转/缩放/切层的自包含报告
fused_pointcloud.ply                 # 兼容路径
hand_pointcloud.ply                  # 兼容路径，现已保留 RGB
object_pointcloud.ply                # 兼容路径，现已保留 RGB
summary.json
scale/root_translation_optimized_hands.npz  # 兼容旁路
kr3/hand_result.npz                        # 兼容旁路
```

`hand_geometry.npz` 只含数值、布尔和 Unicode 数组，loader 固定
`allow_pickle=False`。`manifest.json` 中所有产物路径都限制在输出目录内，并记录大小与
SHA-256。

## 架构边界

| 层 | 位置 | 职责 |
|---|---|---|
| 稳定契约 | `domain.py` | `TriangleMesh`、`TsdfVolume`、`SurfaceRunResult` 和不变量 |
| 观测几何 | `rgbd.py`、`reconstruction.py` | 场景校验、反投影、颜色/视角身份和体素融合 |
| 表面融合 | `fusion/tsdf.py` | 有界、带 mask 的 projective TSDF |
| 表面处理 | `surface/mesh.py`、`surface/quality.py` | 网格提取、清理、法线和质量门禁 |
| 用例编排 | `pipelines/hand_surface.py` | 联结稳定几何步骤，不承载 I/O 细节 |
| 产物 I/O | `io/geometry.py`、`io/artifacts.py` | 原子网格写出、NPZ 和 manifest |
| 展示 | `visualization/surface_report.py` | 只消费落盘契约，不读取关节点结果 |
| 兼容层 | `normalized_output.py`、`interfaces/`、`adapters/` | 保留旧 KR3 行为，不污染新主链 |

详细设计、失败边界和多智能体协作协议见
[架构与核心流程](docs/architecture.md) 和 [目录职责](docs/project_structure.md)。

## 软件调用

```python
from pathlib import Path

from hand_recon import load_surface_geometry_npz, run_mock_reconstruction

result = run_mock_reconstruction(
    scene_dir=Path("mock_data/rgbd_scene_001"),
    output_dir=Path("outputs/mock_rgbd_demo"),
)
assert result.surface_result.ok

geometry = load_surface_geometry_npz(result.output_paths["hand_geometry"])
vertices = geometry["mesh_vertices_m"]
faces = geometry["mesh_faces"]
```

公共入口定义在 `hand_recon.api`。脚本只做参数解析，不应被业务代码 import。

## 质量判据

当前无 GT 闭环至少检查：

- 源点到表面、表面到源点的 mean/P95 距离；
- TSDF 观测体素数、表面顶点/三角面数；
- 单一主连通面、边界边比例和非流形边数量；
- 有观测支持的顶点比例、至少双视角支持的顶点比例；
- 表面积、包围盒和每视角贡献。

mock 默认结果约为 2.8 万顶点、5.46 万三角面、非流形边 0、源点到表面 P95
约 3.5 mm。数值测试使用容差而不是对整份网格做脆弱快照。

## 能力边界

- 当前提交验证的是确定性 mock RGB-D；尚缺一个可提交的小型真实同步 RGB-D fixture。
- 外参被视为可信标定；尚未实现带门禁的多视角 pose-graph refinement。
- TSDF 只恢复观测表面，不推断完全遮挡区域，也不做关节点定位。
- 真实手在多相机曝光期间运动会产生重影，必须由采集同步规范控制。
- mask 边缘和深度噪声是表面精度上限；真实设备阈值应按噪声模型重新标定。
- Re:InterHand MANO mesh/跨帧 ICP 是参考实验，不是本 RGB-D 主链的验收证据。

## 文档索引

- [第一次运行](START_HERE.md)
- [架构与第一性原理](docs/architecture.md)
- [目录职责](docs/project_structure.md)
- [RGB-D I/O 与产物](docs/mock_rgbd_io_schema.md)
- [手部表面重建 SOP](docs/hand_reconstruction_sop.md)
- [表面精度闭环](docs/reconstruction_accuracy_closed_loop.md)
- [多智能体协作协议](docs/multi_agent_closed_loop_task.md)
- [兼容 KR3 接口](docs/kr3_hand_result_interface.md)
- [兼容 KR3 machine-readable schema](schemas/kr3/hand_result_schema.json)
