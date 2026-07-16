# 从第一性原理设计手部表面重建

## 1. 工程目的

输入是多个相机在同一时刻对同一只手的 RGB、米制 depth、hand mask、内参和
`camera_to_world`；输出是能够解释这些观测的世界坐标手部表面及其可信度。

本阶段明确排除关节点定位、骨架拟合和参数手模型。它们是姿态/语义推断问题，不是从
深度恢复表面的必要条件。

## 2. 五条不可破坏的原则

1. **证据优先**：每个几何产物都必须能追溯到输入视角。
2. **坐标与单位显式**：全链以 meter 和声明的 world frame 工作，不猜测 mm/m。
3. **不可观测不伪装**：输出分为观测表面与未来可选的推断补全；当前只实现前者。
4. **质量回到观测**：点数不是精度，必须测距离、支持视角、连通性和拓扑。
5. **资源有上限**：TSDF ROI 来自手部观测 bbox，并由 `max_voxel_count` 防止异常输入耗尽内存。

## 3. 最小最优数据流

```text
RgbdScene
  → validate_scene
  → backproject masked depth + preserve RGB/view id
  → calibrated world-frame point observations
  → colored voxel fusion
  → bounded projective TSDF
  → marching tetrahedra zero surface
  → deduplicate / remove degenerate / keep main component
  → orient by TSDF gradient / vertex normals / nearest observed color
  → surface quality gates
  → ArtifactManifest
  → offline interactive report
```

相机已经标定时不默认运行 ICP。盲目 ICP 可能把分割错误或非同步运动“优化”成错误
几何。未来只有在重叠率充分、校正量受限并且残差显著改善时，才允许 pose-graph
refinement。

## 4. 核心契约

`domain.py` 固定三个边界：

- `TsdfVolume`：origin、voxel size、TSDF values、weights、view support。
- `TriangleMesh`：米制 vertices、faces、normals、RGB 和 coordinate frame；构造时验证 shape、有限值和索引。
- `SurfaceRunResult`：status、mesh、quality 和参数/provenance。

稳定落盘契约是 `manifest.json`。它记录 schema、语义、参数、数量、相对路径、文件大小和
SHA-256。报告、测试和多智能体模块只通过它交接。

## 5. TSDF 与网格

ROI 为融合手部点云 bbox 加固定 padding。每个 voxel 投影到所有相机：仅当投影落在
hand mask、depth 有效且位于截断带后方范围内时积分：

```text
signed_distance = observed_depth - voxel_camera_z
tsdf = clip(signed_distance / truncation, -1, 1)
```

零等值面由每个体素立方体的六个四面体提取。清理步骤删除重复/退化三角形，只保留最大
面连通分量，以 TSDF 梯度统一面方向，再按面积累积顶点法线。颜色取最近融合观测点，
不会由关节点生成。

采用 marching tetrahedra 是为了在 NumPy/SciPy 现有依赖内得到确定性表面；代价是
三角面数量较多。后续若真实数据规模要求更高，可增加 Open3D 后端，但必须实现同一
`SurfaceRunResult` 契约和相同质量门禁。

## 6. 质量状态

`surface/quality.py` 输出：

- source→surface 与 surface→source mean/P95；
- 顶点、三角面、表面积、bbox；
- 连通分量、最大分量占比；
- boundary/non-manifold edge；
- 有 TSDF 支持和多视角支持的顶点比例。

状态：

- `ok`：所有当前门禁通过；
- `partial`：得到可用表面，但至少一项支持或拓扑指标需检查；
- 异常：输入不足、资源越界或没有零交叉，不生成假成功结果。

## 7. I/O 与失败边界

- 场景路径必须在 scene 目录内，NPY/NPZ 禁止 pickle object。
- JSON/NPZ/mesh PLY 使用同目录临时文件并原子替换。
- manifest 中的相对路径再次做 output-root containment 校验。
- 三角面索引越界、NaN、无有效 depth、空 hand mask、TSDF 无零交叉均显式失败。
- HTML 无外网依赖，只嵌入固定上限显示采样；科研产物始终保存全量数据。

## 8. 兼容迁移

原有 `normalized_output`、KR3 22DoF/joints 和 mock pose 不删除，以免破坏既有消费者；
但它们是 `compatibility` 旁路：

- 新表面不从 joints 构造；
- 新报告不读取 `kr3/hand_result.npz`；
- 删除 KR3 和 normalized 文件后仍能生成表面报告；
- manifest 明确声明兼容产物不是主链输入。

## 9. 多智能体协作

运行时重建是确定性数值程序，不需要 LLM agent。多智能体只用于研发，并按失败边界分工：

| 角色 | 独占范围 | 验收 |
|---|---|---|
| 主智能体 | 公共契约、pipeline、集成、最终提交 | 全量测试、端到端、输出审计 |
| 架构智能体 | 目的、数据流、迁移和依赖审计 | 架构缺口与边界清单 |
| 表面智能体 | TSDF、mesh、质量指标审计 | 数值方案、异常和测试矩阵 |
| 可视化智能体 | 产物、HTML、交互和报告审计 | 不依赖关节点的验收方案 |

协作规则：主智能体先冻结 `domain + manifest`；并行角色不同时改公共入口；交付必须包含
假设、文件、命令、数值和风险；合并顺序是契约 → 几何 → 质量/展示 → pipeline/API。

## 10. 剩余风险

- 缺少可提交的真实同步 RGB-D fixture，mock 数值不能代表真实传感器。
- 曝光不同步、外参误差、mask 边缘和多径深度会直接进入网格。
- 当前最近顶点距离是轻量观测一致性指标，未来应补精确 point-to-triangle 和逐视角深度回投影。
- 完全遮挡区域依然未知；未来若做补全，必须输出独立 `completed_surface` 并标注推断来源。
