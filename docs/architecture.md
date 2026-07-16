# 架构与核心执行流程

## 设计目标

工程把“几何计算”“流程编排”“交换契约”“展示”分开。公共 API 只依赖 package
代码；CLI、demo 和批处理脚本依赖公共 API 或 pipeline，不允许核心模块反向依赖脚本。

```text
CLI / demo / scripts
        |
        v
public API ---- reports
        |
        v
pipelines ---- adapters/interfaces
   |  |
   |  +------ normalized output
   v
RGB-D / reconstruction / ICP / pose / evaluation
        |
        v
safe JSON/NPZ and PLY I/O
```

## Mock RGB-D 主流程

1. `config.py` 读取 JSON，拒绝未知字段、非法单位和非法阈值。
2. `mock_data.py` 在场景不存在时生成 4 个确定性视角。
3. `rgbd.py` 校验 metadata、相机参数、相对文件路径和数组 shape。
4. `backproject_depth_to_points` 使用 pinhole 模型把有效 depth 像素恢复到 camera frame。
5. `reconstruction.py` 使用 `camera_to_world` 变换每个视角，并按 mask label 选择点。
6. `icp.voxel_downsample` 对拼接点云做确定性体素均值融合。
7. `pose.py` 生成 mock bbox-centroid pose；`evaluation.py` 运行质量门禁。
8. `normalized_output.py` 生成 left-camera 坐标系的 21 joints 和 20DoF-like 几何角。
9. `interfaces/hand_result.py` 适配为 KR3 22DOF/joints/mesh 契约并严格校验。
10. `io/` 原子写 JSON/NPZ；报告层只消费已经落盘的稳定产物。

默认 pipeline 会对 fused、hand 和 object 分别重建，以保持清晰的每类产物和既有输出
契约。未来若性能成为瓶颈，可以在保持统计口径的前提下复用单次反投影结果。

## 坐标和单位

- 输入深度：meter。
- `camera_to_world`：4×4 齐次变换。
- 融合点云和 pose：world frame，meter。
- normalized hand：第一个视角（rectified left camera）frame，meter/radian。
- KR3 mock adapter：继承 normalized hand 坐标系。

工程不猜测单位。错误的 mm/m 输入会导致灾难性尺度错误，因此当前选择显式拒绝未知
`depth_unit`，而不是静默换算。

## I/O 与失败策略

- metadata 中数组路径必须是场景目录内的相对路径。
- NPY/NPZ 禁止 object/pickle；字符串使用 Unicode dtype。
- JSON 和 NPZ 使用同目录临时文件后原子替换，减少中断造成的半文件。
- 相机参数、矩阵、shape、有限值和 KR3 字段一致性在边界处校验。
- library 抛出 `HandReconError`/`ValueError` 兼容异常；CLI 记录错误并返回非零退出码。
- library 不配置全局日志，只有 CLI/脚本负责 logging policy。

## ICP 流程

点云先按输入 scale 转为 meter，再做体素降采样和确定性随机上限采样。每轮 ICP：

```text
nearest neighbors
  -> distance threshold
  -> trimmed correspondences
  -> SVD best-fit rigid transform
  -> compose transform
  -> tolerance / max-iteration termination
```

SciPy `cKDTree` 是正常路径；只有 SciPy 无法导入时才使用 NumPy 分块 fallback。运行结果
保留 transform、误差、fitness、pair count 和逐轮 history，便于审计。
