# 第一次运行

## 1. 创建环境

```bash
make setup
```

或者执行包含测试、demo 与 ICP 自测的兼容入口：

```bash
bash scripts/bootstrap_kr1_demo.sh
```

## 2. 生成完整表面与可视化

```bash
./run.sh
```

打开：

```text
outputs/mock_rgbd_demo/report/index.html
```

报告可以拖动旋转、滚轮缩放、切换网格/点云，并检查每个视角的 RGB、深度和 mask。
全量几何位于：

```text
outputs/mock_rgbd_demo/geometry/hand_surface.ply
outputs/mock_rgbd_demo/geometry/hand_geometry.npz
outputs/mock_rgbd_demo/manifest.json
```

## 3. 验证

```bash
make check PYTHON=.venv/bin/python
```

## 常见问题

- `No module named hand_recon`：先运行 `make setup`，或命令前设置 `PYTHONPATH=src`。
- `TSDF grid ... exceeding max_voxel_count`：检查单位和异常离群点，不要盲目提高资源上限。
- `no supported zero crossing`：检查 hand mask、有效 depth、外参和截断距离。
- 页面只有采样数据：这是为了交互速度；PLY/NPZ 始终保存全量网格。
- 表面有边界：本阶段只输出实测表面，不凭空补全遮挡区域。
