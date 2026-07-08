# 重建精度闭环方案

重建精度不要只靠可视化判断，建议固定输出一个 `accuracy_report.json`，把每次结果变成可比较、可验收的数字。

## 1. 闭环对象

稳定流程最终结果：

```text
$OUT_DIR/scale/root_translation_optimized_hands.npz
```

基础自检脚本：

```bash
python3 scripts/evaluate_normalized_npz_accuracy.py \
  --prediction-npz outputs/mock_rgbd_demo/scale/root_translation_optimized_hands.npz \
  --output-json outputs/mock_rgbd_demo/scale/accuracy_report.json
```

如果有 GT/reference NPZ：

```bash
python3 scripts/evaluate_normalized_npz_accuracy.py \
  --prediction-npz $OUT_DIR/scale/root_translation_optimized_hands.npz \
  --reference-npz $GT_DIR/root_translation_optimized_hands_gt.npz \
  --output-json $OUT_DIR/scale/accuracy_report.json
```

## 2. 没有 GT 时怎么闭环

没有 GT 时先做自一致性闭环，至少保证结果可读、可用、可稳定比较：

- 字段闭环：检查 `root_translation_optimized_hands.npz` 是否包含全部规范字段。
- shape 闭环：检查每个字段 shape 是否符合文档。
- 有效率闭环：统计 `valid_joint_mask`、`frame_status`、`frame_confidence`。
- 骨长闭环：统计 20 条手部骨段长度，排查异常尺度和爆点。
- 时序闭环：按 `frame_index + hand_side` 统计 wrist/root 相邻帧位移，排查抖动。

这类闭环不能证明“绝对准确”，但能证明输出格式稳定、尺度没有明显错误、时序没有明显跳变。

## 3. 有 GT 时怎么闭环

有 GT 或高可信 reference 时做定量精度闭环：

- `MPJPE`: `joints_3d_left_m` 的平均 3D 关节误差，单位 m/mm。
- `P95 joint error`: 95 分位关节误差，避免只看均值。
- `root_translation_rmse_m`: wrist/root translation RMSE。
- `hand_angle_mae_deg`: 20DoF-like 几何角平均绝对误差。

匹配规则优先使用：

```text
frame_index + hand_side
```

不要只依赖行顺序，因为最终 NPZ 是逐行手实例结构。

## 4. 双目/2D 重投影闭环

真实流程建议再补 2D 重投影闭环：

```text
joints_3d_left_m
-> 使用 K 投影到 rectified left image
-> 使用 baseline_m 投影到 rectified right image
-> 对比 WiLoR left/right 2D joints
```

关键指标：

- left reprojection mean / median / p95 pixel error
- right reprojection mean / median / p95 pixel error
- per-joint reprojection error
- per-frame reprojection pass/fail

这一步能检查 3D 结果是否真的解释了左右目 2D 观测，是最重要的无 GT 闭环之一。

## 5. 点云/深度闭环

如果有 mask depth 或融合点云，可做几何闭环：

- 将最终 hand joints 或 hand mesh 投影回深度图，检查深度残差。
- 将重建点云与参考点云做 Chamfer distance。
- 对真实 mesh/多帧点云可先 ICP 对齐，再统计 point-to-point / point-to-plane RMSE。

建议报告：

```text
depth_residual_mean_m
depth_residual_p95_m
chamfer_l2_m
icp_rmse_m
icp_fitness
```

## 6. 建议验收阈值

阈值要按数据源调整。初始可用：

| 指标 | 建议阈值 |
|---|---:|
| `valid_joint_ratio` | `>= 0.95` |
| `ok_frame_ratio` | `>= 0.95` |
| `MPJPE` | `< 20 mm` |
| `root_translation_rmse_m` | `< 10 mm` |
| `hand_angle_mae_deg` | `< 10 deg` |
| 2D reprojection mean | `< 5 px` |
| 2D reprojection p95 | `< 15 px` |
| depth residual mean | `< 10 mm` |

## 7. 最终闭环产物

每次跑完整流程后，保留：

```text
$OUT_DIR/scale/root_translation_optimized_hands.npz
$OUT_DIR/scale/accuracy_report.json
```

`accuracy_report.json` 用于版本对比、验收、回归测试和提交记录。
