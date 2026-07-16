# KR1 交付报告：mock 多视图 RGB-D 重建闭环（历史快照）

> 本文保留当时交付状态用于追溯。当前运行命令、安全约束和测试结果以根目录 README、
> machine-readable schema 与自动化测试为准。

报告日期：2026-07-09

## 目标

KR1：完成基于 mock 数据的多视图 RGB-D 重建、手部/物体位姿输出和质量评估接口。

## 交付结论

KR1 已完成 mock 闭环交付。当前工程可以自动生成 4 视角 mock RGB-D 数据，完成 depth 反投影、多视角点云融合、hand/object 点云拆分、mock 位姿输出、质量评估和规范化手部 NPZ 输出。

## 核心代码与文档

| 类型 | 文件 |
|---|---|
| demo 入口 | `demo/run_mock_rgbd_pipeline.py` |
| mock 数据生成 | `src/hand_recon/mock_data.py` |
| RGB-D 读取/反投影 | `src/hand_recon/rgbd.py` |
| 多视角重建 | `src/hand_recon/reconstruction.py` |
| 手/物体位姿输出 | `src/hand_recon/pose.py` |
| 质量评估 | `src/hand_recon/evaluation.py` |
| 规范化手部输出 | `src/hand_recon/normalized_output.py` |
| 测试 | `tests/test_mock_rgbd_pipeline.py` |
| I/O schema | `docs/mock_rgbd_io_schema.md` |
| 精度闭环说明 | `docs/reconstruction_accuracy_closed_loop.md` |

## 运行与验收

一键验收命令：

```bash
bash scripts/run_kr1_checks.sh
```

该命令执行：

```bash
python3 demo/run_mock_rgbd_pipeline.py --output-dir outputs/mock_rgbd_demo
python3 scripts/evaluate_normalized_npz_accuracy.py \
  --prediction-npz outputs/mock_rgbd_demo/scale/root_translation_optimized_hands.npz \
  --output-json outputs/mock_rgbd_demo/scale/accuracy_report.json
python3 -m pytest tests/test_mock_rgbd_pipeline.py
python3 scripts/run_icp_registration.py --selftest
```

当前验收结果：

```text
KR1 checks passed.
tests/test_mock_rgbd_pipeline.py: 1 passed
ICP selftest: status=ok
```

## 产物路径

| 产物 | 路径 |
|---|---|
| 融合点云 | `outputs/mock_rgbd_demo/fused_pointcloud.ply` |
| 手部点云 | `outputs/mock_rgbd_demo/hand_pointcloud.ply` |
| 物体点云 | `outputs/mock_rgbd_demo/object_pointcloud.ply` |
| 手/物体位姿 | `outputs/mock_rgbd_demo/pose_output.json` |
| 质量报告 | `outputs/mock_rgbd_demo/quality_report.json` |
| 运行摘要 | `outputs/mock_rgbd_demo/summary.json` |
| 规范化手部输出 | `outputs/mock_rgbd_demo/scale/root_translation_optimized_hands.npz` |
| 自一致性精度报告 | `outputs/mock_rgbd_demo/scale/accuracy_report.json` |

## 关键指标

来自 `outputs/mock_rgbd_demo/quality_report.json`：

| 指标 | 当前值 |
|---|---:|
| `passed` | `true` |
| `view_count` | `4` |
| `depth_valid_ratio_mean` | `0.070516` |
| `hand_point_count` | `1984` |
| `object_point_count` | `1351` |
| `raw_point_count` | `3466` |
| `fused_point_count` | `3335` |
| `views_with_valid_depth` | `4` |
| `coverage_score` | `1.0` |
| `pose_confidence_mean` | `0.95` |

来自 `outputs/mock_rgbd_demo/scale/accuracy_report.json`：

| 指标 | 当前值 |
|---|---:|
| `passed` | `true` |
| `row_count` | `1` |
| `ok_frame_ratio` | `1.0` |
| `valid_joint_ratio` | `1.0` |
| `frame_confidence_mean` | `1.0` |
| `bone_length_mean_m` | `0.0391128434` |
| `bone_length_p95_m` | `0.0861504817` |

## 周报展示口径

可以表述为：

```text
KR1 已完成 mock 多视图 RGB-D 闭环：工程可自动生成 4 视角 RGB-D mock 数据，完成点云融合、手部/物体点云拆分、mock 位姿 JSON 输出、质量评估 JSON 输出，以及规范化手部 NPZ 输出。当前 `scripts/run_kr1_checks.sh` 通过，质量报告 `passed=true`，融合点云 `3335` 点，覆盖率 `1.0`。
```

## 剩余风险

- 当前是 mock 数据闭环，不代表真实相机/真实手部数据精度。
- 当前位姿为 `mock_bbox_centroid`，不是 MANO/WiLoR/真实优化位姿。
- 真实流程仍需要接入真实 GT/reference、2D 重投影、depth residual 和 MPJPE 等评测。
