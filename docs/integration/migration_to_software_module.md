# 迁移说明：从实验脚本到可集成模块

## 重构目标

本次重构不删除现有数据和运行产物，也不破坏原有脚本入口。目标是把实验型工程逐步收束为可被软件系统集成的 Python package。

## 主要变化

| 旧用法 | 新用法 |
|---|---|
| 直接运行 `demo/run_mock_rgbd_pipeline.py` 获取结果 | 软件调用 `hand_recon.run_mock_reconstruction(...)` |
| 直接读取 `outputs/mock_rgbd_demo/kr3/hand_result.npz` | 软件调用 `hand_recon.load_hand_result_npz(...)` |
| 直接调用 `scripts/generate_hand_visual_report.py` | 软件调用 `hand_recon.generate_mock_visual_report(...)` |
| 直接调用 ICP 脚本处理 Re:InterHand | 软件调用 `hand_recon.run_reinterhand_best_data_visualization(...)` |
| 到处查找底层模块 | 从 `hand_recon.core`、`hand_recon.pipelines`、`hand_recon.adapters`、`hand_recon.reports` 进入 |

## 保留兼容

以下旧入口仍然可用：

```text
demo/run_mock_rgbd_pipeline.py
scripts/run_kr1_checks.sh
scripts/run_kr3_checks.sh
scripts/run_hand_visual_report.sh
scripts/run_best_data_visual_report.sh
scripts/run_multi_agent_validation_report.sh
```

这些脚本现在应被视为 CLI 包装或验收工具，不再是软件集成的主入口。

## 新目录说明

```text
src/hand_recon/
  api.py
  core/
  io/
  pipelines/
  adapters/
  reports/
  interfaces/
```

`interfaces/` 保留已有接口定义，`adapters/` 提供更面向软件集成的命名；二者当前兼容同一套 hand result payload。

## 下一步建议

- 把真实 DMA、真值系统、super-labelator 接入写成 `src/hand_recon/adapters/*`，不要写进报告脚本。
- 把真实配置放到 `configs/`，不要写死在源码里。
- 若后续作为正式包发布，补 `pyproject.toml` 和包版本号。
- 对外 API 稳定后，避免随意改 `hand_recon.api` 函数签名。
