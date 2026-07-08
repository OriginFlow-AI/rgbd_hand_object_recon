# KR 提交交付规范

KR 提交不是状态汇报，而是可验收交付。所有开发人员提交 KR 时，必须写清楚时间、产出、证据、指标和技术结论。

多智能体协同和一次性闭环任务的下达方式见 [多智能体协同闭环任务协议](multi_agent_closed_loop_task.md)。

## 核心原则

- 不写空状态：不要只写“完成了”“进行中”“未完成”“已优化”。
- 必须有 Key Result：每条 KR 都要能被检查、复现或对比。
- 必须有交付物：至少包含 doc、code、视频/截图、指标报告、技术结论中的一种；工程类 KR 默认需要 code/doc/report。
- 必须有时间：说明完成时间、统计周期或本次提交覆盖的时间范围。
- 必须有证据：给出文件路径、命令、报告路径、截图/视频路径、PR/commit 或指标数值。

## 提交模板

```text
KR 名称：
提交时间：
负责人：
本次范围：

Key Results：
1. KR1：
   - 产出：
   - 证据：
   - 指标：
   - 技术结论：
2. KR2：
   - 产出：
   - 证据：
   - 指标：
   - 技术结论：

验收命令：

风险/未完成项：
下一步：
```

## 交付物要求

| 类型 | 合格交付 | 不合格写法 |
|---|---|---|
| 文档 | `docs/xxx.md`，说明用途、字段、流程或结论 | “已写文档” |
| 代码 | 具体文件、模块、脚本、测试路径 | “代码已完成” |
| 视频/截图 | 截图路径、视频路径、关键画面说明 | “效果正常” |
| 指标 | 数值、报告路径、通过阈值 | “精度不错” |
| 技术结论 | 可验证判断和边界条件 | “方案可行” |

## 工程类 KR 必填项

- 代码产出：列出新增/修改的核心文件。
- 文档产出：列出接口、字段、流程或使用说明。
- 验收命令：列出实际运行过的命令。
- 指标报告：列出 JSON/report 路径和关键数值。
- 技术结论：说明当前能力边界、限制和下一步依赖。

## 示例：合格提交

```text
KR 名称：规范化手部重建 NPZ 输出
提交时间：2026-07-08 13:42
负责人：xxx
本次范围：KR1 mock RGB-D demo 输出规范化手部结果，并补充精度闭环自检。

Key Results：
1. 规范化 NPZ 输出已接入 demo。
   - 产出：src/hand_recon/normalized_output.py、demo/run_mock_rgbd_pipeline.py
   - 证据：outputs/mock_rgbd_demo/scale/root_translation_optimized_hands.npz
   - 指标：字段 27 个，核心 shape 为 joints_3d_left_m=(1,21,3)、hand_angles_20dof_rad=(1,20)
   - 技术结论：当前 mock demo 无 WiLoR/stereo optimizer，left_row/right_row 使用 -1 占位，字段格式与稳定流程对齐。

2. 精度闭环报告已生成。
   - 产出：scripts/evaluate_normalized_npz_accuracy.py、docs/reconstruction_accuracy_closed_loop.md
   - 证据：outputs/mock_rgbd_demo/scale/accuracy_report.json
   - 指标：passed=true，valid_joint_ratio=1.0，ok_frame_ratio=1.0，warnings=[]
   - 技术结论：当前闭环覆盖字段、shape、有效率、骨长和 root 时序自检；有 GT 后可计算 MPJPE/root RMSE/角度 MAE。

验收命令：
bash scripts/run_kr1_checks.sh

风险/未完成项：
当前是 mock 几何 joint，不代表真实 WiLoR/MANO 精度；真实流程需要接入 2D 重投影和 GT/reference 评测。

下一步：
接入真实 stereo sparse triangulation 后，补 left/right reprojection error。
```

## 示例：不合格提交

```text
今天完成了 KR1，代码已经写完，测试也过了，后面继续优化。
```

原因：没有时间范围、没有文件路径、没有验收命令、没有指标数值、没有技术结论，无法验收。

## 最低验收线

每次 KR 提交至少回答这五个问题：

1. 什么时候完成？
2. 交付了什么文件或产物？
3. 怎么复现或查看？
4. 关键指标是多少？
5. 技术结论是什么，边界在哪里？
