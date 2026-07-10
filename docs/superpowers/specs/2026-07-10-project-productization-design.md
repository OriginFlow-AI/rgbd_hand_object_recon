# RGB-D 手物重建工程产品化设计

## 1. 目标

把当前工程整理成一个可运行、可理解、可测试、可集成、可量化的数据产品，而不是继续堆叠实验脚本。

工程的核心目的只有一条：接收多视角 RGB-D 或等价几何观测，输出统一坐标系中的手部/物体重建结果、手部姿态结果和可审计的质量指标。

本次交付同时解决六件事：

1. 建立标准 Python 包、统一命令行和一键运行入口。
2. 明确源码、数据、产物、文档和第三方工具的边界。
3. 实现时间戳对齐、pose 检出率和数据可用率分析。
4. 生成 JSON、CSV 和自包含 HTML 可视化结果。
5. 用测试、静态检查和多角色复核建立质量门禁。
6. 保留工作文件，删除旧 Git 历史，重新初始化并做一次中文首次提交。

## 2. 已确认事实

- 当前包名是 `hand_recon`，稳定入口是 `src/hand_recon/api.py`。
- 现有 mock 流程能够生成四视角 RGB-D、融合点云、手/物 pose、质量报告和统一 hand result。
- 重构前基线为 `4 passed`，mock 质量检查通过，融合点云为 3,335 点。
- mock 四个视角使用同一个常量时间戳，因此样例的名义跨视角偏差为 0 ms；它不能证明真实硬件同步质量。
- 当前 mock hand pose 是点云包围盒质心占位，不是真实 MANO、UmeTrack 或视觉模型检测。
- 本地 Re:InterHand 选定序列有 10,531 个期望帧，左右手 MANO 文件对这些帧均齐全；这是标注/拟合文件可用率，不是模型检出率。
- 若改用 `frame_list_orig.txt` 的 14,018 帧为分母，左右手各有 13,868 帧 MANO 文件，可用率均为 98.929947%；两种分母必须随指标一起展示。
- 本地 Re:InterHand 资料只有 frame id，没有逐相机原始时间戳、时钟域和帧率，真实多相机同步误差不可计算。
- `01stock_gain_cli copy.py` 不在本仓库，它属于另一个股票工具目录；其无限轮询、行情网络访问和硬编码持仓与本工程无关，不纳入运行入口或提交。
- 旧 `.git` 约 1.2 GiB，而跟踪内容主要是轻量源码和文档，重新初始化能去掉无关历史对象。

## 3. 方案选择

采用渐进式产品化，不做全量重写。

- 保留已经过测试的公开 API、核心算法和现有数据。
- 新增标准打包、统一 CLI、分析子域和一键入口。
- 把旧 demo/验收脚本降级为兼容入口，不让业务软件依赖它们。
- 整理重复或过时文档，但不删除本地真实数据和运行产物。
- 对暂时缺少证据的能力显式标注 `not_evaluable`，不生成虚假指标。
- 当前交付定位为“可集成的重建基础库、确定性演示和数据质量分析工具”；真实模型推理、真实多相机 RGB-D 重建精度和动态播放不在本轮伪装为已完成能力。

## 4. 目标架构

```text
rgbd_hand_object_recon/
├── pyproject.toml              # 包、依赖、测试和代码质量配置
├── Makefile                    # 面向开发者的标准任务入口
├── run.sh                      # 最短可见效果入口
├── START_HERE.md               # 新人从这里开始
├── configs/                    # 可提交的参数示例
├── docs/                       # 架构、采集、分析、集成和字段参考
├── schemas/                    # 机器可读交换契约
├── src/hand_recon/
│   ├── api.py                  # 稳定软件 API
│   ├── cli.py                  # 统一 CLI
│   ├── __main__.py             # python -m hand_recon
│   ├── analysis/               # 时间对齐、检出率、数据适配和报告
│   ├── pipelines/              # 可组合业务流程
│   ├── interfaces/             # 稳定结果契约
│   ├── adapters/               # 外部系统适配
│   ├── reports/                # 重建结果展示
│   └── 现有几何模块             # RGB-D、点云、ICP、pose、评估
├── tests/                      # 单元、集成和 CLI 测试
├── scripts/                    # 数据准备和兼容脚本
├── third_party/                # 明确标注来源的第三方下载工具
├── data/                       # 原始数据，不进 Git
└── outputs/                    # 可再生成产物，不进 Git
```

第一性原理约束：

- 算法核心不读取命令行参数，也不依赖 HTML。
- CLI 只负责解析参数、调用服务和输出路径。
- 原始数据只读；所有派生产物写入 `outputs/`。
- 每个指标必须带分母、单位、阈值、数据来源和能力边界。
- 下游软件只依赖 `hand_recon` API 或 schema，不依赖脚本内部实现。
- synthetic/mock 来源必须显式标记，不能冒充 `super_labelator` 等真实上游。
- NPZ 默认禁止 pickle；字符串字段使用普通 Unicode dtype，不可信输入不能触发对象反序列化。

## 5. 分析数据契约

标准观测记录使用以下字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `sequence_id` | string | 序列标识 |
| `frame_id` | string | 逻辑帧标识 |
| `camera_id` | string | 相机标识 |
| `timestamp_ns` | integer/null | 指定时钟域下的纳秒时间戳 |
| `clock_domain` | string/null | hardware、device、host_monotonic 等 |
| `pose_status` | string | ok、missing、invalid、not_run |
| `pose_confidence` | float/null | 0 到 1 的置信度 |
| `valid_joint_count` | integer/null | 有效 3D 关节点数量 |
| `source` | string | mock、detector、mano_fit、human_label 等 |

JSONL 是通用输入格式。当前工程另提供 mock scene 和 Re:InterHand 本地目录适配器，适配结果仍进入同一指标模型。

## 6. 指标定义

### 6.1 时间戳对齐

对同一 `sequence_id + frame_id` 的多相机记录，在时钟域一致且时间戳完整时计算：

```text
skew_ns = max(timestamp_ns) - min(timestamp_ns)
aligned = skew_ns <= tolerance_ns
```

报告完整同步组数、可计算组数、对齐率、skew 的 min/mean/p50/p95/max 和随帧漂移趋势。时间戳缺失或时钟域不同的组进入 `not_evaluable`，不计入对齐率分母。

### 6.2 pose 检出率

先定义“应尝试检测的合格帧”作为分母。单条 pose 只有同时满足以下条件才算检出：

- `pose_status == "ok"`；
- `pose_confidence >= confidence_threshold`；
- `valid_joint_count >= min_valid_joints`。

```text
pose_detection_rate = detected_eligible_records / eligible_records
```

同时报告 frame 级、camera 级和 source 级分组。MANO 文件覆盖率、检测率、MPJPE/重投影误差是三个不同指标，禁止互相替代。

### 6.3 数据可用率

Re:InterHand 适配器报告 frame list 中期望帧与左右手 MANO params/mesh 的交集，并明确标记为 annotation availability。frame id 只能用于顺序和缺失分析，不能擅自换算成真实时间。

## 7. 可视化与产物

一次分析生成：

- `outputs/analysis/summary.json`：机器可读结论、指标、阈值和限制。
- `outputs/analysis/alignment_groups.csv`：逐同步组的 camera 数、skew 和状态。
- `outputs/analysis/pose_records.csv`：逐记录 pose 判定和失败原因。
- `outputs/analysis/report.html`：无需外网资源的自包含报告。

HTML 至少包含：

- 结论卡片和数据能力边界；
- 相机时间偏差散点/时间线；
- skew 分布；
- 帧 × 相机完整性矩阵；
- pose 检出率和失败原因；
- Re:InterHand 文件覆盖率；
- 本次实际输入、阈值和复现命令。

## 8. 运行与集成

最短可见效果：

```bash
./run.sh
```

标准开发入口：

```bash
make setup
make demo
make analyze
make check
```

标准 CLI：

```bash
python -m hand_recon demo
python -m hand_recon analyze
python -m hand_recon verify
```

软件集成继续从 `from hand_recon import ...` 开始，新增分析 API 通过显式导出提供，不要求下游 shell 调用。

## 9. 质量与错误处理

- 核心指标函数先写失败测试，再写最小实现。
- 覆盖完美同步、超阈值、时间戳缺失、跨时钟域、pose 低置信、关节不足和空输入。
- CLI 对缺文件、坏 JSONL、非法阈值返回非零退出码和可行动错误信息。
- 打包测试验证 `pip install -e .` 后无需手工修改 `sys.path` 即可导入和运行。
- 验收至少运行 pytest、compileall、CLI help、mock demo、实际数据分析和报告内容检查。
- 实现后由独立智能体分别做需求符合性审查和代码质量审查，主智能体复跑全量命令。

## 10. 文档与清理原则

- README 只保留定位、三分钟启动、核心产物、架构导航和真实能力边界。
- `START_HERE.md` 面向第一次运行的人；架构、采集、分析和集成分别成文。
- 历史 KR 报告若仍有追溯价值则归档，不与当前使用说明混排。
- 删除或归档失效的 Gitee/旧上传说明；GitHub 远程信息只在最终重新初始化时恢复。
- `data/`、`outputs/`、`dist/`、缓存和虚拟环境继续忽略，不删除用户本地内容。

## 11. Git 重新初始化

在所有代码和文档通过验收后执行：

1. 记录原远程 `git@github.com:OriginFlow-AI/rgbd_hand_object_recon.git`。
2. 删除旧 `.git`，不删除其他文件。
3. `git init -b main`，恢复 `origin`。
4. 检查暂存清单不包含数据、输出、dist、缓存或凭据。
5. 使用中文提交名：`初始化：规范化 RGB-D 手物重建与数据质量分析工程`。
6. 只有在远程目标可访问且不会无意覆盖既有分支时才上传；因为新历史与远程 `main` 无共同祖先，禁止未经明确安全判断直接强推覆盖。

## 12. 验收结论格式

最终只保留四类信息：

1. 结论是什么；
2. 下一步做什么；
3. 需要哪些文件/命令；
4. 有什么风险/坑。

每日收口模板：

```text
昨天推进了什么：
卡在哪里：
今天第一步：
```
