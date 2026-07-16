# 工程目录与依赖规则

```text
rgbd_hand_object_recon/
  pyproject.toml                  # 包版本、运行/开发依赖、质量工具
  Makefile                        # setup/demo/test/check
  run.sh                          # 最短完整结果入口
  configs/                        # 可提交的运行配置
  src/hand_recon/
    api.py, cli.py                # 外部入口与命令行
    config.py, exceptions.py      # 配置边界和领域异常
    domain.py                     # 稳定表面数据契约
    rgbd.py                       # 标定场景、输入校验、反投影
    reconstruction.py             # 带 RGB/view id 的多视角点云融合
    fusion/
      tsdf.py                     # projective TSDF
    surface/
      mesh.py                     # marching tetrahedra、清理、法线
      quality.py                  # 几何质量指标和门禁
    pipelines/
      hand_surface.py             # 新的表面用例
      mock_rgbd.py                # mock/兼容产物总编排
      reinterhand.py              # 数据集参考实验
    io/
      geometry.py                 # mesh PLY、surface NPZ
      artifacts.py                # manifest、校验和、产物边界
      json_io.py, npz.py          # 安全原子基础 I/O
    visualization/
      surface_report.py           # 无关节点离线三维报告
    reports/                      # 兼容 CLI 与其他历史报告
    interfaces/, adapters/        # 兼容 KR3 契约
    normalized_output.py, pose.py # 兼容关节点/位姿旁路
  tests/                          # 数值、契约、安全、CLI、集成测试
  docs/                           # 当前架构、SOP 与历史报告
  scripts/                        # 薄命令行包装和数据准备
  third_party/                    # 上游脚本与本地安全包装
  mock_data/, data/, outputs/     # 生成输入、真实数据、产物（不提交）
```

## 依赖方向

```text
api / cli
   ↓
pipelines
   ↓
domain ← reconstruction ← rgbd
   ↑          ↓
 surface ← fusion
   ↓
io/artifacts → visualization

compatibility adapters ──旁路──→ legacy outputs
```

约束：

- `domain` 不依赖 pipeline、I/O 或报告。
- `fusion` 和 `surface` 是确定性数值模块，不读写文件。
- `visualization` 只读取 manifest 声明的稳定产物，不访问 pipeline 内部对象。
- `interfaces/adapters` 只负责旧格式，不得反向驱动表面重建。
- `scripts/` 不承载业务逻辑；应用只 import `hand_recon` 或 `hand_recon.api`。
- 产物用 `manifest.json` 交接，不使用临时文件名或 HTML 文本作为机器接口。

这个结构刻意没有引入 DAG 引擎、服务总线、数据库或运行时 LLM agent。当前问题是单机
几何重建；增加与失败边界无关的层只会降低可验证性。
