# 第一次运行

## 1. 准备环境

```bash
bash scripts/bootstrap_kr1_demo.sh
```

该命令创建 `.venv`、以 editable 模式安装 `hand_recon`，然后运行测试、mock demo 和
ICP 自测。真实数据不在安装过程中下载。

## 2. 查看结果

```bash
./run.sh
```

打开：

```text
outputs/reports/hand_reconstruction_visual_report.html
```

## 3. 日常开发

```bash
make help
make demo
make test
make check
```

如果不想安装包，命令前加 `PYTHONPATH=src`：

```bash
PYTHONPATH=src python3 -m hand_recon --help
```

## 常见问题

- `No module named hand_recon`：先运行 bootstrap，或设置 `PYTHONPATH=src`。
- 配置报错：从 `configs/mock_rgbd.json` 开始，未知字段会被拒绝。
- demo 退出码为 1：查看 `quality_report.json` 的 `warnings`。
- Re:InterHand 缺文件：先运行 `scripts/prepare_reinterhand_pilot.py --help`，不要把真实
  数据提交到 Git。
- ICP `not_enough_pairs`：检查单位、初值、重叠范围、距离阈值和 `min_pairs`。
