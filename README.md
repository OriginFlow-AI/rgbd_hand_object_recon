# RGB-D Hand/Object Reconstruction

本工程用于把 DexYCB 三维重建思路迁移到“更多视角覆盖”的手部数据上。当前先完成第一阶段：数据集选型、mock RGB-D 闭环和 ICP 刚体配准脚手架。

## 快速开始

新同事 clone 后，如果希望一条命令创建虚拟环境、安装依赖并完成 KR1 验收，运行：

```bash
bash scripts/bootstrap_kr1_demo.sh
```

如果本机已经安装好 `requirements.txt` 里的依赖，可直接运行轻量验收脚本：

```bash
bash scripts/run_kr1_checks.sh
```

该脚本会依次执行：

```bash
python3 demo/run_mock_rgbd_pipeline.py --output-dir outputs/mock_rgbd_demo
python3 -m pytest tests/test_mock_rgbd_pipeline.py
python3 scripts/run_icp_registration.py --selftest
```

## 工程规范

- 目录职责见 [docs/project_structure.md](docs/project_structure.md)。
- mock RGB-D 输入输出格式见 [docs/mock_rgbd_io_schema.md](docs/mock_rgbd_io_schema.md)。
- `root_translation_optimized_hands.npz` 字段说明见 [docs/root_translation_optimized_hands_npz_schema.md](docs/root_translation_optimized_hands_npz_schema.md)。
- 重建精度闭环方案见 [docs/reconstruction_accuracy_closed_loop.md](docs/reconstruction_accuracy_closed_loop.md)。
- KR 提交交付规范见 [docs/kr_delivery_submission_guideline.md](docs/kr_delivery_submission_guideline.md)。
- 多智能体协同和一次性闭环任务指令见 [docs/multi_agent_closed_loop_task.md](docs/multi_agent_closed_loop_task.md)。
- gitee 初始化/同步步骤见 [docs/gitee_sync.md](docs/gitee_sync.md)。
- `data/`、`outputs/` 和自动生成的 `mock_data/rgbd_scene_001/` 不提交到代码仓库。
- `.gitignore` 已配置真实大数据、生成结果、虚拟环境和 Python 缓存。

## 上传版本

生成不含真实数据和运行产物的上传包：

```bash
bash scripts/create_upload_package.sh
```

输出位于 `dist/rgbd_hand_object_recon_upload_*.tar.gz`，包内保留源码、脚本、配置、文档、测试和 `mock_data/.gitkeep`，排除 `data/`、`outputs/`、`mock_data/rgbd_scene_001/`、`.git/`、虚拟环境和缓存。

## 当前结论

优先数据集建议选 **Re:InterHand** 做 pilot：

- 它提供多视角 RGB、mask、camera parameters，以及完整手部 3D mesh/ MANO fits。
- `Mugsy_cameras/envmap_per_frame` 是 20 个第三人称相机，适合验证“比 8 视角更完整覆盖”的流程。
- 它不是原生 RGB-D；后续 TSDF 阶段需要从 mesh/camera 渲染 depth，或先跑 RGB 多视角深度/MVS。

备选：

- **InterHand2.6M**：真实采集，80-140 个标定 RGB 相机，视角数量最多；但 RGB-only，对 TSDF 不够直接。
- **Hand4K / Hand-3D-Studio**：15 个 4K DSLR 相机，有 3D joints/shape annotation，真实 RGB 质量高。
- **HOT3D**：动态手-物交互，多视角 egocentric，适合后续动态点云播放，但不是外部 360 度手部扫描。
- **ShichengChen multiviewDataset**：只有 4 个 RGB-D 视角，但直接给 depth、完整点云和外参，适合 TSDF/ICP sanity check。

详细 SOP 和来源链接见 [docs/hand_reconstruction_sop.md](docs/hand_reconstruction_sop.md)。

## KR1：mock 多视图 RGB-D 重建 demo

当前已补一个不依赖真实相机数据的最小闭环：

- 自动生成 4 视角 mock RGB-D 数据：`rgb.npy`、`depth.npy`、`mask.npy`、`cameras.json`。
- 读取相机内参/外参，把 depth 反投影为 camera-frame 点云。
- 根据 `camera_to_world` 融合到 world frame，并输出体素降采样点云。
- 输出 hand/object 的 mock 位姿 JSON。
- 输出质量评估 JSON，包括深度有效率、点数、覆盖率、bbox extent 和 pose confidence。

一键运行：

```bash
python3 demo/run_mock_rgbd_pipeline.py --output-dir outputs/mock_rgbd_demo
```

如果 mock 场景不存在，demo 会自动生成 `mock_data/rgbd_scene_001/`。

主要输出：

- `outputs/mock_rgbd_demo/fused_pointcloud.ply`
- `outputs/mock_rgbd_demo/hand_pointcloud.ply`
- `outputs/mock_rgbd_demo/object_pointcloud.ply`
- `outputs/mock_rgbd_demo/pose_output.json`
- `outputs/mock_rgbd_demo/quality_report.json`
- `outputs/mock_rgbd_demo/summary.json`
- `outputs/mock_rgbd_demo/scale/root_translation_optimized_hands.npz`
- `outputs/mock_rgbd_demo/scale/accuracy_report.json`（运行精度闭环脚本后生成）

mock RGB-D 输入输出 schema 见 [docs/mock_rgbd_io_schema.md](docs/mock_rgbd_io_schema.md)。

精度闭环：

```bash
python3 scripts/evaluate_normalized_npz_accuracy.py \
  --prediction-npz outputs/mock_rgbd_demo/scale/root_translation_optimized_hands.npz \
  --output-json outputs/mock_rgbd_demo/scale/accuracy_report.json
```

测试：

```bash
python3 -m pytest tests/test_mock_rgbd_pipeline.py
```

## 第一阶段：ICP

已经准备好的 Re:InterHand pilot：

- Capture: `m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands`
- Metadata: `data/reinterhand/.../{CHECKSUM,frame_list.txt,frame_list_orig.txt}`
- MANO meshes: `data/reinterhand/.../mano_fits/meshes`
- Mugsy 20 相机参数: `data/reinterhand/.../Mugsy_cameras/cam_params.json`
- Summary: `outputs/reinterhand_pilot_summary.json`

下载/恢复/校验/解压命令：

```bash
python3 scripts/prepare_reinterhand_pilot.py \
  --skip-metadata \
  --download-mano \
  --download-mugsy-cam-params \
  --extract-mano
```

可运行自测：

```bash
python3 scripts/run_icp_registration.py --selftest
```

对真实点云运行：

```bash
python3 scripts/run_icp_registration.py \
  --inputs target_cloud.ply source_view_01.ply source_view_02.ply \
  --output-dir outputs/icp_registration \
  --voxel-size-m 0.002 \
  --distance-threshold-m 0.035
```

如果已有相机外参或上一帧位姿，可用 JSON 给每个 source 一个 4x4 初值：

```bash
python3 scripts/run_icp_registration.py \
  --target target_cloud.ply \
  --source source_view_01.ply \
  --init-transforms-json init_transforms.json
```

Re:InterHand 的 MANO mesh 是毫米单位，运行 ICP 时用 `--input-scale 0.001` 转成米：

```bash
cap='m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands'
python3 scripts/run_icp_registration.py \
  --target "data/reinterhand/$cap/mano_fits/meshes/100001_right.ply" \
  --source "data/reinterhand/$cap/mano_fits/meshes/100004_right.ply" \
  --output-dir outputs/reinterhand_icp_pilot_right_100001_100004 \
  --input-scale 0.001 \
  --voxel-size-m 0.001 \
  --distance-threshold-m 0.03 \
  --trim-fraction 0.9 \
  --min-pairs 100 \
  --max-iterations 80
```

当前真实 mesh ICP 结果：

- status: `converged`
- iterations: `4`
- mean error: `6.93e-05 m` (`0.069 mm`)
- RMSE: `7.87e-05 m` (`0.079 mm`)
- fitness: `0.901`

输出：

- `aligned_*.ply`：每个 source 对齐到 target 后的点云。
- `merged_aligned_voxel.ply`：合并后体素降采样点云。
- `icp_summary.json`：每个 source 的 4x4 transform、残差、fitness、迭代历史。

## 与 0401 的对应关系

`0401_grundture` 中可复用的主线：

- `scripts/reconstruct_dexycb_hand_pointcloud.py`：分割 depth 反投影到 shared frame，导出 fused hand point cloud。
- `scripts/run_dexycb_tsdf_fusion_pipeline.py`：Open3D `ScalableTSDFVolume` 融合 hand-masked depth，导出 mesh。
- `src/grundture/geometry/pointcloud_section.py` 和 `mesh_section.py`：后续做手指截面/环中心的几何工具。

本工程的计划是：

1. ICP 配准：已放入 `src/hand_recon/icp.py` 和 `scripts/run_icp_registration.py`。
2. TSDF 融合：下一阶段接 Open3D，把多视角 depth 或从 mesh 渲染出的 depth 融合成 mesh。
3. 手部重建/动态播放：静态输出 PLY/mesh；多帧时输出 per-frame PLY 和一个点云播放 HTML。
