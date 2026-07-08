# 多视角手部重建 SOP

调研日期：2026-07-02。目标是找一个比 DexYCB/FreiHAND 这类 8 视角更充分覆盖手部方向的数据源，再按已验证的 DexYCB 路线做 ICP 配准、TSDF 融合、手部重建和动态点云播放。

## 1. 参考已验证的稳定流程

此前已经验证过的核心做法：

1. 读取多相机数据、标定和手部分割。
2. 把每个相机的 depth/mask 反投影为 camera-frame 点云。
3. 通过外参统一到 shared/world frame。
4. 对共享坐标点云做体素降采样、裁剪和几何诊断。
5. 可选用 Open3D TSDF 把多视角 hand-masked depth 融成 mesh。
6. 在 fused point cloud 或 mesh 上做手指截面几何。

对应文件：

- `scripts/reconstruct_dexycb_hand_pointcloud.py`
- `scripts/run_dexycb_tsdf_fusion_pipeline.py`
- `src/grundture/geometry/pointcloud_section.py`
- `src/grundture/geometry/mesh_section.py`

## 2. 数据集调研

| 数据集 | 视角/模态 | 完整几何 | 优点 | 风险 |
|---|---:|---:|---|---|
| Re:InterHand | 20 个第三人称 rendered RGB 相机；另有 ego 设置 | 有 MANO fits 和 PLY mesh | 多视角 RGB + mask + camera params + 稳定完整 3D GT；最适合 pilot | 体量很大；RGB 不是原生 depth，需要渲染 depth 或 MVS |
| InterHand2.6M | 80-140 个真实标定 RGB 相机 | 有 3D joints/MANO fits | 真实采集相机最多，适合多视角 RGB 几何 | RGB-only；发布图像分辨率较低；TSDF 需额外深度 |
| Hand4K / Hand-3D-Studio | 15 个 4K DSLR 相机 | 有 3D joints/shape annotation | 真实高分辨率多视角 RGB，比 8 视角更多 | 下载在百度网盘；许可为科研非商用；depth/点云路径需确认 |
| HOT3D | 1.5M 多视角 egocentric frames，3.7M+ images | 有手/物 3D pose/shape，Aria 有 SLAM scene point cloud | 最适合后续动态手-物交互播放 | 不是外部环绕式全方向手部扫描 |
| ShichengChen multiviewDataset | 4 个 RGB-D 相机 | 有完整手点云、MANO、外参 | 对 TSDF 和 ICP 最直接，可做 sanity check | 只有 4 视角，不满足“更多视角”主目标 |

### 推荐选择

第一批 pilot 选 **Re:InterHand**。

原因：

- 20 个第三人称相机已经明显多于 8 视角。
- 数据直接提供 mask、camera parameters 和完整 hand mesh，能绕开纯 RGB 重建在早期引入的 MVS/NeRF 不确定性。
- 可用完整 mesh 采样点云做 ICP 第一阶段，也可从 mesh 按相机参数渲染 depth，接已验证的 TSDF pipeline。

如果目标更偏“真实相机图像”而不是“完整几何验证”，第二选择是 **InterHand2.6M** 或 **Hand4K**。如果目标更偏“动态点云播放”，后续再接 **HOT3D**。

## 3. 来源链接

- Re:InterHand official page: https://mks0601.github.io/ReInterHand/
  - 页面说明 `Mugsy_cameras/envmap_per_frame` 从 20 个相机渲染 image/mask/camera parameters，并提供 MANO/original mesh。
  - License: CC-BY-NC 4.0。
- InterHand2.6M official page: https://mks0601.github.io/InterHand2.6M/
  - 页面提供 v1.0 images/annotations 下载，图像约 80 GB。
  - 论文说明使用 80-140 个高分辨率标定相机采集。
- InterHand2.6M GitHub: https://github.com/facebookresearch/InterHand2.6M
  - README 指向 dataset homepage，并说明 MANO fittings 和 camera visualization tool。
- Hand4K / Hand-3D-Studio: https://www.yangangwang.com/papers/icassp2020-hand3dstudio/ZHAO-H3S-2020-02.html
  - 页面说明 15 个 DSLR、4K、多视角、22K frames、3D joints/shape annotation、102 GB。
- ShichengChen multiviewDataset: https://github.com/ShichengChen/multiviewDataset
  - 页面说明 4 个 RealSense D415 RGB-D、hand masks、2D/3D joints、MANO、完整手点云和相机内外参。
- HOT3D: https://facebookresearch.github.io/hot3d/
  - 页面说明 1.5M multi-view frames、3.7M+ images、手/物 3D pose/shape、动态 clips 和工具。

## 4. 第一阶段 ICP 配准

当前已实现：

- `src/hand_recon/icp.py`
- `scripts/run_icp_registration.py`
- `scripts/prepare_reinterhand_pilot.py`

功能：

- 读取 `.ply`、`.npz`、`.npy` 点云。
- 对 source 点云做体素降采样和随机上限采样。
- 支持 4x4 初始 transform，用相机外参或上一帧位姿做初值。
- 用 trimmed point-to-point ICP 求 source 到 target 的刚体变换。
- 输出每个 source 的 aligned PLY、合并点云和 `icp_summary.json`。

自测：

```bash
python3 scripts/run_icp_registration.py --selftest
```

当前结果：3 组 synthetic 手形点云均通过，平均残差约 0.56-0.57 mm。

### 已准备的 Re:InterHand pilot

已下载并校验：

- Capture: `m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands`
- `mano_fits/mano_fits.tar.gzaa`: 506,710,106 bytes，md5 校验通过。
- `Mugsy_cameras/cam_params.json`: md5 校验通过。
- 解压后 `mano_fits/meshes/*.ply`: 27,736 个。
- 解压后 `mano_fits/params/*.json`: 27,736 个。
- 有效 frame_list: 10,531 行。

准备命令：

```bash
python3 scripts/prepare_reinterhand_pilot.py \
  --skip-metadata \
  --download-mano \
  --download-mugsy-cam-params \
  --extract-mano
```

注意：Re:InterHand 的 MANO mesh 顶点单位是毫米，ICP 时需要 `--input-scale 0.001` 转成米。

真实数据使用模板：

```bash
python3 scripts/run_icp_registration.py \
  --inputs target_cloud.ply source_view_01.ply source_view_02.ply source_view_03.ply \
  --output-dir outputs/icp_registration \
  --voxel-size-m 0.002 \
  --distance-threshold-m 0.035 \
  --trim-fraction 0.9
```

真实 Re:InterHand mesh ICP smoke test：

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

结果：

- `status=converged`
- 4 iterations
- `mean_error=6.93e-05 m`，约 0.069 mm。
- `rmse=7.87e-05 m`，约 0.079 mm。
- `fitness=0.901`
- 输出目录：`outputs/reinterhand_icp_pilot_right_100001_100004`

若用数据集相机外参作为初值，准备一个 JSON：

```json
{
  "source_view_01": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

然后运行：

```bash
python3 scripts/run_icp_registration.py \
  --target target_cloud.ply \
  --source source_view_01.ply \
  --init-transforms-json init_transforms.json
```

### ICP 质量门槛

同一帧多视角配准：

- `mean_error` 目标：2-5 mm；RGB/MVS 生成点云可放宽到 5-10 mm。
- `fitness` 目标：大于 0.7，遮挡严重时需要按局部手指/掌部分开评估。
- 初值来自相机外参时，ICP 修正量应很小：旋转通常小于 2 度，平移通常小于 2 cm。

跨帧动态配准：

- 全手刚体 ICP 只适合手势变化很小的相邻帧。
- 如果手指姿态变化明显，应按 MANO/joint 做非刚体或局部刚体配准，不能只依赖全局 ICP。

## 5. 后续阶段计划

### 阶段 2：TSDF 融合

沿用此前的 Open3D 路线：

1. 对每个视角准备 depth、mask、intrinsics、extrinsics。
2. RGB-D 数据可直接进 `ScalableTSDFVolume`。
3. RGB-only 数据需要先生成 depth：
   - Re:InterHand：从官方 PLY mesh + camera params 渲染 depth。
   - InterHand2.6M/Hand4K：先用 MVS/NeRF/DepthAnything-style depth 估计，再做严格质量过滤。
4. 输出 TSDF mesh、mesh vertices point cloud、summary JSON。

### 阶段 3：手部重建和动态点云播放

静态：

- 输出 fused point cloud、TSDF mesh、局部手指截面结果。

多帧：

- 每帧输出 `frame_XXXX.ply` 或压缩 `.npz`。
- 构建一个 HTML/Three.js 播放器，按时间切换点云，支持暂停、逐帧、视角旋转和按相机/置信度着色。

## 6. 当前未做

- 没有下载 Re:InterHand/InterHand2.6M/Hand4K 大数据；这些数据体量从几十 GB 到数百 GB，建议先下载一个 capture/segment 或 mesh-only subset。
- 没有安装 Open3D；`requirements.txt` 里先把 Open3D 放在 optional，等进入 TSDF 阶段再装。
- 没有做 RGB-only 的 MVS/depth 估计；第一阶段只处理已有点云/mesh-sampled 点云。
