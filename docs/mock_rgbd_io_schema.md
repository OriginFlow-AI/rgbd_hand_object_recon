# Mock RGB-D I/O Schema

This document describes the KR1 mock multi-view RGB-D reconstruction demo.

## Directory Layout

```text
mock_data/rgbd_scene_001/
  cameras.json
  frames/
    view_00/
      rgb.npy
      depth.npy
      mask.npy
    view_01/
      rgb.npy
      depth.npy
      mask.npy
```

`demo/run_mock_rgbd_pipeline.py` creates the scene automatically when
`mock_data/rgbd_scene_001/cameras.json` does not exist.

## Camera Metadata

`cameras.json` contains:

```json
{
  "scene_id": "rgbd_scene_001",
  "coordinate_frame": "world",
  "depth_unit": "meter",
  "mask_labels": {
    "background": 0,
    "hand": 1,
    "object": 2
  },
  "views": [
    {
      "camera_id": "view_00",
      "width": 128,
      "height": 96,
      "timestamp": "2026-07-07T00:00:00+08:00",
      "intrinsics": {
        "fx": 118.0,
        "fy": 118.0,
        "cx": 63.5,
        "cy": 47.5
      },
      "extrinsics": {
        "camera_to_world": [[1, 0, 0, 0]]
      },
      "files": {
        "rgb": "frames/view_00/rgb.npy",
        "depth": "frames/view_00/depth.npy",
        "mask": "frames/view_00/mask.npy"
      }
    }
  ]
}
```

`camera_to_world` is a 4x4 transform. Depth values are in meters.

## Frame Arrays

- `rgb.npy`: `H x W x 3`, `uint8`.
- `depth.npy`: `H x W`, `float32`, meters. Invalid background depth is `0`.
- `mask.npy`: `H x W`, `uint8`.

Mask labels:

- `0`: background
- `1`: hand
- `2`: object

## Reconstruction Outputs

The demo writes:

```text
outputs/mock_rgbd_demo/
  fused_pointcloud.ply
  hand_pointcloud.ply
  object_pointcloud.ply
  pose_output.json
  quality_report.json
  summary.json
  scale/
    root_translation_optimized_hands.npz
```

`pose_output.json` uses world-frame mock poses:

```json
{
  "status": "ok",
  "scene_id": "rgbd_scene_001",
  "coordinate_frame": "world",
  "timestamp": "2026-07-07T00:00:00+08:00",
  "hands": [
    {
      "id": "hand_0",
      "label": "hand",
      "pose_type": "mock_bbox_centroid",
      "translation_m": [0.0, 0.0, 0.0],
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "bbox_3d_m": {
        "min": [0.0, 0.0, 0.0],
        "max": [0.0, 0.0, 0.0]
      },
      "point_count": 100,
      "confidence": 0.9
    }
  ],
  "objects": []
}
```

`quality_report.json` includes:

- depth valid ratio per view and mean
- hand/object/fused point counts
- views with valid depth
- fused 3D bounding-box extent
- coverage score
- mean pose confidence
- pass/fail flag and warnings

`scale/root_translation_optimized_hands.npz` follows
[root_translation_optimized_hands.npz 字段说明](root_translation_optimized_hands_npz_schema.md).
For the KR1 mock demo, WiLoR/stereo optimizer row fields use `-1`
placeholders, orientation residual is zero, and the 21 hand joints are
deterministic geometric landmarks estimated from the reconstructed hand point
cloud in the first camera coordinate frame.

## Demo Command

```bash
python3 demo/run_mock_rgbd_pipeline.py --output-dir outputs/mock_rgbd_demo
```

Optional:

```bash
python3 demo/run_mock_rgbd_pipeline.py \
  --scene-dir mock_data/rgbd_scene_001 \
  --output-dir outputs/mock_rgbd_demo \
  --voxel-size-m 0.003 \
  --hand-side left \
  --overwrite-mock-data
```

Use `--hand-side left` to write a left-hand normalized NPZ
(`hand_side=0`, `is_right=False`). Use `--hand-side right` for right hand
(`hand_side=1`, `is_right=True`).
