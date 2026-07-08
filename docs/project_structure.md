# Project Structure

This repository keeps the KR1 mock RGB-D pipeline separate from real dataset
assets and generated outputs.

```text
rgbd_hand_object_recon/
  configs/
    mock_rgbd.json
  demo/
    run_mock_rgbd_pipeline.py
  docs/
    hand_reconstruction_sop.md
    kr_delivery_submission_guideline.md
    mock_rgbd_io_schema.md
    project_structure.md
    reconstruction_accuracy_closed_loop.md
    root_translation_optimized_hands_npz_schema.md
    gitee_sync.md
  mock_data/
    .gitkeep
  scripts/
    create_upload_package.sh
    evaluate_normalized_npz_accuracy.py
    prepare_reinterhand_pilot.py
    run_icp_registration.py
    run_kr1_checks.sh
  src/
    hand_recon/
      icp.py
      normalized_output.py
      mock_data.py
      rgbd.py
      reconstruction.py
      pose.py
      evaluation.py
  tests/
    test_mock_rgbd_pipeline.py
```

## Directory Responsibilities

- `configs/`: lightweight config examples. These should be committed.
- `demo/`: runnable demo entrypoints. The KR1 demo is the main smoke test.
- `docs/`: schema, SOP, sync, and handoff documentation.
- `mock_data/`: placeholder only. The default mock scene is generated at runtime
  and ignored by git.
- `scripts/`: dataset preparation, ICP checks, and one-command verification.
- `src/hand_recon/`: importable Python modules.
- `tests/`: pytest tests for the mock RGB-D pipeline.
- `data/`: real datasets. Ignored by git because files can be very large.
- `outputs/`: generated artifacts. Ignored by git.
- `dist/`: upload package output. Ignored by git.

## Commit Policy

Commit source code, configs, docs, tests, and small placeholders.

Do not commit:

- Re:InterHand or other real datasets under `data/`
- generated point clouds and reports under `outputs/`
- generated mock scene arrays under `mock_data/rgbd_scene_001/`
- local virtual environments or Python caches
