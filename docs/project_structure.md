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
    kr3_hand_result_interface.md
    mock_rgbd_io_schema.md
    multi_agent_closed_loop_task.md
    project_structure.md
    reconstruction_accuracy_closed_loop.md
    root_translation_optimized_hands_npz_schema.md
    gitee_sync.md
    reports/
      kr1_delivery_report.md
      kr3_delivery_report.md
    integration/
      software_integration.md
      migration_to_software_module.md
  examples/
    software_api_demo.py
  schemas/
    kr3/
      hand_result_schema.json
  mock_data/
    .gitkeep
  scripts/
    create_upload_package.sh
    evaluate_normalized_npz_accuracy.py
    generate_best_data_visual_report.py
    generate_hand_visual_report.py
    prepare_reinterhand_pilot.py
    generate_multi_agent_validation_report.py
    run_icp_registration.py
    run_best_data_visual_report.sh
    run_kr3_checks.sh
    run_hand_visual_report.sh
    run_multi_agent_validation_report.sh
    run_kr1_checks.sh
  src/
    hand_recon/
      api.py
      adapters/
        hand_result.py
      core/
        __init__.py
      io/
        __init__.py
      pipelines/
        mock_rgbd.py
        reinterhand.py
      reports/
        hand_visual.py
        best_data_visual.py
      interfaces/
        hand_result.py
      icp.py
      normalized_output.py
      mock_data.py
      rgbd.py
      reconstruction.py
      pose.py
      evaluation.py
  tests/
    test_public_api.py
    test_kr3_hand_result_interface.py
    test_mock_rgbd_pipeline.py
```

## Directory Responsibilities

- `configs/`: lightweight config examples. These should be committed.
- `demo/`: runnable demo entrypoints. Demo scripts should delegate to `src/hand_recon/pipelines/`.
- `docs/`: schema, SOP, sync, and handoff documentation.
- `schemas/`: machine-readable interface schemas. `schemas/kr3/` reserves the hand result exchange format for ground truth, DMA vision, and super-labelator adapters.
- `examples/`: software-integration examples that use public package APIs.
- `mock_data/`: placeholder only. The default mock scene is generated at runtime
  and ignored by git.
- `scripts/`: dataset preparation, report generation, and one-command verification. Scripts are CLI wrappers, not the primary software API.
- `src/hand_recon/`: importable Python modules. Use `hand_recon.api` or top-level `hand_recon` exports for software integration.
- `tests/`: pytest tests for the KR1 mock RGB-D pipeline and KR3 interface reservation.
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
