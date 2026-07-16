# Project Structure

```text
rgbd_hand_object_recon/
  pyproject.toml              # package metadata, dependencies, pytest/Ruff config
  Makefile                    # setup/demo/test/lint/check targets
  run.sh                      # shortest visible-result entrypoint
  configs/                    # committed runtime configuration
  demo/                       # backward-compatible thin demo wrappers
  examples/                   # public API integration examples
  schemas/kr3/                # machine-readable hand-result contract
  scripts/                    # dataset/report/verification CLI wrappers
  src/hand_recon/
    api.py                    # stable software integration API
    cli.py, __main__.py       # installed/module CLI
    config.py, exceptions.py  # validated configuration and domain errors
    rgbd.py                   # RGB-D scene model, loading, backprojection
    reconstruction.py         # multi-view world-frame fusion
    icp.py                    # point-cloud I/O and rigid ICP primitives
    pose.py, evaluation.py    # mock pose and quality gates
    normalized_output.py      # normalized hand payload
    io/                       # safe/atomic JSON and NPZ helpers
    pipelines/                # workflow orchestration
    interfaces/, adapters/    # KR3 contract and compatibility surface
    reports/                  # self-contained HTML reports
  tests/                      # unit, contract, security, CLI, regression tests
  third_party/                # vendored Re:InterHand lists plus local safety wrapper
  docs/                       # current design/usage docs and historical reports
  mock_data/                  # generated mock input (ignored except .gitkeep)
  data/                       # real datasets (ignored)
  outputs/                    # generated artifacts (ignored)
  dist/                       # upload packages (ignored)
```

## Dependency rules

- Downstream applications import top-level `hand_recon` or `hand_recon.api`.
- Pipelines may depend on geometry, interfaces and I/O; geometry must not depend on pipelines.
- Scripts delegate to package modules. New business logic does not belong in `scripts/`.
- `third_party/` is not imported by the package. The maintained selective downloader is
  `scripts/prepare_reinterhand_pilot.py`.
- Generated data and local environments are never source dependencies.

## Commit policy

Commit source, schema, configs, docs, tests and small placeholders. Do not commit real datasets,
generated reports/point clouds, upload archives, virtual environments or caches. `create_upload_package.sh`
uses the same boundary.
