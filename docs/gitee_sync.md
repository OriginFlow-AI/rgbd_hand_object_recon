# Gitee Sync Guide

Use this guide when the Gitee repository URL is available.

## Option A: Initialize This Directory

```bash
cd /home/wangjh/code/rgbd_hand_object_recon
git init
git add README.md requirements.txt .gitignore configs demo docs mock_data scripts src tests third_party
git commit -m "Add KR1 mock RGB-D reconstruction demo"
git branch -M main
git remote add origin <gitee_repo_url>
git push -u origin main
```

Use this when Gitee has an empty repository waiting for the initial push.

## Option B: Clone Gitee First, Then Copy Code

```bash
cd /home/wangjh/code
git clone <gitee_repo_url> rgbd_hand_object_recon_gitee
rsync -av \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'data' \
  --exclude 'outputs' \
  --exclude 'mock_data/rgbd_scene_001' \
  --exclude 'dist' \
  /home/wangjh/code/rgbd_hand_object_recon/ /home/wangjh/code/rgbd_hand_object_recon_gitee/
cd /home/wangjh/code/rgbd_hand_object_recon_gitee
git add README.md requirements.txt .gitignore configs demo docs mock_data scripts src tests third_party
git commit -m "Add KR1 mock RGB-D reconstruction demo"
git push
```

Use this when the Gitee repository already contains files or branch protection.

## Pre-Push Check

Before pushing, run:

```bash
bash scripts/run_kr1_checks.sh
```

On a fresh machine, use the bootstrap script instead:

```bash
bash scripts/bootstrap_kr1_demo.sh
```

Expected result:

- mock RGB-D demo exits with `quality_passed=true`
- `tests/test_mock_rgbd_pipeline.py` passes
- `scripts/run_icp_registration.py --selftest` reports `status=ok`

## Files Intended For Gitee

- `.gitignore`
- `README.md`
- `requirements.txt`
- `configs/mock_rgbd.json`
- `demo/run_mock_rgbd_pipeline.py`
- `docs/*.md`
- `mock_data/.gitkeep`
- `scripts/*.py`
- `scripts/*.sh`
- `src/hand_recon/*.py`
- `tests/test_mock_rgbd_pipeline.py`
- `third_party/reinterhand_download/*.py`

Large data and generated outputs are intentionally ignored.
