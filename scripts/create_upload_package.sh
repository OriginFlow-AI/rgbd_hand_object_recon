#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT/dist"
STAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE_NAME="rgbd_hand_object_recon_upload_${STAMP}"
STAGING_DIR="$DIST_DIR/$PACKAGE_NAME"
ARCHIVE_PATH="$DIST_DIR/${PACKAGE_NAME}.tar.gz"

mkdir -p "$DIST_DIR"
rm -rf "$STAGING_DIR"

rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude 'env' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.mypy_cache' \
  --exclude '.ruff_cache' \
  --exclude 'data' \
  --exclude 'outputs' \
  --exclude 'mock_data/rgbd_scene_001' \
  --exclude 'dist' \
  --exclude '*.pyc' \
  "$ROOT"/ "$STAGING_DIR"/

tar -C "$DIST_DIR" -czf "$ARCHIVE_PATH" "$PACKAGE_NAME"

printf 'Created upload directory: %s\n' "$STAGING_DIR"
printf 'Created upload archive:   %s\n' "$ARCHIVE_PATH"
du -h "$ARCHIVE_PATH"
