"""Validated configuration for the mock RGB-D pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hand_recon.exceptions import ConfigurationError
from hand_recon.mock_data import LABEL_BACKGROUND, LABEL_HAND, LABEL_OBJECT

EXPECTED_MASK_LABELS = {
    "background": LABEL_BACKGROUND,
    "hand": LABEL_HAND,
    "object": LABEL_OBJECT,
}


@dataclass(frozen=True)
class MockRgbdConfig:
    """Runtime options accepted by the mock reconstruction workflow."""

    scene_dir: Path = Path("mock_data/rgbd_scene_001")
    output_dir: Path = Path("outputs/mock_rgbd_demo")
    voxel_size_m: float = 0.003
    hand_side: str = "right"
    overwrite_mock_data: bool = False

    def __post_init__(self) -> None:
        if not self.scene_dir:
            raise ConfigurationError("scene_dir must not be empty")
        if not self.output_dir:
            raise ConfigurationError("output_dir must not be empty")
        if not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0:
            raise ConfigurationError("voxel_size_m must be greater than zero")
        if self.hand_side not in {"left", "right"}:
            raise ConfigurationError("hand_side must be 'left' or 'right'")


def load_mock_rgbd_config(path: Path) -> MockRgbdConfig:
    """Load the committed mock JSON config with strict, actionable errors."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"invalid JSON in {config_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration root must be a JSON object: {config_path}")
    return mock_rgbd_config_from_mapping(raw, source=config_path)


def mock_rgbd_config_from_mapping(raw: Mapping[str, Any], *, source: Path | str = "configuration") -> MockRgbdConfig:
    """Build :class:`MockRgbdConfig` while rejecting misspelled keys."""

    allowed = {
        "scene_dir",
        "output_dir",
        "voxel_size_m",
        "hand_side",
        "overwrite_mock_data",
        "depth_unit",
        "mask_labels",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown keys in {source}: {', '.join(unknown)}")

    depth_unit = str(raw.get("depth_unit", "meter"))
    if depth_unit != "meter":
        raise ConfigurationError(f"depth_unit in {source} must be 'meter', got {depth_unit!r}")

    mask_labels = raw.get("mask_labels", EXPECTED_MASK_LABELS)
    if mask_labels != EXPECTED_MASK_LABELS:
        raise ConfigurationError(f"mask_labels in {source} must equal {EXPECTED_MASK_LABELS}, got {mask_labels!r}")

    overwrite_mock_data = raw.get("overwrite_mock_data", False)
    if not isinstance(overwrite_mock_data, bool):
        raise ConfigurationError(f"overwrite_mock_data in {source} must be a boolean")

    try:
        return MockRgbdConfig(
            scene_dir=Path(raw.get("scene_dir", MockRgbdConfig.scene_dir)),
            output_dir=Path(raw.get("output_dir", MockRgbdConfig.output_dir)),
            voxel_size_m=float(raw.get("voxel_size_m", MockRgbdConfig.voxel_size_m)),
            hand_side=str(raw.get("hand_side", MockRgbdConfig.hand_side)),
            overwrite_mock_data=overwrite_mock_data,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid value in {source}: {exc}") from exc
