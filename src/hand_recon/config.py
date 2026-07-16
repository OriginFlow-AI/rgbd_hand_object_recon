"""Validated configuration for the mock RGB-D pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
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
class HandSurfaceConfig:
    """Numerical controls tied to sensor resolution and resource bounds."""

    voxel_size_m: float = 0.003
    truncation_m: float = 0.009
    padding_m: float = 0.012
    min_weight: float = 1.0
    max_voxel_count: int = 2_000_000

    def __post_init__(self) -> None:
        for name in ("voxel_size_m", "truncation_m", "padding_m", "min_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"surface.{name} must be a positive finite value")
        if self.truncation_m < 2.0 * self.voxel_size_m:
            raise ConfigurationError("surface.truncation_m must be at least twice surface.voxel_size_m")
        if self.max_voxel_count <= 0:
            raise ConfigurationError("surface.max_voxel_count must be greater than zero")


@dataclass(frozen=True)
class MockRgbdConfig:
    """Runtime options accepted by the mock reconstruction workflow."""

    scene_dir: Path = Path("mock_data/rgbd_scene_001")
    output_dir: Path = Path("outputs/mock_rgbd_demo")
    voxel_size_m: float = 0.003
    hand_side: str = "right"
    overwrite_mock_data: bool = False
    surface: HandSurfaceConfig = field(default_factory=HandSurfaceConfig)

    def __post_init__(self) -> None:
        if not self.scene_dir:
            raise ConfigurationError("scene_dir must not be empty")
        if not self.output_dir:
            raise ConfigurationError("output_dir must not be empty")
        if not math.isfinite(self.voxel_size_m) or self.voxel_size_m <= 0:
            raise ConfigurationError("voxel_size_m must be greater than zero")
        if self.hand_side not in {"left", "right"}:
            raise ConfigurationError("hand_side must be 'left' or 'right'")
        if not isinstance(self.surface, HandSurfaceConfig):
            raise ConfigurationError("surface must be a HandSurfaceConfig")


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
        "surface",
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
        surface_raw = raw.get("surface", {})
        if not isinstance(surface_raw, Mapping):
            raise ConfigurationError(f"surface in {source} must be an object")
        return MockRgbdConfig(
            scene_dir=Path(raw.get("scene_dir", MockRgbdConfig.scene_dir)),
            output_dir=Path(raw.get("output_dir", MockRgbdConfig.output_dir)),
            voxel_size_m=float(raw.get("voxel_size_m", MockRgbdConfig.voxel_size_m)),
            hand_side=str(raw.get("hand_side", MockRgbdConfig.hand_side)),
            overwrite_mock_data=overwrite_mock_data,
            surface=_surface_config_from_mapping(surface_raw, source=source),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid value in {source}: {exc}") from exc


def _surface_config_from_mapping(raw: Mapping[str, Any], *, source: Path | str) -> HandSurfaceConfig:
    allowed = {"voxel_size_m", "truncation_m", "padding_m", "min_weight", "max_voxel_count"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown surface keys in {source}: {', '.join(unknown)}")
    try:
        return HandSurfaceConfig(
            voxel_size_m=float(raw.get("voxel_size_m", HandSurfaceConfig.voxel_size_m)),
            truncation_m=float(raw.get("truncation_m", HandSurfaceConfig.truncation_m)),
            padding_m=float(raw.get("padding_m", HandSurfaceConfig.padding_m)),
            min_weight=float(raw.get("min_weight", HandSurfaceConfig.min_weight)),
            max_voxel_count=int(raw.get("max_voxel_count", HandSurfaceConfig.max_voxel_count)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid surface value in {source}: {exc}") from exc
