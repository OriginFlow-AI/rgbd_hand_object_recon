from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hand_recon import __version__
from hand_recon.config import MockRgbdConfig, load_mock_rgbd_config, mock_rgbd_config_from_mapping
from hand_recon.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "hand_recon", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_module_cli_and_package_metadata() -> None:
    result = _cli("--help")

    assert result.returncode == 0
    assert "demo" in result.stdout
    assert "verify" in result.stdout
    assert __version__ == "0.2.0"


def test_committed_config_is_loaded_and_validated() -> None:
    config = load_mock_rgbd_config(ROOT / "configs" / "mock_rgbd.json")

    assert config == MockRgbdConfig()
    with pytest.raises(ConfigurationError, match="unknown keys"):
        mock_rgbd_config_from_mapping({"voxel_szie_m": 0.1})
    with pytest.raises(ConfigurationError, match="greater than zero"):
        mock_rgbd_config_from_mapping({"voxel_size_m": 0})
    with pytest.raises(ConfigurationError, match="must be a boolean"):
        mock_rgbd_config_from_mapping({"overwrite_mock_data": "false"})
    with pytest.raises(ConfigurationError, match="unknown surface keys"):
        mock_rgbd_config_from_mapping({"surface": {"voxel_szie_m": 0.003}})


def test_cli_returns_actionable_config_error(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text('{"voxel_size_m": 0}', encoding="utf-8")

    result = _cli("demo", "--config", str(config_path))

    assert result.returncode == 2
    assert "voxel_size_m must be greater than zero" in result.stderr


def test_shell_entrypoints_have_valid_syntax() -> None:
    scripts = [ROOT / "run.sh", *sorted((ROOT / "scripts").glob("*.sh"))]
    result = subprocess.run(["bash", "-n", *map(str, scripts)], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
