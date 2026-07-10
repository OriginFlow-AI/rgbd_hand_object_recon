#!/usr/bin/env python3
"""CLI wrapper for the Re:InterHand best-data visual report generator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hand_recon.reports.best_data_visual import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
