"""JSON serialization helpers with contextual errors and atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from hand_recon.exceptions import DataValidationError


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and report the file and exact parse location."""

    json_path = Path(path)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise DataValidationError(
            f"invalid JSON in {json_path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise DataValidationError(f"JSON root must be an object: {json_path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a UTF-8 JSON document."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
