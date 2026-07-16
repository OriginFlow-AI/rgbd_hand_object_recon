"""Time conversion helpers shared by pipelines and adapters."""

from __future__ import annotations

from datetime import datetime, timezone

from hand_recon.exceptions import DataValidationError


def parse_iso8601_ns(value: str) -> int:
    """Convert a timezone-aware ISO-8601 timestamp to Unix nanoseconds."""

    if not isinstance(value, str) or not value.strip():
        raise DataValidationError("timestamp must be a non-empty ISO-8601 string")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise DataValidationError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DataValidationError(f"timestamp must include a UTC offset: {value!r}")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    timestamp_ns = (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000
    if timestamp_ns < 0:
        raise DataValidationError(f"timestamp must not predate the Unix epoch: {value!r}")
    return timestamp_ns
