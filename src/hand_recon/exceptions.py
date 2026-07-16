"""Domain exceptions exposed by :mod:`hand_recon`.

All validation exceptions also inherit from :class:`ValueError` so existing
callers that already handle ``ValueError`` remain compatible.
"""

from __future__ import annotations


class HandReconError(Exception):
    """Base class for expected hand-reconstruction failures."""


class ConfigurationError(HandReconError, ValueError):
    """Raised when a configuration file or option is invalid."""


class DataValidationError(HandReconError, ValueError):
    """Raised when an input dataset violates its declared contract."""


class UnsafeDataError(DataValidationError):
    """Raised when loading an input would require unsafe deserialization."""
