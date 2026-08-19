"""Typed errors for the Geomodel scientific domain.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from ..core import ErrorCode, GeoBrainError


class GeomodelContractError(GeoBrainError):  # type: ignore[misc, unused-ignore]
    """A Geomodel geometry, metadata, or configuration contract is invalid."""

    default_code = ErrorCode.CONFIG_INVALID


class GeomodelCapabilityError(GeoBrainError):  # type: ignore[misc, unused-ignore]
    """The selected Geomodel capability is unavailable or unsupported."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE


class GeomodelNumericsError(GeoBrainError):  # type: ignore[misc, unused-ignore]
    """A Geomodel numerical operation failed its scientific contract."""

    default_code = ErrorCode.EXECUTION_FAILED


class GeomodelResourceError(GeoBrainError):  # type: ignore[misc, unused-ignore]
    """A Geomodel request exceeds an explicit resource budget."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE


__all__ = [
    "GeomodelCapabilityError",
    "GeomodelContractError",
    "GeomodelNumericsError",
    "GeomodelResourceError",
]
