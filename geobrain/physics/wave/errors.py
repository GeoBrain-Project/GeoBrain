"""Structured diagnostic errors for the Wave physics contracts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core import ErrorCode, GeoBrainError


class WaveContractError(GeoBrainError):  # type: ignore[misc]  # isolated strict import boundary
    """A Wave configuration or public-data contract is invalid."""

    default_code = ErrorCode.CONFIG_INVALID


class WaveCapabilityError(GeoBrainError):  # type: ignore[misc]  # isolated strict import boundary
    """A requested Wave capability is unavailable."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE


class WaveNumericsError(GeoBrainError):  # type: ignore[misc]  # isolated strict import boundary
    """A Wave numerical operation could not complete."""

    default_code = ErrorCode.EXECUTION_FAILED


class WaveResourceError(GeoBrainError):  # type: ignore[misc]  # isolated strict import boundary
    """A requested Wave resource budget or resource is unavailable."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE


__all__ = [
    "WaveCapabilityError",
    "WaveContractError",
    "WaveNumericsError",
    "WaveResourceError",
]
