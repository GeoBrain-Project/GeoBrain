"""Structured errors for Potential public contracts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from geobrain.core import ErrorCode, GeoBrainError


def _type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _diagnostic_key(value: object) -> str:
    """Return a stable JSON-object key without invoking arbitrary ``str``."""
    if isinstance(value, str):
        return value
    if value is None:
        return "<key:null>"
    if isinstance(value, bool):
        return f"<key:builtins.bool:{'true' if value else 'false'}>"
    if isinstance(value, int):
        return f"<key:builtins.int:{value}>"
    if isinstance(value, float):
        if math.isnan(value):
            return "<key:builtins.float:NaN>"
        if math.isinf(value):
            sign = "Infinity" if value > 0.0 else "-Infinity"
            return f"<key:builtins.float:{sign}>"
        return f"<key:builtins.float:{value}>"
    return f"<key:{_type_name(value)}>"


def _diagnostic_value(value: object, active: set[int] | None = None) -> object:
    """Return deterministic JSON-native diagnostic data without object reprs."""
    if active is None:
        active = set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.layout):
        return str(value)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return "<cycle>"
        active.add(identity)
        try:
            return {
                _diagnostic_key(key): _diagnostic_value(item, active)
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        identity = id(value)
        if identity in active:
            return "<cycle>"
        active.add(identity)
        try:
            return [_diagnostic_value(item, active) for item in value]
        finally:
            active.remove(identity)
    return {"type": _type_name(value)}


def _diagnostic_label(value: object, *, role: str, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return f"<{role}:{_type_name(value)}>"


def _diagnostic_hint(value: object) -> str:
    default = "Review the Potential contract and provide a supported value."
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return f"{default} Invalid hint type: {_type_name(value)}."


class _PotentialError(GeoBrainError):
    """Normalize Potential diagnostics before delegating to the core error."""

    def __init__(
        self,
        message: str,
        *,
        object_name: str | None = None,
        field: str | None = None,
        expected: object = None,
        actual: object = None,
        code: ErrorCode | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(
            message,
            object_name=_diagnostic_label(
                object_name,
                role="object_name",
                default=type(self).__name__,
            ),
            field=_diagnostic_label(field, role="field", default="value"),
            expected=(
                "valid Potential contract value"
                if expected is None
                else _diagnostic_value(expected)
            ),
            actual=_diagnostic_value(actual),
            code=code,
            hint=_diagnostic_hint(hint),
        )


class PotentialContractError(_PotentialError):
    """Potential schema, shape, dtype, device, unit, or geometry failure."""

    default_code = ErrorCode.CONFIG_INVALID


class PotentialCapabilityError(_PotentialError):
    """Potential component, strategy, dtype, or device is unsupported."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE


class PotentialResourceError(_PotentialError):
    """Potential resource preflight cannot satisfy the requested policy."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE


class PotentialNumericsError(_PotentialError):
    """Potential execution failed numerically after valid preflight."""

    default_code = ErrorCode.EXECUTION_FAILED


__all__ = [
    "PotentialCapabilityError",
    "PotentialContractError",
    "PotentialNumericsError",
    "PotentialResourceError",
]
