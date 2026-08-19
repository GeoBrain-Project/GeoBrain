"""Structured diagnostic errors for EM contracts and execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from types import MappingProxyType
from typing import ClassVar, TypeAlias, cast

from geobrain.core.errors import ErrorCode, GeoBrainError


JSONScalar: TypeAlias = str | int | float | bool | None
FrozenJSON: TypeAlias = (
    JSONScalar | tuple["FrozenJSON", ...] | Mapping[str, "FrozenJSON"]
)


def _freeze_json(value: object, active: set[int]) -> FrozenJSON:
    """Copy one strict JSON value into immutable, deterministically sorted form.

    Raises plain ``TypeError`` (not the family error): these helpers validate
    the construction of the family errors themselves, so using them here
    would be circular; a malformed ``details`` payload is a programmer
    error at the raise site, not a physics contract violation.
    """
    if value is None or type(value) in (str, int, bool):
        return cast(JSONScalar, value)
    if type(value) is float:
        number = value
        if not math.isfinite(number):
            raise TypeError("EM error details require finite JSON numbers")
        return number
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise TypeError("EM error details cannot contain cycles")
        if any(type(key) is not str for key in value):
            raise TypeError("EM error detail mappings require string keys")
        active.add(identity)
        try:
            source = cast(Mapping[str, object], value)
            frozen = {
                key: _freeze_json(source[key], active)
                for key in sorted(source)
            }
        finally:
            active.remove(identity)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        identity = id(value)
        if identity in active:
            raise TypeError("EM error details cannot contain cycles")
        active.add(identity)
        try:
            return tuple(_freeze_json(item, active) for item in value)
        finally:
            active.remove(identity)
    raise TypeError(
        "EM error details accept only finite JSON scalar, mapping, and sequence "
        f"values; received {type(value).__qualname__}"
    )


def _thaw_json(value: FrozenJSON) -> object:
    """Return ordinary JSON containers for serialization at the boundary."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class _EMError(GeoBrainError):  # type: ignore[misc]  # isolated strict import boundary
    """Base that owns strict immutable EM-only diagnostic details."""

    allowed_codes: ClassVar[frozenset[ErrorCode]]

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object],
        object_name: str | None = None,
        field: str | None = None,
        expected: object = None,
        actual: object = None,
        code: ErrorCode | None = None,
        hint: str | None = None,
    ) -> None:
        """Build an error without retaining mutable or executable payloads."""
        if type(message) is not str or not message.strip():
            raise TypeError("EM error message must be a non-empty string")
        if not isinstance(details, Mapping) or not details:
            raise TypeError("EM error details must be a non-empty mapping")
        selected = self.default_code if code is None else code
        if not isinstance(selected, ErrorCode) or selected not in self.allowed_codes:
            raise TypeError(
                f"{type(self).__name__} does not map to core code {selected!r}"
            )
        frozen = _freeze_json(details, set())
        if not isinstance(frozen, Mapping):
            raise TypeError("EM error details must be a mapping")
        self.details = frozen
        super().__init__(
            message,
            object_name=object_name,
            field=field,
            expected=expected,
            actual=actual,
            code=selected,
            hint=hint,
        )

    def to_dict(self) -> dict[str, object]:
        """Return stable core diagnostics plus detached JSON-safe EM details."""
        payload = cast(dict[str, object], super().to_dict())
        payload["details"] = _thaw_json(self.details)
        return payload


class EMContractError(_EMError):
    """An EM configuration, data, unit, dtype, device, or artifact is invalid."""

    default_code = ErrorCode.CONFIG_INVALID
    allowed_codes = frozenset(
        {
            ErrorCode.CONFIG_INVALID,
            ErrorCode.ARTIFACT_INVALID,
            ErrorCode.SHAPE_MISMATCH,
            ErrorCode.UNIT_MISMATCH,
            ErrorCode.DTYPE_UNSUPPORTED,
            ErrorCode.DEVICE_UNAVAILABLE,
        }
    )


class EMCapabilityError(_EMError):
    """A requested EM dtype, device, solver, mesh, gradient, or layout is unavailable."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
    allowed_codes = frozenset(
        {
            ErrorCode.CAPABILITY_UNAVAILABLE,
            ErrorCode.DTYPE_UNSUPPORTED,
            ErrorCode.DEVICE_UNAVAILABLE,
        }
    )


class EMNumericsError(_EMError):
    """An EM factorization, solve, convergence, or transform failed."""

    default_code = ErrorCode.EXECUTION_FAILED
    allowed_codes = frozenset({ErrorCode.EXECUTION_FAILED})


class EMResourceError(_EMError):
    """A predicted EM allocation exceeds the explicit resource budget."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
    allowed_codes = frozenset({ErrorCode.CAPABILITY_UNAVAILABLE})


__all__ = [
    "EMCapabilityError",
    "EMContractError",
    "EMNumericsError",
    "EMResourceError",
]
