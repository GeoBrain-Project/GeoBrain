"""
Error hierarchy for GeoBrain core.

Every error carries ``(object_name, field, expected, actual)`` so failures are
self-describing and trace straight back to the offending operator or field.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum


class ErrorCode(str, Enum):
    """Stable machine-readable codes for GeoBrain diagnostic failures."""

    CONFIG_INVALID = "GEOBRAIN_CONFIG_INVALID"
    ARTIFACT_INVALID = "GEOBRAIN_ARTIFACT_INVALID"
    SHAPE_MISMATCH = "GEOBRAIN_SHAPE_MISMATCH"
    UNIT_MISMATCH = "GEOBRAIN_UNIT_MISMATCH"
    DTYPE_UNSUPPORTED = "GEOBRAIN_DTYPE_UNSUPPORTED"
    DEVICE_UNAVAILABLE = "GEOBRAIN_DEVICE_UNAVAILABLE"
    CAPABILITY_UNAVAILABLE = "GEOBRAIN_CAPABILITY_UNAVAILABLE"
    EXECUTION_FAILED = "GEOBRAIN_EXECUTION_FAILED"


_CYCLE_MARKER = "<cycle>"


def _json_safe(value: object, active: set[int] | None = None) -> object:
    """Detach diagnostics into strict JSON values without executing them.

    Non-finite floats use the stable strings ``NaN``, ``Infinity``, and
    ``-Infinity``. Recursive references use ``<cycle>``.
    """
    if active is None:
        active = set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            return _CYCLE_MARKER
        active.add(identity)
        try:
            return {
                _safe_text(key): _json_safe(item, active)
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        identity = id(value)
        if identity in active:
            return _CYCLE_MARKER
        active.add(identity)
        try:
            return [_json_safe(item, active) for item in value]
        finally:
            active.remove(identity)
    return _safe_repr(value)


def _safe_text(value: object) -> str:
    try:
        return str(value)
    except Exception:
        return f"<unprintable {type(value).__qualname__}>"


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__qualname__}>"


class GeoBrainError(ValueError):
    """
    Base class for all GeoBrain errors.

    Folds structured diagnostic context: ``object_name``, ``field``,
    ``expected``, ``actual``; into the message and keeps it as attributes.

    Subclasses :class:`ValueError`: every GeoBrain failure is, at bottom, a
    bad value / shape / argument, so ``except ValueError`` catches them all.
    This lets the whole platform raise the structured :class:`GeoBrainError`
    uniformly (including the ``bayes`` / ``data`` / ``io`` families) without
    breaking callers (or tests) that catch the stdlib ``ValueError``.
    """

    default_code: ErrorCode = ErrorCode.EXECUTION_FAILED

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
        """
        Build a structured error message from the supplied context.

        Args:
            message: Human-readable description of what went wrong.
            object_name: Name of the object that raised the error.
            field: The specific field or argument at fault.
            expected: The expected value, type, or shape.
            actual: The actual value, type, or shape encountered.
            code: Stable machine-readable diagnostic code.
            hint: Optional actionable guidance for resolving the failure.
        """
        parts = [message]
        if object_name is not None:
            parts.append(f"object={object_name!r}")
        if field is not None:
            parts.append(f"field={field!r}")
        if expected is not None:
            parts.append(f"expected={expected!r}")
        if actual is not None:
            parts.append(f"actual={actual!r}")
        if hint is not None:
            parts.append(f"hint={hint}")
        super().__init__(" | ".join(parts))
        self.message = message
        self.object_name = object_name
        self.field = field
        self.expected = expected
        self.actual = actual
        self.code = (self.default_code if code is None else code).value
        self.hint = hint

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe structured diagnostics for UI and Agent clients."""
        return {
            "code": self.code,
            "message": self.message,
            "object_name": self.object_name,
            "field": self.field,
            "expected": _json_safe(self.expected),
            "actual": _json_safe(self.actual),
            "hint": self.hint,
        }


class MissingFieldError(GeoBrainError):
    """A required ModelState/ForwardOutput field is missing.

    Carries :attr:`ErrorCode.SHAPE_MISMATCH`: the small stable code
    vocabulary treats a missing field as the zero-arity case of a
    structural mismatch rather than growing a dedicated code.
    """

    default_code = ErrorCode.SHAPE_MISMATCH


class FieldShapeError(GeoBrainError):
    """A field exists but has the wrong shape or dtype."""

    default_code = ErrorCode.SHAPE_MISMATCH


class RegistryError(GeoBrainError):
    """Registry operation failed (duplicate key, unknown key, wrong type)."""

    default_code = ErrorCode.CONFIG_INVALID


class DifferentiabilityError(GeoBrainError):
    """Differentiability contract violated (e.g. a declared trainable input
    carrying an integer/bool dtype that cannot hold gradients)."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
