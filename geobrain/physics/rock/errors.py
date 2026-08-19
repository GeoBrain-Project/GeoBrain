"""Structured diagnostics for rock-family contracts and execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core import ErrorCode, GeoBrainError


class _RockError(GeoBrainError):  # type: ignore[misc]  # isolated strict import boundary
    """GeoBrain error with a non-empty Rock-specific remediation."""

    default_hint = "correct the reported field to match the expected Rock contract"

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
        resolved_hint = (
            hint
            if isinstance(hint, str) and hint.strip()
            else self.default_hint
        )
        super().__init__(
            message,
            object_name=object_name,
            field=field,
            expected=expected,
            actual=actual,
            code=code,
            hint=resolved_hint,
        )


class RockContractError(_RockError):
    """A Rock input, schema, shape, dtype, device, or unit contract failed."""

    default_code = ErrorCode.CONFIG_INVALID


class RockCapabilityError(_RockError):
    """A requested Rock model or execution combination is unavailable."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
    default_hint = "choose a capability listed by the Rock model report"


class RockNumericsError(_RockError):
    """A Rock calculation failed its declared numerical contract."""

    default_code = ErrorCode.EXECUTION_FAILED
    default_hint = (
        "adjust iteration controls or choose inputs inside the calibrated domain"
    )


class RockResourceError(_RockError):
    """A Rock calculation exceeds its caller-provided resource budget."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
    default_hint = "reduce broadcast size or disable gradient recording"


__all__ = [
    "RockCapabilityError",
    "RockContractError",
    "RockNumericsError",
    "RockResourceError",
]
