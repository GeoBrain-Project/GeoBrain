"""Structured diagnostics for Flow contracts and execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import cast

from geobrain.core import ErrorCode, GeoBrainError


class _FlowError(GeoBrainError):  # type: ignore[misc]  # skipped import boundary
    """GeoBrain error carrying a non-empty Flow remediation hint."""

    default_hint = "correct the reported value to satisfy the Flow contract"

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
        diagnostics: object | None = None,
    ) -> None:
        resolved_hint = hint if isinstance(hint, str) and hint.strip() else self.default_hint
        super().__init__(
            message,
            object_name=object_name,
            field=field,
            expected=expected,
            actual=actual,
            code=code,
            hint=resolved_hint,
        )
        self.diagnostics = diagnostics

    def to_dict(self) -> dict[str, object]:
        """Include the typed Flow diagnostic record in Agent/UI payloads."""
        payload = cast(dict[str, object], super().to_dict())
        diagnostics = self.diagnostics
        if diagnostics is not None:
            to_dict = getattr(diagnostics, "to_dict", None)
            payload["diagnostics"] = to_dict() if callable(to_dict) else diagnostics
        return payload


class FlowContractError(_FlowError):
    """A Flow schema, tensor, unit, or imported-data contract failed."""

    default_code = ErrorCode.CONFIG_INVALID


class FlowCapabilityError(_FlowError):
    """A requested Flow model or execution combination is unavailable."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
    default_hint = "choose a capability declared by the selected Flow model"


class FlowConvergenceError(_FlowError):
    """A Flow nonlinear, linear, flash, or timestep solve did not converge."""

    default_code = ErrorCode.EXECUTION_FAILED
    default_hint = "inspect the last diagnostics and revise the solver or timestep controls"


class FlowResourceError(_FlowError):
    """A Flow operation cannot satisfy its explicit resource budget."""

    default_code = ErrorCode.CAPABILITY_UNAVAILABLE
    default_hint = "reduce the problem size or increase the explicit Flow resource budget"


__all__ = [
    "FlowCapabilityError",
    "FlowContractError",
    "FlowConvergenceError",
    "FlowResourceError",
]
