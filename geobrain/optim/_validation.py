"""
Private scalar coercion shared by optimizer execution contracts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import operator
from numbers import Integral, Real
from typing import SupportsIndex, cast

import torch

from geobrain.core import GeoBrainError


def _numeric_error(
    *,
    owner: str,
    field: str,
    expected: object,
    actual: object,
) -> GeoBrainError:
    """Build one structured numeric-boundary diagnostic."""
    return GeoBrainError(
        f"{owner}.{field} must be {expected}",
        object_name=owner,
        field=field,
        expected=expected,
        actual=actual,
    )


def _tensor_scalar_value(
    value: torch.Tensor,
    *,
    owner: str,
    field: str,
    expected: object,
) -> object:
    """Read a materialized zero-dimensional tensor without leaking backend errors."""
    try:
        return value.item()
    except (RuntimeError, TypeError, ValueError, OverflowError):
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=type(value),
        ) from None


def _coerce_real_scalar(
    value: object,
    *,
    owner: str,
    field: str,
    finite: bool,
    minimum: float | None = None,
    maximum_exclusive: float | None = None,
) -> float:
    """Coerce a Python/NumPy real or zero-dimensional real tensor."""
    expected = "finite real scalar" if finite else "real scalar"
    if minimum is not None:
        expected += f" >= {minimum}"
    if maximum_exclusive is not None:
        expected += f" and < {maximum_exclusive}"

    scalar: object
    if isinstance(value, torch.Tensor):
        if (
            value.ndim != 0
            or value.dtype is torch.bool
            or value.is_complex()
        ):
            raise _numeric_error(
                owner=owner,
                field=field,
                expected=expected,
                actual=type(value),
            )
        scalar = _tensor_scalar_value(
            value,
            owner=owner,
            field=field,
            expected=expected,
        )
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=type(value),
        )
    else:
        scalar = value

    try:
        normalized = float(scalar)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=type(value),
        ) from None
    if finite and not math.isfinite(normalized):
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=value,
        )
    if minimum is not None and normalized < minimum:
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=value,
        )
    if maximum_exclusive is not None and normalized >= maximum_exclusive:
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=value,
        )
    return normalized


def _coerce_integral_scalar(
    value: object,
    *,
    owner: str,
    field: str,
    minimum: int,
) -> int:
    """Coerce a Python/NumPy integer or zero-dimensional integer tensor."""
    expected = f"integer >= {minimum}"
    scalar: object
    if isinstance(value, torch.Tensor):
        if (
            value.ndim != 0
            or value.dtype is torch.bool
            or value.is_floating_point()
            or value.is_complex()
        ):
            raise _numeric_error(
                owner=owner,
                field=field,
                expected=expected,
                actual=type(value),
            )
        scalar = _tensor_scalar_value(
            value,
            owner=owner,
            field=field,
            expected=expected,
        )
    elif isinstance(value, bool) or not isinstance(value, Integral):
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=type(value),
        )
    else:
        scalar = value

    try:
        normalized = operator.index(cast(SupportsIndex, scalar))
    except (TypeError, ValueError, OverflowError):
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=type(value),
        ) from None
    if normalized < minimum:
        raise _numeric_error(
            owner=owner,
            field=field,
            expected=expected,
            actual=value,
        )
    return normalized
