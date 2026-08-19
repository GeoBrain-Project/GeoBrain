"""Canonical-SI validation for compositional model state inputs.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real
from typing import SupportsFloat, SupportsIndex, cast

import torch

from .errors import FlowContractError


def positive_real(value: object, *, object_name: str, field: str) -> float:
    """Return a finite positive configuration scalar."""

    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise FlowContractError(
            f"{field} must be positive and finite",
            object_name=object_name,
            field=field,
            expected="> 0 in canonical SI units",
            actual=value,
        )
    return float(value)


def nonnegative_real(value: object, *, object_name: str, field: str) -> float:
    """Return a finite non-negative configuration scalar."""

    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise FlowContractError(
            f"{field} must be non-negative and finite",
            object_name=object_name,
            field=field,
            expected=">= 0 in canonical SI units",
            actual=value,
        )
    return float(value)


def phase_model_config(
    *,
    temperature: object,
    mu_liquid: object,
    mu_vapor: object,
    swl: object,
    sgr: object,
    n_l: object,
    n_v: object,
    object_name: str,
) -> tuple[float, float, float, float, float, float, float]:
    """Validate shared isothermal two-phase configuration scalars."""

    T = positive_real(temperature, object_name=object_name, field="temperature_k")
    mu_l = positive_real(
        mu_liquid, object_name=object_name, field="liquid_viscosity_pa_s"
    )
    mu_v = positive_real(
        mu_vapor, object_name=object_name, field="vapor_viscosity_pa_s"
    )
    try:
        numeric_input = str | bytes | bytearray | SupportsFloat | SupportsIndex
        swl_value = float(cast(numeric_input, swl))
        sgr_value = float(cast(numeric_input, sgr))
    except (TypeError, ValueError) as error:
        raise FlowContractError(
            "residual saturations must be real scalars",
            object_name=object_name,
            field="swl/sgr",
            expected="real scalars",
            actual=(swl, sgr),
        ) from error
    if not (
        math.isfinite(swl_value)
        and math.isfinite(sgr_value)
        and 0.0 <= swl_value < 1.0
        and 0.0 <= sgr_value < 1.0 - swl_value
    ):
        raise FlowContractError(
            "residual saturations must leave a movable interval",
            object_name=object_name,
            field="swl/sgr",
            expected="0 <= swl, sgr and swl + sgr < 1",
            actual=(swl, sgr),
        )
    exponent_l = positive_real(n_l, object_name=object_name, field="n_l")
    exponent_v = positive_real(n_v, object_name=object_name, field="n_v")
    return T, mu_l, mu_v, swl_value, sgr_value, exponent_l, exponent_v


def cell_scalar_input(
    value: object,
    *,
    n_cells: int,
    dtype: torch.dtype,
    device: torch.device,
    field: str,
    positive: bool,
    object_name: str = "compositional.initial_state",
) -> torch.Tensor:
    """Expand a scalar literal or validate a live scalar/cell tensor in place."""

    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise FlowContractError(
                f"{field} must be floating point",
                object_name=object_name,
                field=field,
                expected="floating tensor",
                actual=str(value.dtype),
            )
        if value.dtype != dtype:
            raise FlowContractError(
                f"{field} dtype must match the model dtype",
                object_name=object_name,
                field=f"{field}.dtype",
                expected=str(dtype),
                actual=str(value.dtype),
            )
        if value.device != device:
            raise FlowContractError(
                f"{field} device must match the model device",
                object_name=object_name,
                field=f"{field}.device",
                expected=str(device),
                actual=str(value.device),
            )
        if value.shape == ():
            result = value.expand(n_cells)
        elif value.shape == (n_cells,):
            result = value
        else:
            raise FlowContractError(
                f"{field} shape is invalid",
                object_name=object_name,
                field=field,
                expected=((), (n_cells,)),
                actual=tuple(value.shape),
            )
    elif isinstance(value, Real) and not isinstance(value, bool):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise FlowContractError(
                f"{field} must be finite",
                object_name=object_name,
                field=field,
                expected="finite scalar",
                actual=value,
            )
        result = torch.full((n_cells,), scalar, dtype=dtype, device=device)
    else:
        raise FlowContractError(
            f"{field} must be a scalar literal or floating tensor",
            object_name=object_name,
            field=field,
            expected="real scalar or floating tensor",
            actual=type(value).__name__,
        )
    invalid = ~torch.isfinite(result)
    if positive:
        invalid = invalid | (result <= 0)
    if bool(invalid.any()):
        relation = "> 0" if positive else "finite"
        raise FlowContractError(
            f"{field} is outside its physical domain",
            object_name=object_name,
            field=field,
            expected=relation,
            actual="contains a non-finite or out-of-domain value",
        )
    return result


def composition_input(
    value: object,
    *,
    n_cells: int,
    n_components: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Validate mole fractions without casting or moving live tensors."""

    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise FlowContractError(
                "composition must be floating point",
                object_name="compositional.initial_state",
                field="composition",
                expected="floating tensor",
                actual=str(value.dtype),
            )
        if value.dtype != dtype:
            raise FlowContractError(
                "composition dtype must match the model dtype",
                object_name="compositional.initial_state",
                field="composition.dtype",
                expected=str(dtype),
                actual=str(value.dtype),
            )
        if value.device != device:
            raise FlowContractError(
                "composition device must match the model device",
                object_name="compositional.initial_state",
                field="composition.device",
                expected=str(device),
                actual=str(value.device),
            )
        result = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = torch.tensor(value, dtype=dtype, device=device)
    else:
        raise FlowContractError(
            "composition must be a literal sequence or floating tensor",
            object_name="compositional.initial_state",
            field="composition",
            expected="component sequence or floating tensor",
            actual=type(value).__name__,
        )
    if result.shape == (n_components,):
        result = result.expand(n_cells, n_components)
    elif result.shape != (n_cells, n_components):
        raise FlowContractError(
            "composition shape is invalid",
            object_name="compositional.initial_state",
            field="composition",
            expected=((n_components,), (n_cells, n_components)),
            actual=tuple(result.shape),
        )
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise FlowContractError(
            "composition must be finite and non-negative",
            object_name="compositional.initial_state",
            field="composition",
            expected="finite mole fractions >= 0",
            actual="contains a negative or non-finite value",
        )
    total = result.sum(dim=-1)
    tolerance = max(1.0e-8, 64.0 * torch.finfo(dtype).eps)
    if not bool(
        torch.isclose(
            total,
            torch.ones_like(total),
            rtol=tolerance,
            atol=tolerance,
        ).all()
    ):
        raise FlowContractError(
            "composition must sum to one",
            object_name="compositional.initial_state",
            field="composition",
            expected="sum(component) == 1",
            actual="one or more cells are off the mole-fraction simplex",
        )
    return result


__all__ = [
    "cell_scalar_input",
    "composition_input",
    "nonnegative_real",
    "phase_model_config",
    "positive_real",
]
