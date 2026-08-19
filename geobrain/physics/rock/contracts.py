"""Immutable tensor and iteration contracts for the rock family.

Capability reports live in the sibling ``capabilities.py`` (the family
template module).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import torch

from geobrain.core import DifferentiabilityLevel, ErrorCode

from .errors import RockContractError

_REAL_DTYPES = frozenset({torch.float32, torch.float64})


def _nonempty_string(value: object, field: str, *, object_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RockContractError(
            "invalid Rock contract string",
            object_name=object_name,
            field=field,
            expected="non-empty string",
            actual=value,
        )
    return value


def _string_tuple(
    values: object,
    field: str,
    *,
    object_name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RockContractError(
            "invalid Rock contract sequence",
            object_name=object_name,
            field=field,
            expected="sequence of non-empty strings",
            actual=values,
        )
    owned = tuple(
        _nonempty_string(item, field, object_name=object_name) for item in values
    )
    if not allow_empty and not owned:
        raise RockContractError(
            "Rock contract sequence cannot be empty",
            object_name=object_name,
            field=field,
            expected="at least one entry",
            actual=owned,
        )
    if len(owned) != len(set(owned)):
        raise RockContractError(
            "Rock contract sequence contains duplicates",
            object_name=object_name,
            field=field,
            expected="unique entries",
            actual=owned,
        )
    return owned


def _pair_tuple(
    values: object,
    field: str,
    *,
    object_name: str,
    allow_empty: bool = True,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RockContractError(
            "invalid Rock key/value sequence",
            object_name=object_name,
            field=field,
            expected="sequence of two-string pairs",
            actual=values,
        )
    owned: list[tuple[str, str]] = []
    for pair in values:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
            raise RockContractError(
                "invalid Rock key/value pair",
                object_name=object_name,
                field=field,
                expected="two non-empty strings",
                actual=pair,
            )
        entries = tuple(pair)
        if len(entries) != 2:
            raise RockContractError(
                "invalid Rock key/value pair",
                object_name=object_name,
                field=field,
                expected="two non-empty strings",
                actual=entries,
            )
        key = _nonempty_string(entries[0], field, object_name=object_name)
        value = _nonempty_string(entries[1], field, object_name=object_name)
        owned.append((key, value))
    result = tuple(owned)
    if not allow_empty and not result:
        raise RockContractError(
            "Rock key/value sequence cannot be empty",
            object_name=object_name,
            field=field,
            expected="at least one pair",
            actual=result,
        )
    keys = tuple(key for key, _ in result)
    if len(keys) != len(set(keys)):
        raise RockContractError(
            "Rock key/value sequence contains duplicate keys",
            object_name=object_name,
            field=field,
            expected="unique keys",
            actual=keys,
        )
    return result


@dataclass(frozen=True, slots=True)
class RockFieldSpec:
    """One named SI tensor field in a Rock capability declaration."""

    name: str
    unit: str
    axes: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _nonempty_string(self.name, "name", object_name="RockFieldSpec"),
        )
        object.__setattr__(
            self,
            "unit",
            _nonempty_string(self.unit, "unit", object_name="RockFieldSpec"),
        )
        object.__setattr__(
            self,
            "axes",
            _string_tuple(self.axes, "axes", object_name="RockFieldSpec"),
        )
        if not isinstance(self.description, str):
            raise RockContractError(
                "invalid Rock field description",
                object_name="RockFieldSpec",
                field="description",
                expected="string",
                actual=self.description,
            )

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-ready field metadata."""

        return {
            "name": self.name,
            "unit": self.unit,
            "axes": list(self.axes),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class RockIterationReport:
    """Owned, detached diagnostic snapshots from a Rock iteration.

    This report is a diagnostic boundary, not a physical-result autograd path.
    Differentiable physical tensors remain with the kernel outputs that produced
    them; convergence evidence stored here is detached for safe serialization.
    """

    converged: torch.Tensor
    residual: torch.Tensor
    iterations: int
    tolerance: float

    def __post_init__(self) -> None:
        converged = object.__getattribute__(self, "converged")
        residual = object.__getattribute__(self, "residual")
        iterations = self.iterations
        tolerance = self.tolerance
        if not isinstance(converged, torch.Tensor) or converged.dtype is not torch.bool:
            raise RockContractError(
                "invalid Rock convergence mask",
                object_name="RockIterationReport",
                field="converged",
                expected="Boolean torch.Tensor",
                actual=type(converged)
                if not isinstance(converged, torch.Tensor)
                else str(converged.dtype),
            )
        if not isinstance(residual, torch.Tensor) or residual.dtype not in _REAL_DTYPES:
            raise RockContractError(
                "invalid Rock iteration residual",
                object_name="RockIterationReport",
                field="residual",
                expected="float32 or float64 torch.Tensor",
                actual=type(residual)
                if not isinstance(residual, torch.Tensor)
                else str(residual.dtype),
            )
        if converged.shape != residual.shape:
            raise RockContractError(
                "Rock iteration report shape mismatch",
                object_name="RockIterationReport",
                field="shape",
                expected=list(converged.shape),
                actual=list(residual.shape),
                code=ErrorCode.SHAPE_MISMATCH,
            )
        if converged.device != residual.device:
            raise RockContractError(
                "Rock iteration report device mismatch",
                object_name="RockIterationReport",
                field="device",
                expected=str(converged.device),
                actual=str(residual.device),
            )
        if residual.device.type == "meta":
            raise RockContractError(
                "Rock iteration report requires materialized tensor values",
                object_name="RockIterationReport",
                field="device",
                expected="device with materialized tensor values",
                actual=str(residual.device),
                code=ErrorCode.DEVICE_UNAVAILABLE,
                hint=(
                    "move iteration diagnostics to a materialized device before "
                    "constructing the report"
                ),
            )
        if not bool(torch.isfinite(residual).all()) or bool(
            torch.any(residual < 0)
        ):
            raise RockContractError(
                "Rock iteration residual must be finite and non-negative",
                object_name="RockIterationReport",
                field="residual",
                expected="finite values >= 0",
                actual=residual.tolist(),
            )
        if type(iterations) is not int or iterations <= 0:
            raise RockContractError(
                "invalid Rock iteration count",
                object_name="RockIterationReport",
                field="iterations",
                expected="positive integer",
                actual=iterations,
            )
        if (
            type(tolerance) not in {int, float}
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise RockContractError(
                "invalid Rock iteration tolerance",
                object_name="RockIterationReport",
                field="tolerance",
                expected="positive finite real number",
                actual=tolerance,
            )
        normalized_tolerance = float(tolerance)
        derived_converged = residual <= normalized_tolerance
        if not torch.equal(converged, derived_converged):
            raise RockContractError(
                "Rock convergence mask disagrees with residual and tolerance",
                object_name="RockIterationReport",
                field="converged",
                expected="residual <= tolerance elementwise",
                actual={
                    "converged": converged.tolist(),
                    "derived": derived_converged.tolist(),
                    "tolerance": normalized_tolerance,
                },
                hint="derive converged elementwise as residual <= tolerance",
            )
        object.__setattr__(self, "converged", converged.detach().clone())
        object.__setattr__(self, "residual", residual.detach().clone())
        object.__setattr__(self, "tolerance", normalized_tolerance)

    def __getattribute__(self, name: str) -> object:
        """Clone public tensor fields so callers cannot mutate owned snapshots."""

        value = object.__getattribute__(self, name)
        if name in {"converged", "residual"} and isinstance(value, torch.Tensor):
            return value.clone()
        return value

    def to_dict(self) -> dict[str, object]:
        """Serialize detached diagnostics, never physical-result autograd tensors."""

        converged = object.__getattribute__(self, "converged")
        residual = object.__getattribute__(self, "residual")
        return {
            "converged": converged.tolist(),
            "residual": residual.tolist(),
            "iterations": self.iterations,
            "tolerance": self.tolerance,
        }


def require_compatible_tensors(
    object_name: str,
    *fields: tuple[str, torch.Tensor | int | float],
) -> tuple[torch.Tensor, ...]:
    """Validate Rock dtype/device/broadcast compatibility without casting tensors."""

    validated_object_name = _nonempty_string(
        object_name,
        "object_name",
        object_name="require_compatible_tensors",
    )
    if not fields:
        raise RockContractError(
            "Rock tensor inputs cannot be empty",
            object_name=validated_object_name,
            field="inputs",
            expected="at least one named input containing a tensor",
            actual=(),
        )
    validated_fields: list[tuple[str, torch.Tensor | int | float]] = []
    for entry in fields:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise RockContractError(
                "invalid named Rock tensor input",
                object_name=validated_object_name,
                field="inputs",
                expected="two-item tuples of (non-empty name, value)",
                actual=entry,
            )
        raw_name, value = entry
        name = _nonempty_string(
            raw_name,
            "field",
            object_name=validated_object_name,
        )
        validated_fields.append((name, value))
    names = tuple(name for name, _ in validated_fields)
    if len(names) != len(set(names)):
        raise RockContractError(
            "Rock tensor input names must be unique",
            object_name=validated_object_name,
            field="inputs",
            expected="unique field names",
            actual=names,
        )

    reference = next(
        (
            value
            for _, value in validated_fields
            if isinstance(value, torch.Tensor)
        ),
        None,
    )
    if reference is None:
        raise RockContractError(
            "Rock call requires a tensor reference",
            object_name=validated_object_name,
            field="inputs",
            expected="at least one torch.Tensor",
            actual=[type(value).__qualname__ for _, value in validated_fields],
        )
    if reference.dtype not in _REAL_DTYPES:
        raise RockContractError(
            "unsupported Rock tensor dtype",
            object_name=validated_object_name,
            field=next(
                name for name, value in validated_fields if value is reference
            ),
            expected="torch.float32 or torch.float64",
            actual=str(reference.dtype),
            code=ErrorCode.DTYPE_UNSUPPORTED,
        )

    tensors: list[torch.Tensor] = []
    shapes: dict[str, list[int]] = {}
    for name, value in validated_fields:
        if isinstance(value, torch.Tensor):
            if value.dtype != reference.dtype:
                raise RockContractError(
                    "Rock tensor dtype mismatch",
                    object_name=validated_object_name,
                    field=name,
                    expected=str(reference.dtype),
                    actual=str(value.dtype),
                    code=ErrorCode.DTYPE_UNSUPPORTED,
                )
            if value.device != reference.device:
                raise RockContractError(
                    "Rock tensor device mismatch",
                    object_name=validated_object_name,
                    field=name,
                    expected=str(reference.device),
                    actual=str(value.device),
                )
            tensor = value
        elif isinstance(value, Real) and not isinstance(value, bool):
            try:
                tensor = reference.new_tensor(value)
            except (OverflowError, TypeError, RuntimeError) as exc:
                raise RockContractError(
                    "Rock scalar cannot be represented by the reference tensor",
                    object_name=validated_object_name,
                    field=name,
                    expected=f"scalar representable as {reference.dtype}",
                    actual={"type": type(value).__qualname__, "value": repr(value)},
                    hint="provide a finite scalar within the reference dtype range",
                ) from exc
        else:
            raise RockContractError(
                "invalid Rock tensor input",
                object_name=validated_object_name,
                field=name,
                expected="torch.Tensor or Python real scalar",
                actual=type(value).__qualname__,
            )
        tensors.append(tensor)
        shapes[name] = list(tensor.shape)

    try:
        torch.broadcast_shapes(*(tensor.shape for tensor in tensors))
    except RuntimeError as exc:
        raise RockContractError(
            "Rock tensor inputs are not broadcastable",
            object_name=validated_object_name,
            field="broadcast_shape",
            expected="PyTorch-broadcastable shapes",
            actual=shapes,
            code=ErrorCode.SHAPE_MISMATCH,
            hint="reshape singleton axes explicitly before calling the Rock kernel",
        ) from exc
    return tuple(tensors)


__all__ = [
    "RockFieldSpec",
    "RockIterationReport",
    "require_compatible_tensors",
]
