"""Exact allocation-free byte accounting for rock-family execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import sys

import torch

from geobrain.core import ErrorCode

from .errors import RockContractError, RockResourceError

_MAX_BYTES = sys.maxsize


def _shape_tuple(shape: object, *, object_name: str) -> tuple[int, ...]:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise RockContractError(
            "invalid Rock broadcast shape",
            object_name=object_name,
            field="broadcast_shape",
            expected="sequence of non-negative integers",
            actual=shape,
        )
    owned = tuple(shape)
    for dimension in owned:
        if type(dimension) is not int or dimension < 0 or dimension > _MAX_BYTES:
            raise RockContractError(
                "invalid Rock broadcast dimension",
                object_name=object_name,
                field="broadcast_shape",
                expected=f"integers in [0, {_MAX_BYTES}]",
                actual=owned,
            )
    return owned


def _nonnegative_integer(value: object, field: str, *, object_name: str) -> int:
    if type(value) is not int or value < 0 or value > _MAX_BYTES:
        raise RockContractError(
            "invalid Rock resource count",
            object_name=object_name,
            field=field,
            expected=f"integer in [0, {_MAX_BYTES}]",
            actual=value,
        )
    return value


def _checked_product(
    left: int,
    right: int,
    field: str,
    *,
    object_name: str,
) -> int:
    if left != 0 and right > _MAX_BYTES // left:
        raise RockContractError(
            "Rock resource estimate exceeds addressable integer range",
            object_name=object_name,
            field=field,
            expected=f"<= {_MAX_BYTES} bytes",
            actual={"left": left, "right": right},
            hint=(
                "reduce broadcast dimensions so every contiguous suffix stride "
                "is addressable"
                if field == "broadcast_shape"
                else "reduce tensor counts or broadcast dimensions"
            ),
        )
    return left * right


@dataclass(frozen=True, slots=True)
class RockResourceEstimate:
    """Exact conservative tensor-byte partitions for one Rock evaluation."""

    broadcast_shape: tuple[int, ...]
    input_bytes: int
    output_bytes: int
    workspace_bytes: int
    saved_for_backward_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "broadcast_shape",
            _shape_tuple(
                self.broadcast_shape,
                object_name="RockResourceEstimate",
            ),
        )
        parts = (
            ("input_bytes", self.input_bytes),
            ("output_bytes", self.output_bytes),
            ("workspace_bytes", self.workspace_bytes),
            ("saved_for_backward_bytes", self.saved_for_backward_bytes),
        )
        for field, value in parts:
            _nonnegative_integer(
                value,
                field,
                object_name="RockResourceEstimate",
            )
        expected_total = sum(value for _, value in parts)
        if expected_total > _MAX_BYTES:
            raise RockContractError(
                "Rock resource partitions exceed addressable integer range",
                object_name="RockResourceEstimate",
                field="total_bytes",
                expected=f"<= {_MAX_BYTES}",
                actual=expected_total,
            )
        if type(self.total_bytes) is not int or self.total_bytes != expected_total:
            raise RockContractError(
                "Rock resource total does not equal its byte partitions",
                object_name="RockResourceEstimate",
                field="total_bytes",
                expected=expected_total,
                actual=self.total_bytes,
            )

    def require_budget(self, budget_bytes: int, *, object_name: str) -> None:
        """Reject an invalid or insufficient budget before scientific allocation."""

        if not isinstance(object_name, str) or not object_name.strip():
            raise RockContractError(
                "invalid Rock resource budget object name",
                object_name="RockResourceEstimate.require_budget",
                field="object_name",
                expected="non-empty string",
                actual=object_name,
                hint="pass the concrete Rock facade or model name",
            )
        if type(budget_bytes) is not int or budget_bytes <= 0 or budget_bytes > _MAX_BYTES:
            raise RockContractError(
                "invalid Rock resource budget",
                object_name=object_name,
                field="budget_bytes",
                expected="positive integer",
                actual=budget_bytes,
            )
        if self.total_bytes > budget_bytes:
            raise RockResourceError(
                "Rock resource estimate exceeds budget",
                object_name=object_name,
                field="budget_bytes",
                expected=f">= {self.total_bytes}",
                actual=budget_bytes,
                hint="reduce broadcast size or disable gradient recording",
            )

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-ready resource metadata."""

        return {
            "broadcast_shape": list(self.broadcast_shape),
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "workspace_bytes": self.workspace_bytes,
            "saved_for_backward_bytes": self.saved_for_backward_bytes,
            "total_bytes": self.total_bytes,
        }


def estimate_elementwise_resources(
    *,
    broadcast_shape: Sequence[int],
    dtype: torch.dtype,
    input_tensors: int,
    output_tensors: int,
    workspace_tensors: int,
    saved_tensors: int,
) -> RockResourceEstimate:
    """Estimate exact elementwise tensor bytes using integer arithmetic only."""

    object_name = "estimate_elementwise_resources"
    shape = _shape_tuple(broadcast_shape, object_name=object_name)
    if dtype is not torch.float32 and dtype is not torch.float64:
        raise RockContractError(
            "unsupported Rock resource dtype",
            object_name=object_name,
            field="dtype",
            expected="float32 or float64",
            actual=dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
        )
    counts = (
        _nonnegative_integer(
            input_tensors,
            "input_tensors",
            object_name=object_name,
        ),
        _nonnegative_integer(
            output_tensors,
            "output_tensors",
            object_name=object_name,
        ),
        _nonnegative_integer(
            workspace_tensors,
            "workspace_tensors",
            object_name=object_name,
        ),
        _nonnegative_integer(
            saved_tensors,
            "saved_tensors",
            object_name=object_name,
        ),
    )

    contiguous_stride = 1
    for dimension in reversed(shape):
        contiguous_stride = _checked_product(
            contiguous_stride,
            max(dimension, 1),
            "broadcast_shape",
            object_name=object_name,
        )

    elements = 1
    for dimension in shape:
        elements = _checked_product(
            elements,
            dimension,
            "broadcast_shape",
            object_name=object_name,
        )
    itemsize = 4 if dtype is torch.float32 else 8
    bytes_per_tensor = _checked_product(
        elements,
        itemsize,
        "total_bytes",
        object_name=object_name,
    )
    parts = tuple(
        _checked_product(
            bytes_per_tensor,
            count,
            "total_bytes",
            object_name=object_name,
        )
        for count in counts
    )
    total = sum(parts)
    if total > _MAX_BYTES:
        raise RockContractError(
            "Rock resource estimate exceeds addressable integer range",
            object_name=object_name,
            field="total_bytes",
            expected=f"<= {_MAX_BYTES} bytes",
            actual=total,
        )
    return RockResourceEstimate(
        broadcast_shape=shape,
        input_bytes=parts[0],
        output_bytes=parts[1],
        workspace_bytes=parts[2],
        saved_for_backward_bytes=parts[3],
        total_bytes=total,
    )


__all__ = ["RockResourceEstimate", "estimate_elementwise_resources"]
