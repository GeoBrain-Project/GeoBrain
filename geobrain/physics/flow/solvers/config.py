"""Immutable validated configuration for Flow nonlinear and Krylov solves.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal

from ..errors import FlowContractError


def _positive_integer(value: object, *, object_name: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise FlowContractError(
            f"{field} must be a positive integer",
            object_name=object_name,
            field=field,
            expected="positive integer",
            actual=value,
        )
    return int(value)


def _positive_float(value: object, *, object_name: str, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise FlowContractError(
            f"{field} must be positive and finite",
            object_name=object_name,
            field=field,
            expected="finite value > 0",
            actual=value,
        )
    return float(value)


@dataclass(frozen=True, slots=True)
class NewtonConfig:
    """Validated nonlinear-solve controls."""

    max_iterations: int = 12
    residual_tolerance: float = 1.0e-8
    update_tolerance: float = 1.0e-10
    line_search_max_iterations: int = 8

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        object.__setattr__(
            self,
            "max_iterations",
            _positive_integer(
                self.max_iterations,
                object_name=object_name,
                field="max_iterations",
            ),
        )
        object.__setattr__(
            self,
            "line_search_max_iterations",
            _positive_integer(
                self.line_search_max_iterations,
                object_name=object_name,
                field="line_search_max_iterations",
            ),
        )
        for field in ("residual_tolerance", "update_tolerance"):
            object.__setattr__(
                self,
                field,
                _positive_float(
                    getattr(self, field),
                    object_name=object_name,
                    field=field,
                ),
            )


@dataclass(frozen=True, slots=True)
class KrylovConfig:
    """Validated GMRES/BiCGSTAB controls."""

    method: Literal["gmres", "bicgstab"]
    max_iterations: int
    tolerance: float
    restart: int | None = None

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        if self.method not in {"gmres", "bicgstab"}:
            raise FlowContractError(
                "method must select a supported Krylov algorithm",
                object_name=object_name,
                field="method",
                expected=("gmres", "bicgstab"),
                actual=self.method,
            )
        object.__setattr__(
            self,
            "max_iterations",
            _positive_integer(
                self.max_iterations,
                object_name=object_name,
                field="max_iterations",
            ),
        )
        object.__setattr__(
            self,
            "tolerance",
            _positive_float(
                self.tolerance,
                object_name=object_name,
                field="tolerance",
            ),
        )
        if self.restart is not None:
            object.__setattr__(
                self,
                "restart",
                _positive_integer(
                    self.restart,
                    object_name=object_name,
                    field="restart",
                ),
            )
        if self.method == "bicgstab" and self.restart is not None:
            raise FlowContractError(
                "restart is only defined for GMRES",
                object_name=object_name,
                field="restart",
                expected=None,
                actual=self.restart,
            )


__all__ = ["KrylovConfig", "NewtonConfig"]
