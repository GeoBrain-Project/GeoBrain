"""Immutable execution and retained-history configuration for Flow.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Literal

from .errors import FlowContractError
from .solvers.config import NewtonConfig

AutogradMode = Literal["full", "implicit", "detached"]
HistoryMode = Literal["final", "report", "all", "checkpoint", "recompute"]
LinearSolverName = Literal["dense_direct", "sparse_direct", "gmres", "bicgstab"]

_AUTOGRAD_MODES = frozenset({"full", "implicit", "detached"})
_HISTORY_MODES = frozenset({"final", "report", "all", "checkpoint", "recompute"})
_LINEAR_SOLVERS = frozenset(
    {"dense_direct", "sparse_direct", "gmres", "bicgstab"}
)


def _positive_integer(value: object, *, object_name: str, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise FlowContractError(
            f"{field_name} must be a positive integer",
            object_name=object_name,
            field=field_name,
            expected="positive integer",
            actual=value,
        )
    return int(value)


def _finite_nonnegative_float(
    value: object,
    *,
    object_name: str,
    field_name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise FlowContractError(
            f"{field_name} entries must be finite and non-negative",
            object_name=object_name,
            field=field_name,
            expected="finite seconds >= 0",
            actual=value,
        )
    return float(value)


@dataclass(frozen=True, slots=True)
class FlowHistoryConfig:
    """Declare exactly which accepted states a Flow execution may retain.

    Attributes:
        mode: retention mode, ``'final'`` (last state only), ``'report'``
            (states at ``report_times_s``), ``'all'``, ``'checkpoint'``, or
            ``'recompute'``.
        report_times_s: explicit report times [s].
        checkpoint_interval: steps between checkpoints.
        recompute_segments: recompute granularity for checkpoint mode.
    """

    mode: HistoryMode = "final"
    report_times_s: tuple[float, ...] = ()
    checkpoint_interval: int = 1
    recompute_segments: int = 1

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        if self.mode not in _HISTORY_MODES:
            raise FlowContractError(
                "history mode is unsupported",
                object_name=object_name,
                field="mode",
                expected=sorted(_HISTORY_MODES),
                actual=self.mode,
            )
        report_times = tuple(
            _finite_nonnegative_float(
                value,
                object_name=object_name,
                field_name="report_times_s",
            )
            for value in self.report_times_s
        )
        if report_times != tuple(sorted(set(report_times))):
            raise FlowContractError(
                "report times must be strictly increasing and unique",
                object_name=object_name,
                field="report_times_s",
                expected="strictly increasing unique seconds",
                actual=report_times,
            )
        if self.mode == "report" and not report_times:
            raise FlowContractError(
                "report history requires at least one report time",
                object_name=object_name,
                field="report_times_s",
                expected="non-empty for mode='report'",
                actual=report_times,
            )
        object.__setattr__(self, "report_times_s", report_times)
        object.__setattr__(
            self,
            "checkpoint_interval",
            _positive_integer(
                self.checkpoint_interval,
                object_name=object_name,
                field_name="checkpoint_interval",
            ),
        )
        object.__setattr__(
            self,
            "recompute_segments",
            _positive_integer(
                self.recompute_segments,
                object_name=object_name,
                field_name="recompute_segments",
            ),
        )


@dataclass(frozen=True, slots=True)
class FlowExecutionConfig:
    """One explicit selection of differentiation, history, and solve policy.

    Attributes:
        autograd_mode: gradient mode, ``'implicit'`` (implicit-function-
            theorem adjoint, the default), ``'full'`` (unrolled autograd), or
            ``'detached'`` (forward only). Both differentiable modes carry
            gradients for the STATE-key trainables (pressure/saturations);
            constitutive-module parameters (e.g. ``rock.permeability_m2``)
            get their gradients through :class:`ParametricFlowOperator` /
            :func:`~geobrain.physics.flow.adjoint.newton_solve_with_adjoint`,
            not through the bare transient march.
        history: :class:`FlowHistoryConfig` retention policy.
        nonlinear: Newton configuration.
        linear_solver: inner linear-solver id (platform convention:
            id strings here and in ``wave``'s backend name; ``em``'s
            ``EMExecutionConfig`` adds an object-injection escape hatch).
        resource_budget_bytes: optional preflight memory budget.
    """

    autograd_mode: AutogradMode = "implicit"
    history: FlowHistoryConfig = field(default_factory=FlowHistoryConfig)
    nonlinear: NewtonConfig = field(default_factory=NewtonConfig)
    linear_solver: LinearSolverName = "dense_direct"
    resource_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        if self.autograd_mode not in _AUTOGRAD_MODES:
            raise FlowContractError(
                "autograd_mode is unsupported",
                object_name=object_name,
                field="autograd_mode",
                expected=sorted(_AUTOGRAD_MODES),
                actual=self.autograd_mode,
            )
        if not isinstance(self.history, FlowHistoryConfig):
            raise FlowContractError(
                "history must be a FlowHistoryConfig",
                object_name=object_name,
                field="history",
                expected=FlowHistoryConfig,
                actual=type(self.history),
            )
        if not isinstance(self.nonlinear, NewtonConfig):
            raise FlowContractError(
                "nonlinear must be a NewtonConfig",
                object_name=object_name,
                field="nonlinear",
                expected=NewtonConfig,
                actual=type(self.nonlinear),
            )
        if self.linear_solver not in _LINEAR_SOLVERS:
            raise FlowContractError(
                "linear_solver is unsupported",
                object_name=object_name,
                field="linear_solver",
                expected=sorted(_LINEAR_SOLVERS),
                actual=self.linear_solver,
            )
        budget = self.resource_budget_bytes
        if budget is not None:
            object.__setattr__(
                self,
                "resource_budget_bytes",
                _positive_integer(
                    budget,
                    object_name=object_name,
                    field_name="resource_budget_bytes",
                ),
            )


__all__ = [
    "AutogradMode",
    "FlowExecutionConfig",
    "FlowHistoryConfig",
    "HistoryMode",
    "LinearSolverName",
]
