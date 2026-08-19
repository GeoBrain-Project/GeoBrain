"""
Shared cancellation, stop-reason, and iteration-observation records.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import threading
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, TypeAlias

import torch

from geobrain.core import GeoBrainError
from geobrain.core.validation import validate_param_mapping

from ._validation import _coerce_integral_scalar, _coerce_real_scalar

__all__ = ["CancellationToken", "IterationRecord", "StopReason"]


class StopReason(str, Enum):
    """Why an optimizer returned an ordinary final result.

    ``COMPLETED`` exhausts the request, ``CALLBACK`` follows a post-iteration
    callback stop, ``CANCELLED`` is cooperative cancellation (including before
    the first step), and ``NONFINITE`` records one terminal non-finite loss.
    Exceptions remain exceptions and use a private progress snapshot without a
    fabricated stop reason.
    """

    COMPLETED = "completed"
    CALLBACK = "callback"
    CANCELLED = "cancelled"
    NONFINITE = "nonfinite"


@dataclass
class CancellationToken:
    """Thread-safe cooperative cancellation signal for optimizer runs."""

    _event: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Request cancellation; repeated requests are harmless."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._event.is_set()


@dataclass(frozen=True)
class IterationRecord:
    """Frozen scalar observation with an owned detached parameter snapshot.

    ``iteration`` is zero-based and non-negative; ``loss`` is finite and is not
    a boolean. ``params`` is copied into a read-only mapping whose tensors are
    detached clones. The holder can still modify those tensors' storage; such a
    mutation changes this observation only and cannot reach optimizer state.

    Attributes:
        iteration: 0-based iteration index.
        loss: objective value at this iteration.
        params: owned snapshot of the parameters at this iteration.
    """

    iteration: int
    loss: float
    params: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        """Validate scalars and detach the record from live optimizer state."""
        iteration = _coerce_integral_scalar(
            self.iteration,
            owner="IterationRecord",
            field="iteration",
            minimum=0,
        )
        loss = _coerce_real_scalar(
            self.loss,
            owner="IterationRecord",
            field="loss",
            finite=True,
        )
        if not isinstance(self.params, MappingABC):
            raise GeoBrainError(
                "IterationRecord.params must be a parameter mapping",
                object_name="IterationRecord",
                field="params",
                expected="non-empty mapping of string keys to torch.Tensor",
                actual=type(self.params),
            )
        validate_param_mapping(self.params, "IterationRecord")
        params = MappingProxyType(
            {name: tensor.detach().clone() for name, tensor in self.params.items()}
        )
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "params", params)


OptimizationCallback: TypeAlias = Callable[
    [int, float, Mapping[str, torch.Tensor]], bool | None
]
