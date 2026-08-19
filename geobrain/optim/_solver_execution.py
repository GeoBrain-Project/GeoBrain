"""Shared run progress for deterministic optimizer loops.

This module owns only completed-iteration accounting, detached callback
records, cooperative stop precedence, partial-result error attachment, and
bare-result assembly. Optimizer-specific stepping remains in each solver.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, NoReturn

import torch

from geobrain.core import GeoBrainError, ModelState

from .execution import IterationRecord, OptimizationCallback, StopReason
from .processing import _ObservationCadence
from .results import (
    OptimizationResult,
    _make_partial_result,
)


@dataclass
class SolverProgress:
    """Mutable private accounting whose history contains completed iterations."""

    params: Mapping[str, torch.Tensor]
    requested_iters: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _losses: list[float] = field(default_factory=list)
    _completed_params: Mapping[str, torch.Tensor] = field(init=False)

    def __post_init__(self) -> None:
        self._completed_params = {
            name: tensor.detach().clone()
            for name, tensor in self.params.items()
        }

    @property
    def completed_iters(self) -> int:
        """Return the number of successfully committed outer iterations."""
        return len(self._losses)

    def observe(
        self,
        loss: float,
        callback: OptimizationCallback | None,
    ) -> bool:
        """Call a detached callback, then commit its successful iteration.

        The callback runs before the history commit so a callback exception
        leaves the current iteration out of an attached partial result.
        """
        iteration = self.completed_iters
        should_call = callback is not None and (
            not isinstance(callback, _ObservationCadence)
            or callback.should_observe(iteration)
        )
        if callback is not None and should_call:
            record = IterationRecord(
                iteration=iteration,
                loss=loss,
                params=self.params,
            )
            should_stop = bool(
                callback(record.iteration, record.loss, record.params)
            )
            committed_loss = record.loss
        else:
            should_stop = False
            committed_loss = float(loss)
        self._losses.append(committed_loss)
        self._completed_params = {
            name: tensor.detach().clone()
            for name, tensor in self.params.items()
        }
        return should_stop

    def observe_nonfinite(self, loss: float) -> None:
        """Commit one terminal non-finite diagnostic without a callback."""
        self._losses.append(float(loss))
        self._completed_params = {
            name: tensor.detach().clone()
            for name, tensor in self.params.items()
        }

    def history(self) -> torch.Tensor:
        """Materialize the committed loss history as the result dtype."""
        return torch.tensor(self._losses, dtype=torch.float64)

    def finish(
        self,
        *,
        state_metadata: Mapping[str, Any],
        stop_reason: StopReason,
    ) -> OptimizationResult:
        """Build a bare result with explicit requested/completed semantics."""
        history = self.history()
        return OptimizationResult(
            params=ModelState(
                tensors=self.params,
                metadata=state_metadata,
            ),
            final_loss=(
                float(history[-1])
                if self.completed_iters > 0
                else float("nan")
            ),
            loss_history=history,
            metadata=self.metadata,
            requested_iters=self.requested_iters,
            completed_iters=self.completed_iters,
            stop_reason=stop_reason,
        )


def _raise_execution_error(
    error: Exception,
    *,
    progress: SolverProgress,
    owner: str,
    phase: str,
) -> NoReturn:
    """Attach completed progress and preserve or structure the exception."""
    partial = _make_partial_result(
        params=progress._completed_params,
        requested_iters=progress.requested_iters,
        completed_iters=progress.completed_iters,
        loss_history=progress.history(),
        metadata={
            **progress.metadata,
            "owner": owner,
            "phase": phase,
        },
    )
    if isinstance(error, GeoBrainError):
        setattr(error, "partial_result", partial)
        raise error
    wrapped = GeoBrainError(
        f"{owner} failed during {phase}",
        object_name=owner,
        field=phase,
        expected="successful optimizer execution",
        actual=type(error),
    )
    setattr(wrapped, "partial_result", partial)
    raise wrapped from error


def _stop_after_iteration(
    *,
    callback_stopped: bool,
    cancelled: bool,
) -> StopReason | None:
    """Resolve simultaneous stops; callback truth takes precedence."""
    if callback_stopped:
        return StopReason.CALLBACK
    if cancelled:
        return StopReason.CANCELLED
    return None


def _stop_for_loss(loss: float) -> StopReason | None:
    """Return the terminal non-finite status, if any."""
    return None if math.isfinite(loss) else StopReason.NONFINITE
