"""Bare deterministic Adam execution.

The Tier-3 loop exposes exact post-step history. It therefore evaluates the
objective twice per completed outer iteration: once for backward and once for
the post-step diagnostic/callback record.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ...core import ForwardContext, GeoBrainError
from ...core.validation import (
    validate_non_negative_int,
    validate_trainable_params,
)
from ...inverse import InverseProblemLike
from .._loss_evaluator import LossEvaluator
from .._solver_execution import (
    SolverProgress,
    _raise_execution_error,
    _stop_after_iteration,
    _stop_for_loss,
)
from ..config import AdamConfig
from ..execution import CancellationToken, OptimizationCallback, StopReason
from .base import Optimizer, OptimizationResult, _compile_optimizer


class Adam(Optimizer):
    """Minimal problem-aware Adam loop with exact post-step observations.

    Use :class:`geobrain.optim.Inverter` when regularizers, gradient
    processors, projections, scheduling, or best-parameter tracking are
    required.

    Args:
        problem: objective provider (``InverseProblem`` or compatible).
        params: ``{name: tensor}`` initial values; cloned and owned.
        config: :class:`AdamConfig` hyper-parameters.
        metadata: free-form extras copied onto the result.
    """

    optimizer_config_type = AdamConfig

    def __init__(
        self,
        problem: InverseProblemLike,
        *,
        params: Mapping[str, torch.Tensor],
        config: AdamConfig = AdamConfig(),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Compile ``config`` over caller-owned trainable leaf tensors."""
        if not isinstance(config, AdamConfig):
            raise GeoBrainError(
                "Adam config must be an AdamConfig",
                object_name="Adam",
                field="config",
                expected=AdamConfig,
                actual=type(config),
            )
        validate_trainable_params(params, "Adam")
        self.problem = problem
        self.params = dict(params)
        self.config = config
        self._metadata: Mapping[str, Any] = (
            {} if metadata is None else dict(metadata)
        )
        self._torch_optim = _compile_optimizer(
            config,
            self.params.values(),
        )

    def run(
        self,
        n_iters: int,
        ctx: ForwardContext | None = None,
        callback: OptimizationCallback | None = None,
        cancellation: CancellationToken | None = None,
    ) -> OptimizationResult:
        """Run outer Adam steps with cooperative between-step cancellation."""
        requested = validate_non_negative_int(
            n_iters,
            owner="Adam.run",
            field="n_iters",
        )
        context = ForwardContext() if ctx is None else ctx
        evaluator = LossEvaluator(
            self.problem,
            mode="MLE",
            regularizer=None,
            ctx=context,
            state_metadata=self._metadata,
        )
        progress = SolverProgress(
            params=self.params,
            requested_iters=requested,
        )
        reason = StopReason.COMPLETED

        for _ in range(requested):
            if cancellation is not None and cancellation.is_cancelled:
                reason = StopReason.CANCELLED
                break
            phase = "objective"
            try:
                evaluation = evaluator.evaluate(self.params)
                phase = "backward"
                self._torch_optim.zero_grad(set_to_none=True)
                evaluation.total_loss.backward()
                phase = "step"
                self._torch_optim.step()
                phase = "diagnostics"
                with torch.no_grad():
                    post_loss = float(
                        evaluator.evaluate(self.params).total_loss.detach()
                    )
                loss_stop = _stop_for_loss(post_loss)
                if loss_stop is not None:
                    progress.observe_nonfinite(post_loss)
                    reason = loss_stop
                    break
                phase = "callback"
                callback_stopped = progress.observe(post_loss, callback)
            except Exception as error:
                _raise_execution_error(
                    error,
                    progress=progress,
                    owner="Adam.run",
                    phase=phase,
                )
            iteration_stop = _stop_after_iteration(
                callback_stopped=callback_stopped,
                cancelled=(
                    cancellation is not None
                    and cancellation.is_cancelled
                ),
            )
            if iteration_stop is not None:
                reason = iteration_stop
                break

        return progress.finish(
            state_metadata=self._metadata,
            stop_reason=reason,
        )
