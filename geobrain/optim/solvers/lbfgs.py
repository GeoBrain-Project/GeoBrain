"""Bare deterministic L-BFGS execution.

``LBFGSConfig.max_iter`` controls torch's internal step/line-search work. One
requested/completed iteration always means one outer ``optimizer.step``;
closure evaluations never inflate result accounting or callback indices.

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
from ..config import LBFGSConfig
from ..execution import CancellationToken, OptimizationCallback, StopReason
from .base import Optimizer, OptimizationResult, _compile_optimizer


class LBFGS(Optimizer):
    """Minimal problem-aware L-BFGS loop with post-step observations.

    Args:
        problem: objective provider (``InverseProblem`` or compatible).
        params: ``{name: tensor}`` initial values; cloned and owned.
        config: :class:`LBFGSConfig` hyper-parameters.
        metadata: free-form extras copied onto the result.
    """

    optimizer_config_type = LBFGSConfig

    def __init__(
        self,
        problem: InverseProblemLike,
        *,
        params: Mapping[str, torch.Tensor],
        config: LBFGSConfig = LBFGSConfig(),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Compile every canonical L-BFGS option from ``config``."""
        if not isinstance(config, LBFGSConfig):
            raise GeoBrainError(
                "LBFGS config must be an LBFGSConfig",
                object_name="LBFGS",
                field="config",
                expected=LBFGSConfig,
                actual=type(config),
            )
        validate_trainable_params(params, "LBFGS")
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
        """Run outer L-BFGS steps with exact outer-iteration accounting."""
        requested = validate_non_negative_int(
            n_iters,
            owner="LBFGS.run",
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
        phase = "objective"

        def closure() -> torch.Tensor:
            nonlocal phase
            phase = "objective"
            evaluation = evaluator.evaluate(self.params)
            phase = "backward"
            self._torch_optim.zero_grad(set_to_none=True)
            evaluation.total_loss.backward()
            phase = "step"
            return evaluation.total_loss

        for _ in range(requested):
            if cancellation is not None and cancellation.is_cancelled:
                reason = StopReason.CANCELLED
                break
            try:
                phase = "step"
                self._torch_optim.step(closure)
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
                    owner="LBFGS.run",
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
