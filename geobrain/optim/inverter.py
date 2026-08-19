"""High-level deterministic inversion with one execution contract.

``Inverter`` adds regularizers, gradient processors, step projections,
scheduling, component histories, and best-parameter tracking to the bare
optimizers. The public optimizer choice is an immutable
``AdamConfig | LBFGSConfig``; string names and loose option dictionaries are
handled only by maintained client migration seams.

Adam uses a rolling post-step graph: ``N`` completed steps cost ``N + 1``
objective evaluations. L-BFGS closure calls remain internal to each outer
step and never count as completed iterations or diagnostics.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Callable, Literal, Mapping, Sequence, cast

import torch

from geobrain.core import ForwardContext, GeoBrainError
from geobrain.core.validation import (
    clamp_params_in_place,
    validate_non_negative_int,
    validate_param_mapping,
)
from geobrain.inverse import InverseProblemLike, Likelihood

from ._loss_evaluator import LossEvaluation, LossEvaluator, Regularizer
from ._normalize import (
    _coerce_params,
    _normalize_bounds,
    _normalize_lr_mapping,
)
from ._solver_execution import (
    SolverProgress,
    _raise_execution_error,
    _stop_after_iteration,
    _stop_for_loss,
)
from ._result_assembly import (
    _ResultTransform,
    _assemble_inversion_result,
)
from .config import AdamConfig, LBFGSConfig, OptimizerConfig
from .execution import CancellationToken, OptimizationCallback, StopReason
from .processing import (
    BoundsClamp,
    GradientProcessor,
    NaNGuard,
    StepProjection,
)
from .parameterization import Parameterization
from .results import InversionResult
from .solvers.base import _compile_optimizer

logger = logging.getLogger(__name__)

__all__ = ["Inverter", "InversionResult"]

_ParamDict = dict[str, torch.Tensor]
def _optimizer_name(config: OptimizerConfig) -> str:
    return "adam" if isinstance(config, AdamConfig) else "lbfgs"


def _component_values(
    evaluation: LossEvaluation,
) -> tuple[float, float, float, dict[str, float]]:
    """Detach one post-step diagnostic without another objective call."""
    data_loss = float(evaluation.data_loss.detach())
    regularization_loss = float(
        (
            evaluation.prior_loss
            + evaluation.regularization_loss
        ).detach()
    )
    total_loss = float(evaluation.total_loss.detach())
    term_losses = {
        name: float(value.detach())
        for name, value in evaluation.term_losses.items()
    }
    return total_loss, data_loss, regularization_loss, term_losses


def _validate_term_keys(
    histories: dict[str, list[float]] | None,
    current: Mapping[str, float],
    *,
    completed_iters: int,
) -> dict[str, list[float]] | None:
    """Require one stable named-term schema for all completed iterations."""
    if histories is None:
        if not current:
            return None
        if completed_iters != 0:
            raise GeoBrainError(
                "Inverter term-loss keys appeared after iteration zero",
                object_name="Inverter.run",
                field="term_losses",
                expected="stable term-loss keys for every iteration",
                actual=sorted(current),
            )
        return {name: [] for name in current}
    if set(histories) != set(current):
        raise GeoBrainError(
            "Inverter term-loss keys changed during execution",
            object_name="Inverter.run",
            field="term_losses",
            expected=sorted(histories),
            actual=sorted(current),
        )
    return histories


class Inverter:
    """Tier-2 deterministic inversion facade.

    Every successful outer iteration follows this order:

    ``objective → backward → gradient processors → optimizer step →
    scheduler step → projections → diagnostics → callback → stop check``.

    Callback loss and parameters are the same post-step/post-projection
    observation written to history. Parameters arrive as an owned, detached,
    read-only mapping. Callback truth wins if it also requests cancellation;
    a false/``None`` callback that cancels produces ``StopReason.CANCELLED``.

    Exceptions from objective, backward, processors, step/scheduler,
    projections, diagnostics, or callback remain exceptions. A
    ``GeoBrainError`` is preserved, while other exceptions are wrapped with
    their cause. Both carry a private ``partial_result`` containing only
    iterations whose callback (if any) returned successfully.

    Args:
        problem: Structural inverse problem. MAP prefers
            ``objective(state, ctx)`` for its complete posterior record; MLE
            uses data-only ``loss(state, ctx)``. Legacy third-party protocols
            use the typed private adapter in
            :mod:`geobrain.optim._loss_evaluator`.
        params: Initial tensor mapping, a model exposing ``trainables()``, or
            ``None`` to use ``problem.model.trainables()``.
        optimizer: Frozen Adam or L-BFGS configuration.
        learning_rates: Optional explicit per-parameter Adam learning-rate
            mapping. This preserves Tier-2 parameter groups while the shared
            config remains the only scalar default source.
        regularizer: Optional scalar tensor function of the live parameters.
        bounds: Optional post-step bounds, compiled as the first projection.
        ctx: Forward context reused by every objective call.
        mode: ``"MAP"`` includes ``-log_prior``; ``"MLE"`` does not evaluate
            the prior.
        gradient_processors: Ordered post-backward/pre-step processors.
        step_projections: Ordered post-step projections after bounds.
        scheduler: Optional factory built once over the torch optimizer.
        nan_policy: ``"raise"``, ``"guard"``, or ``"off"`` for gradients.
        result_transform: Private typed seam applied only to detached final
            and best snapshots. Optimization, hooks, and callbacks continue
            to see the original parameter mapping.
    """

    def __init__(
        self,
        problem: InverseProblemLike,
        *,
        params: Mapping[str, torch.Tensor] | Any | None = None,
        optimizer: AdamConfig | LBFGSConfig = AdamConfig(),
        learning_rates: Mapping[str, float] | None = None,
        regularizer: Regularizer | None = None,
        bounds: Mapping[
            str,
            tuple[float | None, float | None],
        ]
        | None = None,
        ctx: ForwardContext | None = None,
        mode: Literal["MAP", "MLE"] = "MAP",
        gradient_processors: Sequence[GradientProcessor] = (),
        step_projections: Sequence[StepProjection] = (),
        scheduler: Callable[[torch.optim.Optimizer], Any] | None = None,
        nan_policy: Literal["guard", "raise", "off"] = "guard",
        result_transform: _ResultTransform | None = None,
    ) -> None:
        if not isinstance(optimizer, (AdamConfig, LBFGSConfig)):
            raise GeoBrainError(
                "Inverter optimizer must be an optimizer config",
                object_name="Inverter",
                field="optimizer",
                expected="AdamConfig or LBFGSConfig",
                actual=type(optimizer),
            )
        if mode not in ("MAP", "MLE"):
            raise GeoBrainError(
                "Inverter mode must be 'MAP' or 'MLE'",
                object_name="Inverter",
                field="mode",
                expected="'MAP' or 'MLE'",
                actual=mode,
            )
        if nan_policy not in ("guard", "raise", "off"):
            raise GeoBrainError(
                "Inverter nan_policy must be 'guard', 'raise', or 'off'",
                object_name="Inverter",
                field="nan_policy",
                expected="'guard', 'raise', or 'off'",
                actual=nan_policy,
            )
        if learning_rates is not None and isinstance(
            optimizer,
            LBFGSConfig,
        ):
            raise GeoBrainError(
                "Inverter per-parameter learning rates require AdamConfig",
                object_name="Inverter",
                field="learning_rates",
                expected="None when optimizer is LBFGSConfig",
                actual=learning_rates,
            )
        if result_transform is not None and not callable(result_transform):
            raise GeoBrainError(
                "Inverter result_transform must be callable",
                object_name="Inverter",
                field="result_transform",
                expected="callable mapping transform or None",
                actual=type(result_transform),
            )

        initial, params_model = _coerce_params(problem, params)
        validate_param_mapping(initial, "Inverter", require_grad=False)
        self._params_model = params_model
        self._params: _ParamDict = {
            name: torch.nn.Parameter(
                tensor.detach().clone().requires_grad_(True)
            )
            for name, tensor in initial.items()
        }
        self._problem = problem
        self._optimizer_config = optimizer
        self._regularizer = regularizer
        self._ctx = ForwardContext() if ctx is None else ctx
        self._mode = mode
        self._loss_evaluator = LossEvaluator(
            self._problem,
            mode=self._mode,
            regularizer=self._regularizer,
            ctx=self._ctx,
        )
        self._nan_policy = nan_policy
        self._nan_warned = False
        self._result_transform = result_transform

        bounds_normalized = _normalize_bounds(bounds, self._params)
        self._step_projections: tuple[StepProjection, ...] = (
            BoundsClamp(bounds_normalized),
            *tuple(step_projections),
        )

        singular = nan_policy == "guard" and (
            bool(
                getattr(
                    getattr(problem, "likelihood", None),
                    "gradient_singular",
                    False,
                )
            )
            or any(
                bool(getattr(item, "gradient_singular", False))
                for item in getattr(problem, "likelihoods", {}).values()
            )
        )
        self._auto_nan_guard = singular
        configured_processors = tuple(gradient_processors)
        self._gradient_processors = (
            (NaNGuard(), *configured_processors)
            if singular
            else configured_processors
        )

        optimizer_params: Any
        if learning_rates is None:
            optimizer_params = self._params.values()
        else:
            normalized = _normalize_lr_mapping(
                learning_rates,
                self._params,
            )
            optimizer_params = [
                {
                    "params": [self._params[name]],
                    "lr": normalized[name],
                }
                for name in self._params
            ]
        self._optimizer = _compile_optimizer(
            optimizer,
            optimizer_params,
        )
        self._scheduler = (
            scheduler(self._optimizer)
            if scheduler is not None
            else None
        )

    @property
    def params(self) -> _ParamDict:
        """Return the live parameter mapping mutated by optimizer steps."""
        return self._params

    @property
    def params_model(self) -> Any | None:
        """Return the model from which ``params`` were coerced, if any."""
        return self._params_model

    @property
    def optimizer_config(self) -> AdamConfig | LBFGSConfig:
        """Return the frozen configuration compiled by this inverter."""
        return self._optimizer_config

    @classmethod
    def from_function(
        cls,
        forward_fn: Callable[..., torch.Tensor],
        observed: torch.Tensor,
        *,
        params: Mapping[str, torch.Tensor],
        likelihood: Likelihood | None = None,
        optimizer: AdamConfig | LBFGSConfig = AdamConfig(),
        learning_rates: Mapping[str, float] | None = None,
        regularizer: Regularizer | None = None,
        bounds: Mapping[
            str,
            tuple[float | None, float | None],
        ]
        | None = None,
        ctx: ForwardContext | None = None,
        gradient_processors: Sequence[GradientProcessor] = (),
        step_projections: Sequence[StepProjection] = (),
    ) -> Inverter:
        """Build an inverter from ``forward_fn(**params) -> Tensor``."""
        from ._factories import _inverter_from_function

        return cast(
            Inverter,
            _inverter_from_function(
                cls,
                forward_fn,
                observed,
                params=params,
                likelihood=likelihood,
                optimizer=optimizer,
                learning_rates=learning_rates,
                regularizer=regularizer,
                bounds=bounds,
                ctx=ctx,
                gradient_processors=gradient_processors,
                step_projections=step_projections,
            ),
        )

    @classmethod
    def from_parameterization(
        cls,
        problem: InverseProblemLike,
        *,
        parameterization: Parameterization,
        theta: torch.Tensor,
        vector_name: str = "theta",
        **kwargs: Any,
    ) -> Inverter:
        """Build an inverter whose internal vector maps to physical fields.

        Optimization, regularizers, bounds, gradient processors, step
        projections, and callbacks all operate on the latent
        ``{vector_name: theta}`` mapping. Normal results expose physical fields
        in ``params``/``best_params`` and owned latent snapshots in metadata.
        Exception ``partial_result.params`` deliberately remains latent and is
        labelled ``metadata["parameter_space"] == "latent"``.
        """
        from ._parameterized_problem import _parameterized_inverter_inputs

        adapted, params, transform = _parameterized_inverter_inputs(
            problem,
            parameterization=parameterization,
            theta=theta,
            vector_name=vector_name,
            kwargs=kwargs,
        )
        return cls(
            adapted,
            params=params,
            result_transform=transform,
            **kwargs,
        )

    def _process_gradients(self) -> None:
        """Apply the configured raw-gradient policy and processors."""
        if self._nan_policy == "raise":
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in self._params.values()
            ):
                raise GeoBrainError(
                    "Inverter encountered a non-finite gradient",
                    object_name="Inverter",
                    field="nan_policy",
                    expected="finite gradients",
                    actual="non-finite gradient",
                )
        elif (
            self._nan_policy == "guard"
            and self._auto_nan_guard
            and not self._nan_warned
        ):
            would_scrub = any(
                parameter.grad is not None
                and parameter.grad.ndim >= 2
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in self._params.values()
            )
            if would_scrub:
                warnings.warn(
                    "Inverter: non-finite gradient scrubbed by the auto "
                    "NaN-guard (nan_policy='guard').",
                    stacklevel=2,
                )
                self._nan_warned = True

        if self._gradient_processors:
            with torch.no_grad():
                for processor in self._gradient_processors:
                    for name, parameter in self._params.items():
                        if parameter.grad is None:
                            continue
                        processed = processor(
                            name,
                            parameter.grad,
                            self._params,
                        )
                        if processed is not parameter.grad:
                            parameter.grad = processed

    def _apply_step_projections(self) -> None:
        """Apply bounds and explicit projections to the post-step iterate."""
        with torch.no_grad():
            for projection in self._step_projections:
                projection(self._params)

    def _apply_bounds(
        self,
        bounds: Mapping[
            str,
            tuple[float | None, float | None],
        ],
    ) -> None:
        """Retain the historical primitive used by bound-seam tests."""
        clamp_params_in_place(self._params, bounds)

    def run(
        self,
        n_iters: int,
        *,
        callback: OptimizationCallback | None = None,
        verbose: bool = False,
        cancellation: CancellationToken | None = None,
    ) -> InversionResult:
        """Run deterministic outer iterations under the shared contract."""
        requested = validate_non_negative_int(
            n_iters,
            owner="Inverter.run",
            field="n_iters",
        )
        evaluator = self._loss_evaluator
        progress = SolverProgress(
            params=self._params,
            requested_iters=requested,
            metadata={
                "optimizer": _optimizer_name(self._optimizer_config),
                **(
                    {"parameter_space": "latent"}
                    if self._result_transform is not None
                    else {}
                ),
            },
        )
        data_history: list[float] = []
        regularization_history: list[float] = []
        term_histories: dict[str, list[float]] | None = None
        best_loss = float("inf")
        best_iter: int | None = None
        best_params: Mapping[str, torch.Tensor] | None = None
        reason = StopReason.COMPLETED
        rolling: LossEvaluation | None = None
        is_lbfgs = isinstance(self._optimizer_config, LBFGSConfig)
        started = time.perf_counter()
        phase = "objective"

        def closure() -> torch.Tensor:
            nonlocal phase
            phase = "objective"
            evaluation = evaluator.evaluate(self._params)
            phase = "backward"
            self._optimizer.zero_grad(set_to_none=True)
            evaluation.total_loss.backward()
            phase = "processors"
            self._process_gradients()
            phase = "step"
            return evaluation.total_loss

        for iteration in range(requested):
            if cancellation is not None and cancellation.is_cancelled:
                reason = StopReason.CANCELLED
                break
            try:
                if is_lbfgs:
                    phase = "step"
                    self._optimizer.step(closure)
                else:
                    if rolling is None:
                        phase = "objective"
                        rolling = evaluator.evaluate(self._params)
                    phase = "backward"
                    self._optimizer.zero_grad(set_to_none=True)
                    rolling.total_loss.backward()
                    phase = "processors"
                    self._process_gradients()
                    phase = "step"
                    self._optimizer.step()

                if self._scheduler is not None:
                    phase = "step"
                    self._scheduler.step()
                phase = "projection"
                self._apply_step_projections()
                phase = "diagnostics"
                post_step = evaluator.evaluate(self._params)
                if not is_lbfgs:
                    rolling = post_step
                (
                    total_value,
                    data_value,
                    regularization_value,
                    term_values,
                ) = _component_values(post_step)
                term_histories = _validate_term_keys(
                    term_histories,
                    term_values,
                    completed_iters=progress.completed_iters,
                )

                loss_stop = _stop_for_loss(total_value)
                if loss_stop is not None:
                    progress.observe_nonfinite(total_value)
                    data_history.append(data_value)
                    regularization_history.append(
                        regularization_value
                    )
                    if term_histories is not None:
                        for name, value in term_values.items():
                            term_histories[name].append(value)
                    reason = loss_stop
                    break

                if verbose:
                    logger.info(
                        "iter %4d: data=%.4e reg=%.4e total=%.4e",
                        iteration,
                        data_value,
                        regularization_value,
                        total_value,
                    )
                phase = "callback"
                callback_stopped = progress.observe(
                    total_value,
                    callback,
                )
                data_history.append(data_value)
                regularization_history.append(regularization_value)
                if term_histories is not None:
                    for name, value in term_values.items():
                        term_histories[name].append(value)

                if total_value < best_loss:
                    best_loss = total_value
                    best_iter = progress.completed_iters - 1
                    best_params = {
                        name: tensor.detach().clone()
                        for name, tensor in self._params.items()
                    }
            except Exception as error:
                _raise_execution_error(
                    error,
                    progress=progress,
                    owner="Inverter.run",
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

        history = progress.history()
        if best_params is None:
            best_loss = float("nan")
            best_iter = None
        return _assemble_inversion_result(
            latent_params=self._params,
            requested_iters=requested,
            completed_iters=progress.completed_iters,
            stop_reason=reason,
            loss_history=history,
            metadata={"optimizer": _optimizer_name(self._optimizer_config)},
            best_latent_params=best_params,
            data_loss_history=data_history,
            reg_loss_history=regularization_history,
            best_loss=best_loss,
            best_iter=best_iter,
            wall_clock_sec=time.perf_counter() - started,
            term_losses=term_histories,
            result_transform=self._result_transform,
        )
