"""NUTS execution protocol, failure boundary, and solver orchestration.

The public facade owns persistent sampler state. This module advances that
state through warmup and trajectories, enforces callback/failure boundaries,
and delegates call-local buffers and result presentation to
:mod:`._nuts_results`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Literal, Mapping, Protocol

import torch

from ...core import ForwardContext, GeoBrainError
from ...core.validation import validate_int
from ..base import (
    LogPosteriorTarget,
    _TargetEvaluationFailure,
    _check_initial_gradients,
    _check_initial_log_post,
    _partial_run_error,
)
from ..execution import (
    _CallbackTransformFailure,
    _NonfiniteEscapeBudget,
    _NonfiniteEscapeExhausted,
    RunAccounting,
    SamplerStopReason,
    callback_snapshot,
)
from ..results import InferenceResult, _PartialInferenceResult
from . import _nuts_warmup
from ._hamiltonian import MassInfo, _log_prob_and_gradient
from ._nuts_results import NUTSRunOutput
from ._nuts_tree import TreeWorkspaceStats, build_trajectory
from ._nuts_warmup import (
    MutableMassInfo,
    WarmupState,
    detached_mass_copy,
    initialize_warmup_state,
    uses_windowed_warmup,
)


OutputSnapshot = Callable[
    [],
    tuple[
        Mapping[str, torch.Tensor],
        WarmupState | None,
        MassInfo | None,
        int,
    ],
]


@dataclass
class NUTSRunBoundary:
    """Translate NUTS execution failures at one truthful phase boundary."""

    output: NUTSRunOutput
    warmup_at_entry: int
    snapshot: OutputSnapshot
    phase: str = "kernel"

    def __enter__(self) -> NUTSRunBoundary:
        """Activate this boundary and allow phase updates inside the run."""
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Preserve first-boundary errors and attach committed private state."""
        del exception_type, traceback
        if exception is None:
            return False
        if isinstance(exception, _TargetEvaluationFailure):
            cause = exception.cause
            if self.phase == "initial" and isinstance(cause, GeoBrainError):
                raise cause
            raise self._structured(cause, field="target") from cause
        if isinstance(exception, _NonfiniteEscapeExhausted):
            error = exception.error
            partial = self._partial()
            if partial is not None:
                setattr(error, "partial_result", partial)
            raise error from None
        if isinstance(exception, _CallbackTransformFailure):
            cause = exception.cause
            raise self._structured(cause, field="transform") from cause
        if self.phase == "initial" and isinstance(exception, GeoBrainError):
            return False
        field = (
            self.phase
            if self.phase in {"target", "callback", "result"}
            else "kernel"
        )
        raise self._structured(exception, field=field) from exception

    def _partial(self) -> _PartialInferenceResult | None:
        if not self.output.committed_since(self.warmup_at_entry):
            return None
        reference, warmup_state, mass_info, n_divergent_total = self.snapshot()
        return self.output.partial(
            reference=reference,
            warmup_state=warmup_state,
            mass_info=mass_info,
            n_divergent_total=n_divergent_total,
        )

    def _structured(
        self,
        cause: BaseException,
        *,
        field: str,
    ) -> GeoBrainError:
        return _partial_run_error(
            "NUTS",
            self.output.accounting.completed_sampling,
            self.output.requested_iters,
            self._partial(),
            cause,
            field=field,
        )


NUTSCallback = Callable[
    [int, float, Mapping[str, torch.Tensor]],
    bool | None,
]


class NUTSSamplerState(Protocol):
    """Persistent facade state consumed and committed by ``run_nuts``."""

    target: LogPosteriorTarget
    params: dict[str, torch.Tensor]
    step_size: float
    max_depth: int
    delta_max: float
    warmup: int
    target_accept: float
    adapt_mass: bool
    mass_type: str
    thin: int
    store_dtype: torch.dtype | None
    _metadata: Mapping[str, Any]
    _field_mass_kind: dict[str, str]
    _injected_mass_info: MutableMassInfo | None
    mass_info_: dict[str, dict[str, Any]] | None
    step_size_final_: float | None
    _theta: dict[str, torch.Tensor]
    _lp_cur: float | None
    _grad_cur: dict[str, torch.Tensor] | None
    _state_ctx: ForwardContext | None
    _default_ctx: ForwardContext
    _completed_warmup: int
    _completed_sampling: int
    _accepted_sampling: int
    _divergent_sampling: int
    _divergent_total: int
    _accept_probability_sum: float
    _run_state: WarmupState | None
    _generator_initial_seed: int

    def _generator_for(
        self,
        device: torch.device,
    ) -> torch.Generator | None:
        ...


def run_nuts(
    sampler: NUTSSamplerState,
    n_iters: int,
    ctx: ForwardContext | None,
    callback: NUTSCallback | None,
) -> InferenceResult:
    """Continue one persistent NUTS chain without replaying adaptation."""
    n_iters = validate_int(
        n_iters,
        owner="NUTS.run",
        field="n_iters",
        minimum=0,
    )
    run_ctx = sampler._default_ctx if ctx is None else ctx
    warmup_at_entry = sampler._completed_warmup
    accounting = RunAccounting(
        requested_iters=n_iters,
        warmup_iters=sampler.warmup,
        completed_warmup=sampler._completed_warmup,
        continued_from_iteration=sampler._completed_sampling,
        continued_accepted_sampling=sampler._accepted_sampling,
        continued_divergent_sampling=sampler._divergent_sampling,
    )
    output = NUTSRunOutput.initialize(
        accounting=accounting,
        reference=sampler._theta,
        warmup=sampler.warmup,
        thin=sampler.thin,
        store_dtype=sampler.store_dtype,
        generator_initial_seed=sampler._generator_initial_seed,
        step_size_initial=sampler.step_size,
        max_depth=sampler.max_depth,
        mass_type_requested=sampler.mass_type,
        mass_injected=sampler._injected_mass_info is not None,
    )
    device = next(iter(sampler._theta.values())).device
    generator = sampler._generator_for(device)
    escape_budget = _NonfiniteEscapeBudget("NUTS")

    def evaluate_gradient(
        position: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return _log_prob_and_gradient(
            sampler.target,
            position,
            sampler._metadata,
            run_ctx,
        )

    def current_mass() -> MassInfo | None:
        return (
            sampler._run_state.mass_info
            if sampler._run_state is not None
            else sampler._injected_mass_info
        )

    boundary = NUTSRunBoundary(
        output=output,
        warmup_at_entry=warmup_at_entry,
        snapshot=lambda: (
            sampler._theta,
            sampler._run_state,
            current_mass(),
            sampler._divergent_total,
        ),
    )
    stop_reason = SamplerStopReason.COMPLETED
    initial = (
        sampler._run_state is None
        and sampler._completed_warmup == 0
        and sampler._completed_sampling == 0
    )
    boundary.phase = "initial" if initial else "kernel"
    with boundary:
        has_work = (
            accounting.completed_warmup < sampler.warmup
            or n_iters > 0
        )
        if sampler._run_state is None and has_work:
            windowed = uses_windowed_warmup(
                sampler.warmup,
                sampler.adapt_mass,
                sampler._field_mass_kind,
            )
            log_prob, gradient = evaluate_gradient(sampler._theta)
            _check_initial_log_post(log_prob, "NUTS")
            _check_initial_gradients(sampler._theta, gradient, "NUTS")
            sampler._lp_cur = float(log_prob.item())
            sampler._grad_cur = gradient
            sampler._state_ctx = run_ctx
            boundary.phase = "kernel"
            epsilon = sampler.step_size
            if windowed:
                epsilon = _nuts_warmup.find_reasonable_epsilon(
                    sampler._theta,
                    sampler._lp_cur,
                    gradient,
                    epsilon,
                    sampler._injected_mass_info,
                    generator,
                    evaluate_gradient,
                )
            sampler._run_state = initialize_warmup_state(
                epsilon,
                warmup=sampler.warmup,
                adapt_mass=sampler.adapt_mass,
                params=sampler.params,
                kinds=sampler._field_mass_kind,
                mass_info=sampler._injected_mass_info,
                windowed=windowed,
            )
        elif (
            has_work
            and sampler._run_state is not None
            and run_ctx is not sampler._state_ctx
        ):
            boundary.phase = "target"
            log_prob, gradient = evaluate_gradient(sampler._theta)
            _check_initial_log_post(log_prob, "NUTS")
            _check_initial_gradients(sampler._theta, gradient, "NUTS")
            sampler._lp_cur = float(log_prob.item())
            sampler._grad_cur = gradient
            sampler._state_ctx = run_ctx
            boundary.phase = "kernel"

        while (
            accounting.completed_warmup < sampler.warmup
            or accounting.completed_sampling < n_iters
        ):
            assert sampler._run_state is not None
            assert sampler._lp_cur is not None
            assert sampler._grad_cur is not None
            state = sampler._run_state
            in_warmup = accounting.completed_warmup < sampler.warmup
            warmup_index = accounting.completed_warmup
            mass_info = state.mass_info
            epsilon = state.epsilon(in_warmup=in_warmup)

            trajectory = build_trajectory(
                sampler._theta,
                sampler._lp_cur,
                sampler._grad_cur,
                step_size=epsilon,
                max_depth=sampler.max_depth,
                delta_max=sampler.delta_max,
                mass_info=mass_info,
                generator=generator,
                device=device,
                evaluate_gradient=evaluate_gradient,
                workspace=TreeWorkspaceStats(),
            )
            proposal = trajectory.proposal
            acceptance = trajectory.accept_sum / max(
                trajectory.accept_count,
                1,
            )
            representative_log_prob = float(proposal.log_prob.item())
            finite = math.isfinite(representative_log_prob)
            escape_budget.observe(finite=finite)
            if not finite:
                continue
            callback_index = (
                warmup_index - sampler.warmup
                if in_warmup
                else accounting.completed_sampling
            )
            should_stop = False
            if callback is not None:
                boundary.phase = "callback"
                should_stop = bool(
                    callback(
                        callback_index,
                        representative_log_prob,
                        callback_snapshot(proposal.position),
                    )
                )
                boundary.phase = "kernel"

            next_state = state
            if in_warmup:
                next_state = state.advance(
                    warmup_index=warmup_index,
                    total_warmup=sampler.warmup,
                    acceptance=acceptance,
                    target_accept=sampler.target_accept,
                    position=proposal.position,
                    adapt_mass=sampler.adapt_mass,
                    params=sampler.params,
                    kinds=sampler._field_mass_kind,
                    restart_epsilon=lambda guess, metric:
                        _nuts_warmup.find_reasonable_epsilon(
                            proposal.position,
                            representative_log_prob,
                            proposal.gradient,
                            guess,
                            metric,
                            generator,
                            evaluate_gradient,
                        ),
                )

            sampler._theta = {
                name: tensor.detach().clone()
                for name, tensor in proposal.position.items()
            }
            sampler._lp_cur = representative_log_prob
            sampler._grad_cur = {
                name: tensor.detach().clone()
                for name, tensor in proposal.gradient.items()
            }
            sampler._run_state = next_state
            if in_warmup:
                accounting.commit_warmup()
                sampler._completed_warmup = accounting.completed_warmup
                sampler._divergent_total += int(trajectory.divergent)
            else:
                accounting.commit_sampling(
                    accepted=True,
                    divergent=trajectory.divergent,
                )
                sampler._completed_sampling += 1
                sampler._accepted_sampling += 1
                sampler._divergent_sampling += int(trajectory.divergent)
                sampler._divergent_total += int(trajectory.divergent)
                sampler._accept_probability_sum += acceptance
                output.record_sampling(
                    sampler._theta,
                    cumulative_iteration=sampler._completed_sampling,
                    log_prob=representative_log_prob,
                    acceptance=acceptance,
                    depth=trajectory.depth,
                )
            if should_stop:
                stop_reason = SamplerStopReason.CALLBACK
                output.record_callback_stop(
                    phase="warmup" if in_warmup else "sampling",
                    iteration=callback_index,
                )
                break

        sampler.mass_info_ = detached_mass_copy(current_mass())
        sampler.step_size_final_ = (
            math.exp(
                sampler._run_state.dual_averaging.log_epsilon_bar
            )
            if sampler._run_state is not None
            else sampler.step_size
        )
        boundary.phase = "result"
        return output.result(
            stop_reason,
            reference=sampler._theta,
            warmup_state=sampler._run_state,
            mass_info=current_mass(),
            n_divergent_total=sampler._divergent_total,
        )

    raise AssertionError("NUTS execution boundary returned without a result")
