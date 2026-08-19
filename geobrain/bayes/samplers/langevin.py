"""Langevin Dynamics: the simplest gradient-based sampler.

ULA (Unadjusted Langevin Algorithm)::

    θ_{t+1} = θ_t + (ε² / 2) · ∇ log p(θ_t) + ε · ξ_t,    ξ_t ~ N(0, I)

MALA (Metropolis-Adjusted Langevin): propose ``θ'`` per ULA, then accept with
probability ``α = min(1, exp(log p(θ') - log p(θ) + log q(θ|θ') - log q(θ'|θ)))``.
The proposal ``q`` is asymmetric (drift-dependent), so the correction matters.

A good baseline and a debugging tool for ``log_posterior`` implementations:
**strictly less efficient than HMC / NUTS** on smooth log-posteriors, but easy to
read and reason about.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch

from ...core import ForwardContext, GeoBrainError, ModelState
from ..base import (
    LogPosteriorTarget,
    Sampler,
    _TargetEvaluationFailure,
    _check_initial_gradients,
    _check_initial_log_post,
    _evaluate_log_posterior,
    _partial_run_error,
    _requires_grad_leaves,
)
from ..execution import (
    _CallbackTransformFailure,
    _NonfiniteEscapeBudget,
    _NonfiniteEscapeExhausted,
    RunAccounting,
    SamplerStopReason,
    callback_snapshot,
    stored_draw,
    validate_chain_storage,
)
from ..results import InferenceResult, _PartialInferenceResult
from ...core.validation import (
    validate_bool,
    validate_finite_float,
    validate_int,
    validate_param_name,
)

@Sampler.register("langevin")
class LangevinDynamics(Sampler):
    """
    Unadjusted Langevin or Metropolis-adjusted Langevin sampling.

    **Tier in the inversion architecture:** Tier 2 (class-based, explicit).

    For most users, prefer the Tier 1 factory::

        samples = problem.as_posterior().sample(
            "langevin", params={"sigma": s0},
            n_iters=500, step_size=0.01, adjusted=True,
        )

    Use this constructor directly when you need the sampler **as an
    object** - for example, custom callback wiring or
    :meth:`LangevinDynamics.from_callable` against a hand-built log-posterior
    closure.

    The first argument is named ``target`` and accepts either an
    :class:`~geobrain.InverseProblem` or a
    :class:`~geobrain.bayes.Posterior` (both satisfy the
    ``.log_posterior(state, ctx)`` duck-type).
    """

    def __init__(
        self,
        target: LogPosteriorTarget | None = None,
        *,
        params: Mapping[str, torch.Tensor] | None = None,
        step_size: float = 0.01,
        adjusted: bool = True,
        warmup: int = 0,
        thin: int = 1,
        store_dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Configure Langevin dynamics over a target's ``log_posterior``.

        Args:
            target: An :class:`~geobrain.InverseProblem` or
                :class:`~geobrain.bayes.Posterior`.
            params: Initial parameter values; one tensor per field.
            step_size: Langevin step size ``ε`` (``> 0``).
            adjusted: ``True`` for MALA (Metropolis-adjusted), ``False`` for ULA.
            warmup: Burn-in iterations (``>= 0``, default ``0``) run BEFORE
                the ``n_iters`` recorded ones and discarded from
                ``samples`` / ``log_post_history``. LangevinDynamics has no
                adaptation, so
 (unlike NUTS) warmup here is **pure burn-in**: it only
                moves the chain off the starting point. The callback contract
                matches NUTS: warmup iterations fire with NEGATIVE indices
                ``-warmup..-1`` and a truthy return CANCELS the run (empty
                chain); acceptance counts post-warmup iterations only.
                ``warmup=0`` is byte-identical to the pre-warmup sampler.
            thin: Record every ``thin``-th post-warmup draw (``>= 1``,
                default ``1``): post-warmup iterations ``thin-1, 2*thin-1,
                ...`` (0-based) are stored, exactly ``dense[thin-1::thin]``
                of a same-seed unthinned run, ``floor(n_iters/thin)`` draws
                total (the Stan/ArviZ keep-every-k-th convention). Thinning
                is STORAGE-ONLY: iteration counting, RNG use, callback
                indices and acceptance statistics are unchanged, and
                ``log_post_history`` is thinned alongside so its length
                always matches ``samples``.
            store_dtype: Optional dtype for the STORED draws (``None`` =
                parameter dtype, today's behaviour). E.g. ``torch.float32``
                halves chain memory for float64 params. The sampling math
                stays in the parameter dtype; only the recorded clones are
                cast (``log_post_history`` stays float64).
            generator: Optional RNG for reproducibility.
            metadata: Optional ``ModelState``-style metadata (e.g. a
                ``{"units": {...}}`` mapping carried over from an
                ``EarthModel.resolve()`` call that seeded ``params``) to stamp
                onto every ``ModelState`` handed to the target's
                ``log_posterior``, so a problem/prior that reads units off
                the state sees them mid-chain. Defaults to empty. Distinct
                from ``InferenceResult.metadata`` (the sampler's own
                diagnostics namespace), which this never enters.

        Raises:
            GeoBrainError: On empty ``params``, non-positive ``step_size``,
                negative ``warmup``, ``thin < 1``, or a non-floating-point
                ``store_dtype``.
        """
        if target is None:
            raise TypeError(
                "LangevinDynamics() missing required argument: 'target'"
            )
        if params is None:
            raise TypeError(
                "LangevinDynamics() missing required argument: 'params'"
            )
        if not params:
            raise GeoBrainError(
                "LangevinDynamics requires at least one parameter",
                object_name="LangevinDynamics",
                field="params",
                expected="non-empty mapping",
                actual={},
            )
        step_size = validate_finite_float(
            step_size,
            owner="LangevinDynamics",
            field="step_size",
            minimum=0.0,
            minimum_inclusive=False,
        )
        adjusted = validate_bool(
            adjusted, owner="LangevinDynamics", field="adjusted"
        )
        warmup = validate_int(
            warmup,
            owner="LangevinDynamics",
            field="warmup",
            minimum=0,
        )
        thin, store_dtype = validate_chain_storage(
            "LangevinDynamics", thin, store_dtype
        )
        for name, tensor in params.items():
            validate_param_name(name, owner="LangevinDynamics")
            if not isinstance(tensor, torch.Tensor):
                raise GeoBrainError(
                    "LangevinDynamics params values must be torch.Tensor",
                    object_name="LangevinDynamics",
                    field=f"params[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(tensor),
                )

        self.target = target
        self.params: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in params.items()
        }
        self.step_size = step_size
        self.adjusted = adjusted
        self.warmup = warmup
        self.thin = thin
        self.store_dtype = store_dtype
        self.generator = generator
        self._metadata: Mapping[str, Any] = dict(metadata) if metadata else {}
        self._theta = {
            name: tensor.detach().clone() for name, tensor in self.params.items()
        }
        self._lp_cur: torch.Tensor | None = None
        self._grad_cur: dict[str, torch.Tensor] | None = None
        self._state_ctx: ForwardContext | None = None
        self._default_ctx = ForwardContext()
        self._completed_warmup = 0
        self._completed_sampling = 0
        self._accepted_sampling = 0
        self._generator_initial_seed = (
            generator.initial_seed() if generator is not None else torch.initial_seed()
        )

    def _lp_and_grad(
        self, theta: Mapping[str, torch.Tensor], ctx: ForwardContext
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        leaves = _requires_grad_leaves(theta)
        lp = _evaluate_log_posterior(
            self.target,
            ModelState(tensors=leaves, metadata=self._metadata),
            ctx,
        )
        # allow_unused so a param absent from the log-posterior graph is a
        # zero-force flat direction (Langevin handles a zero gradient) rather
        # than a bare torch RuntimeError that aborts run(). Matches HMC / NUTS.
        grads = torch.autograd.grad(lp, list(leaves.values()), allow_unused=True)
        return lp.detach(), {
            n: (g.detach() if g is not None else torch.zeros_like(t))
            for (n, t), g in zip(leaves.items(), grads)
        }

    def run(
        self,
        n_iters: int,
        ctx: ForwardContext | None = None,
        callback: Callable[
            [int, float, Mapping[str, torch.Tensor]],
            bool | None,
        ]
        | None = None,
    ) -> InferenceResult:
        """Continue MALA or ULA without repeating completed warmup."""
        n_iters = validate_int(
            n_iters,
            owner="LangevinDynamics.run",
            field="n_iters",
            minimum=0,
        )
        run_ctx = self._default_ctx if ctx is None else ctx
        completed_warmup_at_entry = self._completed_warmup
        accounting = RunAccounting(
            requested_iters=n_iters,
            warmup_iters=self.warmup,
            completed_warmup=self._completed_warmup,
            continued_from_iteration=self._completed_sampling,
            continued_accepted_sampling=self._accepted_sampling,
        )
        chains: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._theta
        }
        history: list[float] = []
        device = next(iter(self._theta.values())).device
        generator = self._generator_for(device)
        escape_budget = _NonfiniteEscapeBudget("LangevinDynamics")
        callback_phase: str | None = None
        callback_iteration: int | None = None

        def metadata() -> dict[str, Any]:
            stored_draws = len(next(iter(chains.values()))) if chains else 0
            reference = next(iter(self._theta.values()))
            sampler_name = "MALA" if self.adjusted else "ULA"
            return {
                "sampler": sampler_name,
                "sample_layout": "draws",
                "generator_initial_seed": self._generator_initial_seed,
                "requested_warmup": self.warmup,
                "completed_warmup": accounting.completed_warmup,
                "thin": self.thin,
                "stored_draws": stored_draws,
                "accepted_sampling": accounting.accepted_sampling,
                "divergent_sampling": 0,
                "continued_from_iteration": accounting.continued_from_iteration,
                "dtype": str(reference.dtype),
                "device": str(reference.device),
                "callback_phase": callback_phase,
                "callback_iteration": callback_iteration,
                "cumulative_completed_sampling": (
                    accounting.cumulative_completed_sampling
                ),
                "cumulative_accepted_sampling": (
                    accounting.cumulative_accepted_sampling
                ),
                "cumulative_divergent_sampling": 0,
                "step_size": self.step_size,
                "adjusted": self.adjusted,
                "warmup": self.warmup,
                "n_accepted": accounting.accepted_sampling,
                "n_completed": (
                    accounting.completed_warmup
                    + accounting.cumulative_completed_sampling
                ),
            }

        def result(reason: SamplerStopReason) -> InferenceResult:
            return InferenceResult(
                samples=self._stack_chains(
                    chains,
                    self._theta,
                    self.store_dtype,
                ),
                log_post_history=torch.as_tensor(history, dtype=torch.float64),
                acceptance_rate=accounting.acceptance_rate,
                requested_iters=n_iters,
                completed_iters=accounting.completed_sampling,
                stop_reason=reason,
                metadata=metadata(),
            )

        def partial_result() -> _PartialInferenceResult:
            return _PartialInferenceResult(
                samples=self._stack_chains(
                    chains,
                    self._theta,
                    self.store_dtype,
                ),
                log_post_history=torch.as_tensor(history, dtype=torch.float64),
                acceptance_rate=accounting.acceptance_rate,
                requested_iters=n_iters,
                completed_iters=accounting.completed_sampling,
                metadata=metadata(),
            )

        def committed_in_call() -> bool:
            return (
                accounting.completed_sampling > 0
                or accounting.completed_warmup > completed_warmup_at_entry
            )

        stop_reason = SamplerStopReason.COMPLETED
        epsilon = self.step_size
        half_epsilon_sq = 0.5 * epsilon * epsilon
        initial_target = (
            self._lp_cur is None
            and self._grad_cur is None
            and self._completed_warmup == 0
            and self._completed_sampling == 0
        )
        execution_phase = "initial" if initial_target else "kernel"
        try:
            if (
                self._lp_cur is None
                or self._grad_cur is None
                or run_ctx is not self._state_ctx
            ):
                if not initial_target:
                    execution_phase = "target"
                lp_cur, grad_cur = self._lp_and_grad(self._theta, run_ctx)
                _check_initial_log_post(lp_cur, "LangevinDynamics")
                _check_initial_gradients(
                    self._theta,
                    grad_cur,
                    "LangevinDynamics",
                )
                self._lp_cur = lp_cur
                self._grad_cur = grad_cur
                self._state_ctx = run_ctx
            execution_phase = "kernel"

            while (
                accounting.completed_warmup < self.warmup
                or accounting.completed_sampling < n_iters
            ):
                in_warmup = accounting.completed_warmup < self.warmup
                theta = self._theta
                lp_cur = self._lp_cur
                grad_cur = self._grad_cur
                assert lp_cur is not None and grad_cur is not None
                noise = {
                    name: torch.randn(
                        tensor.shape,
                        generator=generator,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    for name, tensor in theta.items()
                }
                proposed_theta = {
                    name: (
                        theta[name]
                        + half_epsilon_sq * grad_cur[name]
                        + epsilon * noise[name]
                    )
                    for name in theta
                }

                if self.adjusted:
                    lp_new, grad_new = self._lp_and_grad(
                        proposed_theta,
                        run_ctx,
                    )
                    if not torch.isfinite(lp_new):
                        accepted = False
                    else:
                        log_q_forward: torch.Tensor | float = 0.0
                        log_q_backward: torch.Tensor | float = 0.0
                        for name in theta:
                            forward_mean = (
                                theta[name]
                                + half_epsilon_sq * grad_cur[name]
                            )
                            backward_mean = (
                                proposed_theta[name]
                                + half_epsilon_sq * grad_new[name]
                            )
                            log_q_forward += -(
                                (proposed_theta[name] - forward_mean) ** 2
                            ).sum() / (2 * epsilon * epsilon)
                            log_q_backward += -(
                                (theta[name] - backward_mean) ** 2
                            ).sum() / (2 * epsilon * epsilon)
                        log_alpha = float(
                            (
                                lp_new
                                - lp_cur
                                + log_q_backward
                                - log_q_forward
                            ).item()
                        )
                        unit = torch.rand(
                            (),
                            generator=generator,
                            device=device,
                        ).item()
                        if math.isnan(log_alpha):
                            accepted = False
                        elif log_alpha >= 0.0:
                            accepted = True
                        else:
                            accepted = math.log(unit) < log_alpha
                    next_theta = proposed_theta if accepted else theta
                    next_lp = lp_new if accepted else lp_cur
                    next_grad = grad_new if accepted else grad_cur
                else:
                    lp_new, grad_new = self._lp_and_grad(
                        proposed_theta,
                        run_ctx,
                    )
                    finite_gradient = all(
                        bool(torch.isfinite(value).all())
                        for value in grad_new.values()
                    )
                    if not bool(torch.isfinite(lp_new)) or not finite_gradient:
                        execution_phase = "target"
                        raise GeoBrainError(
                            "ULA proposal log-posterior and gradient must be finite; "
                            "the failed proposal was not committed and RNG state "
                            "was not rewound",
                            object_name="LangevinDynamics",
                            field="target",
                            expected="finite candidate log-posterior and gradients",
                            actual={
                                "log_posterior_finite": bool(torch.isfinite(lp_new)),
                                "gradient_finite": finite_gradient,
                            },
                        )
                    accepted = True
                    next_theta = proposed_theta
                    next_lp = lp_new
                    next_grad = grad_new

                finite_next = bool(torch.isfinite(next_lp))
                escape_budget.observe(finite=finite_next)
                if not finite_next:
                    # Preserve deliberate hard-wall escape semantics without
                    # committing a non-finite returned history entry.
                    continue
                callback_index = (
                    accounting.completed_warmup - self.warmup
                    if in_warmup
                    else accounting.completed_sampling
                )
                should_stop = False
                if callback is not None:
                    execution_phase = "callback"
                    should_stop = bool(
                        callback(
                            callback_index,
                            float(next_lp.item()),
                            callback_snapshot(next_theta),
                        )
                    )
                    execution_phase = "kernel"

                self._theta = {
                    name: tensor.detach().clone()
                    for name, tensor in next_theta.items()
                }
                self._lp_cur = next_lp.detach().clone()
                self._grad_cur = {
                    name: tensor.detach().clone()
                    for name, tensor in next_grad.items()
                }
                if in_warmup:
                    accounting.commit_warmup()
                    self._completed_warmup = accounting.completed_warmup
                else:
                    accounting.commit_sampling(accepted=accepted)
                    self._completed_sampling += 1
                    self._accepted_sampling += int(accepted)
                    history.append(float(next_lp.item()))
                    if self._completed_sampling % self.thin == 0:
                        for name in chains:
                            chains[name].append(
                                stored_draw(
                                    self._theta[name],
                                    self.store_dtype,
                                )
                            )
                if should_stop:
                    stop_reason = SamplerStopReason.CALLBACK
                    callback_phase = "warmup" if in_warmup else "sampling"
                    callback_iteration = callback_index
                    break
            execution_phase = "result"
            return result(stop_reason)
        except _TargetEvaluationFailure as failure:
            cause = failure.cause
            if execution_phase == "initial" and isinstance(cause, GeoBrainError):
                raise cause
            partial = partial_result() if committed_in_call() else None
            raise _partial_run_error(
                "LangevinDynamics",
                accounting.completed_sampling,
                n_iters,
                partial,
                cause,
                field="target",
            ) from cause
        except _NonfiniteEscapeExhausted as failure:
            error = failure.error
            partial = partial_result() if committed_in_call() else None
            if partial is not None:
                setattr(error, "partial_result", partial)
            raise error from None
        except _CallbackTransformFailure as failure:
            cause = failure.cause
            partial = partial_result() if committed_in_call() else None
            raise _partial_run_error(
                "LangevinDynamics",
                accounting.completed_sampling,
                n_iters,
                partial,
                cause,
                field="transform",
            ) from cause
        except Exception as exc:
            if execution_phase == "initial" and isinstance(exc, GeoBrainError):
                raise
            partial = partial_result() if committed_in_call() else None
            field = (
                execution_phase
                if execution_phase in {"target", "callback", "result"}
                else "kernel"
            )
            raise _partial_run_error(
                "LangevinDynamics",
                accounting.completed_sampling,
                n_iters,
                partial,
                exc,
                field=field,
            ) from exc
