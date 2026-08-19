"""
Hamiltonian Monte Carlo for ``InverseProblem``.

Standard formulation: augment the parameter ``θ`` with a momentum ``p`` drawn each
step from ``N(0, M)``. The leapfrog integrator approximates Hamilton's equations
for ``H(θ, p) = -log_posterior(θ) + 0.5 pᵀ M⁻¹ p``. A Metropolis-Hastings step
accepts the proposal with probability ``min(1, exp(H_old - H_new))``.

This implementation:

- Unit mass matrix (``M = I``); mass-matrix preconditioning is a future knob.
- User-set step size and leapfrog length; no dual averaging (see NUTS).
- Per-trajectory step-size jitter (``step_size_jitter``, default 0.2): each
  trajectory integrates with ``eps_t = step_size * (1 + j*(2u-1))``. A fixed
  ``(eps, L)`` with ``eps*L ≈ π*σ_min`` phase-locks the smallest Gaussian
  eigenmode (every trajectory maps it ``x → -x``, so its magnitude never
  mixes: grossly biased std at near-perfect acceptance). Jitter breaks the
  lock; ``step_size_jitter=0.0`` reproduces the fixed-step sampler
  bit-for-bit.
- Divergent trajectories (non-finite Hamiltonian mid-leapfrog) are rejected
  AND counted: ``metadata["n_divergent"]`` (post-warmup; key matches NUTS)
  plus ``"n_divergent_total"`` (warmup included).
- Chain-storage controls: ``warmup`` (pure burn-in: HMC has no adaptation;
  discarded draws, negative callback indices, NUTS's callback contract),
  ``thin`` (keep every thin-th post-warmup draw; storage-only) and
  ``store_dtype`` (cast stored draws only, e.g. float32 to halve memory).
- Per-step ``log_posterior`` recomputed, with gradients via autograd over the full
  ``problem.log_posterior(state, ctx)`` chain.

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
    validate_finite_float,
    validate_int,
    validate_param_name,
)

@Sampler.register("hmc")
class HMC(Sampler):
    """
    Hamiltonian Monte Carlo with leapfrog integrator and unit mass.

    **Tier in the inversion architecture:** Tier 2 (class-based, explicit).

    For most users, prefer the Tier 1 factory::

        samples = problem.as_posterior().sample(
            "hmc", params={"sigma": s0},
            n_iters=500, step_size=0.01, n_leapfrog=20,
        )

    Use this constructor directly when you need the sampler **as an
    object** - for example, to bind it to a variable for custom
    callback wiring, framework integration, or to call
    :meth:`HMC.from_callable` against a hand-built log-posterior closure.

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
        n_leapfrog: int = 20,
        step_size_jitter: float = 0.2,
        warmup: int = 0,
        thin: int = 1,
        store_dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Configure HMC over a target's ``log_posterior``.

        Args:
            target: An :class:`~geobrain.InverseProblem` or
                :class:`~geobrain.bayes.Posterior` (anything with
                ``log_posterior(state, ctx)``).
            params: Initial parameter values; one tensor per field.
            step_size: Leapfrog step size ``ε`` (``> 0``).
            n_leapfrog: Number of leapfrog steps per proposal (``> 0``).
            step_size_jitter: Relative per-trajectory step-size jitter ``j``
                in ``[0, 1)``. Each trajectory integrates with
                ``eps_t = step_size * (1 + j*(2u-1))``, ``u ~ U(0, 1)`` drawn
                from the sampler's generator at the start of the iteration.
                Breaks the leapfrog resonance where a fixed
                ``step_size * n_leapfrog ≈ π*σ_min`` phase-locks the smallest
                eigenmode (silently biased std at near-perfect acceptance).
                ``0.0`` disables jitter, consumes no extra RNG draw, and is
                bit-identical to the pre-jitter sampler. Default ``0.2``
                (note: seeded chains differ from GeoBrain < 0.2.x runs that
                predate this knob; pass ``step_size_jitter=0.0`` to reproduce
                them).
            warmup: Burn-in iterations (``>= 0``, default ``0``) run BEFORE
                the ``n_iters`` recorded ones and discarded from
                ``samples`` / ``log_post_history``. HMC has no step-size or
                mass adaptation, so (unlike NUTS) warmup here is **pure
                burn-in**: it only moves the chain off the starting point.
                The callback contract matches NUTS: warmup iterations fire
                with NEGATIVE indices ``-warmup..-1`` and a truthy return
                CANCELS the run (empty chain); acceptance and ``n_divergent``
                count post-warmup iterations only (``n_divergent_total``
                includes warmup). ``warmup=0`` is byte-identical to the
                pre-warmup sampler.
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
            GeoBrainError: On empty ``params``, non-positive ``step_size`` /
                ``n_leapfrog``, ``step_size_jitter`` outside ``[0, 1)``,
                negative ``warmup``, ``thin < 1``, or a non-floating-point
                ``store_dtype``.
        """
        if target is None:
            raise TypeError("HMC() missing required argument: 'target'")
        if params is None:
            raise TypeError("HMC() missing required argument: 'params'")
        if not params:
            raise GeoBrainError(
                "HMC requires at least one parameter",
                object_name="HMC",
                field="params",
                expected="non-empty mapping",
                actual={},
            )
        step_size = validate_finite_float(
            step_size,
            owner="HMC",
            field="step_size",
            minimum=0.0,
            minimum_inclusive=False,
        )
        n_leapfrog = validate_int(
            n_leapfrog,
            owner="HMC",
            field="n_leapfrog",
            minimum=0,
            minimum_inclusive=False,
        )
        step_size_jitter = validate_finite_float(
            step_size_jitter,
            owner="HMC",
            field="step_size_jitter",
            minimum=0.0,
        )
        if step_size_jitter >= 1.0:
            raise GeoBrainError(
                "HMC step_size_jitter must be in [0, 1)",
                object_name="HMC",
                field="step_size_jitter",
                expected="[0, 1)",
                actual=step_size_jitter,
            )
        warmup = validate_int(
            warmup,
            owner="HMC",
            field="warmup",
            minimum=0,
        )
        thin, store_dtype = validate_chain_storage("HMC", thin, store_dtype)

        for name, tensor in params.items():
            validate_param_name(name, owner="HMC")
            if not isinstance(tensor, torch.Tensor):
                raise GeoBrainError(
                    "HMC params values must be torch.Tensor",
                    object_name="HMC",
                    field=f"params[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(tensor),
                )

        self.target = target
        # Detached working copy; we manage requires_grad ourselves.
        self.params: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in params.items()
        }
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.step_size_jitter = step_size_jitter
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
        self._divergent_sampling = 0
        self._divergent_total = 0
        self._generator_initial_seed = (
            generator.initial_seed() if generator is not None else torch.initial_seed()
        )

    def _log_post_and_grad(
        self, theta: Mapping[str, torch.Tensor], ctx: ForwardContext
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate log_posterior at θ; return (scalar value, dict of gradients)."""
        leaves = _requires_grad_leaves(theta)
        state = ModelState(tensors=leaves, metadata=self._metadata)
        lp = _evaluate_log_posterior(self.target, state, ctx)
        # allow_unused so a param the log-posterior never references (a
        # conditionally-inactive block, or a from_callable closure that ignores
        # a key) is treated as a zero-force flat direction: leapfrog handles a
        # zero gradient correctly: instead of aborting run() with a bare torch
        # RuntimeError ("...appears to not have been used in the graph").
        grads = torch.autograd.grad(
            lp, list(leaves.values()), retain_graph=False, allow_unused=True
        )
        return lp.detach(), {
            name: (g.detach() if g is not None else torch.zeros_like(t))
            for (name, t), g in zip(leaves.items(), grads)
        }

    def _kinetic_energy(self, momentum: Mapping[str, torch.Tensor]) -> torch.Tensor:
        ke = torch.zeros((), dtype=next(iter(momentum.values())).dtype,
                          device=next(iter(momentum.values())).device)
        for p in momentum.values():
            ke = ke + 0.5 * p.pow(2).sum()
        return ke

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
        """Continue this HMC instance for ``n_iters`` sampling transitions.

        Warmup belongs to the instance, resumes after an interrupted call, and
        is never repeated after completion. Callback indices are negative
        cumulative warmup positions and call-local zero-based sampling
        positions. A callback sees the candidate transition through an owned
        snapshot; returning true commits that candidate and stops, while an
        exception leaves it uncommitted.
        """
        n_iters = validate_int(
            n_iters,
            owner="HMC.run",
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
            continued_divergent_sampling=self._divergent_sampling,
        )
        chains: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._theta
        }
        history: list[float] = []
        device = next(iter(self._theta.values())).device
        generator = self._generator_for(device)
        escape_budget = _NonfiniteEscapeBudget("HMC")
        callback_phase: str | None = None
        callback_iteration: int | None = None

        def metadata() -> dict[str, Any]:
            stored_draws = len(next(iter(chains.values()))) if chains else 0
            reference = next(iter(self._theta.values()))
            return {
                "sampler": "HMC",
                "sample_layout": "draws",
                "generator_initial_seed": self._generator_initial_seed,
                "requested_warmup": self.warmup,
                "completed_warmup": accounting.completed_warmup,
                "thin": self.thin,
                "stored_draws": stored_draws,
                "accepted_sampling": accounting.accepted_sampling,
                "divergent_sampling": accounting.divergent_sampling,
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
                "cumulative_divergent_sampling": (
                    accounting.cumulative_divergent_sampling
                ),
                "step_size": self.step_size,
                "step_size_jitter": self.step_size_jitter,
                "n_leapfrog": self.n_leapfrog,
                "warmup": self.warmup,
                "n_accepted": accounting.accepted_sampling,
                "n_completed": (
                    accounting.completed_warmup
                    + accounting.cumulative_completed_sampling
                ),
                "n_divergent": accounting.divergent_sampling,
                "n_divergent_total": self._divergent_total,
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
                lp_cur, grad_cur = self._log_post_and_grad(
                    self._theta,
                    run_ctx,
                )
                _check_initial_log_post(lp_cur, "HMC")
                _check_initial_gradients(self._theta, grad_cur, "HMC")
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

                if self.step_size_jitter > 0.0:
                    unit = torch.rand(
                        (),
                        generator=generator,
                        device=device,
                    ).item()
                    epsilon = self.step_size * (
                        1.0 + self.step_size_jitter * (2.0 * unit - 1.0)
                    )
                else:
                    epsilon = self.step_size

                momentum = {
                    name: torch.randn(
                        tensor.shape,
                        generator=generator,
                        dtype=tensor.dtype,
                        device=tensor.device,
                    )
                    for name, tensor in theta.items()
                }
                old_hamiltonian = -lp_cur + self._kinetic_energy(momentum)
                proposed_theta = {
                    name: tensor.clone() for name, tensor in theta.items()
                }
                proposed_momentum = {
                    name: tensor.clone() for name, tensor in momentum.items()
                }
                for name in proposed_theta:
                    proposed_momentum[name] = (
                        proposed_momentum[name]
                        + 0.5 * epsilon * grad_cur[name]
                    )

                diverged = False
                lp_new = lp_cur
                grad_new = grad_cur
                for leapfrog_index in range(self.n_leapfrog):
                    for name in proposed_theta:
                        proposed_theta[name] = (
                            proposed_theta[name]
                            + epsilon * proposed_momentum[name]
                        )
                    lp_new, grad_new = self._log_post_and_grad(
                        proposed_theta,
                        run_ctx,
                    )
                    if not torch.isfinite(lp_new):
                        diverged = True
                        break
                    kick = (
                        epsilon
                        if leapfrog_index < self.n_leapfrog - 1
                        else 0.5 * epsilon
                    )
                    for name in proposed_theta:
                        proposed_momentum[name] = (
                            proposed_momentum[name] + kick * grad_new[name]
                        )

                new_hamiltonian = -lp_new + self._kinetic_energy(
                    proposed_momentum
                )
                log_alpha = float((old_hamiltonian - new_hamiltonian).item())
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
                finite_next = bool(torch.isfinite(next_lp))
                escape_budget.observe(finite=finite_next)
                if not finite_next:
                    # A ``-inf`` initial state is allowed so the sampler can
                    # escape a hard support wall. An attempt that has not yet
                    # reached finite support consumes RNG but is not a valid
                    # committed transition or history entry.
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
                    self._divergent_total += int(diverged)
                else:
                    accounting.commit_sampling(
                        accepted=accepted,
                        divergent=diverged,
                    )
                    self._completed_sampling += 1
                    self._accepted_sampling += int(accepted)
                    self._divergent_sampling += int(diverged)
                    self._divergent_total += int(diverged)
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
                "HMC",
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
                "HMC",
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
                "HMC",
                accounting.completed_sampling,
                n_iters,
                partial,
                exc,
                field=field,
            ) from exc
