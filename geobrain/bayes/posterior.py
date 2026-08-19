"""
Posterior view over an :class:`~geobrain.InverseProblem`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, cast

import torch

from ..core import ForwardContext, GeoBrainError, ModelState
from .execution import _CallbackTransformFailure, callback_snapshot
from .results import InferenceResult, _PartialInferenceResult
from .transforms import InvertibleTransform

if TYPE_CHECKING:
    from ..inverse import InverseProblem, Prior


class _TransformedTarget:
    """Wraps a :class:`Posterior` so a sampler operates in UNCONSTRAINED space.

    ``log_posterior(u) = base.log_prob(forward(u)) + Σ log|det J_forward(u)|``,
    the change-of-variables density whose pushforward through ``forward`` is the
    constrained posterior. Fields without a transform pass through untransformed.
    Samplers consume this via the ``.log_posterior(state, ctx)`` duck type.
    """

    def __init__(self, base: "Posterior", transforms: Mapping[str, InvertibleTransform]) -> None:
        self._base = base
        self._transforms = transforms

    def log_posterior(
        self, u_state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        constrained: dict[str, torch.Tensor] = {}
        ladj: torch.Tensor | None = None
        for name, u in u_state.tensors.items():
            bij = self._transforms.get(name)
            if bij is None:
                constrained[name] = u
                continue
            constrained[name] = bij.forward(u)
            term = bij.log_abs_det_jacobian(u)
            ladj = term if ladj is None else ladj + term
        # Carry the incoming state's metadata (e.g. resolve()-stamped units)
        # through the constrained rebuild: same contract as the optim-side
        # reparam seams (see geobrain/nn/reparam.py): the tensors are derived,
        # the metadata is not.
        lp = self._base.log_prob(
            ModelState(tensors=constrained, metadata=u_state.metadata), ctx
        )
        return lp if ladj is None else lp + ladj


class Posterior:
    """
    Probabilistic view over an :class:`InverseProblem` plus an optional prior.

    Invariant: this class holds **no** data, likelihood, or forward operator;
    those live on ``self.problem``. The only state stored here is ``self.prior``,
    which either *overrides* ``problem.prior`` (when the caller passed one to
    :meth:`InverseProblem.as_posterior`) or *inherits* it (when the caller passed
    ``None``). When both are ``None`` the posterior reduces to the likelihood (an
    improper flat prior).

    :meth:`log_prob` returns
    ``problem.log_likelihood(state, ctx) + prior.log_prob(state)``.

    Args:
        problem: the inverse problem supplying likelihood and forward.
        prior: optional prior over the sampled parameters; overrides
            ``problem.prior`` when given, inherits it when ``None``.
        transforms: optional ``{param name: bijector}`` mapping, sampling
            then runs in unconstrained space with the Jacobian correction
            applied automatically (:mod:`geobrain.core.transforms`).
    :meth:`sample` dispatches to one of
    ``{HMC, NUTS, LangevinDynamics, SVGD}`` by constructing the sampler
    directly with ``self`` as the target; sampler-specific kwargs
    (``step_size``, ``n_leapfrog``, ``max_depth``, ``adjusted``, ``n_particles``,
    ...) flow through via ``**sampler_kwargs``.

    Note:
        ``log_prob`` and ``log_posterior`` are aliases; the latter exists for
        sampler compatibility (samplers consume targets via the
        ``.log_posterior(state, ctx)`` duck type).
    """

    def __init__(
        self,
        problem: "InverseProblem",
        prior: "Prior | None" = None,
        transforms: Mapping[str, InvertibleTransform] | None = None,
    ) -> None:
        self.problem = problem
        # Explicit prior wins. Otherwise inherit the problem's prior, which
        # may itself be None. ``getattr`` guards against duck-typed problem
        # objects in tests that don't define ``.prior``.
        self.prior = prior if prior is not None else getattr(problem, "prior", None)
        # Optional per-field change-of-variables for constrained parameters
        # (e.g. {"sigma": PositiveTransform()} to sample σ>0 in unconstrained
        # space). Empty = sample every field in its raw space, identical to
        # before.
        self.transforms = self._validate_transforms(transforms)

    @staticmethod
    def _validate_transforms(
        transforms: Mapping[str, InvertibleTransform] | None,
    ) -> dict[str, InvertibleTransform]:
        if not transforms:
            return {}
        out = dict(transforms)
        for name, transform in out.items():
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "Posterior transform names must be non-empty strings",
                    object_name="Posterior",
                    field="transforms",
                    expected="non-empty string keys",
                    actual=name,
                )
            if not isinstance(transform, InvertibleTransform):
                raise GeoBrainError(
                    "Posterior transform values must be InvertibleTransform instances",
                    object_name="Posterior",
                    field=f"transforms[{name!r}]",
                    expected=InvertibleTransform,
                    actual=type(transform),
                )
        return out

    def log_prob(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        """log p(theta | d) up to a constant. Equals likelihood + prior log-prob."""
        ll = self.problem.log_likelihood(state, ctx)
        if self.prior is None:
            return ll
        return ll + self.prior.log_prob(state)

    def log_posterior(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        """Alias for :meth:`log_prob` - kept for naming compatibility."""
        return self.log_prob(state, ctx)

    def sample(
        self,
        method: str,
        *,
        params: Mapping[str, torch.Tensor],
        n_iters: int,
        ctx: ForwardContext | None = None,
        callback: Callable[[int, float, Mapping[str, torch.Tensor]], bool | None] | None = None,
        **sampler_kwargs: Any,
    ) -> InferenceResult:
        """
        Draw samples from this posterior.

        **Tier in the inversion architecture:** Tier 1 (canonical user API
        for Bayesian inversion). Dispatches to
        ``{HMC, NUTS, LangevinDynamics, SVGD}`` by constructing the sampler
        directly with this posterior as the target.

        Args:
            method: ``"hmc"``, ``"nuts"``, ``"langevin"`` or ``"svgd"``
                (case-insensitive).
            params: initial parameter values; one Tensor per field, matching
                the names that ``self.problem.log_likelihood`` (and the
                prior, if any) expect on the ``ModelState``.
            n_iters: number of post-init sampler iterations to draw.
            ctx: optional :class:`ForwardContext` threaded into every
                sampler ``log_posterior`` evaluation.
            callback: optional per-iteration callback forwarded to the
                underlying sampler's ``run``. Receives ``(iteration, log_prob,
                params)``; returning a truthy value stops sampling early.
            **sampler_kwargs: forwarded to the sampler constructor
                (``step_size``, ``n_leapfrog``, ``max_depth``, ``adjusted``,
                ``n_particles``, ``generator``, ...). The MCMC samplers
                (HMC / LangevinDynamics / NUTS) also take the chain-storage /
                burn-in knobs here: ``warmup`` (extra discarded iterations;
                the callback fires with negative indices during warmup),
                ``thin`` (record every ``thin``-th post-warmup draw,
                storage only) and ``store_dtype`` (dtype of the STORED
                draws, e.g. ``torch.float32`` to halve chain memory; the
                sampling math keeps the param dtype).

        Returns:
            :class:`~geobrain.bayes.InferenceResult` from the underlying sampler.

        Note:
            **Log-density space.** When this posterior has transforms (constrained
            parameters), sampling runs in the *unconstrained* space and the returned
            ``samples`` are mapped back to the *constrained* space, but
            ``log_post_history`` is left as the **sampler-space (unconstrained)**
            log-density; it includes the change-of-variables ``log|det J|`` term
            the sampler used. Consequently ``log_post_history[i]`` is **not**
            ``self.log_prob(samples[i])`` (the constrained-space log-prob). To get
            the constrained-space log-prob of a drawn sample, recompute it via the
            model, e.g. ``self.log_prob(ModelState({...samples[i]...}))``. With no
            transforms the two spaces coincide and ``log_post_history`` matches
            ``log_prob`` at each sample.

            **Callback space.** With transforms, a supplied ``callback`` receives
            ``params`` already mapped to the *constrained* space (matching the
            ``params`` you passed and the returned ``samples``), but its
            ``log_post`` argument is the same sampler-space (unconstrained)
            log-density as ``log_post_history``.

            **Partial results.** If the target raises mid-chain, the structured
            error carries the completed draws as ``err.partial_result``; those
            ``samples`` are mapped back to the *constrained* space too, so a
            caller inspecting the partial chain sees the same space as a normal
            return.
        """

        # Lazy import the bayes package so that registering decorators on
        # HMC / NUTS / LangevinDynamics / SVGD fire and populate
        # ``Sampler._registry``
        # even when Posterior is the first symbol the caller touches. Once
        # any sampler module has been imported - directly or via
        # ``import geobrain.bayes`` - this is a no-op.
        import geobrain.bayes  # noqa: F401

        from .base import Sampler  # local to avoid moving the module-level
        # import: keeps Posterior cheap when never used to sample.

        sampler_cls = Sampler.get(method)
        sampler_factory = cast(Any, sampler_cls)

        if not self.transforms:
            sampler = sampler_factory(self, params=params, **sampler_kwargs)
            return cast(
                InferenceResult,
                sampler.run(n_iters=n_iters, ctx=ctx, callback=callback),
            )

        unknown_transforms = set(self.transforms) - set(params)
        if unknown_transforms:
            raise GeoBrainError(
                "Posterior transforms contain unknown parameter names",
                object_name="Posterior.sample",
                field="transforms",
                expected=f"keys in {sorted(params)}",
                actual=sorted(unknown_transforms),
            )

        # Constrained parameters: sample in the UNCONSTRAINED space defined by
        # the transforms (target carries the log|det J| correction), then map the
        # draws back to the constrained space. ``params`` are given in the
        # constrained space and must lie in each transform's domain.
        target = _TransformedTarget(self, self.transforms)
        u_params = {
            name: (self.transforms[name].inverse(v) if name in self.transforms else v)
            for name, v in params.items()
        }
        # Forward-map the callback's params back to CONSTRAINED space so an
        # early-stop / monitor sees the same space as the caller's ``params``
        # and the returned ``samples``: the sampler would otherwise hand it the
        # raw unconstrained iterates (e.g. an ``x['sigma'] > 5`` stop would only
        # fire at true sigma > e^5). The log-density passes through unchanged:
        # it stays sampler-space (unconstrained, with the change-of-variables
        # term), consistent with ``log_post_history``. The constrained log-prob
        # is intentionally not recomputed here because SVGD's callback carries
        # the whole particle cloud, not a single model state.
        run_callback: Callable[
            [int, float, Mapping[str, torch.Tensor]],
            bool | None,
        ] | None = callback
        if callback is not None:
            _user_callback = callback

            def constrained_callback(
                iteration: int,
                log_post: float,
                unconstrained: Mapping[str, torch.Tensor],
            ) -> bool | None:
                try:
                    x_iter = {
                        name: (
                            self.transforms[name].forward(v)
                            if name in self.transforms
                            else v
                        )
                        for name, v in unconstrained.items()
                    }
                except Exception as exc:
                    raise _CallbackTransformFailure(exc) from exc
                return _user_callback(
                    iteration,
                    log_post,
                    callback_snapshot(x_iter),
                )
            run_callback = constrained_callback

        def _to_constrained(
            samples: Mapping[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            # The transforms are elementwise, so ``forward`` maps a whole stacked
            # ``(n_draws, *field_shape)`` (MCMC) or ``(n_particles, ...)`` (SVGD)
            # draw tensor back to the constrained space in one call.
            return {
                name: (
                    self.transforms[name].forward(s)
                    if name in self.transforms
                    else s
                )
                for name, s in samples.items()
            }

        def _metadata_in_space(
            metadata: Mapping[str, Any],
            parameter_space: str,
            **diagnostics: Any,
        ) -> dict[str, Any]:
            return {
                **dict(metadata),
                "parameter_space": parameter_space,
                **diagnostics,
            }

        sampler = sampler_factory(target, params=u_params, **sampler_kwargs)
        try:
            result = sampler.run(n_iters=n_iters, ctx=ctx, callback=run_callback)
        except Exception as exc:
            # A mid-chain failure attaches its completed draws as
            # ``exc.partial_result``: in the sampler's UNCONSTRAINED space. Remap
            # them to the constrained space so a caller reading the partial chain
            # sees the same space as a normal return (samples in x-space;
            # ``log_post_history`` stays sampler-space, matching the documented
            # contract for a full return).
            partial = getattr(exc, "partial_result", None)
            if isinstance(partial, _PartialInferenceResult):
                sampler_space_snapshot = _PartialInferenceResult(
                    samples=partial.samples,
                    log_post_history=partial.log_post_history,
                    acceptance_rate=partial.acceptance_rate,
                    requested_iters=partial.requested_iters,
                    completed_iters=partial.completed_iters,
                    metadata=_metadata_in_space(
                        partial.metadata,
                        "unconstrained",
                    ),
                )
                try:
                    constrained_partial = _PartialInferenceResult(
                        samples=_to_constrained(partial.samples),
                        log_post_history=partial.log_post_history,
                        acceptance_rate=partial.acceptance_rate,
                        requested_iters=partial.requested_iters,
                        completed_iters=partial.completed_iters,
                        metadata=_metadata_in_space(
                            partial.metadata,
                            "constrained",
                        ),
                    )
                except Exception as remap_error:
                    # Remapping diagnostics must never replace the sampler's
                    # primary exception or its original cause chain. Retain a
                    # truthful sampler-space snapshot and record why it could
                    # not cross the transform boundary.
                    unconstrained_partial = _PartialInferenceResult(
                        samples=sampler_space_snapshot.samples,
                        log_post_history=sampler_space_snapshot.log_post_history,
                        acceptance_rate=sampler_space_snapshot.acceptance_rate,
                        requested_iters=sampler_space_snapshot.requested_iters,
                        completed_iters=sampler_space_snapshot.completed_iters,
                        metadata=_metadata_in_space(
                            sampler_space_snapshot.metadata,
                            "unconstrained",
                            partial_remap_error=(
                                f"{type(remap_error).__name__}: {remap_error}"
                            ),
                        ),
                    )
                    setattr(exc, "partial_result", unconstrained_partial)
                else:
                    setattr(exc, "partial_result", constrained_partial)
            raise
        sampler_space_snapshot = _PartialInferenceResult(
            samples=result.samples,
            log_post_history=result.log_post_history,
            acceptance_rate=result.acceptance_rate,
            requested_iters=result.requested_iters,
            completed_iters=result.completed_iters,
            metadata=_metadata_in_space(
                result.metadata,
                "unconstrained",
            ),
        )
        try:
            return InferenceResult(
                samples=_to_constrained(result.samples),
                log_post_history=result.log_post_history,
                acceptance_rate=result.acceptance_rate,
                requested_iters=result.requested_iters,
                completed_iters=result.completed_iters,
                stop_reason=result.stop_reason,
                metadata=_metadata_in_space(
                    result.metadata,
                    "constrained",
                ),
            )
        except Exception as remap_error:
            unconstrained_partial = _PartialInferenceResult(
                samples=sampler_space_snapshot.samples,
                log_post_history=sampler_space_snapshot.log_post_history,
                acceptance_rate=sampler_space_snapshot.acceptance_rate,
                requested_iters=sampler_space_snapshot.requested_iters,
                completed_iters=sampler_space_snapshot.completed_iters,
                metadata=_metadata_in_space(
                    sampler_space_snapshot.metadata,
                    "unconstrained",
                ),
            )
            error = GeoBrainError(
                "Posterior could not transform a completed sampler result to "
                "constrained parameter space; the complete unconstrained "
                "snapshot is attached as private .partial_result",
                object_name="Posterior.sample",
                field="transform",
                expected="constrained owned inference result",
                actual=f"{type(remap_error).__name__}: {remap_error}",
            )
            error.partial_result = unconstrained_partial
            raise error from remap_error
