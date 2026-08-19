"""
Stein Variational Gradient Descent.

SVGD evolves ``n_particles`` particles toward the posterior by combining a
score-driven attractive term and a kernel-driven repulsive term:

    φ(x) = (1/N) Σ_j [ k(x_j, x) ∇log p(x_j) + ∇_{x_j} k(x_j, x) ]
    x_i ← x_i + ε · φ(x_i)

With an RBF kernel ``k(x, x') = exp(-||x - x'||² / 2 h²)``,

    ∇_{x_j} k(x_j, x) = k(x_j, x) · (x - x_j) / h²,
    so  Σ_j ∇_{x_j} k(x_j, x_i)  =  (1/h²) (rowsum(K) ⊙ x_i − K @ X).

The bandwidth ``h`` uses the median-distance heuristic
``h² = median(pairwise sq. dist) / log N`` (Liu & Wang, 2016).

Note: ``n_iters`` here means *update iterations*. The output ``samples``
holds, per field, the final particle ensemble ``(n_particles, *field_shape)``
that approximates the posterior.

Multi-field: several trainable fields are supported by running SVGD on the
PRODUCT space: each particle is the flat concatenation of the fields (in
``params`` order), the RBF kernel and its median-heuristic bandwidth act on
that joint vector, and samples / callbacks / partial results are unpacked
back to per-field tensors. All fields must share one dtype and device (the
joint kernel adds their squared distances). With a single field the particle
cloud keeps its natural ``(n_particles, *field_shape)`` layout, exactly as
before.

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
    _check_initial_log_post,
    _evaluate_log_posterior,
    _partial_run_error,
)
from ..execution import (
    _CallbackTransformFailure,
    RunAccounting,
    SamplerStopReason,
    callback_snapshot,
)
from ..results import InferenceResult, _PartialInferenceResult
from ...core.validation import (
    validate_finite_float,
    validate_int,
    validate_param_name,
)


# Sentinel distinguishing "init_spread not passed" from an explicit
# ``init_spread=1.0`` (which happens to equal the historical default), the
# ``init_particles=`` path needs to tell the two apart to reject an explicit
# init_spread rather than silently ignore it (see the ctor docstring).
_UNSET = object()


@Sampler.register("svgd")
class SVGD(Sampler):
    """
    Stein Variational Gradient Descent (one or more trainable fields).

    **Tier in the inversion architecture:** Tier 2 (class-based, explicit).

    For most users, prefer the Tier 1 factory::

        samples = problem.as_posterior().sample(
            "svgd", params={"sigma": s0},
            n_iters=200, n_particles=20, step_size=0.1,
        )

    Use this constructor directly when you need the sampler **as an
    object** - for example, to inspect ``self.particles`` mid-run, wire
    custom callbacks, or call :meth:`SVGD.from_callable` against a
    hand-built log-posterior closure.

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
        n_particles: int = 20,
        step_size: float = 0.1,
        init_spread: float = _UNSET,  # type: ignore[assignment]
        init_particles: Mapping[str, torch.Tensor] | None = None,
        generator: torch.Generator | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Configure SVGD over a target's ``log_posterior``.

        Args:
            target: An :class:`~geobrain.InverseProblem` or
                :class:`~geobrain.bayes.Posterior`.
            params: ``{name: tensor}`` trainable fields (one or more). With
                several fields, all must share one dtype and device (the
                particle is their flat concatenation and the kernel acts on
                the joint vector). Required unless ``init_particles=`` is
                given (in which case it is optional; see ``init_particles``
                below); when both are given their field-name sets must
                match.
            n_particles: Number of particles (``>= 2``). Ignored when
                ``init_particles=`` is given; the particle count is read
                off ``init_particles``' leading dimension instead.
            step_size: SVGD step size ``ε`` (``> 0``).
            init_spread: Std of the Gaussian jitter used to seed particles
                around the initial value (``> 0``). Mutually exclusive with
                ``init_particles=`` (raises if both are explicitly given;
                there is no anchor to jitter around once particles are
                supplied directly).
            init_particles: ``{name: Tensor(n_particles, *field_shape)}``,
                a pre-built particle cloud per field, e.g. from
                :func:`geobrain.geomodel.bridge.ensemble_to_particles`.
                Bypasses the anchor+jitter initialisation entirely: the
                cloud is used AS GIVEN (bitwise, modulo a dtype/device cast
                to the first field's dtype/device), consistent with the
                "flat concatenation of fields in ``field_names`` order"
                layout ``params=`` builds. All fields must share one
                ``n_particles`` (leading dim) and one dtype/device (same
                product-space rule as ``params=``).
            generator: Optional RNG for reproducibility. Unused when
                ``init_particles=`` is given (no random draw happens).
            metadata: Optional ``ModelState``-style metadata (e.g. a
                ``{"units": {...}}`` mapping carried over from an
                ``EarthModel.resolve()`` call that seeded ``params``) to stamp
                onto every per-particle ``ModelState`` handed to the target's
                ``log_posterior``, so a problem/prior that reads units off
                the state sees them mid-run. Defaults to empty. Distinct
                from ``InferenceResult.metadata`` (the sampler's own
                diagnostics namespace), which this never enters.

        Raises:
            GeoBrainError: On empty ``params``, mixed field dtypes / devices,
                ``n_particles < 2``, non-positive ``step_size`` /
                ``init_spread``, ``init_spread=`` combined with
                ``init_particles=``, inconsistent ``init_particles`` leading
                dimensions, or a ``params``/``init_particles`` field-name
                mismatch (when both are given).
        """
        if target is None:
            raise TypeError("SVGD() missing required argument: 'target'")

        # Shared by both init paths (params= anchor+jitter and init_particles=).
        self._metadata: Mapping[str, Any] = dict(metadata) if metadata else {}

        if init_particles is not None:
            self._init_from_particles(
                target=target, params=params, init_particles=init_particles,
                step_size=step_size, init_spread=init_spread, generator=generator,
            )
            return

        if params is None:
            raise TypeError("SVGD() missing required argument: 'params'")
        if not params:
            raise GeoBrainError(
                "SVGD params must contain at least one trainable field",
                object_name="SVGD",
                field="params",
                expected="non-empty mapping",
                actual={},
            )
        n_particles = validate_int(
            n_particles,
            owner="SVGD",
            field="n_particles",
            minimum=2,
        )
        step_size = validate_finite_float(
            step_size,
            owner="SVGD",
            field="step_size",
            minimum=0.0,
            minimum_inclusive=False,
        )

        self.target = target
        anchors: list[torch.Tensor] = []
        for name, anchor in params.items():
            validate_param_name(name, owner="SVGD")
            if not isinstance(anchor, torch.Tensor):
                raise GeoBrainError(
                    "SVGD params value must be a torch.Tensor",
                    object_name="SVGD",
                    field=f"params[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(anchor),
                )
            anchors.append(anchor)
        self.field_names = tuple(params.keys())
        self.field_shapes = {
            name: tuple(a.shape) for name, a in zip(self.field_names, anchors)
        }
        ref = anchors[0]
        for name, a in zip(self.field_names[1:], anchors[1:]):
            if a.dtype != ref.dtype or a.device != ref.device:
                raise GeoBrainError(
                    "SVGD multi-field params must share one dtype and device "
                    "(the particle is their flat concatenation)",
                    object_name="SVGD",
                    field=f"params[{name!r}]",
                    expected=f"{ref.dtype} on {ref.device}",
                    actual=f"{a.dtype} on {a.device}",
                )
        numels = [int(a.numel()) for a in anchors]
        self._dim_total = sum(numels)
        offsets = [0]
        for n in numels:
            offsets.append(offsets[-1] + n)
        self._field_slices = tuple(
            (offsets[i], offsets[i + 1]) for i in range(len(numels))
        )
        self._multi = len(self.field_names) > 1
        # Single-field compatibility surface (pre-multi-field API).
        self.name = self.field_names[0]
        self.field_shape = self.field_shapes[self.name]

        self.n_particles = n_particles
        self.step_size = step_size
        self.generator = generator

        if init_spread is _UNSET:
            init_spread = 1.0
        init_spread = validate_finite_float(
            init_spread,
            owner="SVGD",
            field="init_spread",
            minimum=0.0,
            minimum_inclusive=False,
        )
        # ``_generator_for`` reconciles the injected generator with the anchor
        # device (no-op when they match) so a CPU-seeded SVGD seeds CUDA
        # particles instead of crashing on this draw.
        if not self._multi:
            # Single field: keep the historical (n_particles, *field_shape)
            # cloud layout and the exact RNG draw, bit-for-bit.
            anchor = anchors[0]
            noise = torch.randn(
                (self.n_particles,) + self.field_shape,
                dtype=anchor.dtype, device=anchor.device,
                generator=self._generator_for(anchor.device),
            )
            self.particles = anchor.detach().unsqueeze(0).expand_as(noise).clone() \
                + init_spread * noise
        else:
            # Product space: each particle is the flat concat of the fields
            # (params order); one joint draw seeds the jitter.
            flat_anchor = torch.cat([a.detach().reshape(-1) for a in anchors])
            noise = torch.randn(
                (self.n_particles, self._dim_total),
                dtype=ref.dtype, device=ref.device,
                generator=self._generator_for(ref.device),
            )
            self.particles = flat_anchor.unsqueeze(0) + init_spread * noise
        self._initialize_execution_state()

    def _init_from_particles(
        self,
        *,
        target: LogPosteriorTarget,
        params: Mapping[str, torch.Tensor] | None,
        init_particles: Mapping[str, torch.Tensor],
        step_size: float,
        init_spread: float,
        generator: torch.Generator | None,
    ) -> None:
        """``init_particles=`` path: build ``self.particles`` directly from a
        pre-supplied per-field cloud; no anchor, no jitter, no RNG draw.
        Mirrors the field bookkeeping the ``params=`` path derives from
        anchors (``field_names`` / ``field_shapes`` / ``_field_slices`` /
        ``_dim_total`` / ``_multi`` / ``name`` / ``field_shape``), but reads
        the per-particle SHAPE (not a single anchor's shape) off each
        tensor's trailing dims.
        """
        if init_spread is not _UNSET:
            raise GeoBrainError(
                "SVGD init_spread cannot be combined with init_particles, "
                "init_particles bypasses the anchor+jitter initialisation "
                "entirely, so there is no anchor to jitter around",
                object_name="SVGD",
                field="init_spread",
                expected="omit init_spread when init_particles is given",
                actual=init_spread,
            )
        if not isinstance(init_particles, Mapping) or not init_particles:
            raise GeoBrainError(
                "SVGD init_particles must be a non-empty mapping",
                object_name="SVGD",
                field="init_particles",
                expected="non-empty mapping",
                actual=init_particles,
            )
        if params is not None and set(params) != set(init_particles):
            raise GeoBrainError(
                "SVGD params and init_particles must name the same fields "
                "when both are given",
                object_name="SVGD",
                field="params/init_particles",
                expected=sorted(init_particles),
                actual=sorted(params),
            )
        step_size = validate_finite_float(
            step_size,
            owner="SVGD",
            field="step_size",
            minimum=0.0,
            minimum_inclusive=False,
        )

        field_names = tuple(init_particles.keys())
        clouds: list[torch.Tensor] = []
        for name in field_names:
            validate_param_name(name, owner="SVGD")
            t = init_particles[name]
            if not isinstance(t, torch.Tensor):
                raise GeoBrainError(
                    "SVGD init_particles value must be a torch.Tensor",
                    object_name="SVGD",
                    field=f"init_particles[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(t),
                )
            if t.ndim < 1:
                raise GeoBrainError(
                    "SVGD init_particles value must have a leading "
                    "n_particles dimension",
                    object_name="SVGD",
                    field=f"init_particles[{name!r}]",
                    expected=">= 1-D tensor",
                    actual=f"{t.ndim}-D",
                )
            clouds.append(t)

        n_by_field = {name: int(t.shape[0]) for name, t in zip(field_names, clouds)}
        if len(set(n_by_field.values())) != 1:
            raise GeoBrainError(
                "SVGD init_particles fields must share one n_particles "
                "(the leading dimension)",
                object_name="SVGD",
                field="init_particles",
                expected="identical leading dim across fields",
                actual=n_by_field,
            )
        n_particles = validate_int(
            next(iter(n_by_field.values())),
            owner="SVGD",
            field="init_particles n_particles",
            minimum=2,
        )

        ref = clouds[0]
        for name, t in zip(field_names[1:], clouds[1:]):
            if t.dtype != ref.dtype or t.device != ref.device:
                raise GeoBrainError(
                    "SVGD multi-field init_particles must share one dtype "
                    "and device (the particle is their flat concatenation)",
                    object_name="SVGD",
                    field=f"init_particles[{name!r}]",
                    expected=f"{ref.dtype} on {ref.device}",
                    actual=f"{t.dtype} on {t.device}",
                )

        self.target = target
        self.field_names = field_names
        self.field_shapes = {
            name: tuple(t.shape[1:]) for name, t in zip(field_names, clouds)
        }
        numels = [int(t[0].numel()) for t in clouds]
        self._dim_total = sum(numels)
        offsets = [0]
        for nel in numels:
            offsets.append(offsets[-1] + nel)
        self._field_slices = tuple(
            (offsets[i], offsets[i + 1]) for i in range(len(numels))
        )
        self._multi = len(field_names) > 1
        self.name = field_names[0]
        self.field_shape = self.field_shapes[self.name]

        self.n_particles = n_particles
        self.step_size = step_size
        self.generator = generator

        if not self._multi:
            self.particles = clouds[0].detach().clone()
        else:
            flat_parts = [t.detach().reshape(n_particles, -1) for t in clouds]
            self.particles = torch.cat(flat_parts, dim=1).clone()
        self._initialize_execution_state()

    def _initialize_execution_state(self) -> None:
        """Initialize continuation accounting after either particle init path."""
        self._default_ctx = ForwardContext()
        self._completed_sampling = 0
        self._generator_initial_seed = (
            self.generator.initial_seed()
            if self.generator is not None
            else torch.initial_seed()
        )

    # SVGD inherits ``Sampler.from_callable``; ``init`` may hold one field
    # (historical cloud layout) or several (flat product-space particles).

    def unpack_particles(
        self, cloud: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Per-field views of a particle cloud (default: ``self.particles``).

        Single field: ``{name: cloud}`` unchanged. Multi-field: slices the
        flat ``(n_particles, D_total)`` cloud back into
        ``(n_particles, *field_shape)`` tensors, in ``params`` order.
        """
        if cloud is None:
            cloud = self.particles
        if not self._multi:
            return {self.name: cloud}
        n = cloud.shape[0]
        return {
            name: cloud[:, s:e].reshape((n,) + self.field_shapes[name])
            for name, (s, e) in zip(self.field_names, self._field_slices)
        }

    def _leaves_for_row(
        self, row: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        """Detached-leaf tensors for one particle, keyed per field."""
        if not self._multi:
            leaf = row.detach().clone().requires_grad_(True)
            return {self.name: leaf}, [leaf]
        leaves: dict[str, torch.Tensor] = {}
        leaf_list: list[torch.Tensor] = []
        for name, (s, e) in zip(self.field_names, self._field_slices):
            leaf = (
                row[s:e]
                .reshape(self.field_shapes[name])
                .detach()
                .clone()
                .requires_grad_(True)
            )
            leaves[name] = leaf
            leaf_list.append(leaf)
        return leaves, leaf_list

    def _gradient_log_p_per_particle(
        self, particles: torch.Tensor, ctx: ForwardContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_p [N], ∇log_p stacked like ``particles``)."""
        log_ps: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        for i in range(self.n_particles):
            leaves, leaf_list = self._leaves_for_row(particles[i])
            state = ModelState(tensors=leaves, metadata=self._metadata)
            lp = _evaluate_log_posterior(self.target, state, ctx)
            gs = torch.autograd.grad(lp, leaf_list)
            if not self._multi:
                g = gs[0]
            else:
                g = torch.cat([gi.reshape(-1) for gi in gs])
            log_ps.append(lp.detach())
            grads.append(g.detach())
        return torch.stack(log_ps), torch.stack(grads)

    @staticmethod
    def _rbf_kernel_and_repulsive(
        x_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RBF kernel matrix + summed repulsive direction + cloud scale.

        x_flat: (N, D). Returns K (N, N), ``S`` (N, D) where
        ``S[i] = Σ_j ∇_{x_j} k(x_j, x_i)``, and ``med``, the median
        off-diagonal pairwise squared distance (the cloud's squared length
        scale, used by ``run`` to bound the repulsive displacement).
        """
        n = x_flat.shape[0]
        sq = (x_flat.unsqueeze(0) - x_flat.unsqueeze(1)).pow(2).sum(-1)  # (N, N)
        # Median heuristic over the OFF-DIAGONAL pairwise distances (i≠j), per
        # Liu & Wang (2016). The full matrix carries N exact zeros on the
        # diagonal: at the documented minimum N=2 the median of [0, d, d, 0] is
        # 0, so h floors to 1e-8, K→I, the repulsive term vanishes, and the
        # particles collapse onto one mode (degenerate zero-spread posterior).
        # Small even N is biased down the same way. clamp stays only as a guard
        # for genuinely coincident particles.
        iu = torch.triu_indices(n, n, offset=1, device=sq.device)
        med = torch.median(sq[iu[0], iu[1]])
        h_sq = (med / math.log(max(n, 2.0))).clamp(min=1e-8)
        K = torch.exp(-sq / (2.0 * h_sq))
        row_sum = K.sum(dim=1, keepdim=True)              # (N, 1)
        # Σ_j ∇_{x_j} k(x_j, x_i)  =  (1/h²) (rowsum(K) ⊙ x_i − K @ X)
        S = (row_sum * x_flat - K @ x_flat) / h_sq        # (N, D)
        return K, S, med

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
        """Continue the current SVGD particle ensemble for one call."""
        n_iters = validate_int(
            n_iters,
            owner="SVGD.run",
            field="n_iters",
            minimum=0,
        )
        run_ctx = self._default_ctx if ctx is None else ctx
        accounting = RunAccounting(
            requested_iters=n_iters,
            warmup_iters=0,
            continued_from_iteration=self._completed_sampling,
            continued_accepted_sampling=self._completed_sampling,
        )
        history: list[float] = []
        callback_phase: str | None = None
        callback_iteration: int | None = None

        def metadata() -> dict[str, Any]:
            reference = self.particles
            return {
                "sampler": "SVGD",
                "sample_layout": "particles",
                "generator_initial_seed": self._generator_initial_seed,
                "requested_warmup": 0,
                "completed_warmup": 0,
                "thin": 1,
                "stored_draws": self.n_particles,
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
                "n_particles": self.n_particles,
                "step_size": self.step_size,
                "n_completed": accounting.cumulative_completed_sampling,
            }

        def result(reason: SamplerStopReason) -> InferenceResult:
            return InferenceResult(
                samples={
                    name: tensor
                    for name, tensor in self.unpack_particles().items()
                },
                log_post_history=torch.as_tensor(history, dtype=torch.float64),
                acceptance_rate=(
                    1.0 if accounting.completed_sampling > 0 else 0.0
                ),
                requested_iters=n_iters,
                completed_iters=accounting.completed_sampling,
                stop_reason=reason,
                metadata=metadata(),
            )

        def partial_result() -> _PartialInferenceResult:
            return _PartialInferenceResult(
                samples={
                    name: tensor
                    for name, tensor in self.unpack_particles().items()
                },
                log_post_history=torch.as_tensor(history, dtype=torch.float64),
                acceptance_rate=(
                    1.0 if accounting.completed_sampling > 0 else 0.0
                ),
                requested_iters=n_iters,
                completed_iters=accounting.completed_sampling,
                metadata=metadata(),
            )

        stop_reason = SamplerStopReason.COMPLETED
        particle_count = self.n_particles
        dimension = self._dim_total
        execution_phase = (
            "initial" if self._completed_sampling == 0 else "kernel"
        )
        try:
            for iteration in range(n_iters):
                log_probabilities, gradients = (
                    self._gradient_log_p_per_particle(
                        self.particles,
                        run_ctx,
                    )
                )
                if accounting.completed_sampling == 0:
                    _check_initial_log_post(log_probabilities, "SVGD")
                execution_phase = "kernel"
                flat_particles = self.particles.view(
                    particle_count,
                    dimension,
                )
                flat_gradients = gradients.view(particle_count, dimension)
                kernel, repulsive, median_squared = (
                    self._rbf_kernel_and_repulsive(flat_particles)
                )
                repulsive_displacement = (
                    self.step_size / particle_count
                ) * repulsive.norm(dim=1, keepdim=True)
                cloud_scale = median_squared.sqrt()
                repulsive = torch.where(
                    repulsive_displacement > cloud_scale,
                    repulsive * (cloud_scale / repulsive_displacement),
                    repulsive,
                )
                direction = (
                    kernel @ flat_gradients + repulsive
                ) / particle_count
                proposed_particles = (
                    self.particles
                    + self.step_size * direction.view_as(self.particles)
                )
                mean_log_prob = float(log_probabilities.mean().item())
                if not (
                    math.isfinite(mean_log_prob)
                    and bool(torch.isfinite(proposed_particles).all())
                ):
                    partial = (
                        partial_result()
                        if accounting.completed_sampling > 0
                        else None
                    )
                    err = GeoBrainError(
                        f"SVGD ensemble diverged at iteration {iteration}: "
                        "the candidate particle cloud or mean log-posterior "
                        "became non-finite; current transition was not "
                        "committed and the last finite ensemble is attached "
                        "as .partial_result",
                        object_name="SVGD",
                        field="step_size",
                        actual=self.step_size,
                    )
                    if partial is not None:
                        err.partial_result = partial
                    raise err

                should_stop = False
                if callback is not None:
                    execution_phase = "callback"
                    should_stop = bool(
                        callback(
                            accounting.completed_sampling,
                            mean_log_prob,
                            callback_snapshot(
                                self.unpack_particles(proposed_particles)
                            ),
                        )
                    )
                    execution_phase = "kernel"

                self.particles = proposed_particles.detach().clone()
                accounting.commit_sampling(accepted=True)
                self._completed_sampling += 1
                history.append(mean_log_prob)
                if should_stop:
                    stop_reason = SamplerStopReason.CALLBACK
                    callback_phase = "sampling"
                    callback_iteration = accounting.completed_sampling - 1
                    break
            execution_phase = "result"
            return result(stop_reason)
        except _TargetEvaluationFailure as failure:
            cause = failure.cause
            if execution_phase == "initial" and isinstance(cause, GeoBrainError):
                raise cause
            partial = (
                partial_result() if accounting.completed_sampling > 0 else None
            )
            raise _partial_run_error(
                "SVGD",
                accounting.completed_sampling,
                n_iters,
                partial,
                cause,
                field="target",
            ) from cause
        except _CallbackTransformFailure as failure:
            cause = failure.cause
            partial = (
                partial_result() if accounting.completed_sampling > 0 else None
            )
            raise _partial_run_error(
                "SVGD",
                accounting.completed_sampling,
                n_iters,
                partial,
                cause,
                field="transform",
            ) from cause
        except Exception as exc:
            if execution_phase == "initial" and isinstance(exc, GeoBrainError):
                raise
            if getattr(exc, "partial_result", None) is not None:
                raise
            partial = (
                partial_result() if accounting.completed_sampling > 0 else None
            )
            field = (
                execution_phase
                if execution_phase in {"callback", "result"}
                else "kernel"
            )
            raise _partial_run_error(
                "SVGD",
                accounting.completed_sampling,
                n_iters,
                partial,
                exc,
                field=field,
            ) from exc
