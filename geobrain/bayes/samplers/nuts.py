"""No-U-Turn Sampler (NUTS) with optional dual averaging and mass adaptation.

Implements the **progressive multinomial** NUTS formulation (Betancourt 2017,
"A Conceptual Introduction to Hamiltonian Monte Carlo", §A.4, the algorithm
Stan runs) for the trajectory: tree doubling, U-turn termination, divergence
detection, and multinomial selection over leaves weighted by
``exp(log_joint - log_joint_init)``. Each subtree keeps one representative
proposal plus its accumulated log-sum-exp weight. Merging resamples that
representative, so memory per trajectory is ``O(max_depth)`` rather than the
``2**max_depth`` full clones held by a naive candidate list. The chosen draw
carries its log-posterior and gradient, avoiding an extra target evaluation.

Two optional adaptations are activated by ``warmup > 0``:

- **Dual averaging** tunes the step size toward ``target_accept``. Warmup uses
  the live Robbins-Monro step size; sampling uses its smoothed value.
- **Mass-matrix adaptation** supports ``"identity"``, ``"diagonal"``, and
  per-field ``"dense"`` covariance. Dense fields above ``max_dense_dim`` fall
  back to diagonal storage, avoiding accidental quadratic allocation.

With ``adapt_mass=True`` and ``warmup >= 20``, adaptation follows a scaled
Stan window schedule: an initial step-size-only buffer, expanding Welford mass
windows, and a terminal step-size-only buffer. Closing a mass window applies
the new metric immediately, clears the accumulator, searches for a reasonable
step size under that metric, and restarts dual averaging. Shorter warmups keep
the historical single-window scheme. Injected ``mass_matrix`` values require
``adapt_mass=False`` and bypass windowed mass adaptation.

``warmup=0`` therefore runs fixed-step NUTS; ``samples`` always contains only
post-warmup draws. Trajectory construction lives in :mod:`._nuts_tree`,
adaptation lives in :mod:`._nuts_warmup`, execution orchestration lives in
:mod:`._nuts_execution`, and call-local result presentation lives in
:mod:`._nuts_results`. This module owns the public facade and persistent chain
state.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from ...core import ForwardContext, GeoBrainError
from ...core.validation import validate_param_name
from ..base import LogPosteriorTarget, Sampler
from ..execution import validate_chain_storage
from ..results import InferenceResult
from ._nuts_execution import run_nuts
from ._nuts_tree import validate_tree_options
from ._nuts_warmup import (
    WarmupState,
    build_injected_mass_info,
    field_mass_kinds,
    validate_warmup_options,
)


@Sampler.register("nuts")
class NUTS(Sampler):
    """NUTS with tree doubling and optional step-size/mass adaptation.

    **Tier in the inversion architecture:** Tier 2 (class-based, explicit).

    For most users, prefer the Tier 1 factory::

        samples = problem.as_posterior().sample(
            "nuts",
            params={"sigma": s0},
            n_iters=500,
            warmup=200,
            target_accept=0.8,
        )

    Construct this class directly to inspect adapted state, warm-start a later
    run, attach callbacks, or call :meth:`NUTS.from_callable` against a custom
    log-posterior. ``target`` accepts an
    :class:`~geobrain.InverseProblem`, a
    :class:`~geobrain.bayes.Posterior`, or any object satisfying the
    ``log_posterior(state, ctx)`` protocol.

    Attributes:
        mass_info_: Detached copies of the per-field mass entries in effect at
            the end of the latest run. Diagonal entries contain ``"diag"``;
            dense entries contain ``"M_inv"``, ``"chol_M"``, and
            ``"field_shape"``. ``None`` denotes identity mass or no run.
            A diagonal may be passed back through
            ``mass_matrix={name: mass_info_[name]["diag"]}``.
        step_size_final_: Dual-averaged final step size from the latest run,
            equal to ``InferenceResult.metadata["step_size_final"]``.
            ``None`` before any run.
    """

    def __init__(
        self,
        target: LogPosteriorTarget | None = None,
        *,
        params: Mapping[str, torch.Tensor] | None = None,
        step_size: float = 0.01,
        max_depth: int = 8,
        delta_max: float = 1000.0,
        warmup: int = 0,
        target_accept: float = 0.8,
        adapt_mass: bool = True,
        mass_type: str = "diagonal",
        max_dense_dim: int = 200,
        mass_matrix: Mapping[str, torch.Tensor] | None = None,
        thin: int = 1,
        store_dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure NUTS over a target's ``log_posterior``.

        Args:
            target: An :class:`~geobrain.InverseProblem`,
                :class:`~geobrain.bayes.Posterior`, or compatible
                :class:`~geobrain.bayes.LogPosteriorTarget`.
            params: Initial parameter values, one tensor per field.
            step_size: Initial leapfrog step size ``epsilon`` (``> 0``).
                Windowed warmup treats this as the starting guess for its
                reasonable-step-size search, so its exact value matters less
                than in fixed-step sampling. It is adapted whenever
                ``warmup > 0``.
            max_depth: Maximum tree depth per trajectory (``> 0``).
            delta_max: Positive divergence threshold on energy error.
            warmup: Once-only adaptation iterations (``>= 0``). Warmup draws
                are discarded. With mass adaptation and at least 20 warmup
                steps, scaled Stan windows apply each new mass mid-warmup and
                restart step-size adaptation. Shorter warmups finalize mass
                once at the end.
            target_accept: Dual-averaging target probability in ``(0, 1)``.
            adapt_mass: Whether warmup estimates a mass matrix. Must be
                ``False`` when ``mass_matrix`` is supplied.
            mass_type: ``"identity"``, ``"diagonal"``, or ``"dense"``.
            max_dense_dim: Per-field flat-size cap above which dense
                adaptation falls back to diagonal (``>= 1``).
            mass_matrix: Optional precomputed mass ``M`` by field. A tensor
                with ``numel == field.numel()`` supplies the diagonal; a
                two-dimensional ``(n, n)`` tensor supplies a dense symmetric
                positive-definite mass. Missing fields use identity mass.
                Injection requires ``adapt_mass=False`` and is active from the
                first transition, including when ``warmup=0``. This accepts
                values recovered from :attr:`mass_info_` together with
                :attr:`step_size_final_`; the optimal ``M`` is the inverse
                posterior covariance.
            thin: Store every ``thin``-th post-warmup draw (``>= 1``), using
                zero-based post-warmup indices
                ``thin - 1, 2 * thin - 1, ...``. This is exactly
                ``dense[thin - 1::thin]`` from a same-seed unthinned run and
                stores ``floor(n_iters / thin)`` draws. Thinning is
                STORAGE-ONLY: transition counting, RNG use, adaptation,
                callback indices, and acceptance statistics are unchanged.
                ``log_post_history`` remains dense with one value per committed
                sampling transition; only ``samples`` is thinned.
            store_dtype: Optional dtype for stored parameter clones
                (``None`` keeps the parameter dtype). For example, float32
                storage halves chain memory for float64 parameters. Sampling
                trajectories and adaptation remain in the parameter dtype;
                ``log_post_history`` remains float64.
            generator: Optional device-correct random generator for exact
                seeded reproducibility.
            metadata: Optional ``ModelState``-style metadata stamped onto each
                target evaluation, for example the ``{"units": {...}}``
                mapping carried from an ``EarthModel.resolve()`` call that
                seeded ``params``. A problem or prior can therefore read units
                from every mid-chain ``ModelState``. This defaults to empty,
                is distinct from the sampler diagnostics in
                ``InferenceResult.metadata``, and never enters that namespace.

        Raises:
            GeoBrainError: On invalid ``step_size``, ``max_depth``, ``warmup``,
                ``target_accept``, ``mass_type``, ``max_dense_dim``, ``thin``,
                or ``store_dtype``; non-tensor parameters; or an invalid
                ``mass_matrix`` (unknown field, shape mismatch, non-finite or
                non-positive-definite values, or use with
                ``adapt_mass=True``).
        """
        if target is None:
            raise TypeError("NUTS() missing required argument: 'target'")
        if params is None:
            raise TypeError("NUTS() missing required argument: 'params'")
        if not params:
            raise GeoBrainError(
                "NUTS requires at least one parameter",
                object_name="NUTS",
                field="params",
                expected="non-empty mapping",
                actual={},
            )
        step_size, max_depth, delta_max = validate_tree_options(
            step_size, max_depth, delta_max,
        )
        (
            warmup, target_accept, adapt_mass, mass_type, max_dense_dim,
        ) = validate_warmup_options(
            warmup, target_accept, adapt_mass, mass_type, max_dense_dim,
        )
        thin, store_dtype = validate_chain_storage("NUTS", thin, store_dtype)
        for name, tensor in params.items():
            validate_param_name(name, owner="NUTS")
            if not isinstance(tensor, torch.Tensor):
                raise GeoBrainError(
                    "NUTS params values must be torch.Tensor",
                    object_name="NUTS", field=f"params[{name!r}]",
                    expected=torch.Tensor, actual=type(tensor),
                )

        self.target = target
        self.params = {
            name: tensor.detach().clone() for name, tensor in params.items()
        }
        self.step_size = step_size
        self.max_depth = max_depth
        self.delta_max = delta_max
        self.warmup = warmup
        self.target_accept = target_accept
        self.adapt_mass = adapt_mass
        self.mass_type = mass_type
        self.max_dense_dim = max_dense_dim
        self.thin = thin
        self.store_dtype = store_dtype
        self.generator = generator
        self._metadata: Mapping[str, Any] = dict(metadata) if metadata else {}
        self._field_mass_kind = field_mass_kinds(
            self.params, mass_type, max_dense_dim,
        )
        self._injected_mass_info = (
            build_injected_mass_info(
                mass_matrix, params=self.params, adapt_mass=adapt_mass,
            )
            if mass_matrix is not None
            else None
        )
        self.mass_info_: dict[str, dict[str, Any]] | None = None
        self.step_size_final_: float | None = None
        self._theta = {
            name: tensor.detach().clone() for name, tensor in self.params.items()
        }
        self._lp_cur: float | None = None
        self._grad_cur: dict[str, torch.Tensor] | None = None
        self._state_ctx: ForwardContext | None = None
        self._default_ctx = ForwardContext()
        self._completed_warmup = 0
        self._completed_sampling = 0
        self._accepted_sampling = 0
        self._divergent_sampling = 0
        self._divergent_total = 0
        self._accept_probability_sum = 0.0
        self._run_state: WarmupState | None = None
        self._generator_initial_seed = (
            generator.initial_seed() if generator is not None
            else torch.initial_seed()
        )

    def run(
        self,
        n_iters: int,
        ctx: ForwardContext | None = None,
        callback: Callable[
            [int, float, Mapping[str, torch.Tensor]], bool | None
        ] | None = None,
    ) -> InferenceResult:
        """Continue the owned chain without replaying committed adaptation."""
        return run_nuts(self, n_iters, ctx, callback)
