"""
Shared inverse problem definition.

Binds ``(forward, observed, likelihood, prior)`` into a thin object that every
solver, deterministic (:mod:`geobrain.optim`), Bayesian (:mod:`geobrain.bayes`),
or decision-rule; consumes identically. There is *no* ``BayesianInverseProblem``;
switching solvers is enough to move from MAP / least-squares to full posterior
sampling.

Statistical ingredients live in sibling modules so this file stays focused on the
inverse-problem definition itself:

- :mod:`geobrain.inverse.likelihood`: :class:`Likelihood` / :class:`GaussianLikelihood`
- :mod:`geobrain.inverse.prior`: :class:`Prior` / :class:`GaussianPrior`

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Protocol, runtime_checkable

import torch

from ..core.composition import OperatorBundle, OperatorChain
from ..core.errors import GeoBrainError
from ._units import check_units
from .evaluation import ObjectiveEvaluation
from .likelihood import Likelihood
from ..core.operator import ForwardOperator, Operator
from .prior import Prior
from ..core.containers import ModelState, ForwardOutput
from ..core.context import ForwardContext

if TYPE_CHECKING:
    from geobrain.bayes import Posterior
    from geobrain.bayes.transforms import InvertibleTransform
    from geobrain.optim import Inverter


@runtime_checkable
class InverseProblemLike(Protocol):
    """Structural type an :class:`~geobrain.optim.Inverter` (or any other
    deterministic-optimisation consumer) can drive without depending on the
    concrete :class:`InverseProblem` class, the Tier-1/2 optimisation-side
    counterpart of :class:`~geobrain.bayes.LogPosteriorTarget` (which plays
    the same role for the Bayesian samplers).

    Any object exposing ``loss`` / ``log_likelihood`` / ``log_prior`` methods
    and a ``prior`` attribute satisfies this structurally, no inheritance
    needed (``isinstance(obj, InverseProblemLike)`` works via
    ``@runtime_checkable``). :class:`InverseProblem` itself satisfies it; a
    hand-rolled adapter around a third-party misfit function can too.
    """

    prior: Optional[Prior]

    def loss(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        ...

    def log_likelihood(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        ...

    def log_prior(self, state: ModelState) -> torch.Tensor:
        ...


class InverseProblem:
    """
    Inverse-problem definition: given observations, infer model parameters.

    A pure-data binding of the forward operator, observed data, likelihood, and
    optional prior into one object that drives both deterministic inversion and
    Bayesian inference. Switching solvers, not subclassing; is what moves
    between a point estimate and a full posterior.

    Attributes:
        forward: An :class:`Operator` producing a :class:`ForwardOutput`. May be a
            single :class:`ForwardOperator`, an :class:`OperatorChain` ending in
            one, or a :class:`OperatorBundle` of several.
        observed: Observed-data mapping (channel name → tensor). Keys must overlap
            with the forward operator's ``output_keys``.
        likelihood: Any object satisfying the :class:`Likelihood` protocol.
        prior: Optional. When present, used by Bayesian solvers via
            :meth:`log_posterior`. Absent → flat (improper) prior; samplers still
            run but the posterior reduces to the likelihood.
        metadata: Free-form, frozen provenance mapping that travels *with* the
            problem (coordinate frame, units, dataset id, noise-floor assumptions,
            …). Not used by solvers and **not** part of problem identity (excluded
            from ``__eq__`` / ``__hash__``), mirroring ``ModelState.metadata`` /
            ``ForwardOutput.metadata``. Reserved as the round-trip slot a future
            checkpoint/serialization layer can use without breaking this frozen
            surface.
    """

    forward: Operator
    observed: Mapping[str, torch.Tensor]
    likelihood: Likelihood
    prior: Optional[Prior]
    metadata: Mapping[str, Any]
    _observed_units: Mapping[str, str]

    def __init__(
        self,
        forward: Operator,
        observed: Mapping[str, torch.Tensor] | None = None,
        likelihood: Likelihood | None = None,
        *,
        prior: Optional[Prior] = None,
        metadata: Mapping[str, Any] | None = None,
        observed_units: Mapping[str, str] | None = None,
    ) -> None:
        """
        Bind the problem components and validate their consistency.

        Args:
            forward: Observation-producing operator (or chain / joint).
            observed: Observed-data mapping (channel name → tensor).
            likelihood: Data-misfit model satisfying the :class:`Likelihood`
                protocol.
            prior: Optional prior over the model parameters.

        Raises:
            TypeError: If ``observed`` or ``likelihood`` is missing.
            GeoBrainError: If the forward operator or observed channels are
                invalid.
        """
        if observed is None:
            raise TypeError(
                "InverseProblem() missing required argument: 'observed'"
            )
        if likelihood is None:
            raise TypeError(
                "InverseProblem() missing required argument: 'likelihood'"
            )

        # Immutable by design: a problem is a pure-data binding (mirrors the
        # ModelState / ForwardOutput frozen pattern in core.containers). Assign via
        # object.__setattr__ to bypass the guarding __setattr__ below; after this
        # __init__ the object is read-only.
        observed_frozen = MappingProxyType(dict(observed))
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "observed", observed_frozen)
        object.__setattr__(self, "likelihood", likelihood)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(metadata or {})),
        )
        # Optional per-channel observed units for the opt-in unit contract.
        # Empty mapping when unset → the check in ``predict`` is a no-op
        # (validate-when-present, fully backward compatible).
        object.__setattr__(
            self, "_observed_units", MappingProxyType(dict(observed_units or {})),
        )

        self._validate()

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"InverseProblem is immutable; cannot set attribute {name!r}. "
            "Construct a new InverseProblem instead."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"InverseProblem is immutable; cannot delete attribute {name!r}."
        )

    def __repr__(self) -> str:
        return (
            f"InverseProblem(forward={self.forward!r}, "
            f"observed={dict(self.observed)!r}, likelihood={self.likelihood!r}, "
            f"prior={self.prior!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InverseProblem):
            return NotImplemented
        return (
            self.forward is other.forward
            and self.observed is other.observed
            and self.likelihood is other.likelihood
            and self.prior is other.prior
        )

    def __hash__(self) -> int:
        # Match the previous frozen-dataclass identity-style hashing semantics
        # (mappings/tensors aren't hashable; we hash by object identity).
        return hash((id(self.forward), id(self.observed), id(self.likelihood), id(self.prior)))

    def _validate(self) -> None:
        """Validate forward-operator type and observed-channel consistency."""
        # forward must produce a ForwardOutput
        if isinstance(self.forward, OperatorChain):
            if not self.forward.is_observation:
                raise GeoBrainError(
                    "forward must end in a ForwardOperator",
                    object_name="InverseProblem",
                    field="forward",
                    expected="OperatorChain terminating in ForwardOperator",
                    actual="OperatorChain (all transforms)",
                )
        elif not isinstance(self.forward, (ForwardOperator, OperatorBundle)):
            raise GeoBrainError(
                "forward must be a ForwardOperator, OperatorChain "
                "ending in one, or OperatorBundle",
                object_name="InverseProblem",
                field="forward",
                expected="observation-producing Operator",
                actual=type(self.forward).__name__,
            )

        if not self.observed:
            raise GeoBrainError(
                "InverseProblem requires at least one observed-data channel",
                object_name="InverseProblem",
                field="observed",
                expected="non-empty mapping",
                actual={},
            )
        for name, tensor in self.observed.items():
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "observed channel name must be a non-empty string",
                    object_name="InverseProblem",
                    field="observed",
                    actual=name,
                )
            if not isinstance(tensor, torch.Tensor):
                raise GeoBrainError(
                    "observed values must be torch.Tensor",
                    object_name="InverseProblem",
                    field=f"observed[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(tensor),
                )

        # Observed keys must be reachable from the forward operator's declared
        # channels. Contract: if forward declares a non-empty output_keys, every
        # observed channel MUST be in that set: construction fails fast (this is
        # a hard raise, not a warning). When output_keys is empty this pre-check
        # is skipped and the observed-vs-produced match is deferred to
        # Likelihood.misfit at call time. Note this deferral does NOT permit an
        # operator to *emit* undeclared channels: ForwardOperator.forward
        # separately enforces emitted data keys ⊆ output_keys, so an empty
        # output_keys means the forward produces no declared data channels (only
        # a composition (chain/bundle) carries the observable keys upward).
        declared = set(self.forward.differentiability.output_keys)
        if declared:
            unknown = set(self.observed) - declared
            if unknown:
                raise GeoBrainError(
                    "observed channel not produced by forward operator",
                    object_name="InverseProblem",
                    field="observed",
                    expected=f"subset of forward output_keys {sorted(declared)}",
                    actual=sorted(unknown),
                )

    # -----------------------------------------------------------------
    # Forward evaluation
    # -----------------------------------------------------------------

    def predict(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> ForwardOutput:
        """
        Run the forward operator (no misfit).

        Args:
            state: Model parameters.
            ctx: Per-call runtime configuration. Defaults to an empty
                :class:`ForwardContext`.

        Returns:
            The :class:`ForwardOutput` produced by the forward operator.
        """
        if ctx is None:
            ctx = ForwardContext()
        pred: ForwardOutput = self.forward(state, ctx)
        # Opt-in unit contract: when the caller declared ``observed_units`` and the
        # forward output declares per-channel ``metadata["units"]``, verify the
        # shared channels agree (a no-op if either side omits a channel's unit).
        if self._observed_units:
            check_units(
                pred.metadata.get("units", {}),
                self._observed_units,
                object_name="InverseProblem",
            )
        return pred

    def evaluate(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> tuple[ForwardOutput, torch.Tensor]:
        """
        Run the forward model and score it against the observations.

        Args:
            state: Model parameters to evaluate.
            ctx: Per-call runtime configuration (wavelet, survey, mesh). Defaults
                to an empty :class:`ForwardContext`.

        Returns:
            Tuple ``(prediction, misfit)``, the :class:`ForwardOutput` and the scalar
            data-misfit ``likelihood.misfit(prediction, observed)``.
        """
        result = self.objective(state, ctx)
        return result.prediction, result.data_loss

    def objective(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> ObjectiveEvaluation:
        """Evaluate every inverse-objective term from one forward prediction.

        Args:
            state: Model parameters to evaluate.
            ctx: Optional per-call runtime configuration.

        Returns:
            An immutable :class:`ObjectiveEvaluation` containing the prediction,
            scalar data loss, log-likelihood, log-prior, empty per-term mapping,
            and derived total loss.
        """
        prediction = self.predict(state, ctx)
        data_loss = self.likelihood.misfit(
            prediction.data,
            self.observed,
        )
        log_likelihood = self.likelihood.log_likelihood(
            prediction.data,
            self.observed,
        )
        log_prior = self.log_prior(state)
        return ObjectiveEvaluation(
            prediction=prediction,
            data_loss=data_loss,
            log_likelihood=log_likelihood,
            log_prior=log_prior,
        )

    def loss(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        """
        Evaluate only the scalar data-misfit loss from one prediction.

        Args:
            state: Model parameters.
            ctx: Per-call runtime configuration.

        Returns:
            The scalar misfit. This data-only route does not evaluate the
            log-likelihood or prior; consumers needing the complete
            probabilistic record use :meth:`objective`.
        """
        prediction = self.predict(state, ctx)
        return self.likelihood.misfit(
            prediction.data,
            self.observed,
        )

    # -----------------------------------------------------------------
    # Probabilistic interface
    # -----------------------------------------------------------------

    def log_likelihood(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        """
        Log-likelihood ``log p(d | θ)``. Equals ``-misfit`` up to a constant.

        Args:
            state: Model parameters.
            ctx: Per-call runtime configuration.

        Returns:
            Scalar log-likelihood, for samplers.
        """
        return self.objective(state, ctx).log_likelihood

    def log_prior(self, state: ModelState) -> torch.Tensor:
        """
        Log-prior ``log p(θ)``.

        Args:
            state: Model parameters.

        Returns:
            Scalar ``prior.log_prob(state)``, or ``0`` when ``prior is None``.
        """
        if self.prior is None:
            # Flat prior → 0. Match GaussianPrior.log_prob's empty-state guard:
            # an empty ModelState has no anchor tensor for dtype/device, so fall
            # back to a bare zero scalar rather than raising StopIteration.
            anchor = next(iter(state.tensors.values()), None)
            if anchor is None:
                return torch.zeros(())
            return torch.zeros((), dtype=anchor.dtype, device=anchor.device)
        return self.prior.log_prob(state)

    def log_posterior(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> torch.Tensor:
        """
        Unnormalised log-posterior ``log p(θ | d) = log_likelihood + log_prior``.

        Args:
            state: Model parameters.
            ctx: Per-call runtime configuration.

        Returns:
            Scalar log-posterior, up to an additive constant.
        """
        result = self.objective(state, ctx)
        return result.log_likelihood + result.log_prior

    # -----------------------------------------------------------------
    # View factories
    # -----------------------------------------------------------------
    #
    # InverseProblem is the single source of truth. The two helpers below build
    # thin *views* over it: never copies. ``create_inverter`` opens the
    # deterministic optimisation view (Adam / L-BFGS); ``as_posterior`` opens the
    # probabilistic view (HMC / NUTS / LangevinDynamics / SVGD), optionally
    # swapping the prior.
    # Both helpers lazy-import their target modules so this file stays cycle-free.

    def create_inverter(self, **kwargs: Any) -> "Inverter":
        """
        Build an :class:`~geobrain.optim.Inverter` for this problem.

        Tier 1 (canonical user API for deterministic inversion); sugar for
        ``Inverter(self, **kwargs)``. When ``self.prior`` is set, the default
        ``mode="MAP"`` folds ``-log_prior(state)`` into the loss so the point
        estimate is the MAP point; pass ``mode="MLE"`` to run on the pure data
        misfit instead (e.g. when handling the prior via a hand-rolled
        ``regularizer``). Lazy-imports :mod:`geobrain.optim` to avoid an
        ``inverse_problem`` ↔ ``optim`` import cycle.

        Args:
            **kwargs: Forwarded verbatim to :meth:`Inverter.__init__`, typically
                ``params``, an explicit ``optimizer=AdamConfig(...)`` or
                ``optimizer=LBFGSConfig(...)``, ``regularizer``, ``bounds``,
                ``ctx``, ``mode``, ``gradient_processors``, and
                ``step_projections``.

        Returns:
            A configured :class:`~geobrain.optim.Inverter`.

        Example::

            from geobrain.optim import AdamConfig

            result = problem.create_inverter(
                params={"sigma": sigma_init},
                optimizer=AdamConfig(lr=0.01),
                bounds={"sigma": (1e-4, 1.0)},
            ).run(n_iters=100)
        """
        from geobrain.optim import Inverter

        return Inverter(self, **kwargs)

    def as_posterior(
        self,
        prior: "Prior | None" = None,
        transforms: "Mapping[str, InvertibleTransform] | None" = None,
    ) -> "Posterior":
        """
        Open the probabilistic view of this problem.

        Tier 1 (canonical user API for Bayesian inversion). Returns a thin
        :class:`~geobrain.bayes.Posterior` that combines :meth:`log_likelihood`
        with an optional prior. When ``prior`` is None the view inherits
        ``self.prior`` (itself possibly None, an improper flat prior, so the
        posterior reduces to the likelihood). The view holds no copies of the
        data, likelihood, or forward operator, only a reference to ``self`` and
        the optional prior override.

        Args:
            prior: Optional prior override. Defaults to ``self.prior``.
            transforms: Optional per-parameter map of
                :class:`~geobrain.bayes.transforms.InvertibleTransform` change-of-variables.
                The sampler explores each parameter in an *unconstrained* space
                ``u`` and the transform maps it back to the constrained model
                space ``x`` (with the log-abs-det-Jacobian folded into the
                density), so the chain never proposes out-of-domain values. A
                parameter with no entry is sampled directly (identity). The
                canonical use is a **positivity constraint** on a strictly
                positive property such as velocity or conductivity, e.g.
                ``transforms={"vp": PositiveTransform()}`` lets the sampler roam over
                ``u = log(vp)`` while the forward always sees ``vp = exp(u) > 0``.

        Returns:
            A :class:`~geobrain.bayes.Posterior` view compatible with HMC / NUTS /
            LangevinDynamics / SVGD.

        Example::

            # Inherit the problem's prior:
            posterior = problem.as_posterior()
            samples = posterior.sample(
                "hmc", params={"sigma": s0},
                n_iters=500, step_size=0.01, n_leapfrog=20,
            )

            # Explicit prior override:
            posterior = problem.as_posterior(prior=GaussianPrior(...))

            # Positivity constraint via a log-transform on velocity:
            from geobrain.bayes.transforms import PositiveTransform
            posterior = problem.as_posterior(
                transforms={"vp": PositiveTransform()}
            )
        """
        from geobrain.bayes import Posterior

        return Posterior(self, prior=prior, transforms=transforms)
