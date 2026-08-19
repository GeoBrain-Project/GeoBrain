"""Orchestrate typed multi-physics objectives over one shared Earth model.

Binding validation and per-term input construction live in
:mod:`geobrain.inverse.joint_binding`; immutable weighting policies live in
:mod:`geobrain.inverse.joint_weights`. This module owns only the joint
objective record, shared-model resolution, term execution, gradient
accumulation, diagnostics, and the current calibrated weight run state.

The deterministic data loss is the weighted sum of raw term misfits. The
joint log-likelihood remains unweighted so Bayesian consumers see the true
likelihood rather than an optimization-balancing artifact. Gradient-norm
weights calibrate lazily once and remain frozen until :meth:`JointProblem.rebalance`
is explicitly called.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from ..core.containers import ForwardOutput, ModelState
from ..core.context import ForwardContext
from ..core.errors import GeoBrainError
from ..core.operator import Operator
from . import joint_binding as _binding
from . import joint_weights as _weights
from .likelihood import Likelihood

__all__ = ["JointObjectiveEvaluation", "JointProblem"]


@dataclass(frozen=True)
class JointObjectiveEvaluation:
    """Immutable result of one complete joint-objective evaluation.

    Args:
        predictions: Full forward output for each named physics term.
        data_loss: Scalar weighted sum of the per-term data losses.
        log_likelihood: Scalar unweighted joint log-likelihood.
        log_prior: Scalar joint log-prior, currently zero.
        term_losses: Raw, unweighted scalar data loss for every term.

    Raises:
        GeoBrainError: If a joint or per-term objective value is not a scalar
            :class:`torch.Tensor`.
    """

    predictions: Mapping[str, ForwardOutput]
    data_loss: torch.Tensor
    log_likelihood: torch.Tensor
    log_prior: torch.Tensor
    term_losses: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        """Freeze mappings and validate every objective tensor is scalar."""
        object.__setattr__(self, "predictions", MappingProxyType(dict(self.predictions)))
        object.__setattr__(self, "term_losses", MappingProxyType(dict(self.term_losses)))
        objective_terms = (
            ("data_loss", self.data_loss),
            ("log_likelihood", self.log_likelihood),
            ("log_prior", self.log_prior),
        )
        for name, value in objective_terms:
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise GeoBrainError(
                    "joint objective terms must be scalar tensors",
                    object_name="JointObjectiveEvaluation",
                    field=name,
                    expected="0-d torch.Tensor",
                    actual=type(value).__name__,
                )
        for name, value in self.term_losses.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise GeoBrainError(
                    "joint per-term losses must be scalar tensors",
                    object_name="JointObjectiveEvaluation",
                    field=f"term_losses[{name!r}]",
                    expected="0-d torch.Tensor",
                    actual=type(value).__name__,
                )

    @property
    def total_loss(self) -> torch.Tensor:
        """Return the scalar MAP objective ``data_loss - log_prior``."""
        return self.data_loss - self.log_prior


class JointProblem:
    """Coordinate a named set of forward terms over one shared Earth model.

    Args:
        model: EarthModel-shaped object exposing functional ``with_values``,
            ``resolve``, and ``trainables`` methods.
        forwards: Non-empty mapping of term names to bare operators or explicit
            :class:`~geobrain.inverse.joint_binding.JointForward` bindings.
        observed: One observed tensor for each term name.
        likelihoods: One structural likelihood for each term name.
        weights: ``None`` for unit weights, an exact fixed-weight mapping, or
            ``"grad_norm"`` for lazy gradient-norm calibration.
        ctx: Optional default forward context.

    Raises:
        GeoBrainError: If construction data, binding contracts, weights, or
            shared-model trainable reachability are invalid.
    """

    __slots__ = (
        "model",
        "forwards",
        "observed",
        "likelihoods",
        "likelihood",
        "prior",
        "_ctx",
        "_term_names",
        "_compiled",
        "_union_needed",
        "_diag",
        "_weight_policy",
        "_weights",
    )

    model: _binding._EarthModelLike
    forwards: Mapping[str, Operator | _binding.JointForward]
    observed: Mapping[str, torch.Tensor]
    likelihoods: Mapping[str, Likelihood]
    likelihood: None
    prior: None
    _ctx: ForwardContext | None
    _term_names: tuple[str, ...]
    _compiled: Mapping[str, _binding.CompiledJointForward]
    _union_needed: tuple[str, ...]
    _diag: dict[str, dict[str, float]]
    _weight_policy: _weights.JointWeightPolicy
    _weights: Mapping[str, float] | None

    def __init__(
        self,
        model: _binding._EarthModelLike,
        forwards: Mapping[str, Operator | _binding.JointForward],
        observed: Mapping[str, torch.Tensor],
        likelihoods: Mapping[str, Likelihood],
        weights: None | Mapping[str, float] | str = None,
        ctx: ForwardContext | None = None,
    ) -> None:
        for attribute in ("with_values", "resolve", "trainables"):
            if not callable(getattr(model, attribute, None)):
                raise GeoBrainError(
                    "JointProblem model= must be an EarthModel-shaped object "
                    "exposing with_values()/resolve()/trainables()",
                    object_name="JointProblem",
                    field="model",
                    expected="EarthModel-like",
                    actual=type(model),
                )
        if not isinstance(forwards, Mapping) or not forwards:
            raise GeoBrainError(
                "JointProblem forwards= must be a non-empty mapping",
                object_name="JointProblem",
                field="forwards",
                expected=("non-empty Mapping[str, Operator | JointForward]"),
                actual=forwards,
            )

        forward_values = dict(forwards)
        observed_values = dict(observed)
        likelihood_values = dict(likelihoods)
        for name in forward_values:
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "JointProblem term names must be non-empty strings",
                    object_name="JointProblem",
                    field="forwards",
                    expected="non-empty string keys",
                    actual=name,
                )

        forward_names = set(forward_values)
        observed_names = set(observed_values)
        likelihood_names = set(likelihood_values)
        if not (forward_names == observed_names == likelihood_names):
            raise GeoBrainError(
                "JointProblem forwards/observed/likelihoods key sets must match exactly",
                object_name="JointProblem",
                field="forwards/observed/likelihoods",
                expected=sorted(forward_names),
                actual={
                    "forwards": sorted(forward_names),
                    "observed": sorted(observed_names),
                    "likelihoods": sorted(likelihood_names),
                },
            )
        for name, tensor in observed_values.items():
            if not isinstance(tensor, torch.Tensor):
                raise GeoBrainError(
                    "JointProblem observed values must be torch.Tensor",
                    object_name="JointProblem",
                    field=f"observed[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(tensor),
                )
        for name, likelihood in likelihood_values.items():
            if not isinstance(likelihood, Likelihood):
                raise GeoBrainError(
                    "JointProblem likelihoods values must satisfy the "
                    "Likelihood protocol (misfit/log_likelihood)",
                    object_name="JointProblem",
                    field=f"likelihoods[{name!r}]",
                    expected=Likelihood,
                    actual=type(likelihood),
                )

        compiled = {
            name: _binding.compile_joint_forward(name, binding)
            for name, binding in forward_values.items()
        }
        term_names = tuple(forward_values)
        union_needed = tuple(
            sorted({field for item in compiled.values() for field in item.model_fields})
        )
        policy, current_weights = _weights._build_weight_state(term_names, weights)

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "forwards", MappingProxyType(forward_values))
        object.__setattr__(self, "observed", MappingProxyType(observed_values))
        object.__setattr__(self, "likelihoods", MappingProxyType(likelihood_values))
        object.__setattr__(self, "likelihood", None)
        object.__setattr__(self, "prior", None)
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_term_names", term_names)
        object.__setattr__(self, "_compiled", MappingProxyType(compiled))
        object.__setattr__(self, "_union_needed", union_needed)
        object.__setattr__(self, "_diag", {"term_losses": {}})
        object.__setattr__(self, "_weight_policy", policy)
        object.__setattr__(self, "_weights", current_weights)
        _binding._validate_model_reachability(model, union_needed)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"JointProblem is immutable; cannot set attribute {name!r}. "
            "Construct a new JointProblem instead."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"JointProblem is immutable; cannot delete attribute {name!r}.")

    def __repr__(self) -> str:
        """Return the sorted named-term summary."""
        return f"JointProblem(terms={sorted(self._term_names)})"

    def _base_context(
        self,
        context: ForwardContext | None,
    ) -> ForwardContext:
        if context is not None:
            return context
        if self._ctx is not None:
            return self._ctx
        return ForwardContext()

    def _term_misfit(
        self,
        name: str,
        state: ModelState,
        base_context: ForwardContext,
    ) -> torch.Tensor:
        compiled = self._compiled[name]
        resolved = self.model.with_values(**dict(state.tensors)).resolve(*compiled.model_fields)
        prediction = compiled.run(resolved.tensors, base_context)
        predicted = {name: prediction.data[compiled.output_name]}
        return self.likelihoods[name].misfit(
            predicted,
            {name: self.observed[name]},
        )

    def _predictions(
        self,
        state: ModelState,
        base_context: ForwardContext,
    ) -> dict[str, ForwardOutput]:
        resolved = self.model.with_values(**dict(state.tensors)).resolve(*self._union_needed)
        return {
            name: self._compiled[name].run(resolved.tensors, base_context)
            for name in self._term_names
        }

    def _misfits(
        self,
        predictions: Mapping[str, ForwardOutput],
    ) -> dict[str, torch.Tensor]:
        return {
            name: self.likelihoods[name].misfit(
                {name: predictions[name].data[self._compiled[name].output_name]},
                {name: self.observed[name]},
            )
            for name in self._term_names
        }

    def _joint_log_likelihood(
        self,
        predictions: Mapping[str, ForwardOutput],
    ) -> torch.Tensor:
        total: torch.Tensor | None = None
        for name in self._term_names:
            value = self.likelihoods[name].log_likelihood(
                {name: predictions[name].data[self._compiled[name].output_name]},
                {name: self.observed[name]},
            )
            total = value if total is None else total + value
        assert total is not None
        return total

    def _calibrate_weights(
        self,
        state: ModelState,
        base_context: ForwardContext,
    ) -> Mapping[str, float]:
        norms = {
            name: _weights._gradient_norm(
                self._term_misfit(name, state, base_context),
                state.tensors.values(),
            )
            for name in self._term_names
        }
        return self._weight_policy.calibrate(norms)

    def _current_weights(
        self,
        state: ModelState,
        base_context: ForwardContext,
        term_losses: Mapping[str, torch.Tensor] | None = None,
    ) -> Mapping[str, float]:
        if self._weights is None:
            calibrated = (
                self._calibrate_weights(state, base_context)
                if term_losses is None
                else self._weight_policy.calibrate(
                    _weights._gradient_norms_from_losses(
                        term_losses,
                        state.tensors.values(),
                    )
                )
            )
            object.__setattr__(self, "_weights", calibrated)
        assert self._weights is not None
        return self._weights

    def objective(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> JointObjectiveEvaluation:
        """Evaluate all joint terms from one shared-model resolution.

        Args:
            state: Live model trainable tensors.
            ctx: Optional per-call context overriding the construction default.

        Returns:
            Immutable predictions, weighted loss, unweighted log-likelihood,
            zero log-prior, and raw per-term losses.
        """
        base_context = self._base_context(ctx)
        predictions = self._predictions(state, base_context)
        term_losses = self._misfits(predictions)
        weights = self._current_weights(state, base_context, term_losses)
        self._diag["term_losses"] = {
            name: float(value.detach()) for name, value in term_losses.items()
        }
        return JointObjectiveEvaluation(
            predictions=predictions,
            data_loss=_weights._weighted_loss(term_losses, weights),
            log_likelihood=self._joint_log_likelihood(predictions),
            log_prior=self.log_prior(state),
            term_losses=term_losses,
        )

    def loss(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor:
        """Return the weighted joint data loss."""
        base_context = self._base_context(ctx)
        term_losses = self._misfits(self._predictions(state, base_context))
        weights = self._current_weights(state, base_context, term_losses)
        self._diag["term_losses"] = {
            name: float(value.detach()) for name, value in term_losses.items()
        }
        return _weights._weighted_loss(term_losses, weights)

    def log_likelihood(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor:
        """Return the unweighted joint log-likelihood."""
        return self._joint_log_likelihood(
            self._predictions(state, self._base_context(ctx)),
        )

    def log_prior(self, state: ModelState) -> torch.Tensor:
        """Return a scalar zero because joint priors are not configured."""
        anchor = next(iter(state.tensors.values()), None)
        if anchor is None:
            return torch.zeros(())
        return torch.zeros(
            (),
            dtype=anchor.dtype,
            device=anchor.device,
        )

    def accumulate_gradients(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> float:
        """Accumulate weighted term gradients without retaining term graphs.

        Each term resolves, runs, and backpropagates independently. This
        preserves the bounded graph lifetime and the numerical equivalence to
        ``loss(state, ctx).backward()`` within the established tolerance.
        """
        base_context = self._base_context(ctx)
        weights = self._current_weights(state, base_context)
        term_losses: dict[str, float] = {}
        total = 0.0
        for name in self._term_names:
            misfit = self._term_misfit(name, state, base_context)
            term_losses[name] = float(misfit.detach())
            weighted = weights[name] * misfit
            weighted.backward()
            total += float(weighted.detach())
        self._diag["term_losses"] = term_losses
        return total

    def rebalance(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> Mapping[str, float]:
        """Explicitly replace current gradient-norm weights at ``state``.

        Args:
            state: Live state with at least one gradient-requiring leaf.
            ctx: Optional per-call context overriding the construction default.

        Returns:
            The newly calibrated immutable weight mapping.

        Raises:
            GeoBrainError: If the problem uses fixed weights or the supplied
                state has no gradient-requiring leaf.
        """
        if not isinstance(
            self._weight_policy,
            _weights.GradientNormWeights,
        ):
            raise GeoBrainError(
                "rebalance() only applies to weights='grad_norm' problems; "
                "this problem was constructed with fixed weights",
                object_name="JointProblem",
                field="weights",
                expected="weights='grad_norm' at construction",
                actual="fixed weights",
            )
        if not any(tensor.requires_grad for tensor in state.tensors.values()):
            raise GeoBrainError(
                "rebalance() needs a state whose leaves require grad (pass "
                "the LIVE optimisation tensors, not .detach()ed copies)",
                object_name="JointProblem",
                field="state",
                expected="at least one requires_grad leaf",
                actual="all leaves detached",
            )
        refreshed = self._calibrate_weights(
            state,
            self._base_context(ctx),
        )
        object.__setattr__(self, "_weights", refreshed)
        return refreshed

    @property
    def term_losses(self) -> dict[str, float]:
        """Return raw per-term losses from the most recent loss evaluation."""
        return dict(self._diag["term_losses"])

    @property
    def weights(self) -> Mapping[str, float] | None:
        """Return current immutable weights, or ``None`` before calibration."""
        return self._weights
