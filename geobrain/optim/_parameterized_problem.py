"""Private physical-state adapter for flat optimizer parameterizations.

The adapter keeps optimization state flat while presenting a physical
``ModelState`` to an inverse problem. Each delegated method parameterizes its
input exactly once. The MAP objective fallback also reuses that one physical
state for both the legacy data-loss and prior calls.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast

import torch

from geobrain.core import (
    FieldShapeError,
    ForwardContext,
    GeoBrainError,
    ModelState,
)
from geobrain.core.validation import validate_mapping_key
from geobrain.inverse import InverseProblemLike, ObjectiveEvaluationLike

from .parameterization import Parameterization
from ._result_assembly import _ResultTransform


@dataclass(frozen=True)
class _FallbackObjective:
    """Complete-enough objective record for a legacy loss/prior protocol."""

    data_loss: torch.Tensor
    log_likelihood: torch.Tensor
    log_prior: torch.Tensor
    term_losses: Mapping[str, torch.Tensor]


def _require_method(
    problem: object,
    name: str,
) -> Callable[..., Any]:
    method = getattr(problem, name, None)
    if not callable(method):
        raise GeoBrainError(
            "_ParameterizedProblem physical problem is missing a method",
            object_name="_ParameterizedProblem",
            field="problem",
            expected=f"callable {name}(physical_state, ...)",
            actual={"type": type(problem), "missing": name},
        )
    return cast(Callable[..., Any], method)


def _legacy_term_losses(
    raw_terms: object,
    *,
    reference: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Normalize optional legacy diagnostics without another forward call."""
    if raw_terms is None:
        return MappingProxyType({})
    if not isinstance(raw_terms, MappingABC):
        raise GeoBrainError(
            "_ParameterizedProblem term_losses must be a mapping",
            object_name="_ParameterizedProblem",
            field="term_losses",
            expected="mapping of names to scalar tensors or real numbers",
            actual=type(raw_terms),
        )
    normalized: dict[str, torch.Tensor] = {}
    for raw_name, raw_value in raw_terms.items():
        validate_mapping_key(
            "_ParameterizedProblem",
            "term_losses",
            raw_name,
        )
        name = cast(str, raw_name)
        if isinstance(raw_value, torch.Tensor):
            normalized[name] = raw_value
        elif isinstance(raw_value, Real) and not isinstance(raw_value, bool):
            normalized[name] = reference.new_tensor(float(raw_value))
        else:
            raise GeoBrainError(
                "_ParameterizedProblem term losses must be scalar",
                object_name="_ParameterizedProblem",
                field=f"term_losses[{name!r}]",
                expected="scalar torch.Tensor or real number",
                actual=type(raw_value),
            )
    return MappingProxyType(normalized)


class _ParameterizedProblem:
    """Adapt a physical inverse problem to one named latent vector."""

    def __init__(
        self,
        problem: object,
        parameterization: Parameterization,
        *,
        vector_name: str = "theta",
    ) -> None:
        if not isinstance(parameterization, Parameterization):
            raise GeoBrainError(
                "_ParameterizedProblem requires a Parameterization",
                object_name="_ParameterizedProblem",
                field="parameterization",
                expected=Parameterization,
                actual=type(parameterization),
            )
        if not isinstance(vector_name, str) or not vector_name:
            raise GeoBrainError(
                "_ParameterizedProblem vector_name must be a non-empty string",
                object_name="Inverter.from_parameterization",
                field="vector_name",
                expected="non-empty str",
                actual=vector_name,
            )
        self._problem = problem
        self._parameterization = parameterization
        self._vector_name = vector_name
        self.prior: Any | None = getattr(problem, "prior", None)

    @property
    def likelihood(self) -> Any | None:
        """Expose an optional likelihood for Inverter gradient policy."""
        return getattr(self._problem, "likelihood", None)

    @property
    def likelihoods(self) -> Mapping[str, Any]:
        """Expose optional joint likelihoods without assuming their presence."""
        likelihoods = getattr(self._problem, "likelihoods", None)
        if isinstance(likelihoods, MappingABC):
            return cast(Mapping[str, Any], likelihoods)
        return {}

    @property
    def term_losses(self) -> object:
        """Delegate the current named diagnostics without reevaluation."""
        return getattr(self._problem, "term_losses", None)

    def _physical_state(self, state: ModelState) -> ModelState:
        (theta,) = state.fetch(self._vector_name)
        return cast(ModelState, self._parameterization(theta))

    def objective(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> ObjectiveEvaluationLike:
        """Evaluate a physical objective from one parameterized state."""
        physical = self._physical_state(state)
        objective = getattr(self._problem, "objective", None)
        if callable(objective):
            return cast(
                ObjectiveEvaluationLike,
                cast(Callable[..., Any], objective)(physical, ctx),
            )

        data_loss = cast(
            torch.Tensor,
            _require_method(self._problem, "loss")(physical, ctx),
        )
        log_prior = cast(
            torch.Tensor,
            _require_method(self._problem, "log_prior")(physical),
        )
        return _FallbackObjective(
            data_loss=data_loss,
            log_likelihood=-data_loss,
            log_prior=log_prior,
            term_losses=_legacy_term_losses(
                self.term_losses,
                reference=data_loss,
            ),
        )

    def loss(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor:
        """Delegate the data-only loss on one physical state."""
        physical = self._physical_state(state)
        return cast(
            torch.Tensor,
            _require_method(self._problem, "loss")(physical, ctx),
        )

    def log_likelihood(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor:
        """Delegate log-likelihood on one physical state."""
        physical = self._physical_state(state)
        return cast(
            torch.Tensor,
            _require_method(self._problem, "log_likelihood")(
                physical,
                ctx,
            ),
        )

    def log_prior(self, state: ModelState) -> torch.Tensor:
        """Delegate log-prior on one physical state."""
        physical = self._physical_state(state)
        return cast(
            torch.Tensor,
            _require_method(self._problem, "log_prior")(physical),
        )


def _parameterized_inverter_inputs(
    problem: InverseProblemLike,
    *,
    parameterization: Parameterization,
    theta: torch.Tensor,
    vector_name: str,
    kwargs: Mapping[str, Any],
) -> tuple[
    InverseProblemLike,
    Mapping[str, torch.Tensor],
    _ResultTransform,
]:
    """Validate the flat-vector factory and return its three constructor inputs."""
    if not isinstance(parameterization, Parameterization):
        raise GeoBrainError(
            "Inverter.from_parameterization requires a Parameterization",
            object_name="Inverter.from_parameterization",
            field="parameterization",
            expected=Parameterization,
            actual=type(parameterization),
        )
    if not isinstance(vector_name, str) or not vector_name:
        raise GeoBrainError(
            "Inverter.from_parameterization vector_name must be non-empty",
            object_name="Inverter.from_parameterization",
            field="vector_name",
            expected="non-empty str",
            actual=vector_name,
        )
    if not isinstance(theta, torch.Tensor):
        raise GeoBrainError(
            "Inverter.from_parameterization theta must be a tensor",
            object_name="Inverter.from_parameterization",
            field="theta",
            expected=torch.Tensor,
            actual=type(theta),
        )
    if theta.ndim != 1 or theta.numel() != parameterization.size:
        raise FieldShapeError(
            "Inverter.from_parameterization theta has wrong shape",
            object_name="Inverter.from_parameterization",
            field="theta",
            expected=(parameterization.size,),
            actual=tuple(theta.shape),
        )
    for reserved in ("params", "result_transform"):
        if reserved in kwargs:
            raise GeoBrainError(
                "Inverter.from_parameterization owns its parameter seam",
                object_name="Inverter.from_parameterization",
                field=reserved,
                expected=f"{reserved} omitted",
                actual=kwargs[reserved],
            )

    adapted = _ParameterizedProblem(
        problem,
        parameterization,
        vector_name=vector_name,
    )

    def to_physical(
        params: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if vector_name not in params:
            raise GeoBrainError(
                "Parameterized result is missing its latent vector",
                object_name="Inverter.run",
                field=vector_name,
                expected=f"present in {sorted(params)}",
            )
        physical = parameterization(params[vector_name])
        return cast(Mapping[str, torch.Tensor], physical.tensors)

    return (
        cast(InverseProblemLike, adapted),
        {vector_name: theta},
        to_physical,
    )
