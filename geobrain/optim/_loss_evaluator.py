"""Typed loss evaluation for deterministic optimization.

MAP prefers the one-forward ``problem.objective`` contract because it needs a
complete posterior record. MLE deliberately calls the data-only
``problem.loss`` route so an attached prior is never evaluated. A private
structural adapter supports older third-party problem shapes without adding an
unrequested likelihood evaluation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol, cast

import torch

from geobrain.core import ForwardContext, GeoBrainError, ModelState
from geobrain.inverse import ObjectiveEvaluationLike

Regularizer = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]


class _ObjectiveProblem(Protocol):
    def objective(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> ObjectiveEvaluationLike: ...


class _DataProblem(Protocol):
    def loss(
        self,
        state: ModelState,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor: ...


class _LegacyMapProblem(_DataProblem, Protocol):
    def log_prior(self, state: ModelState) -> torch.Tensor: ...


@dataclass(frozen=True)
class LossEvaluation:
    """One live scalar objective and its structured component losses."""

    data_loss: torch.Tensor
    prior_loss: torch.Tensor
    regularization_loss: torch.Tensor
    total_loss: torch.Tensor
    term_losses: Mapping[str, torch.Tensor]


def _require_scalar(
    value: object,
    *,
    field: str,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 0
        or not value.is_floating_point()
    ):
        raise GeoBrainError(
            f"LossEvaluator.{field} must be a real floating scalar tensor",
            object_name="LossEvaluator",
            field=field,
            expected="0-d real floating torch.Tensor",
            actual=(
                {
                    "type": type(value),
                    "shape": tuple(value.shape),
                    "dtype": value.dtype,
                    "device": value.device,
                }
                if isinstance(value, torch.Tensor)
                else type(value)
            ),
        )
    if reference is not None and (
        value.dtype != reference.dtype or value.device != reference.device
    ):
        raise GeoBrainError(
            f"LossEvaluator.{field} must match data_loss dtype and device",
            object_name="LossEvaluator",
            field=field,
            expected={
                "dtype": reference.dtype,
                "device": reference.device,
            },
            actual={"dtype": value.dtype, "device": value.device},
        )
    return value


def _require_term_name(name: object) -> str:
    """Require a meaningful string key without changing the caller's name."""
    if not isinstance(name, str) or not name.strip():
        raise GeoBrainError(
            "LossEvaluator term_losses keys must be non-empty strings",
            object_name="LossEvaluator",
            field="term_losses",
            expected="non-empty string keys",
            actual=name,
        )
    return name


def _objective_method(problem: object) -> _ObjectiveProblem | None:
    method = getattr(problem, "objective", None)
    return cast(_ObjectiveProblem, problem) if callable(method) else None


def _data_problem(problem: object) -> _DataProblem:
    if not callable(getattr(problem, "loss", None)):
        raise GeoBrainError(
            "LossEvaluator problem does not provide a data-loss protocol",
            object_name="LossEvaluator",
            field="problem",
            expected="callable loss(state, ctx)",
            actual=type(problem),
        )
    return cast(_DataProblem, problem)


def _legacy_map_problem(problem: object) -> _LegacyMapProblem:
    missing = [
        name
        for name in ("loss", "log_prior")
        if not callable(getattr(problem, name, None))
    ]
    if missing:
        raise GeoBrainError(
            "LossEvaluator problem does not provide an objective protocol",
            object_name="LossEvaluator",
            field="problem",
            expected=(
                "callable objective(state, ctx), or legacy callable "
                "loss/log_prior methods"
            ),
            actual={"type": type(problem), "missing": missing},
        )
    return cast(_LegacyMapProblem, problem)


def _diagnostic_term_losses(
    problem: object,
    *,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Copy optional maintained named diagnostics without another forward."""
    raw_terms = getattr(problem, "term_losses", None)
    if raw_terms is None:
        return {}
    if not isinstance(raw_terms, MappingABC):
        raise GeoBrainError(
            "LossEvaluator diagnostic term_losses must be a mapping",
            object_name="LossEvaluator",
            field="term_losses",
            expected="mapping of names to scalar tensors or real numbers",
            actual=type(raw_terms),
        )
    terms: dict[str, torch.Tensor] = {}
    for raw_name, value in raw_terms.items():
        name = _require_term_name(raw_name)
        field = f"term_losses[{name!r}]"
        if isinstance(value, torch.Tensor):
            terms[name] = _require_scalar(
                value,
                field=field,
                reference=reference,
            )
        elif isinstance(value, Real) and not isinstance(value, bool):
            terms[name] = reference.new_tensor(float(value))
        else:
            raise GeoBrainError(
                "LossEvaluator diagnostic term losses must be scalar",
                object_name="LossEvaluator",
                field=field,
                expected="0-d real floating torch.Tensor or real number",
                actual=type(value),
            )
    return terms


def _validate_finite_components(
    *,
    data_loss: torch.Tensor,
    prior_loss: torch.Tensor,
    regularization_loss: torch.Tensor,
    total_loss: torch.Tensor,
    term_losses: Mapping[str, torch.Tensor],
) -> None:
    """Reject a finite total paired with any non-finite constituent."""
    if not bool(torch.isfinite(total_loss)):
        return
    components = {
        "data_loss": data_loss,
        "prior_loss": prior_loss,
        "regularization_loss": regularization_loss,
        **{
            f"term_losses[{name!r}]": value
            for name, value in term_losses.items()
        },
    }
    for field, value in components.items():
        if bool(torch.isfinite(value)):
            continue
        raise GeoBrainError(
            "LossEvaluator finite total_loss requires finite components",
            object_name="LossEvaluator",
            field=field,
            expected="finite scalar when total_loss is finite",
            actual=float(value.detach()),
        )


class LossEvaluator:
    """Evaluate MAP/MLE losses without hidden extra forward executions."""

    def __init__(
        self,
        problem: object,
        *,
        mode: Literal["MLE", "MAP"],
        regularizer: Regularizer | None,
        ctx: ForwardContext,
        state_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in ("MLE", "MAP"):
            raise GeoBrainError(
                "LossEvaluator mode must be 'MLE' or 'MAP'",
                object_name="LossEvaluator",
                field="mode",
                expected="'MLE' or 'MAP'",
                actual=mode,
            )
        self._problem = problem
        self._mode = mode
        self._regularizer = regularizer
        self._ctx = ctx
        self._state_metadata = (
            {} if state_metadata is None else dict(state_metadata)
        )

    def evaluate(
        self,
        params: Mapping[str, torch.Tensor],
    ) -> LossEvaluation:
        """Return one structured live loss graph at ``params``."""
        state = ModelState(
            tensors=params,
            metadata=self._state_metadata,
        )
        objective_problem = _objective_method(self._problem)
        if self._mode == "MAP" and objective_problem is not None:
            objective = objective_problem.objective(state, self._ctx)
            data_loss = _require_scalar(
                getattr(objective, "data_loss", None),
                field="data_loss",
            )
            log_prior = _require_scalar(
                getattr(objective, "log_prior", None),
                field="log_prior",
                reference=data_loss,
            )
            raw_terms = getattr(objective, "term_losses", None)
            if not isinstance(raw_terms, MappingABC):
                raise GeoBrainError(
                    "LossEvaluator objective term_losses must be a mapping",
                    object_name="LossEvaluator",
                    field="term_losses",
                    expected="mapping of names to scalar tensors",
                    actual=type(raw_terms),
                )
            term_losses: dict[str, torch.Tensor] = {}
            for raw_name, value in raw_terms.items():
                name = _require_term_name(raw_name)
                term_losses[name] = _require_scalar(
                    value,
                    field=f"term_losses[{name!r}]",
                    reference=data_loss,
                )
        elif self._mode == "MLE":
            data = _data_problem(self._problem)
            data_loss = _require_scalar(
                data.loss(state, self._ctx),
                field="data_loss",
            )
            log_prior = data_loss.new_zeros(())
            term_losses = _diagnostic_term_losses(
                self._problem,
                reference=data_loss,
            )
        else:
            legacy = _legacy_map_problem(self._problem)
            data_loss = _require_scalar(
                legacy.loss(state, self._ctx),
                field="data_loss",
            )
            log_prior = _require_scalar(
                legacy.log_prior(state),
                field="log_prior",
                reference=data_loss,
            )
            term_losses = _diagnostic_term_losses(
                self._problem,
                reference=data_loss,
            )

        prior_loss = -log_prior if self._mode == "MAP" else log_prior
        regularization_loss = (
            _require_scalar(
                self._regularizer(params),
                field="regularization_loss",
                reference=data_loss,
            )
            if self._regularizer is not None
            else data_loss.new_zeros(())
        )
        total_loss = data_loss + prior_loss + regularization_loss
        _validate_finite_components(
            data_loss=data_loss,
            prior_loss=prior_loss,
            regularization_loss=regularization_loss,
            total_loss=total_loss,
            term_losses=term_losses,
        )
        return LossEvaluation(
            data_loss=data_loss,
            prior_loss=prior_loss,
            regularization_loss=regularization_loss,
            total_loss=total_loss,
            term_losses=MappingProxyType(term_losses),
        )
