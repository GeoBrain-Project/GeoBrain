"""
Immutable, typed records for inverse-objective evaluations.

This module defines the structural result contract shared by inverse-objective
consumers and the concrete single-problem record returned by
:meth:`geobrain.inverse.InverseProblem.objective`. It contains no solver or
joint-decomposition policy.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import torch

from ..core.containers import ForwardOutput
from ..core.errors import GeoBrainError

__all__ = ["ObjectiveEvaluationLike", "ObjectiveEvaluation"]


@runtime_checkable
class ObjectiveEvaluationLike(Protocol):
    """Structural contract for a fully evaluated inverse objective.

    Implementations expose scalar data, likelihood, and prior terms together
    with an immutable-by-convention mapping of named scalar term losses. The
    protocol is structural: third-party records need not inherit from a
    GeoBrain class.
    """

    @property
    def data_loss(self) -> torch.Tensor:
        """Return the scalar data-misfit loss."""
        ...

    @property
    def log_likelihood(self) -> torch.Tensor:
        """Return the scalar log-likelihood."""
        ...

    @property
    def log_prior(self) -> torch.Tensor:
        """Return the scalar log-prior."""
        ...

    @property
    def term_losses(self) -> Mapping[str, torch.Tensor]:
        """Return named scalar loss terms, empty when no decomposition exists."""
        ...


@dataclass(frozen=True)
class ObjectiveEvaluation:
    """Immutable result of one complete inverse-objective evaluation.

    Args:
        prediction: Forward output produced by the single forward execution.
        data_loss: Scalar data-misfit tensor.
        log_likelihood: Scalar log-likelihood tensor.
        log_prior: Scalar log-prior tensor.
        term_losses: Optional named scalar data-loss terms. The mapping is
            defensively copied and exposed read-only.

    Raises:
        GeoBrainError: If an objective value or named term is not a scalar
            :class:`torch.Tensor`.
    """

    prediction: ForwardOutput
    data_loss: torch.Tensor
    log_likelihood: torch.Tensor
    log_prior: torch.Tensor
    term_losses: Mapping[str, torch.Tensor] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Validate scalar terms and freeze a defensive copy of term losses."""
        for name in ("data_loss", "log_likelihood", "log_prior"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise GeoBrainError(
                    "objective terms must be scalar tensors",
                    object_name="ObjectiveEvaluation",
                    field=name,
                    expected="0-d torch.Tensor",
                    actual=type(value).__name__,
                )
        frozen_terms = MappingProxyType(dict(self.term_losses))
        for name, value in frozen_terms.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise GeoBrainError(
                    "per-term objective values must be scalar tensors",
                    object_name="ObjectiveEvaluation",
                    field=f"term_losses[{name!r}]",
                    expected="0-d torch.Tensor",
                    actual=type(value).__name__,
                )
        object.__setattr__(self, "term_losses", frozen_terms)

    @property
    def total_loss(self) -> torch.Tensor:
        """Return the scalar MAP objective ``data_loss - log_prior``."""
        return self.data_loss - self.log_prior
