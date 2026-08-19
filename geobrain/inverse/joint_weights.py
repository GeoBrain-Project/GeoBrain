"""Immutable weighting policies for named joint-inversion objective terms.

Policies hold configuration only. A :class:`geobrain.inverse.JointProblem`
owns the current calibrated mapping as run state, initializes it lazily once
for gradient-norm weighting, and replaces it only during explicit rebalancing.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, cast

import torch

from ..core.errors import GeoBrainError

__all__ = [
    "FixedWeights",
    "GradientNormWeights",
    "JointWeightPolicy",
]


class JointWeightPolicy(Protocol):
    """Structural policy that calibrates named weights from gradient norms."""

    def calibrate(
        self,
        gradient_norms: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Return immutable weights for the exact supplied term names."""
        ...


def _validate_term_names(
    term_names: tuple[str, ...],
    *,
    object_name: str,
) -> None:
    valid = (
        isinstance(term_names, tuple)
        and bool(term_names)
        and len(set(term_names)) == len(term_names)
        and all(isinstance(name, str) and bool(name) for name in term_names)
    )
    if not valid:
        raise GeoBrainError(
            "joint weight term names must be non-empty and unique",
            object_name=object_name,
            field="term_names",
            expected="non-empty tuple of unique non-empty strings",
            actual=term_names,
        )


def _validated_values(
    values: object,
    *,
    term_names: tuple[str, ...],
    object_name: str,
    field: str,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise GeoBrainError(
            "joint weight inputs must be mappings",
            object_name=object_name,
            field=field,
            expected="Mapping[str, float]",
            actual=values,
        )
    mapping = cast(Mapping[str, float], values)
    actual_names = set(mapping)
    expected_names = set(term_names)
    if actual_names != expected_names:
        raise GeoBrainError(
            "joint weight terms must match exactly",
            object_name=object_name,
            field=field,
            expected=sorted(expected_names),
            actual=sorted(actual_names),
        )

    validated: dict[str, float] = {}
    for name in term_names:
        raw_value = mapping[name]
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
            or float(raw_value) < 0.0
        ):
            raise GeoBrainError(
                "joint weight values must be finite and non-negative",
                object_name=object_name,
                field=f"{field}[{name!r}]",
                expected="finite non-negative float",
                actual=raw_value,
            )
        validated[name] = float(raw_value)
    return validated


@dataclass(frozen=True)
class FixedWeights:
    """Immutable fixed weights for an exact set of joint terms.

    Args:
        values: Non-empty mapping from unique non-empty term names to finite,
            non-negative weights. The mapping is defensively copied.

    Raises:
        GeoBrainError: If term names or weight values are invalid.
    """

    values: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate and defensively freeze the configured weights."""
        if not isinstance(self.values, Mapping):
            raise GeoBrainError(
                "joint weight inputs must be mappings",
                object_name="FixedWeights",
                field="values",
                expected="Mapping[str, float]",
                actual=self.values,
            )
        term_names = tuple(self.values)
        _validate_term_names(term_names, object_name="FixedWeights")
        validated = _validated_values(
            self.values,
            term_names=term_names,
            object_name="FixedWeights",
            field="values",
        )
        object.__setattr__(
            self,
            "values",
            MappingProxyType(validated),
        )

    def calibrate(
        self,
        gradient_norms: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Return configured weights after validating the exact term keys."""
        _validated_values(
            gradient_norms,
            term_names=tuple(self.values),
            object_name="FixedWeights",
            field="gradient_norms",
        )
        return self.values


@dataclass(frozen=True)
class GradientNormWeights:
    """Calibrate immutable weights that balance per-term gradient norms.

    Args:
        term_names: Ordered, non-empty tuple of unique non-empty term names.
        relative_floor: Positive finite floor relative to the median norm.

    Raises:
        GeoBrainError: If the names or relative floor are invalid.

    Notes:
        For an even number of terms, the median is the arithmetic mean of the
        two central values. This preserves GeoBrain's pre-extraction numerical
        behavior. When every norm is zero, every weight is one.
    """

    term_names: tuple[str, ...]
    relative_floor: float = 1e-3

    def __post_init__(self) -> None:
        """Validate immutable policy configuration."""
        _validate_term_names(
            self.term_names,
            object_name="GradientNormWeights",
        )
        if (
            isinstance(self.relative_floor, bool)
            or not isinstance(self.relative_floor, (int, float))
            or not math.isfinite(float(self.relative_floor))
            or float(self.relative_floor) <= 0.0
        ):
            raise GeoBrainError(
                "gradient-norm relative floor must be positive and finite",
                object_name="GradientNormWeights",
                field="relative_floor",
                expected="finite float > 0",
                actual=self.relative_floor,
            )

    def calibrate(
        self,
        gradient_norms: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Return read-only weights calibrated from exact finite norms.

        Args:
            gradient_norms: Mapping whose keys exactly equal
                :attr:`term_names` and whose values are finite and
                non-negative.

        Returns:
            Immutable calibrated weight mapping in :attr:`term_names` order.

        Raises:
            GeoBrainError: If keys or norm values violate the contract.
        """
        norms = _validated_values(
            gradient_norms,
            term_names=self.term_names,
            object_name="GradientNormWeights",
            field="gradient_norms",
        )
        ordered = sorted(norms.values())
        midpoint = len(ordered) // 2
        if len(ordered) % 2 == 1:
            median = ordered[midpoint]
        else:
            lower = ordered[midpoint - 1]
            upper = ordered[midpoint]
            median = lower + (upper - lower) * 0.5

        if median <= 0.0:
            weights = {name: 1.0 for name in self.term_names}
        else:
            floor = float(self.relative_floor) * median
            weights = {}
            for name in self.term_names:
                denominator = max(norms[name], floor)
                value = median / denominator if denominator > 0.0 else math.inf
                if not math.isfinite(value) or value <= 0.0:
                    raise GeoBrainError(
                        "calibrated gradient-norm weights must be finite and positive",
                        object_name="GradientNormWeights",
                        field=f"weights[{name!r}]",
                        expected="finite positive float",
                        actual=value,
                    )
                weights[name] = value
        return MappingProxyType(weights)


def _build_weight_state(
    term_names: tuple[str, ...],
    weights: None | Mapping[str, float] | str,
) -> tuple[JointWeightPolicy, Mapping[str, float] | None]:
    if weights is None:
        policy = FixedWeights({name: 1.0 for name in term_names})
        return policy, policy.values
    if isinstance(weights, str):
        if weights != "grad_norm":
            raise GeoBrainError(
                "JointProblem weights= string value must be 'grad_norm'",
                object_name="JointProblem",
                field="weights",
                expected="'grad_norm'",
                actual=weights,
            )
        return GradientNormWeights(term_names), None
    if isinstance(weights, Mapping):
        weight_names = set(weights)
        expected_names = set(term_names)
        if weight_names != expected_names:
            raise GeoBrainError(
                "JointProblem weights= keys must match forwards keys exactly",
                object_name="JointProblem",
                field="weights",
                expected=sorted(expected_names),
                actual=sorted(weight_names),
            )
        policy = FixedWeights({name: weights[name] for name in term_names})
        return policy, policy.values
    raise GeoBrainError(
        "JointProblem weights= must be None, a {name: float} mapping, or 'grad_norm'",
        object_name="JointProblem",
        field="weights",
        expected="None | Mapping[str, float] | 'grad_norm'",
        actual=weights,
    )


def _gradient_norm(
    objective: torch.Tensor,
    tensors: Iterable[torch.Tensor],
    *,
    retain_graph: bool = False,
) -> float:
    leaves = [tensor for tensor in tensors if tensor.requires_grad]
    if not leaves or not objective.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        objective,
        leaves,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    squared_norm: torch.Tensor | None = None
    for gradient in gradients:
        if gradient is None:
            continue
        term = gradient.detach().pow(2).sum()
        squared_norm = term if squared_norm is None else squared_norm + term
    if squared_norm is None:
        return 0.0
    return float(torch.sqrt(squared_norm))


def _gradient_norms_from_losses(
    term_losses: Mapping[str, torch.Tensor],
    tensors: Iterable[torch.Tensor],
) -> Mapping[str, float]:
    leaves = tuple(tensor for tensor in tensors if tensor.requires_grad)
    return MappingProxyType(
        {
            name: _gradient_norm(loss, leaves, retain_graph=True)
            for name, loss in term_losses.items()
        }
    )


def _weighted_loss(
    term_losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> torch.Tensor:
    total: torch.Tensor | None = None
    for name, term_loss in term_losses.items():
        weighted = weights[name] * term_loss
        total = weighted if total is None else total + weighted
    assert total is not None
    return total
