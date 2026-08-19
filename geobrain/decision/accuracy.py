"""Bayes-action decision-accuracy and expected-utility gain.

These functions evaluate the value of an acquisition in decision currency.
They optimize the action both before and after each possible acquisition
outcome; they are not mutual-information estimators.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from geobrain.core.validation import validate_positive_int
from geobrain.core.errors import GeoBrainError
from geobrain.decision._metadata import freeze_metadata, thaw_metadata


FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class DecisionAccuracyResult:
    """Owned spatial decision-accuracy result with explicit units.

    Attributes:
        gain_map: per-cell accuracy gain.
        prior_accuracy: baseline accuracy without the data.
        expected_posterior_accuracy: accuracy with the data.
        mean_gain: spatial mean of ``gain_map``.
        n_realizations: ensemble size behind the estimate.
        estimator: which estimator produced it.
        units: unit string of the mapped quantity.
        gain_standard_error: optional per-cell standard error.
        metadata: estimator extras.
    """

    gain_map: torch.Tensor
    prior_accuracy: torch.Tensor
    expected_posterior_accuracy: torch.Tensor
    mean_gain: float
    n_realizations: int
    estimator: str
    units: Literal["normalized_accuracy", "accuracy_probability"]
    gain_standard_error: torch.Tensor | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        owner = "DecisionAccuracyResult"
        tensors = (
            ("gain_map", self.gain_map),
            ("prior_accuracy", self.prior_accuracy),
            (
                "expected_posterior_accuracy",
                self.expected_posterior_accuracy,
            ),
        )
        owned: dict[str, torch.Tensor] = {}
        for field, value in tensors:
            if not isinstance(value, torch.Tensor):
                raise GeoBrainError(
                    f"{field} must be a tensor",
                    object_name=owner,
                    field=field,
                    expected="finite tensor",
                    actual=type(value),
                )
            clone = value.detach().clone()
            if clone.numel() < 1 or not bool(torch.isfinite(clone).all()):
                raise GeoBrainError(
                    f"{field} must be non-empty and finite",
                    object_name=owner,
                    field=field,
                    expected="non-empty finite tensor",
                    actual=tuple(clone.shape),
                )
            owned[field] = clone
        shape = owned["gain_map"].shape
        if (
            owned["prior_accuracy"].shape != shape
            or owned["expected_posterior_accuracy"].shape != shape
        ):
            raise GeoBrainError(
                "result tensors must share the grid shape",
                object_name=owner,
                field="gain_map",
                expected=shape,
                actual=(
                    owned["prior_accuracy"].shape,
                    owned["expected_posterior_accuracy"].shape,
                ),
            )
        if (
            isinstance(self.mean_gain, bool)
            or not isinstance(self.mean_gain, (int, float))
            or not math.isfinite(float(self.mean_gain))
            or not math.isclose(
                float(self.mean_gain),
                float(owned["gain_map"].mean()),
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
        ):
            raise GeoBrainError(
                "mean_gain must equal the finite grid mean",
                object_name=owner,
                field="mean_gain",
                expected=float(owned["gain_map"].mean()),
                actual=self.mean_gain,
            )
        n_realizations = validate_positive_int(
            self.n_realizations,
            owner=owner,
            field="n_realizations",
        )
        if not isinstance(self.estimator, str) or not self.estimator:
            raise GeoBrainError(
                "estimator must be a non-empty string",
                object_name=owner,
                field="estimator",
                expected="non-empty string",
                actual=self.estimator,
            )
        if self.units not in ("normalized_accuracy", "accuracy_probability"):
            raise GeoBrainError(
                "units must name the decision-accuracy scale",
                object_name=owner,
                field="units",
                expected="'normalized_accuracy' or 'accuracy_probability'",
                actual=self.units,
            )
        standard_error: torch.Tensor | None = None
        if self.gain_standard_error is not None:
            if not isinstance(self.gain_standard_error, torch.Tensor):
                raise GeoBrainError(
                    "gain_standard_error must be a tensor or None",
                    object_name=owner,
                    field="gain_standard_error",
                    expected=f"finite non-negative tensor with shape {shape}",
                    actual=type(self.gain_standard_error),
                )
            standard_error = self.gain_standard_error.detach().clone()
            if (
                standard_error.shape != shape
                or not bool(torch.isfinite(standard_error).all())
                or bool((standard_error < 0).any())
            ):
                raise GeoBrainError(
                    "gain_standard_error must match the grid and be non-negative",
                    object_name=owner,
                    field="gain_standard_error",
                    expected=f"finite non-negative tensor with shape {shape}",
                    actual=tuple(standard_error.shape),
                )
        for field, value in owned.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "mean_gain", float(self.mean_gain))
        object.__setattr__(self, "n_realizations", n_realizations)
        object.__setattr__(self, "gain_standard_error", standard_error)
        object.__setattr__(
            self,
            "metadata",
            freeze_metadata(self.metadata, object_name=owner),
        )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return "\n".join([
            "=== Spatial Decision Accuracy Result ===",
            f"Mean gain           : {self.mean_gain:.4f}",
            f"Max gain (per cell) : {float(self.gain_map.max()):.4f}",
            f"Grid shape          : {tuple(self.gain_map.shape)}",
            f"Prior ensemble      : {self.n_realizations} realizations",
            f"Estimator           : {self.estimator}",
            f"Units               : {self.units}",
        ])

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain representation."""
        return {
            "gain_map": self.gain_map.detach().cpu().numpy().copy(),
            "prior_accuracy": (
                self.prior_accuracy.detach().cpu().numpy().copy()
            ),
            "expected_posterior_accuracy": (
                self.expected_posterior_accuracy.detach().cpu().numpy().copy()
            ),
            "mean_gain": self.mean_gain,
            "n_realizations": self.n_realizations,
            "estimator": self.estimator,
            "units": self.units,
            "gain_standard_error": (
                None
                if self.gain_standard_error is None
                else self.gain_standard_error.detach().cpu().numpy().copy()
            ),
            "metadata": thaw_metadata(self.metadata),
        }


def _numeric_array(
    value: ArrayLike,
    *,
    object_name: str,
    field: str,
) -> FloatArray:
    """Return a float64 array while rejecting booleans and coercion failures."""
    try:
        untyped = np.asarray(value)
        if np.issubdtype(untyped.dtype, np.bool_):
            raise TypeError("boolean values are not numeric probabilities")
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeoBrainError(
            f"{field} must contain numeric values",
            object_name=object_name,
            field=field,
            expected="numeric array without bool values",
            actual=value,
        ) from exc
    return result


def _probability_array(
    value: ArrayLike,
    field: str,
    *,
    object_name: str,
) -> FloatArray:
    result = _numeric_array(value, object_name=object_name, field=field)
    if (
        result.size == 0
        or not bool(np.all(np.isfinite(result)))
        or bool(np.any(result < 0.0))
        or bool(np.any(result > 1.0))
    ):
        raise GeoBrainError(
            "probabilities must be finite values in [0, 1]",
            object_name=object_name,
            field=field,
            expected="non-empty finite array in [0, 1]",
            actual=value,
        )
    return result


def _outcome_weights(
    weights: ArrayLike | None,
    n_outcomes: int,
    *,
    object_name: str,
) -> FloatArray:
    if n_outcomes < 1:
        raise GeoBrainError(
            "posterior probabilities require at least one outcome",
            object_name=object_name,
            field="posterior_state_probabilities",
            expected="non-empty leading outcome axis",
            actual=n_outcomes,
        )
    if weights is None:
        return cast(
            FloatArray,
            np.full(n_outcomes, 1.0 / n_outcomes, dtype=np.float64),
        )
    result = _numeric_array(
        weights,
        object_name=object_name,
        field="weights",
    )
    if (
        result.ndim != 1
        or result.shape[0] != n_outcomes
        or not bool(np.all(np.isfinite(result)))
        or bool(np.any(result < 0.0))
    ):
        raise GeoBrainError(
            "outcome weights must be finite, non-negative, and non-zero",
            object_name=object_name,
            field="weights",
            expected=(
                f"{n_outcomes} finite non-negative values with positive sum"
            ),
            actual=weights,
        )
    with np.errstate(over="ignore"):
        total = float(result.sum())
    if total <= 0.0:
        raise GeoBrainError(
            "outcome weights must be finite, non-negative, and non-zero",
            object_name=object_name,
            field="weights",
            expected=(
                f"{n_outcomes} finite non-negative values with positive sum"
            ),
            actual=weights,
        )
    if math.isfinite(total):
        return cast(
            FloatArray,
            np.asarray(result / total, dtype=np.float64),
        )
    maximum = float(result.max())
    scaled = result / maximum
    return cast(
        FloatArray,
        np.asarray(scaled / scaled.sum(), dtype=np.float64),
    )


def _utility_array(utilities: ArrayLike) -> FloatArray:
    result = _numeric_array(
        utilities,
        object_name="expected_utility_gain",
        field="utilities",
    )
    if (
        result.ndim != 2
        or result.shape[0] < 1
        or result.shape[1] < 1
        or not bool(np.all(np.isfinite(result)))
    ):
        raise GeoBrainError(
            "utilities must be a finite action-by-state matrix",
            object_name="expected_utility_gain",
            field="utilities",
            expected="non-empty finite [n_actions, n_states] array",
            actual=utilities,
        )
    return result


def expected_utility_gain(
    prior_state_probabilities: ArrayLike,
    posterior_state_probabilities: ArrayLike,
    utilities: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> FloatArray:
    """Return expected improvement in optimal action utility.

    ``utilities[a, s]`` is the payoff for action ``a`` in state ``s``.
    The final axis of each probability array is the state axis; the posterior
    adds a leading acquisition-outcome axis. Outcome weights are normalized.

    A negative result is retained. It signals that the supplied posterior
    outcomes and weights are not Bayes-consistent with the supplied prior;
    silently clamping it would conceal that information.

    Args:
        prior_state_probabilities: prior state distribution.
        posterior_state_probabilities: per-outcome posterior distributions.
        utilities: decision-by-state utility table.
        weights: optional outcome weights (defaults to uniform).
    """
    prior = _probability_array(
        prior_state_probabilities,
        "prior_state_probabilities",
        object_name="expected_utility_gain",
    )
    posterior = _probability_array(
        posterior_state_probabilities,
        "posterior_state_probabilities",
        object_name="expected_utility_gain",
    )
    utility = _utility_array(utilities)
    n_states = utility.shape[1]
    if (
        prior.ndim == 0
        or prior.shape[-1] != n_states
        or posterior.ndim != prior.ndim + 1
        or posterior.shape[1:] != prior.shape
    ):
        raise GeoBrainError(
            "probability shapes must align with outcome and state axes",
            object_name="expected_utility_gain",
            field="posterior_state_probabilities",
            expected=("n_outcomes", *prior.shape),
            actual=posterior.shape,
        )
    if not bool(np.allclose(prior.sum(axis=-1), 1.0)) or not bool(
        np.allclose(posterior.sum(axis=-1), 1.0)
    ):
        raise GeoBrainError(
            "state probabilities must sum to one",
            object_name="expected_utility_gain",
            field="state_probabilities",
            expected="sum(state_axis) == 1",
            actual="non-normalized probabilities",
        )
    outcome_weights = _outcome_weights(
        weights,
        posterior.shape[0],
        object_name="expected_utility_gain",
    )
    prior_value = np.einsum("...s,as->...a", prior, utility).max(axis=-1)
    posterior_value = np.einsum(
        "o...s,as->o...a",
        posterior,
        utility,
    ).max(axis=-1)
    return cast(
        FloatArray,
        np.asarray(
            np.tensordot(
                outcome_weights,
                posterior_value,
                axes=([0], [0]),
            )
            - prior_value,
            dtype=np.float64,
        ),
    )


def expected_accuracy_gain(
    prior_probability: ArrayLike,
    posterior_probabilities: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    normalize: bool = True,
) -> FloatArray:
    """Return the expected gain in optimal binary-classification accuracy.

    Accuracy for event probability ``p`` is ``max(p, 1 - p)`` because both
    binary actions are considered. With ``normalize=True``, the raw
    probability gain is divided cellwise by its perfect-information ceiling
    ``1 - max(p, 1 - p)``. A zero ceiling has defined normalized gain zero.

    Args:
        prior_probability: prior success probability per decision cell.
        posterior_probabilities: per-outcome posterior probabilities.
        weights: optional outcome weights (defaults to uniform).
        normalize: divide the gain by the prior baseline when True.
    """
    if not isinstance(normalize, bool):
        raise GeoBrainError(
            "normalize must be a bool",
            object_name="expected_accuracy_gain",
            field="normalize",
            expected=bool,
            actual=type(normalize),
        )
    prior = _probability_array(
        prior_probability,
        "prior_probability",
        object_name="expected_accuracy_gain",
    )
    posterior = _probability_array(
        posterior_probabilities,
        "posterior_probabilities",
        object_name="expected_accuracy_gain",
    )
    if posterior.ndim == 0:
        raise GeoBrainError(
            "posterior probabilities require an outcome axis",
            object_name="expected_accuracy_gain",
            field="posterior_probabilities",
            expected="array with leading outcome axis",
            actual=posterior_probabilities,
        )
    if posterior.shape[1:] != prior.shape:
        raise GeoBrainError(
            "posterior event shape must match the prior event shape",
            object_name="expected_accuracy_gain",
            field="posterior_probabilities",
            expected=("n_outcomes", *prior.shape),
            actual=posterior.shape,
        )
    prior_states = np.stack((1.0 - prior, prior), axis=-1)
    posterior_states = np.stack((1.0 - posterior, posterior), axis=-1)
    prior_accuracy = np.maximum(prior, 1.0 - prior)
    gain = expected_utility_gain(
        prior_states,
        posterior_states,
        np.eye(2, dtype=np.float64),
        weights=weights,
    )
    if not normalize:
        return gain
    ceiling = 1.0 - prior_accuracy
    return cast(
        FloatArray,
        np.divide(
            gain,
            ceiling,
            out=np.zeros_like(gain, dtype=np.float64),
            where=ceiling > 0.0,
        ),
    )


__all__ = [
    "DecisionAccuracyResult",
    "expected_accuracy_gain",
    "expected_utility_gain",
]
