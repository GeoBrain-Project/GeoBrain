"""Continuous indicator kriging with explicit CDF corrections.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ...conditioning import ConditioningPolicy, ConditioningSet
from ...frames import GeoFrame, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError, GeomodelNumericsError
from ...neighbourhood import NeighbourhoodSpec, StaticKDTreeNeighbourhood
from .._domain import make_estimation_result
from ..models.covariance import CovarianceModel
from .covariance_matrix import covariance_matrix, covariance_vector
from .kriging import _hard_arrays, _validate_common
from .kriging_kernel import (
    KrigingSolvePolicy,
    default_neighbourhood,
    solve_kriging_system,
)

__all__ = ["IndicatorKriging"]


def _order_relation_correction(values: FloatArray) -> FloatArray:
    """Return the deterministic bounded, monotone projection used by SISIM."""
    return _correct_cdf_with_count(values)[0]


def _correct_cdf_with_count(values: FloatArray) -> tuple[FloatArray, int]:
    corrected = np.array(values, dtype=np.float64, copy=True)
    corrections = 0
    for index, value in enumerate(corrected):
        bounded = min(1.0, max(0.0, float(value)))
        if bounded != float(value):
            corrections += 1
        corrected[index] = bounded
    for index in range(1, corrected.size):
        if corrected[index] < corrected[index - 1]:
            corrected[index] = corrected[index - 1]
            corrections += 1
    return as_float_array(corrected), corrections


def _etype_mean(thresholds: tuple[float, ...], probabilities: FloatArray) -> float:
    total = thresholds[0]
    for left in range(len(thresholds) - 1):
        width = thresholds[left + 1] - thresholds[left]
        cdf_midpoint = 0.5 * (float(probabilities[left]) + float(probabilities[left + 1]))
        total += width * (1.0 - cdf_midpoint)
    return float(total)


def _indicator_system(
    model: CovarianceModel,
    nearby: FloatArray,
    target: FloatArray,
    indicator_values: FloatArray,
    solve_policy: KrigingSolvePolicy,
    *,
    object_name: str,
) -> tuple[float, float, float]:
    count = int(nearby.shape[0])
    matrix: FloatArray = as_float_array(np.zeros((count + 1, count + 1), dtype=np.float64))
    rhs: FloatArray = as_float_array(np.zeros(count + 1, dtype=np.float64))
    matrix[:count, :count] = covariance_matrix(model, nearby, nearby)
    covariance_dt = covariance_vector(model, nearby, target)
    rhs[:count] = covariance_dt
    matrix[count, :count] = 1.0
    matrix[:count, count] = 1.0
    rhs[count] = 1.0
    solution, jitter, residual = solve_kriging_system(
        as_float_array(matrix),
        as_float_array(rhs),
        data_count=count,
        policy=solve_policy,
        object_name=object_name,
    )
    probability = float(np.dot(solution[:count], indicator_values))
    if not math.isfinite(probability):
        raise GeomodelNumericsError(
            "indicator probability is not finite",
            object_name=object_name,
            field="probability",
            expected="finite value",
            actual=probability,
        )
    return probability, jitter, residual


class IndicatorKriging:
    """Ordinary kriging of threshold indicators on one shared neighbourhood.

    Args:
        variograms: one covariance model per threshold.
        thresholds: indicator cutoffs.
        property: output :class:`PropertyMetadata` (name = data column).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        conditioning_policy: how conditioning data conflicts are handled.
        solve_policy: singular-system handling policy.
    """

    def __init__(
        self,
        variograms: Sequence[CovarianceModel],
        thresholds: Sequence[float],
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec | None = None,
        conditioning_policy: ConditioningPolicy = ConditioningPolicy(),
        solve_policy: KrigingSolvePolicy = KrigingSolvePolicy(),
    ) -> None:
        models = tuple(variograms)
        cutoffs = tuple(float(value) for value in thresholds)
        representative = models[0] if models else None
        _validate_common(
            representative,
            property,
            neighbourhood,
            conditioning_policy,
            solve_policy,
            object_name=type(self).__name__,
        )
        if len(models) != len(cutoffs) or len(cutoffs) < 2:
            raise GeomodelContractError(
                "indicator kriging requires at least two aligned thresholds and models",
                object_name=type(self).__name__,
                field="variograms/thresholds",
                expected="equal lengths >= 2",
                actual=(len(models), len(cutoffs)),
            )
        if any(not isinstance(model, CovarianceModel) for model in models):
            raise GeomodelContractError(
                "indicator models must be covariance models",
                object_name=type(self).__name__,
                field="variograms",
                expected="CovarianceModel sequence",
                actual=[type(model).__name__ for model in models],
            )
        if any(not math.isfinite(value) for value in cutoffs) or any(
            right <= left for left, right in zip(cutoffs, cutoffs[1:])
        ):
            raise GeomodelContractError(
                "indicator thresholds must be finite and strictly increasing",
                object_name=type(self).__name__,
                field="thresholds",
                expected="strictly increasing finite values",
                actual=cutoffs,
            )
        self.variograms = models
        self.thresholds = cutoffs
        self.property = property
        self.neighbourhood = neighbourhood
        self.conditioning_policy = conditioning_policy
        self.solve_policy = solve_policy

    def __call__(self, data: GeoFrame | ConditioningSet, domain: Any) -> GeoFrame:
        coords, values, geometry, targets = _hard_arrays(
            data,
            domain,
            self.property,
            self.conditioning_policy,
            object_name=type(self).__name__,
        )
        for model in self.variograms:
            model.require_stationary_covariance(object_name=type(self).__name__)
        ndim = int(coords.shape[1])
        spec = self.neighbourhood or default_neighbourhood(self.variograms[0], ndim)
        if spec.ndim != ndim:
            raise GeomodelContractError(
                "indicator neighbourhood dimension does not match the domain",
                object_name=type(self).__name__,
                field="neighbourhood",
                expected=f"{ndim}-D",
                actual=f"{spec.ndim}-D",
            )
        index = StaticKDTreeNeighbourhood.from_arrays(
            coords, np.arange(coords.shape[0], dtype=np.int64)
        )
        probabilities = np.empty((len(self.thresholds), targets.shape[0]), dtype=np.float64)
        estimates = np.empty(targets.shape[0], dtype=np.float64)
        variances = np.empty(targets.shape[0], dtype=np.float64)
        selected: list[tuple[int, ...]] = []
        checks = 0
        jitters: list[float] = []
        residuals: list[float] = []
        correction_count = 0
        encoded = tuple(
            as_float_array((values <= threshold).astype(np.float64))
            for threshold in self.thresholds
        )

        for target_index, target in enumerate(targets):
            selection = index.query(target, spec)
            ids = tuple(int(value) for value in selection.ids.tolist())
            selected.append(ids)
            checks += selection.distance_checks
            if selection.status == "insufficient":
                raise GeomodelNumericsError(
                    "indicator-kriging neighbourhood is insufficient",
                    object_name=type(self).__name__,
                    field="neighbourhood",
                    expected=f"at least {spec.min_neighbors} neighbours",
                    actual=len(ids),
                )
            row_ids = np.asarray(selection.ids, dtype=np.int64)
            raw: FloatArray = as_float_array(np.empty(len(self.thresholds), dtype=np.float64))
            target_jitter = 0.0
            target_residual = 0.0
            for cutoff_index, model in enumerate(self.variograms):
                probability, jitter, residual = _indicator_system(
                    model,
                    coords[row_ids],
                    target,
                    encoded[cutoff_index][row_ids],
                    self.solve_policy,
                    object_name=type(self).__name__,
                )
                raw[cutoff_index] = probability
                target_jitter = max(target_jitter, jitter)
                target_residual = max(target_residual, residual)
            corrected, corrections = _correct_cdf_with_count(as_float_array(raw))
            correction_count += corrections
            probabilities[:, target_index] = corrected
            estimates[target_index] = _etype_mean(self.thresholds, corrected)
            median_probability = float(corrected[len(self.thresholds) // 2])
            variances[target_index] = median_probability * (1.0 - median_probability)
            jitters.append(target_jitter)
            residuals.append(target_residual)

        diagnostics: dict[str, object] = {
            "coordinate_dimension": ndim,
            "neighbourhood_backend": "StaticKDTreeNeighbourhood",
            "selected_ids": tuple(selected),
            "distance_checks": checks,
            "jitter_relative": tuple(jitters),
            "solve_residual": tuple(residuals),
            "constraint_residual": tuple(0.0 for _ in targets),
            "cdf_corrections": correction_count,
        }
        probability_properties = {
            f"prob_{index}": as_float_array(probabilities[index])
            for index in range(len(self.thresholds))
        }
        probability_metadata = {
            name: PropertyMetadata(name, "probability", "1") for name in probability_properties
        }
        return make_estimation_result(
            geometry,
            as_float_array(estimates),
            as_float_array(variances),
            property=self.property,
            diagnostics=diagnostics,
            extra_properties=probability_properties,
            extra_metadata=probability_metadata,
        )

    def __repr__(self) -> str:
        return f"IndicatorKriging(thresholds={self.thresholds!r})"
