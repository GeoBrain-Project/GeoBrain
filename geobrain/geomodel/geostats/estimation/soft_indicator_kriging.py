"""Soft indicator kriging for declared probability properties.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from ...conditioning import ConditioningPolicy
from ...frames import GeoFrame, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError, GeomodelNumericsError
from ...neighbourhood import NeighbourhoodSpec, StaticKDTreeNeighbourhood
from .._domain import make_estimation_result, resolve_domain
from ..models.covariance import CovarianceModel
from .indicator_kriging import _correct_cdf_with_count, _etype_mean, _indicator_system
from .kriging import _hard_arrays, _validate_common
from .kriging_kernel import KrigingSolvePolicy, default_neighbourhood

__all__ = ["SoftIndicatorKriging"]


class SoftIndicatorKriging:
    """Kriging of ordered soft-CDF columns on one shared static index.

    Args:
        variograms: one covariance model per threshold.
        thresholds: indicator cutoffs.
        soft_columns: data columns carrying soft probabilities.
        property: output :class:`PropertyMetadata` (name = data column).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        conditioning_policy: how conditioning data conflicts are handled.
        solve_policy: singular-system handling policy.
    """

    def __init__(
        self,
        variograms: Sequence[CovarianceModel],
        thresholds: Sequence[float],
        soft_columns: Sequence[str],
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec | None = None,
        conditioning_policy: ConditioningPolicy = ConditioningPolicy(),
        solve_policy: KrigingSolvePolicy = KrigingSolvePolicy(),
    ) -> None:
        models = tuple(variograms)
        cutoffs = tuple(float(value) for value in thresholds)
        columns = tuple(str(value) for value in soft_columns)
        representative = models[0] if models else None
        _validate_common(
            representative,
            property,
            neighbourhood,
            conditioning_policy,
            solve_policy,
            object_name=type(self).__name__,
        )
        if len(models) != len(cutoffs) or len(columns) != len(cutoffs) or len(cutoffs) < 2:
            raise GeomodelContractError(
                "soft indicator inputs must align and contain at least two thresholds",
                object_name=type(self).__name__,
                field="variograms/thresholds/soft_columns",
                expected="equal lengths >= 2",
                actual=(len(models), len(cutoffs), len(columns)),
            )
        if any(not isinstance(model, CovarianceModel) for model in models):
            raise GeomodelContractError(
                "soft indicator models must be covariance models",
                object_name=type(self).__name__,
                field="variograms",
                expected="CovarianceModel sequence",
                actual=[type(model).__name__ for model in models],
            )
        if any(not math.isfinite(value) for value in cutoffs) or any(
            right <= left for left, right in zip(cutoffs, cutoffs[1:])
        ):
            raise GeomodelContractError(
                "soft indicator thresholds must be finite and strictly increasing",
                object_name=type(self).__name__,
                field="thresholds",
                expected="strictly increasing finite values",
                actual=cutoffs,
            )
        if any(not column for column in columns) or len(set(columns)) != len(columns):
            raise GeomodelContractError(
                "soft indicator columns must be non-empty and unique",
                object_name=type(self).__name__,
                field="soft_columns",
                expected="unique non-empty names",
                actual=columns,
            )
        self.variograms = models
        self.thresholds = cutoffs
        self.soft_columns = columns
        self.property = property
        self.neighbourhood = neighbourhood
        self.conditioning_policy = conditioning_policy
        self.solve_policy = solve_policy

    def __call__(self, data: GeoFrame, domain: Any) -> GeoFrame:
        if not isinstance(data, GeoFrame):
            raise GeomodelContractError(
                "soft indicator conditioning must be a GeoFrame",
                object_name=type(self).__name__,
                field="data",
                expected="GeoFrame with probability metadata",
                actual=type(data).__name__,
            )
        geometry, targets = resolve_domain(
            domain, object_name=type(self).__name__, preserve_dimension=True
        )
        coordinate_sets: list[np.ndarray] = []
        soft_values: list[np.ndarray] = []
        for column in self.soft_columns:
            if column not in data:
                raise GeomodelContractError(
                    "soft probability column is missing",
                    object_name=type(self).__name__,
                    field="soft_columns",
                    expected=data.columns,
                    actual=column,
                )
            metadata = data.metadata_for(column)
            if metadata.kind != "probability":
                raise GeomodelContractError(
                    "soft indicator columns require probability metadata",
                    object_name=type(self).__name__,
                    field=column,
                    expected="probability",
                    actual=metadata.kind,
                )
            coords, values, _, _ = _hard_arrays(
                data,
                geometry,
                metadata,
                self.conditioning_policy,
                object_name=type(self).__name__,
            )
            coordinate_sets.append(coords)
            soft_values.append(values)
        coords = coordinate_sets[0]
        if any(not np.array_equal(coords, item) for item in coordinate_sets[1:]):
            raise GeomodelContractError(
                "normalized soft probability rows do not align",
                object_name=type(self).__name__,
                field="soft_columns",
                expected="identical normalized coordinates",
                actual="column-specific rows differ",
            )
        matrix_values = np.stack(soft_values)
        if not np.all((matrix_values >= 0.0) & (matrix_values <= 1.0)):
            raise GeomodelContractError(
                "soft probabilities must lie in the closed unit interval",
                object_name=type(self).__name__,
                field="soft_columns",
                expected="values in [0, 1]",
                actual={
                    "minimum": float(matrix_values.min()),
                    "maximum": float(matrix_values.max()),
                },
            )
        order_violations = np.argwhere(np.diff(matrix_values, axis=0) < 0.0)
        if order_violations.size:
            cutoff_index, row_index = (int(value) for value in order_violations[0])
            raise GeomodelContractError(
                "soft probabilities must form a monotone conditioning CDF",
                object_name=type(self).__name__,
                field="soft_columns",
                expected="non-decreasing probabilities across ordered thresholds",
                actual={
                    "row": row_index,
                    "lower_column": self.soft_columns[cutoff_index],
                    "lower_probability": float(matrix_values[cutoff_index, row_index]),
                    "upper_column": self.soft_columns[cutoff_index + 1],
                    "upper_probability": float(matrix_values[cutoff_index + 1, row_index]),
                },
            )
        for model in self.variograms:
            model.require_stationary_covariance(object_name=type(self).__name__)
        ndim = int(coords.shape[1])
        spec = self.neighbourhood or default_neighbourhood(self.variograms[0], ndim)
        if spec.ndim != ndim:
            raise GeomodelContractError(
                "soft indicator neighbourhood dimension does not match the domain",
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

        for target_index, target in enumerate(targets):
            selection = index.query(target, spec)
            ids = tuple(int(value) for value in selection.ids.tolist())
            selected.append(ids)
            checks += selection.distance_checks
            if selection.status == "insufficient":
                raise GeomodelNumericsError(
                    "soft-indicator neighbourhood is insufficient",
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
                    as_float_array(coords[row_ids]),
                    as_float_array(target),
                    as_float_array(matrix_values[cutoff_index, row_ids]),
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
        return (
            f"SoftIndicatorKriging(thresholds={self.thresholds!r}, "
            f"soft_columns={self.soft_columns!r})"
        )
