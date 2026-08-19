"""Dimension-aware simple and ordinary block kriging.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np

from ...conditioning import ConditioningPolicy, ConditioningSet
from ...frames import GeoFrame, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError, GeomodelNumericsError
from ...neighbourhood import NeighbourhoodSpec, StaticKDTreeNeighbourhood
from .._domain import make_estimation_result
from ..models.covariance import CovarianceModel
from .covariance_matrix import covariance_matrix
from .kriging import _hard_arrays, _validate_common
from .kriging_kernel import (
    KrigingSolvePolicy,
    default_neighbourhood,
    solve_kriging_system,
    validated_variance,
)

__all__ = ["BlockKriging"]


def _block_offsets(size: tuple[float, ...], discretization: tuple[int, ...]) -> FloatArray:
    axes: list[np.ndarray] = []
    for width, count in zip(size, discretization, strict=True):
        if count == 1:
            axes.append(np.asarray([0.0], dtype=np.float64))
        else:
            half = width / 2.0
            axes.append(np.linspace(-half + half / count, half - half / count, count))
    return as_float_array(np.asarray(tuple(product(*axes)), dtype=np.float64))


class BlockKriging:
    """Discretized block-support kriging with explicit units and solve policy.

    Args:
        variogram: covariance/variogram model of the field.
        property: output :class:`PropertyMetadata` (name = data column).
        block_size_m: block dimensions [m].
        block_discretization: sub-points per block axis.
        kind: ``'ordinary'`` or ``'simple'``.
        mean: stationary mean for simple kriging.
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        conditioning_policy: how conditioning data conflicts are handled.
        solve_policy: singular-system handling policy.
    """

    def __init__(
        self,
        variogram: CovarianceModel,
        *,
        property: PropertyMetadata,
        block_size_m: tuple[float, ...],
        block_discretization: tuple[int, ...],
        kind: str = "ordinary",
        mean: float = 0.0,
        neighbourhood: NeighbourhoodSpec | None = None,
        conditioning_policy: ConditioningPolicy = ConditioningPolicy(),
        solve_policy: KrigingSolvePolicy = KrigingSolvePolicy(),
    ) -> None:
        _validate_common(
            variogram,
            property,
            neighbourhood,
            conditioning_policy,
            solve_policy,
            object_name=type(self).__name__,
        )
        try:
            size = tuple(float(value) for value in block_size_m)
            discretization = tuple(int(value) for value in block_discretization)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "block support must be numeric",
                object_name=type(self).__name__,
                field="block_size_m/block_discretization",
                expected="dimension-matched tuples",
                actual=(block_size_m, block_discretization),
            ) from exc
        if (
            len(size) not in (2, 3)
            or len(discretization) != len(size)
            or any(not math.isfinite(value) or value <= 0.0 for value in size)
            or any(value < 1 for value in discretization)
        ):
            raise GeomodelContractError(
                "block support is invalid",
                object_name=type(self).__name__,
                field="block_size_m/block_discretization",
                expected="positive 2-D or 3-D size and integer counts >= 1",
                actual=(size, discretization),
            )
        if kind not in ("simple", "ordinary"):
            raise GeomodelContractError(
                "block-kriging kind is invalid",
                object_name=type(self).__name__,
                field="kind",
                expected="'simple' or 'ordinary'",
                actual=kind,
            )
        try:
            resolved_mean = float(mean)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "block-kriging mean must be finite",
                object_name=type(self).__name__,
                field="mean",
                expected="finite float",
                actual=mean,
            ) from exc
        if not math.isfinite(resolved_mean):
            raise GeomodelContractError(
                "block-kriging mean must be finite",
                object_name=type(self).__name__,
                field="mean",
                expected="finite float",
                actual=mean,
            )
        self.variogram = variogram
        self.property = property
        self.block_size_m = size
        self.block_discretization = discretization
        self.kind = kind
        self.mean = resolved_mean
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
        ndim = int(coords.shape[1])
        if len(self.block_size_m) != ndim:
            raise GeomodelContractError(
                "block support dimension does not match the domain",
                object_name=type(self).__name__,
                field="block_size_m",
                expected=f"{ndim} values",
                actual=self.block_size_m,
            )
        self.variogram.require_stationary_covariance(object_name=type(self).__name__)
        spec = self.neighbourhood or default_neighbourhood(self.variogram, ndim)
        if spec.ndim != ndim:
            raise GeomodelContractError(
                "block neighbourhood dimension does not match the domain",
                object_name=type(self).__name__,
                field="neighbourhood",
                expected=f"{ndim}-D",
                actual=f"{spec.ndim}-D",
            )
        offsets = _block_offsets(self.block_size_m, self.block_discretization)
        c_vv = float(np.mean(covariance_matrix(self.variogram, offsets, offsets)))
        index = StaticKDTreeNeighbourhood.from_arrays(
            coords, np.arange(coords.shape[0], dtype=np.int64)
        )
        estimates = np.empty(targets.shape[0], dtype=np.float64)
        variances = np.empty(targets.shape[0], dtype=np.float64)
        selected: list[tuple[int, ...]] = []
        checks = 0
        jitters: list[float] = []
        residuals: list[float] = []
        constraints: list[float] = []
        ordinary = self.kind == "ordinary"

        for target_index, target in enumerate(targets):
            selection = index.query(target, spec)
            ids = tuple(int(value) for value in selection.ids.tolist())
            selected.append(ids)
            checks += selection.distance_checks
            if selection.status == "insufficient":
                if not ordinary:
                    estimates[target_index] = self.mean
                    variances[target_index] = c_vv
                    jitters.append(0.0)
                    residuals.append(0.0)
                    constraints.append(0.0)
                    continue
                raise GeomodelNumericsError(
                    "block-kriging neighbourhood is insufficient",
                    object_name=type(self).__name__,
                    field="neighbourhood",
                    expected=f"at least {spec.min_neighbors} neighbours",
                    actual=len(ids),
                )
            row_ids = np.asarray(selection.ids, dtype=np.int64)
            nearby = coords[row_ids]
            covariance_dd = covariance_matrix(self.variogram, nearby, nearby)
            block_points = target + offsets
            covariance_db = as_float_array(
                np.mean(covariance_matrix(self.variogram, nearby, block_points), axis=1)
            )
            count = int(row_ids.size)
            extra = 1 if ordinary else 0
            matrix: FloatArray = as_float_array(
                np.zeros((count + extra, count + extra), dtype=np.float64)
            )
            rhs: FloatArray = as_float_array(np.zeros(count + extra, dtype=np.float64))
            matrix[:count, :count] = covariance_dd
            rhs[:count] = covariance_db
            if ordinary:
                matrix[count, :count] = 1.0
                matrix[:count, count] = 1.0
                rhs[count] = 1.0
            solution, jitter, residual = solve_kriging_system(
                as_float_array(matrix),
                as_float_array(rhs),
                data_count=count,
                policy=self.solve_policy,
                object_name=type(self).__name__,
            )
            weights = solution[:count]
            estimates[target_index] = (
                float(np.dot(weights, values[row_ids]))
                if ordinary
                else float(self.mean + np.dot(weights, values[row_ids] - self.mean))
            )
            variance = c_vv - float(np.dot(weights, covariance_db))
            if ordinary:
                variance -= float(solution[count])
            variances[target_index] = validated_variance(
                variance,
                sill=float(self.variogram.sill),
                object_name=type(self).__name__,
            )
            jitters.append(jitter)
            residuals.append(residual)
            constraints.append(0.0 if not ordinary else abs(float(np.sum(weights)) - 1.0))

        diagnostics: dict[str, object] = {
            "coordinate_dimension": ndim,
            "neighbourhood_backend": "StaticKDTreeNeighbourhood",
            "selected_ids": tuple(selected),
            "distance_checks": checks,
            "jitter_relative": tuple(jitters),
            "solve_residual": tuple(residuals),
            "constraint_residual": tuple(constraints),
            "block_discretization": self.block_discretization,
        }
        return make_estimation_result(
            geometry,
            as_float_array(estimates),
            as_float_array(variances),
            property=self.property,
            diagnostics=diagnostics,
        )

    def __repr__(self) -> str:
        return (
            f"BlockKriging(kind={self.kind!r}, block_size_m={self.block_size_m!r}, "
            f"block_discretization={self.block_discretization!r})"
        )
