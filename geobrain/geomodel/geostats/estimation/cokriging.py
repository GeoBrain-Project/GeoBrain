"""Dimension-aware MM1/MM2 collocated cokriging.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...conditioning import ConditioningPolicy
from ...frames import GeoFrame, PropertyMetadata
from ...frames._arrays import as_float_array
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
    validated_variance,
)

__all__ = ["CollocatedCokriging"]


class CollocatedCokriging:
    """Markov-model collocated cokriging with explicit property identities.

    Args:
        variogram: covariance/variogram model of the field.
        correlation: primary-secondary correlation coefficient.
        primary_property / secondary_property: the two properties.
        secondary_sill: secondary variance (``None`` = from data).
        kind: ``'ordinary'`` or ``'simple'``.
        markov_model: Markov screening assumption id.
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        conditioning_policy: how conditioning data conflicts are handled.
        solve_policy: singular-system handling policy.
    """

    def __init__(
        self,
        variogram: CovarianceModel,
        correlation: float,
        *,
        primary_property: PropertyMetadata,
        secondary_property: PropertyMetadata,
        secondary_sill: float | None = None,
        kind: str = "ordinary",
        markov_model: str = "mm1",
        secondary_variogram: CovarianceModel | None = None,
        primary_mean: float = 0.0,
        secondary_mean: float = 0.0,
        neighbourhood: NeighbourhoodSpec | None = None,
        conditioning_policy: ConditioningPolicy = ConditioningPolicy(),
        solve_policy: KrigingSolvePolicy = KrigingSolvePolicy(),
    ) -> None:
        _validate_common(
            variogram,
            primary_property,
            neighbourhood,
            conditioning_policy,
            solve_policy,
            object_name=type(self).__name__,
        )
        if (
            not isinstance(secondary_property, PropertyMetadata)
            or secondary_property.kind != "continuous"
        ):
            raise GeomodelContractError(
                "secondary property must be continuous metadata",
                object_name=type(self).__name__,
                field="secondary_property",
                expected="continuous PropertyMetadata",
                actual=secondary_property,
            )
        rho = float(correlation)
        if not math.isfinite(rho) or not -1.0 <= rho <= 1.0:
            raise GeomodelContractError(
                "cokriging correlation is invalid",
                object_name=type(self).__name__,
                field="correlation",
                expected="finite value in [-1, 1]",
                actual=correlation,
            )
        if kind not in ("simple", "ordinary"):
            raise GeomodelContractError(
                "cokriging kind is invalid",
                object_name=type(self).__name__,
                field="kind",
                expected="'simple' or 'ordinary'",
                actual=kind,
            )
        if markov_model not in ("mm1", "mm2"):
            raise GeomodelContractError(
                "cokriging Markov model is invalid",
                object_name=type(self).__name__,
                field="markov_model",
                expected="'mm1' or 'mm2'",
                actual=markov_model,
            )
        if markov_model == "mm2" and not isinstance(secondary_variogram, CovarianceModel):
            raise GeomodelContractError(
                "MM2 requires a secondary covariance model",
                object_name=type(self).__name__,
                field="secondary_variogram",
                expected="CovarianceModel",
                actual=type(secondary_variogram).__name__,
            )
        resolved_secondary_sill = (
            float(secondary_variogram.sill)  # type: ignore[union-attr]
            if markov_model == "mm2"
            else float(variogram.sill if secondary_sill is None else secondary_sill)
        )
        if not math.isfinite(resolved_secondary_sill) or resolved_secondary_sill <= 0.0:
            raise GeomodelContractError(
                "secondary sill must be positive and finite",
                object_name=type(self).__name__,
                field="secondary_sill",
                expected="> 0 and finite",
                actual=resolved_secondary_sill,
            )
        try:
            resolved_primary_mean = float(primary_mean)
            resolved_secondary_mean = float(secondary_mean)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "cokriging means must be finite",
                object_name=type(self).__name__,
                field="primary_mean/secondary_mean",
                expected="finite floats",
                actual=(primary_mean, secondary_mean),
            ) from exc
        if not math.isfinite(resolved_primary_mean) or not math.isfinite(resolved_secondary_mean):
            raise GeomodelContractError(
                "cokriging means must be finite",
                object_name=type(self).__name__,
                field="primary_mean/secondary_mean",
                expected="finite floats",
                actual=(primary_mean, secondary_mean),
            )
        self.variogram = variogram
        self.correlation = rho
        self.primary_property = primary_property
        self.secondary_property = secondary_property
        self.secondary_sill = resolved_secondary_sill
        self.kind = kind
        self.markov_model = markov_model
        self.secondary_variogram = secondary_variogram
        self.primary_mean = resolved_primary_mean
        self.secondary_mean = resolved_secondary_mean
        self.neighbourhood = neighbourhood
        self.conditioning_policy = conditioning_policy
        self.solve_policy = solve_policy

    def __call__(self, data: GeoFrame, domain: Any) -> GeoFrame:
        if not isinstance(data, GeoFrame):
            raise GeomodelContractError(
                "collocated cokriging requires a GeoFrame",
                object_name=type(self).__name__,
                field="data",
                expected="GeoFrame",
                actual=type(data).__name__,
            )
        coords, primary_values, geometry, targets = _hard_arrays(
            data,
            domain,
            self.primary_property,
            self.conditioning_policy,
            object_name=type(self).__name__,
        )
        secondary_coords, secondary_values, _, _ = _hard_arrays(
            data,
            geometry,
            self.secondary_property,
            self.conditioning_policy,
            object_name=type(self).__name__,
        )
        if not np.array_equal(coords, secondary_coords):
            raise GeomodelContractError(
                "primary and secondary conditioning rows do not align",
                object_name=type(self).__name__,
                field="data",
                expected="collocated complete primary/secondary observations",
                actual="normalized coordinates differ",
            )
        if isinstance(domain, GeoFrame) and self.secondary_property.name in domain:
            if domain.metadata_for(self.secondary_property.name) != self.secondary_property:
                raise GeomodelContractError(
                    "target secondary metadata does not match the estimator",
                    object_name=type(self).__name__,
                    field="secondary_property",
                    expected=self.secondary_property.to_dict(),
                    actual=domain.metadata_for(self.secondary_property.name).to_dict(),
                )
            target_secondary = as_float_array(domain[self.secondary_property.name])
        else:
            target_secondary = as_float_array(
                np.full(targets.shape[0], float(np.mean(secondary_values)), dtype=np.float64)
            )

        self.variogram.require_stationary_covariance(object_name=type(self).__name__)
        if self.secondary_variogram is not None:
            self.secondary_variogram.require_stationary_covariance(object_name=type(self).__name__)
        ndim = int(coords.shape[1])
        spec = self.neighbourhood or default_neighbourhood(self.variogram, ndim)
        if spec.ndim != ndim:
            raise GeomodelContractError(
                "cokriging neighbourhood dimension does not match the domain",
                object_name=type(self).__name__,
                field="neighbourhood",
                expected=f"{ndim}-D",
                actual=f"{spec.ndim}-D",
            )

        sill_primary = float(self.variogram.sill)
        sigma_primary = math.sqrt(sill_primary)
        sigma_secondary = math.sqrt(self.secondary_sill)
        cross_zero = self.correlation * sigma_primary * sigma_secondary
        cross_model: CovarianceModel
        if self.markov_model == "mm1":
            cross_model = self.variogram
            cross_scale = self.correlation * sigma_secondary / sigma_primary
        else:
            assert self.secondary_variogram is not None
            cross_model = self.secondary_variogram
            cross_scale = self.correlation * sigma_primary / sigma_secondary
        secondary_data_mean = float(np.mean(secondary_values))
        ordinary = self.kind == "ordinary"
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

        for target_index, target in enumerate(targets):
            selection = index.query(target, spec)
            ids = tuple(int(value) for value in selection.ids.tolist())
            selected.append(ids)
            checks += selection.distance_checks
            if selection.status == "insufficient":
                if ordinary:
                    raise GeomodelNumericsError(
                        "cokriging neighbourhood is insufficient",
                        object_name=type(self).__name__,
                        field="neighbourhood",
                        expected=f"at least {spec.min_neighbors} neighbours",
                        actual=len(ids),
                    )
                estimates[target_index] = self.primary_mean
                variances[target_index] = sill_primary
                jitters.append(0.0)
                residuals.append(0.0)
                constraints.append(0.0)
                continue
            row_ids = np.asarray(selection.ids, dtype=np.int64)
            nearby = coords[row_ids]
            covariance_dd = covariance_matrix(self.variogram, nearby, nearby)
            covariance_dt = covariance_vector(self.variogram, nearby, target)
            cross_dt = cross_scale * covariance_vector(cross_model, nearby, target)
            count = int(row_ids.size)
            data_count = count + 1
            extra = 1 if ordinary else 0
            matrix = as_float_array(
                np.zeros((data_count + extra, data_count + extra), dtype=np.float64)
            )
            rhs = as_float_array(np.zeros(data_count + extra, dtype=np.float64))
            matrix[:count, :count] = covariance_dd
            matrix[:count, count] = cross_dt
            matrix[count, :count] = cross_dt
            matrix[count, count] = self.secondary_sill
            rhs[:count] = covariance_dt
            rhs[count] = cross_zero
            if ordinary:
                matrix[data_count, :count] = 1.0
                matrix[:count, data_count] = 1.0
                rhs[data_count] = 1.0
            solution, jitter, residual = solve_kriging_system(
                as_float_array(matrix),
                as_float_array(rhs),
                data_count=data_count,
                policy=self.solve_policy,
                object_name=type(self).__name__,
            )
            primary_weights = solution[:count]
            secondary_weight = float(solution[count])
            secondary_residual = float(target_secondary[target_index]) - (
                secondary_data_mean if ordinary else self.secondary_mean
            )
            estimates[target_index] = (
                float(
                    np.dot(primary_weights, primary_values[row_ids])
                    + secondary_weight * secondary_residual
                )
                if ordinary
                else float(
                    self.primary_mean
                    + np.dot(primary_weights, primary_values[row_ids] - self.primary_mean)
                    + secondary_weight * secondary_residual
                )
            )
            variance = sill_primary - float(np.dot(solution[:data_count], rhs[:data_count]))
            if ordinary:
                variance -= float(solution[data_count])
            variances[target_index] = validated_variance(
                variance,
                sill=sill_primary,
                object_name=type(self).__name__,
            )
            jitters.append(jitter)
            residuals.append(residual)
            constraints.append(0.0 if not ordinary else abs(float(np.sum(primary_weights)) - 1.0))

        diagnostics: dict[str, object] = {
            "coordinate_dimension": ndim,
            "neighbourhood_backend": "StaticKDTreeNeighbourhood",
            "selected_ids": tuple(selected),
            "distance_checks": checks,
            "jitter_relative": tuple(jitters),
            "solve_residual": tuple(residuals),
            "constraint_residual": tuple(constraints),
            "markov_model": self.markov_model,
        }
        return make_estimation_result(
            geometry,
            as_float_array(estimates),
            as_float_array(variances),
            property=self.primary_property,
            diagnostics=diagnostics,
        )

    def __repr__(self) -> str:
        return (
            f"CollocatedCokriging(kind={self.kind!r}, correlation={self.correlation:.3f}, "
            f"primary={self.primary_property.name!r}, secondary={self.secondary_property.name!r})"
        )
