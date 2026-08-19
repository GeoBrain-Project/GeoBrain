"""Simple, ordinary, and universal kriging scientific interfaces.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from ...conditioning import ConditioningPolicy, ConditioningSet, normalize_conditioning
from ...frames import GeoFrame, Geometry, PropertyMetadata
from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError
from ...neighbourhood import NeighbourhoodSpec
from .._domain import make_estimation_result, resolve_domain
from ..models.covariance import CovarianceModel
from .drift import ALLOWED_DRIFT_TERMS
from .kriging_kernel import (
    KrigingSolvePolicy,
    default_neighbourhood,
    krige_loop,
)

__all__ = ["KrigingSolvePolicy", "SimpleKriging", "OrdinaryKriging", "UniversalKriging"]


def _validate_common(
    model: object,
    property: object,
    neighbourhood: object,
    conditioning_policy: object,
    solve_policy: object,
    *,
    object_name: str,
) -> None:
    checks = (
        (model, CovarianceModel, "variogram"),
        (property, PropertyMetadata, "property"),
        (conditioning_policy, ConditioningPolicy, "conditioning_policy"),
        (solve_policy, KrigingSolvePolicy, "solve_policy"),
    )
    for value, expected_type, field in checks:
        if not isinstance(value, expected_type):
            raise GeomodelContractError(
                "invalid kriging configuration",
                object_name=object_name,
                field=field,
                expected=expected_type.__name__,
                actual=type(value).__name__,
            )
    property_metadata = cast(PropertyMetadata, property)
    if property_metadata.kind != "continuous":
        raise GeomodelContractError(
            "point kriging requires a continuous property",
            object_name=object_name,
            field="property.kind",
            expected="continuous",
            actual=property_metadata.kind,
        )
    if neighbourhood is not None and not isinstance(neighbourhood, NeighbourhoodSpec):
        raise GeomodelContractError(
            "invalid kriging neighbourhood",
            object_name=object_name,
            field="neighbourhood",
            expected="NeighbourhoodSpec or None",
            actual=type(neighbourhood).__name__,
        )


def _hard_arrays(
    data: GeoFrame | ConditioningSet,
    domain: Any,
    property: PropertyMetadata,
    policy: ConditioningPolicy,
    *,
    object_name: str,
) -> tuple[FloatArray, FloatArray, Geometry, FloatArray]:
    geometry, targets = resolve_domain(domain, object_name=object_name, preserve_dimension=True)
    normalized = normalize_conditioning(data, geometry, property, policy)
    hard = normalized.hard_values
    if hard is None:
        raise GeomodelContractError(
            "kriging requires hard conditioning values",
            object_name=object_name,
            field="data",
            expected=f"hard observations for {property.name!r}",
            actual=None,
        )
    present = ~np.ma.getmaskarray(hard)
    if not np.any(present):
        raise GeomodelContractError(
            "kriging requires at least one hard conditioning value",
            object_name=object_name,
            field="data",
            expected="one or more observations",
            actual=0,
        )
    coords = as_float_array(np.asarray(normalized.coordinates_m)[present])
    values = as_float_array(np.asarray(np.ma.getdata(hard))[present])
    return coords, values, geometry, targets


class _PointKriging:
    _ktype: int

    def __init__(
        self,
        variogram: CovarianceModel,
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec | None,
        conditioning_policy: ConditioningPolicy,
        solve_policy: KrigingSolvePolicy,
    ) -> None:
        _validate_common(
            variogram,
            property,
            neighbourhood,
            conditioning_policy,
            solve_policy,
            object_name=type(self).__name__,
        )
        self.variogram = variogram
        self.property = property
        self.neighbourhood = neighbourhood
        self.conditioning_policy = conditioning_policy
        self.solve_policy = solve_policy

    def _run(
        self,
        data: GeoFrame | ConditioningSet,
        domain: Any,
        *,
        mean: float = 0.0,
        drift_terms: tuple[str, ...] = (),
    ) -> GeoFrame:
        coords, values, geometry, targets = _hard_arrays(
            data,
            domain,
            self.property,
            self.conditioning_policy,
            object_name=type(self).__name__,
        )
        spec = self.neighbourhood or default_neighbourhood(self.variogram, coords.shape[1])
        estimates, variances, diagnostics = krige_loop(
            coords,
            values,
            targets,
            self.variogram,
            ktype=self._ktype,
            mean=mean,
            drift_terms=drift_terms,
            neighbourhood=spec,
            solve_policy=self.solve_policy,
            object_name=type(self).__name__,
        )
        return make_estimation_result(
            geometry,
            estimates,
            variances,
            property=self.property,
            diagnostics=diagnostics,
        )


class SimpleKriging(_PointKriging):
    """Simple kriging with one explicit continuous-property contract.

    Args:
        variogram: covariance/variogram model of the field.
        property: output :class:`PropertyMetadata` (name = data column).
        mean: known stationary mean.
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        conditioning_policy: how conditioning data conflicts are handled.
        solve_policy: singular-system handling policy.
    """

    _ktype = 0

    def __init__(
        self,
        variogram: CovarianceModel,
        *,
        property: PropertyMetadata,
        mean: float = 0.0,
        neighbourhood: NeighbourhoodSpec | None = None,
        conditioning_policy: ConditioningPolicy = ConditioningPolicy(),
        solve_policy: KrigingSolvePolicy = KrigingSolvePolicy(),
    ) -> None:
        super().__init__(
            variogram,
            property=property,
            neighbourhood=neighbourhood,
            conditioning_policy=conditioning_policy,
            solve_policy=solve_policy,
        )
        try:
            self.mean = float(mean)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "simple-kriging mean must be numeric",
                object_name=type(self).__name__,
                field="mean",
                expected="finite float",
                actual=mean,
            ) from exc
        if not math.isfinite(self.mean):
            raise GeomodelContractError(
                "simple-kriging mean must be finite",
                object_name=type(self).__name__,
                field="mean",
                expected="finite float",
                actual=mean,
            )

    def __call__(self, data: GeoFrame | ConditioningSet, domain: Any) -> GeoFrame:
        return self._run(data, domain, mean=self.mean)

    def __repr__(self) -> str:
        return f"SimpleKriging(mean={self.mean}, variogram={self.variogram!r})"


class OrdinaryKriging(_PointKriging):
    """Ordinary kriging with an exact sum-to-one constraint.

    Args:
        variogram: covariance/variogram model of the field.
        property: output :class:`PropertyMetadata` (name = data column).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        conditioning_policy: how conditioning data conflicts are handled.
        solve_policy: singular-system handling policy.
    """

    _ktype = 1

    def __init__(
        self,
        variogram: CovarianceModel,
        *,
        property: PropertyMetadata,
        neighbourhood: NeighbourhoodSpec | None = None,
        conditioning_policy: ConditioningPolicy = ConditioningPolicy(),
        solve_policy: KrigingSolvePolicy = KrigingSolvePolicy(),
    ) -> None:
        super().__init__(
            variogram,
            property=property,
            neighbourhood=neighbourhood,
            conditioning_policy=conditioning_policy,
            solve_policy=solve_policy,
        )

    def __call__(self, data: GeoFrame | ConditioningSet, domain: Any) -> GeoFrame:
        return self._run(data, domain)

    def __repr__(self) -> str:
        return f"OrdinaryKriging(variogram={self.variogram!r})"


class UniversalKriging(_PointKriging):
    """Universal kriging with declared polynomial drift constraints.

    Args:
        variogram: covariance/variogram model of the field.
        property: output :class:`PropertyMetadata` (name = data column).
        drift_terms: polynomial drift terms (e.g. ``('x', 'y')``).
        neighbourhood: search-neighbourhood spec (``None`` = exhaustive).
        conditioning_policy: how conditioning data conflicts are handled.
        solve_policy: singular-system handling policy.
    """

    _ktype = 2

    def __init__(
        self,
        variogram: CovarianceModel,
        *,
        property: PropertyMetadata,
        drift_terms: tuple[str, ...] = ("x", "y"),
        neighbourhood: NeighbourhoodSpec | None = None,
        conditioning_policy: ConditioningPolicy = ConditioningPolicy(),
        solve_policy: KrigingSolvePolicy = KrigingSolvePolicy(),
    ) -> None:
        super().__init__(
            variogram,
            property=property,
            neighbourhood=neighbourhood,
            conditioning_policy=conditioning_policy,
            solve_policy=solve_policy,
        )
        terms = tuple(str(term).lower() for term in drift_terms)
        if not terms or any(term not in ALLOWED_DRIFT_TERMS for term in terms):
            raise GeomodelContractError(
                "universal-kriging drift terms are invalid",
                object_name=type(self).__name__,
                field="drift_terms",
                expected=f"non-empty subset of {sorted(ALLOWED_DRIFT_TERMS)}",
                actual=terms,
            )
        self.drift_terms = terms

    def __call__(self, data: GeoFrame | ConditioningSet, domain: Any) -> GeoFrame:
        return self._run(data, domain, drift_terms=self.drift_terms)

    def __repr__(self) -> str:
        return f"UniversalKriging(variogram={self.variogram!r}, drift_terms={self.drift_terms!r})"
