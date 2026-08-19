"""
Shared domain-resolution helpers for numpy geomodel estimators.

The high-level kriging and simulation classes accept the same target-domain
shapes: a GeoFrame, a Geometry, or a raw ``(m, 2|3)`` coordinate array. This
module keeps that contract in one place so individual algorithms can focus on
their geostatistical kernel.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np

from ...core import GeoBrainError
from ..frames._arrays import FloatArray, as_float_array
from ..frames import ColumnRole, GeoFrame, Geometry, GeoPoints, PropertyMetadata
from ..errors import GeomodelContractError
from .models.covariance import CovarianceModel


def coords_3d(coords: object, *, object_name: str = "geomodel") -> FloatArray:
    """Return coordinates as ``(n, 3)`` float64, padding 2-D inputs with z=0."""
    arr = as_float_array(coords)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise GeoBrainError(
            "coords must be (n, 2) or (n, 3)",
            object_name=object_name,
            field="coords",
            expected="(n, 2) or (n, 3)",
            actual=tuple(arr.shape),
        )
    if arr.shape[1] == 2:
        arr = as_float_array(np.column_stack([arr, np.zeros(arr.shape[0], dtype=np.float64)]))
    return arr


def resolve_domain(
    domain: Any,
    *,
    object_name: str,
    preserve_dimension: bool = False,
) -> tuple[Geometry, FloatArray]:
    """
    Resolve a target domain.

    ``preserve_dimension=True`` is the 0.2 estimator contract. The temporary
    default 3-D embedding keeps pre-Task-5 simulation kernels internally
    aligned until their own dimension-aware migration; padding is therefore
    explicit and never used by the kriging estimators.

    Accepted inputs are:
    - ``GeoFrame``: use its geometry;
    - ``Geometry``: use it directly;
    - raw ``(m, 2|3)`` array-like: wrap as ``GeoPoints``.
    """
    if isinstance(domain, GeoFrame):
        coords = as_float_array(domain.geometry.coords)
        return domain.geometry, coords if preserve_dimension else coords_3d(coords)
    if isinstance(domain, Geometry):
        coords = as_float_array(domain.coords)
        return domain, coords if preserve_dimension else coords_3d(coords)
    try:
        arr = as_float_array(domain)
    except (TypeError, ValueError) as exc:
        raise GeoBrainError(
            "domain must be GeoFrame / Geometry / (m, 2|3) array",
            object_name=object_name,
            field="domain",
            expected="GeoFrame, Geometry, or (m, 2|3) array",
            actual=type(domain).__name__,
        ) from exc
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise GeoBrainError(
            "domain must be GeoFrame / Geometry / (m, 2|3) array",
            object_name=object_name,
            field="domain",
            expected="GeoFrame, Geometry, or (m, 2|3) array",
            actual=tuple(arr.shape),
        )
    return GeoPoints(arr), arr if preserve_dimension else coords_3d(arr)


def default_search_radius(
    variogram: CovarianceModel,
    search_radius: float | tuple[float, ...] | list[float] | None,
) -> float:
    """Resolve the search radius convention shared by kriging/simulation APIs."""
    if search_radius is not None:
        if isinstance(search_radius, (tuple, list)):
            return float(max(search_radius))
        return float(search_radius)
    if variogram.structures:
        return float(max(s.range_max for s in variogram.structures) * 3.0)
    return 1.0e21


def validate_property_column(data: GeoFrame, column: str | None, *, object_name: str) -> str:
    """Return the selected property column, defaulting to the first column."""
    if not data.columns:
        raise GeoBrainError(
            "data GeoFrame has no property columns",
            object_name=object_name,
            field="data.columns",
            expected="non-empty",
            actual=[],
        )
    col = column or data.columns[0]
    if col not in data.columns:
        raise GeoBrainError(
            f"data column {col!r} is missing",
            object_name=object_name,
            field="column",
            expected=f"one of {data.columns}",
            actual=col,
        )
    return col


class EstimationResult(GeoFrame):  # type: ignore[misc, unused-ignore]
    """GeoFrame estimate carrying owned, copy-on-read solver diagnostics."""

    def __init__(self, *args: Any, diagnostics: dict[str, object], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._estimation_diagnostics = copy.deepcopy(diagnostics)

    @property
    def diagnostics(self) -> dict[str, object]:
        return copy.deepcopy(self._estimation_diagnostics)


def _squared_unit(unit: str) -> str:
    return "1" if unit == "1" else f"({unit})^2"


def make_estimation_result(
    geometry: Geometry,
    estimates: FloatArray,
    variances: FloatArray,
    *,
    property: PropertyMetadata,
    diagnostics: dict[str, object],
    extra_properties: dict[str, FloatArray] | None = None,
    extra_metadata: dict[str, PropertyMetadata] | None = None,
) -> EstimationResult:
    """Build the standard kriging result table with estimate/variance roles."""
    estimate_metadata = PropertyMetadata("estimate", "continuous", property.unit)
    variance_metadata = PropertyMetadata(
        "variance", "continuous", _squared_unit(property.unit or "1")
    )
    properties = {"estimate": estimates, "variance": variances}
    metadata = {"estimate": estimate_metadata, "variance": variance_metadata}
    if extra_properties:
        properties.update(extra_properties)
    if extra_metadata:
        metadata.update(extra_metadata)
    out = EstimationResult(
        geometry,
        properties=properties,
        metadata=metadata,
        diagnostics=diagnostics,
    )
    out.set_role("estimate", ColumnRole.ESTIMATE)
    out.set_role("variance", ColumnRole.VARIANCE)
    return out


def derive_realization_seeds(seed: int | None, count: int) -> np.ndarray:
    """Derive ``count`` independent, reproducible per-realisation seeds.

    Every simulator splits one master ``seed`` into per-realisation seeds the
    same way so that results are reproducible and independent across
    realisations (and identical between serial and process-parallel runs).
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)):
        raise GeomodelContractError(
            "realization count must be an exact non-negative integer",
            object_name="derive_realization_seeds",
            field="count",
            expected="non-negative int",
            actual=count,
        )
    resolved_count = int(count)
    if resolved_count < 0:
        raise GeomodelContractError(
            "realization count must be non-negative",
            object_name="derive_realization_seeds",
            field="count",
            expected="non-negative int",
            actual=count,
        )
    # ``default_rng`` owns generator state, so this neither reads nor mutates
    # NumPy's process-global RandomState.  This exact draw expression is an
    # audited deterministic compatibility contract.
    master_rng = np.random.default_rng(seed)
    return cast(np.ndarray, master_rng.integers(0, 2**63 - 1, size=resolved_count))


def make_simulation_result(geometry: Geometry, sims: Any) -> list[GeoFrame]:
    """Build the standard simulation result tables, one per realisation.

    Each realisation array ``sim`` is wrapped in a single-column GeoFrame on
    ``geometry`` with the ``"simulation"`` role. This is the result-assembly
    tail shared by the single-property simulators (SGSIM, CoSGSIM, LUSIM,
    SISIM, DSSIM, FFTMA, DirectSampling, ImageQuilting, FilterSim, SNESIM).
    """
    realisations: list[GeoFrame] = []
    for sim in sims:
        gt = GeoFrame(geometry, properties={"simulation": sim})
        gt.set_role("simulation", ColumnRole.SIMULATION)
        realisations.append(gt)
    return realisations


__all__ = [
    "coords_3d",
    "default_search_radius",
    "derive_realization_seeds",
    "EstimationResult",
    "make_estimation_result",
    "make_simulation_result",
    "resolve_domain",
    "validate_property_column",
]
