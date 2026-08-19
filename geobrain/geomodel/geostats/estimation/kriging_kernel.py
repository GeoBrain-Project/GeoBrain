"""Dimension-aware kriging systems with explicit numerical policy.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from ...frames._arrays import FloatArray, as_float_array
from ...errors import GeomodelContractError, GeomodelNumericsError
from ...neighbourhood import NeighbourhoodSpec, StaticKDTreeNeighbourhood
from ..models.covariance import CovarianceModel
from .covariance_matrix import covariance_matrix, covariance_vector
from .drift import _kriging_drift_basis

_DEKKER_SPLITTER = 134_217_729.0

__all__ = [
    "KrigingSolvePolicy",
    "constraint_count",
    "default_neighbourhood",
    "krige_loop",
    "solve_kriging_system",
    "validated_variance",
]


@dataclass(frozen=True, slots=True)
class KrigingSolvePolicy:
    """Explicit response to singular kriging systems.

    Attributes:
        on_singular: ``'error'`` / ``'jitter'`` singular-system handling.
        jitter_relative: relative jitter ladder tried when jittering.
    """

    on_singular: Literal["error", "stabilize"] = "error"
    jitter_relative: tuple[float, ...] = (1.0e-12, 1.0e-10, 1.0e-8)

    def __post_init__(self) -> None:
        if self.on_singular not in ("error", "stabilize"):
            raise GeomodelContractError(
                "invalid kriging singular-system policy",
                object_name=type(self).__name__,
                field="on_singular",
                expected="'error' or 'stabilize'",
                actual=self.on_singular,
            )
        try:
            jitter = tuple(float(value) for value in self.jitter_relative)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "kriging jitter schedule must be numeric",
                object_name=type(self).__name__,
                field="jitter_relative",
                expected="strictly increasing positive finite values",
                actual=self.jitter_relative,
            ) from exc
        if (
            not jitter
            or any(not math.isfinite(value) or value <= 0.0 for value in jitter)
            or any(right <= left for left, right in zip(jitter, jitter[1:]))
        ):
            raise GeomodelContractError(
                "kriging jitter schedule is invalid",
                object_name=type(self).__name__,
                field="jitter_relative",
                expected="strictly increasing positive finite values",
                actual=jitter,
            )
        object.__setattr__(self, "jitter_relative", jitter)


def constraint_count(ktype: int, drift_terms: tuple[str, ...]) -> int:
    """Return the exact number of Lagrange constraints."""
    if ktype == 0:
        return 0
    if ktype == 1:
        return 1
    if ktype == 2:
        return 1 + len(drift_terms)
    raise GeomodelContractError(
        "unsupported kriging type",
        object_name="constraint_count",
        field="ktype",
        expected="0, 1, or 2",
        actual=ktype,
    )


def default_neighbourhood(model: CovarianceModel, ndim: int) -> NeighbourhoodSpec:
    """Derive one dimension-matched search ellipsoid from model ranges."""
    if ndim not in (2, 3):
        raise GeomodelContractError(
            "kriging coordinates must be two- or three-dimensional",
            object_name="default_neighbourhood",
            field="ndim",
            expected="2 or 3",
            actual=ndim,
        )
    radius = max((item.range_max for item in model.structures), default=1.0e20) * 3.0
    if not math.isfinite(radius) or radius <= 0.0:
        radius = 1.0e20
    angles = (0.0,) if ndim == 2 else (0.0, 0.0, 0.0)
    return NeighbourhoodSpec((radius,) * ndim, angles)


def _validate_covariance_block(
    block: FloatArray,
    *,
    object_name: str,
) -> tuple[float, float, FloatArray]:
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise GeomodelNumericsError(
            "kriging covariance block must be square",
            object_name=object_name,
            field="covariance",
            expected="square matrix",
            actual=tuple(block.shape),
        )
    if not np.isfinite(block).all():
        raise GeomodelNumericsError(
            "kriging covariance block must be finite",
            object_name=object_name,
            field="covariance",
            expected="finite matrix",
            actual="contains NaN or infinity",
        )
    diagonal = as_float_array(np.diag(block))
    if np.any(diagonal <= 0.0):
        raise GeomodelNumericsError(
            "kriging covariance diagonal must be positive",
            object_name=object_name,
            field="covariance",
            expected="strictly positive marginal variances",
            actual=diagonal.tolist(),
        )
    standard_deviations = as_float_array(np.sqrt(diagonal))
    normalized = as_float_array(block / np.multiply.outer(standard_deviations, standard_deviations))
    symmetry_tolerance = 1.0e-12
    if not np.allclose(normalized, normalized.T, rtol=0.0, atol=symmetry_tolerance):
        raise GeomodelNumericsError(
            "kriging covariance block must be symmetric",
            object_name=object_name,
            field="covariance",
            expected=f"relative asymmetry <= {symmetry_tolerance:.3g}",
            actual=float(np.max(np.abs(normalized - normalized.T))),
        )
    try:
        eigenvalues = np.linalg.eigvalsh(normalized)
    except np.linalg.LinAlgError as exc:
        raise GeomodelNumericsError(
            "kriging covariance eigenvalue check failed",
            object_name=object_name,
            field="covariance",
            expected="finite positive-semidefinite covariance",
            actual="eigendecomposition failed",
        ) from exc
    tolerance = 1.0e-10
    minimum = float(eigenvalues[0])
    if not np.isfinite(eigenvalues).all() or minimum < -tolerance:
        raise GeomodelNumericsError(
            "kriging covariance is not admissible",
            object_name=object_name,
            field="covariance",
            expected=f"relative minimum eigenvalue >= {-tolerance:.3g}",
            actual=minimum,
        )
    return minimum, tolerance, standard_deviations


def _equilibrate_system(
    matrix: FloatArray,
    rhs: FloatArray,
    *,
    data_count: int,
    standard_deviations: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return a diagonally congruent, dimensionless kriging system."""
    scales = np.ones(matrix.shape[0], dtype=np.float64)
    scales[:data_count] = standard_deviations
    for column in range(data_count, matrix.shape[0]):
        constraint_scale = float(np.max(np.abs(matrix[:data_count, column]) / standard_deviations))
        if math.isfinite(constraint_scale) and constraint_scale > 0.0:
            scales[column] = constraint_scale
    scale_products = np.multiply.outer(scales, scales)
    normalized_matrix = as_float_array(matrix / scale_products)
    normalized_rhs = as_float_array(rhs / scales)
    return normalized_matrix, normalized_rhs, as_float_array(scales)


def _refine_solution(
    matrix: FloatArray,
    rhs: FloatArray,
    solution: FloatArray,
) -> FloatArray:
    """Apply bounded extended-precision residual refinement to a float64 solve."""
    extended = np.longdouble
    matrix_extended = np.asarray(matrix, dtype=extended)
    rhs_extended = np.asarray(rhs, dtype=extended)
    refined = np.asarray(solution, dtype=np.float64)
    for _ in range(3):
        residual = np.asarray(
            rhs_extended - matrix_extended @ np.asarray(refined, dtype=extended),
            dtype=np.float64,
        )
        if not np.any(residual):
            break
        try:
            correction = np.linalg.solve(matrix, residual)
        except np.linalg.LinAlgError:
            break
        candidate = refined + correction
        if np.array_equal(candidate, refined):
            break
        refined = candidate
    return as_float_array(refined)


def _accurate_dot(left: FloatArray, right: FloatArray) -> float:
    """Return a scale-safe compensated dot product of finite float64 vectors."""
    left_mantissa, left_exponent = np.frexp(left)
    right_mantissa, right_exponent = np.frexp(right)
    product = left_mantissa * right_mantissa
    left_split = _DEKKER_SPLITTER * left_mantissa
    left_high = left_split - (left_split - left_mantissa)
    left_low = left_mantissa - left_high
    right_split = _DEKKER_SPLITTER * right_mantissa
    right_high = right_split - (right_split - right_mantissa)
    right_low = right_mantissa - right_high
    product_error = (
        (left_high * right_high - product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    exponent = left_exponent + right_exponent
    nonzero = (product != 0.0) | (product_error != 0.0)
    if not np.any(nonzero):
        return 0.0
    maximum_exponent = int(np.max(exponent[nonzero]))
    scaled = math.fsum(
        [
            term
            for value, error, value_exponent in zip(product, product_error, exponent)
            for term in (
                math.ldexp(float(value), int(value_exponent) - maximum_exponent),
                math.ldexp(float(error), int(value_exponent) - maximum_exponent),
            )
        ]
    )
    if scaled == 0.0:
        # A complete leading cancellation can expose a component more than
        # 1074 binary exponents below the initial scale.  This rare bounded-size
        # fallback retains that component exactly.
        exact = sum(
            (
                Fraction.from_float(float(left_value)) * Fraction.from_float(float(right_value))
                for left_value, right_value in zip(left, right)
            ),
            start=Fraction(0),
        )
        return float(exact)
    mantissa, local_exponent = math.frexp(scaled)
    try:
        return math.ldexp(mantissa, maximum_exponent + local_exponent)
    except OverflowError:
        return math.copysign(math.inf, mantissa)


def solve_kriging_system(
    matrix: FloatArray,
    rhs: FloatArray,
    *,
    data_count: int,
    policy: KrigingSolvePolicy,
    object_name: str,
) -> tuple[FloatArray, float, float]:
    """Validate and solve one system, returning solution, jitter and residual."""
    a = as_float_array(matrix)
    b = as_float_array(rhs)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (a.shape[0],):
        raise GeomodelNumericsError(
            "kriging linear-system shapes are invalid",
            object_name=object_name,
            field="matrix/rhs",
            expected="square matrix and aligned vector",
            actual={"matrix": tuple(a.shape), "rhs": tuple(b.shape)},
        )
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise GeomodelNumericsError(
            "kriging linear system must be finite",
            object_name=object_name,
            field="matrix/rhs",
            expected="finite values",
            actual="contains NaN or infinity",
        )
    if not np.allclose(a, a.T, rtol=1.0e-12, atol=0.0):
        raise GeomodelNumericsError(
            "kriging linear system must be symmetric",
            object_name=object_name,
            field="matrix",
            expected="relative asymmetry <= 1e-12",
            actual=float(np.max(np.abs(a - a.T))),
        )
    minimum_eigenvalue, singular_tolerance, standard_deviations = _validate_covariance_block(
        a[:data_count, :data_count], object_name=object_name
    )
    near_singular = minimum_eigenvalue <= singular_tolerance
    if near_singular and policy.on_singular == "error":
        raise GeomodelNumericsError(
            "kriging covariance block is singular or numerically rank deficient",
            object_name=object_name,
            field="covariance",
            expected=f"minimum eigenvalue > {singular_tolerance:.6g}",
            actual=minimum_eigenvalue,
        )
    schedule = policy.jitter_relative if near_singular else (0.0,)
    if policy.on_singular == "stabilize" and not near_singular:
        schedule += policy.jitter_relative
    last_error: Exception | None = None
    for relative in schedule:
        candidate = np.array(a, dtype=np.float64, copy=True)
        if relative:
            candidate[np.arange(data_count), np.arange(data_count)] += (
                relative * standard_deviations * standard_deviations
            )
        normalized_candidate, normalized_rhs, scales = _equilibrate_system(
            as_float_array(candidate),
            b,
            data_count=data_count,
            standard_deviations=standard_deviations,
        )
        if policy.on_singular == "stabilize":
            condition_number = float(np.linalg.cond(normalized_candidate[:data_count, :data_count]))
            if not math.isfinite(condition_number) or condition_number > 1.0e9:
                last_error = GeomodelNumericsError(
                    "stabilized covariance remains too ill-conditioned",
                    object_name=object_name,
                    field="covariance",
                    expected="dimensionless condition number <= 1e9",
                    actual=condition_number,
                )
                continue
        try:
            normalized_solution = np.linalg.solve(normalized_candidate, normalized_rhs)
        except np.linalg.LinAlgError as exc:
            last_error = exc
            continue
        normalized_solution = _refine_solution(
            normalized_candidate,
            normalized_rhs,
            as_float_array(normalized_solution),
        )
        solution = normalized_solution / scales
        residual = float(
            np.linalg.norm(
                normalized_candidate @ normalized_solution - normalized_rhs,
                ord=np.inf,
            )
        )
        backward_scale = float(
            np.linalg.norm(normalized_candidate, ord=np.inf)
            * np.linalg.norm(normalized_solution, ord=np.inf)
            + np.linalg.norm(normalized_rhs, ord=np.inf)
        )
        limit = 1.0e-10 * backward_scale
        if np.isfinite(solution).all() and math.isfinite(residual) and residual <= limit:
            return as_float_array(solution), float(relative), residual
        last_error = GeomodelNumericsError(
            "kriging solve residual exceeds tolerance",
            object_name=object_name,
            field="residual",
            expected=f"<= {limit:.6g}",
            actual=residual,
        )

    raise GeomodelNumericsError(
        "kriging system is singular or failed its residual contract",
        object_name=object_name,
        field="matrix",
        expected=(
            "admissible nonsingular system"
            if policy.on_singular == "error"
            else f"system stabilized by one of {policy.jitter_relative}"
        ),
        actual="solve failed",
    ) from last_error


def validated_variance(value: float, *, sill: float, object_name: str) -> float:
    """Return a finite non-negative variance or raise on material invalidity."""
    sill_scale = abs(sill)
    tolerance = max(1.0e-10 * sill_scale, 8.0 * float(np.spacing(sill_scale)))
    if not math.isfinite(value) or value < -tolerance:
        raise GeomodelNumericsError(
            "kriging variance is not scientifically admissible",
            object_name=object_name,
            field="variance",
            expected=f"finite value >= {-tolerance:.3g}",
            actual=value,
        )
    return 0.0 if value < 0.0 else value


def krige_loop(
    data_coords: FloatArray,
    values: FloatArray,
    target_coords: FloatArray,
    model: CovarianceModel,
    *,
    ktype: int,
    mean: float = 0.0,
    drift_terms: tuple[str, ...] = (),
    neighbourhood: NeighbourhoodSpec,
    solve_policy: KrigingSolvePolicy,
    object_name: str,
) -> tuple[FloatArray, FloatArray, dict[str, object]]:
    """Run one indexed, dimension-aware kriging call."""
    model.require_stationary_covariance(object_name=object_name)
    data = as_float_array(data_coords)
    targets = as_float_array(target_coords)
    vals = as_float_array(values)
    if (
        data.ndim != 2
        or targets.ndim != 2
        or data.shape[1] not in (2, 3)
        or targets.shape[1] != data.shape[1]
        or vals.shape != (data.shape[0],)
    ):
        raise GeomodelContractError(
            "kriging coordinate/value dimensions do not align",
            object_name=object_name,
            field="data/targets/values",
            expected="aligned 2-D or 3-D arrays",
            actual={
                "data": tuple(data.shape),
                "targets": tuple(targets.shape),
                "values": tuple(vals.shape),
            },
        )
    if not np.isfinite(data).all() or not np.isfinite(targets).all() or not np.isfinite(vals).all():
        raise GeomodelContractError(
            "kriging inputs must be finite",
            object_name=object_name,
            field="data/targets/values",
            expected="finite arrays",
            actual="contains NaN or infinity",
        )
    if neighbourhood.ndim != data.shape[1]:
        raise GeomodelContractError(
            "kriging neighbourhood dimension does not match coordinates",
            object_name=object_name,
            field="neighbourhood",
            expected=f"{data.shape[1]}-D NeighbourhoodSpec",
            actual=f"{neighbourhood.ndim}-D",
        )

    n_extra = constraint_count(ktype, drift_terms)
    sill = float(model.sill)
    estimates = np.empty(targets.shape[0], dtype=np.float64)
    variances = np.empty(targets.shape[0], dtype=np.float64)
    source_ids = np.arange(data.shape[0], dtype=np.int64)
    index = StaticKDTreeNeighbourhood.from_arrays(data, source_ids)

    selected_ids: list[tuple[int, ...]] = []
    distance_checks = 0
    jitter_used: list[float] = []
    solve_residuals: list[float] = []
    constraint_residuals: list[float] = []

    for target_index, target in enumerate(targets):
        selection = index.query(target, neighbourhood)
        distance_checks += selection.distance_checks
        ids = tuple(int(value) for value in selection.ids.tolist())
        selected_ids.append(ids)
        if selection.status == "insufficient":
            if ktype == 0:
                estimates[target_index] = mean
                variances[target_index] = sill
                jitter_used.append(0.0)
                solve_residuals.append(0.0)
                constraint_residuals.append(0.0)
                continue
            raise GeomodelNumericsError(
                "kriging neighbourhood cannot satisfy the estimator constraints",
                object_name=object_name,
                field="neighbourhood",
                expected=f"at least {neighbourhood.min_neighbors} neighbours",
                actual=len(ids),
            )

        idx = np.asarray(selection.ids, dtype=np.int64)
        nearby = data[idx]
        nearby_values = vals[idx]
        nd = int(idx.size)
        covariance_dd = covariance_matrix(model, nearby, nearby)
        covariance_dt = covariance_vector(model, nearby, target)
        system: FloatArray = as_float_array(
            np.zeros((nd + n_extra, nd + n_extra), dtype=np.float64)
        )
        rhs: FloatArray = as_float_array(np.zeros(nd + n_extra, dtype=np.float64))
        system[:nd, :nd] = covariance_dd
        rhs[:nd] = covariance_dt

        if ktype in (1, 2):
            system[nd, :nd] = 1.0
            system[:nd, nd] = 1.0
            rhs[nd] = 1.0
        if ktype == 2:
            data_basis = _kriging_drift_basis(nearby, target, list(drift_terms))
            for drift_index in range(len(drift_terms)):
                column = nd + 1 + drift_index
                system[column, :nd] = data_basis[:, drift_index]
                system[:nd, column] = data_basis[:, drift_index]

        solution, jitter, residual = solve_kriging_system(
            as_float_array(system),
            as_float_array(rhs),
            data_count=nd,
            policy=solve_policy,
            object_name=object_name,
        )
        weights = solution[:nd]
        estimate = (
            float(mean + _accurate_dot(weights, as_float_array(nearby_values - mean)))
            if ktype == 0
            else _accurate_dot(weights, nearby_values)
        )
        variance = sill - _accurate_dot(weights, covariance_dt)
        if n_extra:
            variance -= float(np.dot(solution[nd:], rhs[nd:]))
        if not math.isfinite(estimate):
            raise GeomodelNumericsError(
                "kriging estimate is not finite",
                object_name=object_name,
                field="estimate",
                expected="finite result",
                actual=estimate,
            )
        estimates[target_index] = estimate
        variances[target_index] = validated_variance(
            variance,
            sill=sill,
            object_name=object_name,
        )
        jitter_used.append(jitter)
        solve_residuals.append(residual)
        constraint_residuals.append(
            0.0 if n_extra == 0 else float(np.max(np.abs(system[nd:, :nd] @ weights - rhs[nd:])))
        )

    diagnostics: dict[str, object] = {
        "coordinate_dimension": int(data.shape[1]),
        "neighbourhood_backend": "StaticKDTreeNeighbourhood",
        "selected_ids": tuple(selected_ids),
        "distance_checks": int(distance_checks),
        "jitter_relative": tuple(jitter_used),
        "solve_residual": tuple(solve_residuals),
        "constraint_residual": tuple(constraint_residuals),
    }
    return as_float_array(estimates), as_float_array(variances), diagnostics
