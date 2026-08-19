"""Canonical SI inclusion and effective-medium kernels.

The functions in this module preserve the first tensor input's dtype and
device, support PyTorch broadcasting, and expose elementwise convergence
evidence for every iterative calculation.  Bulk and shear moduli are in Pa;
fractions and convergence residuals are dimensionless.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TypeAlias, cast

import torch

from geobrain.core import ErrorCode

from .contracts import RockIterationReport, require_compatible_tensors
from .errors import RockContractError, RockNumericsError

TensorInput: TypeAlias = torch.Tensor | int | float

# Taylor coefficients in x = alpha**2 - 1.  The series are analytic from
# both sides of the spherical limit and avoid subtracting nearly equal terms.
# The first omitted coefficient and convergence-radius safety cap below give
# a dtype-aware switch whose geometric tail is below one quarter of dtype epsilon.
_SPHEROID_SERIES_MAX_ABS_ARGUMENT = 0.25
_SPHEROID_THETA_COEFFICIENTS = (
    2.0 / 3.0,
    2.0 / 15.0,
    -8.0 / 105.0,
    16.0 / 315.0,
    -128.0 / 3465.0,
    256.0 / 9009.0,
    -1024.0 / 45045.0,
    2048.0 / 109395.0,
    -32768.0 / 2078505.0,
    65536.0 / 4849845.0,
    -262144.0 / 22309287.0,
)
_SPHEROID_SHAPE_COEFFICIENTS = (
    -2.0 / 5.0,
    -6.0 / 35.0,
    8.0 / 105.0,
    -16.0 / 385.0,
    128.0 / 5005.0,
    -256.0 / 15015.0,
    1024.0 / 85085.0,
    -2048.0 / 230945.0,
    32768.0 / 4849845.0,
    -196608.0 / 37182145.0,
    786432.0 / 185910725.0,
)
_SPHEROID_FIRST_OMITTED_MAX_COEFFICIENT = 524288.0 / 50702925.0


@dataclass(frozen=True, slots=True)
class EffectiveModuliResult:
    """Effective bulk/shear moduli and detached convergence diagnostics."""

    k_eff: torch.Tensor
    mu_eff: torch.Tensor
    iteration: RockIterationReport


@dataclass(frozen=True, slots=True)
class InclusionFactors:
    """Dimensionless bulk and shear strain-concentration factors."""

    p: torch.Tensor
    q: torch.Tensor


@dataclass(frozen=True, slots=True)
class XuWhiteResult:
    """Xu--White dry-frame moduli, density, and convergence evidence."""

    k_dry: torch.Tensor
    mu_dry: torch.Tensor
    rho_dry: torch.Tensor
    iteration: RockIterationReport


def _tensor_polynomial(
    value: torch.Tensor,
    coefficients: tuple[float, ...],
) -> torch.Tensor:
    """Evaluate low-to-high coefficients with Horner's rule."""
    result = torch.full_like(value, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        result = result * value + coefficient
    return result


def _float_polynomial(value: float, coefficients: tuple[float, ...]) -> float:
    """Scalar counterpart of :func:`_tensor_polynomial`."""
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = result * value + coefficient
    return result


def _spheroid_series_limit(dtype: torch.dtype) -> float:
    """Return a conservative dtype-aware bound for the degree-ten series."""
    epsilon = torch.finfo(dtype).eps
    tail_factor = 1.0 / (1.0 - _SPHEROID_SERIES_MAX_ABS_ARGUMENT)
    error_limited = (0.25 * epsilon / (_SPHEROID_FIRST_OMITTED_MAX_COEFFICIENT * tail_factor)) ** (
        1.0 / 11.0
    )
    return float(min(_SPHEROID_SERIES_MAX_ABS_ARGUMENT, error_limited))


def _spheroid_shape_coefficients(
    aspect_ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable Berryman theta/shape coefficients for alpha > 0."""
    series_limit = _spheroid_series_limit(aspect_ratio.dtype)
    lower = math.sqrt(1.0 - series_limit)
    upper = math.sqrt(1.0 + series_limit)
    near_sphere = (aspect_ratio >= lower) & (aspect_ratio <= upper)
    is_oblate = aspect_ratio < 1.0

    # Safe inactive values keep both torch.where branches finite in backward.
    series_alpha = torch.where(near_sphere, aspect_ratio, torch.ones_like(aspect_ratio))
    series_argument = series_alpha.square() - 1.0
    theta_series = _tensor_polynomial(series_argument, _SPHEROID_THETA_COEFFICIENTS)
    shape_series = _tensor_polynomial(series_argument, _SPHEROID_SHAPE_COEFFICIENTS)

    direct_oblate = is_oblate & ~near_sphere
    oblate_alpha = torch.where(
        direct_oblate,
        aspect_ratio,
        torch.full_like(aspect_ratio, 0.5),
    )
    oblate_base = 1.0 - oblate_alpha.square()
    oblate_root = torch.sqrt(oblate_base)
    theta_oblate = (
        oblate_alpha / oblate_base**1.5 * (torch.acos(oblate_alpha) - oblate_alpha * oblate_root)
    )
    shape_oblate = oblate_alpha.square() * (3.0 * theta_oblate - 2.0) / oblate_base

    direct_prolate = ~is_oblate & ~near_sphere
    prolate_alpha = torch.where(
        direct_prolate,
        aspect_ratio,
        torch.full_like(aspect_ratio, 2.0),
    )
    inverse_alpha = prolate_alpha.reciprocal()
    prolate_base = 1.0 - inverse_alpha.square()
    theta_prolate = prolate_base.reciprocal() - (
        torch.acosh(prolate_alpha)
        * inverse_alpha.square()
        / (prolate_base * torch.sqrt(prolate_base))
    )
    shape_prolate = -(3.0 * theta_prolate - 2.0) / prolate_base

    theta_direct = torch.where(is_oblate, theta_oblate, theta_prolate)
    shape_direct = torch.where(is_oblate, shape_oblate, shape_prolate)
    return (
        torch.where(near_sphere, theta_series, theta_direct),
        torch.where(near_sphere, shape_series, shape_direct),
    )


def _validated_inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    tensors = cast(
        tuple[torch.Tensor, ...],
        require_compatible_tensors(object_name, *fields),
    )
    for (field, _), tensor in zip(fields, tensors, strict=True):
        if tensor.layout is not torch.strided:
            raise RockContractError(
                "inclusion kernels require strided tensors",
                object_name=object_name,
                field=field,
                expected="torch.strided layout",
                actual=str(tensor.layout),
            )
        if tensor.device.type == "meta":
            raise RockContractError(
                "inclusion kernels require materialized tensors",
                object_name=object_name,
                field=field,
                expected="materialized CPU or accelerator tensor",
                actual=str(tensor.device),
                code=ErrorCode.DEVICE_UNAVAILABLE,
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "inclusion input must be finite",
                object_name=object_name,
                field=field,
                expected="finite values",
                actual="non-finite value(s)",
            )
    return cast(tuple[torch.Tensor, ...], torch.broadcast_tensors(*tensors))


def _extrema(value: torch.Tensor) -> dict[str, float]:
    return {"minimum": value.amin().item(), "maximum": value.amax().item()}


def _require_positive(object_name: str, field: str, value: torch.Tensor) -> None:
    if bool(torch.any(value <= 0.0)):
        raise RockContractError(
            "elastic bulk modulus must be positive",
            object_name=object_name,
            field=field,
            expected="> 0 Pa",
            actual=_extrema(value),
        )


def _require_nonnegative(object_name: str, field: str, value: torch.Tensor) -> None:
    if bool(torch.any(value < 0.0)):
        raise RockContractError(
            "elastic shear modulus must be non-negative",
            object_name=object_name,
            field=field,
            expected=">= 0 Pa",
            actual=_extrema(value),
        )


def _require_fraction(object_name: str, field: str, value: torch.Tensor) -> None:
    if bool(torch.any((value < 0.0) | (value > 1.0))):
        raise RockContractError(
            "phase fraction must lie in the unit interval",
            object_name=object_name,
            field=field,
            expected="0 <= value <= 1",
            actual=_extrema(value),
        )


def _positive_float(object_name: str, field: str, value: object) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RockContractError(
            "iteration control must be positive and finite",
            object_name=object_name,
            field=field,
            expected="positive finite real number",
            actual=value,
        )
    return float(value)


def _positive_int(object_name: str, field: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise RockContractError(
            "iteration control must be a positive integer",
            object_name=object_name,
            field=field,
            expected="positive integer",
            actual=value,
        )
    return value


def _safe_scale(k: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    tiny = k.new_tensor(torch.finfo(k.dtype).tiny)
    return torch.maximum(k.abs(), mu.abs()).clamp_min(tiny)


def _require_finite_moduli(
    object_name: str,
    k_eff: torch.Tensor,
    mu_eff: torch.Tensor,
) -> None:
    for field, value in (("k_eff", k_eff), ("mu_eff", mu_eff)):
        if not bool(torch.isfinite(value).all()):
            raise RockNumericsError(
                "effective-medium iteration produced non-finite moduli",
                object_name=object_name,
                field=field,
                expected="finite Pa values",
                actual="non-finite value(s)",
            )
        if bool(torch.any(value < 0.0)):
            raise RockNumericsError(
                "effective-medium iteration produced negative moduli",
                object_name=object_name,
                field=field,
                expected=">= 0 Pa",
                actual=_extrema(value),
            )


def spherical_inclusion_factors(
    k_matrix: torch.Tensor,
    mu_matrix: TensorInput,
    k_inclusion: TensorInput,
    mu_inclusion: TensorInput,
) -> InclusionFactors:
    """Return Berryman spherical-inclusion ``P`` and ``Q`` factors."""

    object_name = "spherical_inclusion_factors"
    k_m, mu_m, k_i, mu_i = _validated_inputs(
        object_name,
        ("k_matrix", k_matrix),
        ("mu_matrix", mu_matrix),
        ("k_inclusion", k_inclusion),
        ("mu_inclusion", mu_inclusion),
    )
    _require_positive(object_name, "k_matrix", k_m)
    _require_nonnegative(object_name, "mu_matrix", mu_m)
    _require_nonnegative(object_name, "k_inclusion", k_i)
    _require_nonnegative(object_name, "mu_inclusion", mu_i)
    if bool(torch.any(mu_m <= 0.0)):
        raise RockContractError(
            "the spherical-inclusion matrix must resist shear",
            object_name=object_name,
            field="mu_matrix",
            expected="> 0 Pa",
            actual=_extrema(mu_m),
        )

    zeta = (mu_m / 6.0) * (9.0 * k_m + 8.0 * mu_m) / (k_m + 2.0 * mu_m)
    p = (k_m + (4.0 / 3.0) * mu_m) / (k_i + (4.0 / 3.0) * mu_m)
    q = (mu_m + zeta) / (mu_i + zeta)
    return InclusionFactors(p, q)


def kuster_toksoz_moduli(
    k_matrix: torch.Tensor,
    mu_matrix: TensorInput,
    k_inclusion: TensorInput,
    mu_inclusion: TensorInput,
    inclusion_fraction: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dilute spherical Kuster--Toksöz effective moduli in Pa."""

    object_name = "kuster_toksoz_moduli"
    k_m, mu_m, k_i, mu_i, fraction = _validated_inputs(
        object_name,
        ("k_matrix", k_matrix),
        ("mu_matrix", mu_matrix),
        ("k_inclusion", k_inclusion),
        ("mu_inclusion", mu_inclusion),
        ("inclusion_fraction", inclusion_fraction),
    )
    _require_positive(object_name, "k_matrix", k_m)
    _require_nonnegative(object_name, "mu_matrix", mu_m)
    _require_nonnegative(object_name, "k_inclusion", k_i)
    _require_nonnegative(object_name, "mu_inclusion", mu_i)
    _require_fraction(object_name, "inclusion_fraction", fraction)
    factors = spherical_inclusion_factors(k_m, mu_m, k_i, mu_i)
    zeta = (mu_m / 6.0) * (9.0 * k_m + 8.0 * mu_m) / (k_m + 2.0 * mu_m)
    bulk_ratio = fraction * (k_i - k_m) * factors.p / (k_m + (4.0 / 3.0) * mu_m)
    shear_ratio = fraction * (mu_i - mu_m) * factors.q / (mu_m + zeta)
    k_eff = (k_m + bulk_ratio * (4.0 / 3.0) * mu_m) / (1.0 - bulk_ratio)
    mu_eff = (mu_m + shear_ratio * zeta) / (1.0 - shear_ratio)
    _require_finite_moduli(object_name, k_eff, mu_eff)
    return k_eff, mu_eff


def ellipsoidal_inclusion_factors(
    k_matrix: torch.Tensor,
    mu_matrix: TensorInput,
    k_inclusion: TensorInput,
    mu_inclusion: TensorInput,
    aspect_ratio: TensorInput,
) -> InclusionFactors:
    """Return Berryman ``P``/``Q`` factors for spheroidal inclusions."""

    object_name = "ellipsoidal_inclusion_factors"
    k_m, mu_m, k_i, mu_i, alpha = _validated_inputs(
        object_name,
        ("k_matrix", k_matrix),
        ("mu_matrix", mu_matrix),
        ("k_inclusion", k_inclusion),
        ("mu_inclusion", mu_inclusion),
        ("aspect_ratio", aspect_ratio),
    )
    _require_positive(object_name, "k_matrix", k_m)
    _require_positive(object_name, "mu_matrix", mu_m)
    _require_nonnegative(object_name, "k_inclusion", k_i)
    _require_nonnegative(object_name, "mu_inclusion", mu_i)
    _require_positive(object_name, "aspect_ratio", alpha)

    theta, shape = _spheroid_shape_coefficients(alpha)
    shear_ratio = mu_i / mu_m
    bulk_ratio = k_i / k_m
    a_value = shear_ratio - 1.0
    b_value = (bulk_ratio - shear_ratio) / 3.0
    r_value = mu_m / (k_m + (4.0 / 3.0) * mu_m)
    three_minus_four_r = 3.0 - 4.0 * r_value
    f_1_contrast = 1.5 * (shape + theta) - r_value * (1.5 * shape + 2.5 * theta - 4.0 / 3.0)
    f_1 = 1.0 - f_1_contrast + shear_ratio * f_1_contrast
    f_2_delta = 1.5 * (shape + theta) - r_value * (1.5 * shape + 2.5 * theta)
    f_2 = (
        shear_ratio
        + a_value * f_2_delta
        + b_value * three_minus_four_r
        + a_value
        * (bulk_ratio - 1.0)
        * (1.5 - 2.0 * r_value)
        * (shape + theta - r_value * (shape - theta + 2.0 * theta.square()))
    )
    f_3_delta = -shape - 1.5 * theta + r_value * (shape + theta)
    f_3 = shear_ratio + a_value * f_3_delta
    f_4 = 1.0 + (a_value / 4.0) * (shape + 3.0 * theta - r_value * (shape - theta))
    f_5 = (
        a_value * (-shape + r_value * (shape + theta - 4.0 / 3.0))
        + b_value * theta * three_minus_four_r
    )
    f_6_delta = shape - r_value * (shape + theta)
    f_6 = shear_ratio + a_value * f_6_delta + b_value * (1.0 - theta) * three_minus_four_r
    f_7 = (
        2.0
        + (a_value / 4.0) * (3.0 * shape + 9.0 * theta - r_value * (3.0 * shape + 5.0 * theta))
        + b_value * theta * three_minus_four_r
    )
    f_8 = (
        a_value
        * (
            1.0
            - 2.0 * r_value
            + (shape / 2.0) * (r_value - 1.0)
            + (theta / 2.0) * (5.0 * r_value - 3.0)
        )
        + b_value * (1.0 - theta) * three_minus_four_r
    )
    f_9 = a_value * ((r_value - 1.0) * shape - r_value * theta) + (
        b_value * theta * three_minus_four_r
    )
    p_general = f_1 / f_2
    q_general = (2.0 / f_3 + 1.0 / f_4 + (f_4 * f_5 + f_6 * f_7 - f_8 * f_9) / (f_2 * f_4)) / 5.0
    for field, value in (("p", p_general), ("q", q_general)):
        if not bool(torch.isfinite(value).all()):
            raise RockNumericsError(
                "ellipsoidal concentration factor is non-finite",
                object_name=object_name,
                field=field,
                expected="finite dimensionless values",
                actual="non-finite value(s)",
            )
    return InclusionFactors(p_general, q_general)


def swiss_cheese_moduli(
    k_solid: torch.Tensor,
    mu_solid: TensorInput,
    porosity: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dilute spherical dry-pore moduli."""

    object_name = "swiss_cheese_moduli"
    k_s, mu_s, phi = _validated_inputs(
        object_name,
        ("k_solid", k_solid),
        ("mu_solid", mu_solid),
        ("porosity", porosity),
    )
    _require_positive(object_name, "k_solid", k_s)
    _require_positive(object_name, "mu_solid", mu_s)
    _require_fraction(object_name, "porosity", phi)
    k_eff = k_s / (1.0 + (1.0 + 3.0 * k_s / (4.0 * mu_s)) * phi)
    mu_eff = mu_s / (1.0 + (15.0 * k_s + 20.0 * mu_s) / (9.0 * k_s + 8.0 * mu_s) * phi)
    _require_finite_moduli(object_name, k_eff, mu_eff)
    return k_eff, mu_eff


def dilute_crack_moduli(
    k_solid: torch.Tensor,
    mu_solid: TensorInput,
    crack_density: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Walsh dilute random-crack moduli."""

    object_name = "dilute_crack_moduli"
    k_s, mu_s, density = _validated_inputs(
        object_name,
        ("k_solid", k_solid),
        ("mu_solid", mu_solid),
        ("crack_density", crack_density),
    )
    _require_positive(object_name, "k_solid", k_s)
    _require_positive(object_name, "mu_solid", mu_s)
    _require_nonnegative(object_name, "crack_density", density)
    poisson = (3.0 * k_s - 2.0 * mu_s) / (2.0 * (3.0 * k_s + mu_s))
    k_eff = k_s / (1.0 + (16.0 / 9.0) * (1.0 - poisson.square()) / (1.0 - 2.0 * poisson) * density)
    mu_eff = mu_s / (
        1.0 + 32.0 * (1.0 - poisson) * (5.0 - poisson) / (45.0 * (2.0 - poisson)) * density
    )
    _require_finite_moduli(object_name, k_eff, mu_eff)
    return k_eff, mu_eff


def self_consistent_dilute_moduli(
    k_matrix: torch.Tensor,
    mu_matrix: TensorInput,
    k_inclusion: TensorInput,
    mu_inclusion: TensorInput,
    inclusion_fraction: TensorInput,
    *,
    mode: str = "stress",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the stress- or strain-form dilute self-consistent estimate."""

    object_name = "self_consistent_dilute_moduli"
    if mode not in {"stress", "strain"}:
        raise RockContractError(
            "unknown dilute self-consistent mode",
            object_name=object_name,
            field="mode",
            expected="stress or strain",
            actual=mode,
        )
    k_m, mu_m, k_i, mu_i, fraction = _validated_inputs(
        object_name,
        ("k_matrix", k_matrix),
        ("mu_matrix", mu_matrix),
        ("k_inclusion", k_inclusion),
        ("mu_inclusion", mu_inclusion),
        ("inclusion_fraction", inclusion_fraction),
    )
    _require_positive(object_name, "k_matrix", k_m)
    _require_positive(object_name, "mu_matrix", mu_m)
    _require_nonnegative(object_name, "k_inclusion", k_i)
    _require_nonnegative(object_name, "mu_inclusion", mu_i)
    _require_fraction(object_name, "inclusion_fraction", fraction)
    poisson = 0.5 * (3.0 * k_m - 2.0 * mu_m) / (3.0 * k_m + mu_m)
    s_1 = (1.0 / 3.0) * (1.0 + poisson) / (1.0 - poisson)
    s_2 = (2.0 / 15.0) * (4.0 - 5.0 * poisson) / (1.0 - poisson)
    equal_bulk = k_m == k_i
    equal_shear = mu_m == mu_i
    bulk_difference = torch.where(
        equal_bulk,
        torch.ones_like(k_m),
        k_m - k_i,
    )
    shear_difference = torch.where(
        equal_shear,
        torch.ones_like(mu_m),
        mu_m - mu_i,
    )
    bulk_increment = (k_m / bulk_difference - s_1).reciprocal()
    shear_increment = (mu_m / shear_difference - s_2).reciprocal()
    if mode == "stress":
        k_eff = k_m / (1.0 + fraction * bulk_increment)
        mu_eff = mu_m / (1.0 + fraction * shear_increment)
    else:
        k_eff = k_m * (1.0 - fraction * bulk_increment)
        mu_eff = mu_m * (1.0 - fraction * shear_increment)
    k_eff = torch.where(equal_bulk, k_m, k_eff)
    mu_eff = torch.where(equal_shear, mu_m, mu_eff)
    _require_finite_moduli(object_name, k_eff, mu_eff)
    return k_eff, mu_eff


def _dem_integrate(
    k_host: torch.Tensor,
    mu_host: torch.Tensor,
    k_inclusion: torch.Tensor,
    mu_inclusion: torch.Tensor,
    fraction: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_eff = k_host
    mu_eff = mu_host
    increment = fraction / steps

    for step in range(steps):
        phase_fraction = increment * step
        denominator = 1.0 - phase_fraction
        zeta = (mu_eff / 6.0) * (9.0 * k_eff + 8.0 * mu_eff) / (k_eff + 2.0 * mu_eff)
        p = (k_eff + (4.0 / 3.0) * mu_eff) / (k_inclusion + (4.0 / 3.0) * mu_eff)
        q = (mu_eff + zeta) / (mu_inclusion + zeta)
        dk_1 = (k_inclusion - k_eff) * p / denominator
        dmu_1 = (mu_inclusion - mu_eff) * q / denominator

        k_mid = k_eff + 0.5 * increment * dk_1
        mu_mid = mu_eff + 0.5 * increment * dmu_1
        phase_mid = phase_fraction + 0.5 * increment
        zeta_mid = (mu_mid / 6.0) * (9.0 * k_mid + 8.0 * mu_mid) / (k_mid + 2.0 * mu_mid)
        p_mid = (k_mid + (4.0 / 3.0) * mu_mid) / (k_inclusion + (4.0 / 3.0) * mu_mid)
        q_mid = (mu_mid + zeta_mid) / (mu_inclusion + zeta_mid)
        dk_2 = (k_inclusion - k_mid) * p_mid / (1.0 - phase_mid)
        dmu_2 = (mu_inclusion - mu_mid) * q_mid / (1.0 - phase_mid)
        k_eff = k_eff + increment * dk_2
        mu_eff = mu_eff + increment * dmu_2
    return k_eff, mu_eff


def differential_effective_medium(
    k_host: torch.Tensor,
    mu_host: TensorInput,
    k_inclusion: TensorInput,
    mu_inclusion: TensorInput,
    inclusion_fraction: TensorInput,
    *,
    steps: int = 50,
    tolerance: float = 1.0e-3,
) -> EffectiveModuliResult:
    """Integrate the spherical DEM ODE and estimate step-discretization error.

    The returned moduli use ``2 * steps`` midpoint steps.  The diagnostic
    residual is the normalized difference from a companion ``steps`` solve.
    """

    object_name = "differential_effective_medium"
    validated_steps = _positive_int(object_name, "steps", steps)
    validated_tolerance = _positive_float(object_name, "tolerance", tolerance)
    k_h, mu_h, k_i, mu_i, fraction = _validated_inputs(
        object_name,
        ("k_host", k_host),
        ("mu_host", mu_host),
        ("k_inclusion", k_inclusion),
        ("mu_inclusion", mu_inclusion),
        ("inclusion_fraction", inclusion_fraction),
    )
    _require_positive(object_name, "k_host", k_h)
    _require_positive(object_name, "mu_host", mu_h)
    _require_nonnegative(object_name, "k_inclusion", k_i)
    _require_nonnegative(object_name, "mu_inclusion", mu_i)
    _require_fraction(object_name, "inclusion_fraction", fraction)
    if bool(torch.any(fraction >= 1.0)):
        raise RockContractError(
            "DEM inclusion fraction must stay below its singular endpoint",
            object_name=object_name,
            field="inclusion_fraction",
            expected="0 <= value < 1",
            actual=_extrema(fraction),
        )

    coarse_k, coarse_mu = _dem_integrate(k_h, mu_h, k_i, mu_i, fraction, validated_steps)
    fine_k, fine_mu = _dem_integrate(k_h, mu_h, k_i, mu_i, fraction, 2 * validated_steps)
    scale = _safe_scale(fine_k, fine_mu)
    residual = (
        torch.maximum(
            (fine_k - coarse_k).abs(),
            (fine_mu - coarse_mu).abs(),
        )
        / scale
    )
    _require_finite_moduli(object_name, fine_k, fine_mu)
    report = RockIterationReport(
        residual <= validated_tolerance,
        residual,
        2 * validated_steps,
        validated_tolerance,
    )
    return EffectiveModuliResult(fine_k, fine_mu, report)


def _self_consistent_update(
    k_eff: torch.Tensor,
    mu_eff: torch.Tensor,
    k_1: torch.Tensor,
    mu_1: torch.Tensor,
    k_2: torch.Tensor,
    mu_2: torch.Tensor,
    fraction_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    fraction_2 = 1.0 - fraction_1
    zeta = (mu_eff / 6.0) * (9.0 * k_eff + 8.0 * mu_eff) / (k_eff + 2.0 * mu_eff)
    k_plus_four_thirds_mu = k_eff + (4.0 / 3.0) * mu_eff
    p_1 = k_plus_four_thirds_mu / (k_1 + (4.0 / 3.0) * mu_eff)
    p_2 = k_plus_four_thirds_mu / (k_2 + (4.0 / 3.0) * mu_eff)
    q_1 = (mu_eff + zeta) / (mu_1 + zeta)
    q_2 = (mu_eff + zeta) / (mu_2 + zeta)
    k_candidate = (fraction_1 * k_1 * p_1 + fraction_2 * k_2 * p_2) / (
        fraction_1 * p_1 + fraction_2 * p_2
    )
    mu_candidate = (fraction_1 * mu_1 * q_1 + fraction_2 * mu_2 * q_2) / (
        fraction_1 * q_1 + fraction_2 * q_2
    )
    return k_candidate, mu_candidate


def self_consistent_moduli(
    k_phase_1: torch.Tensor,
    mu_phase_1: TensorInput,
    k_phase_2: TensorInput,
    mu_phase_2: TensorInput,
    phase_1_fraction: TensorInput,
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 200,
) -> EffectiveModuliResult:
    """Solve the two-phase spherical self-consistent equations.

    Args:
        k_phase_1 / mu_phase_1: phase-1 moduli [Pa].
        k_phase_2 / mu_phase_2: phase-2 moduli [Pa].
        phase_1_fraction: volume fraction of phase 1.
        tolerance: fixed-point convergence tolerance.
        max_iterations: iteration cap.
    """

    object_name = "self_consistent_moduli"
    validated_tolerance = _positive_float(object_name, "tolerance", tolerance)
    validated_iterations = _positive_int(object_name, "max_iterations", max_iterations)
    k_1, mu_1, k_2, mu_2, fraction_1 = _validated_inputs(
        object_name,
        ("k_phase_1", k_phase_1),
        ("mu_phase_1", mu_phase_1),
        ("k_phase_2", k_phase_2),
        ("mu_phase_2", mu_phase_2),
        ("phase_1_fraction", phase_1_fraction),
    )
    _require_positive(object_name, "k_phase_1", k_1)
    _require_nonnegative(object_name, "mu_phase_1", mu_1)
    _require_positive(object_name, "k_phase_2", k_2)
    _require_nonnegative(object_name, "mu_phase_2", mu_2)
    _require_fraction(object_name, "phase_1_fraction", fraction_1)

    fraction_2 = 1.0 - fraction_1
    k_arithmetic = fraction_1 * k_1 + fraction_2 * k_2
    k_harmonic = 1.0 / (fraction_1 / k_1 + fraction_2 / k_2)
    k_eff = 0.5 * (k_arithmetic + k_harmonic)
    # A phase may be a fluid.  The arithmetic mean is a finite, positive
    # initial guess and avoids inventing a shear modulus in a harmonic floor.
    mu_eff = fraction_1 * mu_1 + fraction_2 * mu_2
    fluid_only = (mu_1 == 0.0) & (mu_2 == 0.0)
    k_eff = torch.where(fluid_only, k_harmonic, k_eff)
    mu_eff = torch.where(fluid_only, torch.zeros_like(mu_eff), mu_eff)
    pure_1 = fraction_1 == 1.0
    pure_2 = fraction_1 == 0.0
    converged = pure_1 | pure_2 | fluid_only
    residual = torch.where(
        converged,
        torch.zeros_like(k_eff),
        torch.full_like(k_eff, torch.inf),
    )

    for _ in range(validated_iterations):
        iteration_k = torch.where(converged, torch.ones_like(k_eff), k_eff)
        iteration_mu = torch.where(converged, torch.ones_like(mu_eff), mu_eff)
        k_candidate, mu_candidate = _self_consistent_update(
            iteration_k,
            iteration_mu,
            k_1,
            mu_1,
            k_2,
            mu_2,
            fraction_1,
        )
        candidate_scale = _safe_scale(k_candidate, mu_candidate)
        candidate_residual = (
            torch.maximum(
                (k_candidate - k_eff).abs(),
                (mu_candidate - mu_eff).abs(),
            )
            / candidate_scale
        )
        active = ~converged
        k_eff = torch.where(active, k_candidate, k_eff)
        mu_eff = torch.where(active, mu_candidate, mu_eff)
        residual = torch.where(active, candidate_residual, residual)
        converged = converged | (candidate_residual <= validated_tolerance)

    k_eff = torch.where(pure_1, k_1, torch.where(pure_2, k_2, k_eff))
    mu_eff = torch.where(pure_1, mu_1, torch.where(pure_2, mu_2, mu_eff))
    _require_finite_moduli(object_name, k_eff, mu_eff)
    report = RockIterationReport(
        residual <= validated_tolerance,
        residual,
        validated_iterations,
        validated_tolerance,
    )
    return EffectiveModuliResult(k_eff, mu_eff, report)


def sc_flex_moduli(
    k_matrix: torch.Tensor,
    mu_matrix: TensorInput,
    k_inclusion: TensorInput,
    mu_inclusion: TensorInput,
    inclusion_fraction: TensorInput,
    *,
    tolerance: float = 1.0e-6,
    max_iterations: int = 100,
) -> EffectiveModuliResult:
    """Solve the flexible two-phase self-consistent formulation."""

    object_name = "sc_flex_moduli"
    validated_tolerance = _positive_float(object_name, "tolerance", tolerance)
    validated_iterations = _positive_int(object_name, "max_iterations", max_iterations)
    k_m, mu_m, k_i, mu_i, fraction = _validated_inputs(
        object_name,
        ("k_matrix", k_matrix),
        ("mu_matrix", mu_matrix),
        ("k_inclusion", k_inclusion),
        ("mu_inclusion", mu_inclusion),
        ("inclusion_fraction", inclusion_fraction),
    )
    _require_positive(object_name, "k_matrix", k_m)
    _require_positive(object_name, "mu_matrix", mu_m)
    _require_nonnegative(object_name, "k_inclusion", k_i)
    _require_nonnegative(object_name, "mu_inclusion", mu_i)
    _require_fraction(object_name, "inclusion_fraction", fraction)

    k_eff = k_m
    mu_eff = mu_m
    pure_matrix = fraction == 0.0
    pure_inclusion = fraction == 1.0
    converged = pure_matrix | pure_inclusion
    residual = torch.where(
        converged,
        torch.zeros_like(k_eff),
        torch.full_like(k_eff, torch.inf),
    )
    tiny = k_eff.new_tensor(torch.finfo(k_eff.dtype).tiny)

    for _ in range(validated_iterations):
        iteration_k = torch.where(converged, k_m, k_eff)
        iteration_mu = torch.where(converged, mu_m, mu_eff)
        nu = 0.5 * (3.0 * iteration_k - 2.0 * iteration_mu) / (3.0 * iteration_k + iteration_mu)
        s_1 = (1.0 / 3.0) * (1.0 + nu) / (1.0 - nu)
        s_2 = (2.0 / 15.0) * (4.0 - 5.0 * nu) / (1.0 - nu)
        equal_bulk = k_m == k_i
        equal_shear = mu_m == mu_i
        bulk_difference = torch.where(
            equal_bulk,
            torch.ones_like(iteration_k),
            iteration_k - k_i,
        )
        shear_difference = torch.where(
            equal_shear,
            torch.ones_like(iteration_mu),
            iteration_mu - mu_i,
        )
        bulk_contrast = (iteration_k / bulk_difference - s_1).reciprocal()
        shear_contrast = (iteration_mu / shear_difference - s_2).reciprocal()
        k_candidate = (
            1.0 - fraction * iteration_k * (k_m - k_i) / (k_m * bulk_difference) * bulk_contrast
        ) * k_m
        mu_candidate = (
            1.0
            - fraction * iteration_mu * (mu_m - mu_i) / (mu_m * shear_difference) * shear_contrast
        ) * mu_m
        # Exact equal-modulus cases have removable 0/0 contrasts.
        k_candidate = torch.where(k_m == k_i, k_m, k_candidate)
        mu_candidate = torch.where(mu_m == mu_i, mu_m, mu_candidate)
        scale = torch.maximum(k_candidate.abs(), mu_candidate.abs()).clamp_min(tiny)
        candidate_residual = (
            torch.maximum(
                (k_candidate - k_eff).abs(),
                (mu_candidate - mu_eff).abs(),
            )
            / scale
        )
        active = ~converged
        k_eff = torch.where(active, k_candidate, k_eff)
        mu_eff = torch.where(active, mu_candidate, mu_eff)
        residual = torch.where(active, candidate_residual, residual)
        converged = converged | (candidate_residual <= validated_tolerance)

    k_eff = torch.where(pure_matrix, k_m, torch.where(pure_inclusion, k_i, k_eff))
    mu_eff = torch.where(pure_matrix, mu_m, torch.where(pure_inclusion, mu_i, mu_eff))
    _require_finite_moduli(object_name, k_eff, mu_eff)
    report = RockIterationReport(
        residual <= validated_tolerance,
        residual,
        validated_iterations,
        validated_tolerance,
    )
    return EffectiveModuliResult(k_eff, mu_eff, report)


def oconnell_budiansky_fluid_moduli(
    k_dry_matrix: torch.Tensor,
    mu_dry_matrix: TensorInput,
    k_fluid: TensorInput,
    crack_density: TensorInput,
    aspect_ratio: TensorInput,
    *,
    tolerance: float = 1.0e-6,
    max_iterations: int = 100,
) -> EffectiveModuliResult:
    """Return fluid-filled penny-crack moduli with fixed-loop diagnostics."""

    object_name = "oconnell_budiansky_fluid_moduli"
    validated_tolerance = _positive_float(object_name, "tolerance", tolerance)
    validated_iterations = _positive_int(object_name, "max_iterations", max_iterations)
    k_0, mu_0, k_fl, density, alpha = _validated_inputs(
        object_name,
        ("k_dry_matrix", k_dry_matrix),
        ("mu_dry_matrix", mu_dry_matrix),
        ("k_fluid", k_fluid),
        ("crack_density", crack_density),
        ("aspect_ratio", aspect_ratio),
    )
    _require_positive(object_name, "k_dry_matrix", k_0)
    _require_positive(object_name, "mu_dry_matrix", mu_0)
    _require_positive(object_name, "k_fluid", k_fl)
    _require_nonnegative(object_name, "crack_density", density)
    _require_positive(object_name, "aspect_ratio", alpha)
    if bool(torch.any(alpha >= 1.0)):
        raise RockContractError(
            "penny-crack aspect ratio must stay below one",
            object_name=object_name,
            field="aspect_ratio",
            expected="0 < value < 1",
            actual=_extrema(alpha),
        )

    nu_0 = (3.0 * k_0 - 2.0 * mu_0) / (6.0 * k_0 + 2.0 * mu_0)
    fluid_ratio = k_fl / (alpha * k_0)
    nu_eff = torch.full_like(k_0, 0.2)
    crack_factor = torch.full_like(k_0, 0.9)
    no_cracks = density == 0.0
    converged = no_cracks
    residual = torch.where(
        converged,
        torch.zeros_like(k_0),
        torch.full_like(k_0, torch.inf),
    )
    safe_density = torch.where(no_cracks, torch.ones_like(density), density)
    tiny = torch.full_like(k_0, torch.finfo(k_0.dtype).tiny)

    for _ in range(validated_iterations):
        numerator = (45.0 / 16.0) * (nu_0 - nu_eff) / (1.0 - nu_eff.square()) * (2.0 - nu_eff)
        denominator = crack_factor * (1.0 + 3.0 * nu_0) * (2.0 - nu_eff) - 2.0 * (1.0 - 2.0 * nu_0)
        safe_denominator = torch.where(
            denominator.abs() > tiny, denominator, torch.ones_like(denominator)
        )
        equation_residual = torch.where(
            denominator.abs() > tiny,
            numerator / safe_denominator - density,
            torch.ones_like(denominator),
        )
        b_coefficient = -(
            density
            + (9.0 / 16.0) * (1.0 - 2.0 * nu_eff) / (1.0 - nu_0.square())
            + 3.0 * fluid_ratio / (4.0 * math.pi)
        )
        c_coefficient = (9.0 / 16.0) * (1.0 - 2.0 * nu_eff) / (1.0 - nu_0.square())
        discriminant = b_coefficient.square() - 4.0 * density * c_coefficient
        if bool(torch.any(discriminant < 0.0)):
            raise RockNumericsError(
                "fluid-crack iteration reached a negative discriminant",
                object_name=object_name,
                field="discriminant",
                expected=">= 0",
                actual=_extrema(discriminant),
            )
        factor_candidate = (-b_coefficient - torch.sqrt(discriminant)) / (2.0 * safe_density)
        factor_candidate = factor_candidate.clamp(min=0.01, max=1.0)
        nu_candidate = (nu_eff + 0.1 * equation_residual).clamp(min=-0.5, max=0.499)
        state_residual = torch.maximum(
            (nu_candidate - nu_eff).abs(),
            (factor_candidate - crack_factor).abs(),
        )

        candidate_numerator = (
            (45.0 / 16.0)
            * (nu_0 - nu_candidate)
            / (1.0 - nu_candidate.square())
            * (2.0 - nu_candidate)
        )
        candidate_denominator = factor_candidate * (1.0 + 3.0 * nu_0) * (
            2.0 - nu_candidate
        ) - 2.0 * (1.0 - 2.0 * nu_0)
        coupled_term = density * candidate_denominator
        equation_scale = torch.maximum(
            torch.maximum(candidate_numerator.abs(), coupled_term.abs()),
            tiny,
        )
        coupled_residual = (candidate_numerator - coupled_term).abs() / equation_scale
        coupled_residual = torch.where(
            candidate_denominator.abs() > tiny,
            coupled_residual,
            torch.ones_like(coupled_residual),
        )

        candidate_b = -(
            density
            + (9.0 / 16.0) * (1.0 - 2.0 * nu_candidate) / (1.0 - nu_0.square())
            + 3.0 * fluid_ratio / (4.0 * math.pi)
        )
        candidate_c = (9.0 / 16.0) * (1.0 - 2.0 * nu_candidate) / (1.0 - nu_0.square())
        quadratic_a_term = density * factor_candidate.square()
        quadratic_b_term = candidate_b * factor_candidate
        quadratic_scale = torch.maximum(
            torch.maximum(quadratic_a_term.abs(), quadratic_b_term.abs()),
            torch.maximum(candidate_c.abs(), tiny),
        )
        quadratic_residual = (
            quadratic_a_term + quadratic_b_term + candidate_c
        ).abs() / quadratic_scale
        candidate_residual = torch.maximum(
            state_residual,
            torch.maximum(coupled_residual, quadratic_residual),
        )
        active = ~converged
        nu_eff = torch.where(active, nu_candidate, nu_eff)
        crack_factor = torch.where(active, factor_candidate, crack_factor)
        residual = torch.where(active, candidate_residual, residual)
        converged = converged | (candidate_residual <= validated_tolerance)

    k_eff = k_0 * (
        1.0 - 16.0 * (1.0 - nu_eff.square()) * density * crack_factor / (9.0 * (1.0 - 2.0 * nu_eff))
    )
    mu_eff = mu_0 * (
        1.0 - (32.0 / 45.0) * (1.0 - nu_eff) * (crack_factor + 3.0 / (2.0 - nu_eff)) * density
    )
    k_eff = torch.where(no_cracks, k_0, k_eff)
    mu_eff = torch.where(no_cracks, mu_0, mu_eff)
    _require_finite_moduli(object_name, k_eff, mu_eff)
    report = RockIterationReport(
        residual <= validated_tolerance,
        residual,
        validated_iterations,
        validated_tolerance,
    )
    return EffectiveModuliResult(k_eff, mu_eff, report)


def _oblate_shape_coefficients(aspect_ratio: float) -> tuple[float, float]:
    series_argument = aspect_ratio * aspect_ratio - 1.0
    if abs(series_argument) <= _spheroid_series_limit(torch.float64):
        return (
            _float_polynomial(series_argument, _SPHEROID_THETA_COEFFICIENTS),
            _float_polynomial(series_argument, _SPHEROID_SHAPE_COEFFICIENTS),
        )
    one_minus_square = 1.0 - aspect_ratio * aspect_ratio
    root = math.sqrt(one_minus_square)
    theta = (aspect_ratio / one_minus_square**1.5) * (math.acos(aspect_ratio) - aspect_ratio * root)
    shape = aspect_ratio**2 * (3.0 * theta - 2.0) / one_minus_square
    return theta, shape


def _oblate_factors(
    k_matrix: torch.Tensor,
    mu_matrix: torch.Tensor,
    theta: float,
    shape: float,
) -> InclusionFactors:
    poisson = (3.0 * k_matrix - 2.0 * mu_matrix) / (2.0 * (3.0 * k_matrix + mu_matrix))
    r_value = (1.0 - 2.0 * poisson) / (2.0 * (1.0 - poisson))
    a_value = -1.0
    b_value = 0.0
    three_minus_four_r = 3.0 - 4.0 * r_value
    f_1 = 1.0 + a_value * (
        1.5 * (shape + theta) - r_value * (1.5 * shape + 2.5 * theta - 4.0 / 3.0)
    )
    f_2 = (
        1.0
        + a_value * (1.0 + 1.5 * (shape + theta) - (r_value / 2.0) * (3.0 * shape + 5.0 * theta))
        + b_value * three_minus_four_r
        + (a_value / 2.0)
        * (a_value + 3.0 * b_value)
        * three_minus_four_r
        * (shape + theta - r_value * (shape - theta + 2.0 * theta * theta))
    )
    f_3 = 1.0 + a_value * (1.0 - (shape + 1.5 * theta) + r_value * (shape + theta))
    f_4 = 1.0 + (a_value / 4.0) * (shape + 3.0 * theta - r_value * (shape - theta))
    f_5 = (
        a_value * (-shape + r_value * (shape + theta - 4.0 / 3.0))
        + b_value * theta * three_minus_four_r
    )
    f_6 = (
        1.0
        + a_value * (1.0 + shape - r_value * (shape + theta))
        + b_value * (1.0 - theta) * three_minus_four_r
    )
    f_7 = (
        2.0
        + (a_value / 4.0) * (3.0 * shape + 9.0 * theta - r_value * (3.0 * shape + 5.0 * theta))
        + b_value * theta * three_minus_four_r
    )
    f_8 = (
        a_value
        * (
            1.0
            - 2.0 * r_value
            + (shape / 2.0) * (r_value - 1.0)
            + (theta / 2.0) * (5.0 * r_value - 3.0)
        )
        + b_value * (1.0 - theta) * three_minus_four_r
    )
    f_9 = a_value * ((r_value - 1.0) * shape - r_value * theta) + (
        b_value * theta * three_minus_four_r
    )
    trace = 3.0 * f_1 / f_2
    shear_term = 2.0 / f_3 + 1.0 / f_4 + (f_4 * f_5 + f_6 * f_7 - f_8 * f_9) / (f_2 * f_4)
    return InclusionFactors(trace / 3.0, shear_term / 5.0)


def _apply_dilute_factors(
    k_matrix: torch.Tensor,
    mu_matrix: torch.Tensor,
    fraction: torch.Tensor,
    factors: InclusionFactors,
) -> tuple[torch.Tensor, torch.Tensor]:
    zeta = (mu_matrix / 6.0) * ((9.0 * k_matrix + 8.0 * mu_matrix) / (k_matrix + 2.0 * mu_matrix))
    bulk_ratio = -fraction * k_matrix * factors.p / (k_matrix + (4.0 / 3.0) * mu_matrix)
    shear_ratio = -fraction * mu_matrix * factors.q / (mu_matrix + zeta)
    k_eff = (k_matrix + bulk_ratio * (4.0 / 3.0) * mu_matrix) / (1.0 - bulk_ratio)
    mu_eff = (mu_matrix + shear_ratio * zeta) / (1.0 - shear_ratio)
    return k_eff, mu_eff


def _xu_white_integrate(
    k_mineral: torch.Tensor,
    mu_mineral: torch.Tensor,
    sand_porosity: torch.Tensor,
    clay_porosity: torch.Tensor,
    sand_shape: tuple[float, float],
    clay_shape: tuple[float, float],
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    k_eff = k_mineral
    mu_eff = mu_mineral
    sand_increment = sand_porosity / steps
    clay_increment = clay_porosity / steps
    for _ in range(steps):
        factors = _oblate_factors(k_eff, mu_eff, *sand_shape)
        k_eff, mu_eff = _apply_dilute_factors(k_eff, mu_eff, sand_increment, factors)
    for _ in range(steps):
        factors = _oblate_factors(k_eff, mu_eff, *clay_shape)
        k_eff, mu_eff = _apply_dilute_factors(k_eff, mu_eff, clay_increment, factors)
    return k_eff, mu_eff


def xu_white_moduli(
    shale_volume: torch.Tensor,
    porosity: TensorInput,
    *,
    k_quartz: float = 37.0e9,
    mu_quartz: float = 44.0e9,
    rho_quartz: float = 2650.0,
    k_clay: float = 21.0e9,
    mu_clay: float = 7.0e9,
    rho_clay: float = 2580.0,
    sand_aspect_ratio: float = 0.12,
    clay_aspect_ratio: float = 0.03,
    steps: int = 20,
    tolerance: float = 5.0e-2,
) -> XuWhiteResult:
    """Return Xu--White dry-frame properties with step-refinement evidence."""

    object_name = "xu_white_moduli"
    validated_steps = _positive_int(object_name, "steps", steps)
    validated_tolerance = _positive_float(object_name, "tolerance", tolerance)
    shale, phi = _validated_inputs(
        object_name,
        ("shale_volume", shale_volume),
        ("porosity", porosity),
    )
    _require_fraction(object_name, "shale_volume", shale)
    _require_fraction(object_name, "porosity", phi)
    if bool(torch.any(phi >= 1.0)):
        raise RockContractError(
            "Xu--White porosity must stay below one",
            object_name=object_name,
            field="porosity",
            expected="0 <= value < 1",
            actual=_extrema(phi),
        )
    if bool(torch.any(shale + phi > 1.0)):
        raise RockContractError(
            "shale volume plus porosity cannot exceed the bulk volume",
            object_name=object_name,
            field="shale_volume + porosity",
            expected="<= 1",
            actual=_extrema(shale + phi),
        )

    constants = {
        "k_quartz": k_quartz,
        "mu_quartz": mu_quartz,
        "rho_quartz": rho_quartz,
        "k_clay": k_clay,
        "mu_clay": mu_clay,
        "rho_clay": rho_clay,
    }
    for field, value in constants.items():
        _positive_float(object_name, field, value)
    for field, value in (
        ("sand_aspect_ratio", sand_aspect_ratio),
        ("clay_aspect_ratio", clay_aspect_ratio),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
        ):
            raise RockContractError(
                "pore aspect ratio must lie strictly between zero and one",
                object_name=object_name,
                field=field,
                expected="0 < value < 1",
                actual=value,
            )

    solid_fraction = 1.0 - phi
    clay_solid_fraction = shale / solid_fraction
    quartz_solid_fraction = 1.0 - clay_solid_fraction
    k_voigt = quartz_solid_fraction * k_quartz + clay_solid_fraction * k_clay
    k_reuss = 1.0 / (quartz_solid_fraction / k_quartz + clay_solid_fraction / k_clay)
    mu_voigt = quartz_solid_fraction * mu_quartz + clay_solid_fraction * mu_clay
    mu_reuss = 1.0 / (quartz_solid_fraction / mu_quartz + clay_solid_fraction / mu_clay)
    k_mineral = 0.5 * (k_voigt + k_reuss)
    mu_mineral = 0.5 * (mu_voigt + mu_reuss)
    sand_porosity = phi * quartz_solid_fraction
    clay_porosity = phi * clay_solid_fraction
    sand_shape = _oblate_shape_coefficients(float(sand_aspect_ratio))
    clay_shape = _oblate_shape_coefficients(float(clay_aspect_ratio))
    coarse_k, coarse_mu = _xu_white_integrate(
        k_mineral,
        mu_mineral,
        sand_porosity,
        clay_porosity,
        sand_shape,
        clay_shape,
        validated_steps,
    )
    fine_k, fine_mu = _xu_white_integrate(
        k_mineral,
        mu_mineral,
        sand_porosity,
        clay_porosity,
        sand_shape,
        clay_shape,
        2 * validated_steps,
    )
    residual = torch.maximum(
        (fine_k - coarse_k).abs(),
        (fine_mu - coarse_mu).abs(),
    ) / _safe_scale(fine_k, fine_mu)
    rho_dry = solid_fraction * (quartz_solid_fraction * rho_quartz + clay_solid_fraction * rho_clay)
    _require_finite_moduli(object_name, fine_k, fine_mu)
    if not bool(torch.isfinite(rho_dry).all()) or bool(torch.any(rho_dry <= 0.0)):
        raise RockNumericsError(
            "Xu--White calculation produced invalid dry density",
            object_name=object_name,
            field="rho_dry",
            expected="> 0 kg/m^3",
            actual=_extrema(rho_dry),
        )
    report = RockIterationReport(
        residual <= validated_tolerance,
        residual,
        4 * validated_steps,
        validated_tolerance,
    )
    return XuWhiteResult(fine_k, fine_mu, rho_dry, report)


def require_converged(result: EffectiveModuliResult, *, object_name: str) -> None:
    """Raise a structured numerical error when any broadcast item failed."""

    converged = result.iteration.converged
    if not bool(converged.all()):
        residual = result.iteration.residual
        raise RockNumericsError(
            "effective-medium solver did not converge for every element",
            object_name=object_name,
            field="convergence",
            expected=f"residual <= {result.iteration.tolerance}",
            actual={
                "failed": int((~converged).sum().item()),
                "maximum_residual": residual.max().item(),
                "iterations": result.iteration.iterations,
            },
            hint="increase the iteration budget or relax the declared tolerance",
        )


__all__ = [
    "EffectiveModuliResult",
    "InclusionFactors",
    "XuWhiteResult",
    "differential_effective_medium",
    "dilute_crack_moduli",
    "ellipsoidal_inclusion_factors",
    "kuster_toksoz_moduli",
    "oconnell_budiansky_fluid_moduli",
    "require_converged",
    "sc_flex_moduli",
    "self_consistent_moduli",
    "self_consistent_dilute_moduli",
    "spherical_inclusion_factors",
    "swiss_cheese_moduli",
    "xu_white_moduli",
]
