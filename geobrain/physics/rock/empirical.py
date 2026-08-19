"""Canonical SI empirical rock-physics relations.

The functions in this module are the only scientific implementations of the
empirical relations the platform retains as canon. Operator facades and
compatibility adapters delegate to these tensor-first kernels.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TypeAlias, cast

import torch

from .contracts import require_compatible_tensors
from .errors import RockContractError

TensorInput: TypeAlias = torch.Tensor | int | float


def _inputs(
    object_name: str,
    *fields: tuple[str, TensorInput],
) -> tuple[torch.Tensor, ...]:
    tensors = cast(
        tuple[torch.Tensor, ...], require_compatible_tensors(object_name, *fields)
    )
    for (field, _), tensor in zip(fields, tensors, strict=True):
        if tensor.layout is not torch.strided or tensor.device.type == "meta":
            raise RockContractError(
                "empirical relation requires a materialized strided tensor",
                object_name=object_name,
                field=field,
                expected="materialized torch.strided tensor",
                actual={"layout": str(tensor.layout), "device": str(tensor.device)},
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "empirical input must be finite",
                object_name=object_name,
                field=field,
                expected="finite values",
                actual="non-finite value(s)",
            )
    return tensors


def _extrema(value: torch.Tensor) -> dict[str, float]:
    return {"minimum": value.amin().item(), "maximum": value.amax().item()}


def _positive(object_name: str, field: str, value: torch.Tensor) -> None:
    if bool(torch.any(value <= 0.0)):
        raise RockContractError(
            "empirical input must be positive",
            object_name=object_name,
            field=field,
            expected="> 0",
            actual=_extrema(value),
        )


def _unit_interval(object_name: str, field: str, value: torch.Tensor) -> None:
    if bool(torch.any((value < 0.0) | (value > 1.0))):
        raise RockContractError(
            "empirical fraction must lie in the unit interval",
            object_name=object_name,
            field=field,
            expected="0 <= value <= 1",
            actual=_extrema(value),
            hint="supply physical fractions without clipping",
        )


def gardner_density(
    vp: torch.Tensor,
    coefficient: TensorInput = 310.0,
    exponent: TensorInput = 0.25,
) -> torch.Tensor:
    """Return Gardner density in kg/m3 for ``vp`` in m/s.

    Args:
        vp: P-wave velocity [m/s].
        coefficient: Gardner ``a`` (SI form; default 310).
        exponent: Gardner ``b`` (default 0.25).
    """

    vp, coefficient, exponent = _inputs(
        "gardner_density",
        ("vp", vp),
        ("coefficient", coefficient),
        ("exponent", exponent),
    )
    _positive("gardner_density", "vp", vp)
    _positive("gardner_density", "coefficient", coefficient)
    _positive("gardner_density", "exponent", exponent)
    return coefficient * vp.pow(exponent)


def castagna_shear_velocity(
    vp: torch.Tensor,
    slope: TensorInput = 0.804,
    intercept: TensorInput = -856.0,
) -> torch.Tensor:
    """Return the Castagna mudrock-line shear velocity in m/s."""

    vp, slope, intercept = _inputs(
        "castagna_shear_velocity",
        ("vp", vp),
        ("slope", slope),
        ("intercept", intercept),
    )
    _positive("castagna_shear_velocity", "vp", vp)
    _positive("castagna_shear_velocity", "slope", slope)
    result = slope * vp + intercept
    if bool(torch.any(result <= 0.0)):
        raise RockContractError(
            "Castagna relation is outside its positive-velocity calibration",
            object_name="castagna_shear_velocity",
            field="vp",
            expected="slope * vp + intercept > 0 m/s",
            actual=_extrema(vp),
        )
    return result


_GC_COEFFICIENTS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.80416, -0.85588),
    (0.0, 0.76969, -0.86735),
    (-0.05509, 1.01677, -1.03049),
    (0.0, 0.58321, -0.07775),
)


def greenberg_castagna_shear_velocity(
    vp: torch.Tensor,
    sand_fraction: TensorInput,
    shale_fraction: TensorInput,
    limestone_fraction: TensorInput,
    dolomite_fraction: TensorInput,
) -> torch.Tensor:
    """Return the Greenberg-Castagna four-mineral Hill average in m/s."""

    vp, sand, shale, limestone, dolomite = _inputs(
        "greenberg_castagna_shear_velocity",
        ("vp", vp),
        ("sand_fraction", sand_fraction),
        ("shale_fraction", shale_fraction),
        ("limestone_fraction", limestone_fraction),
        ("dolomite_fraction", dolomite_fraction),
    )
    _positive("greenberg_castagna_shear_velocity", "vp", vp)
    for field, fraction in (
        ("sand_fraction", sand),
        ("shale_fraction", shale),
        ("limestone_fraction", limestone),
        ("dolomite_fraction", dolomite),
    ):
        if bool(torch.any(fraction < 0.0)):
            raise RockContractError(
                "mineral fraction must be non-negative",
                object_name="greenberg_castagna_shear_velocity",
                field=field,
                expected=">= 0",
                actual=_extrema(fraction),
            )
    total = sand + shale + limestone + dolomite
    _positive("greenberg_castagna_shear_velocity", "mineral_fraction_sum", total)
    fractions = tuple(item / total for item in (sand, shale, limestone, dolomite))
    vp_km_s = vp / vp.new_tensor(1000.0)
    velocities = tuple(
        (vp_km_s.square() * a + vp_km_s * b + c) * vp.new_tensor(1000.0)
        for a, b, c in _GC_COEFFICIENTS
    )
    for velocity in velocities:
        _positive("greenberg_castagna_shear_velocity", "calibrated_vs", velocity)

    # The canonical mixture helper is two-phase; fold the exact Voigt/Reuss
    # sums explicitly for the published four-mineral Hill construction.
    voigt = torch.zeros_like(vp)
    reciprocal = torch.zeros_like(vp)
    for fraction, velocity in zip(fractions, velocities, strict=True):
        voigt = voigt + fraction * velocity
        reciprocal = reciprocal + fraction / velocity
    reuss = reciprocal.reciprocal()
    return (voigt + reuss) * vp.new_tensor(0.5)


def han_shaly_sandstone(
    porosity: torch.Tensor,
    clay_fraction: TensorInput,
    *,
    vp_coefficients: tuple[float, float, float] = (5.59, -6.93, -2.18),
    vs_coefficients: tuple[float, float, float] = (3.52, -4.91, -1.89),
    mineral_density: TensorInput = 2650.0,
    fluid_density: TensorInput = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Han (1986) ``vp``, ``vs`` and density in SI units."""

    if len(vp_coefficients) != 3 or len(vs_coefficients) != 3:
        raise RockContractError(
            "Han coefficient vectors must contain three entries",
            object_name="han_shaly_sandstone",
            field="coefficients",
            expected="(intercept, porosity slope, clay slope)",
            actual=(vp_coefficients, vs_coefficients),
        )
    porosity, clay, mineral_density, fluid_density = _inputs(
        "han_shaly_sandstone",
        ("porosity", porosity),
        ("clay_fraction", clay_fraction),
        ("mineral_density", mineral_density),
        ("fluid_density", fluid_density),
    )
    _unit_interval("han_shaly_sandstone", "porosity", porosity)
    _unit_interval("han_shaly_sandstone", "clay_fraction", clay)
    _positive("han_shaly_sandstone", "mineral_density", mineral_density)
    _positive("han_shaly_sandstone", "fluid_density", fluid_density)
    ap, bp, cp = vp_coefficients
    ass, bs, cs = vs_coefficients
    vp = (ap + bp * porosity + cp * clay) * porosity.new_tensor(1000.0)
    vs = (ass + bs * porosity + cs * clay) * porosity.new_tensor(1000.0)
    _positive("han_shaly_sandstone", "vp", vp)
    _positive("han_shaly_sandstone", "vs", vs)
    rho = mineral_density * (1.0 - porosity) + fluid_density * porosity
    return vp, vs, rho


def macbeth_dry_moduli(
    effective_pressure: torch.Tensor,
    *,
    k_infinite: TensorInput = 24.0e9,
    mu_infinite: TensorInput = 18.0e9,
    delta_k: TensorInput = 9.0e9,
    delta_mu: TensorInput = 6.0e9,
    pressure_scale_k: TensorInput = 20.0e6,
    pressure_scale_mu: TensorInput = 25.0e6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return MacBeth pressure-sensitive dry moduli in Pa."""

    pressure, k_inf, mu_inf, dk, dmu, pk, pmu = _inputs(
        "macbeth_dry_moduli",
        ("effective_pressure", effective_pressure),
        ("k_infinite", k_infinite),
        ("mu_infinite", mu_infinite),
        ("delta_k", delta_k),
        ("delta_mu", delta_mu),
        ("pressure_scale_k", pressure_scale_k),
        ("pressure_scale_mu", pressure_scale_mu),
    )
    if bool(torch.any(pressure < 0.0)):
        raise RockContractError(
            "effective pressure must be non-negative",
            object_name="macbeth_dry_moduli",
            field="effective_pressure",
            expected=">= 0 Pa",
            actual=_extrema(pressure),
        )
    for field, value in (("k_infinite", k_inf), ("mu_infinite", mu_inf), ("pressure_scale_k", pk), ("pressure_scale_mu", pmu)):
        _positive("macbeth_dry_moduli", field, value)
    if bool(torch.any(dk < 0.0)) or bool(torch.any(dmu < 0.0)):
        raise RockContractError(
            "MacBeth modulus decrements must be non-negative",
            object_name="macbeth_dry_moduli",
            field="delta_moduli",
            expected=">= 0 Pa",
            actual={"delta_k": _extrema(dk), "delta_mu": _extrema(dmu)},
        )
    if bool(torch.any(k_inf <= dk)) or bool(torch.any(mu_inf <= dmu)):
        raise RockContractError(
            "MacBeth zero-pressure moduli must remain positive",
            object_name="macbeth_dry_moduli",
            field="asymptotic_moduli",
            expected="k_infinite > delta_k and mu_infinite > delta_mu",
            actual="non-positive zero-pressure modulus",
        )
    return k_inf - dk * torch.exp(-pressure / pk), mu_inf - dmu * torch.exp(-pressure / pmu)


def raymer_hunt_gardner_velocity(
    porosity: torch.Tensor,
    mineral_velocity: TensorInput = 5500.0,
    fluid_velocity: TensorInput = 1500.0,
) -> torch.Tensor:
    """Return Raymer-Hunt-Gardner P-wave velocity in m/s."""

    porosity, mineral, fluid = _inputs(
        "raymer_hunt_gardner_velocity",
        ("porosity", porosity),
        ("mineral_velocity", mineral_velocity),
        ("fluid_velocity", fluid_velocity),
    )
    _unit_interval("raymer_hunt_gardner_velocity", "porosity", porosity)
    _positive("raymer_hunt_gardner_velocity", "mineral_velocity", mineral)
    _positive("raymer_hunt_gardner_velocity", "fluid_velocity", fluid)
    return (1.0 - porosity).square() * mineral + porosity * fluid


def wyllie_time_average_velocity(
    porosity: torch.Tensor,
    mineral_velocity: TensorInput = 5500.0,
    fluid_velocity: TensorInput = 1500.0,
) -> torch.Tensor:
    """Return Wyllie's time-average P-wave velocity in m/s."""

    porosity, mineral, fluid = _inputs(
        "wyllie_time_average_velocity",
        ("porosity", porosity),
        ("mineral_velocity", mineral_velocity),
        ("fluid_velocity", fluid_velocity),
    )
    _unit_interval("wyllie_time_average_velocity", "porosity", porosity)
    _positive("wyllie_time_average_velocity", "mineral_velocity", mineral)
    _positive("wyllie_time_average_velocity", "fluid_velocity", fluid)
    return (porosity / fluid + (1.0 - porosity) / mineral).reciprocal()


def castagna_mudrock_velocity(vp: torch.Tensor) -> torch.Tensor:
    """Return the Castagna et al. mudrock-line shear velocity in m/s."""

    (vp,) = _inputs("castagna_mudrock_velocity", ("vp", vp))
    _positive("castagna_mudrock_velocity", "vp", vp)
    result = (vp.new_tensor(0.862) * (vp / 1000.0) - 1.172) * 1000.0
    _positive("castagna_mudrock_velocity", "calibrated_vs", result)
    return result


def krief_dry_moduli(
    porosity: torch.Tensor,
    k_mineral: TensorInput,
    mu_mineral: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Krief dry-frame moduli in Pa."""

    porosity, k_mineral, mu_mineral = _inputs(
        "krief_dry_moduli",
        ("porosity", porosity),
        ("k_mineral", k_mineral),
        ("mu_mineral", mu_mineral),
    )
    if bool(torch.any((porosity < 0.0) | (porosity >= 1.0))):
        raise RockContractError(
            "Krief porosity lies outside the calibrated half-open interval",
            object_name="krief_dry_moduli",
            field="porosity",
            expected="0 <= porosity < 1",
            actual=_extrema(porosity),
        )
    _positive("krief_dry_moduli", "k_mineral", k_mineral)
    _positive("krief_dry_moduli", "mu_mineral", mu_mineral)
    scale = (1.0 - porosity).pow(3.0 / (1.0 - porosity))
    return k_mineral * scale, mu_mineral * scale


def storvoll_velocity(depth: torch.Tensor) -> torch.Tensor:
    """Return the Storvoll velocity-depth trend in m/s for depth in m."""

    (depth,) = _inputs("storvoll_velocity", ("depth", depth))
    if bool(torch.any(depth < 0.0)):
        raise RockContractError(
            "depth must be non-negative",
            object_name="storvoll_velocity",
            field="depth",
            expected=">= 0 m",
            actual=_extrema(depth),
        )
    return depth / 1.76 + 2600.0


def japsen_velocity(depth: torch.Tensor) -> torch.Tensor:
    """Return the Japsen four-segment velocity-depth trend in m/s."""

    (depth,) = _inputs("japsen_velocity", ("depth", depth))
    if bool(torch.any(depth < 0.0)):
        raise RockContractError(
            "depth must be non-negative",
            object_name="japsen_velocity",
            field="depth",
            expected=">= 0 m",
            actual=_extrema(depth),
        )
    return torch.where(
        depth < 1393.0,
        1550.0 + 0.6 * depth,
        torch.where(
            depth < 2000.0,
            -400.0 + 2.0 * depth,
            torch.where(depth < 3500.0, 2600.0 + 0.5 * depth, 3475.0 + 0.25 * depth),
        ),
    )


def hillis_velocity(depth: torch.Tensor) -> torch.Tensor:
    """Return the Hillis transit-time velocity trend in m/s."""

    (depth,) = _inputs("hillis_velocity", ("depth", depth))
    if bool(torch.any((depth < 0.0) | (depth >= 6721.0))):
        raise RockContractError(
            "depth is outside the positive-transit-time calibration",
            object_name="hillis_velocity",
            field="depth",
            expected="0 <= depth < 6721 m",
            actual=_extrema(depth),
        )
    return 304800.0 / (135.9 - 20.22 * depth / 1000.0)


def scherbaum_velocity(depth: torch.Tensor) -> torch.Tensor:
    """Return the Scherbaum velocity-depth trend in m/s."""

    (depth,) = _inputs("scherbaum_velocity", ("depth", depth))
    if bool(torch.any(depth < 0.0)):
        raise RockContractError(
            "depth must be non-negative",
            object_name="scherbaum_velocity",
            field="depth",
            expected=">= 0 m",
            actual=_extrema(depth),
        )
    return 2325.0 + 0.51 * depth


def hjelstuen_velocity(depth: torch.Tensor) -> torch.Tensor:
    """Return the Hjelstuen velocity-depth trend in m/s."""

    (depth,) = _inputs("hjelstuen_velocity", ("depth", depth))
    if bool(torch.any(depth < 0.0)):
        raise RockContractError(
            "depth must be non-negative",
            object_name="hjelstuen_velocity",
            field="depth",
            expected=">= 0 m",
            actual=_extrema(depth),
        )
    return (1.87 + 0.55 * depth / 1000.0) * 1000.0


def ehrenberg_porosity(depth: torch.Tensor) -> torch.Tensor:
    """Return the Ehrenberg linear porosity-depth correlation."""

    (depth,) = _inputs("ehrenberg_porosity", ("depth", depth))
    if bool(torch.any((depth < 0.0) | (depth > 5206.0))):
        raise RockContractError(
            "depth is outside the non-negative Ehrenberg porosity domain",
            object_name="ehrenberg_porosity",
            field="depth",
            expected="0 <= depth <= 5206 m",
            actual=_extrema(depth),
        )
    return -0.0922 * depth / 1000.0 + 0.48


def ramm_porosity(depth: torch.Tensor, *, region: str = "haltenbanken") -> torch.Tensor:
    """Return the Ramm-Bjorlykke regional porosity-depth correlation."""

    if region not in {"haltenbanken", "north_sea"}:
        raise RockContractError(
            "unknown Ramm porosity region",
            object_name="ramm_porosity",
            field="region",
            expected="haltenbanken or north_sea",
            actual=region,
        )
    (depth,) = _inputs("ramm_porosity", ("depth", depth))
    if bool(torch.any(depth < 0.0)):
        raise RockContractError(
            "depth must be non-negative",
            object_name="ramm_porosity",
            field="depth",
            expected=">= 0 m",
            actual=_extrema(depth),
        )
    result = (46.4 - 0.0085 * depth) / 100.0 if region == "haltenbanken" else (42.7 - 0.0069 * depth) / 100.0
    if bool(torch.any(result < 0.0)):
        raise RockContractError(
            "depth is outside the non-negative Ramm porosity calibration",
            object_name="ramm_porosity",
            field="depth",
            expected="depth producing porosity >= 0",
            actual=_extrema(depth),
        )
    return result


def sclater_depth_from_porosity(porosity: torch.Tensor) -> torch.Tensor:
    """Return Sclater-Christie depth in metres from porosity."""

    (porosity,) = _inputs("sclater_depth_from_porosity", ("porosity", porosity))
    if bool(torch.any((porosity <= 0.0) | (porosity > 0.49))):
        raise RockContractError(
            "porosity is outside the Sclater-Christie calibration",
            object_name="sclater_depth_from_porosity",
            field="porosity",
            expected="0 < porosity <= 0.49",
            actual=_extrema(porosity),
        )
    depth_km = 3.7 * torch.log(0.49 / porosity)
    return depth_km * 1000.0


def sclater_porosity_from_depth(depth: torch.Tensor) -> torch.Tensor:
    """Return Sclater-Christie porosity for depth in metres."""

    (depth,) = _inputs("sclater_porosity_from_depth", ("depth", depth))
    if bool(torch.any(depth < 0.0)):
        raise RockContractError(
            "depth must be non-negative",
            object_name="sclater_porosity_from_depth",
            field="depth",
            expected=">= 0 m",
            actual=_extrema(depth),
        )
    depth_km = depth / 1000.0
    return 0.49 * torch.exp(-depth_km / 3.7)


def st_peter_velocities(
    effective_pressure: torch.Tensor,
    *,
    sample: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return St. Peter sandstone velocities for 0--150 MPa effective pressure.

    The public boundary is Pa; the Eberhart-Phillips (1989) coefficients use
    the published 0--1.5 kbar pressure coordinate internally.
    """

    if type(sample) is not int or sample not in (1, 2):
        raise RockContractError(
            "St. Peter sample must be one or two",
            object_name="st_peter_velocities",
            field="sample",
            expected="exact integer 1 or 2",
            actual=sample,
        )
    (pressure,) = _inputs(
        "st_peter_velocities", ("effective_pressure", effective_pressure)
    )
    if bool(torch.any((pressure < 0.0) | (pressure > 1.5e8))):
        raise RockContractError(
            "effective pressure is outside the St. Peter calibration",
            object_name="st_peter_velocities",
            field="effective_pressure",
            expected="0 <= value <= 1.5e8 Pa",
            actual=_extrema(pressure),
        )
    pressure_kbar = pressure / 1.0e8
    if sample == 1:
        vp = 4.21 + 0.187 * pressure_kbar - 0.746 * torch.exp(-24.0 * pressure_kbar)
        vs = 2.58 + 0.160 * pressure_kbar - 0.781 * torch.exp(-20.0 * pressure_kbar)
    else:
        vp = 4.55 + 0.298 * pressure_kbar - 0.8 * torch.exp(-19.0 * pressure_kbar)
        vs = 2.83 + 0.195 * pressure_kbar - 0.741 * torch.exp(-16.0 * pressure_kbar)
    return vp * 1000.0, vs * 1000.0


def bulk_density(
    porosity: torch.Tensor,
    mineral_density: TensorInput = 2650.0,
    fluid_density: TensorInput = 1000.0,
) -> torch.Tensor:
    """Return two-phase bulk density in kg/m3."""

    porosity, mineral, fluid = _inputs(
        "bulk_density",
        ("porosity", porosity),
        ("mineral_density", mineral_density),
        ("fluid_density", fluid_density),
    )
    _unit_interval("bulk_density", "porosity", porosity)
    _positive("bulk_density", "mineral_density", mineral)
    _positive("bulk_density", "fluid_density", fluid)
    return (1.0 - porosity) * mineral + porosity * fluid


__all__ = [
    "bulk_density",
    "castagna_mudrock_velocity",
    "castagna_shear_velocity",
    "ehrenberg_porosity",
    "gardner_density",
    "greenberg_castagna_shear_velocity",
    "han_shaly_sandstone",
    "hillis_velocity",
    "hjelstuen_velocity",
    "japsen_velocity",
    "krief_dry_moduli",
    "macbeth_dry_moduli",
    "ramm_porosity",
    "raymer_hunt_gardner_velocity",
    "scherbaum_velocity",
    "sclater_depth_from_porosity",
    "sclater_porosity_from_depth",
    "st_peter_velocities",
    "storvoll_velocity",
    "wyllie_time_average_velocity",
]
