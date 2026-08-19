"""Canonical SI petrophysical permeability and resistivity relations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TypeAlias, cast

import torch

from .contracts import require_compatible_tensors
from .errors import RockContractError

TensorInput: TypeAlias = torch.Tensor | int | float

MILLIDARCY_TO_SQUARE_METRE = 9.869233e-16


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
                "petrophysical relation requires a materialized strided tensor",
                object_name=object_name,
                field=field,
                expected="materialized torch.strided tensor",
                actual={"layout": str(tensor.layout), "device": str(tensor.device)},
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RockContractError(
                "petrophysical input must be finite",
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
            "petrophysical input must be positive",
            object_name=object_name,
            field=field,
            expected="> 0",
            actual=_extrema(value),
            hint="supply physical values without clipping",
        )


def _nonnegative(object_name: str, field: str, value: torch.Tensor) -> None:
    if bool(torch.any(value < 0.0)):
        raise RockContractError(
            "petrophysical input must be non-negative",
            object_name=object_name,
            field=field,
            expected=">= 0",
            actual=_extrema(value),
        )


def _open_porosity(object_name: str, porosity: torch.Tensor) -> None:
    if bool(torch.any((porosity <= 0.0) | (porosity >= 1.0))):
        raise RockContractError(
            "porosity must lie strictly between zero and one",
            object_name=object_name,
            field="porosity",
            expected="0 < porosity < 1",
            actual=_extrema(porosity),
            hint="supply physical porosity without clipping",
        )


def archie_resistivity(
    porosity: torch.Tensor,
    water_saturation: TensorInput,
    water_resistivity: TensorInput,
    *,
    tortuosity_factor: TensorInput = 1.0,
    cementation_exponent: TensorInput = 2.0,
    saturation_exponent: TensorInput = 2.0,
) -> torch.Tensor:
    """Return Archie formation resistivity in ohm metre.

    Args:
        porosity: fractional porosity.
        water_saturation: fractional water saturation.
        water_resistivity: brine resistivity [Ohm·m].
        tortuosity_factor: Archie ``a``.
        cementation_exponent: Archie ``m``.
        saturation_exponent: Archie ``n``.
    """

    porosity, saturation, rw, a, m, n = _inputs(
        "archie_resistivity",
        ("porosity", porosity),
        ("water_saturation", water_saturation),
        ("water_resistivity", water_resistivity),
        ("tortuosity_factor", tortuosity_factor),
        ("cementation_exponent", cementation_exponent),
        ("saturation_exponent", saturation_exponent),
    )
    _open_porosity("archie_resistivity", porosity)
    if bool(torch.any((saturation <= 0.0) | (saturation > 1.0))):
        raise RockContractError(
            "water saturation is outside the Archie calibration",
            object_name="archie_resistivity",
            field="water_saturation",
            expected="0 < water_saturation <= 1",
            actual=_extrema(saturation),
            hint="zero saturation is singular; do not clip it",
        )
    for field, value in (("water_resistivity", rw), ("tortuosity_factor", a), ("cementation_exponent", m), ("saturation_exponent", n)):
        _positive("archie_resistivity", field, value)
    return a * rw * porosity.pow(-m) * saturation.pow(-n)


def archie_conductivity(
    porosity: torch.Tensor,
    water_saturation: TensorInput,
    water_conductivity: TensorInput,
    *,
    tortuosity_factor: TensorInput = 1.0,
    cementation_exponent: TensorInput = 2.0,
    saturation_exponent: TensorInput = 2.0,
) -> torch.Tensor:
    """Return Archie conductivity in S/m, including the dry zero limit."""

    porosity, saturation, sigma_w, a, m, n = _inputs(
        "archie_conductivity",
        ("porosity", porosity),
        ("water_saturation", water_saturation),
        ("water_conductivity", water_conductivity),
        ("tortuosity_factor", tortuosity_factor),
        ("cementation_exponent", cementation_exponent),
        ("saturation_exponent", saturation_exponent),
    )
    _open_porosity("archie_conductivity", porosity)
    if bool(torch.any((saturation < 0.0) | (saturation > 1.0))):
        raise RockContractError(
            "water saturation is outside the Archie calibration",
            object_name="archie_conductivity",
            field="water_saturation",
            expected="0 <= water_saturation <= 1",
            actual=_extrema(saturation),
        )
    for field, value in (("water_conductivity", sigma_w), ("tortuosity_factor", a), ("cementation_exponent", m), ("saturation_exponent", n)):
        _positive("archie_conductivity", field, value)
    return sigma_w * porosity.pow(m) * saturation.pow(n) / a


def archie_formation_factor(
    porosity: torch.Tensor,
    *,
    tortuosity_factor: TensorInput = 1.0,
    cementation_exponent: TensorInput = 2.0,
) -> torch.Tensor:
    """Return Archie formation factor."""

    porosity, a, m = _inputs(
        "archie_formation_factor",
        ("porosity", porosity),
        ("tortuosity_factor", tortuosity_factor),
        ("cementation_exponent", cementation_exponent),
    )
    _open_porosity("archie_formation_factor", porosity)
    _positive("archie_formation_factor", "tortuosity_factor", a)
    _positive("archie_formation_factor", "cementation_exponent", m)
    return a * porosity.pow(-m)


def archie_resistivity_index(
    water_saturation: torch.Tensor,
    *,
    saturation_exponent: TensorInput = 2.0,
) -> torch.Tensor:
    """Return the Archie resistivity index."""

    saturation, exponent = _inputs(
        "archie_resistivity_index",
        ("water_saturation", water_saturation),
        ("saturation_exponent", saturation_exponent),
    )
    if bool(torch.any((saturation <= 0.0) | (saturation > 1.0))):
        raise RockContractError(
            "water saturation is outside the Archie calibration",
            object_name="archie_resistivity_index",
            field="water_saturation",
            expected="0 < water_saturation <= 1",
            actual=_extrema(saturation),
        )
    _positive("archie_resistivity_index", "saturation_exponent", exponent)
    return saturation.pow(-exponent)


def archie_water_saturation(
    formation_resistivity: torch.Tensor,
    porosity: TensorInput,
    water_resistivity: TensorInput,
    *,
    tortuosity_factor: TensorInput = 1.0,
    cementation_exponent: TensorInput = 2.0,
    saturation_exponent: TensorInput = 2.0,
) -> torch.Tensor:
    """Invert Archie's law without clipping an out-of-domain solution."""

    rt, porosity, rw, a, m, n = _inputs(
        "archie_water_saturation",
        ("formation_resistivity", formation_resistivity),
        ("porosity", porosity),
        ("water_resistivity", water_resistivity),
        ("tortuosity_factor", tortuosity_factor),
        ("cementation_exponent", cementation_exponent),
        ("saturation_exponent", saturation_exponent),
    )
    _positive("archie_water_saturation", "formation_resistivity", rt)
    _open_porosity("archie_water_saturation", porosity)
    for field, value in (("water_resistivity", rw), ("tortuosity_factor", a), ("cementation_exponent", m), ("saturation_exponent", n)):
        _positive("archie_water_saturation", field, value)
    saturation = (a * rw / (rt * porosity.pow(m))).pow(n.reciprocal())
    if bool(torch.any((saturation <= 0.0) | (saturation > 1.0))):
        raise RockContractError(
            "inverted Archie saturation lies outside its physical domain",
            object_name="archie_water_saturation",
            field="water_saturation",
            expected="0 < water_saturation <= 1",
            actual=_extrema(saturation),
        )
    return saturation


def kozeny_carman_permeability(
    porosity: torch.Tensor,
    grain_diameter: TensorInput,
    *,
    kozeny_constant: TensorInput = 180.0,
) -> torch.Tensor:
    """Return Kozeny-Carman permeability in m2 for grain diameter in m.

    Args:
        porosity: fractional porosity.
        grain_diameter: representative grain diameter [m].
        kozeny_constant: Kozeny-Carman constant (default 180).
    """

    porosity, diameter, constant = _inputs(
        "kozeny_carman_permeability",
        ("porosity", porosity),
        ("grain_diameter", grain_diameter),
        ("kozeny_constant", kozeny_constant),
    )
    _open_porosity("kozeny_carman_permeability", porosity)
    _positive("kozeny_carman_permeability", "grain_diameter", diameter)
    _positive("kozeny_carman_permeability", "kozeny_constant", constant)
    return diameter.square() * porosity.pow(3.0) / (constant * (1.0 - porosity).square())


def kozeny_carman_percolation_permeability(
    porosity: torch.Tensor,
    percolation_porosity: TensorInput,
    grain_diameter: TensorInput,
    coefficient: TensorInput,
) -> torch.Tensor:
    """Return percolation-corrected permeability in m2."""

    porosity, threshold, diameter, coefficient = _inputs(
        "kozeny_carman_percolation_permeability",
        ("porosity", porosity),
        ("percolation_porosity", percolation_porosity),
        ("grain_diameter", grain_diameter),
        ("coefficient", coefficient),
    )
    _open_porosity("kozeny_carman_percolation_permeability", porosity)
    if bool(torch.any((threshold < 0.0) | (threshold >= 1.0))):
        raise RockContractError(
            "percolation porosity is outside its physical domain",
            object_name="kozeny_carman_percolation_permeability",
            field="percolation_porosity",
            expected="0 <= value < 1",
            actual=_extrema(threshold),
        )
    if bool(torch.any(porosity < threshold)):
        raise RockContractError(
            "porosity is below the calibrated percolation threshold",
            object_name="kozeny_carman_percolation_permeability",
            field="porosity",
            expected="porosity >= percolation_porosity",
            actual=_extrema(porosity),
        )
    _positive("kozeny_carman_percolation_permeability", "grain_diameter", diameter)
    _positive("kozeny_carman_percolation_permeability", "coefficient", coefficient)
    delta = porosity - threshold
    return coefficient * diameter.square() * delta.pow(3.0) / (1.0 + threshold - porosity).square()


def perm_logs(
    porosity: torch.Tensor,
    irreducible_water_saturation: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Tixier, Timur, Coates and Coates-Dumanoir in m2."""

    porosity, saturation = _inputs(
        "perm_logs",
        ("porosity", porosity),
        ("irreducible_water_saturation", irreducible_water_saturation),
    )
    _open_porosity("perm_logs", porosity)
    if bool(torch.any((saturation <= 0.0) | (saturation > 1.0))):
        raise RockContractError(
            "irreducible saturation is outside the correlation domain",
            object_name="perm_logs",
            field="irreducible_water_saturation",
            expected="0 < value <= 1",
            actual=_extrema(saturation),
        )
    factor = porosity.new_tensor(MILLIDARCY_TO_SQUARE_METRE)
    tixier = 62500.0 * porosity.pow(6.0) / saturation.square()
    timur = 10000.0 * porosity.pow(4.5) / saturation.square()
    coates = (
        10000.0
        * porosity.pow(4.0)
        * (1.0 - saturation).square()
        / saturation.square()
    )
    coates_dumanoir = 352.0 * porosity.pow(4.0) / saturation.pow(4.0)
    return (
        tixier * factor,
        timur * factor,
        coates * factor,
        coates_dumanoir * factor,
    )


def owolabi_permeability(
    porosity: torch.Tensor,
    irreducible_water_saturation: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Owolabi oil and gas permeability correlations in m2."""

    porosity, saturation = _inputs(
        "owolabi_permeability",
        ("porosity", porosity),
        ("irreducible_water_saturation", irreducible_water_saturation),
    )
    _open_porosity("owolabi_permeability", porosity)
    if bool(torch.any((saturation < 0.0) | (saturation > 1.0))):
        raise RockContractError(
            "irreducible water saturation is outside its physical domain",
            object_name="owolabi_permeability",
            field="irreducible_water_saturation",
            expected="0 <= value <= 1",
            actual=_extrema(saturation),
        )
    shared = (porosity * saturation).square()
    oil_millidarcy = 307.0 + 26552.0 * porosity.square() - 34540.0 * shared
    gas_millidarcy = 30.7 + 2655.0 * porosity.square() - 3454.0 * shared
    if bool(torch.any(oil_millidarcy < 0.0)) or bool(torch.any(gas_millidarcy < 0.0)):
        raise RockContractError(
            "Owolabi inputs produce negative permeability outside calibration",
            object_name="owolabi_permeability",
            field="porosity/irreducible_water_saturation",
            expected="non-negative oil and gas permeability",
            actual={
                "oil_minimum_mD": oil_millidarcy.amin().item(),
                "gas_minimum_mD": gas_millidarcy.amin().item(),
            },
        )
    factor = porosity.new_tensor(MILLIDARCY_TO_SQUARE_METRE)
    return oil_millidarcy * factor, gas_millidarcy * factor


def panda_lake_permeability(
    grain_diameter: torch.Tensor,
    sorting_coefficient: TensorInput,
    specific_surface_factor: TensorInput,
    tortuosity: TensorInput,
    porosity: TensorInput,
) -> torch.Tensor:
    """Return Panda-Lake permeability in the units implied by grain diameter."""

    diameter, cement, surface, tortuosity, porosity = _inputs(
        "panda_lake_permeability",
        ("grain_diameter", grain_diameter),
        ("sorting_coefficient", sorting_coefficient),
        ("specific_surface_factor", specific_surface_factor),
        ("tortuosity", tortuosity),
        ("porosity", porosity),
    )
    _positive("panda_lake_permeability", "grain_diameter", diameter)
    _nonnegative("panda_lake_permeability", "sorting_coefficient", cement)
    _positive("panda_lake_permeability", "specific_surface_factor", surface)
    _positive("panda_lake_permeability", "tortuosity", tortuosity)
    _open_porosity("panda_lake_permeability", porosity)
    numerator = diameter.square() * porosity.pow(3.0) * (
        cement.pow(3.0) * surface + 3.0 * cement.square() + 1.0
    ).square()
    denominator = 72.0 * tortuosity * (1.0 - porosity).square() * (
        cement.square() + 1.0
    ).square()
    return numerator / denominator


def panda_lake_cemented_permeability(
    porosity: torch.Tensor,
    grain_diameter: TensorInput,
) -> torch.Tensor:
    """Return the Panda-Lake cemented-sand correlation."""

    porosity, diameter = _inputs(
        "panda_lake_cemented_permeability",
        ("porosity", porosity),
        ("grain_diameter", grain_diameter),
    )
    _open_porosity("panda_lake_cemented_permeability", porosity)
    _positive("panda_lake_cemented_permeability", "grain_diameter", diameter)
    return 3.34 * diameter.square() * porosity.pow(3.0) / (1.0 - porosity).square()


def revil_permeability(
    porosity: torch.Tensor,
    grain_diameter: TensorInput,
) -> torch.Tensor:
    """Return the Revil shaly-rock permeability correlation."""

    porosity, diameter = _inputs(
        "revil_permeability",
        ("porosity", porosity),
        ("grain_diameter", grain_diameter),
    )
    _open_porosity("revil_permeability", porosity)
    _positive("revil_permeability", "grain_diameter", diameter)
    return 1000.0 * diameter.square() * porosity.pow(4.5) / 24.0


def fredrich_permeability(
    porosity: torch.Tensor,
    grain_diameter: TensorInput,
    geometry_factor: TensorInput,
) -> torch.Tensor:
    """Return the Fredrich pore-geometry permeability correlation."""

    porosity, diameter, factor = _inputs(
        "fredrich_permeability",
        ("porosity", porosity),
        ("grain_diameter", grain_diameter),
        ("geometry_factor", geometry_factor),
    )
    _open_porosity("fredrich_permeability", porosity)
    _positive("fredrich_permeability", "grain_diameter", diameter)
    _positive("fredrich_permeability", "geometry_factor", factor)
    formation_factor = 2.5 / porosity
    specific_surface = 6.0 * (1.0 - porosity) / diameter
    return (porosity / specific_surface).square() / (factor * formation_factor)


def bloch_porosity_permeability(
    sorting: torch.Tensor,
    clay_fraction: TensorInput,
    grain_size_index: TensorInput,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Bloch porosity as a fraction and permeability in m2."""

    sorting, clay, grain_index = _inputs(
        "bloch_porosity_permeability",
        ("sorting", sorting),
        ("clay_fraction", clay_fraction),
        ("grain_size_index", grain_size_index),
    )
    _positive("bloch_porosity_permeability", "sorting", sorting)
    if bool(torch.any((clay < 0.0) | (clay > 1.0))):
        raise RockContractError(
            "clay fraction is outside its physical domain",
            object_name="bloch_porosity_permeability",
            field="clay_fraction",
            expected="0 <= value <= 1",
            actual=_extrema(clay),
        )
    _positive("bloch_porosity_permeability", "grain_size_index", grain_index)
    clay_percent = clay * 100.0
    porosity_percent = -6.1 + 9.8 / sorting + 0.17 * clay_percent
    permeability_millidarcy = 10.0 ** (
        -4.67 + 1.34 * grain_index + 4.08 / sorting + 3.42 * clay
    )
    if bool(torch.any((porosity_percent <= 0.0) | (porosity_percent >= 100.0))):
        raise RockContractError(
            "Bloch inputs produce porosity outside the physical fraction domain",
            object_name="bloch_porosity_permeability",
            field="porosity",
            expected="0 < porosity_percent < 100",
            actual=_extrema(porosity_percent),
        )
    return (
        porosity_percent / 100.0,
        permeability_millidarcy * sorting.new_tensor(MILLIDARCY_TO_SQUARE_METRE),
    )


def bernabe_permeability(
    porosity: torch.Tensor,
    crack_fraction: TensorInput,
    crack_aperture: TensorInput,
    tube_radius: TensorInput,
) -> torch.Tensor:
    """Return Bernabe dual-porosity permeability."""

    porosity, crack_fraction, aperture, radius = _inputs(
        "bernabe_permeability",
        ("porosity", porosity),
        ("crack_fraction", crack_fraction),
        ("crack_aperture", crack_aperture),
        ("tube_radius", tube_radius),
    )
    _open_porosity("bernabe_permeability", porosity)
    if bool(torch.any((crack_fraction < 0.0) | (crack_fraction > 1.0))):
        raise RockContractError(
            "crack fraction is outside its physical domain",
            object_name="bernabe_permeability",
            field="crack_fraction",
            expected="0 <= value <= 1",
            actual=_extrema(crack_fraction),
        )
    _positive("bernabe_permeability", "crack_aperture", aperture)
    _positive("bernabe_permeability", "tube_radius", radius)
    crack_porosity = porosity * crack_fraction
    tube_porosity = porosity - crack_porosity
    return aperture.square() * crack_porosity / 30.0 + radius.square() * tube_porosity / 20.0


__all__ = [
    "MILLIDARCY_TO_SQUARE_METRE",
    "archie_conductivity",
    "archie_formation_factor",
    "archie_resistivity",
    "archie_resistivity_index",
    "archie_water_saturation",
    "bernabe_permeability",
    "bloch_porosity_permeability",
    "fredrich_permeability",
    "kozeny_carman_percolation_permeability",
    "kozeny_carman_permeability",
    "owolabi_permeability",
    "panda_lake_cemented_permeability",
    "panda_lake_permeability",
    "perm_logs",
    "revil_permeability",
]
