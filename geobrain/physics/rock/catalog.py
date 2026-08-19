"""Immutable SI mineral/fluid catalog with canonical derived mixtures.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from . import mixtures
from .contracts import require_compatible_tensors
from .errors import RockContractError


@dataclass(frozen=True, slots=True)
class Mineral:
    """Immutable mineral properties with moduli in Pa and density in kg/m³."""

    name: str
    K: float
    G: float
    rho: float


@dataclass(frozen=True, slots=True)
class Fluid:
    """Immutable fluid properties with bulk modulus in Pa and density in kg/m³."""

    name: str
    K: float
    rho: float


_mineral_data = {
    "quartz": Mineral("Quartz", K=36.6e9, G=45.0e9, rho=2650.0),
    "feldspar": Mineral("Feldspar", K=37.5e9, G=15.0e9, rho=2620.0),
    "plagioclase": Mineral("Plagioclase", K=75.6e9, G=25.6e9, rho=2630.0),
    "mica": Mineral("Mica", K=42.0e9, G=22.0e9, rho=2790.0),
    "muscovite": Mineral("Muscovite", K=52.0e9, G=31.0e9, rho=2820.0),
    "biotite": Mineral("Biotite", K=50.0e9, G=28.0e9, rho=3050.0),
    "clay": Mineral("Clay (average)", K=21.0e9, G=7.0e9, rho=2580.0),
    "illite": Mineral("Illite", K=21.0e9, G=7.0e9, rho=2660.0),
    "kaolinite": Mineral("Kaolinite", K=1.5e9, G=1.4e9, rho=2600.0),
    "smectite": Mineral("Smectite", K=9.3e9, G=6.9e9, rho=2700.0),
    "montmorillonite": Mineral("Montmorillonite", K=9.3e9, G=6.9e9, rho=2500.0),
    "chlorite": Mineral("Chlorite", K=82.0e9, G=28.0e9, rho=2800.0),
    "calcite": Mineral("Calcite", K=76.8e9, G=32.0e9, rho=2710.0),
    "dolomite": Mineral("Dolomite", K=94.9e9, G=45.0e9, rho=2870.0),
    "aragonite": Mineral("Aragonite", K=47.0e9, G=39.0e9, rho=2930.0),
    "halite": Mineral("Halite", K=24.8e9, G=14.9e9, rho=2160.0),
    "anhydrite": Mineral("Anhydrite", K=56.1e9, G=29.1e9, rho=2980.0),
    "gypsum": Mineral("Gypsum", K=36.0e9, G=15.0e9, rho=2320.0),
    "kerogen": Mineral("Kerogen", K=2.9e9, G=2.7e9, rho=1300.0),
    "pyrite": Mineral("Pyrite", K=147.4e9, G=132.5e9, rho=4930.0),
    "siderite": Mineral("Siderite", K=124.0e9, G=51.0e9, rho=3960.0),
    "magnetite": Mineral("Magnetite", K=161.0e9, G=91.0e9, rho=5180.0),
    "hematite": Mineral("Hematite", K=100.0e9, G=95.0e9, rho=5240.0),
}

_fluid_data = {
    "water": Fluid("Water", K=2.25e9, rho=1000.0),
    "brine": Fluid("Brine", K=2.80e9, rho=1050.0),
    "brine_light": Fluid("Light Brine", K=2.50e9, rho=1020.0),
    "brine_heavy": Fluid("Heavy Brine", K=3.20e9, rho=1150.0),
    "oil": Fluid("Oil", K=1.00e9, rho=800.0),
    "oil_light": Fluid("Light Oil", K=1.20e9, rho=750.0),
    "oil_medium": Fluid("Medium Oil", K=1.00e9, rho=850.0),
    "oil_heavy": Fluid("Heavy Oil", K=0.70e9, rho=950.0),
    "gas": Fluid("Gas", K=0.02e9, rho=100.0),
    "gas_shallow": Fluid("Shallow Gas", K=0.01e9, rho=50.0),
    "gas_deep": Fluid("Deep Gas", K=0.10e9, rho=300.0),
    "co2": Fluid("CO2 (supercritical)", K=0.05e9, rho=500.0),
    "co2_liquid": Fluid("CO2 (liquid)", K=0.10e9, rho=700.0),
}

MINERALS: Mapping[str, Mineral] = MappingProxyType(_mineral_data.copy())
FLUIDS: Mapping[str, Fluid] = MappingProxyType(_fluid_data.copy())
del _mineral_data, _fluid_data


def _key(name: str, *, object_name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise RockContractError(
            "Rock catalog name must be a non-empty string",
            object_name=object_name,
            field="name",
            expected="non-empty string",
            actual=name,
        )
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def get_mineral(name: str) -> Mineral:
    """Look up an immutable mineral record by normalized name."""

    key = _key(name, object_name="get_mineral")
    try:
        return MINERALS[key]
    except KeyError as exc:
        raise RockContractError(
            "unknown Rock mineral",
            object_name="get_mineral",
            field="name",
            expected=tuple(sorted(MINERALS)),
            actual=name,
        ) from exc


def get_fluid(name: str) -> Fluid:
    """Look up an immutable fluid record by normalized name."""

    key = _key(name, object_name="get_fluid")
    try:
        return FLUIDS[key]
    except KeyError as exc:
        raise RockContractError(
            "unknown Rock fluid",
            object_name="get_fluid",
            field="name",
            expected=tuple(sorted(FLUIDS)),
            actual=name,
        ) from exc


def list_minerals() -> tuple[str, ...]:
    """Return the immutable sorted mineral-name inventory."""

    return tuple(sorted(MINERALS))


def list_fluids() -> tuple[str, ...]:
    """Return the immutable sorted fluid-name inventory."""

    return tuple(sorted(FLUIDS))


def mix_minerals_vrh(
    mineral_1: str,
    mineral_2: str,
    phase_1_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix two catalog minerals through the canonical VRH/Voigt functions."""

    (phase_1_fraction,) = require_compatible_tensors(
        "mix_minerals_vrh", ("phase_1_fraction", phase_1_fraction)
    )
    first = get_mineral(mineral_1)
    second = get_mineral(mineral_2)
    bulk_modulus = mixtures.vrh_average(
        phase_1_fraction.new_tensor(first.K),
        phase_1_fraction.new_tensor(second.K),
        phase_1_fraction,
    )
    shear_modulus = mixtures.vrh_average(
        phase_1_fraction.new_tensor(first.G),
        phase_1_fraction.new_tensor(second.G),
        phase_1_fraction,
    )
    density = mixtures.voigt_average(
        phase_1_fraction.new_tensor(first.rho),
        phase_1_fraction.new_tensor(second.rho),
        phase_1_fraction,
    )
    return bulk_modulus, shear_modulus, density


def mix_fluids_wood(
    fluid_1: str,
    fluid_2: str,
    fluid_1_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix two catalog fluids through the canonical Wood function."""

    (fluid_1_fraction,) = require_compatible_tensors(
        "mix_fluids_wood", ("fluid_1_fraction", fluid_1_fraction)
    )
    first = get_fluid(fluid_1)
    second = get_fluid(fluid_2)
    return mixtures.wood_fluid_mix(
        fluid_1_fraction.new_tensor(first.K),
        fluid_1_fraction.new_tensor(second.K),
        fluid_1_fraction.new_tensor(first.rho),
        fluid_1_fraction.new_tensor(second.rho),
        fluid_1_fraction,
    )


def mix_fluids_brie(
    fluid_1: str,
    fluid_2: str,
    fluid_1_fraction: torch.Tensor,
    *,
    exponent: torch.Tensor | int | float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix two catalog fluids through the canonical Brie function."""

    (fluid_1_fraction,) = require_compatible_tensors(
        "mix_fluids_brie", ("fluid_1_fraction", fluid_1_fraction)
    )
    first = get_fluid(fluid_1)
    second = get_fluid(fluid_2)
    return mixtures.brie_fluid_mix(
        fluid_1_fraction.new_tensor(first.K),
        fluid_1_fraction.new_tensor(second.K),
        fluid_1_fraction.new_tensor(first.rho),
        fluid_1_fraction.new_tensor(second.rho),
        fluid_1_fraction,
        exponent=exponent,
    )


__all__ = [
    "FLUIDS",
    "MINERALS",
    "Fluid",
    "Mineral",
    "get_fluid",
    "get_mineral",
    "list_fluids",
    "list_minerals",
    "mix_fluids_brie",
    "mix_fluids_wood",
    "mix_minerals_vrh",
]
