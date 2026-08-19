"""
Materials database (minerals and fluids) plus mixing helpers.

Two frozen dataclasses
(:class:`Mineral`, :class:`Fluid`) plus two dicts indexed by lowercase
name (:data:`MINERALS` (23 entries; :data:`FLUIDS`) 13 entries), and
three pure-tensor mixing helpers that preserve autograd through the
``composition`` fractions.

Unit conventions:
    K, G, bulk / shear modulus in Pa
    rho:   density in kg/m³

Fluids are assumed inviscid (G = 0) at the seismic frequencies these
relations target.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ....core import RegistryError


@dataclass(frozen=True)
class Mineral:
    """Mineral elastic properties (K, G in Pa, ρ in kg/m³)."""

    name: str
    K: float
    G: float
    rho: float


@dataclass(frozen=True)
class Fluid:
    """Fluid properties (G assumed zero)."""

    name: str
    K: float
    rho: float


# --- Mineral database ------------------------------------------------------


MINERALS: dict[str, Mineral] = {
    # Framework silicates
    "quartz": Mineral("Quartz", K=36.6e9, G=45.0e9, rho=2650.0),
    "feldspar": Mineral("Feldspar", K=37.5e9, G=15.0e9, rho=2620.0),
    "plagioclase": Mineral("Plagioclase", K=75.6e9, G=25.6e9, rho=2630.0),
    # Micas
    "mica": Mineral("Mica", K=42.0e9, G=22.0e9, rho=2790.0),
    "muscovite": Mineral("Muscovite", K=52.0e9, G=31.0e9, rho=2820.0),
    "biotite": Mineral("Biotite", K=50.0e9, G=28.0e9, rho=3050.0),
    # Clay minerals
    "clay": Mineral("Clay (average)", K=21.0e9, G=7.0e9, rho=2580.0),
    "illite": Mineral("Illite", K=21.0e9, G=7.0e9, rho=2660.0),
    "kaolinite": Mineral("Kaolinite", K=1.5e9, G=1.4e9, rho=2600.0),
    "smectite": Mineral("Smectite", K=9.3e9, G=6.9e9, rho=2700.0),
    "montmorillonite": Mineral("Montmorillonite", K=9.3e9, G=6.9e9, rho=2500.0),
    "chlorite": Mineral("Chlorite", K=82.0e9, G=28.0e9, rho=2800.0),
    # Carbonates
    "calcite": Mineral("Calcite", K=76.8e9, G=32.0e9, rho=2710.0),
    "dolomite": Mineral("Dolomite", K=94.9e9, G=45.0e9, rho=2870.0),
    "aragonite": Mineral("Aragonite", K=47.0e9, G=39.0e9, rho=2930.0),
    # Evaporites
    "halite": Mineral("Halite", K=24.8e9, G=14.9e9, rho=2160.0),
    "anhydrite": Mineral("Anhydrite", K=56.1e9, G=29.1e9, rho=2980.0),
    "gypsum": Mineral("Gypsum", K=36.0e9, G=15.0e9, rho=2320.0),
    # Other
    "kerogen": Mineral("Kerogen", K=2.9e9, G=2.7e9, rho=1300.0),
    "pyrite": Mineral("Pyrite", K=147.4e9, G=132.5e9, rho=4930.0),
    "siderite": Mineral("Siderite", K=124.0e9, G=51.0e9, rho=3960.0),
    "magnetite": Mineral("Magnetite", K=161.0e9, G=91.0e9, rho=5180.0),
    "hematite": Mineral("Hematite", K=100.0e9, G=95.0e9, rho=5240.0),
}


# --- Fluid database --------------------------------------------------------


FLUIDS: dict[str, Fluid] = {
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


# --- Lookup ----------------------------------------------------------------


def _normalize(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def get_mineral(name: str) -> Mineral:
    """Look up a mineral by name (case-insensitive)."""
    key = _normalize(name)
    if key not in MINERALS:
        available = ", ".join(sorted(MINERALS.keys()))
        raise RegistryError(
            f"Unknown mineral: '{name}'. Available minerals: {available}",
            object_name="MINERALS", field="name",
            expected=f"one of {available}", actual=name,
        )
    return MINERALS[key]


def get_fluid(name: str) -> Fluid:
    """Look up a fluid by name (case-insensitive)."""
    key = _normalize(name)
    if key not in FLUIDS:
        available = ", ".join(sorted(FLUIDS.keys()))
        raise RegistryError(
            f"Unknown fluid: '{name}'. Available fluids: {available}",
            object_name="FLUIDS", field="name",
            expected=f"one of {available}", actual=name,
        )
    return FLUIDS[key]


def list_minerals() -> list[str]:
    return sorted(MINERALS.keys())


def list_fluids() -> list[str]:
    return sorted(FLUIDS.keys())


# --- Mixing helpers --------------------------------------------------------


def _t(value):
    """Tensor coercion that preserves any caller-supplied tensor (for autograd)."""
    return value if torch.is_tensor(value) else torch.as_tensor(value)


def mix_minerals_vrh(
    composition: dict[str, float], normalize: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Voigt-Reuss-Hill mix for a mineral composition.

    Returns ``(K, G, rho)`` torch tensors. The 0-d tensors auto-broadcast
    against per-cell ``phi``/``Sw`` downstream so the workflow stays
    differentiable end-to-end.
    """
    if not composition:
        raise ValueError("Composition dictionary cannot be empty")
    total = sum(composition.values())
    if total <= 0:
        raise ValueError("Total fraction must be positive")

    K_voigt = _t(0.0)
    K_reuss_inv = _t(0.0)
    G_voigt = _t(0.0)
    G_reuss_inv = _t(0.0)
    rho_mix = _t(0.0)
    for name, frac in composition.items():
        f = _t(frac) / _t(total) if normalize else _t(frac)
        m = get_mineral(name)
        K_voigt = K_voigt + f * m.K
        G_voigt = G_voigt + f * m.G
        K_reuss_inv = K_reuss_inv + f / m.K
        G_reuss_inv = G_reuss_inv + f / m.G
        rho_mix = rho_mix + f * m.rho
    K = 0.5 * (K_voigt + 1.0 / K_reuss_inv)
    G = 0.5 * (G_voigt + 1.0 / G_reuss_inv)
    return K, G, rho_mix


def mix_fluids_wood(
    composition: dict[str, float], normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Wood (Reuss) mix: ``1 / K_eff = Σ f_i / K_i``, ``ρ_eff = Σ f_i ρ_i``."""
    if not composition:
        raise ValueError("Composition dictionary cannot be empty")
    total = sum(composition.values())
    if total <= 0:
        raise ValueError("Total fraction must be positive")

    K_inv = _t(0.0)
    rho_mix = _t(0.0)
    for name, frac in composition.items():
        f = _t(frac) / _t(total) if normalize else _t(frac)
        fl = get_fluid(name)
        K_inv = K_inv + f / fl.K
        rho_mix = rho_mix + f * fl.rho
    return 1.0 / K_inv, rho_mix


def mix_fluids_brie(
    K_water: float, K_gas: float,
    rho_water: float, rho_gas: float,
    Sw: float, e: float = 3.0,
) -> tuple[Tensor, Tensor]:
    """Brie patchy-saturation mix between water and gas."""
    K_w = _t(K_water)
    K_g = _t(K_gas)
    rho_w = _t(rho_water)
    rho_g = _t(rho_gas)
    Sw_t = _t(Sw)
    K = K_g + (K_w - K_g) * (Sw_t ** e)
    rho = Sw_t * rho_w + (1.0 - Sw_t) * rho_g
    return K, rho


__all__ = [
    "FLUIDS",
    "Fluid",
    "MINERALS",
    "Mineral",
    "get_fluid",
    "get_mineral",
    "list_fluids",
    "list_minerals",
    "mix_fluids_brie",
    "mix_fluids_wood",
    "mix_minerals_vrh",
]
