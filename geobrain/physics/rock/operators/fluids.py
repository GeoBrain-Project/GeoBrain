"""Fluid-mixture, EOS, substitution, and poroelastic Rock facades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping

import torch

from geobrain.core import ModelState

from ..fluid_eos import batzle_wang_brine
from ..fluid_substitution import gassmann_saturated_properties
from ..mixtures import wood_fluid_mix
from ..poroelastic import biot_high_frequency_limits
from ._base import RockForwardOperator, RockOperatorDeclaration, field


class WoodFluidMix(RockForwardOperator):
    """Two-fluid Wood mixture facade."""

    declaration = RockOperatorDeclaration(
        model="wood_fluid_mix",
        subfamily="mixtures",
        inputs=(
            field("k_water", "Pa", "Water/brine bulk modulus."),
            field("k_other", "Pa", "Second-fluid bulk modulus."),
            field("rho_water", "kg/m3", "Water/brine density."),
            field("rho_other", "kg/m3", "Second-fluid density."),
            field("water_saturation", "1", "Water saturation fraction."),
        ),
        outputs=(
            field("bulk_modulus", "Pa", "Mixture bulk modulus."),
            field("rho", "kg/m3", "Mixture density."),
        ),
        citation="Wood (1941) harmonic fluid bulk-modulus mixture.",
        calibrated_domain=(("water_saturation", "0 <= value <= 1"),),
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        bulk, rho = wood_fluid_mix(
            *state.fetch("k_water", "k_other", "rho_water", "rho_other", "water_saturation")
        )
        return {"bulk_modulus": bulk, "rho": rho}, {}


class BatzleWangBrine(RockForwardOperator):
    """Batzle-Wang brine properties with an SI boundary."""

    declaration = RockOperatorDeclaration(
        model="batzle_wang_brine",
        subfamily="fluid_eos",
        inputs=(
            field("temperature", "K", "Absolute temperature."),
            field("pressure", "Pa", "Absolute pressure."),
            field("salinity", "1", "NaCl mass fraction."),
        ),
        outputs=(
            field("bulk_modulus", "Pa", "Brine bulk modulus."),
            field("rho", "kg/m3", "Brine density."),
        ),
        citation="Batzle & Wang (1992), Geophysics 57, 1396-1408.",
        calibrated_domain=(
            ("temperature", "273.15 <= value <= 623.15 K"),
            ("pressure", "0.1e6 <= value <= 100e6 Pa"),
            ("salinity", "0 <= value <= 0.35"),
        ),
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = batzle_wang_brine(*state.fetch("temperature", "pressure", "salinity"))
        return {"bulk_modulus": result.bulk_modulus, "rho": result.rho}, {}


class Gassmann(RockForwardOperator):
    """Canonical Gassmann saturated-property facade."""

    declaration = RockOperatorDeclaration(
        model="gassmann",
        subfamily="fluid_substitution",
        inputs=(
            field("k_dry", "Pa", "Dry-frame bulk modulus."),
            field("mu_dry", "Pa", "Dry-frame shear modulus."),
            field("k_mineral", "Pa", "Mineral bulk modulus."),
            field("k_fluid", "Pa", "Pore-fluid bulk modulus."),
            field("porosity", "1", "Porosity fraction."),
            field("rho_mineral", "kg/m3", "Mineral density."),
            field("rho_fluid", "kg/m3", "Pore-fluid density."),
        ),
        outputs=(
            field("k_sat", "Pa", "Saturated bulk modulus."),
            field("mu_sat", "Pa", "Saturated shear modulus."),
            field("rho", "kg/m3", "Saturated density."),
            field("vp", "m/s", "Compressional velocity."),
            field("vs", "m/s", "Shear velocity."),
        ),
        citation="Gassmann (1951).",
        calibrated_domain=(("porosity", "0 <= value < 1"),),
        workspace_tensors=9,
        saved_tensors=9,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = gassmann_saturated_properties(
            *state.fetch(
                "k_dry",
                "mu_dry",
                "k_mineral",
                "k_fluid",
                "porosity",
                "rho_mineral",
                "rho_fluid",
            )
        )
        return {
            "k_sat": result.k_sat,
            "mu_sat": result.mu_sat,
            "rho": result.rho,
            "vp": result.vp,
            "vs": result.vs,
        }, {}


class BiotHighFrequency(RockForwardOperator):
    """Biot high-frequency fast/slow P and shear velocity facade."""

    declaration = RockOperatorDeclaration(
        model="biot_high_frequency",
        subfamily="poroelastic",
        inputs=(
            field("k_dry", "Pa", "Dry-frame bulk modulus."),
            field("shear_modulus", "Pa", "Dry-frame shear modulus."),
            field("k_mineral", "Pa", "Mineral bulk modulus."),
            field("k_fluid", "Pa", "Fluid bulk modulus."),
            field("rho_mineral", "kg/m3", "Mineral density."),
            field("rho_fluid", "kg/m3", "Fluid density."),
            field("porosity", "1", "Porosity fraction."),
            field("tortuosity", "1", "High-frequency tortuosity."),
        ),
        outputs=(
            field("fast_p_velocity", "m/s", "Fast compressional velocity."),
            field("slow_p_velocity", "m/s", "Slow compressional velocity."),
            field("shear_velocity", "m/s", "Shear velocity."),
        ),
        citation="Biot (1956), JASA 28, parts I-II.",
        calibrated_domain=(("porosity", "0 < value < 1"),),
        workspace_tensors=16,
        saved_tensors=12,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = biot_high_frequency_limits(
            *state.fetch(
                "k_dry",
                "shear_modulus",
                "k_mineral",
                "k_fluid",
                "rho_mineral",
                "rho_fluid",
                "porosity",
                "tortuosity",
            )
        )
        return {
            "fast_p_velocity": result.fast_p_velocity,
            "slow_p_velocity": result.slow_p_velocity,
            "shear_velocity": result.shear_velocity,
        }, {}


__all__ = ["BatzleWangBrine", "BiotHighFrequency", "Gassmann", "WoodFluidMix"]
