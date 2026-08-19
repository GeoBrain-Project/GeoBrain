"""Elastic Rock operator facades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping

import torch

from geobrain.core import ModelState

from ..elastic import bulk_modulus_from_velocities, velocities_from_moduli
from ._base import RockForwardOperator, RockOperatorDeclaration, field


class VelocitiesFromModuli(RockForwardOperator):
    """Compute isotropic P/S velocities from bulk/shear moduli and density."""

    declaration = RockOperatorDeclaration(
        model="velocities_from_moduli",
        subfamily="elastic",
        inputs=(
            field("bulk_modulus", "Pa", "Bulk modulus."),
            field("shear_modulus", "Pa", "Shear modulus."),
            field("rho", "kg/m3", "Bulk density."),
        ),
        outputs=(
            field("vp", "m/s", "Compressional velocity."),
            field("vs", "m/s", "Shear velocity."),
        ),
        citation="Isotropic linear elasticity.",
        calibrated_domain=(("moduli", "> 0 Pa"), ("rho", "> 0 kg/m3")),
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        vp, vs = velocities_from_moduli(*state.fetch("bulk_modulus", "shear_modulus", "rho"))
        return {"vp": vp, "vs": vs}, {}


class ModuliFromVelocities(RockForwardOperator):
    """Compute isotropic bulk/shear moduli from P/S velocities and density."""

    declaration = RockOperatorDeclaration(
        model="moduli_from_velocities",
        subfamily="elastic",
        inputs=(
            field("vp", "m/s", "Compressional velocity."),
            field("vs", "m/s", "Shear velocity."),
            field("rho", "kg/m3", "Bulk density."),
        ),
        outputs=(
            field("bulk_modulus", "Pa", "Bulk modulus."),
            field("shear_modulus", "Pa", "Shear modulus."),
        ),
        citation="Isotropic linear elasticity.",
        calibrated_domain=(("vp,vs", "> 0 m/s"), ("rho", "> 0 kg/m3")),
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        bulk, shear = bulk_modulus_from_velocities(*state.fetch("vp", "vs", "rho"))
        return {"bulk_modulus": bulk, "shear_modulus": shear}, {}


__all__ = ["ModuliFromVelocities", "VelocitiesFromModuli"]
