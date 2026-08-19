"""Empirical and petrophysical Rock operator facades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping

import torch

from geobrain.core import ModelState

from ..empirical import gardner_density
from ..petrophysics import archie_resistivity, kozeny_carman_permeability
from ._base import RockForwardOperator, RockOperatorDeclaration, field


class Gardner(RockForwardOperator):
    """Gardner compressional-velocity to density facade in SI units."""

    declaration = RockOperatorDeclaration(
        model="gardner",
        subfamily="empirical",
        inputs=(field("vp", "m/s", "Compressional velocity."),),
        outputs=(field("rho", "kg/m3", "Bulk density."),),
        citation="Gardner, Gardner & Gregory (1974), Geophysics 39.",
        calibrated_domain=(("vp", "> 0 m/s"),),
        workspace_tensors=2,
        saved_tensors=2,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        return {"rho": gardner_density(state.tensors["vp"])}, {}


class ArchieResistivity(RockForwardOperator):
    """Archie clean-formation resistivity facade."""

    declaration = RockOperatorDeclaration(
        model="archie_resistivity",
        subfamily="petrophysics",
        inputs=(
            field("porosity", "1", "Connected porosity fraction."),
            field("water_saturation", "1", "Water saturation fraction."),
            field("water_resistivity", "Ohm·m", "Formation-water resistivity."),
        ),
        outputs=(field("resistivity", "Ohm·m", "Formation resistivity."),),
        citation="Archie (1942), Transactions of the AIME 146.",
        calibrated_domain=(
            ("porosity", "0 < value < 1"),
            ("water_saturation", "0 < value <= 1"),
        ),
        workspace_tensors=5,
        saved_tensors=5,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        value = archie_resistivity(
            *state.fetch("porosity", "water_saturation", "water_resistivity")
        )
        return {"resistivity": value}, {}


class KozenyCarman(RockForwardOperator):
    """Kozeny-Carman permeability facade."""

    declaration = RockOperatorDeclaration(
        model="kozeny_carman",
        subfamily="petrophysics",
        inputs=(
            field("porosity", "1", "Connected porosity fraction."),
            field("grain_diameter", "m", "Representative grain diameter."),
        ),
        outputs=(field("permeability", "m2", "Intrinsic permeability."),),
        citation="Kozeny (1927); Carman (1937).",
        calibrated_domain=(("porosity", "0 < value < 1"),),
        workspace_tensors=5,
        saved_tensors=4,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        value = kozeny_carman_permeability(
            *state.fetch("porosity", "grain_diameter")
        )
        return {"permeability": value}, {}


__all__ = ["ArchieResistivity", "Gardner", "KozenyCarman"]
