"""Anisotropic-crack Rock operator facades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping

import torch

from geobrain.core import ModelState

from ..anisotropy import hudson_phase_properties, sayers_kachanov_phase_properties
from ._base import RockForwardOperator, RockOperatorDeclaration, field


_PHASE_OUTPUTS = (
    field("stiffness", "Pa", "Voigt 6x6 stiffness tensor."),
    field("vp0", "m/s", "Vertical compressional velocity."),
    field("vs0", "m/s", "Vertical shear velocity."),
    field("epsilon", "1", "Thomsen epsilon."),
    field("delta", "1", "Thomsen delta."),
    field("gamma", "1", "Thomsen gamma."),
)


def _phase_data(result: object) -> Mapping[str, torch.Tensor]:
    return {
        "stiffness": result.stiffness,
        "vp0": result.vp0,
        "vs0": result.vs0,
        "epsilon": result.epsilon,
        "delta": result.delta,
        "gamma": result.gamma,
    }


class Hudson(RockForwardOperator):
    """First-order aligned-crack Hudson VTI facade."""

    declaration = RockOperatorDeclaration(
        model="hudson",
        subfamily="anisotropy",
        inputs=(
            field("k_iso", "Pa", "Isotropic bulk modulus."),
            field("mu_iso", "Pa", "Isotropic shear modulus."),
            field("rho", "kg/m3", "Bulk density."),
            field("crack_density", "1", "Crack-density parameter."),
            field("aspect_ratio", "1", "Crack aspect ratio."),
            field("k_fluid", "Pa", "Crack-fluid bulk modulus."),
            field("mu_fluid", "Pa", "Crack-fluid shear modulus."),
        ),
        outputs=_PHASE_OUTPUTS,
        citation="Hudson (1981), Geophysical Journal 64.",
        calibrated_domain=(("aspect_ratio", "0 < value <= 1"),),
        workspace_tensors=24,
        saved_tensors=16,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = hudson_phase_properties(
            *state.fetch(
                "k_iso",
                "mu_iso",
                "rho",
                "crack_density",
                "aspect_ratio",
            ),
            k_fluid=state.tensors["k_fluid"],
            mu_fluid=state.tensors["mu_fluid"],
        )
        return _phase_data(result), {}


class SayersKachanov(RockForwardOperator):
    """Sayers-Kachanov crack-compliance VTI facade."""

    declaration = RockOperatorDeclaration(
        model="sayers_kachanov",
        subfamily="anisotropy",
        inputs=(
            field("k_iso", "Pa", "Isotropic bulk modulus."),
            field("mu_iso", "Pa", "Isotropic shear modulus."),
            field("rho", "kg/m3", "Bulk density."),
            field("normal_compliance", "Pa^-1", "Normal crack compliance."),
            field("tangential_compliance", "Pa^-1", "Tangential crack compliance."),
        ),
        outputs=_PHASE_OUTPUTS,
        citation="Sayers & Kachanov (1995), Geophysical Prospecting 43.",
        calibrated_domain=(("compliances", ">= 0 Pa^-1"),),
        workspace_tensors=18,
        saved_tensors=12,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = sayers_kachanov_phase_properties(
            *state.fetch(
                "k_iso",
                "mu_iso",
                "rho",
                "normal_compliance",
                "tangential_compliance",
            )
        )
        return _phase_data(result), {}


__all__ = ["Hudson", "SayersKachanov"]
