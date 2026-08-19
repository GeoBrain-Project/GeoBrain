"""Quantitative-interpretation Rock operator facades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping

import torch

from geobrain.core import ModelState

from ..qi import estimate_rock_physics_template_resources, rock_physics_template
from ..resources import RockResourceEstimate
from ._base import RockForwardOperator, RockOperatorDeclaration, field


class RockPhysicsTemplate(RockForwardOperator):
    """Cartesian porosity/saturation rock-physics-template facade."""

    declaration = RockOperatorDeclaration(
        model="rock_physics_template",
        subfamily="qi",
        inputs=(
            field("porosity", "1", "Rank-one porosity axis."),
            field("water_saturation", "1", "Rank-one water-saturation axis."),
            field("k_dry", "Pa", "Dry-frame bulk modulus."),
            field("mu_dry", "Pa", "Dry-frame shear modulus."),
            field("k_mineral", "Pa", "Mineral bulk modulus."),
            field("rho_mineral", "kg/m3", "Mineral density."),
            field("k_brine", "Pa", "Brine bulk modulus."),
            field("rho_brine", "kg/m3", "Brine density."),
            field("k_hydrocarbon", "Pa", "Hydrocarbon bulk modulus."),
            field("rho_hydrocarbon", "kg/m3", "Hydrocarbon density."),
        ),
        outputs=(
            field("impedance", "kg/(m2*s)", "Acoustic impedance."),
            field("vp_vs_ratio", "1", "Compressional-to-shear velocity ratio."),
            field("vp", "m/s", "Compressional velocity."),
            field("vs", "m/s", "Shear velocity."),
            field("density", "kg/m3", "Bulk density."),
            field("fluid_bulk_modulus", "Pa", "Mixture fluid bulk modulus."),
        ),
        citation="Avseth, Mukerji & Mavko (2005), Quantitative Seismic Interpretation.",
        calibrated_domain=(
            ("porosity", "rank-1, 0 <= value < 1"),
            ("water_saturation", "rank-1, 0 <= value <= 1"),
        ),
        workspace_tensors=8,
        saved_tensors=8,
    )

    def estimate_resources(
        self,
        state: ModelState,
        *,
        autograd_enabled: bool,
        budget_bytes: int | None = None,
    ) -> RockResourceEstimate:
        estimate = estimate_rock_physics_template_resources(
            state.tensors["porosity"], state.tensors["water_saturation"]
        )
        shape = (
            state.tensors["porosity"].numel(),
            state.tensors["water_saturation"].numel(),
        )
        elements = shape[0] * shape[1]
        itemsize = estimate.bytes_per_element
        input_bytes = 10 * elements * itemsize
        output_bytes = estimate.output_elements * itemsize
        workspace_bytes = estimate.workspace_elements * itemsize
        saved_bytes = workspace_bytes if autograd_enabled else 0
        result = RockResourceEstimate(
            broadcast_shape=shape,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            workspace_bytes=workspace_bytes,
            saved_for_backward_bytes=saved_bytes,
            total_bytes=input_bytes + output_bytes + workspace_bytes + saved_bytes,
        )
        selected_budget = self._budget_bytes if budget_bytes is None else budget_bytes
        if selected_budget is not None:
            result.require_budget(selected_budget, object_name=type(self).__name__)
        return result

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = rock_physics_template(
            *state.fetch(
                "porosity",
                "water_saturation",
                "k_dry",
                "mu_dry",
                "k_mineral",
                "rho_mineral",
                "k_brine",
                "rho_brine",
                "k_hydrocarbon",
                "rho_hydrocarbon",
            )
        )
        return {
            "impedance": result.impedance,
            "vp_vs_ratio": result.vp_vs_ratio,
            "vp": result.vp,
            "vs": result.vs,
            "density": result.density,
            "fluid_bulk_modulus": result.fluid_bulk_modulus,
        }, {}


__all__ = ["RockPhysicsTemplate"]
