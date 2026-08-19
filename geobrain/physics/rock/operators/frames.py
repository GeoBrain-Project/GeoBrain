"""Granular-contact and effective-medium Rock facades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Mapping

import torch

from geobrain.core import ModelState

from ..granular import hertz_mindlin_moduli
from ..inclusions import require_converged, self_consistent_moduli
from ._base import RockForwardOperator, RockOperatorDeclaration, field


class HertzMindlin(RockForwardOperator):
    """Hertz-Mindlin dry-frame endpoint facade."""

    declaration = RockOperatorDeclaration(
        model="hertz_mindlin",
        subfamily="granular",
        inputs=(
            field("effective_pressure", "Pa", "Effective pressure."),
            field("k_mineral", "Pa", "Mineral bulk modulus."),
            field("mu_mineral", "Pa", "Mineral shear modulus."),
            field("critical_porosity", "1", "Critical porosity."),
            field("coordination_number", "1", "Grain coordination number."),
            field("friction_factor", "1", "Tangential contact factor."),
        ),
        outputs=(
            field("k_dry", "Pa", "Dry-frame bulk modulus."),
            field("mu_dry", "Pa", "Dry-frame shear modulus."),
        ),
        citation="Mindlin (1949); Dvorkin & Nur (1996).",
        calibrated_domain=(("effective_pressure", "> 0 Pa"),),
        workspace_tensors=8,
    )

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = hertz_mindlin_moduli(
            *state.fetch(
                "effective_pressure",
                "k_mineral",
                "mu_mineral",
                "critical_porosity",
                "coordination_number",
                "friction_factor",
            )
        )
        return {"k_dry": result.k_dry, "mu_dry": result.mu_dry}, {}


class SelfConsistent(RockForwardOperator):
    """Two-phase spherical self-consistent effective-medium facade.

    Args:
        tolerance: fixed-point convergence tolerance on the moduli.
        max_iterations: iteration cap for the self-consistent loop.
        budget_bytes: optional memory-preflight budget.
    """

    declaration = RockOperatorDeclaration(
        model="self_consistent",
        subfamily="effective_medium",
        inputs=(
            field("k_phase_1", "Pa", "Phase-one bulk modulus."),
            field("mu_phase_1", "Pa", "Phase-one shear modulus."),
            field("k_phase_2", "Pa", "Phase-two bulk modulus."),
            field("mu_phase_2", "Pa", "Phase-two shear modulus."),
            field("phase_1_fraction", "1", "Phase-one volume fraction."),
        ),
        outputs=(
            field("k_eff", "Pa", "Effective bulk modulus."),
            field("mu_eff", "Pa", "Effective shear modulus."),
        ),
        citation="Budiansky (1965) self-consistent spherical inclusions.",
        calibrated_domain=(("phase_1_fraction", "0 <= value <= 1"),),
        workspace_tensors=12,
        saved_tensors=10,
        iterative=True,
    )

    def __init__(
        self,
        *,
        tolerance: float = 1.0e-8,
        max_iterations: int = 200,
        budget_bytes: int | None = None,
    ) -> None:
        super().__init__(budget_bytes=budget_bytes)
        self._tolerance = tolerance
        self._max_iterations = max_iterations

    def _evaluate(
        self, state: ModelState
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
        result = self_consistent_moduli(
            *state.fetch(
                "k_phase_1",
                "mu_phase_1",
                "k_phase_2",
                "mu_phase_2",
                "phase_1_fraction",
            ),
            tolerance=self._tolerance,
            max_iterations=self._max_iterations,
        )
        require_converged(result, object_name="SelfConsistent")
        return {
            "k_eff": result.k_eff,
            "mu_eff": result.mu_eff,
        }, {"iteration": result.iteration.to_dict()}


__all__ = ["HertzMindlin", "SelfConsistent"]
