"""
Hudson crack-stiffness effective-medium family.

Private submodule of :mod:`geobrain.physics.rock.models.effective`.
Public symbols are re-exported from ``effective.py``.

Models:
    HudsonStiffness: Aligned penny-shaped cracks (VTI/HTI 6×6)
    HudsonRandom:     Randomly oriented cracks (isotropic)
    HudsonOrtho:      Three orthogonal aligned-crack sets (orthorhombic 6×6)
    HudsonCone:       Cone-distributed crack normals (VTI 6×6)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from numbers import Integral

from torch import Tensor

from ....core import GeoBrainError
from ..anisotropy import (
    _legacy_degrees_to_radians,
    hudson_cone_stiffness,
    hudson_orthogonal_stiffness,
    hudson_random_moduli,
    hudson_stiffness,
)
from ._categories import EffectiveModel
from ._registry import register


@register("HudsonStiffness", aliases=["hudson_stiffness"])
class HudsonStiffness(EffectiveModel):
    """
    Hudson aligned penny-shaped cracks → 6×6 VTI (axis=3) or HTI (axis=1) stiffness.

    Different from the Thomsen-output Hudson in the anisotropy category;
    this one returns the full Voigt-notation stiffness for downstream
    anisotropic-Gassmann or velocity computations.

    Args:
        order: 1 (first-order) or 2 (Cheng's 2nd-order correction).
        axis: 3 → VTI, 1 → HTI (symmetry axis x₁).
    """

    def __init__(self, order: int = 1, axis: int = 3) -> None:
        super().__init__()
        if not isinstance(order, Integral) or isinstance(order, bool) or int(order) not in (1, 2):
            raise GeoBrainError(
                "HudsonStiffness order must be 1 or 2",
                object_name="HudsonStiffness",
                field="order",
                expected="1 or 2",
                actual=order,
            )
        if not isinstance(axis, Integral) or isinstance(axis, bool) or int(axis) not in (1, 3):
            raise GeoBrainError(
                "HudsonStiffness axis must be 1 (HTI) or 3 (VTI)",
                object_name="HudsonStiffness",
                field="axis",
                expected="1 or 3",
                actual=axis,
            )
        self.order = int(order)
        self.axis = int(axis)

    def forward(
        self,
        K: Tensor,
        G: Tensor,
        Ki: Tensor,
        Gi: Tensor,
        alpha: Tensor,
        crd: Tensor,
    ) -> Tensor:
        return hudson_stiffness(
            K,
            G,
            crd,
            alpha,
            k_fluid=Ki,
            mu_fluid=Gi,
            axis=self.axis,
            order=self.order,
        )


@register("HudsonRandom", aliases=["hudson_random"])
class HudsonRandom(EffectiveModel):
    """Hudson randomly oriented cracks → isotropic ``(K_eff, μ_eff)``."""

    def forward(
        self,
        K: Tensor,
        G: Tensor,
        Ki: Tensor,
        Gi: Tensor,
        alpha: Tensor,
        crd: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return hudson_random_moduli(
            K,
            G,
            crd,
            alpha,
            k_fluid=Ki,
            mu_fluid=Gi,
        )


@register("HudsonOrtho", aliases=["hudson_ortho"])
class HudsonOrtho(EffectiveModel):
    """
    Hudson three orthogonal aligned-crack sets → orthorhombic stiffness.

    Inputs ``alpha`` and ``crd`` are length-3 vectors (one per crack set).
    Returns a 6×6 orthorhombic stiffness matrix.
    """

    def forward(
        self,
        K: Tensor,
        G: Tensor,
        Ki: Tensor,
        Gi: Tensor,
        alpha: Tensor,
        crd: Tensor,
    ) -> Tensor:
        return hudson_orthogonal_stiffness(
            K,
            G,
            crd,
            alpha,
            k_fluid=Ki,
            mu_fluid=Gi,
        )


@register("HudsonCone", aliases=["hudson_cone"])
class HudsonCone(EffectiveModel):
    """
    Hudson cone-distributed crack normals → VTI stiffness.

    Cone half-angle ``theta`` (degrees) measures the spread of crack
    normals around the vertical axis.
    """

    def forward(
        self,
        K: Tensor,
        G: Tensor,
        Ki: Tensor,
        Gi: Tensor,
        alpha: Tensor,
        crd: Tensor,
        theta_deg: Tensor | float,
    ) -> Tensor:
        return hudson_cone_stiffness(
            K,
            G,
            crd,
            alpha,
            cone_angle_radians=_legacy_degrees_to_radians(
                K,
                theta_deg,
                object_name="HudsonCone",
                field="theta_deg",
            ),
            k_fluid=Ki,
            mu_fluid=Gi,
        )


__all__ = [
    "HudsonStiffness",
    "HudsonRandom",
    "HudsonOrtho",
    "HudsonCone",
]
