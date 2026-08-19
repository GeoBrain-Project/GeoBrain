"""
Linearized AVO approximations: Aki-Richards and Shuey.

Both approximations linearise the full Zoeppritz solution around small contrasts
and moderate incidence angles. Their valid range is implementation-defined; the
value reported in ``self._valid_angles`` is the conventional industry cutoff (40°
for Aki-Richards, 45° for Shuey).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from .base import AVOModel, normalize_reflectivity_inputs, vectorize_angles
from .registry import register_avo


@register_avo(
    "AkiRichardsReflectivity",
    aliases=["aki", "aki_richards", "ar", "AkiRichards"],
    description="Aki-Richards linearised 3-term approximation (valid 0-40°)",
)
class AkiRichardsReflectivity(AVOModel):
    """
    Aki-Richards three-term linearised P-P AVO reflectivity (math nn.Module).

    Distinct name from the user-facing ForwardOperator wrapper
    :class:`geobrain.physics.wave.reflectivity.operators.AkiRichards`: this is
    the pure math class returning ``R(θ)`` per interface, the operator
    wraps it for ``ModelState → ForwardOutput`` use.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "aki_richards"
        self._valid_angles = (0, 40)

    @vectorize_angles
    def forward(
        self,
        vp1: Tensor, vs1: Tensor, rho1: Tensor,
        vp2: Tensor, vs2: Tensor, rho2: Tensor,
        theta: Tensor,
    ) -> Tensor:
        vp = (vp1 + vp2) / 2
        vs = (vs1 + vs2) / 2
        rho = (rho1 + rho2) / 2

        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1

        sin2 = torch.sin(theta) ** 2
        tan2 = torch.tan(theta) ** 2
        k2 = (vs / vp) ** 2

        term1 = 0.5 * (1 + tan2) * (dvp / vp)
        term2 = -4 * k2 * sin2 * (dvs / vs)
        term3 = 0.5 * (1 - 4 * k2 * sin2) * (drho / rho)
        return term1 + term2 + term3


@register_avo(
    "ShueyReflectivity",
    aliases=["shuey", "Shuey"],
    description="Shuey 3-term AVO approximation (intercept R0, gradient G, curvature F)",
)
class ShueyReflectivity(AVOModel):
    """
    Shuey three-term P-P AVO reflectivity (math nn.Module).

    Distinct name from the user-facing ForwardOperator wrapper
    :class:`geobrain.physics.wave.reflectivity.operators.Shuey`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._name = "shuey"
        self._valid_angles = (0, 45)

    def _compute_attributes(
        self,
        vp1: Tensor, vs1: Tensor, rho1: Tensor,
        vp2: Tensor, vs2: Tensor, rho2: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        vp = (vp1 + vp2) / 2
        vs = (vs1 + vs2) / 2
        rho = (rho1 + rho2) / 2

        dvp = vp2 - vp1
        dvs = vs2 - vs1
        drho = rho2 - rho1
        k2 = (vs / vp) ** 2

        R0 = 0.5 * (dvp / vp + drho / rho)
        G = 0.5 * (dvp / vp) - 2 * k2 * (drho / rho + 2 * dvs / vs)
        F = 0.5 * (dvp / vp)
        return R0, G, F

    @vectorize_angles
    def forward(
        self,
        vp1: Tensor, vs1: Tensor, rho1: Tensor,
        vp2: Tensor, vs2: Tensor, rho2: Tensor,
        theta: Tensor,
    ) -> Tensor:
        R0, G, F = self._compute_attributes(vp1, vs1, rho1, vp2, vs2, rho2)
        sin2 = torch.sin(theta) ** 2
        tan2 = torch.tan(theta) ** 2
        return R0 + G * sin2 + F * (tan2 - sin2)

    def avo_attributes(
        self,
        vp1: object,
        vs1: object,
        rho1: object,
        vp2: object,
        vs2: object,
        rho2: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(R0, G, F)`` intercept / gradient / curvature for the
        interface pair. Useful for AVO attribute extraction without
        evaluating reflectivity at specific angles."""
        values = normalize_reflectivity_inputs(
            (
                ("vp1", vp1),
                ("vs1", vs1),
                ("rho1", rho1),
                ("vp2", vp2),
                ("vs2", vs2),
                ("rho2", rho2),
            ),
            owner=type(self).__name__,
        )
        return self._compute_attributes(*values)


__all__ = ["AkiRichardsReflectivity", "ShueyReflectivity"]
