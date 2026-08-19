"""
Miscellaneous effective-medium models (Mori-Tanaka + O'Connell-Budiansky).

Private submodule of :mod:`geobrain.physics.rock.models.effective`.
Public symbols are re-exported from ``effective.py``.

Models:
    MTAverage:            Mori-Tanaka three-phase average
    OConnellBudiansky:    Dry penny-shaped cracks (closed form)
    OConnellBudianskyFl: Fluid-saturated penny-shaped cracks (iterative SC)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from torch import Tensor

from ._categories import EffectiveModel
from ._registry import register
from ._types import EPS
from ..inclusions import oconnell_budiansky_fluid_moduli, require_converged


# --- Mori-Tanaka --------------------------------------------------------


@register("MTAverage", aliases=["MoriTanaka", "mori_tanaka"])
class MTAverage(EffectiveModel):
    """Modified Mori-Tanaka three-phase average (matrix + two inclusions)."""

    def forward(
        self,
        f: Tensor,
        Kmat: Tensor,
        Gmat: Tensor,
        K1: Tensor,
        G1: Tensor,
        K2: Tensor,
        G2: Tensor,
    ) -> tuple[Tensor, Tensor]:
        nu = (3.0 * Kmat - 2.0 * Gmat) / (2.0 * (3.0 * Kmat + Gmat) + EPS)
        alpha_mt = (1.0 + nu) / (3.0 * (1.0 - nu) + EPS)
        beta = 2.0 * (4.0 - 5.0 * nu) / (15.0 * (1.0 - nu) + EPS)

        dK1 = Kmat - (Kmat - K1) * alpha_mt
        dK2 = Kmat - (Kmat - K2) * alpha_mt
        dG1 = Gmat - (Gmat - G1) * beta
        dG2 = Gmat - (Gmat - G2) * beta

        K_ave = (f * Kmat * K1 / (dK1 + EPS) + (1.0 - f) * Kmat * K2 / (dK2 + EPS)) / (
            f * Kmat / (dK1 + EPS) + (1.0 - f) * Kmat / (dK2 + EPS) + EPS
        )
        G_ave = (f * Gmat * G1 / (dG1 + EPS) + (1.0 - f) * Gmat * G2 / (dG2 + EPS)) / (
            f * Gmat / (dG1 + EPS) + (1.0 - f) * Gmat / (dG2 + EPS) + EPS
        )
        return K_ave, G_ave


# --- O'Connell-Budiansky penny-shaped cracks --------------------------------


@register("OConnellBudiansky", aliases=["ob", "oconnell_budiansky"])
class OConnellBudiansky(EffectiveModel):
    """O'Connell & Budiansky (1974) dry penny-shaped cracks (closed form)."""

    def forward(
        self,
        K0: Tensor,
        G0: Tensor,
        crd: Tensor,
    ) -> tuple[Tensor, Tensor]:
        nu0 = (3.0 * K0 - 2.0 * G0) / (6.0 * K0 + 2.0 * G0 + EPS)
        nu_eff = nu0 * (1.0 - 16.0 * crd / 9.0)
        K_dry = K0 * (1.0 - 16.0 * (1.0 - nu_eff**2) * crd / (9.0 * (1.0 - 2.0 * nu_eff) + EPS))
        G_dry = G0 * (
            1.0 - 32.0 * (1.0 - nu_eff) * (5.0 - nu_eff) * crd / (45.0 * (2.0 - nu_eff) + EPS)
        )
        return K_dry, G_dry


@register("OConnellBudianskyFl", aliases=["ob_fluid"])
class OConnellBudianskyFl(EffectiveModel):
    """O'Connell & Budiansky fluid-saturated penny-shaped cracks (iterative SC)."""

    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        super().__init__()
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def forward(
        self,
        K0: Tensor,
        G0: Tensor,
        Kfl: Tensor,
        crd: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = oconnell_budiansky_fluid_moduli(
            K0,
            G0,
            Kfl,
            crd,
            alpha,
            tolerance=self.tol,
            max_iterations=self.max_iter,
        )
        require_converged(result, object_name="OConnellBudianskyFl")
        return result.k_eff, result.mu_eff


__all__ = [
    "MTAverage",
    "OConnellBudiansky",
    "OConnellBudianskyFl",
]
