"""
Bounds + critical-porosity effective-medium models.

Private submodule of :mod:`geobrain.physics.rock.models.effective`.
Public symbols are re-exported from ``effective.py``.

Models:
    VRH:               Voigt-Reuss-Hill 2-component average (legacy signature)
    HashinShtrikman:   Hashin-Shtrikman bounds, Hill-averaged
    Voigt:             Voigt upper bound (iso-strain)
    Reuss:             Reuss lower bound (iso-stress)
    CriticalPorosity: Nur et al. (1998) critical-porosity model

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..moduli import vrh_average
from ._categories import EffectiveModel
from ._registry import register
from ._types import EPS


@register("VRH", aliases=["vrh", "voigt_reuss_hill"])
class VRH(EffectiveModel):
    """
    Voigt-Reuss-Hill average for a 2-component isotropic mixture.

    Inputs: ``K1, K2, f1`` (volume fraction of phase 1); ``f2 = 1 − f1``
    inferred. Returns the Hill-averaged effective modulus.
    """

    def forward(self, K1: Tensor, K2: Tensor, f1: Tensor) -> Tensor:
        f2 = 1.0 - f1
        return vrh_average([K1, K2], [f1, f2], eps=1e-20)


@register("HashinShtrikman", aliases=["hs", "hashin_shtrikman"])
class HashinShtrikman(EffectiveModel):
    """
    Hashin-Shtrikman bounds for a 2-phase isotropic mixture, Hill-averaged.

    Returns ``(K_HS, mu_HS)``; the arithmetic mean of HS⁺ and HS⁻.
    """

    def forward(
        self,
        K1: Tensor, mu1: Tensor, K2: Tensor, mu2: Tensor, f1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        f2 = 1.0 - f1
        eps = 1e-20
        K_plus = K1 + f2 / (1.0 / (K2 - K1 + eps) + 3.0 * f1 / (3.0 * K1 + 4.0 * mu1))
        K_minus = K2 + f1 / (1.0 / (K1 - K2 + eps) + 3.0 * f2 / (3.0 * K2 + 4.0 * mu2))
        K_HS = 0.5 * (K_plus + K_minus)

        zeta1 = 2.0 * f1 * (K1 + 2.0 * mu1) / (5.0 * mu1 * (K1 + 4.0 / 3.0 * mu1))
        zeta2 = 2.0 * f2 * (K2 + 2.0 * mu2) / (5.0 * mu2 * (K2 + 4.0 / 3.0 * mu2))
        mu_plus = mu1 + f2 / (1.0 / (mu2 - mu1 + eps) + zeta1)
        mu_minus = mu2 + f1 / (1.0 / (mu1 - mu2 + eps) + zeta2)
        mu_HS = 0.5 * (mu_plus + mu_minus)
        return K_HS, mu_HS


@register("Voigt", aliases=["voigt"])
class Voigt(EffectiveModel):
    """
    Voigt upper bound (iso-strain): ``M_V = Σ f_i · M_i``.

    Inputs: ``M`` (per-phase moduli) and ``f`` (per-phase volume fractions),
    same length. Both tensors of any 1-D length.
    """

    def forward(self, M: Tensor, f: Tensor) -> Tensor:
        return torch.sum(f * M)


@register("Reuss", aliases=["reuss"])
class Reuss(EffectiveModel):
    """Reuss lower bound (iso-stress): ``1 / M_R = Σ f_i / M_i``."""

    def forward(self, M: Tensor, f: Tensor) -> Tensor:
        return 1.0 / torch.sum(f / (M + EPS))


@register("CriticalPorosity", aliases=["critical_porosity", "cp"])
class CriticalPorosity(EffectiveModel):
    """
    Nur et al. (1998) critical-porosity model: ``M_dry = M_0 · (1 − φ/φ_c)``.

    Returns ``(K_dry, μ_dry)`` clamped to ≥ 0 (φ > φ_c is non-physical).
    """

    def __init__(self, phi_c: float = 0.40) -> None:
        super().__init__()
        self.phi_c = float(phi_c)

    def forward(
        self, K0: Tensor, G0: Tensor, phi: Tensor,
        phi_c: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        phi_c_val = phi_c if phi_c is not None else self.phi_c
        factor = torch.clamp(1.0 - phi / phi_c_val, min=0.0)
        return K0 * factor, G0 * factor

    def compute_dry_rock(self, K0, G0, phi, **params):
        """Workflow hook: pulls phi_c from params or uses constructor default."""
        return self.forward(K0, G0, phi, phi_c=params.get("phi_c"))


__all__ = [
    "VRH",
    "HashinShtrikman",
    "Voigt",
    "Reuss",
    "CriticalPorosity",
]
