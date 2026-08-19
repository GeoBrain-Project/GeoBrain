"""
Fluid mixing models (Wood, Brie).

Private submodule of :mod:`geobrain.physics.rock.models.fluid`.
Public symbols are re-exported from ``fluid.py``.

Models:
    Wood: Reuss harmonic average for two-phase fluid mixing
    Brie: Empirical patchy-saturation mixing

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from torch import Tensor

from ....core import GeoBrainError
from ._categories import FluidModel
from ._registry import register


@register("Wood", aliases=["wood"])
class Wood(FluidModel):
    """Wood's law: Reuss harmonic average for two-phase fluid mixing.

    ``1 / K_fluid = Sw / K_water + (1 − Sw) / K_other``
    ``ρ_fluid    = Sw · ρ_water + (1 − Sw) · ρ_other``

    Inputs (forward): ``Sw, K_water, rho_water``; returns ``(K_fluid, ρ_fluid)``.
    """

    def __init__(self, *, K_other: float = 0.10e9, rho_other: float = 200.0) -> None:
        super().__init__()
        for name, value in (("K_other", K_other), ("rho_other", rho_other)):
            if value <= 0:
                raise GeoBrainError(
                    f"Wood {name} must be positive",
                    object_name="Wood",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        self.K_other = float(K_other)
        self.rho_other = float(rho_other)

    def forward(
        self, Sw: Tensor, K_water: Tensor | float, rho_water: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        Sg = 1.0 - Sw
        eps = 1e-20
        K_fluid = 1.0 / (Sw / K_water + Sg / self.K_other + eps)
        rho_fluid = Sw * rho_water + Sg * self.rho_other
        return K_fluid, rho_fluid

    def compute_fluid_mix(self, fl1_K, fl2_K, fl1_rho, fl2_rho, Sw, **params):
        """
        Workflow hook: phase 1 = water, phase 2 = ``K_other``/``rho_other``-equivalent.

        Note this ignores the model's stored ``K_other`` / ``rho_other`` and
        uses the values passed in, useful when the workflow centralises
        the hydrocarbon/gas endmember.
        """
        Sg = 1.0 - Sw
        K_eff = 1.0 / (Sw / fl1_K + Sg / fl2_K + 1e-20)
        rho_eff = Sw * fl1_rho + Sg * fl2_rho
        return K_eff, rho_eff


@register("Brie", aliases=["brie"])
class Brie(FluidModel):
    """
    Brie et al. (1995) empirical patchy-saturation mixing.

    ``K_fluid = (K_water − K_other) · Sw^e + K_other``
    ``ρ_fluid = Sw · ρ_water + (1 − Sw) · ρ_other``  (Wood-style mass mix)

    Exponent ``e`` controls patchiness: ``e=1`` ≈ patchy end, ``e→∞``
    approaches uniform saturation. ``e≈3`` is the common default.
    """

    def __init__(
        self,
        *,
        K_other: float = 0.10e9,
        rho_other: float = 200.0,
        exponent: float = 3.0,
    ) -> None:
        super().__init__()
        for name, value in (("K_other", K_other), ("rho_other", rho_other)):
            if value <= 0:
                raise GeoBrainError(
                    f"Brie {name} must be positive",
                    object_name="Brie",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        if exponent <= 0:
            raise GeoBrainError(
                "Brie exponent must be positive",
                object_name="Brie",
                field="exponent",
                expected="> 0",
                actual=exponent,
            )
        self.K_other = float(K_other)
        self.rho_other = float(rho_other)
        self.exponent = float(exponent)

    def forward(
        self, Sw: Tensor, K_water: Tensor | float, rho_water: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        Sg = 1.0 - Sw
        Sw_safe = Sw.clamp(min=0.0, max=1.0)
        K_fluid = (K_water - self.K_other) * Sw_safe.pow(self.exponent) + self.K_other
        rho_fluid = Sw * rho_water + Sg * self.rho_other
        return K_fluid, rho_fluid

    def compute_fluid_mix(self, fl1_K, fl2_K, fl1_rho, fl2_rho, Sw, **params):
        Sg = 1.0 - Sw
        Sw_safe = Sw.clamp(min=0.0, max=1.0)
        K_eff = (fl1_K - fl2_K) * Sw_safe.pow(self.exponent) + fl2_K
        rho_eff = Sw * fl1_rho + Sg * fl2_rho
        return K_eff, rho_eff


__all__ = ["Wood", "Brie"]
