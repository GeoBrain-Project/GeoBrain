"""
CO₂ + live oil EOS classes.

Private submodule of :mod:`geobrain.physics.rock.models.fluid`.
Public symbols are re-exported from ``fluid.py``.

Models:
    CO2Properties: Supercritical CO₂ ``(ρ, K)`` via modified Batzle-Wang
    LiveOil:        Live (gas-saturated) oil ``(ρ, K)``
    CO2Brine:       CO₂-brine mixture (composes CO2Properties + BatzleWang)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._categories import FluidModel
from ._registry import register
from ._types import EPS
from ._fluid_batzlewang import BatzleWang


@register("CO2Properties", aliases=["co2_props"])
class CO2Properties(FluidModel):
    """Supercritical CO₂ ``(ρ, K)`` from modified Batzle-Wang. ρ in kg/m³, K in Pa (SI)."""

    def forward(
        self, P: Tensor, T: Tensor, G: Tensor | float = 1.5349,
    ) -> tuple[Tensor, Tensor]:
        P, T = torch.as_tensor(P), torch.as_tensor(T)
        G = torch.as_tensor(G, dtype=T.dtype, device=T.device)
        R = 8.3145
        Ta = T + 273.15
        P_pr = P / 7.4
        T_pr = Ta / (31.1 + 273.5)

        E = (
            0.109 * (3.85 - T_pr) ** 2
            * torch.exp(-(0.45 + 8.0 * (0.56 - 1.0 / T_pr) ** 2) * P_pr ** 1.2 / T_pr)
        )
        Z = (
            (0.03 + 0.00527 * (3.5 - T_pr) ** 3) * P_pr
            + (0.642 * T_pr - 0.007 * T_pr ** 4 - 0.52) + E
        )
        rho = 28.8 * G * P / (Z * R * Ta + EPS)
        r_0 = (
            0.85 + 5.6 / (P_pr + 2) + 27.1 / (P_pr + 3.5) ** 2
            - 8.7 * torch.exp(-0.65 * (P_pr + 1))
        )
        dzdp = (0.03 + 0.00527 * (3.5 - T_pr) ** 3) + 0.109 * (
            3.85 - T_pr
        ) ** 2 * 1.2 * P_pr ** 0.2 * (
            -(0.45 + 8.0 * (0.56 - 1.0 / T_pr) ** 2) / T_pr
        ) * torch.exp(-(0.45 + 8.0 * (0.56 - 1.0 / T_pr) ** 2) * P_pr ** 1.2 / T_pr)
        K = P / (1.0 - P_pr * dzdp / (Z + EPS) + EPS) * r_0 / 1000.0
        return rho * 1000.0, K * 1e9        # SI (kg/m³, Pa)


@register("LiveOil", aliases=["live_oil"])
class LiveOil(FluidModel):
    """
    Live (gas-saturated) oil ``(ρ, K)``. ρ in kg/m³, K in Pa (SI).

    Gas-oil ratio ``Rg`` auto-computed from (P, ρ₀, T) when ``None``. The
    ``den`` reference-density parameter stays in its historical native scale
    internally (same ``rho_0``-style caution as ``BatzleWangOilDead``, the
    correlation constants are calibrated for that scale); only the RETURN
    boundary changed to SI.
    """

    def forward(
        self,
        P: Tensor, T: Tensor, den: Tensor, G: Tensor,
        Rg: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        P, T = torch.as_tensor(P), torch.as_tensor(T)
        den = torch.as_tensor(den)
        G = torch.as_tensor(G, dtype=T.dtype, device=T.device)

        if Rg is None:
            Rg = 0.02123 * G * (P * torch.exp(4.072 / den - 0.00377 * T)) ** 1.205
        else:
            Rg = torch.as_tensor(Rg, dtype=T.dtype, device=T.device)

        B = 0.972 + 0.00038 * (2.4 * Rg * (G / den) ** 0.5 + T + 17.8) ** 1.175
        rho_p = den * (1.0 + 0.001 * Rg) ** (-1) * B ** (-1)
        v = (
            2096.0 * (rho_p / (2.6 - rho_p + EPS)) ** 0.5
            - 3.7 * T + 4.64 * P
            + 0.0115 * (4.12 * (1.08 / (rho_p + EPS) - 1.0) ** 0.5 - 1.0) * T * P
        )
        rho_g = (den + 0.0012 * G * Rg) / (B + EPS)
        K = rho_g * v ** 2 / 1e6
        return rho_g * 1000.0, K * 1e9        # SI (kg/m³, Pa)


@register("CO2Brine", aliases=["co2_brine"])
class CO2Brine(FluidModel):
    """CO₂-brine mixture using :class:`CO2Properties` + :class:`BatzleWang`."""

    def __init__(self) -> None:
        super().__init__()
        self._co2 = CO2Properties()
        self._bw = BatzleWang()

    def forward(
        self,
        T: Tensor, P: Tensor,
        salinity: Tensor, Sco2: Tensor,
        brie_e: Tensor | float | None = None,
    ) -> tuple[Tensor, Tensor]:
        rho_co2, K_co2 = self._co2(P, T)
        rho_brine, K_brine = self._bw.brine(T, P, salinity)
        den_mix = (1.0 - Sco2) * rho_brine + Sco2 * rho_co2
        if brie_e is None:
            Kf_mix = ((1.0 - Sco2) / (K_brine + EPS) + Sco2 / (K_co2 + EPS)) ** (-1)
        else:
            brie_e_t = torch.as_tensor(brie_e, dtype=T.dtype, device=T.device)
            Kf_mix = (K_brine - K_co2) * (1.0 - Sco2) ** brie_e_t + K_co2
        return den_mix, Kf_mix


__all__ = ["CO2Properties", "LiveOil", "CO2Brine"]
