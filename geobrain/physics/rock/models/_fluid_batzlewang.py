"""
Batzle-Wang fluid EOS classes (brine, gas, oil dead/live + unified).

Private submodule of :mod:`geobrain.physics.rock.models.fluid`.
Public symbols are re-exported from ``fluid.py``.

Models:
    BatzleWangBrine:    Brine K/ρ vs (T, P, salinity)
    BatzleWangGas:      Hydrocarbon gas K/ρ vs (T, P, γ_G)
    BatzleWangOilDead: Dead crude oil K/ρ vs (T, P, ρ₀)
    BatzleWangOilLive: Live crude oil K/ρ vs (T, P, ρ₀, GOR, γ_G)
    BatzleWang:         Unified dispatcher (brine/gas/oil) returning BW units

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ....core import GeoBrainError
from ._categories import FluidModel
from ._registry import register


# --- Batzle & Wang 1992 brine table -----------------------------------------

# v_water(T, P) = Σ_{i=0..4} Σ_{j=0..3} W_ij · T^i · P^j
# (Batzle & Wang 1992 Table 1; Mavko et al. 2009 Table 6.4.4)
_BW_W = (
    (1402.85, 1.524, 3.437e-3, -1.197e-5),
    (4.871, -0.0111, 1.739e-4, -1.628e-6),
    (-0.04783, 2.747e-4, -2.135e-6, 1.237e-8),
    (1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10),
    (-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13),
)


def _bw_velocity_water(T: Tensor, P: Tensor) -> Tensor:
    v = torch.zeros_like(T)
    for i, row in enumerate(_BW_W):
        Tp = T.pow(i)
        for j, w in enumerate(row):
            v = v + w * Tp * P.pow(j)
    return v


def _bw_density_water(T: Tensor, P: Tensor) -> Tensor:
    return 1.0 + 1.0e-6 * (
        -80.0 * T
        - 3.3 * T.pow(2)
        + 0.00175 * T.pow(3)
        + 489.0 * P
        - 2.0 * T * P
        + 0.016 * T.pow(2) * P
        - 1.3e-5 * T.pow(3) * P
        - 0.333 * P.pow(2)
        - 0.002 * T * P.pow(2)
    )


# --- Single-source Batzle-Wang physics (SI: K in Pa, rho in kg/m^3) ---------
# The registered split operators AND the convenience ``BatzleWang`` dispatcher
# both delegate here, so there is exactly one implementation per fluid, a
# second implementation invites silent divergence (a re-implemented gas path
# once drifted ~6% from the literature-validated one). Inputs: T in °C,
# P in MPa; the phase parameter (salinity / gas gravity / reference density) may
# be a scalar or a tensor.


def _bw_brine_si(T: Tensor, P: Tensor, S: Tensor | float) -> tuple[Tensor, Tensor]:
    """Brine ``(K [Pa], rho [kg/m^3])``: Batzle & Wang 1992."""
    rho_w = _bw_density_water(T, P)
    rho_brine = rho_w + S * (
        0.668 + 0.44 * S
        + 1.0e-6 * (
            300.0 * P
            - 2400.0 * P * S
            + T * (80.0 + 3.0 * T - 3300.0 * S - 13.0 * P + 47.0 * P * S)
        )
    )
    v_w = _bw_velocity_water(T, P)
    v_brine = (
        v_w
        + S * (1170.0 - 9.6 * T + 0.055 * T.pow(2) - 8.5e-5 * T.pow(3)
               + 2.6 * P - 0.0029 * T * P - 0.0476 * P.pow(2))
        + (S ** 1.5) * (780.0 - 10.0 * P + 0.16 * P.pow(2))
        # -1820 S²: Batzle & Wang (1992) eq. 29 (also Rock Physics Handbook).
        # A dropped leading digit here (-820) shifts K/K by ~1.5e-3 at
        # S=0.035: the EOS registry-parity suite pins the exact value.
        # rho does not depend on this coefficient.
        - 1820.0 * (S ** 2)
    )
    rho_si = rho_brine * 1000.0
    return rho_si * v_brine.pow(2), rho_si


def _bw_gas_si(T_C: Tensor, P: Tensor, G: Tensor | float) -> tuple[Tensor, Tensor]:
    """Hydrocarbon gas ``(K [Pa], rho [kg/m^3])``: Standing-Katz Z (Mavko 7.30–7.37)."""
    T_a = T_C + 273.15
    T_pc = 94.72 + 170.75 * G
    P_pc = 4.892 - 0.4048 * G
    T_pr = T_a / T_pc
    P_pr = P / P_pc

    a = 0.03 + 0.00527 * (3.5 - T_pr).pow(3)
    b = 0.642 * T_pr - 0.007 * T_pr.pow(4) - 0.52
    alpha = 0.45 + 8.0 * (0.56 - 1.0 / T_pr).pow(2)
    beta = alpha / T_pr
    exp_arg = -beta * P_pr.pow(1.2)
    c = 0.109 * (3.85 - T_pr).pow(2)
    E = c * torch.exp(exp_arg)
    Z = a * P_pr + b + E

    R_gas_const = 8.314
    M_air = 28.97e-3
    rho_gas = M_air * G * (P * 1.0e6) / (Z * R_gas_const * T_a)

    dE_dPpr = -1.2 * E * beta * P_pr.pow(0.2)
    dZ_dPpr = a + dE_dPpr
    gamma0 = (
        0.85
        + 5.6 / (P_pr + 2.0)
        + 27.1 / (P_pr + 3.5).pow(2)
        - 8.7 * torch.exp(-0.65 * (P_pr + 1.0))
    )
    denom = (1.0 - P_pr * dZ_dPpr / Z).clamp(min=1e-6)
    K_gas = gamma0 * (P * 1.0e6) / denom
    return K_gas, rho_gas


def _bw_oil_dead_si(T: Tensor, P: Tensor, rho_0: Tensor | float) -> tuple[Tensor, Tensor]:
    """Dead crude oil ``(K [Pa], rho [kg/m^3])``: Mavko 7.41–7.42."""
    rho_TP = (
        rho_0
        + (0.00277 * P - 1.71e-7 * P.pow(3)) * (rho_0 - 1.15) ** 2
        + 3.49e-4 * P
    )
    rho_oil = rho_TP / (0.972 + 3.81e-4 * (T + 17.78).pow(1.175))
    v_oil = (
        2096.0 * (rho_0 / (2.6 - rho_0)) ** 0.5
        - 3.7 * T
        + 4.64 * P
        + 0.0115 * (4.12 * (1.08 / rho_0 - 1.0) ** 0.5 - 1.0) * T * P
    )
    rho_si = rho_oil * 1000.0
    return rho_si * v_oil.pow(2), rho_si


@register("BatzleWangBrine", aliases=["bw_brine"])
class BatzleWangBrine(FluidModel):
    """
    Brine ``(K, ρ)`` vs ``(T, P, S)`` per Batzle & Wang 1992.

    Inputs (forward): ``temperature_C`` (°C), ``pressure_MPa`` (MPa).
    Returns ``(K_fluid, ρ_fluid)`` in Pa, kg/m³.
    """

    def __init__(self, *, salinity: float = 0.035) -> None:
        super().__init__()
        if not 0.0 <= salinity < 1.0:
            raise GeoBrainError(
                "Batzle-Wang salinity must be in [0, 1)",
                object_name="BatzleWangBrine",
                field="salinity",
                expected="[0, 1)",
                actual=salinity,
            )
        self.salinity = float(salinity)

    def forward(self, T: Tensor, P: Tensor) -> tuple[Tensor, Tensor]:
        return _bw_brine_si(T, P, self.salinity)


@register("BatzleWangGas", aliases=["bw_gas"])
class BatzleWangGas(FluidModel):
    """
    Hydrocarbon gas ``(K, ρ)`` vs ``(T, P)`` per Batzle & Wang 1992.

    Standing-Katz Z-factor approximation (Mavko et al. 2009 eqs. 7.30–7.37).
    Returns ``(K_fluid, ρ_fluid)`` in Pa, kg/m³.
    """

    def __init__(self, *, gas_gravity: float = 0.65) -> None:
        super().__init__()
        if not 0.5 < gas_gravity < 2.0:
            raise GeoBrainError(
                "BatzleWangGas gas_gravity must be in (0.5, 2.0)",
                object_name="BatzleWangGas",
                field="gas_gravity",
                expected="(0.5, 2.0)",
                actual=gas_gravity,
            )
        self.gas_gravity = float(gas_gravity)

    def forward(self, T_C: Tensor, P: Tensor) -> tuple[Tensor, Tensor]:
        return _bw_gas_si(T_C, P, self.gas_gravity)


@register("BatzleWangOilDead", aliases=["bw_oil_dead"])
class BatzleWangOilDead(FluidModel):
    """
    Dead crude oil ``(K, ρ)`` vs ``(T, P, ρ₀)`` per Batzle & Wang 1992.

    Mavko et al. 2009 eqs. 7.41–7.42. Returns ``(K_fluid, ρ_fluid)``
    in Pa, kg/m³.
    """

    def __init__(self, *, reference_density: float = 0.85) -> None:
        super().__init__()
        if not 0.5 < reference_density < 1.1:
            raise GeoBrainError(
                "BatzleWangOilDead reference_density must be in (0.5, 1.1) g/cm³",  # SI-EXEMPT: rho_0 is correlation-native g/cm^3
                object_name="BatzleWangOilDead",
                field="reference_density",
                expected="(0.5, 1.1)",
                actual=reference_density,
            )
        self.rho_0 = float(reference_density)

    def forward(self, T: Tensor, P: Tensor) -> tuple[Tensor, Tensor]:
        return _bw_oil_dead_si(T, P, self.rho_0)


@register("BatzleWangOilLive", aliases=["bw_oil_live"])
class BatzleWangOilLive(FluidModel):
    """
    Live crude oil ``(K, ρ)`` vs ``(T, P, ρ₀, GOR, γ_G)`` per Batzle & Wang 1992.

    Mavko et al. 2009 eqs. 7.43–7.47. Returns ``(K_fluid, ρ_fluid)``
    in Pa, kg/m³.
    """

    def __init__(
        self,
        *,
        reference_density: float = 0.85,
        gas_oil_ratio: float = 85.0,
        gas_gravity: float = 0.65,
    ) -> None:
        super().__init__()
        if not 0.5 < reference_density < 1.1:
            raise GeoBrainError(
                "BatzleWangOilLive reference_density must lie in (0.5, 1.1) g/cm³",  # SI-EXEMPT: rho_0 is correlation-native g/cm^3
                object_name="BatzleWangOilLive",
                field="reference_density",
                expected="(0.5, 1.1)",
                actual=reference_density,
            )
        if not 0.0 <= gas_oil_ratio <= 500.0:
            raise GeoBrainError(
                "BatzleWangOilLive gas_oil_ratio must lie in [0, 500] L/L",
                object_name="BatzleWangOilLive",
                field="gas_oil_ratio",
                expected="[0, 500]",
                actual=gas_oil_ratio,
            )
        if not 0.5 < gas_gravity < 2.0:
            raise GeoBrainError(
                "BatzleWangOilLive gas_gravity must lie in (0.5, 2.0)",
                object_name="BatzleWangOilLive",
                field="gas_gravity",
                expected="(0.5, 2.0)",
                actual=gas_gravity,
            )
        self.rho_0 = float(reference_density)
        self.R_G = float(gas_oil_ratio)
        self.G = float(gas_gravity)

    def forward(self, T: Tensor, P: Tensor) -> tuple[Tensor, Tensor]:
        rho_0, R_G, G = self.rho_0, self.R_G, self.G
        B_0 = 0.972 + 3.8e-4 * (2.4 * R_G * (G / rho_0) ** 0.5 + T + 17.8).pow(1.175)
        rho_R = (rho_0 + 0.0012 * R_G * G) / B_0
        rho_pseudo = (rho_0 / B_0) * (1.0 / (1.0 + 0.001 * R_G))

        rho_TP = (
            rho_R
            + (0.00277 * P - 1.71e-7 * P.pow(3)) * (rho_R - 1.15).pow(2)
            + 3.49e-4 * P
        )
        rho_live = rho_TP / (0.972 + 3.81e-4 * (T + 17.78).pow(1.175))

        v_live = (
            2096.0 * torch.sqrt(rho_pseudo / (2.6 - rho_pseudo).clamp(min=1e-6))
            - 3.7 * T
            + 4.64 * P
            + 0.0115 * (4.12 * torch.sqrt((1.08 / rho_pseudo - 1.0).clamp(min=1e-6))
                        - 1.0) * T * P
        )
        rho_si = rho_live * 1000.0
        K_oil = rho_si * v_live.pow(2)
        return K_oil, rho_si


# --- Unified BatzleWang dispatcher ------------------------------------------


@register("BatzleWang", aliases=["BW", "batzle_wang"])
class BatzleWang(FluidModel):
    """
    Unified Batzle-Wang dispatcher: brine / gas / oil from (T, P).

    Methods ``brine``, ``gas``, ``oil`` return ``(rho, K)`` in SI units
    (rho in kg/m³, K in Pa); ``forward(T, P,
    fluid_type=…)`` dispatches by string. The four T2a split classes
    (BatzleWangBrine/Gas/OilDead/OilLive) remain available for cases
    where SI-unit outputs (kg/m³, Pa) and per-class registration suit
    better.
    """

    def brine(self, T: Tensor, P: Tensor, S: Tensor | float = 0.035) -> tuple[Tensor, Tensor]:
        T, P = torch.as_tensor(T), torch.as_tensor(P)
        S = torch.as_tensor(S, dtype=T.dtype, device=T.device)
        K_pa, rho_si = _bw_brine_si(T, P, S)
        return rho_si, K_pa        # SI (kg/m³, Pa)

    def gas(self, T: Tensor, P: Tensor, G: Tensor | float = 0.6) -> tuple[Tensor, Tensor]:
        T, P = torch.as_tensor(T), torch.as_tensor(P)
        G = torch.as_tensor(G, dtype=T.dtype, device=T.device)
        K_pa, rho_si = _bw_gas_si(T, P, G)
        return rho_si, K_pa        # SI (kg/m³, Pa)

    def oil(self, T: Tensor, P: Tensor, rho_0: Tensor | float = 0.8) -> tuple[Tensor, Tensor]:
        T, P = torch.as_tensor(T), torch.as_tensor(P)
        rho_0 = torch.as_tensor(rho_0, dtype=T.dtype, device=T.device)
        K_pa, rho_si = _bw_oil_dead_si(T, P, rho_0)
        return rho_si, K_pa        # SI (kg/m³, Pa)

    _ACCEPTED_KW = {"brine": ("S",), "gas": ("G",), "oil": ("rho_0",)}

    def forward(
        self, T: Tensor, P: Tensor,
        fluid_type: str = "brine", **kw,
    ) -> tuple[Tensor, Tensor]:
        # Unknown kwargs must not fall through ``kw.get`` silently;
        # ``salinity=0.05`` would be ignored and the default 0.035 used.
        # Reject anything but the documented per-fluid names.
        accepted = self._ACCEPTED_KW.get(fluid_type, ())
        unknown = set(kw) - set(accepted)
        if unknown:
            raise GeoBrainError(
                f"BatzleWang.forward got unknown keyword(s) {sorted(unknown)} "
                f"for fluid_type={fluid_type!r}, accepted: "
                f"{', '.join(repr(a) for a in accepted) or 'none'}",
                object_name="BatzleWang",
                field="kwargs",
                expected=accepted,
                actual=sorted(unknown),
            )
        if fluid_type == "brine":
            return self.brine(T, P, kw.get("S", 0.035))
        if fluid_type == "gas":
            return self.gas(T, P, kw.get("G", 0.6))
        if fluid_type == "oil":
            return self.oil(T, P, kw.get("rho_0", 0.8))
        raise GeoBrainError(
            f"Unknown fluid type: {fluid_type!r} (use 'brine', 'gas', or 'oil')",
            object_name="BatzleWang", field="fluid_type",
            expected="brine|gas|oil", actual=fluid_type,
        )


__all__ = [
    "BatzleWangBrine",
    "BatzleWangGas",
    "BatzleWangOilDead",
    "BatzleWangOilLive",
    "BatzleWang",
]
