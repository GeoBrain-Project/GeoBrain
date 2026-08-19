"""
Pressure-temperature CO₂ / brine PVT correlations (differentiable torch).

This submodule supplies the *pressure(+temperature)-dependent* viscosity and the
CO₂-in-brine solubility / equilibrium ratio that the flow CO₂-storage PVT
(:class:`~geobrain.physics.flow.properties.pvt_co2.PVTCO2Brine`) needs but the
rock-physics density EOS does not carry. All functions are pure ``torch`` so the
mobility, accumulation and equilibrium terms they feed stay end-to-end
differentiable for inversion.

Published correlations implemented here:

* **CO₂ density**: adjusted Redlich-Kwong cubic equation of state with the
  constant-attraction parameters of *Spycher, Pruess & Ennis-King (2003)*. The
  gas / liquid (supercritical) molar-volume branch is the physical root of the
  cubic, selected by the lower molar Gibbs energy (Spycher et al., 2003, Appx
  B.2). This density feeds the CO₂ viscosity correlation below.
* **Brine (H₂O + NaCl) density**: specific-volume correlation of *Rowe & Chou
  (1970)*; this is the brine density the CO₂-storage PVT tables are built on
  (the brine analogue of the adjusted Redlich-Kwong CO₂ density). Salt-free
  reduces to the pure-water branch.
* **CO₂ viscosity**: *Fenghour, Wakeham & Vesovic (1998)* zero-density +
  excess-density form (with the Vesovic et al. 1990 critical scaling), as
  collected by Hassanzadeh et al. (2008). Function of temperature and the CO₂
  mass density.
* **Brine (H₂O + NaCl) viscosity**: pure-water viscosity of *Mao & Duan
  (2009)*, the NaCl factor of Mao & Duan (2009) and the dissolved-CO₂
  thickening factor of *Islam & Carlson (2012)*. Salt-free reduces to pure
  water; ``w_co2 = 0`` removes the dissolved-CO₂ term.
* **CO₂ solubility / K-value**: mutual solubilities of CO₂ and H₂O from
  *Spycher, Pruess & Ennis-King (2003)* (the Redlich-Kwong fugacity / activity
  formulation), giving the aqueous CO₂ mole fraction ``x_CO2`` and the vapour
  H₂O mole fraction ``y_H2O``; the CO₂ equilibrium ratio is
  ``K_CO2 = (1 − y_H2O) / x_CO2``.

Public inputs and outputs use canonical SI: pressure in **Pa**, temperature in
**K**, density in **kg/m³**, and viscosity in **Pa·s**. Published-correlation
pressure scales (bar or MPa) remain private implementation details.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

# --- physical constants (correlation units) -------------------------------
_R_BAR = 83.1447  # gas constant [bar·cm³/(mol·K)]
_R_SI = 8.31446  # gas constant [J/(mol·K)]
_MW_CO2 = 44.0095e-3  # CO₂ molar mass [kg/mol]
_MW_H2O = 18.01528e-3  # H₂O molar mass [kg/mol]
_M_H2O = 1.0 / _MW_H2O  # moles H₂O per kg H₂O [mol/kg]
_EPS = 1e-30


def _as_t(x: Tensor | float, like: Tensor) -> Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


# --- CO₂ density: adjusted Redlich-Kwong (Spycher et al., 2003) -----------


def _co2_molar_volume_rk_bar(p_bar: Tensor, T_K: Tensor) -> Tensor:
    """Molar volume of CO₂ [cm³/mol] from the adjusted Redlich-Kwong EOS.

    Solves the cubic ``V³ − c2·V² − c1·V − c0 = 0`` (Spycher et al., 2003) and
    returns the physical root: the gas branch (largest root) or the liquid /
    supercritical branch (smallest root), whichever has the lower molar Gibbs
    energy. The constant-attraction mixing parameters assume pure CO₂.
    """
    p = p_bar
    T = T_K
    a_m = 7.54e7 - 4.13e4 * T  # bar·cm⁶·K^0.5/mol²  (Spycher et al., 2003)
    b_m = _as_t(27.8, T)  # cm³/mol

    # Cubic V³ − c2 V² − c1 V − c0 = 0
    c2 = _R_BAR * T / p
    c1 = (_R_BAR * T * b_m) / p - a_m / (p * torch.sqrt(T)) + b_m**2
    c0 = a_m * b_m / (p * torch.sqrt(T))

    # Depressed cubic substitution V = t + c2/3
    shift = c2 / 3.0
    # x³ + P x + Q = 0
    P = -(c2**2) / 3.0 - c1
    Q = -2.0 * (c2**3) / 27.0 - (c2 * c1) / 3.0 - c0

    disc = (Q / 2.0) ** 2 + (P / 3.0) ** 3
    sqrt_negP3 = torch.sqrt(torch.clamp(-P / 3.0, min=0.0))

    # --- one real root (disc > 0): Cardano ---
    sd = torch.sqrt(torch.clamp(disc, min=0.0))
    u = torch.sign(-Q / 2.0 + sd) * torch.abs(-Q / 2.0 + sd) ** (1.0 / 3.0)
    v = torch.sign(-Q / 2.0 - sd) * torch.abs(-Q / 2.0 - sd) ** (1.0 / 3.0)
    root_single = u + v + shift

    # --- three real roots (disc <= 0): trigonometric ---
    # protect acos argument
    denom = 2.0 * sqrt_negP3**3 + _EPS
    arg = torch.clamp((3.0 * Q) / (P * denom + _EPS * torch.sign(P)), -1.0, 1.0)
    theta = torch.acos(arg)
    r1 = 2.0 * sqrt_negP3 * torch.cos(theta / 3.0) + shift
    r2 = 2.0 * sqrt_negP3 * torch.cos((theta + 2.0 * torch.pi) / 3.0) + shift
    r3 = 2.0 * sqrt_negP3 * torch.cos((theta + 4.0 * torch.pi) / 3.0) + shift

    three = disc <= 0.0
    V_gas = torch.where(three, torch.maximum(torch.maximum(r1, r2), r3), root_single)
    V_liq = torch.where(three, torch.minimum(torch.minimum(r1, r2), r3), root_single)

    # Gibbs-energy selection between gas and liquid branches (Spycher Appx B.2):
    #   pick the branch giving the lower departure Gibbs energy.
    w1 = p * (V_gas - V_liq)
    w21 = _R_BAR * T * torch.log(torch.clamp((V_gas - b_m) / (V_liq - b_m + _EPS), min=_EPS))
    w22 = (
        a_m
        / (b_m * torch.sqrt(T))
        * torch.log(torch.clamp(((V_gas + b_m) * V_liq) / ((V_liq + b_m) * V_gas + _EPS), min=_EPS))
    )
    w2 = w21 + w22
    pick_gas = (w2 - w1) > 0.0
    V = torch.where(three, torch.where(pick_gas, V_gas, V_liq), root_single)
    return V


def co2_molar_volume_rk(pressure_pa: Tensor, temperature_k: Tensor) -> Tensor:
    """CO₂ molar volume [cm³/mol] for pressure [Pa] and temperature [K]."""

    return _co2_molar_volume_rk_bar(pressure_pa * 1.0e-5, temperature_k)


def co2_density_rk(pressure_pa: Tensor, temperature_k: Tensor) -> Tensor:
    """CO₂ mass density [kg/m³] from the adjusted Redlich-Kwong EOS."""
    V = co2_molar_volume_rk(pressure_pa, temperature_k)  # cm³/mol
    rhox = (1.0 / V) * 1e6  # mol/m³
    return rhox * _MW_CO2  # kg/m³


# --- Brine density: Rowe & Chou (1970) ------------------------------------


def brine_density_rowe_chou(
    pressure_pa: Tensor,
    temperature_k: Tensor,
    salinity: float | Tensor = 0.0,
) -> Tensor:
    """Brine (H₂O + NaCl) mass density [kg/m³] of *Rowe & Chou (1970)*.

    Specific-volume correlation of A. M. Rowe & J. C. S. Chou, *Pressure-volume-
    temperature-concentration relation of aqueous NaCl solutions* (J. Chem. Eng.
    Data 15(1), 1970). ``salinity`` is the NaCl weight fraction (0 = salt-free,
    where the salt terms vanish). Valid over T ≈ 293–423 K and P ≤ 350 bar. Pure
    ``torch`` and differentiable in pressure (and temperature). This is the brine
    density the CO₂-storage PVT tables of reservoir simulators are built on, so
    it is the brine analogue of the adjusted Redlich-Kwong CO₂ density above.
    """
    T = temperature_k
    S = _as_t(salinity, T)
    q = pressure_pa * 1.0e-5 * 1.01972  # Pa → bar → kgf/cm²

    a1 = 5.916365 - 0.01035794 * T + 0.9270048e-5 * T**2 - 1127.522 / T + 100674.1 / T**2
    a2 = 0.520491e-2 - 0.10482101e-4 * T + 0.8328532e-8 * T**2 - 1.1702939 / T + 102.2783 / T**2
    a3 = 0.118547e-7 - 0.6599143e-10 * T
    a4 = -2.5166 + 0.0111766 * T - 0.170522e-4 * T**2
    a5 = 2.84851 - 0.0154305 * T + 0.223982e-4 * T**2
    a6 = -0.0014814 + 0.82969e-5 * T - 0.12469e-7 * T**2
    a7 = 0.0027141 - 0.15391e-4 * T + 0.22655e-7 * T**2
    a8 = 0.62158e-6 - 0.40075e-8 * T + 0.65972e-11 * T**2

    specific_volume = (
        a1
        - q * a2
        - q**2 * a3
        + a4 * S
        + a5 * S**2
        - q * a6 * S
        - q * a7 * S**2
        - 0.5 * q**2 * a8 * S
    ) * 0.001  # cm³/g → m³/kg
    return 1.0 / specific_volume  # kg/m³


# --- CO₂ viscosity: Fenghour-Wakeham-Vesovic (1998) -----------------------


def co2_viscosity_fenghour(T_K: Tensor, rho_co2: Tensor) -> Tensor:
    """CO₂ dynamic viscosity [Pa·s] (Fenghour, Wakeham & Vesovic, 1998).

    Zero-density term from the Vesovic et al. (1990) collision integral plus the
    excess-viscosity polynomial in mass density ``rho_co2`` [kg/m³]. Below
    313.15 K a small linear-in-density correction is added (Hassanzadeh et al.,
    2008 collection).
    """
    T = T_K
    rho = rho_co2
    g = [
        0.4071119e-2,
        0.7198037e-4,
        0.2411697e-16,
        0.2971072e-22,
        -0.1627888e-22,
    ]
    e = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
    f = [5.5934e-3, 6.1757e-5, 0.0, 2.6430e-11]

    T_x = T / 251.196
    logTx = torch.log(T_x)
    # Reduced collision integral of the Fenghour-Wakeham-Vesovic (1998)
    # zero-density viscosity (ε/k = 251.196 K):
    #   ln Ψ_μ = Σ_{i=0..4} e_i (ln T*)^i
    # i.e. a degree-4 polynomial in ln T*: NOT the outer product of the
    # coefficient sum with the power sum (Σ e_i)·(Σ (ln T*)^j), which is a
    # different number and over-predicts the dilute-gas viscosity by ~40%.
    ln_psi = e[0] + e[1] * logTx + e[2] * logTx**2 + e[3] * logTx**3 + e[4] * logTx**4
    psi_m = torch.exp(ln_psi)

    a1 = 1.00697 * torch.sqrt(T) / psi_m
    a2 = g[0] * rho
    a3 = g[1] * rho**2
    a4 = g[2] * rho**6 / T_x**3
    a5 = g[3] * rho**8
    a6 = g[4] * rho**8 / T_x
    # Low-temperature linear-in-density correction (only T < 313.15 K).
    a7_full = f[0] * rho + f[1] * rho + f[2] * rho + f[3] * rho
    a7 = torch.where(T < 313.15, a7_full, torch.zeros_like(a7_full))

    mu = (a1 + a2 + a3 + a4 + a5 + a6 + a7) / 1e6  # μPa·s → Pa·s
    return mu


def co2_viscosity(pressure_pa: Tensor, temperature_k: Tensor) -> Tensor:
    """CO₂ viscosity [Pa·s] as a function of pressure and temperature.

    Composes the Redlich-Kwong density with the Fenghour-Wakeham-Vesovic
    viscosity correlation.
    """
    rho = co2_density_rk(pressure_pa, temperature_k)
    return co2_viscosity_fenghour(temperature_k, rho)


# --- Brine viscosity: Mao & Duan (2009) + Islam-Carlson (2012) ------------


def brine_viscosity(
    pressure_pa: Tensor,
    temperature_k: Tensor,
    m_nacl: Tensor | float = 0.0,
    w_co2: Tensor | float = 0.0,
) -> Tensor:
    """Brine (H₂O + NaCl + dissolved CO₂) dynamic viscosity [Pa·s].

    Pure-water viscosity of Mao & Duan (2009); NaCl multiplier of Mao & Duan
    (2009); dissolved-CO₂ thickening of Islam & Carlson (2012). ``m_nacl`` is
    NaCl molality [mol/kg], ``w_co2`` the aqueous CO₂ mass fraction. Salt-free
    (``m_nacl = 0``) uses the Islam-Carlson pure-water-plus-CO₂ branch.
    """
    T = temperature_k
    m = _as_t(m_nacl, T)
    w = _as_t(w_co2, T)
    P_mpa = pressure_pa * 1.0e-6

    # Pure-water density (Islam & Carlson, 2012) [g/cm³]
    a = 1.34136579e2
    b = [-4.07743800e3, 1.63192756e4, 1.37091355e3]
    c = [-5.56126409e-3, -1.07149234e-2, -5.46294495e-4]
    d = [4.45861703e-1, -4.51029739e-4]
    rho_h2o = (
        a
        + b[0] * 10.0 ** (c[0] * T)
        + b[1] * 10.0 ** (c[1] * T)
        + b[2] * 10.0 ** (c[2] * T)
        + d[0] * P_mpa
        + d[1] * P_mpa**2
    ) / 1e3

    # Pure-water viscosity (Mao & Duan, 2009)
    dd = [
        0.28853170e7,
        -0.11072577e5,
        -0.90834095e1,
        0.30925651e-1,
        -0.27407100e-4,
        -0.19283851e7,
        0.56216046e4,
        0.13827250e2,
        -0.47609523e-1,
        0.35545041e-4,
    ]
    # term1: sum_{k=1..5} dd[k] * T^(k-3);  term2: sum * rho * T^(k-8)
    t1 = dd[0] * T ** (-2) + dd[1] * T ** (-1) + dd[2] * T**0 + dd[3] * T**1 + dd[4] * T**2
    t2 = rho_h2o * (
        dd[5] * T ** (-2) + dd[6] * T ** (-1) + dd[7] * T**0 + dd[8] * T**1 + dd[9] * T**2
    )
    mu_h2o = torch.exp(t1 + t2)

    # Salt-free branch: H₂O + dissolved CO₂ (Islam & Carlson, 2012)
    a_ic = (1.632609119e3, -9.46077673946e2)
    b_ic = (-1.047187396332e4, 3.68325597e1)
    mu_r = 1.0 + (a_ic[0] * w + a_ic[1] * w**2) / (b_ic[0] + b_ic[1] * T)
    mu_free = mu_r * mu_h2o

    # Saline branch: NaCl factor (Mao & Duan, 2009) + CO₂ thickening (IC2012)
    A = -0.21319213 + 0.13651589e-2 * T - 0.12191756e-5 * T**2
    B = 0.69161945e-1 - 0.27292263e-3 * T + 0.20852448e-6 * T**2
    C = -0.25988855e-2 + 0.77989227e-5 * T
    mu_b = torch.exp(A * m + B * m**2 + C * m**3) * mu_h2o
    mu_saline = mu_b * (1.0 + 4.65 * torch.clamp(w, min=0.0) ** 1.0134)

    return torch.where(m == 0.0, mu_free, mu_saline)


# --- CO₂ solubility / K-value: Spycher, Pruess & Ennis-King (2003) --------


def co2_brine_equilibrium(
    pressure_pa: Tensor,
    temperature_k: Tensor,
) -> tuple[Tensor, Tensor]:
    """Mutual CO₂ / H₂O solubility (Spycher, Pruess & Ennis-King, 2003).

    Returns ``(x_co2, y_h2o)``: the aqueous CO₂ mole fraction and the vapour H₂O
    mole fraction for a pure-water brine (no salt activity correction). Uses the
    adjusted Redlich-Kwong fugacity coefficients and the temperature-dependent
    equilibrium constants of Spycher et al. (2003), with the partial molar
    volumes of CO₂ and H₂O for the Poynting pressure correction.
    """
    P = pressure_pa * 1.0e-5
    T = temperature_k
    T_C = T - 273.15
    p0 = 1.0  # reference pressure [bar]

    a_co2 = 7.54e7 - 4.13e4 * T
    a_h2o_co2 = _as_t(7.89e7, T)
    b_co2 = _as_t(27.8, T)
    b_h2o = _as_t(18.18, T)

    Vp_co2 = _as_t(32.6, T)  # cm³/mol
    Vp_h2o = _as_t(18.1, T)

    # Equilibrium constants at p0 = 1 bar (Spycher et al., 2003)
    kap0_co2_g = 10.0 ** (1.189 + 1.304e-2 * T_C - 5.446e-5 * T_C**2)
    kap0_co2_l = 10.0 ** (1.169 + 1.368e-2 * T_C - 5.380e-5 * T_C**2)
    kap0_h2o = 10.0 ** (-2.209 + 3.097e-2 * T_C - 1.098e-4 * T_C**2 + 2.048e-7 * T_C**3)

    # Pure-CO₂ mixing rules
    a_m = a_co2
    b_m = b_co2
    V = _co2_molar_volume_rk_bar(P, T)  # cm³/mol
    Z = P * V / (_R_BAR * T)

    fa = torch.log(V / (V - b_m))
    fb = b_co2 / (V - b_m)
    fc = (2.0 * a_co2 / (_R_BAR * T**1.5 * b_m)) * torch.log((V + b_m) / V)
    fd = a_m * b_co2 / (_R_BAR * T**1.5 * b_m**2) * (torch.log((V + b_m) / V) - b_m / (V + b_m))
    fe = torch.log(Z)
    phi_co2 = torch.exp(fa + fb - fc + fd - fe)

    fbw = b_h2o / (V - b_m)
    fcw = (2.0 * a_h2o_co2 / (_R_BAR * T**1.5 * b_m)) * torch.log((V + b_m) / V)
    fdw = a_m * b_h2o / (_R_BAR * T**1.5 * b_m**2) * (torch.log((V + b_m) / V) - b_m / (V + b_m))
    phi_h2o = torch.exp(fa + fbw - fcw + fdw - fe)

    # Liquid CO₂ equilibrium constant below the CO₂ critical T at dense states.
    use_l = (T_C < 31.0) & (V < 94.0)
    kap0_co2 = torch.where(use_l, kap0_co2_l, kap0_co2_g)

    B = phi_co2 * P / (_M_H2O * kap0_co2) * torch.exp(-(P - p0) * Vp_co2 / (_R_BAR * T))
    A = kap0_h2o / (phi_h2o * P) * torch.exp((P - p0) * Vp_h2o / (_R_BAR * T))

    y_h2o = (1.0 - B) / (1.0 / A - B)
    x_co2 = B * (1.0 - y_h2o)

    y_h2o = torch.clamp(y_h2o, 0.0, 1.0)
    x_co2 = torch.clamp(x_co2, 0.0, 1.0)
    return x_co2, y_h2o


def co2_solubility(pressure_pa: Tensor, temperature_k: Tensor) -> Tensor:
    """Aqueous CO₂ mole fraction ``x_CO2`` (Spycher et al., 2003)."""
    x_co2, _ = co2_brine_equilibrium(pressure_pa, temperature_k)
    return x_co2


def co2_kvalue(pressure_pa: Tensor, temperature_k: Tensor) -> Tensor:
    """CO₂ equilibrium ratio ``K_CO2 = (1 − y_H2O) / x_CO2`` (Spycher et al., 2003)."""
    x_co2, y_h2o = co2_brine_equilibrium(pressure_pa, temperature_k)
    y_co2 = torch.clamp(1.0 - y_h2o, min=1e-20)
    x_co2 = torch.clamp(x_co2, min=1e-20)
    return y_co2 / x_co2


__all__ = [
    "co2_molar_volume_rk",
    "co2_density_rk",
    "brine_density_rowe_chou",
    "co2_viscosity_fenghour",
    "co2_viscosity",
    "brine_viscosity",
    "co2_brine_equilibrium",
    "co2_solubility",
    "co2_kvalue",
]
