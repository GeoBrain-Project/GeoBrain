"""
Gassmann + Biot + Geertsma-Smit + Brown-Korringa + Mavko-Jizba.

Private submodule of :mod:`geobrain.physics.rock.models.fluid`.
Public symbols are re-exported from ``fluid.py``.

Models:
    Gassmann:                Isotropic fluid substitution
    GassmannInverse:         Recover K_dry from K_sat
    GassmannFluidSub:        K_sat(fl1) → K_sat(fl2) via dry intermediate
    BiotHF:                  Biot high-frequency limiting velocities
    BiotDispersion:          Full frequency-dependent Biot dispersion
    GeertsmaSmitHF:          Geertsma-Smit high-frequency approximation
    GeertsmaSmitLF:          Geertsma-Smit low/middle-frequency interpolation
    BrownKorringa*:          Anisotropic dry/sat compliance variants
    MavkoJizba:              Mavko-Jizba squirt-flow high-frequency moduli

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ....core import GeoBrainError
from ..anisotropy import compliance_from_stiffness, stiffness_from_compliance
from ._categories import FluidModel
from ._registry import register
from ._types import EPS, PI
from ..fluid_substitution import (
    brown_korringa_dry_compliance,
    brown_korringa_saturated_compliance,
    brown_korringa_substitute_fluid,
)


def gassmann_k_sat(
    K_dry: Tensor,
    K_mineral: Tensor | float,
    K_fluid: Tensor | float,
    phi: Tensor | float,
    *,
    rtol: float = 1e-9,
) -> Tensor:
    """Saturated bulk modulus via Gassmann fluid substitution (stiffness form)::

        K_sat = K_dry + (1 - K_dry/K_min)^2
                / (phi/K_fl + (1-phi)/K_min - K_dry/K_min^2)

    The **single source of truth** for the Gassmann K_sat formula, shared by
    :class:`Gassmann` and the composite rock model. All moduli must be in the
    **same** units (Pa or GPa); the result carries that unit.

    In the incompressible/degenerate balance the denominator vanishes and
    ``K_sat -> K_dry``. The degeneracy test is **relative** to the pore/mineral
    compressibility ``scale``, a fixed absolute threshold false-triggers in Pa,
    where the denominator is O(1e-10). The ``where`` guard keeps the autograd
    gradient finite (a bare ``A / B`` back-propagates ``0*inf = NaN`` even where
    the quotient is masked out).
    """
    A = (1.0 - K_dry / K_mineral) ** 2
    scale = phi / K_fluid + (1.0 - phi) / K_mineral  # positive compressibility scale
    B = scale - K_dry / (K_mineral * K_mineral)
    if isinstance(B, torch.Tensor):
        scale_t = scale if isinstance(scale, torch.Tensor) else B.new_tensor(float(scale))
        cond = B.abs() < rtol * scale_t.abs()
        safe_B = torch.where(cond, torch.ones_like(B), B)
        return K_dry + torch.where(cond, torch.zeros_like(B), A / safe_B)
    thr = rtol * abs(scale)
    return K_dry + (0.0 if abs(B) < thr else A / B)


# --- Gassmann fluid substitution -------------------------------------------


@register("Gassmann", aliases=["gassmann"])
class Gassmann(FluidModel):
    """
    Gassmann isotropic fluid substitution.

    ``K_sat = K_dry + (1 − K_dry/K_min)² / (φ/K_fl + (1−φ)/K_min − K_dry/K_min²)``
    ``ρ_sat = (1 − φ) · ρ_min + φ · ρ_fl``
    ``vp    = √((K_sat + 4 μ_dry / 3) / ρ_sat)``

    Returns ``(vp, ρ)``, vp in m/s, density in SI kg/m³.
    """

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        rho_mineral: float = 2650.0,
        mu_dry: float = 20.0e9,
    ) -> None:
        super().__init__()
        for name, value in (
            ("K_mineral", K_mineral),
            ("rho_mineral", rho_mineral),
            ("mu_dry", mu_dry),
        ):
            if value <= 0:
                raise GeoBrainError(
                    f"Gassmann {name} must be positive",
                    object_name="Gassmann",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        self.K_mineral = float(K_mineral)
        self.rho_mineral = float(rho_mineral)
        self.mu_dry = float(mu_dry)

    def forward(
        self,
        K_dry: Tensor,
        phi: Tensor,
        K_fluid: Tensor | float,
        rho_fluid: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        K_sat = gassmann_k_sat(K_dry, self.K_mineral, K_fluid, phi)

        rho_si = (1.0 - phi) * self.rho_mineral + phi * rho_fluid
        vp = torch.sqrt((K_sat + (4.0 / 3.0) * self.mu_dry) / rho_si)
        return vp, rho_si  # SI kg/m³ (the /1000 conversion boundary is gone)


@register("GassmannInverse", aliases=["gassmann_inverse"])
class GassmannInverse(FluidModel):
    """
    Inverse Gassmann: recover ``K_dry`` from ``K_sat``.

    Returns ``(K_dry, G_sat)``. ``G_sat = G_dry`` by Gassmann invariance.
    """

    def forward(
        self,
        K_sat: Tensor,
        G_sat: Tensor,
        K0: Tensor,
        K_fl: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        A = K_sat * (phi * K0 / K_fl + 1 - phi) - K0
        B = phi * K0 / K_fl + K_sat / K0 - 1 - phi
        K_dry = A / (B + EPS)
        return K_dry, G_sat


@register("GassmannFluidSub", aliases=["FluidSub", "gassmann_fluid_sub"])
class GassmannFluidSub(FluidModel):
    """
    Gassmann fluid-to-fluid substitution: ``K_sat(fl1) → K_sat(fl2)`` via dry.

    Composes :class:`GassmannInverse` (sat→dry with fl1) and a Gassmann
    forward (dry→sat with fl2).
    """

    def __init__(self) -> None:
        super().__init__()
        self._inv = GassmannInverse()

    def forward(
        self,
        K_sat1: Tensor,
        G_sat1: Tensor,
        K0: Tensor,
        K_fl1: Tensor,
        K_fl2: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        K_dry, G_dry = self._inv(K_sat1, G_sat1, K0, K_fl1, phi)
        # Re-saturate with fluid 2 through the single shared Gassmann core. Its
        # divisor guard is **relative**: a fixed absolute 1e-10 threshold false-
        # triggers in Pa units (where B ~ 1e-10) and would silently zero the
        # substitution back to K_dry for a stiff replacement fluid (e.g. brine).
        K_sat2 = gassmann_k_sat(K_dry, K0, K_fl2, phi)
        return K_sat2, G_dry


# --- Biot poroelasticity ---------------------------------------------------


@register("BiotHF", aliases=["biot_hf"])
class BiotHF(FluidModel):
    """Biot high-frequency limiting velocities: returns ``(Vp_fast, Vp_slow, Vs)`` in m/s."""

    def forward(
        self,
        K_dry: Tensor,
        G_dry: Tensor,
        K0: Tensor,
        K_fl: Tensor,
        rho0: Tensor,
        rho_fl: Tensor,
        phi: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rho = (1.0 - phi) * rho0 + phi * rho_fl
        rho12 = (1.0 - alpha) * phi * rho_fl
        rho22 = alpha * phi * rho_fl
        rho11 = (1.0 - phi) * rho0 - (1.0 - alpha) * phi * rho_fl

        T1 = 1.0 - phi - K_dry / (K0 + EPS)
        T2 = phi * K0 / (K_fl + EPS)
        R = phi**2 * K0 / (T1 + T2 + EPS)
        Q = T1 * phi * K0 / (T1 + T2 + EPS)
        P = ((1.0 - phi) * T1 * K0 + T2 * K_dry) / (T1 + T2 + EPS) + 4.0 * G_dry / 3.0

        Delta = P * rho22 + R * rho11 - 2.0 * Q * rho12
        T3 = rho11 * rho22 - rho12**2
        T4 = P * R - Q**2
        disc = torch.clamp(Delta**2 - 4.0 * T3 * T4, min=EPS)

        Vp_fast = torch.sqrt((Delta + torch.sqrt(disc)) / (2.0 * T3 + EPS)) * 1000.0
        Vp_slow = (
            torch.sqrt(torch.clamp((Delta - torch.sqrt(disc)) / (2.0 * T3 + EPS), min=EPS)) * 1000.0
        )
        Vs = torch.sqrt(G_dry / (rho - phi * rho_fl / (alpha + EPS) + EPS)) * 1000.0
        return Vp_fast, Vp_slow, Vs


@register("BiotDispersion", aliases=["biot_dispersion"])
class BiotDispersion(FluidModel):
    """Full frequency-dependent Biot dispersion: returns ``(Vp_fast, Vp_slow, Vs, 1/Q_P1, 1/Q_P2, 1/Qs)``."""

    def forward(
        self,
        K_dry: Tensor,
        G_dry: Tensor,
        K0: Tensor,
        K_fl: Tensor,
        rho0: Tensor,
        rho_fl: Tensor,
        eta: Tensor,
        phi: Tensor,
        kapa: Tensor,
        a: Tensor,
        alpha: Tensor,
        freq: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        rho = (1.0 - phi) * rho0 + phi * rho_fl
        D = K0 * (1.0 + phi * (K0 / (K_fl + EPS) - 1.0))
        M = K0**2 / (D - K_dry + EPS)
        C_coeff = (K0 - K_dry) * K0 / (D - K_dry + EPS)
        H = K_dry + 4.0 * G_dry / 3.0 + (K0 - K_dry) ** 2 / (D - K_dry + EPS)

        w = 2.0 * PI * freq
        zeta = torch.sqrt(w * a**2 * rho_fl / (eta + EPS))
        F = torch.where(zeta < 0.1, torch.ones_like(zeta), zeta / 4.0)

        q_real = alpha * rho_fl / (phi + EPS)
        q_imag = eta * F / (w * kapa + EPS)
        q = q_real.to(torch.complex64) - 1j * q_imag.to(torch.complex64)

        rho_c = rho.to(torch.complex64)
        rho_fl_c = rho_fl.to(torch.complex64)
        H_c = H.to(torch.complex64)
        C_c = C_coeff.to(torch.complex64)
        M_c = M.to(torch.complex64)
        G_dry_c = G_dry.to(torch.complex64)

        Ta = C_c**2 - M_c * H_c
        Tb = H_c * q + M_c * rho_c - 2.0 * C_c * rho_fl_c
        Tc = rho_fl_c**2 - rho_c * q
        disc = Tb**2 - 4.0 * Ta * Tc
        P1_s2 = (-Tb + torch.sqrt(disc)) / (2.0 * Ta + EPS)
        P2_s2 = (-Tb - torch.sqrt(disc)) / (2.0 * Ta + EPS)
        S_s2 = (rho_c * q - rho_fl_c**2) / (G_dry_c * q + EPS)

        Vp_fast = 1.0 / torch.sqrt(P1_s2).real * 1000.0
        Vp_slow = 1.0 / torch.sqrt(P2_s2).real * 1000.0
        Vs = 1.0 / torch.sqrt(S_s2).real * 1000.0
        QP1_inv = (1.0 / P1_s2).imag / ((1.0 / P1_s2).real + EPS)
        QP2_inv = (1.0 / P2_s2).imag / ((1.0 / P2_s2).real + EPS)
        Qs_inv = (1.0 / S_s2).imag / ((1.0 / S_s2).real + EPS)
        return Vp_fast, Vp_slow, Vs, QP1_inv, QP2_inv, Qs_inv


# --- Geertsma-Smit approximations ------------------------------------------


@register("GeertsmaSmitHF", aliases=["gs_hf"])
class GeertsmaSmitHF(FluidModel):
    """Geertsma-Smit high-frequency approximation: returns ``(Vp_fast, Vs)`` in m/s."""

    def forward(
        self,
        K_dry: Tensor,
        G_dry: Tensor,
        K0: Tensor,
        K_fl: Tensor,
        rho0: Tensor,
        rho_fl: Tensor,
        phi: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, Tensor]:
        rho = (1.0 - phi) * rho0 + phi * rho_fl
        rho_biot = rho0 * (1.0 - phi) + phi * rho_fl * (1.0 - 1.0 / (alpha + EPS))
        Hdry = K_dry + 4.0 * G_dry / 3.0
        T1 = phi * rho / (rho_fl * alpha + EPS)
        alpha_biot = 1.0 - K_dry / (K0 + EPS)

        Vp_fast = (
            torch.sqrt(
                1.0
                / (rho_biot + EPS)
                * (
                    Hdry
                    + (T1 + alpha_biot * (alpha_biot - 2.0 * phi / (alpha + EPS)))
                    / ((alpha_biot - phi) / (K0 + EPS) + phi / (K_fl + EPS) + EPS)
                )
            )
            * 1000.0
        )
        Vs = torch.sqrt(G_dry / (rho_biot + EPS)) * 1000.0
        return Vp_fast, Vs


@register("GeertsmaSmitLF", aliases=["gs_lf"])
class GeertsmaSmitLF(FluidModel):
    """Geertsma-Smit low/middle-frequency interpolation between Vp(low) and Vp(∞)."""

    def forward(
        self,
        Vp0: Tensor,
        Vpinf: Tensor,
        freq: Tensor,
        phi: Tensor,
        rho_fl: Tensor,
        kapa: Tensor,
        eta: Tensor,
    ) -> Tensor:
        fc = phi * eta / (2.0 * PI * rho_fl * kapa + EPS)
        a_coeff = (fc / (freq + EPS)) ** 2
        return torch.sqrt((Vpinf**4 + Vp0**4 * a_coeff) / (Vpinf**2 + Vp0**2 * a_coeff + EPS))


# --- Brown-Korringa compliance variants ------------------------------------


@register("BrownKorringaDry2Sat", aliases=["bk_dry2sat"])
class BrownKorringaDry2Sat(FluidModel):
    """
    Brown-Korringa anisotropic dry → saturated compliance.

    Different API from the anisotropy-category ``BrownKorringa``: this one
    operates on the **6×6 compliance matrix** ``S_dry`` directly, returning
    the saturated 6×6 ``S_sat``. Use it when you already have the dry
    stiffness/compliance available.
    """

    def forward(
        self,
        Sdry: Tensor,
        K0: Tensor,
        G0: Tensor,
        K_fl: Tensor,
        phi: Tensor,
    ) -> Tensor:
        return brown_korringa_saturated_compliance(Sdry, K0, G0, K_fl, phi)


@register("BrownKorringaSat2Dry", aliases=["bk_sat2dry"])
class BrownKorringaSat2Dry(FluidModel):
    """Brown-Korringa anisotropic saturated → dry compliance inverse."""

    def forward(
        self,
        Ssat: Tensor,
        K0: Tensor,
        G0: Tensor,
        K_fl: Tensor,
        phi: Tensor,
    ) -> Tensor:
        return brown_korringa_dry_compliance(Ssat, K0, G0, K_fl, phi)


@register("BrownKorringaSub", aliases=["bk_sub"])
class BrownKorringaSub(FluidModel):
    """
    Brown-Korringa anisotropic fluid 1 → fluid 2 substitution.

    Composes sat→dry (with fluid 1) and dry→sat (with fluid 2). Returns
    ``(C_sat2, S_sat2)``: both 6×6, stiffness and compliance.
    """

    def forward(
        self,
        Csat: Tensor,
        K0: Tensor,
        G0: Tensor,
        K_fl1: Tensor,
        K_fl2: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        Ssat = compliance_from_stiffness(Csat)
        Ssat2 = brown_korringa_substitute_fluid(Ssat, K0, G0, K_fl1, K_fl2, phi)
        Csat2 = stiffness_from_compliance(Ssat2)
        return Csat2, Ssat2


# --- Mavko-Jizba squirt flow -----------------------------------------------


@register("MavkoJizba", aliases=["mj", "mavko_jizba"])
class MavkoJizba(FluidModel):
    """
    Mavko-Jizba squirt-flow high-frequency saturated moduli.

    Returns ``(K_uf_sat, G_uf_sat, Vp_hf, Vs_hf)``, saturated unrelaxed
    moduli and the corresponding high-frequency velocities (m/s).
    """

    def forward(
        self,
        Vp_hs: Tensor,
        Vs_hs: Tensor,
        Vpdry: Tensor,
        Vsdry: Tensor,
        K0: Tensor,
        rhodry: Tensor,
        rho_fl: Tensor,
        K_fl: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        vp_km, vs_km = Vpdry / 1000.0, Vsdry / 1000.0
        K_dry = rhodry * vp_km**2 - 4.0 / 3.0 * rhodry * vs_km**2
        G_dry = rhodry * vs_km**2

        vp_hs_km, vs_hs_km = Vp_hs / 1000.0, Vs_hs / 1000.0
        Khs = rhodry * vp_hs_km**2 - 4.0 / 3.0 * rhodry * vs_hs_km**2
        Kuf = Khs

        A = (1.0 - Kuf / (K0 + EPS)) ** 2
        B = phi / (K_fl + EPS) + (1.0 - phi) / (K0 + EPS) - Kuf / (K0**2 + EPS)
        Kuf_sat = Kuf + A / (B + EPS)

        Guf_sat_inv = 1.0 / (G_dry + EPS) - 4.0 / 15.0 * (1.0 / (K_dry + EPS) - 1.0 / (Kuf + EPS))
        Guf_sat = 1.0 / (Guf_sat_inv + EPS)

        rho_sat = rhodry + phi * rho_fl
        Vp_hf = torch.sqrt((Kuf_sat + 4.0 / 3.0 * Guf_sat) / (rho_sat + EPS)) * 1000.0
        Vs_hf = torch.sqrt(Guf_sat / (rho_sat + EPS)) * 1000.0
        return Kuf_sat, Guf_sat, Vp_hf, Vs_hf


__all__ = [
    "Gassmann",
    "GassmannInverse",
    "GassmannFluidSub",
    "BiotHF",
    "BiotDispersion",
    "GeertsmaSmitHF",
    "GeertsmaSmitLF",
    "BrownKorringaDry2Sat",
    "BrownKorringaSat2Dry",
    "BrownKorringaSub",
    "MavkoJizba",
]
