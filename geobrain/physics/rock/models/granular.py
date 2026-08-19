"""
Granular-medium ``ComponentModel`` implementations.

Pure-math layer. The ``*Operator`` wrappers are colocated in this module.

Models:
    HertzMindlin: Mindlin contact theory at the critical porosity
                   (Mavko, Mukerji & Dvorkin 2009, eq. 5.43)
    SoftSand:      Modified HS lower bound (friable / uncemented sand)
    StiffSand:     Modified HS upper bound (cemented / consolidated sand)
    Walton:        Walton (1987) frictionless-contact alternative

All operate on dry-frame moduli; downstream Gassmann handles fluid effects.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    PropertyTransform,
)
from ._categories import GranularModel
from ._registry import register
from ._types import EPS, PI
from ..granular import (
    constant_cement_moduli,
    contact_cement_moduli,
    hertz_mindlin_moduli,
    modified_upper_hashin_shtrikman,
    patchy_cement_moduli,
    soft_sand_moduli,
    stiff_sand_moduli,
    varying_patchy_cement_moduli,
    walton_moduli,
)
from ..moduli import check_bounded


def _validate_positive_scalars(class_name: str, **pairs: float) -> None:
    for name, value in pairs.items():
        if value <= 0:
            raise GeoBrainError(
                f"{class_name} {name} must be positive",
                object_name=class_name,
                field=name,
                expected="> 0",
                actual=value,
            )


def _validate_phi_lt_one(class_name: str, phi: float) -> None:
    if phi >= 1:
        raise GeoBrainError(
            f"{class_name} phi_critical must be < 1",
            object_name=class_name,
            field="phi_critical",
            expected="< 1",
            actual=phi,
        )


def _mineral_poisson(K: float, mu: float) -> float:
    """Poisson's ratio from K, μ of a single isotropic mineral."""
    return (3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu))


# --- Hertz-Mindlin ----------------------------------------------------------


@register("HertzMindlin", aliases=["hertz_mindlin", "HM"])
class HertzMindlin(GranularModel):
    """
    Hertz-Mindlin dry-frame moduli at the critical porosity.

    K_HM ∝ (P_eff)^(1/3); μ_HM also ∝ (P_eff)^(1/3) with a Poisson-ratio
    factor. Returns ``(K_HM, μ_HM)`` in Pa.
    """

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
        coordination_n: float = 9.0,
    ) -> None:
        super().__init__()
        _validate_positive_scalars(
            "HertzMindlin",
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
            coordination_n=coordination_n,
        )
        _validate_phi_lt_one("HertzMindlin", phi_critical)
        self.K_mineral = float(K_mineral)
        self.mu_mineral = float(mu_mineral)
        self.phi_c = float(phi_critical)
        self.n = float(coordination_n)
        self.nu = _mineral_poisson(self.K_mineral, self.mu_mineral)

    def forward(self, P_eff: Tensor) -> tuple[Tensor, Tensor]:
        result = hertz_mindlin_moduli(
            P_eff,
            self.K_mineral,
            self.mu_mineral,
            self.phi_c,
            self.n,
        )
        return result.k_dry, result.mu_dry

    def compute_dry_rock(self, K0, G0, phi, **params):
        """Workflow hook: varies K0/G0/phi_c/Cn/P at call time (not constructor)."""
        from ._types import CN as _CN, F as _F, P as _P, PHI_C as _PHI_C

        return hm_moduli_v05(
            K0,
            G0,
            params.get("phi_c", _PHI_C),
            params.get("Cn", _CN),
            params.get("P", _P),
            params.get("f", _F),
        )


# --- Soft / Stiff sand bounds ----------------------------------------------


@register("SoftSand", aliases=["soft_sand", "friable_sand"])
class SoftSand(GranularModel):
    """
    Modified HS *lower* bound (friable / uncemented sand line).

    Interpolates between Hertz-Mindlin at φ = φ_c and the pure mineral
    at φ = 0 with ζ evaluated at the *soft* (HM) endpoint:

        ζ_HM = (μ_HM / 6) · (9 K_HM + 8 μ_HM) / (K_HM + 2 μ_HM)
        u    = φ / φ_c

        K_dry = (u/(K_HM + 4 μ_HM/3) + (1−u)/(K_min + 4 μ_HM/3))⁻¹ − 4 μ_HM/3
        μ_dry = (u/(μ_HM + ζ_HM)    + (1−u)/(μ_min + ζ_HM))⁻¹    − ζ_HM
    """

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
    ) -> None:
        super().__init__()
        _validate_positive_scalars(
            "SoftSand",
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
        )
        _validate_phi_lt_one("SoftSand", phi_critical)
        self.K_mineral = float(K_mineral)
        self.mu_mineral = float(mu_mineral)
        self.phi_c = float(phi_critical)

    def forward(
        self,
        K_HM: Tensor,
        mu_HM: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return soft_sand_math(self.K_mineral, self.mu_mineral, K_HM, mu_HM, phi, self.phi_c)

    def compute_dry_rock(self, K0, G0, phi, **params):
        """Workflow hook: varies all parameters at call time, not constructor."""
        from ._types import CN as _CN, F as _F, P as _P, PHI_C as _PHI_C

        phi_c = params.get("phi_c", _PHI_C)
        K_HM, mu_HM = hm_moduli_v05(
            K0,
            G0,
            phi_c,
            params.get("Cn", _CN),
            params.get("P", _P),
            params.get("f", _F),
        )
        return soft_sand_math(K0, G0, K_HM, mu_HM, phi, phi_c)


def soft_sand_math(
    K0,
    G0,
    K_HM,
    mu_HM,
    phi,
    phi_c,
) -> tuple[Tensor, Tensor]:
    """Compatibility view of the canonical HS-lower kernel."""
    result = soft_sand_moduli(phi, K0, G0, K_HM, mu_HM, phi_c)
    return result.k_dry, result.mu_dry


@register("StiffSand", aliases=["stiff_sand", "consolidated_sand"])
class StiffSand(GranularModel):
    """
    Modified HS *upper* bound (cemented / consolidated sand line).

    Same form as :class:`SoftSand` but ζ evaluated at the *stiff*
    (mineral) endpoint, and the K-formula denominators use μ_mineral
    everywhere:

        ζ_min = (μ_min / 6) · (9 K_min + 8 μ_min) / (K_min + 2 μ_min)
        u     = φ / φ_c

        K_dry = (u/(K_HM + 4 μ_min/3) + (1−u)/(K_min + 4 μ_min/3))⁻¹ − 4 μ_min/3
        μ_dry = (u/(μ_HM + ζ_min)    + (1−u)/(μ_min + ζ_min))⁻¹    − ζ_min
    """

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
    ) -> None:
        super().__init__()
        _validate_positive_scalars(
            "StiffSand",
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
        )
        _validate_phi_lt_one("StiffSand", phi_critical)
        self.K_mineral = float(K_mineral)
        self.mu_mineral = float(mu_mineral)
        self.phi_c = float(phi_critical)
        # ζ at the stiff (mineral) endpoint: scalar, precompute.
        self._zeta_min = (
            (self.mu_mineral / 6.0)
            * (9.0 * self.K_mineral + 8.0 * self.mu_mineral)
            / (self.K_mineral + 2.0 * self.mu_mineral)
        )

    def forward(
        self,
        K_HM: Tensor,
        mu_HM: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return _stiff_sand_math(
            self.K_mineral,
            self.mu_mineral,
            K_HM,
            mu_HM,
            phi,
            self.phi_c,
            self._zeta_min,
        )

    def compute_dry_rock(self, K0, G0, phi, **params):
        """Workflow hook: varies all parameters at call time."""
        from ._types import CN as _CN, F as _F, P as _P, PHI_C as _PHI_C

        phi_c = params.get("phi_c", _PHI_C)
        K_HM, mu_HM = hm_moduli_v05(
            K0,
            G0,
            phi_c,
            params.get("Cn", _CN),
            params.get("P", _P),
            params.get("f", _F),
        )
        zeta_min = G0 / 6.0 * (9.0 * K0 + 8.0 * G0) / (K0 + 2.0 * G0 + EPS)
        return _stiff_sand_math(K0, G0, K_HM, mu_HM, phi, phi_c, zeta_min)


def _stiff_sand_math(
    K0,
    G0,
    K_HM,
    mu_HM,
    phi,
    phi_c,
    zeta_min,
) -> tuple[Tensor, Tensor]:
    """Compatibility view of the canonical HS-upper kernel."""
    del zeta_min
    result = stiff_sand_moduli(phi, K0, G0, K_HM, mu_HM, phi_c)
    return result.k_dry, result.mu_dry


# --- Walton (frictionless contact) ------------------------------------------


@register("Walton", aliases=["walton"])
class Walton(GranularModel):
    """
    Walton (1987) frictionless granular contact.

    K_W = K_HM; μ_W = (3/5) K_W. Returns ``(K_W, μ_W)`` in Pa.
    Hertz-Mindlin and Walton bracket the realistic friction range, HM
    (rough contacts) is the upper bound, Walton (smooth) the lower.
    """

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
        coordination_n: float = 9.0,
    ) -> None:
        super().__init__()
        _validate_positive_scalars(
            "Walton",
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
            coordination_n=coordination_n,
        )
        _validate_phi_lt_one("Walton", phi_critical)
        self.K_mineral = float(K_mineral)
        self.mu_mineral = float(mu_mineral)
        self.phi_c = float(phi_critical)
        self.n = float(coordination_n)
        self.nu = _mineral_poisson(self.K_mineral, self.mu_mineral)

    def forward(self, P_eff: Tensor) -> tuple[Tensor, Tensor]:
        result = walton_moduli(
            P_eff,
            self.K_mineral,
            self.mu_mineral,
            self.phi_c,
            self.n,
        )
        return result.k_dry, result.mu_dry


# ============================================================================
# Forward-time-signature model variants
# ============================================================================
#
# These all use the "everything-at-forward-time" signature where
# (phi_c, Cn, P, mineral moduli, cement moduli) flow through ``forward``.
# That differs from the T2a operator-friendly signature where those are
# constructor scalars; both styles coexist for different consumers.


def hm_moduli_v05(
    K0: Tensor,
    G0: Tensor,
    phi_c: Tensor | float,
    Cn: Tensor | float,
    P: Tensor | float,
    f: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """
    Hertz-Mindlin moduli with the forward-time signature (used internally by
    :class:`PCM`, :class:`VPCM`, and any T6 model that needs to vary the
    HM parameters at forward time).

    SI throughout: ``K0`` / ``G0`` and effective pressure ``P`` in
    **Pa**; returns ``(K_HM, G_HM)`` in **Pa**. The power law is scale-free
    so any single consistent unit works; Pa is the platform convention.
    """
    pressure = torch.as_tensor(P, dtype=K0.dtype, device=K0.device)
    result = hertz_mindlin_moduli(
        pressure,
        K0,
        G0,
        phi_c,
        Cn,
        f,
    )
    return result.k_dry, result.mu_dry


def _canonical_contact_cement(
    K0: Tensor,
    G0: Tensor,
    Kc: Tensor,
    Gc: Tensor,
    phi: Tensor,
    phi_c: Tensor | float,
    Cn: Tensor | float,
    scheme: int = 2,
) -> tuple[Tensor, Tensor]:
    """Compatibility adapter to the canonical full contact equation."""
    result = contact_cement_moduli(
        K0,
        G0,
        Kc,
        Gc,
        phi_c - phi,
        phi_c,
        Cn,
        scheme=scheme,
    )
    return result.k_dry, result.mu_dry


# --- Contact-cement family --------------------------------------------------


@register("ContactCement", aliases=["contact_cement", "cc"])
class ContactCement(GranularModel):
    """
    Canonical full contact-cement model (Dvorkin-Nur 1996).

    Args:
        scheme: 1 (cement at contacts only) or 2 (cement uniformly).
    """

    def __init__(self, scheme: int = 2) -> None:
        super().__init__()
        if type(scheme) is not int or scheme not in (1, 2):
            raise GeoBrainError(
                "ContactCement scheme must be 1 or 2",
                object_name="ContactCement",
                field="scheme",
                expected="1 or 2",
                actual=scheme,
            )
        self.scheme = int(scheme)

    def forward(
        self,
        K0: Tensor,
        G0: Tensor,
        Kc: Tensor,
        Gc: Tensor,
        phi: Tensor,
        phi_c: Tensor | float = 0.36,
        Cn: Tensor | float = 8.5,
    ) -> tuple[Tensor, Tensor]:
        return _canonical_contact_cement(K0, G0, Kc, Gc, phi, phi_c, Cn, self.scheme)

    def compute_dry_rock(self, K0, G0, phi, **params):
        """Workflow hook: cement moduli default to mineral when not given."""
        from ._types import CN as _CN, PHI_C as _PHI_C

        Kc = params.get("Kc", K0)
        Gc = params.get("Gc", G0)
        return _canonical_contact_cement(
            K0,
            G0,
            Kc,
            Gc,
            phi,
            params.get("phi_c", _PHI_C),
            params.get("Cn", _CN),
            self.scheme,
        )


@register("ContactCementFull", aliases=["contact_cement_full"])
class ContactCementFull(GranularModel):
    """Compatibility name for the canonical contact-cement equation."""

    def __init__(self, scheme: int = 2) -> None:
        super().__init__()
        if type(scheme) is not int or scheme not in (1, 2):
            raise GeoBrainError(
                "ContactCementFull scheme must be 1 or 2",
                object_name="ContactCementFull",
                field="scheme",
                expected="1 or 2",
                actual=scheme,
            )
        self.scheme = int(scheme)

    def forward(
        self,
        K0: Tensor,
        G0: Tensor,
        Kc: Tensor,
        Gc: Tensor,
        phi: Tensor,
        phi_c: Tensor | float = 0.36,
        Cn: Tensor | float = 8.5,
    ) -> tuple[Tensor, Tensor]:
        return _canonical_contact_cement(
            K0,
            G0,
            Kc,
            Gc,
            phi,
            phi_c,
            Cn,
            self.scheme,
        )


# --- HS-style cement mixers (MUHS / PCM) -----------------------------------


@register("MUHS", aliases=["IncreasingCement", "muhs"])
class MUHS(GranularModel):
    """
    Modified upper Hashin-Shtrikman bound for increasing cement.

    Mixes a contact-cement endpoint (at ``phi_i``) with the pure mineral
    (at φ = 0) along the HS upper bound. Returns ``(K_eff, μ_eff)`` in
    the same units as the inputs.
    """

    def __init__(self, scheme: int = 2) -> None:
        super().__init__()
        self._ccm = ContactCement(scheme=scheme)

    def forward(
        self,
        K0: Tensor,
        G0: Tensor,
        Kc: Tensor,
        Gc: Tensor,
        phi: Tensor,
        phi_i: Tensor,
        phi_c: Tensor | float = 0.36,
        Cn: Tensor | float = 8.5,
    ) -> tuple[Tensor, Tensor]:
        result = modified_upper_hashin_shtrikman(
            phi,
            K0,
            G0,
            Kc,
            Gc,
            phi_c - phi_i,
            phi_c,
            Cn,
            scheme=self._ccm.scheme,
        )
        return result.k_dry, result.mu_dry


@register("PCM", aliases=["PatchyCement", "pcm"])
class PCM(GranularModel):
    """Patchy Cement Model: mixes uncemented (HM) and cemented (CCM) patches."""

    def __init__(self, scheme: int = 2) -> None:
        super().__init__()
        self._ccm = ContactCement(scheme=scheme)

    def forward(
        self,
        K0: Tensor,
        G0: Tensor,
        Kc: Tensor,
        Gc: Tensor,
        phi: Tensor,
        phi_c: Tensor,
        v_cem: Tensor,
        f_cem: Tensor,
        Cn: Tensor | float = 8.5,
        P: Tensor | float = 20.0e6,
        f_hm: Tensor | float = 0.5,
        mode: str = "stiff",
    ) -> tuple[Tensor, Tensor]:
        result = patchy_cement_moduli(
            phi,
            K0,
            G0,
            Kc,
            Gc,
            phi_c,
            v_cem,
            f_cem,
            Cn,
            P,
            f_hm,
            scheme=self._ccm.scheme,
            mode=mode,
        )
        return result.k_dry, result.mu_dry


# --- Thomas-Stieber + shaly-sand family ------------------------------------


@register("ThomasStieber", aliases=["thomas_stieber"])
class ThomasStieber(GranularModel):
    """
    Thomas-Stieber sand-shale porosity bounds.

    Returns ``(phi_dispersed, phi_laminated)``, the dispersed and
    laminated-shale porosity endpoints for a given ``vsh``.
    """

    def forward(
        self,
        phi_sand: Tensor,
        phi_sh: Tensor,
        vsh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        phi_dirty = phi_sand - (1 - phi_sh) * vsh
        phi_structural = phi_sh * vsh
        phi_dispersed = torch.maximum(phi_dirty, phi_structural)
        phi_laminated = phi_sand + (phi_sh - phi_sand) * vsh
        return phi_dispersed, phi_laminated


@register("SiltyShale", aliases=["silty_shale"])
class SiltyShale(GranularModel):
    """Dvorkin-Gutierrez silty shale (HS lower bound with shale matrix)."""

    def forward(
        self,
        C: Tensor,
        Kq: Tensor,
        Gq: Tensor,
        Ksh: Tensor,
        Gsh: Tensor,
    ) -> tuple[Tensor, Tensor]:
        K_sat = (C / (Ksh + 4 * Gsh / 3) + (1 - C) / (Kq + 4 * Gsh / 3 + EPS)) ** (-1) - 4 * Gsh / 3
        Zsh = Gsh / 6 * (9 * Ksh + 8 * Gsh) / (Ksh + 2 * Gsh + EPS)
        G_sat = (C / (Gsh + Zsh) + (1 - C) / (Gq + Zsh + EPS)) ** (-1) - Zsh
        return K_sat, G_sat


@register("ShalySand", aliases=["shaly_sand"])
class ShalySand(GranularModel):
    """Shaly sand via HS lower bound (sand matrix carrying dispersed shale)."""

    def forward(
        self,
        phi_s: Tensor,
        C: Tensor,
        Kss: Tensor,
        Gss: Tensor,
        Kcc: Tensor,
        Gcc: Tensor,
    ) -> tuple[Tensor, Tensor]:
        K_sat = (
            (1 - C / phi_s) / (Kss + 4 * Gss / 3) + (C / phi_s) / (Kcc + 4 * Gss / 3 + EPS)
        ) ** (-1) - 4 * Gss / 3
        Zss = Gss / 6 * (9 * Kss + 8 * Gss) / (Kss + 2 * Gss + EPS)
        G_sat = ((1 - C / phi_s) / (Gss + Zss) + (C / phi_s) / (Gcc + Zss + EPS)) ** (-1) - Zss
        return K_sat, G_sat


# --- Digby bonded spheres + ConstantCement / VPCM / Diluting ---------------


@register("Digby", aliases=["digby"])
class Digby(GranularModel):
    """
    Digby 1981 bonded sphere pack (Newton iteration on cubic).

    Inputs ``a_R`` is the initial (zero-stress) bonded-contact radius
    relative to grain radius. Returns ``(K_eff, G_eff)`` in Pa given
    ``sigma`` (effective stress) in Pa (SI).
    """

    def forward(
        self,
        K0: Tensor,
        G0: Tensor,
        phi: Tensor,
        Cn: Tensor,
        sigma: Tensor,
        a_R: Tensor,
    ) -> tuple[Tensor, Tensor]:
        nu = (3.0 * K0 - 2.0 * G0) / (6.0 * K0 + 2.0 * G0 + EPS)
        rhs = 3.0 * PI * (1.0 - nu) * sigma / (2.0 * Cn * (1.0 - phi) * G0 + EPS)
        x = rhs ** (1.0 / 3.0)
        for _ in range(20):
            fx = x**3 + 1.5 * a_R**2 * x - rhs
            dfx = 3.0 * x**2 + 1.5 * a_R**2 + EPS
            x = (x - fx / dfx).clamp(min=EPS)
        b_R = torch.sqrt(x**2 + a_R**2)
        Sn_R = 4.0 * G0 * b_R / (1.0 - nu + EPS)
        St_R = 8.0 * G0 * a_R / (2.0 - nu + EPS)
        Keff = Cn * (1.0 - phi) * Sn_R / (12.0 * PI)
        Geff = Cn * (1.0 - phi) * (Sn_R + 1.5 * St_R) / (20.0 * PI)
        return Keff, Geff


@register("ConstantCement", aliases=["constant_cement"])
class ConstantCement(GranularModel):
    """Constant-cement model (Avseth et al. 2000)."""

    def forward(
        self,
        phi_b: Tensor,
        K0: Tensor,
        G0: Tensor,
        Kc: Tensor,
        Gc: Tensor,
        phi: Tensor,
        phi_c: Tensor,
        Cn: Tensor,
        scheme: int = 1,
    ) -> tuple[Tensor, Tensor]:
        result = constant_cement_moduli(
            phi,
            K0,
            G0,
            Kc,
            Gc,
            phi_c - phi_b,
            phi_c,
            Cn,
            scheme=scheme,
        )
        return result.k_dry, result.mu_dry


@register("VPCM", aliases=["varying_patchy_cement", "vpcm"])
class VPCM(GranularModel):
    """
    Varying Patchiness Cement Model (Yu et al. 2023).

    Stress-dependent patchy-cement model that interpolates between stiff
    and soft Hashin-Shtrikman branches via ``alpha_d``. Returns
    ``(K_dry, G_dry)``.
    """

    def forward(
        self,
        alpha_d: Tensor,
        f: Tensor,
        sigma: Tensor,
        K0: Tensor,
        G0: Tensor,
        phi: Tensor,
        phi_c: Tensor,
        v_cem: Tensor,
        v_ci: Tensor,
        Kc: Tensor,
        Gc: Tensor,
        Cn: Tensor,
        scheme: int = 1,
        f_hm: Tensor | float = 0.5,
    ) -> tuple[Tensor, Tensor]:
        result = varying_patchy_cement_moduli(
            alpha_d,
            phi,
            K0,
            G0,
            Kc,
            Gc,
            phi_c,
            v_cem,
            v_ci,
            f,
            Cn,
            sigma,
            f_hm,
            scheme=scheme,
        )
        return result.k_dry, result.mu_dry


@register("Diluting", aliases=["diluting"])
class Diluting(GranularModel):
    """
    Stress-dependent diluting parameter for :class:`VPCM`.

    Returns ``α_d = k · (1 − σ/σ₀)^m`` clamped to ≥ 0 (so post-yield
    stresses don't produce negative diluting).
    """

    def forward(
        self,
        k: Tensor,
        sigma0: Tensor,
        sigma: Tensor,
        m: Tensor,
    ) -> Tensor:
        ratio = (1.0 - sigma / (sigma0 + EPS)).clamp(min=0.0)
        return k * ratio**m


__all__ = [
    "ConstantCement",
    "ContactCement",
    "ContactCementFull",
    "Digby",
    "Diluting",
    "HertzMindlin",
    "HertzMindlinOperator",
    "MUHS",
    "PCM",
    "ShalySand",
    "SiltyShale",
    "SoftSand",
    "SoftSandOperator",
    "StiffSand",
    "StiffSandOperator",
    "ThomasStieber",
    "VPCM",
    "Walton",
    "WaltonOperator",
    "_SandBaseOperator",
]


# ============================================================================
# Operator wrappers (PropertyTransform)
#
# Merged from rock/granular.py.
# ============================================================================

# Private aliases
_HMModel = HertzMindlin
_SoftSandModel = SoftSand
_StiffSandModel = StiffSand
_WaltonModel = Walton


class HertzMindlinOperator(PropertyTransform):
    """Hertz-Mindlin dry-frame moduli at critical porosity (operator)."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("P_eff",),
        output_keys=("K_HM", "mu_HM"),
    )

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
        coordination_n: float = 9.0,
    ) -> None:
        super().__init__()
        self._model = _HMModel(
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
            coordination_n=coordination_n,
        )
        self.K_mineral = self._model.K_mineral
        self.mu_mineral = self._model.mu_mineral
        self.phi_c = self._model.phi_c
        self.n = self._model.n
        self.nu = self._model.nu

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (P_eff,) = state.fetch("P_eff")
        K_HM, mu_HM = self._model(P_eff)
        return state.with_tensors(K_HM=K_HM, mu_HM=mu_HM)


class _SandBaseOperator(PropertyTransform):
    """Shared shape-checking for the two modified-HS sand bounds."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("K_HM", "mu_HM", "phi"),
        output_keys=("K_dry", "mu_dry"),
    )

    def _check_shapes(self, K_HM, mu_HM, phi, op_name: str) -> None:
        for name, t in (("mu_HM", mu_HM), ("phi", phi)):
            if t.shape != K_HM.shape:
                raise GeoBrainError(
                    f"{op_name} inputs must share shape",
                    object_name=op_name,
                    field=name,
                    expected=tuple(K_HM.shape),
                    actual=tuple(t.shape),
                )


class SoftSandOperator(_SandBaseOperator):
    """Modified HS lower bound between HM-pack and pure mineral."""

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
    ) -> None:
        super().__init__()
        self._model = _SoftSandModel(
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
        )
        self.K_mineral = self._model.K_mineral
        self.mu_mineral = self._model.mu_mineral
        self.phi_c = self._model.phi_c

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K_HM, mu_HM, phi = state.fetch("K_HM", "mu_HM", "phi")
        self._check_shapes(K_HM, mu_HM, phi, "SoftSand")
        check_bounded(phi, "phi", owner="SoftSand", upper=self.phi_c)
        K_dry, mu_dry = self._model(K_HM, mu_HM, phi)
        return state.with_tensors(K_dry=K_dry, mu_dry=mu_dry)


class StiffSandOperator(_SandBaseOperator):
    """Modified HS upper bound between HM-pack and pure mineral."""

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
    ) -> None:
        super().__init__()
        self._model = _StiffSandModel(
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
        )
        self.K_mineral = self._model.K_mineral
        self.mu_mineral = self._model.mu_mineral
        self.phi_c = self._model.phi_c
        self._zeta_min = self._model._zeta_min

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        K_HM, mu_HM, phi = state.fetch("K_HM", "mu_HM", "phi")
        self._check_shapes(K_HM, mu_HM, phi, "StiffSand")
        check_bounded(phi, "phi", owner="StiffSand", upper=self.phi_c)
        K_dry, mu_dry = self._model(K_HM, mu_HM, phi)
        return state.with_tensors(K_dry=K_dry, mu_dry=mu_dry)


class WaltonOperator(PropertyTransform):
    """Walton 1987 frictionless granular contact (operator)."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("P_eff",),
        output_keys=("K_W", "mu_W"),
    )

    def __init__(
        self,
        *,
        K_mineral: float = 37.0e9,
        mu_mineral: float = 44.0e9,
        phi_critical: float = 0.40,
        coordination_n: float = 9.0,
    ) -> None:
        super().__init__()
        self._model = _WaltonModel(
            K_mineral=K_mineral,
            mu_mineral=mu_mineral,
            phi_critical=phi_critical,
            coordination_n=coordination_n,
        )
        self.K_mineral = self._model.K_mineral
        self.mu_mineral = self._model.mu_mineral
        self.phi_c = self._model.phi_c
        self.n = self._model.n
        self.nu = self._model.nu

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (P_eff,) = state.fetch("P_eff")
        K_W, mu_W = self._model(P_eff)
        return state.with_tensors(K_W=K_W, mu_W=mu_W)
