"""
Inclusion-based effective-medium models (DEM, KT, SC, Eshelby-Cheng,
dilute crack/pore, Berryman P/Q).

Private submodule of :mod:`geobrain.physics.rock.models.effective`.
Public symbols are re-exported from ``effective.py``.

Models:
    DEM:            Differential Effective Medium (Berryman 1992 ODE)
    KusterToksoz:   Sparse spherical-inclusion closed form
    SelfConsistent: Berryman 1980 self-consistent EMT
    XuWhite:        Two-pore-shape clastic rock model
    EshelbyCheng:   Cracked-isotropic VTI stiffness
    SwissCheese:    Dilute spherical pores
    DiluteCrack:    Walsh (1965) dilute random cracks
    SCDilute:       Dilute self-consistent two-phase
    SCFlex:         Iterative SC two-phase
    PQ:             Berryman ellipsoidal P, Q factors

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..inclusions import (
    differential_effective_medium,
    dilute_crack_moduli,
    ellipsoidal_inclusion_factors,
    kuster_toksoz_moduli,
    require_converged,
    sc_flex_moduli,
    self_consistent_dilute_moduli,
    self_consistent_moduli,
    swiss_cheese_moduli,
    xu_white_moduli,
)
from ....core import GeoBrainError
from ._categories import EffectiveModel
from ._registry import register
from ._types import EPS
from ..errors import RockNumericsError


# --- Differential Effective Medium ------------------------------------------


@register("DEM", aliases=["dem", "differential_effective_medium"])
class DEM(EffectiveModel):
    """
    Differential Effective Medium (Berryman 1992 ODE for spherical inclusions).

    Returns ``(K_eff, mu_eff)`` in Pa.
    """

    def __init__(
        self,
        *,
        K_host: float = 37.0e9,
        mu_host: float = 44.0e9,
        K_inc: float = 2.25e9,
        mu_inc: float = 0.0,
        n_steps: int = 50,
        tolerance: float = 1.0e-3,
    ) -> None:
        super().__init__()
        for name, value in (("K_host", K_host), ("mu_host", mu_host), ("K_inc", K_inc)):
            if value <= 0:
                raise GeoBrainError(
                    f"DEM {name} must be positive",
                    object_name="DEM",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        if mu_inc < 0:
            raise GeoBrainError(
                "DEM mu_inc must be non-negative (use 0 for fluid)",
                object_name="DEM",
                field="mu_inc",
                expected=">= 0",
                actual=mu_inc,
            )
        if n_steps < 1:
            raise GeoBrainError(
                "DEM n_steps must be >= 1",
                object_name="DEM",
                field="n_steps",
                expected=">= 1",
                actual=n_steps,
            )
        self.K_host = float(K_host)
        self.mu_host = float(mu_host)
        self.K_inc = float(K_inc)
        self.mu_inc = float(mu_inc)
        self.n_steps = int(n_steps)
        self.tolerance = float(tolerance)

    def forward(self, phi: Tensor) -> tuple[Tensor, Tensor]:
        result = differential_effective_medium(
            phi.new_tensor(self.K_host),
            self.mu_host,
            self.K_inc,
            self.mu_inc,
            phi,
            steps=self.n_steps,
            tolerance=self.tolerance,
        )
        require_converged(result, object_name="DEM")
        return result.k_eff, result.mu_eff

    def compute_dry_rock(self, K0, G0, phi, **params):
        """Workflow hook: DEM with dry pores (K_inc=G_inc=0) starting from (K0, G0)."""
        n_steps = params.get("n_steps", self.n_steps)
        tolerance = params.get("tolerance", self.tolerance)
        result = differential_effective_medium(
            torch.ones_like(phi) * K0,
            torch.ones_like(phi) * G0,
            0.0,
            0.0,
            phi,
            steps=n_steps,
            tolerance=tolerance,
        )
        require_converged(result, object_name="DEM.compute_dry_rock")
        return result.k_eff, result.mu_eff


# --- Kuster-Toksöz ----------------------------------------------------------


@register("KusterToksoz", aliases=["kuster_toksoz", "kt"])
class KusterToksoz(EffectiveModel):
    """
    Kuster-Toksöz 1974 closed-form for sparse spherical inclusions.

    Returns ``(K_KT, mu_KT)`` in Pa. Strictly dilute (φ ≲ 0.2) for
    physical accuracy.
    """

    def __init__(
        self,
        *,
        K_inclusion: float = 2.25e9,
        mu_inclusion: float = 0.0,
    ) -> None:
        super().__init__()
        if K_inclusion <= 0:
            raise GeoBrainError(
                "KusterToksoz K_inclusion must be positive",
                object_name="KusterToksoz",
                field="K_inclusion",
                expected="> 0",
                actual=K_inclusion,
            )
        if mu_inclusion < 0:
            raise GeoBrainError(
                "KusterToksoz mu_inclusion must be non-negative",
                object_name="KusterToksoz",
                field="mu_inclusion",
                expected=">= 0",
                actual=mu_inclusion,
            )
        self.K_inclusion = float(K_inclusion)
        self.mu_inclusion = float(mu_inclusion)

    def forward(
        self,
        K_m: Tensor,
        mu_m: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return kuster_toksoz_moduli(
            K_m,
            mu_m,
            self.K_inclusion,
            self.mu_inclusion,
            phi,
        )


# --- Self-Consistent --------------------------------------------------------


@register("SelfConsistent", aliases=["sc", "self_consistent"])
class SelfConsistent(EffectiveModel):
    """
    Berryman 1980 self-consistent EMT for 2-phase spherical inclusions.

    Fixed-point iteration starting from a finite mixed-phase estimate. Returns
    ``(K_SC, mu_SC)`` in Pa and rejects non-convergence rather than silently
    returning the final iterate.
    """

    def __init__(self, *, n_iter: int = 60, tolerance: float = 1.0e-3) -> None:
        super().__init__()
        if n_iter < 1:
            raise GeoBrainError(
                "SelfConsistent n_iter must be >= 1",
                object_name="SelfConsistent",
                field="n_iter",
                expected=">= 1",
                actual=n_iter,
            )
        self.n_iter = int(n_iter)
        self.tolerance = float(tolerance)

    def forward(
        self,
        K1: Tensor,
        mu1: Tensor,
        K2: Tensor,
        mu2: Tensor,
        f1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = self_consistent_moduli(
            K1,
            mu1,
            K2,
            mu2,
            f1,
            tolerance=self.tolerance,
            max_iterations=self.n_iter,
        )
        require_converged(result, object_name="SelfConsistent")
        return result.k_eff, result.mu_eff


@register("XuWhite", aliases=["xu_white"])
class XuWhite(EffectiveModel):
    """
    Xu-White 1995 two-pore-shape clastic rock model.

    Mineral mix via V-R-H + Kuster-Toksöz with oblate-spheroid pores
    (sand-shape family + clay-shape family), added in ``n_steps`` Euler
    substeps to keep KT in the dilute regime.

    Returns ``(K_dry, mu_dry, rho_dry)`` in Pa, Pa, kg/m³.
    """

    def __init__(
        self,
        *,
        K_quartz: float = 37.0e9,
        mu_quartz: float = 44.0e9,
        rho_quartz: float = 2650.0,
        K_clay: float = 21.0e9,
        mu_clay: float = 7.0e9,
        rho_clay: float = 2580.0,
        alpha_sand: float = 0.12,
        alpha_clay: float = 0.03,
        n_steps: int = 20,
        tolerance: float = 5.0e-2,
    ) -> None:
        super().__init__()
        for name, value in (
            ("K_quartz", K_quartz),
            ("mu_quartz", mu_quartz),
            ("rho_quartz", rho_quartz),
            ("K_clay", K_clay),
            ("mu_clay", mu_clay),
            ("rho_clay", rho_clay),
        ):
            if value <= 0:
                raise GeoBrainError(
                    f"XuWhite {name} must be positive",
                    object_name="XuWhite",
                    field=name,
                    expected="> 0",
                    actual=value,
                )
        for name, value in (("alpha_sand", alpha_sand), ("alpha_clay", alpha_clay)):
            if not 0.0 < value < 1.0:
                raise GeoBrainError(
                    f"XuWhite {name} must lie strictly in (0, 1)",
                    object_name="XuWhite",
                    field=name,
                    expected="(0, 1)",
                    actual=value,
                )
        if n_steps < 1:
            raise GeoBrainError(
                "XuWhite n_steps must be >= 1",
                object_name="XuWhite",
                field="n_steps",
                expected=">= 1",
                actual=n_steps,
            )
        self.K_quartz = float(K_quartz)
        self.mu_quartz = float(mu_quartz)
        self.rho_quartz = float(rho_quartz)
        self.K_clay = float(K_clay)
        self.mu_clay = float(mu_clay)
        self.rho_clay = float(rho_clay)
        self.alpha_sand = float(alpha_sand)
        self.alpha_clay = float(alpha_clay)
        self.n_steps = int(n_steps)
        self.tolerance = float(tolerance)

    def forward(self, V_sh: Tensor, phi: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        result = xu_white_moduli(
            V_sh,
            phi,
            k_quartz=self.K_quartz,
            mu_quartz=self.mu_quartz,
            rho_quartz=self.rho_quartz,
            k_clay=self.K_clay,
            mu_clay=self.mu_clay,
            rho_clay=self.rho_clay,
            sand_aspect_ratio=self.alpha_sand,
            clay_aspect_ratio=self.alpha_clay,
            steps=self.n_steps,
            tolerance=self.tolerance,
        )
        if not bool(result.iteration.converged.all()):
            raise RockNumericsError(
                "XuWhite step refinement did not converge for every element",
                object_name="XuWhite",
                field="convergence",
                expected=f"residual <= {result.iteration.tolerance}",
                actual=result.iteration.to_dict(),
                hint="increase n_steps or relax the declared tolerance",
            )
        return result.k_dry, result.mu_dry, result.rho_dry


# --- Eshelby-Cheng oblate-spheroid VTI -------------------------------------


@register("EshelbyCheng", aliases=["eshelby_cheng"])
class EshelbyCheng(EffectiveModel):
    """
    Eshelby-Cheng (Cheng 1993) cracked-isotropic-rock VTI stiffness.

    Penny-shaped oblate-spheroid cracks of aspect ratio α in an
    isotropic background ``(K, G)`` with optional fluid bulk modulus
    ``Kf``. Returns 6×6 VTI stiffness.
    """

    def forward(
        self,
        K: Tensor,
        G: Tensor,
        phi: Tensor,
        alpha: Tensor,
        Kf: Tensor | float = 0.0,
    ) -> Tensor:
        from ..moduli import write_vti_matrix
        from ._types import PI

        # Spherical-limit guard: as alpha -> 1 the crack shape integrals
        # diverge (Sa = sqrt(1 - alpha^2) -> 0, and Sa^3 sits in every
        # denominator), so the stiffness blows up to nonsense (C11 ~ -3.6e24
        # at alpha == 1). Bound alpha away from the removable singularity.
        alpha_t = torch.as_tensor(alpha)
        if bool((torch.abs(alpha_t - 1.0) < 1e-2).any()):
            raise GeoBrainError(
                "EshelbyCheng: aspect ratio alpha must be bounded away from the "
                "spherical limit (require |alpha - 1| >= 1e-2); the penny-crack "
                "shape integrals diverge as alpha -> 1 (Sa^3 -> 0).",
                object_name="EshelbyCheng",
                field="alpha",
                expected="|alpha - 1| >= 1e-2",
                actual=f"[{float(alpha_t.min()):.4g}, {float(alpha_t.max()):.4g}]",
            )

        lamda = K - 2.0 * G / 3.0
        sigma = (3.0 * K - 2.0 * G) / (6.0 * K + 2.0 * G + EPS)
        R = (1.0 - 2.0 * sigma) / (8.0 * PI * (1.0 - sigma) + EPS)
        Q = 3.0 * R / (1.0 - 2.0 * sigma + EPS)

        Sa = torch.sqrt(1.0 - alpha**2 + EPS)
        Ia = 2.0 * PI * alpha * (torch.acos(alpha) - alpha * Sa) / (Sa**3 + EPS)
        Ic = 4.0 * PI - 2.0 * Ia
        Iac = (Ic - Ia) / (3.0 * Sa**2 + EPS)
        Iaa = PI - 3.0 * Iac / 4.0
        Iab = Iaa / 3.0

        S11 = Q * Iaa + R * Ia
        S33 = Q * (4.0 * PI / 3.0 - 2.0 * Iac * alpha**2) + Ic * R
        S12 = Q * Iab - R * Ia
        S13 = Q * Iac * alpha**2 - R * Ia
        S31 = Q * Iac - R * Ic
        S1212 = Q * Iab + R * Ia
        S1313 = Q * (1.0 + alpha**2) * Iac / 2.0 + R * (Ia + Ic) / 2.0

        C_factor = Kf / (3.0 * (K - Kf) + EPS)
        D = (
            S33 * S11
            + S33 * S12
            - 2.0 * S31 * S13
            - (S11 + S12 + S33 - 1.0 - 3.0 * C_factor)
            - C_factor * (S11 + S12 + 2.0 * (S33 - S13 - S31))
        )
        E = (
            S33 * S11
            - S31 * S13
            - (S33 + S11 - 2.0 * C_factor - 1.0)
            + C_factor * (S31 + S13 - S11 - S33)
        )

        C11_1 = lamda * (S31 - S33 + 1.0) + 2.0 * G * E / (D * (S12 - S11 + 1.0) + EPS)
        C33_1 = (
            (lamda + 2.0 * G) * (-S12 - S11 + 1.0) + 2.0 * lamda * S13 + 4.0 * G * C_factor
        ) / (D + EPS)
        C13_1 = (
            (lamda + 2.0 * G) * (S13 + S31)
            - 4.0 * G * C_factor
            + lamda * (S13 - S12 - S11 - S33 + 2.0)
        ) / (2.0 * D + EPS)
        C44_1 = G / (1.0 - 2.0 * S1313 + EPS)
        C66_1 = G / (1.0 - 2.0 * S1212 + EPS)

        C11 = lamda + 2.0 * G - phi * C11_1
        C33 = lamda + 2.0 * G - phi * C33_1
        C13 = lamda - phi * C13_1
        C44 = G - phi * (G - C44_1)
        C66 = G - phi * (G - C66_1)
        return write_vti_matrix(C11, C33, C13, C44, C66)


# --- Dilute pore / crack models --------------------------------------------


@register("SwissCheese", aliases=["swiss_cheese"])
class SwissCheese(EffectiveModel):
    """
    Dilute spherical pores in a homogeneous isotropic solid.

    ``K_dry / K_s = 1 / (1 + (1 + 3K_s/(4G_s))·φ)``
    """

    def forward(
        self,
        Ks: Tensor,
        Gs: Tensor,
        phi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return swiss_cheese_moduli(Ks, Gs, phi)


@register("DiluteCrack", aliases=["dilute_crack", "walsh"])
class DiluteCrack(EffectiveModel):
    """
    Walsh (1965) non-interacting random crack model.

    ``cd`` is the crack density ``N·a³/V``.
    """

    def forward(
        self,
        Ks: Tensor,
        Gs: Tensor,
        cd: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return dilute_crack_moduli(Ks, Gs, cd)


# --- Self-consistent extensions --------------------------------------------


@register("SCDilute", aliases=["sc_dilute"])
class SCDilute(EffectiveModel):
    """
    Dilute spherical micro-inclusions in a two-phase composite.

    ``mode='stress'`` (default): expansion in inverse moduli (gives
    matrix-form). ``mode='strain'`` gives the complementary expansion.
    """

    def __init__(self, mode: str = "stress") -> None:
        super().__init__()
        if mode not in ("stress", "strain"):
            raise GeoBrainError(
                "SCDilute mode must be 'stress' or 'strain'",
                object_name="SCDilute",
                field="mode",
                expected="stress|strain",
                actual=mode,
            )
        self.mode = mode

    def forward(
        self,
        Km: Tensor,
        Gm: Tensor,
        Ki: Tensor,
        Gi: Tensor,
        f: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self_consistent_dilute_moduli(
            Km,
            Gm,
            Ki,
            Gi,
            f,
            mode=self.mode,
        )


@register("SCFlex", aliases=["sc_flex"])
class SCFlex(EffectiveModel):
    """Iterative self-consistent two-phase composite (flexible inclusion shape)."""

    def __init__(self, max_iter: int = 50, tol: float = 1.0e-6) -> None:
        super().__init__()
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def forward(
        self,
        f: Tensor,
        Km: Tensor,
        Ki: Tensor,
        Gm: Tensor,
        Gi: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = sc_flex_moduli(
            Km,
            Gm,
            Ki,
            Gi,
            f,
            tolerance=self.tol,
            max_iterations=self.max_iter,
        )
        require_converged(result, object_name="SCFlex")
        return result.k_eff, result.mu_eff


# --- Berryman P, Q strain-concentration factors ----------------------------


@register("PQ", aliases=["pq", "berryman_pq"])
class PQ(EffectiveModel):
    """
    Berryman (1980) P, Q strain-concentration factors for ellipsoidal inclusions.

    Returns ``(P, Q)`` for the shape specified by aspect ratio ``alpha``
    (alpha < 1 = oblate, alpha > 1 = prolate, alpha ≈ 1 = sphere). The
    sphere branch is handled specially because the general formula has
    a coordinate singularity at α = 1.
    """

    def forward(
        self,
        Km: Tensor,
        Gm: Tensor,
        Ki: Tensor,
        Gi: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, Tensor]:
        factors = ellipsoidal_inclusion_factors(Km, Gm, Ki, Gi, alpha)
        return factors.p, factors.q


__all__ = [
    "DEM",
    "KusterToksoz",
    "SelfConsistent",
    "XuWhite",
    "EshelbyCheng",
    "SwissCheese",
    "DiluteCrack",
    "SCDilute",
    "SCFlex",
    "PQ",
]
