"""
Generalized two-parameter cubic equation of state (Peng-Robinson / SRK).

A differentiable PyTorch implementation of the generalized cubic EOS.
The generalized cubic is

    P = RT/(v − b) − a / [(v + m₁·b)(v + m₂·b)]

with ``(m₁, m₂)`` selecting the family, Peng-Robinson ``(1+√2, 1−√2)`` or
Soave-Redlich-Kwong ``(0, 1)``. In reduced form per component ``i``::

    A_i = ω_a · α_i(T) · p_r,i / T_r,i² ,   B_i = ω_b · p_r,i / T_r,i
    α_i = [1 + m(ω_i)·(1 − √T_r,i)]²        (m: PR or SRK acentric polynomial)
    p_r,i = p/p_c,i ,   T_r,i = T/T_c,i

van-der-Waals mixing with binary interaction k_ij::

    A_ij = √(A_i·A_j)·(1 − k_ij) ,   A = ΣΣ x_i x_j A_ij ,   B = Σ x_i B_i

The compressibility factor solves the monic cubic ``Z³ + aZ² + bZ + c = 0`` with

    a = (m₁+m₂−1)B − 1
    b = A − (m₁+m₂−m₁m₂)B² − (m₁+m₂)B
    c = −[A·B + m₁m₂·B²·(B+1)]

(liquid = smallest, vapor = largest real root), and the fugacity coefficient is

    ln φ_i = −ln(Z−B) + (B_i/B)(Z−1)
             + A/[(m₁−m₂)B] · ln[(Z+m₂B)/(Z+m₁B)] · (2·A_s,i/A − B_i/B),
    A_s,i = Σ_j A_ij·x_j .

Units are SI (Pa, K).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ....core import GeoBrainError
from .mixture import Mixture

# (ω_a, ω_b, m₁, m₂) and the acentric α-polynomial coefficients, per family.
_FAMILIES = {
    "PR": dict(
        omega_a=0.457235529,
        omega_b=0.077796074,
        m1=1.0 + math.sqrt(2.0),
        m2=1.0 - math.sqrt(2.0),
        c0=0.37464,
        c1=1.54226,
        c2=0.26992,
    ),
    "SRK": dict(
        omega_a=0.4274802327, omega_b=0.08664035, m1=0.0, m2=1.0, c0=0.48, c1=1.574, c2=0.176
    ),
}


class CubicEOS(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """Peng-Robinson / SRK cubic EOS over a :class:`Mixture` (SI units).

    Args:
        mixture: the :class:`Mixture` (critical properties + BIP).
        kind:    ``"PR"`` (Peng-Robinson) or ``"SRK"`` (Soave-Redlich-Kwong).
    """

    def __init__(self, mixture: Mixture, kind: str = "PR") -> None:
        super().__init__()
        if kind not in _FAMILIES:
            raise GeoBrainError(f"kind must be 'PR' or 'SRK', got {kind!r}")
        self.mixture = mixture
        self.kind = kind
        f = _FAMILIES[kind]
        self.omega_a = f["omega_a"]
        self.omega_b = f["omega_b"]
        self.m1 = f["m1"]
        self.m2 = f["m2"]
        self._c0, self._c1, self._c2 = f["c0"], f["c1"], f["c2"]

    @property
    def n_components(self) -> int:
        return self.mixture.n_components

    # ------------------------------------------------------------------
    # Component A_i, B_i  (reduced, dimensionless)
    # ------------------------------------------------------------------
    def ab_components(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(A_i, B_i)`` per component, shape ``(..., n_components)``."""
        mix = self.mixture
        p, T = mix.validate_state_tensors(p, T, object_name="CubicEOS.ab_components")
        p_r = p.unsqueeze(-1) / mix.critical_pressure_pa
        T_r = T.unsqueeze(-1) / mix.critical_temperature_k
        omega = mix.acentric_factor
        m = self._c0 + self._c1 * omega - self._c2 * omega * omega
        alpha_half = 1.0 + m * (1.0 - T_r.sqrt())
        alpha = alpha_half * alpha_half
        A_i = self.omega_a * alpha * p_r / (T_r * T_r)
        B_i = self.omega_b * p_r / T_r
        return A_i, B_i

    def mixture_ab(
        self,
        comp: torch.Tensor,
        A_i: torch.Tensor,
        B_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """van-der-Waals mixing → ``(A, B, A_ij)``. ``comp`` is the phase mole
        fraction ``(..., n_components)``; ``A_ij`` is the attractive matrix."""
        one_minus_k = 1.0 - self.mixture.bip
        A_ij = (A_i.unsqueeze(-1) * A_i.unsqueeze(-2)).sqrt() * one_minus_k
        A = (comp.unsqueeze(-1) * comp.unsqueeze(-2) * A_ij).sum(dim=(-1, -2))
        B = (comp * B_i).sum(dim=-1)
        return A, B, A_ij

    # ------------------------------------------------------------------
    # Compressibility factor Z (root of the cubic), differentiable
    # ------------------------------------------------------------------
    def _cubic_coeffs(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m1, m2 = self.m1, self.m2
        a = (m1 + m2 - 1.0) * B - 1.0
        b = A - (m1 + m2 - m1 * m2) * B * B - (m1 + m2) * B
        c = -(A * B + m1 * m2 * B * B * (B + 1.0))
        return a, b, c

    def compressibility(
        self, A: torch.Tensor, B: torch.Tensor, root: str = "vapor"
    ) -> torch.Tensor:
        """Real root ``Z`` of ``Z³+aZ²+bZ+c``: ``root='vapor'`` (largest) or
        ``'liquid'`` (smallest). Differentiable in ``A, B`` via an
        implicit-function-theorem cleanup over a no-grad companion-eigvals root."""
        if root not in ("vapor", "liquid"):
            raise GeoBrainError(f"root must be 'vapor' or 'liquid', got {root!r}")
        a, b, c = self._cubic_coeffs(A, B)
        want_max = root == "vapor"
        with torch.no_grad():
            shape = a.shape
            comp = a.new_zeros(*shape, 3, 3)
            comp[..., 0, 0] = -a
            comp[..., 0, 1] = -b
            comp[..., 0, 2] = -c
            comp[..., 1, 0] = 1.0
            comp[..., 2, 1] = 1.0
            # A non-finite companion entry makes LAPACK's DGEBAL balancing step
            # segfault the whole process; sanitise before the eigensolve. Finite
            # garbage roots for such cells are handled by the Z>B guard below and
            # rejected downstream (flash convergence / stability mask).
            comp = torch.nan_to_num(comp)
            roots = torch.linalg.eigvals(comp)  # (..., 3) complex
            real = roots.real
            is_real = roots.imag.abs() <= 1e-9 * (1.0 + real.abs())
            fill = real.new_full(real.shape, -1e30 if want_max else 1e30)
            cand = torch.where(is_real, real, fill)
            Z0 = cand.amax(dim=-1) if want_max else cand.amin(dim=-1)
            # guard: Z must exceed B (positive v − b); clamp degenerate picks.
            Z0 = torch.maximum(Z0, B + 1e-8)
        # one differentiable Newton step from the (≈exact) root
        f = ((Z0 + a) * Z0 + b) * Z0 + c
        df = (3.0 * Z0 + 2.0 * a) * Z0 + b
        return Z0 - f / df

    # ------------------------------------------------------------------
    # Fugacity coefficient
    # ------------------------------------------------------------------
    def ln_phi(
        self,
        comp: torch.Tensor,
        A_i: torch.Tensor,
        B_i: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        A_ij: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """``ln φ_i`` for every component (the equilibrium driver of the flash)."""
        Zc = Z.unsqueeze(-1)
        Bc = B.unsqueeze(-1)
        Ac = A.unsqueeze(-1)
        A_s = (A_ij * comp.unsqueeze(-2)).sum(dim=-1)  # Σ_j A_ij x_j
        term1 = -torch.log(Z - B).unsqueeze(-1) + (B_i / Bc) * (Zc - 1.0)
        log_ratio = torch.log((Zc + self.m2 * Bc) / (Zc + self.m1 * Bc))
        term2 = Ac / ((self.m1 - self.m2) * Bc) * log_ratio * (2.0 * A_s / Ac - B_i / Bc)
        return term1 + term2

    def phase(
        self,
        comp: torch.Tensor,
        p: torch.Tensor,
        T: torch.Tensor,
        root: str = "vapor",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convenience: composition + (p, T) → ``(Z, ln φ_i)`` for one phase."""
        comp = self.mixture.validate_composition_tensor(
            comp,
            object_name="CubicEOS.phase",
        )
        A_i, B_i = self.ab_components(p, T)
        A, B, A_ij = self.mixture_ab(comp, A_i, B_i)
        Z = self.compressibility(A, B, root=root)
        return Z, self.ln_phi(comp, A_i, B_i, A, B, A_ij, Z)


__all__ = ["CubicEOS"]
