"""Viscoelastic wave equation: first-order velocity-stress with a single
generalized standard-linear-solid (GSLS) relaxation mechanism, separate P- and
S-wave attenuation (``Qp``, ``Qs``).

Extends :class:`ElasticVelocityStress` with three stress memory variables
(``r_xx``, ``r_zz``, ``r_xz``, co-located with the stresses they relax) that
introduce frequency-dependent attenuation and dispersion controlled by the
quality factors ``Qp`` (dilatational) and ``Qs`` (shear). The discretization is
the Robertsson–Blanch–Symes (1994) viscoelastic scheme as implemented in SOFI2D
(Bohlen 2002): the τ-method places one relaxation mechanism at the reference
frequency ``ω₀ = 2π·ref_freq`` (``τ_σ = 1/ω₀``), so for a single mechanism

    τ_p = 2 / (Qp − 1) ,   τ_s = 2 / (Qs − 1)       (level reproducing Q at ω₀)

The *relaxed* moduli (reduced so the phase velocity at ω₀ equals the input
``vp``/``vs``) and the *unrelaxed* moduli that drive the instantaneous response::

    π = ρ vp² / (1 + τ_p/2) ,   μ = ρ vs² / (1 + τ_s/2)     (relaxed)
    π_u = π (1 + τ_p) ,         μ_u = μ (1 + τ_s)            (unrelaxed)

With strains ``ex = ∂̃vx/∂x``, ``ez = ∂̃vz/∂z``, ``γ = ∂̃vx/∂z + ∂̃vz/∂x`` (all
CPML-stretched), ``η = Δt ω₀``, ``b = 1/(1+η/2)``, ``c = 1−η/2``, the memory
variables update by the trapezoidal rule and feed back into the stresses::

    r_xx ← b ( r_xx c − π τ_p η (ex+ez) + 2 μ τ_s η ez )
    r_zz ← b ( r_zz c − π τ_p η (ex+ez) + 2 μ τ_s η ex )
    r_xz ← b ( r_xz c − μ τ_s η γ )
    sxx += Δt ( π_u (ex+ez) − 2 μ_u ez ) + (Δt/2)(r_xx_old + r_xx_new)
    szz += Δt ( π_u (ex+ez) − 2 μ_u ex ) + (Δt/2)(r_zz_old + r_zz_new)
    sxz += Δt   μ_u γ                    + (Δt/2)(r_xz_old + r_xz_new)

Lossless limit ``Qp, Qs → ∞`` gives ``τ_p, τ_s → 0`` ⇒ ``r → 0``,
``π_u → ρvp²``, ``μ_u → ρvs²``, recovering :class:`ElasticVelocityStress`
exactly. Differentiable w.r.t. ``vp, vs, rho, Qp, Qs`` (Q-FWI is first-class).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from geobrain.core import GeoBrainError

import math
from typing import Callable, Mapping, Sequence

import torch
from torch import Tensor

from ..boundaries.cpml import CPML
from ..fd.derivative import (
    diff_x_backward,
    diff_x_forward,
    diff_z_backward,
    diff_z_forward,
)
from .base import FieldSpec, ModelSpec
from .elastic import ElasticVelocityStress


class ViscoElasticVelocityStress(ElasticVelocityStress):
    """Constant-Q viscoelastic wave equation (single GSLS mechanism).

    Models: ``vp``, ``vs`` (m/s, phase velocities at ``ref_freq``), ``rho``
    (kg/m³), ``Qp``, ``Qs`` (P- and S-wave quality factors, dimensionless).
    The reference frequency is fixed at construction; the mechanism reproduces
    ``Qp``/``Qs`` at that frequency. ``Qp, Qs → ∞`` recovers the elastic operator.
    """

    # Elastic wavefields + three SLS stress memory variables (real wavefields,
    # NOT CPML memory vars: carried in the state, reset to zero each shot) +
    # the eight elastic CPML memory variables (inherited order).
    FIELD_SPECS = (
        FieldSpec("vx", "x"),
        FieldSpec("vz", "z"),
        FieldSpec("sxx", "00"),
        FieldSpec("szz", "00"),
        FieldSpec("sxz", "x"),
        FieldSpec("r_xx", "00"),
        FieldSpec("r_zz", "00"),
        FieldSpec("r_xz", "x"),
        FieldSpec("psi_sxx_x", "x", is_memory=True),
        FieldSpec("psi_sxz_z", "00", is_memory=True),
        FieldSpec("psi_sxz_x", "00", is_memory=True),
        FieldSpec("psi_szz_z", "z", is_memory=True),
        FieldSpec("psi_vx_x", "00", is_memory=True),
        FieldSpec("psi_vz_z", "00", is_memory=True),
        FieldSpec("psi_vx_z", "z", is_memory=True),
        FieldSpec("psi_vz_x", "x", is_memory=True),
    )
    MODEL_SPECS = (
        ModelSpec("vp"), ModelSpec("vs"), ModelSpec("rho"),
        ModelSpec("Qp"), ModelSpec("Qs"),
    )

    def __init__(self, fd_order: int = 8, ref_freq: float = 15.0) -> None:
        super().__init__(fd_order)
        if ref_freq <= 0.0:
            raise GeoBrainError(f"ref_freq must be positive, got {ref_freq}")
        self.ref_freq = float(ref_freq)
        self.omega0 = 2.0 * math.pi * self.ref_freq

    def prepare(
        self, models: Mapping[str, Tensor], dt: float, dx: float, dz: float
    ) -> Mapping[str, Tensor]:
        """Relaxed/unrelaxed moduli + memory coefficients (single mechanism).

        Single GSLS mechanism at the reference frequency ``ω₀`` (relaxation time
        ``τ_σ = 1/ω₀``): the τ-level reproducing ``Q`` at ``ω₀`` is ``τ = 2/(Q−1)``
        and the modulus defect is ``1 + τ/2`` (the L=1, FL=ref_freq case of the
        SOFI2D constant-Q τ-method, where the general weights ``A = Σ aₗ²/(1+aₗ²)``,
        ``B = Σ aₗ/(1+aₗ²)`` collapse to ``A = B = 1/2``).
        """
        vp, vs, rho = models["vp"], models["vs"], models["rho"]
        Qp, Qs = models["Qp"], models["Qs"]
        tau_p = 2.0 / (Qp - 1.0)
        tau_s = 2.0 / (Qs - 1.0)
        pi_relaxed = rho * vp * vp / (1.0 + 0.5 * tau_p)   # phase velocity at ω₀ == vp
        mu_relaxed = rho * vs * vs / (1.0 + 0.5 * tau_s)
        eta = dt * self.omega0                              # η = Δt/τ_σ
        pi_u = pi_relaxed * (1.0 + tau_p)                   # unrelaxed (instantaneous) moduli
        mu_u = mu_relaxed * (1.0 + tau_s)
        return {
            "c11": pi_u,
            "c13": pi_u - 2.0 * mu_u,
            "c33": pi_u,
            "c44": mu_u,
            "buoyancy": 1.0 / rho,
            "e_p": pi_relaxed * tau_p * eta,                # memory drive: P modulus
            "d_s": mu_relaxed * tau_s * eta,                # memory drive: S modulus
            "mem_b": torch.as_tensor(1.0 / (1.0 + 0.5 * eta), dtype=vp.dtype, device=vp.device),
            "mem_c": torch.as_tensor(1.0 - 0.5 * eta, dtype=vp.dtype, device=vp.device),
        }

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        """Unrelaxed (fastest) P velocity drives the CFL limit.

        ``vp_unrelaxed = vp · sqrt((1+τ_p)/(1+τ_p/2)) ≥ vp``: the instantaneous
        response is faster than the reference-frequency phase velocity.
        """
        vp, Qp = models["vp"], models["Qp"]
        tau_p = 2.0 / (Qp - 1.0)
        fac = torch.sqrt((1.0 + tau_p) / (1.0 + 0.5 * tau_p))
        return float((vp * fac).detach().max())

    def step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        cpml: CPML,
        dt: float,
        dx: float,
        dz: float,
    ) -> tuple[Tensor, ...]:
        (vx, vz, sxx, szz, sxz, r_xx, r_zz, r_xz,
         psi_sxx_x, psi_sxz_z, psi_sxz_x, psi_szz_z,
         psi_vx_x, psi_vz_z, psi_vx_z, psi_vz_x) = state
        c11, c13, c33, c44, buoy = (
            coeffs["c11"], coeffs["c13"], coeffs["c33"], coeffs["c44"], coeffs["buoyancy"]
        )
        e_p, d_s, mem_b, mem_c = coeffs["e_p"], coeffs["d_s"], coeffs["mem_b"], coeffs["mem_c"]
        c = self._coeffs

        # --- velocity update (uses current stresses): identical to elastic ---
        d_sxx_x = diff_x_forward(sxx, c, dx)
        d_sxz_z = diff_z_backward(sxz, c, dz)
        s1, psi_sxx_x = CPML.stretch(d_sxx_x, psi_sxx_x, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        s2, psi_sxz_z = CPML.stretch(d_sxz_z, psi_sxz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        vx = vx + dt * buoy * (s1 + s2)

        d_sxz_x = diff_x_backward(sxz, c, dx)
        d_szz_z = diff_z_forward(szz, c, dz)
        s3, psi_sxz_x = CPML.stretch(d_sxz_x, psi_sxz_x, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        s4, psi_szz_z = CPML.stretch(d_szz_z, psi_szz_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        vz = vz + dt * buoy * (s3 + s4)

        # --- strains (CPML-stretched), shared by stress and memory updates ----
        d_vx_x = diff_x_backward(vx, c, dx)
        d_vz_z = diff_z_backward(vz, c, dz)
        ex, psi_vx_x = CPML.stretch(d_vx_x, psi_vx_x, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        ez, psi_vz_z = CPML.stretch(d_vz_z, psi_vz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        d_vx_z = diff_z_forward(vx, c, dz)
        d_vz_x = diff_x_forward(vz, c, dx)
        g1, psi_vx_z = CPML.stretch(d_vx_z, psi_vx_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        g2, psi_vz_x = CPML.stretch(d_vz_x, psi_vz_x, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        gxz = g1 + g2

        # --- SLS memory variables (trapezoidal, backward-stable b<1) ----------
        theta = ex + ez
        r_xx_new = mem_b * (r_xx * mem_c - e_p * theta + 2.0 * d_s * ez)
        r_zz_new = mem_b * (r_zz * mem_c - e_p * theta + 2.0 * d_s * ex)
        r_xz_new = mem_b * (r_xz * mem_c - d_s * gxz)

        # --- stress update: unrelaxed elastic term + trapezoidal memory -------
        sxx = sxx + dt * (c11 * ex + c13 * ez) + 0.5 * dt * (r_xx + r_xx_new)
        szz = szz + dt * (c13 * ex + c33 * ez) + 0.5 * dt * (r_zz + r_zz_new)
        sxz = sxz + dt * c44 * gxz + 0.5 * dt * (r_xz + r_xz_new)

        return (vx, vz, sxx, szz, sxz, r_xx_new, r_zz_new, r_xz_new,
                psi_sxx_x, psi_sxz_z, psi_sxz_x, psi_szz_z,
                psi_vx_x, psi_vz_z, psi_vx_z, psi_vz_x)

    def inverse_step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        dt: float,
        dx: float,
        dz: float,
        set_rim: Callable[[Tensor, str], Tensor],
    ) -> list[Tensor]:
        """Boundary-saving is intentionally unavailable for viscoelastic.

        The dissipative memory variables decay each forward step (``b < 1``), so
        reverse-time reconstruction grows geometrically and amplifies
        floating-point error across the window; attenuation is not
        time-reversible. (Override the inherited elastic ``inverse_step``, which
        would silently discard the memory variables.) Use
        ``MemoryConfig('checkpoint')`` (forward recompute, exact) for
        memory-bounded viscoelastic backward passes.
        """
        raise NotImplementedError(
            "ViscoElasticVelocityStress does not support boundary-saving "
            "(dissipative memory variables make reverse reconstruction "
            "ill-conditioned); use MemoryConfig('checkpoint') instead."
        )
