"""Visco-acoustic wave equation: first-order velocity-stress with a single
Standard-Linear-Solid (SLS) relaxation mechanism (constant-Q at a reference
frequency).

Extends :class:`AcousticVelocityStress` with one memory variable ``r``
(co-located with pressure) that introduces frequency-dependent attenuation and
dispersion controlled by the quality factor ``Q``. The relaxation times follow
the standard single-mechanism constant-Q parameterisation that reproduces ``Q``
at the angular reference frequency ``ω₀`` (Blanch & Robertsson 1995; Carcione):

    τ_σ = (√(1 + Q⁻²) − Q⁻¹) / ω₀ ,   τ_ε = 1 / (ω₀² τ_σ) ,   τ = τ_ε/τ_σ − 1

The lossless limit ``Q → ∞`` gives ``τ → 0``, ``r → 0`` and recovers the
isotropic acoustic operator exactly.

Leapfrog update (κ = ρ c², buoyancy = 1/ρ, divergence ``θ = ∂̃vx/∂x+∂̃vz/∂z``)::

    r  ← (r − Δt/τ_σ · κ τ θ) / (1 + Δt/τ_σ)          # backward-Euler, stable
    p  ← p − Δt ( κ (1+τ) θ + (r_old+r_new)/2 )
    vx ← vx − Δt (1/ρ) ∂̃p/∂x ,   vz ← vz − Δt (1/ρ) ∂̃p/∂z

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
from .acoustic import AcousticVelocityStress
from .base import FieldSpec, ModelSpec


class ViscoAcousticVelocityStress(AcousticVelocityStress):
    """Constant-Q visco-acoustic wave equation (single SLS mechanism).

    Models: ``vp`` (m/s), ``rho`` (kg/m³), ``Q`` (quality factor, dimensionless).
    The reference frequency is fixed at construction (``ref_freq`` in Hz); the
    SLS reproduces ``Q`` at that frequency.
    """# p, vx, vz, r (SLS memory, a real wavefield: NOT a CPML memory var), then
    # the four CPML memory variables.
    FIELD_SPECS = (
        FieldSpec("p", "00"),
        FieldSpec("vx", "x"),
        FieldSpec("vz", "z"),
        FieldSpec("r", "00"),
        FieldSpec("psi_vxx", "00", is_memory=True),
        FieldSpec("psi_vzz", "00", is_memory=True),
        FieldSpec("psi_px", "x", is_memory=True),
        FieldSpec("psi_pz", "z", is_memory=True),
    )
    MODEL_SPECS = (ModelSpec("vp"), ModelSpec("rho"), ModelSpec("Q"))

    def __init__(self, fd_order: int = 8, ref_freq: float = 15.0) -> None:
        super().__init__(fd_order)
        if ref_freq <= 0.0:
            raise GeoBrainError(f"ref_freq must be positive, got {ref_freq}")
        self.ref_freq = float(ref_freq)
        self.omega0 = 2.0 * math.pi * self.ref_freq

    def prepare(
        self, models: Mapping[str, Tensor], dt: float, dx: float, dz: float
    ) -> Mapping[str, Tensor]:
        vp = models["vp"]
        rho = models["rho"]
        Q = models["Q"]
        kappa = rho * vp * vp
        buoyancy = 1.0 / rho
        # Single-mechanism constant-Q relaxation times at ω₀ (per cell via Q).
        inv_Q = 1.0 / Q
        t_sigma = (torch.sqrt(1.0 + inv_Q * inv_Q) - inv_Q) / self.omega0
        t_epsilon = 1.0 / (self.omega0 * self.omega0 * t_sigma)
        tau = t_epsilon / t_sigma - 1.0
        return {
            "kappa": kappa,
            "buoyancy": buoyancy,
            "tau": tau,
            "inv_tsig": 1.0 / t_sigma,
        }

    def cfl_limit(
        self, model: Mapping[str, Tensor], spacing: tuple[float, ...]
    ) -> float:
        """Use the unrelaxed SLS speed only for explicit-step stability.

        The instantaneous pressure update uses ``kappa * (1 + tau)``.  With
        the single-SLS parametrisation, ``sqrt(1 + tau)`` is the stable
        conjugate ``sqrt(1 + Q**-2) + Q**-1``; using the input relaxed,
        zero-frequency ``vp`` alone therefore underestimates the CFL speed at
        finite Q.  CPML profile calibration deliberately retains the frozen
        0.2.0 ``max_velocity`` convention; this override changes only the
        strict time-step limit.
        """
        vp, quality_factor = model["vp"], model["Q"]
        inverse_q = quality_factor.reciprocal()
        speed_factor = torch.sqrt(1.0 + inverse_q.square()) + inverse_q
        maximum_speed = float((vp * speed_factor).detach().max())
        return float(
            self.cfl_dt_max(maximum_speed, *tuple(reversed(spacing)))
        )

    def step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        cpml: CPML,
        dt: float,
        dx: float,
        dz: float,
    ) -> tuple[Tensor, ...]:
        p, vx, vz, r, psi_vxx, psi_vzz, psi_px, psi_pz = state
        kappa = coeffs["kappa"]
        buoyancy = coeffs["buoyancy"]
        tau = coeffs["tau"]
        inv_tsig = coeffs["inv_tsig"]
        c = self._coeffs

        # --- pressure update: divergence of velocity (half -> integer) + CPML --
        dvx_dx = diff_x_backward(vx, c, dx)
        dvz_dz = diff_z_backward(vz, c, dz)
        sx, psi_vxx = CPML.stretch(dvx_dx, psi_vxx, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        sz, psi_vzz = CPML.stretch(dvz_dz, psi_vzz, cpml.bz_int, cpml.az_int, cpml.kz_int)
        theta = sx + sz

        # SLS memory variable (backward-Euler in r ⇒ unconditionally stable):
        #   dr/dt = -(1/τ_σ)(r + κ τ θ)
        r_new = (r - dt * inv_tsig * (kappa * tau * theta)) / (1.0 + dt * inv_tsig)

        # pressure: dp/dt = -[κ(1+τ)θ + r̄],  r̄ = (r + r_new)/2
        p = p - dt * (kappa * (1.0 + tau) * theta + 0.5 * (r + r_new))

        # --- velocity update: pressure gradient (integer -> half) + CPML -------
        dp_dx = diff_x_forward(p, c, dx)
        dp_dz = diff_z_forward(p, c, dz)
        tx, psi_px = CPML.stretch(dp_dx, psi_px, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        tz, psi_pz = CPML.stretch(dp_dz, psi_pz, cpml.bz_half, cpml.az_half, cpml.kz_half)
        vx = vx - dt * buoyancy * tx
        vz = vz - dt * buoyancy * tz

        return p, vx, vz, r_new, psi_vxx, psi_vzz, psi_px, psi_pz

    def inverse_step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        dt: float,
        dx: float,
        dz: float,
        set_rim: Callable[[Tensor, str], Tensor],
    ) -> list[Tensor]:
        """Boundary-saving is intentionally unavailable for visco-acoustic.

        Although the single-SLS leapfrog is *algebraically* invertible, the
        memory variable decays each forward step by ``α = 1/(1+Δt/τ_σ) < 1``, so
        reverse-time reconstruction grows by ``1/α > 1`` and amplifies
        floating-point error geometrically across the time window, attenuation
        is not time-reversible. (We must override the inherited acoustic
        ``inverse_step``, which would otherwise silently discard ``r``.) Use
        ``MemoryConfig("checkpoint")`` (forward recompute, exact) for
        memory-bounded visco-acoustic backward passes.
        """
        raise NotImplementedError(
            "ViscoAcousticVelocityStress does not support boundary-saving "
            "(dissipative memory variable makes reverse reconstruction "
            "ill-conditioned); use MemoryConfig('checkpoint') instead."
        )
