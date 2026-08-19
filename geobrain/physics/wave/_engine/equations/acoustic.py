"""Acoustic wave equation: first-order pressure–velocity (velocity-stress) form.

State (all stored on equally sized ``(B, 1, nz, nx)`` tensors; the staggering is
carried by the ``D+`` / ``D−`` operators rather than by different array sizes):

    p    pressure              integer nodes
    vx   x particle velocity   x half nodes
    vz   z particle velocity   z half nodes
    ψ_vxx, ψ_vzz               CPML memory for ∂vx/∂x, ∂vz/∂z (pressure update)
    ψ_px,  ψ_pz                CPML memory for ∂p/∂x, ∂p/∂z   (velocity update)

Leapfrog update (κ = ρ c², buoyancy = 1/ρ)::

    p  ← p  − Δt κ ( ∂̃vx/∂x + ∂̃vz/∂z )
    vx ← vx − Δt (1/ρ) ∂̃p/∂x
    vz ← vz − Δt (1/ρ) ∂̃p/∂z

where ``∂̃`` denotes the CPML-stretched derivative. Pressure differences velocity
with the backward operator (half→integer); velocity differences pressure with the
forward operator (integer→half).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Callable, ClassVar, Mapping, Sequence

import torch
from torch import Tensor

from ..boundaries.cpml import CPML
from ..boundaries.free_surface import zero_free_surface_row
from ..fd.coefficients import staggered_first_derivative_coeffs
from ..fd.derivative import (
    diff_x_backward,
    diff_x_forward,
    diff_z_backward,
    diff_z_backward_freesurface_even,
    diff_z_forward,
    diff_z_forward_freesurface_odd,
)
from .base import FieldSpec, ModelSpec, WaveEquation


class AcousticVelocityStress(WaveEquation):
    """Isotropic acoustic wave equation, first-order velocity-stress."""

    FIELD_SPECS: ClassVar[tuple[FieldSpec, ...]] = (
        FieldSpec("p", "00"),
        FieldSpec("vx", "x"),
        FieldSpec("vz", "z"),
        FieldSpec("psi_vxx", "00", is_memory=True),
        FieldSpec("psi_vzz", "00", is_memory=True),
        FieldSpec("psi_px", "x", is_memory=True),
        FieldSpec("psi_pz", "z", is_memory=True),
    )
    MODEL_SPECS: ClassVar[tuple[ModelSpec, ...]] = (
        ModelSpec("vp"),
        ModelSpec("rho"),
    )

    def __init__(
        self, fd_order: int = 8, *, normalize_source_by_cell_volume: bool = False
    ) -> None:
        super().__init__(
            fd_order,
            normalize_source_by_cell_volume=normalize_source_by_cell_volume,
        )
        self._coeffs = staggered_first_derivative_coeffs(fd_order)

    @property
    def source_field(self) -> str:
        return "p"

    @property
    def default_receiver_field(self) -> str:
        return "p"

    def prepare(
        self, models: Mapping[str, Tensor], dt: float, dx: float, dz: float
    ) -> Mapping[str, Tensor]:
        vp = models["vp"]
        rho = models["rho"]
        kappa = rho * vp * vp          # bulk modulus, co-located with p
        buoyancy = 1.0 / rho           # co-located with v (no half-cell averaging)
        return {"kappa": kappa, "buoyancy": buoyancy}

    def step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        cpml: CPML,
        dt: float,
        dx: float,
        dz: float,
    ) -> tuple[Tensor, ...]:
        p, vx, vz, psi_vxx, psi_vzz, psi_px, psi_pz = state
        kappa = coeffs["kappa"]
        buoyancy = coeffs["buoyancy"]
        c = self._coeffs
        fs = self._free_surface

        # --- pressure update: difference velocities (half -> integer) -----
        # Free surface (pressure release): the top-edge ∂vz/∂z uses the EVEN image
        # of vz (symmetric normal velocity) instead of the zero-fill rigid wall.
        dvx_dx = diff_x_backward(vx, c, dx)
        dvz_dz = (diff_z_backward_freesurface_even(vz, c, dz) if fs
                  else diff_z_backward(vz, c, dz))
        sx, psi_vxx = CPML.stretch(dvx_dx, psi_vxx, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        sz, psi_vzz = CPML.stretch(dvz_dz, psi_vzz, cpml.bz_int, cpml.az_int, cpml.kz_int)
        p = p - dt * kappa * (sx + sz)
        if fs:
            # Enforce the pressure node p = 0 on the surface row.
            p = zero_free_surface_row(p, self._surface_row)

        # --- velocity update: difference pressure (integer -> half) -------
        # Free surface: the top-edge ∂p/∂z uses the ODD image of the (node)
        # pressure so the surface-adjacent vz feels the mirrored -p above.
        dp_dx = diff_x_forward(p, c, dx)
        dp_dz = (diff_z_forward_freesurface_odd(p, c, dz) if fs
                 else diff_z_forward(p, c, dz))
        tx, psi_px = CPML.stretch(dp_dx, psi_px, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        tz, psi_pz = CPML.stretch(dp_dz, psi_pz, cpml.bz_half, cpml.az_half, cpml.kz_half)
        vx = vx - dt * buoyancy * tx
        vz = vz - dt * buoyancy * tz

        return p, vx, vz, psi_vxx, psi_vzz, psi_px, psi_pz

    def inverse_step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        dt: float,
        dx: float,
        dz: float,
        set_rim: Callable[[Tensor, str], Tensor],
    ) -> list[Tensor]:
        p, vx, vz, *psi = state  # source already removed by the caller
        kappa, buoy = coeffs["kappa"], coeffs["buoyancy"]
        c = self._coeffs
        # Forward updated p (from old v) then v (from new p); invert in reverse.
        vx_k = set_rim(vx + dt * buoy * diff_x_forward(p, c, dx), "vx")
        vz_k = set_rim(vz + dt * buoy * diff_z_forward(p, c, dz), "vz")
        p_k = set_rim(
            p + dt * kappa * (diff_x_backward(vx_k, c, dx) + diff_z_backward(vz_k, c, dz)),
            "p",
        )
        return [p_k, vx_k, vz_k, *(torch.zeros_like(p) for _ in psi)]

    def cfl_dt_max(self, vmax: float, dx: float, dz: float) -> float:
        # Von Neumann limit for the 2-D staggered scheme:
        #   dt_max = h / (vmax · sqrt(2) · Σ|c_k|).
        # For 4th order Σ|c| = 9/8 + 1/24 ⇒ dt_max ≈ 0.606 h/vmax, matching the
        # GeoBrain reference kernel.
        c_sum = sum(abs(ck) for ck in self._coeffs)
        h = min(dx, dz)
        return h / (vmax * math.sqrt(2.0) * c_sum)
