"""3-D acoustic wave equation: first-order pressure-velocity.

State on public model-storage ``(B, 1, nz, nx, ny)`` tensors (staggering
encoded by ``D±``):

    p              integer nodes
    vx, vy, vz     half nodes in x / y / z
    ψ_vxx, ψ_vyy, ψ_vzz   CPML memory for the pressure update (integer)
    ψ_px,  ψ_py,  ψ_pz    CPML memory for the velocity updates (half)

Update (κ = ρc², buoyancy = 1/ρ)::

    p  ← p − Δt κ ( ∂̃vx/∂x + ∂̃vy/∂y + ∂̃vz/∂z )
    vx ← vx − Δt (1/ρ) ∂̃p/∂x   (and y, z)

The derivative helpers retain their historical tensor-axis names: ``diff_y``
operates on axis ``-2`` (public x) and ``diff_x`` on axis ``-1`` (public y).
This adapter maps them to the public ``vx``/``vy`` component semantics. The
shared packed eager backend consumes the equation through the internal
``WaveEquationProtocol``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from geobrain.core import GeoBrainError

import math
from collections.abc import Callable
from typing import ClassVar, Mapping, Sequence

import torch
from torch import Tensor

from ..boundaries.cpml3d import CPML3D
from ..fd.coefficients import staggered_first_derivative_coeffs
from ..fd.derivative import (
    diff_x_backward,
    diff_x_forward,
    diff_y_backward,
    diff_y_forward,
    diff_z_backward,
    diff_z_forward,
)
from .base import FieldSpec, ModelSpec, ReceiverRecording


class AcousticVelocityStress3D(ReceiverRecording):
    """Isotropic 3-D acoustic wave equation, first-order velocity-stress.

    Fields/params are declared via ``FIELD_SPECS`` / ``MODEL_SPECS`` (the same
    declarative vocabulary as the 2-D :class:`AcousticVelocityStress`); the
    introspection (``state_names`` / ``wavefield_names`` / ``memory_names`` /
    ``field_index``) is inherited from :class:`ReceiverRecording`, with CPML
    memory variables flagged by ``is_memory=True`` rather than a name heuristic.
    """

    DIMENSION = 3
    FIELD_SPECS: ClassVar[tuple[FieldSpec, ...]] = (
        FieldSpec("p"), FieldSpec("vx", "x"), FieldSpec("vy", "y"),
        FieldSpec("vz", "z"),
        FieldSpec("psi_vxx", is_memory=True), FieldSpec("psi_vyy", is_memory=True),
        FieldSpec("psi_vzz", is_memory=True), FieldSpec("psi_px", "x", is_memory=True),
        FieldSpec("psi_py", "y", is_memory=True), FieldSpec("psi_pz", "z", is_memory=True),
    )
    MODEL_SPECS: ClassVar[tuple[ModelSpec, ...]] = (
        ModelSpec("vp"),
        ModelSpec("rho"),
    )
    source_field = "p"
    snapshot_field = "p"
    default_receiver_field = "p"

    def __init__(self, fd_order: int = 8) -> None:
        if fd_order < 2 or fd_order % 2:
            raise GeoBrainError(f"fd_order must be a positive even integer: {fd_order}")
        self.fd_order = fd_order
        self._coeffs = staggered_first_derivative_coeffs(fd_order)

    def init_state(
        self,
        batch: int,
        nz: int,
        nx: int,
        ny: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> list[Tensor]:
        return [torch.zeros((batch, 1, nz, nx, ny), device=device, dtype=dtype)
                for _ in self.FIELD_SPECS]

    def prepare(
        self,
        models: Mapping[str, Tensor],
        dt: float,
        dy: float,
        dx: float,
        dz: float,
    ) -> Mapping[str, Tensor]:
        vp, rho = models["vp"], models["rho"]
        return {"kappa": rho * vp * vp, "buoyancy": 1.0 / rho}

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        return float(models["vp"].detach().max())

    def step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        cpml: CPML3D,
        dt: float,
        dy: float,
        dx: float,
        dz: float,
    ) -> tuple[Tensor, ...]:
        p, vx, vy, vz, psi_vxx, psi_vyy, psi_vzz, psi_px, psi_py, psi_pz = state
        kappa, buoy = coeffs["kappa"], coeffs["buoyancy"]
        c = self._coeffs

        # Public storage is (z, x, y): helper diff_y owns axis -2 (public x),
        # while helper diff_x owns axis -1 (public y). CPML by/bx follow those
        # tensor axes, respectively.
        # Retain the historical y→x→z construction order so pressure/model
        # gradients remain bit-exact while the state fields carry public names.
        dvy_dy = diff_x_backward(vy, c, dy)
        dvx_dx = diff_y_backward(vx, c, dx)
        dvz_dz = diff_z_backward(vz, c, dz)
        sy, psi_vyy = CPML3D.stretch(dvy_dy, psi_vyy, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        sx, psi_vxx = CPML3D.stretch(dvx_dx, psi_vxx, cpml.by_int, cpml.ay_int, cpml.ky_int)
        sz, psi_vzz = CPML3D.stretch(dvz_dz, psi_vzz, cpml.bz_int, cpml.az_int, cpml.kz_int)
        p = p - dt * kappa * (sy + sx + sz)

        dp_dy = diff_x_forward(p, c, dy)
        dp_dx = diff_y_forward(p, c, dx)
        dp_dz = diff_z_forward(p, c, dz)
        ty, psi_py = CPML3D.stretch(dp_dy, psi_py, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        tx, psi_px = CPML3D.stretch(dp_dx, psi_px, cpml.by_half, cpml.ay_half, cpml.ky_half)
        tz, psi_pz = CPML3D.stretch(dp_dz, psi_pz, cpml.bz_half, cpml.az_half, cpml.kz_half)
        vy = vy - dt * buoy * ty
        vx = vx - dt * buoy * tx
        vz = vz - dt * buoy * tz

        return p, vx, vy, vz, psi_vxx, psi_vyy, psi_vzz, psi_px, psi_py, psi_pz

    def add_source(
        self,
        state: Sequence[Tensor],
        iz: Tensor,
        ix: Tensor,
        iy: Tensor,
        amp: Tensor,
        dt: float,
    ) -> list[Tensor]:
        idx = self.field_index(self.source_field)
        new = list(state)
        new[idx] = _index_add_3d(new[idx], iz, ix, iy, dt * amp)
        return new

    # record_receivers (default p), record_field, multi-component + illumination
    # come from the ReceiverRecording mixin (dimension-agnostic via *coords).

    def inverse_step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        dt: float,
        dy: float,
        dx: float,
        dz: float,
        set_rim: Callable[[Tensor, str], Tensor],
    ) -> list[Tensor]:
        p, vx, vy, vz, *psi = state  # source already removed by the caller
        kappa, buoy = coeffs["kappa"], coeffs["buoyancy"]
        c = self._coeffs
        vx_k = set_rim(vx + dt * buoy * diff_y_forward(p, c, dx), "vx")
        vy_k = set_rim(vy + dt * buoy * diff_x_forward(p, c, dy), "vy")
        vz_k = set_rim(vz + dt * buoy * diff_z_forward(p, c, dz), "vz")
        p_k = set_rim(
            p + dt * kappa * (diff_y_backward(vx_k, c, dx)
                              + diff_x_backward(vy_k, c, dy)
                              + diff_z_backward(vz_k, c, dz)),
            "p",
        )
        return [p_k, vx_k, vy_k, vz_k, *(torch.zeros_like(p) for _ in psi)]

    def cfl_dt_max(self, vmax: float, dy: float, dx: float, dz: float) -> float:
        c_sum = sum(abs(ck) for ck in self._coeffs)
        return min(dx, dy, dz) / (vmax * math.sqrt(3.0) * c_sum)


def _index_add_3d(
    field: Tensor,
    iz: Tensor,
    ix: Tensor,
    iy: Tensor,
    values: Tensor,
) -> Tensor:
    """Out-of-place ``field[b, 0, iz[b], ix[b], iy[b]] += values[b]``."""
    n = field.shape[0]
    batch = torch.arange(n, device=field.device)
    chan = torch.zeros(n, dtype=torch.long, device=field.device)
    return torch.index_put(field, (batch, chan, iz, ix, iy), values, accumulate=True)
