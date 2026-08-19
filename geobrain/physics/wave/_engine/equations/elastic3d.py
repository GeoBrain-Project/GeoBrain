"""3-D elastic wave equation: first-order velocity-stress (Virieux, 3-D).

Wavefields on public model-storage ``(B, 1, nz, nx, ny)`` tensors
(collocated storage; staggering encoded by ``D±``):

    vx, vy, vz                     particle velocities (half in x / y / z)
    sxx, syy, szz                  normal stresses (integer)
    sxy, sxz, syz                  shear stresses (half in two axes each)

plus 18 CPML memory variables (one per stretched spatial derivative).

Stiffness uses the VTI symmetry (c22=c11, c23=c13, c55=c44), so the six
constants ``c11, c12, c13, c33, c44, c66`` cover both isotropic
(:class:`ElasticVelocityStress3D`) and VTI (:class:`ElasticVTI3D`). Update (ρ, buoyancy=1/ρ),
velocities then stresses::

    vx += Δt b ( ∂̃sxx/∂x + ∂̃sxy/∂y + ∂̃sxz/∂z )      (and cyclic for vy, vz)
    sxx += Δt ( c11 ∂̃vx/∂x + c12 ∂̃vy/∂y + c13 ∂̃vz/∂z )
    syy += Δt ( c12 ∂̃vx/∂x + c11 ∂̃vy/∂y + c13 ∂̃vz/∂z )
    szz += Δt ( c13 ∂̃vx/∂x + c13 ∂̃vy/∂y + c33 ∂̃vz/∂z )
    sxy += Δt c66 ( ∂̃vx/∂y + ∂̃vy/∂x )
    sxz += Δt c44 ( ∂̃vx/∂z + ∂̃vz/∂x )
    syz += Δt c44 ( ∂̃vy/∂z + ∂̃vz/∂y )

Consumed through ``WaveEquationProtocol`` by the shared packed eager loop.
The derivative helpers retain their historical tensor-axis names: ``diff_y``
operates on axis ``-2`` (public x) and ``diff_x`` on axis ``-1`` (public y).
The isotropic adapter maps all fields and CPML memories to public physical
component names while retaining the historical y→x→z operation order.

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
    avg1_hh_to_ii,
    avg1_ii_to_hh,
    diff_x_backward,
    diff_x_forward,
    diff_y_backward,
    diff_y_forward,
    diff_z_backward,
    diff_z_forward,
)
from .acoustic3d import _index_add_3d
from .base import FieldSpec, ModelSpec, ReceiverRecording
from .elastic import _vti_max_phase_velocity

# Engine-internal storage is (B, 1, nz, ny, nx): the engine's own axes are
# z=-3, y=-2, x=-1. Under the platform ``(nz, nx, ny)`` convention the model is
# fed AS-IS (no transpose) with a consistently x↔y-swapped grid, so PHYSICALLY
# array axis ``_AX_Y`` (=-2) is x and ``_AX_X`` (=-1) is y. The isotropic /
# acoustic / VTI updates are x↔y symmetric and don't care; only TTI (below) does,
# and it is pointed at the physical x–z tilt plane accordingly.
_AX_X, _AX_Y, _AX_Z = -1, -2, -3


class ElasticVelocityStress3D(ReceiverRecording):
    """Isotropic 3-D elastic wave equation, first-order velocity-stress."""

    DIMENSION = 3
    # Public names, ordered by the historical physical y→x→z graph traversal.
    SOURCE_FIELDS = ("syy", "sxx", "szz")
    _PRESSURE_FIELDS = ("syy", "sxx", "szz")
    FIELD_SPECS: ClassVar[tuple[FieldSpec, ...]] = (
        FieldSpec("vx", "x"), FieldSpec("vy", "y"), FieldSpec("vz", "z"),
        FieldSpec("sxx"), FieldSpec("syy"), FieldSpec("szz"),
        FieldSpec("sxy"), FieldSpec("sxz"), FieldSpec("syz"),
        # velocity-update memory
        FieldSpec("psi_sxx_x", is_memory=True), FieldSpec("psi_sxy_y", is_memory=True),
        FieldSpec("psi_sxz_z", is_memory=True),
        FieldSpec("psi_sxy_x", is_memory=True), FieldSpec("psi_syy_y", is_memory=True),
        FieldSpec("psi_syz_z", is_memory=True),
        FieldSpec("psi_sxz_x", is_memory=True), FieldSpec("psi_syz_y", is_memory=True),
        FieldSpec("psi_szz_z", is_memory=True),
        # stress-update memory
        FieldSpec("psi_vx_x", is_memory=True), FieldSpec("psi_vy_y", is_memory=True),
        FieldSpec("psi_vz_z", is_memory=True),
        FieldSpec("psi_vx_y", is_memory=True), FieldSpec("psi_vy_x", is_memory=True),
        FieldSpec("psi_vx_z", is_memory=True), FieldSpec("psi_vz_x", is_memory=True),
        FieldSpec("psi_vy_z", is_memory=True), FieldSpec("psi_vz_y", is_memory=True),
    )
    MODEL_SPECS: ClassVar[tuple[ModelSpec, ...]] = (
        ModelSpec("vp"),
        ModelSpec("vs"),
        ModelSpec("rho"),
    )
    source_field = "sxx"
    snapshot_field = "sxx"

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
        vp, vs, rho = models["vp"], models["vs"], models["rho"]
        mu = rho * vs * vs
        c33 = rho * vp * vp
        return {
            "c11": c33, "c12": c33 - 2.0 * mu, "c13": c33 - 2.0 * mu,
            "c33": c33, "c44": mu, "c66": mu, "buoyancy": 1.0 / rho,
        }

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
        (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
         p_sxx_x, p_sxy_y, p_sxz_z, p_sxy_x, p_syy_y, p_syz_z,
         p_sxz_x, p_syz_y, p_szz_z,
         p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y) = state
        c11, c12, c13 = coeffs["c11"], coeffs["c12"], coeffs["c13"]
        c33, c44, c66, buoy = coeffs["c33"], coeffs["c44"], coeffs["c66"], coeffs["buoyancy"]
        c = self._coeffs
        st = CPML3D.stretch

        # Public storage is (z, x, y): helper diff_y owns axis -2 (public x),
        # while helper diff_x owns axis -1 (public y). CPML by/bx follow those
        # tensor axes, respectively. Retain the historical y→x→z construction
        # order so pressure/model/wavelet gradients remain bit-exact.

        # --- velocity update (from current stresses) ----------------------
        a1, p_syy_y = st(diff_x_forward(syy, c, dy), p_syy_y, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        a2, p_sxy_x = st(diff_y_backward(sxy, c, dx), p_sxy_x, cpml.by_int, cpml.ay_int, cpml.ky_int)
        a3, p_syz_z = st(diff_z_backward(syz, c, dz), p_syz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        vy = vy + dt * buoy * (a1 + a2 + a3)

        b1, p_sxy_y = st(diff_x_backward(sxy, c, dy), p_sxy_y, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        b2, p_sxx_x = st(diff_y_forward(sxx, c, dx), p_sxx_x, cpml.by_half, cpml.ay_half, cpml.ky_half)
        b3, p_sxz_z = st(diff_z_backward(sxz, c, dz), p_sxz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        vx = vx + dt * buoy * (b1 + b2 + b3)

        d1, p_syz_y = st(diff_x_backward(syz, c, dy), p_syz_y, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        d2, p_sxz_x = st(diff_y_backward(sxz, c, dx), p_sxz_x, cpml.by_int, cpml.ay_int, cpml.ky_int)
        d3, p_szz_z = st(diff_z_forward(szz, c, dz), p_szz_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        vz = vz + dt * buoy * (d1 + d2 + d3)

        # --- stress update (from new velocities) --------------------------
        ey, p_vy_y = st(diff_x_backward(vy, c, dy), p_vy_y, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        ex, p_vx_x = st(diff_y_backward(vx, c, dx), p_vx_x, cpml.by_int, cpml.ay_int, cpml.ky_int)
        ez, p_vz_z = st(diff_z_backward(vz, c, dz), p_vz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        syy = syy + dt * (c11 * ey + c12 * ex + c13 * ez)
        sxx = sxx + dt * (c12 * ey + c11 * ex + c13 * ez)
        szz = szz + dt * (c13 * ey + c13 * ex + c33 * ez)

        g1, p_vy_x = st(diff_y_forward(vy, c, dx), p_vy_x, cpml.by_half, cpml.ay_half, cpml.ky_half)
        g2, p_vx_y = st(diff_x_forward(vx, c, dy), p_vx_y, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        sxy = sxy + dt * c66 * (g1 + g2)

        h1, p_vy_z = st(diff_z_forward(vy, c, dz), p_vy_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        h2, p_vz_y = st(diff_x_forward(vz, c, dy), p_vz_y, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        syz = syz + dt * c44 * (h1 + h2)

        i1, p_vx_z = st(diff_z_forward(vx, c, dz), p_vx_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        i2, p_vz_x = st(diff_y_forward(vz, c, dx), p_vz_x, cpml.by_half, cpml.ay_half, cpml.ky_half)
        sxz = sxz + dt * c44 * (i1 + i2)

        return (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                p_sxx_x, p_sxy_y, p_sxz_z, p_sxy_x, p_syy_y, p_syz_z,
                p_sxz_x, p_syz_y, p_szz_z,
                p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y)

    def add_source(
        self,
        state: Sequence[Tensor],
        iz: Tensor,
        ix: Tensor,
        iy: Tensor,
        amp: Tensor,
        dt: float,
    ) -> list[Tensor]:
        """Explosive source: drive sxx, syy, szz equally."""
        new = list(state)
        for name in self.source_fields:
            j = self.field_index(name)
            new[j] = _index_add_3d(new[j], iz, ix, iy, dt * amp)
        return new

    def _record_primary(
        self, state: Sequence[Tensor], *coords: Tensor
    ) -> Tensor:
        """Record pressure ``-(sxx+syy+szz)/3`` at the receivers."""
        p = self.primary_wavefield(state)
        return p[(slice(None), 0, *coords)]

    def primary_wavefield(self, state: Sequence[Tensor]) -> Tensor:
        """Return derived pressure ``-(sxx+syy+szz)/3`` over the full grid."""
        first, second, third = (
            state[self.field_index(name)] for name in self._PRESSURE_FIELDS
        )
        # Preserve the historical physical y→x→z graph while exposing public names.
        return -(first + second + third) / 3.0

    def illumination_fields(self, state: Sequence[Tensor]) -> dict[str, Tensor]:
        """Illumination keyed to legacy ``forward_wavefield_{p,vx,vy,vz}``:
        pressure ``-(sxx+syy+szz)/3`` energy plus the three particle velocities."""
        first, second, third = (
            state[self.field_index(name)] for name in self._PRESSURE_FIELDS
        )
        return {
            "p": -(first + second + third) / 3.0,
            "vx": state[self.field_index("vx")],
            "vy": state[self.field_index("vy")],
            "vz": state[self.field_index("vz")],
        }

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
        (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, *psi) = state
        c11, c12, c13 = coeffs["c11"], coeffs["c12"], coeffs["c13"]
        c33, c44, c66, buoy = coeffs["c33"], coeffs["c44"], coeffs["c66"], coeffs["buoyancy"]
        c = self._coeffs
        # Forward: v from old s, then s from new v. Invert: s (from known v), then v.
        ey = diff_x_backward(vy, c, dy)
        ex = diff_y_backward(vx, c, dx)
        ez = diff_z_backward(vz, c, dz)
        syy_k = set_rim(syy - dt * (c11 * ey + c12 * ex + c13 * ez), "syy")
        sxx_k = set_rim(sxx - dt * (c12 * ey + c11 * ex + c13 * ez), "sxx")
        szz_k = set_rim(szz - dt * (c13 * ey + c13 * ex + c33 * ez), "szz")
        sxy_k = set_rim(sxy - dt * c66 * (diff_y_forward(vy, c, dx) + diff_x_forward(vx, c, dy)), "sxy")
        syz_k = set_rim(syz - dt * c44 * (diff_z_forward(vy, c, dz) + diff_x_forward(vz, c, dy)), "syz")
        sxz_k = set_rim(sxz - dt * c44 * (diff_z_forward(vx, c, dz) + diff_y_forward(vz, c, dx)), "sxz")
        vy_k = set_rim(vy - dt * buoy * (diff_x_forward(syy_k, c, dy)
                                         + diff_y_backward(sxy_k, c, dx)
                                         + diff_z_backward(syz_k, c, dz)), "vy")
        vx_k = set_rim(vx - dt * buoy * (diff_x_backward(sxy_k, c, dy)
                                         + diff_y_forward(sxx_k, c, dx)
                                         + diff_z_backward(sxz_k, c, dz)), "vx")
        vz_k = set_rim(vz - dt * buoy * (diff_x_backward(syz_k, c, dy)
                                         + diff_y_backward(sxz_k, c, dx)
                                         + diff_z_forward(szz_k, c, dz)), "vz")
        return [vx_k, vy_k, vz_k, sxx_k, syy_k, szz_k, sxy_k, sxz_k, syz_k,
                *(torch.zeros_like(vx) for _ in psi)]

    def cfl_dt_max(self, vmax: float, dy: float, dx: float, dz: float) -> float:
        c_sum = sum(abs(ck) for ck in self._coeffs)
        return min(dx, dy, dz) / (vmax * math.sqrt(3.0) * c_sum)


class ElasticVTI3D(ElasticVelocityStress3D):
    """3-D VTI elastic wave equation (Thomsen ε, δ, γ).

    With ``c33 = ρ vp²`` (vertical P) and ``c44 = ρ vs²`` (vertical S)::

        c11 = c33 (1 + 2ε),   c66 = c44 (1 + 2γ),   c12 = c11 − 2 c66
        c13 = sqrt( (c33−c44)² + 2 δ c33 (c33−c44) ) − c44

    ε = δ = γ = 0 reproduces :class:`ElasticVelocityStress3D` exactly.
    """

    MODEL_SPECS: ClassVar[tuple[ModelSpec, ...]] = (
        ModelSpec("vp"),
        ModelSpec("vs"),
        ModelSpec("rho"),
        ModelSpec("epsilon"),
        ModelSpec("delta"),
        ModelSpec("gamma"),
    )

    def prepare(
        self,
        models: Mapping[str, Tensor],
        dt: float,
        dy: float,
        dx: float,
        dz: float,
    ) -> Mapping[str, Tensor]:
        vp, vs, rho = models["vp"], models["vs"], models["rho"]
        eps, delta, gamma = models["epsilon"], models["delta"], models["gamma"]
        c33 = rho * vp * vp
        c44 = rho * vs * vs
        c11 = c33 * (1.0 + 2.0 * eps)
        c66 = c44 * (1.0 + 2.0 * gamma)
        c12 = c11 - 2.0 * c66
        diff = c33 - c44
        c13 = torch.sqrt(diff * diff + 2.0 * delta * c33 * diff) - c44
        return {
            "c11": c11,
            "c12": c12,
            "c13": c13,
            "c33": c33,
            "c44": c44,
            "c66": c66,
            "buoyancy": 1.0 / rho,
        }

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        return _vti_max_phase_velocity(models)


def _vti_rotated_about_y(
    a: Tensor,
    c: Tensor,
    f: Tensor,
    c44: Tensor,
    c66: Tensor,
    theta: Tensor,
) -> dict[str, Tensor]:
    """VTI stiffness rotated by ``theta`` about the out-of-plane horizontal axis →
    monoclinic constants (mirror plane = material 1–3).

    ``a=C11=C22, c=C33, f=C13=C23, c44=C44=C55, c66=C66``
    (vertical-axis VTI).
    Returns the 13 non-zero rotated Voigt constants in the MATERIAL frame (axes
    1,2,3 with 2 the rotation axis; the symmetry axis tilts from 3 toward 1).
    ``_tti3d_stress`` assigns those material axes to engine slots so the tilt lands
    in the physical x–z plane (about physical y). ``theta=0`` is identity.
    """
    m, n = torch.cos(theta), torch.sin(theta)
    m2, n2 = m * m, n * n
    m4, n4 = m2 * m2, n2 * n2
    mn = m * n
    fp = f + 2.0 * c44
    return {
        # in-plane (x–z) block: identical to 2-D TTI
        "c11": a * m4 + c * n4 + 2.0 * fp * m2 * n2,
        "c33": a * n4 + c * m4 + 2.0 * fp * m2 * n2,
        "c13": f * (m4 + n4) + (a + c - 4.0 * c44) * m2 * n2,
        "c55": (a + c - 2.0 * f) * m2 * n2 + c44 * (m2 - n2) ** 2,
        # c15/c35 carry the Bond-consistent (+θ) sign: see test_tti3d.
        "c15": -mn * (a * m2 - c * n2 - fp * (m2 - n2)),
        "c35": -mn * (a * n2 - c * m2 + fp * (m2 - n2)),
        # y-coupling block
        "c22": a,
        "c12": (a - 2.0 * c66) * m2 + f * n2,
        "c23": (a - 2.0 * c66) * n2 + f * m2,
        "c25": mn * (f - (a - 2.0 * c66)),
        "c44": c44 * m2 + c66 * n2,
        "c66": c44 * n2 + c66 * m2,
        "c46": (c44 - c66) * mn,
    }


class ElasticTTI3D(ElasticVelocityStress3D):
    """3-D tilted TI elastic wave equation (dip tilt ``θ`` about the physical
    y-axis, i.e. the symmetry axis dips in the physical x–z plane).

    The VTI stiffness is rotated into a monoclinic tensor; the stress update gains
    off-diagonal coupling: under the platform ``(nz, nx, ny)`` convention the
    normal stresses and the physical **x–z** shear couple through ``c15/c25/c35``,
    and the remaining two shears couple through ``c46``. Those cross terms live on
    different staggered subgrids and are bilinearly averaged (single-axis averages
    in physical x and z). The momentum update is unchanged. ``θ = 0`` reproduces
    :class:`ElasticVTI3D` exactly; a model invariant in physical ``y`` reproduces
    the 2-D :class:`ElasticTTI` solution in the x–z plane.

    Models: ``vp, vs, rho, epsilon, delta, gamma, theta`` (θ in radians).
    """

    MODEL_SPECS: ClassVar[tuple[ModelSpec, ...]] = (
        ModelSpec("vp"),
        ModelSpec("vs"),
        ModelSpec("rho"),
        ModelSpec("epsilon"),
        ModelSpec("delta"),
        ModelSpec("gamma"),
        ModelSpec("theta"),
    )
    # TTI retains its independently frozen legacy engine-frame implementation.
    SOURCE_FIELDS = ("sxx", "syy", "szz")
    _PRESSURE_FIELDS = ("sxx", "syy", "szz")
    _PUBLIC_TO_ENGINE_FIELD: ClassVar[dict[str, str]] = {
        "vx": "vy",
        "vy": "vx",
        "sxx": "syy",
        "syy": "sxx",
        "sxz": "syz",
        "syz": "sxz",
    }

    def _public_field_index(self, name: str) -> int:
        """Resolve a public component without changing canonical state indices."""
        engine_name = self._PUBLIC_TO_ENGINE_FIELD.get(name, name)
        return super().field_index(engine_name)

    def record_field(
        self, state: Sequence[Tensor], field: str, *coords: Tensor
    ) -> Tensor:
        """Sample a raw component through the retained engine-frame adapter."""
        index = self._public_field_index(field)
        return state[index][(slice(None), 0, *coords)]

    def sample_receivers(
        self,
        state: Sequence[Tensor],
        receiver_indices: Tensor,
        receiver_shot_index: Tensor,
        components: tuple[str, ...],
    ) -> Mapping[str, Tensor]:
        """Expose packed receiver components under physical public names."""
        engine_components = tuple(
            self._PUBLIC_TO_ENGINE_FIELD.get(name, name) for name in components
        )
        sampled = super().sample_receivers(
            state,
            receiver_indices,
            receiver_shot_index,
            engine_components,
        )
        return {
            public_name: sampled[engine_name]
            for public_name, engine_name in zip(components, engine_components, strict=True)
        }

    def snapshot_fields(
        self, state: Sequence[Tensor]
    ) -> Mapping[str, Tensor]:
        """Expose the physical public x-normal diagnostic snapshot."""
        return {"wavefield": state[self._public_field_index(self.snapshot_field)]}

    def illumination_fields(
        self, state: Sequence[Tensor]
    ) -> dict[str, Tensor]:
        """Expose pressure and physical public velocity illumination fields."""
        pressure = self.primary_wavefield(state)
        return {
            "p": pressure,
            "vx": state[self._public_field_index("vx")],
            "vy": state[self._public_field_index("vy")],
            "vz": state[self._public_field_index("vz")],
        }

    @property
    def halo_width(self) -> int:
        return self.fd_order // 2 + 1

    def prepare(
        self,
        models: Mapping[str, Tensor],
        dt: float,
        dy: float,
        dx: float,
        dz: float,
    ) -> Mapping[str, Tensor]:
        vp, vs, rho = models["vp"], models["vs"], models["rho"]
        eps, delta, gamma, theta = (
            models["epsilon"],
            models["delta"],
            models["gamma"],
            models["theta"],
        )
        c = rho * vp * vp  # C33 (vertical P)
        c44 = rho * vs * vs  # C44 = C55 (vertical S)
        a = c * (1.0 + 2.0 * eps)  # C11 = C22
        c66 = c44 * (1.0 + 2.0 * gamma)
        diff = c - c44
        f = torch.sqrt(diff * diff + 2.0 * delta * c * diff) - c44
        out = _vti_rotated_about_y(a, c, f, c44, c66, theta)
        out["buoyancy"] = 1.0 / rho
        return out

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        return _vti_max_phase_velocity(models)

    def _strains(
        self,
        vx: Tensor,
        vy: Tensor,
        vz: Tensor,
        cpml: CPML3D,
        psis: Sequence[Tensor],
        dt: float,
        dx: float,
        dy: float,
        dz: float,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        tuple[Tensor, ...],
    ]:
        """Compute the six stretched strain-rate components and updated ψ."""
        (p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y) = psis
        c = self._coeffs
        st = CPML3D.stretch
        ex, p_vx_x = st(diff_x_backward(vx, c, dx), p_vx_x, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        ey, p_vy_y = st(diff_y_backward(vy, c, dy), p_vy_y, cpml.by_int, cpml.ay_int, cpml.ky_int)
        ez, p_vz_z = st(diff_z_backward(vz, c, dz), p_vz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        gxy1, p_vx_y = st(diff_y_forward(vx, c, dy), p_vx_y, cpml.by_half, cpml.ay_half, cpml.ky_half)
        gxy2, p_vy_x = st(diff_x_forward(vy, c, dx), p_vy_x, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        gxz1, p_vx_z = st(diff_z_forward(vx, c, dz), p_vx_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        gxz2, p_vz_x = st(diff_x_forward(vz, c, dx), p_vz_x, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        gyz1, p_vy_z = st(diff_z_forward(vy, c, dz), p_vy_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        gyz2, p_vz_y = st(diff_y_forward(vz, c, dy), p_vz_y, cpml.by_half, cpml.ay_half, cpml.ky_half)
        psis_new = (p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y)
        return ex, ey, ez, gxy1 + gxy2, gxz1 + gxz2, gyz1 + gyz2, psis_new

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
        # Preserve the frozen TTI engine-frame x/y positional semantics while
        # exposing the same public argument order as the 3-D equation base.
        dx, dy = dy, dx
        (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
         p_sxx_x, p_sxy_y, p_sxz_z, p_sxy_x, p_syy_y, p_syz_z,
         p_sxz_x, p_syz_y, p_szz_z,
         p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y) = state
        buoy = coeffs["buoyancy"]
        c = self._coeffs
        st = CPML3D.stretch

        # --- velocity update (momentum; unchanged from ElasticVelocityStress3D) ---------
        a1, p_sxx_x = st(diff_x_forward(sxx, c, dx), p_sxx_x, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        a2, p_sxy_y = st(diff_y_backward(sxy, c, dy), p_sxy_y, cpml.by_int, cpml.ay_int, cpml.ky_int)
        a3, p_sxz_z = st(diff_z_backward(sxz, c, dz), p_sxz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        vx = vx + dt * buoy * (a1 + a2 + a3)
        b1, p_sxy_x = st(diff_x_backward(sxy, c, dx), p_sxy_x, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        b2, p_syy_y = st(diff_y_forward(syy, c, dy), p_syy_y, cpml.by_half, cpml.ay_half, cpml.ky_half)
        b3, p_syz_z = st(diff_z_backward(syz, c, dz), p_syz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        vy = vy + dt * buoy * (b1 + b2 + b3)
        d1, p_sxz_x = st(diff_x_backward(sxz, c, dx), p_sxz_x, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        d2, p_syz_y = st(diff_y_backward(syz, c, dy), p_syz_y, cpml.by_int, cpml.ay_int, cpml.ky_int)
        d3, p_szz_z = st(diff_z_forward(szz, c, dz), p_szz_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        vz = vz + dt * buoy * (d1 + d2 + d3)

        # --- stress update with monoclinic coupling ----------------------
        psis = (p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y)
        ex, ey, ez, gxy, gxz, gyz, psis = self._strains(vx, vy, vz, cpml, psis, dt, dx, dy, dz)
        (p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y) = psis
        sxx, syy, szz, sxy, sxz, syz = _tti3d_stress(
            coeffs, dt, sxx, syy, szz, sxy, sxz, syz, ex, ey, ez, gxy, gxz, gyz)

        return (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                p_sxx_x, p_sxy_y, p_sxz_z, p_sxy_x, p_syy_y, p_syz_z,
                p_sxz_x, p_syz_y, p_szz_z,
                p_vx_x, p_vy_y, p_vz_z, p_vx_y, p_vy_x, p_vx_z, p_vz_x, p_vy_z, p_vz_y)

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
        # Match :meth:`step` without changing the retained engine-frame axes.
        dx, dy = dy, dx
        (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, *psi) = state
        buoy = coeffs["buoyancy"]
        c = self._coeffs
        # Invert stress update first (strains from the known final velocities, no CPML).
        ex = diff_x_backward(vx, c, dx)
        ey = diff_y_backward(vy, c, dy)
        ez = diff_z_backward(vz, c, dz)
        gxy = diff_y_forward(vx, c, dy) + diff_x_forward(vy, c, dx)
        gxz = diff_z_forward(vx, c, dz) + diff_x_forward(vz, c, dx)
        gyz = diff_z_forward(vy, c, dz) + diff_y_forward(vz, c, dy)
        nsxx, nsyy, nszz, nsxy, nsxz, nsyz = _tti3d_stress(
            coeffs, dt, sxx, syy, szz, sxy, sxz, syz, ex, ey, ez, gxy, gxz, gyz, sign=-1.0)
        sxx_k = set_rim(nsxx, "sxx")
        syy_k = set_rim(nsyy, "syy")
        szz_k = set_rim(nszz, "szz")
        sxy_k = set_rim(nsxy, "sxy")
        sxz_k = set_rim(nsxz, "sxz")
        syz_k = set_rim(nsyz, "syz")
        # Then invert the velocity update (momentum law, same as ElasticVelocityStress3D).
        vx_k = set_rim(vx - dt * buoy * (diff_x_forward(sxx_k, c, dx)
                                         + diff_y_backward(sxy_k, c, dy)
                                         + diff_z_backward(sxz_k, c, dz)), "vx")
        vy_k = set_rim(vy - dt * buoy * (diff_x_backward(sxy_k, c, dx)
                                         + diff_y_forward(syy_k, c, dy)
                                         + diff_z_backward(syz_k, c, dz)), "vy")
        vz_k = set_rim(vz - dt * buoy * (diff_x_backward(sxz_k, c, dx)
                                         + diff_y_backward(syz_k, c, dy)
                                         + diff_z_forward(szz_k, c, dz)), "vz")
        return [vx_k, vy_k, vz_k, sxx_k, syy_k, szz_k, sxy_k, sxz_k, syz_k,
                *(torch.zeros_like(vx) for _ in psi)]


def _tti3d_stress(
    coeffs: Mapping[str, Tensor],
    dt: float,
    sxx: Tensor,
    syy: Tensor,
    szz: Tensor,
    sxy: Tensor,
    sxz: Tensor,
    syz: Tensor,
    ex: Tensor,
    ey: Tensor,
    ez: Tensor,
    gxy: Tensor,
    gxz: Tensor,
    gyz: Tensor,
    sign: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply (sign=+1) or invert (sign=-1) the monoclinic stress increment.

    Platform axis convention ``(nz, nx, ny)``: the model is fed AS-IS, so array
    axis ``_AX_Y`` (=-2) is the physical **x** axis and ``_AX_X`` (=-1) is the
    physical **y** axis. The TTI symmetry axis tilts in the physical **x–z**
    plane (about the physical y-axis), so the in-plane shear is the physical
    x–z shear, carried in the engine ``syz``/``gyz`` slot, which lives on the
    (x½, z½) subgrid. This is the x↔y mirror of the historical (engine-frame
    x–z-plane) formulation; VTI is x↔y symmetric, so the mirror is exact and the
    material stiffness constants (``c11 … c66`` from ``_vti_rotated_about_y``) are
    unchanged, only which engine slot plays each material role moves.

    Cross terms are averaged between subgrids: the in-plane x–z shear ``gyz``
    (at x½z½) to the integer node and back; the other two shears across the x and
    z axes.
    """
    c11, c12, c13 = coeffs["c11"], coeffs["c12"], coeffs["c13"]
    c22, c23, c25 = coeffs["c22"], coeffs["c23"], coeffs["c25"]
    c33, c35, c15 = coeffs["c33"], coeffs["c35"], coeffs["c15"]
    c44, c46, c55, c66 = coeffs["c44"], coeffs["c46"], coeffs["c55"], coeffs["c66"]
    s = sign * dt

    # in-plane x–z shear gyz (x½,z½) -> integer node for the normal stresses
    gyz_ii = avg1_hh_to_ii(avg1_hh_to_ii(gyz, _AX_Y), _AX_Z)
    # normal strains (integer) -> (x½,z½) for the x–z-shear stress
    ex_h = avg1_ii_to_hh(avg1_ii_to_hh(ex, _AX_Y), _AX_Z)
    ey_h = avg1_ii_to_hh(avg1_ii_to_hh(ey, _AX_Y), _AX_Z)
    ez_h = avg1_ii_to_hh(avg1_ii_to_hh(ez, _AX_Y), _AX_Z)
    # gxz (x½,z½,yI) -> (x½,y½,zI) for sxy ; gxy (x½,y½,zI) -> (x½,yI,z½) for sxz
    gxz_xy = avg1_hh_to_ii(avg1_ii_to_hh(gxz, _AX_Y), _AX_Z)
    gxy_xz = avg1_ii_to_hh(avg1_hh_to_ii(gxy, _AX_Y), _AX_Z)

    syy = syy + s * (c11 * ey + c12 * ex + c13 * ez + c15 * gyz_ii)
    sxx = sxx + s * (c12 * ey + c22 * ex + c23 * ez + c25 * gyz_ii)
    szz = szz + s * (c13 * ey + c23 * ex + c33 * ez + c35 * gyz_ii)
    syz = syz + s * (c15 * ey_h + c25 * ex_h + c35 * ez_h + c55 * gyz)
    sxy = sxy + s * (c66 * gxy + c46 * gxz_xy)
    sxz = sxz + s * (c44 * gxz + c46 * gxy_xz)
    return sxx, syy, szz, sxy, sxz, syz
