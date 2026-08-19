"""Isotropic elastic wave equation: first-order velocity-stress (Virieux, 1986).

Staggered layout (collocated storage; staggering encoded by ``D±``):

    vx   x velocity     (i,   j+1/2)   x half, z int
    vz   z velocity     (i+1/2, j)     x int,  z half
    sxx  normal stress  (i,   j)       integer
    szz  normal stress  (i,   j)       integer
    sxz  shear stress   (i+1/2, j+1/2) x half, z half

Update (ρ, λ, μ; c11 = λ+2μ), velocities then stresses::

    vx  += Δt (1/ρ) ( ∂̃sxx/∂x + ∂̃sxz/∂z )
    vz  += Δt (1/ρ) ( ∂̃sxz/∂x + ∂̃szz/∂z )
    sxx += Δt ( c11 ∂̃vx/∂x + λ  ∂̃vz/∂z )
    szz += Δt ( λ  ∂̃vx/∂x + c11 ∂̃vz/∂z )
    sxz += Δt μ ( ∂̃vx/∂z + ∂̃vz/∂x )

For each derivative the staggered operator (``D+`` int→half, ``D−`` half→int) and
the CPML profile (matched to the *result* node) are fixed by the table above.

Acoustic limit: ``vs = 0`` ⇒ μ = 0, λ = ρc² ; then ``sxz`` decouples and
``sxx = szz``; the pressure trace ``-(sxx+szz)/2`` reproduces the acoustic
solution (up to the source-sign convention).

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
    avg_hh_to_ii,
    avg_ii_to_hh,
    diff_x_backward,
    diff_x_forward,
    diff_z_backward,
    diff_z_backward_freesurface_even,
    diff_z_backward_freesurface_odd,
    diff_z_forward,
    diff_z_forward_freesurface_even,
    diff_z_forward_freesurface_odd,
)
from .base import FieldSpec, ModelSpec, WaveEquation, _index_add, _index_add_multi


def _vti_max_phase_velocity(models: Mapping[str, Tensor]) -> float:
    """Return the exact maximum VTI phase speed over every propagation angle.

    The qP/qSV eigenvalues depend on ``t = sin(angle)^2`` through a linear
    trace plus the square root of a quadratic. Squaring its stationary-point
    equation gives a quadratic whose in-domain roots, together with both
    principal axes, contain the global maximum. Extra roots introduced by
    squaring are harmless because they are still values of the same phase-speed
    curve. The azimuthal SH maximum is included when ``gamma`` is present.
    """
    vp, vs = models["vp"], models["vs"]
    epsilon, delta = models["epsilon"], models["delta"]
    c33 = vp.square()
    c44 = vs.square()
    c11 = c33 * (1.0 + 2.0 * epsilon)
    difference = c33 - c44
    c13 = torch.sqrt(difference.square() + 2.0 * delta * c33 * difference) - c44
    scale = torch.stack((c11.abs(), c13.abs(), c33.abs(), c44.abs())).amax(dim=0)
    c11, c13, c33, c44 = (coefficient / scale for coefficient in (c11, c13, c33, c44))
    coupling = c13 + c44

    trace1 = c11 - c33
    difference0 = c44 - c33
    difference1 = c11 + c33 - 2.0 * c44
    square0 = difference0.square()
    square1 = 2.0 * difference0 * difference1 + 4.0 * coupling.square()
    square2 = difference1.square() - 4.0 * coupling.square()

    def qp_eigenvalue(t: Tensor) -> Tensor:
        diagonal_x = c44 + (c11 - c44) * t
        diagonal_z = c33 + (c44 - c33) * t
        coupling_squared = coupling.square() * t * (1.0 - t)
        discriminant = (diagonal_x - diagonal_z).square() + 4.0 * coupling_squared
        return 0.5 * (diagonal_x + diagonal_z + torch.sqrt(discriminant))

    zero = torch.zeros_like(vp)
    one = torch.ones_like(vp)
    candidates = [zero]

    common = square2 - trace1.square()
    quadratic = 4.0 * square2 * common
    linear = 4.0 * square1 * common
    constant = square1.square() - 4.0 * trace1.square() * square0
    discriminant = linear.square() - 4.0 * quadratic * constant
    quadratic_case = quadratic != 0.0
    real_roots = discriminant >= 0.0
    root_span = torch.sqrt(torch.where(real_roots, discriminant, zero))
    denominator = torch.where(quadratic_case, 2.0 * quadratic, one)
    for sign in (-1.0, 1.0):
        root = (-linear + sign * root_span) / denominator
        valid = quadratic_case & real_roots & (root > 0.0) & (root < 1.0)
        safe_root = torch.where(valid, root, zero)
        candidates.append(torch.where(valid, qp_eigenvalue(safe_root), zero))

    linear_case = (~quadratic_case) & (linear != 0.0)
    linear_root = -constant / torch.where(linear_case, linear, one)
    valid_linear = linear_case & (linear_root > 0.0) & (linear_root < 1.0)
    safe_linear_root = torch.where(valid_linear, linear_root, zero)
    candidates.append(torch.where(valid_linear, qp_eigenvalue(safe_linear_root), zero))

    interior_velocity = torch.sqrt(torch.stack(candidates).amax(dim=0) * scale)
    horizontal_qp = vp * torch.sqrt(1.0 + 2.0 * epsilon)
    maximum_velocity = torch.maximum(vp, horizontal_qp)
    maximum_velocity = torch.maximum(maximum_velocity, vs)
    maximum_velocity = torch.maximum(maximum_velocity, interior_velocity)
    if "gamma" in models:
        horizontal_sh = vs * torch.sqrt(1.0 + 2.0 * models["gamma"])
        maximum_velocity = torch.maximum(maximum_velocity, horizontal_sh)
    return float(maximum_velocity.detach().max())


class ElasticVelocityStress(WaveEquation):
    """Isotropic elastic wave equation, first-order velocity-stress."""

    SOURCE_FIELDS = ("sxx", "szz")
    FIELD_SPECS: ClassVar[tuple[FieldSpec, ...]] = (
        FieldSpec("vx", "x"),
        FieldSpec("vz", "z"),
        FieldSpec("sxx", "00"),
        FieldSpec("szz", "00"),
        FieldSpec("sxz", "x"),  # genuinely (x-half, z-half); label is informational
        # CPML memory variables, one per stretched derivative
        FieldSpec("psi_sxx_x", "x", is_memory=True),
        FieldSpec("psi_sxz_z", "00", is_memory=True),
        FieldSpec("psi_sxz_x", "00", is_memory=True),
        FieldSpec("psi_szz_z", "z", is_memory=True),
        FieldSpec("psi_vx_x", "00", is_memory=True),
        FieldSpec("psi_vz_z", "00", is_memory=True),
        FieldSpec("psi_vx_z", "z", is_memory=True),
        FieldSpec("psi_vz_x", "x", is_memory=True),
    )
    MODEL_SPECS: ClassVar[tuple[ModelSpec, ...]] = (
        ModelSpec("vp"),
        ModelSpec("vs"),
        ModelSpec("rho"),
    )

    def __init__(self, fd_order: int = 8) -> None:
        super().__init__(fd_order)
        self._coeffs = staggered_first_derivative_coeffs(fd_order)

    @property
    def source_field(self) -> str:
        return "sxx"

    @property
    def default_receiver_field(self) -> str:
        return "sxx"  # overridden by record_receivers (records pressure)

    def prepare(
        self, models: Mapping[str, Tensor], dt: float, dx: float, dz: float
    ) -> Mapping[str, Tensor]:
        """Isotropic stiffness in the general ``(c11, c13, c33, c44)`` form.

        Isotropic ⇒ ``c11 = c33 = λ+2μ``, ``c13 = λ``, ``c44 = μ``, so the same
        ``step`` serves anisotropic (VTI) subclasses that only override this.
        """
        vp, vs, rho = models["vp"], models["vs"], models["rho"]
        mu = rho * vs * vs
        c33 = rho * vp * vp
        return {
            "c11": c33,
            "c13": c33 - 2.0 * mu,
            "c33": c33,
            "c44": mu,
            "buoyancy": 1.0 / rho,
        }

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        return float(models["vp"].detach().max())

    def step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        cpml: CPML,
        dt: float,
        dx: float,
        dz: float,
    ) -> tuple[Tensor, ...]:
        (vx, vz, sxx, szz, sxz,
         psi_sxx_x, psi_sxz_z, psi_sxz_x, psi_szz_z,
         psi_vx_x, psi_vz_z, psi_vx_z, psi_vz_x) = state
        c11, c13, c33, c44, buoy = (
            coeffs["c11"], coeffs["c13"], coeffs["c33"], coeffs["c44"], coeffs["buoyancy"]
        )
        c = self._coeffs
        fs = self._free_surface

        # Traction-free surface (Levander/Robertsson stress-imaging). Surface on
        # integer z-row ``_surface_row``: σzz, σxx, vx live there; σxz, vz sit half
        # a cell below. The z-derivatives that cross the surface use imaged
        # operators instead of the plain zero-fill (rigid wall). Mirror parities
        # about the surface plane:
        #   P-part  (exactly the acoustic pressure-release pair):
        #     σzz ODD (=0 forced on the surface row), normal velocity vz EVEN.
        #   SV-part:
        #     shear stress σxz ODD (→ 0 at the surface), horizontal velocity vx
        #     EVEN (free to slide).
        # σzz is forced to zero on the surface row (below); its antisymmetric image
        # then mirrors that as −σzz on the ghost rows the vz update reaches.

        # --- velocity update (uses current stresses) ----------------------
        d_sxx_x = diff_x_forward(sxx, c, dx)   # -> vx (x-half)
        d_sxz_z = (diff_z_backward_freesurface_odd(sxz, c, dz) if fs      # σxz odd
                   else diff_z_backward(sxz, c, dz))  # -> vx (z-int)
        s1, psi_sxx_x = CPML.stretch(d_sxx_x, psi_sxx_x, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        s2, psi_sxz_z = CPML.stretch(d_sxz_z, psi_sxz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        vx = vx + dt * buoy * (s1 + s2)

        d_sxz_x = diff_x_backward(sxz, c, dx)  # -> vz (x-int)
        d_szz_z = (diff_z_forward_freesurface_odd(szz, c, dz) if fs       # σzz odd (=0)
                   else diff_z_forward(szz, c, dz))  # -> vz (z-half)
        s3, psi_sxz_x = CPML.stretch(d_sxz_x, psi_sxz_x, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        s4, psi_szz_z = CPML.stretch(d_szz_z, psi_szz_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        vz = vz + dt * buoy * (s3 + s4)

        # --- stress update (uses new velocities) --------------------------
        d_vx_x = diff_x_backward(vx, c, dx)    # -> sxx/szz (x-int)
        d_vz_z = (diff_z_backward_freesurface_even(vz, c, dz) if fs       # vz even
                  else diff_z_backward(vz, c, dz))  # -> sxx/szz (z-int)
        e1, psi_vx_x = CPML.stretch(d_vx_x, psi_vx_x, cpml.bx_int, cpml.ax_int, cpml.kx_int)
        e2, psi_vz_z = CPML.stretch(d_vz_z, psi_vz_z, cpml.bz_int, cpml.az_int, cpml.kz_int)
        sxx = sxx + dt * (c11 * e1 + c13 * e2)
        szz = szz + dt * (c13 * e1 + c33 * e2)
        if fs:
            # Impose the normal-traction condition σzz = 0 on the surface row.
            szz = zero_free_surface_row(szz, self._surface_row)

        d_vx_z = (diff_z_forward_freesurface_even(vx, c, dz) if fs        # vx even
                  else diff_z_forward(vx, c, dz))   # -> sxz (z-half)
        d_vz_x = diff_x_forward(vz, c, dx)     # -> sxz (x-half)
        g1, psi_vx_z = CPML.stretch(d_vx_z, psi_vx_z, cpml.bz_half, cpml.az_half, cpml.kz_half)
        g2, psi_vz_x = CPML.stretch(d_vz_x, psi_vz_x, cpml.bx_half, cpml.ax_half, cpml.kx_half)
        sxz = sxz + dt * c44 * (g1 + g2)

        return (vx, vz, sxx, szz, sxz,
                psi_sxx_x, psi_sxz_z, psi_sxz_x, psi_szz_z,
                psi_vx_x, psi_vz_z, psi_vx_z, psi_vz_x)

    def add_source(
        self,
        state: Sequence[Tensor],
        src_z: Tensor,
        src_x: Tensor,
        amp: Tensor,
        dt: float,
    ) -> list[Tensor]:
        """Explosive (isotropic) source: drive ``sxx`` and ``szz`` equally."""
        new = list(state)
        i_sxx = self.field_index("sxx")
        i_szz = self.field_index("szz")
        new[i_sxx] = _index_add(new[i_sxx], src_z, src_x, dt * amp)
        new[i_szz] = _index_add(new[i_szz], src_z, src_x, dt * amp)
        return new

    def add_source_multi(
        self,
        state: Sequence[Tensor],
        iz: Tensor,
        ix: Tensor,
        amp: Tensor,
        dt: float,
    ) -> list[Tensor]:
        """Explosive encoded source: drive ``sxx`` and ``szz`` at all positions."""
        new = list(state)
        i_sxx = self.field_index("sxx")
        i_szz = self.field_index("szz")
        new[i_sxx] = _index_add_multi(new[i_sxx], iz, ix, dt * amp)
        new[i_szz] = _index_add_multi(new[i_szz], iz, ix, dt * amp)
        return new

    def _record_primary(self, state: Sequence[Tensor], *coords: Tensor) -> Tensor:
        """Record pressure ``-(sxx + szz) / 2`` at the receivers."""
        p = self.primary_wavefield(state)
        return p[(slice(None), 0, *coords)]

    def primary_wavefield(self, state: Sequence[Tensor]) -> Tensor:
        """Return derived pressure ``-(sxx+szz)/2`` over the full grid."""
        sxx = state[self.field_index("sxx")]
        szz = state[self.field_index("szz")]
        return -0.5 * (sxx + szz)

    def illumination_fields(self, state: Sequence[Tensor]) -> dict[str, Tensor]:
        """Illumination maps keyed to the legacy ``forward_wavefield_{p,vx,vz}``:
        pressure ``-(sxx+szz)/2`` energy plus the two particle velocities."""
        sxx = state[self.field_index("sxx")]
        szz = state[self.field_index("szz")]
        return {
            "p": -0.5 * (sxx + szz),
            "vx": state[self.field_index("vx")],
            "vz": state[self.field_index("vz")],
        }

    def inverse_step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        dt: float,
        dx: float,
        dz: float,
        set_rim: Callable[[Tensor, str], Tensor],
    ) -> list[Tensor]:
        vx, vz, sxx, szz, sxz, *psi = state  # source already removed by the caller
        c11, c13, c33, c44, buoy = (
            coeffs["c11"], coeffs["c13"], coeffs["c33"], coeffs["c44"], coeffs["buoyancy"]
        )
        c = self._coeffs
        # Forward: v from old s, then s from new v. Invert in reverse: s, then v.
        d_vx_x = diff_x_backward(vx, c, dx)
        d_vz_z = diff_z_backward(vz, c, dz)
        d_vx_z = diff_z_forward(vx, c, dz)
        d_vz_x = diff_x_forward(vz, c, dx)
        sxx_k = set_rim(sxx - dt * (c11 * d_vx_x + c13 * d_vz_z), "sxx")
        szz_k = set_rim(szz - dt * (c13 * d_vx_x + c33 * d_vz_z), "szz")
        sxz_k = set_rim(sxz - dt * c44 * (d_vx_z + d_vz_x), "sxz")
        vx_k = set_rim(
            vx - dt * buoy * (diff_x_forward(sxx_k, c, dx) + diff_z_backward(sxz_k, c, dz)),
            "vx",
        )
        vz_k = set_rim(
            vz - dt * buoy * (diff_x_backward(sxz_k, c, dx) + diff_z_forward(szz_k, c, dz)),
            "vz",
        )
        return [vx_k, vz_k, sxx_k, szz_k, sxz_k, *(torch.zeros_like(vx) for _ in psi)]

    def cfl_dt_max(self, vmax: float, dx: float, dz: float) -> float:
        c_sum = sum(abs(ck) for ck in self._coeffs)
        return min(dx, dz) / (vmax * math.sqrt(2.0) * c_sum)


class ElasticVTI(ElasticVelocityStress):
    """Transversely isotropic (VTI) elastic wave equation, Thomsen parameters.

    Inherits the velocity-stress fields, source, receiver, and step from
    :class:`ElasticVelocityStress`; only the stiffness mapping changes. With
    vertical velocities ``vp``/``vs`` and Thomsen ``ε``/``δ`` (and ``c33 = ρ vp²``,
    ``c44 = ρ vs²``)::

        c11 = c33 (1 + 2ε)
        c13 = sqrt( (c33 − c44)² + 2 δ c33 (c33 − c44) ) − c44
        c33 = c33 ,   c44 = c44

    Isotropic limit ``ε = δ = 0`` ⇒ ``c11 = c33``, ``c13 = λ``, recovering
    :class:`ElasticVelocityStress` exactly.
    """

    MODEL_SPECS = (
        ModelSpec("vp"), ModelSpec("vs"), ModelSpec("rho"),
        ModelSpec("epsilon"), ModelSpec("delta"),
    )

    def prepare(
        self, models: Mapping[str, Tensor], dt: float, dx: float, dz: float
    ) -> Mapping[str, Tensor]:
        vp, vs, rho = models["vp"], models["vs"], models["rho"]
        eps, delta = models["epsilon"], models["delta"]
        c33 = rho * vp * vp
        c44 = rho * vs * vs
        c11 = c33 * (1.0 + 2.0 * eps)
        diff = c33 - c44
        c13 = torch.sqrt(diff * diff + 2.0 * delta * c33 * diff) - c44
        return {"c11": c11, "c13": c13, "c33": c33, "c44": c44, "buoyancy": 1.0 / rho}

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        return _vti_max_phase_velocity(models)


class ElasticTTI(ElasticVelocityStress):
    """Tilted transversely isotropic (TTI) elastic wave equation.

    The VTI stiffness (vertical symmetry axis) is rotated by a tilt angle ``θ``
    (radians, from vertical) into a full in-plane stiffness with off-diagonal
    coupling ``c15, c35``::

        sxx += Δt ( c11 ∂̃vx/∂x + c13 ∂̃vz/∂z + c15 γ̃xz )
        szz += Δt ( c13 ∂̃vx/∂x + c33 ∂̃vz/∂z + c35 γ̃xz )
        sxz += Δt ( c15 ∂̃vx/∂x + c35 ∂̃vz/∂z + c55 γ̃xz )

    where ``γxz = ∂vx/∂z + ∂vz/∂x``. The momentum (velocity) update is unchanged,
    anisotropy lives only in the constitutive law. Because the normal stresses
    sit on the integer node and the shear strain ``γxz`` on the ``(x½, z½)`` node,
    the ``c15/c35`` cross terms are bilinearly averaged between the two subgrids
    (the standard staggered-grid TTI treatment).

    Models: ``vp, vs, rho`` (vertical velocities) and Thomsen ``epsilon, delta``
    plus ``theta`` (tilt, radians). ``θ = 0`` reproduces :class:`ElasticVTI`
    exactly (``c15 = c35 = 0``, ``c55 = c44``).
    """

    MODEL_SPECS = (
        ModelSpec("vp"), ModelSpec("vs"), ModelSpec("rho"),
        ModelSpec("epsilon"), ModelSpec("delta"), ModelSpec("theta"),
    )

    @property
    def halo_width(self) -> int:
        # The c15/c35 cross terms bilinearly average across one extra cell.
        return self.fd_order // 2 + 1

    def prepare(
        self, models: Mapping[str, Tensor], dt: float, dx: float, dz: float
    ) -> Mapping[str, Tensor]:
        vp, vs, rho = models["vp"], models["vs"], models["rho"]
        eps, delta, theta = models["epsilon"], models["delta"], models["theta"]
        # VTI constants in the symmetry-axis frame: a=c11, c=c33, f=c13, l=c44.
        cc = rho * vp * vp
        ll = rho * vs * vs
        aa = cc * (1.0 + 2.0 * eps)
        diff = cc - ll
        ff = torch.sqrt(diff * diff + 2.0 * delta * cc * diff) - ll
        # Bond rotation by theta about the out-of-plane axis.
        m = torch.cos(theta)
        n = torch.sin(theta)
        m2, n2 = m * m, n * n
        m4, n4 = m2 * m2, n2 * n2
        mn = m * n
        fp = ff + 2.0 * ll
        c11 = aa * m4 + cc * n4 + 2.0 * fp * m2 * n2
        c33 = aa * n4 + cc * m4 + 2.0 * fp * m2 * n2
        c13 = ff * (m4 + n4) + (aa + cc - 4.0 * ll) * m2 * n2
        c55 = (aa + cc - 2.0 * ff) * m2 * n2 + ll * (m2 - n2) ** 2
        # Bond-consistent (+θ) sign for the off-diagonal coupling.
        c15 = -mn * (aa * m2 - cc * n2 - fp * (m2 - n2))
        c35 = -mn * (aa * n2 - cc * m2 + fp * (m2 - n2))
        return {"c11": c11, "c13": c13, "c33": c33, "c55": c55,
                "c15": c15, "c35": c35, "buoyancy": 1.0 / rho}

    def max_velocity(self, models: Mapping[str, Tensor]) -> float:
        return _vti_max_phase_velocity(models)

    def step(
        self,
        state: Sequence[Tensor],
        coeffs: Mapping[str, Tensor],
        cpml: CPML,
        dt: float,
        dx: float,
        dz: float,
    ) -> tuple[Tensor, ...]:
        (vx, vz, sxx, szz, sxz,
         psi_sxx_x, psi_sxz_z, psi_sxz_x, psi_szz_z,
         psi_vx_x, psi_vz_z, psi_vx_z, psi_vz_x) = state
        c11, c13, c33 = coeffs["c11"], coeffs["c13"], coeffs["c33"]
        c55, c15, c35, buoy = coeffs["c55"], coeffs["c15"], coeffs["c35"], coeffs["buoyancy"]
        c = self._coeffs

        # --- velocity update (momentum; isotropy-independent) -------------
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

        # --- stress update with anisotropic coupling ----------------------
        ex, psi_vx_x = CPML.stretch(diff_x_backward(vx, c, dx), psi_vx_x,
                                    cpml.bx_int, cpml.ax_int, cpml.kx_int)   # at II
        ez, psi_vz_z = CPML.stretch(diff_z_backward(vz, c, dz), psi_vz_z,
                                    cpml.bz_int, cpml.az_int, cpml.kz_int)   # at II
        g1, psi_vx_z = CPML.stretch(diff_z_forward(vx, c, dz), psi_vx_z,
                                    cpml.bz_half, cpml.az_half, cpml.kz_half)
        g2, psi_vz_x = CPML.stretch(diff_x_forward(vz, c, dx), psi_vz_x,
                                    cpml.bx_half, cpml.ax_half, cpml.kx_half)
        gxz = g1 + g2                                                       # at HH
        gxz_ii = avg_hh_to_ii(gxz)
        ex_hh = avg_ii_to_hh(ex)
        ez_hh = avg_ii_to_hh(ez)
        sxx = sxx + dt * (c11 * ex + c13 * ez + c15 * gxz_ii)
        szz = szz + dt * (c13 * ex + c33 * ez + c35 * gxz_ii)
        sxz = sxz + dt * (c15 * ex_hh + c35 * ez_hh + c55 * gxz)

        return (vx, vz, sxx, szz, sxz,
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
        vx, vz, sxx, szz, sxz, *psi = state  # source already removed
        c11, c13, c33 = coeffs["c11"], coeffs["c13"], coeffs["c33"]
        c55, c15, c35, buoy = coeffs["c55"], coeffs["c15"], coeffs["c35"], coeffs["buoyancy"]
        c = self._coeffs
        # Invert stress update first (strains from the known final velocities).
        ex = diff_x_backward(vx, c, dx)
        ez = diff_z_backward(vz, c, dz)
        gxz = diff_z_forward(vx, c, dz) + diff_x_forward(vz, c, dx)
        gxz_ii = avg_hh_to_ii(gxz)
        ex_hh = avg_ii_to_hh(ex)
        ez_hh = avg_ii_to_hh(ez)
        sxx_k = set_rim(sxx - dt * (c11 * ex + c13 * ez + c15 * gxz_ii), "sxx")
        szz_k = set_rim(szz - dt * (c13 * ex + c33 * ez + c35 * gxz_ii), "szz")
        sxz_k = set_rim(sxz - dt * (c15 * ex_hh + c35 * ez_hh + c55 * gxz), "sxz")
        # Then invert the velocity update (same as isotropic momentum law).
        vx_k = set_rim(
            vx - dt * buoy * (diff_x_forward(sxx_k, c, dx) + diff_z_backward(sxz_k, c, dz)),
            "vx",
        )
        vz_k = set_rim(
            vz - dt * buoy * (diff_x_backward(sxz_k, c, dx) + diff_z_forward(szz_k, c, dz)),
            "vz",
        )
        return [vx_k, vz_k, sxx_k, szz_k, sxz_k, *(torch.zeros_like(vx) for _ in psi)]
