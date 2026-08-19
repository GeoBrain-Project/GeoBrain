"""
Oil-water-gas three-phase transient flow on a general 2-D grid using the MPFA-O
multi-point flux.

The three-phase extension of :class:`MPFATwoPhaseModel`: same MPFA-O multi-point
geometric flux ``G^α_f = Σ_c T_c·Φ^α_c`` (stencils from
:func:`mpfa_o_face_flux_stencils_full`, built once from geometry + absolute
permeability) and phase-potential upwinding, now over three phases. The oil
relperm is Stone-II (``ThreePhaseRelPerm``), depending on both ``S_w`` and
``S_g``. Per-cell mass balance, one row per phase (SI)::

    V·φ·(ρ_α S_α − ρ_α S_α|old)/Δt  +  Σ_f (±F^α_f)  −  q^α  =  0 ,
    F^α_f = (ρ_α·k_rα/μ_α)|_up · G^α_f ,   up = sign(G^α_f)

State is ``[p (oil pressure), S_w, S_g]`` of length ``3·n_cells`` (``S_o = 1 −
S_w − S_g``). Phase potentials carry the two capillary pressures and gravity,
all routed through the same MPFA stencil:

    Φ_o = p (+ρ_o g z),  Φ_w = p − Pc_ow(S_w) (+ρ_w g z),  Φ_g = p + Pc_og(S_g) (+ρ_g g z)

Boundaries are no-flow (interior MPFA faces only); flow is driven by per-phase
sources. On a K-orthogonal grid the stencil collapses to two-point and this
reproduces the TPFA three-phase solution exactly; with ``S_g ≡ 0`` and no gas
source it reduces to the two-phase oil-water model.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, SupportsFloat, TypeAlias

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from .._defaults import S_MAX, S_MIN
from ..discretization.flux import scatter_internal_face_flux, upwind_cell
from ..discretization.mpfa import MPFAGrid2D, _stable_polygon_areas, mpfa_o_face_flux_stencils_full
from ..discretization.mpfa3d import MPFAGrid3D, hex_cell_volumes, mpfa_o_face_flux_stencils_3d_full
from ..properties import ThreePhaseRelPerm
from ..wells import FlowSourceTerms, source_block
from .mpfa_two_phase import nnc_phase_flux, register_nnc

ScalarInput: TypeAlias = SupportsFloat | torch.Tensor


if TYPE_CHECKING:
    class _ModuleBase:
        """Static interface for ``torch.nn.Module`` when imports are skipped."""

        def __init__(self) -> None:
            pass

        def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
            pass
else:
    _ModuleBase = nn.Module


class MPFAThreePhaseModel(_ModuleBase):
    """Oil-water-gas three-phase Darcy flow with an MPFA-O multi-point flux (2-D, SI).

    Args:
        grid: :class:`MPFAGrid2D`.
        perm_tensor: ``(n_cells, 2, 2)`` cell permeability tensors [m²].
        porosity: scalar or ``(n_cells,)``.
        relperm: a :class:`ThreePhaseRelPerm` (``kr_water(sw)``, ``kr_oil(sw,sg)``,
            ``kr_gas(sg)``).
        rho_*_ref / mu_* / c_*: per-phase reference density [kg/m³], viscosity
            [Pa·s] and compressibility [1/Pa] (``ρ_α = ρ_ref(1+c_α(p−p_ref))``).
        pc_ow: optional ``Pc_ow(S_w) = p_o − p_w`` [Pa]; pc_og: ``Pc_og(S_g) =
            p_g − p_o`` [Pa].
        depth: optional ``(n_cells,)`` depth (+down) [m] enabling gravity.
    """

    schema = _flow_model_schema(
        model_name="MPFAThreePhaseModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
            ("sg", "1", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
            ("gas_mass", "kg/s", "mass", "sg"),
        ),
        grid_kinds=("mpfa-2d",),
        phases=("water", "oil", "gas"),
        structured_sources=True,
    )
    grid: MPFAGrid2D | MPFAGrid3D
    relperm: ThreePhaseRelPerm
    pc_ow: Callable[[torch.Tensor], torch.Tensor] | None
    pc_og: Callable[[torch.Tensor], torch.Tensor] | None
    phi: torch.Tensor
    V: torch.Tensor
    depth: torch.Tensor | None
    L: torch.Tensor
    face_lr: torch.Tensor
    nnc_pairs: torch.Tensor | None
    nnc_trans: torch.Tensor | None

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: ScalarInput,
        relperm: ThreePhaseRelPerm,
        *,
        rho_w_ref: float = 1000.0,
        rho_o_ref: float = 800.0,
        rho_g_ref: float = 100.0,
        mu_w: float = 1e-3,
        mu_o: float = 2e-3,
        mu_g: float = 2e-5,
        c_w: float = 0.0,
        c_o: float = 0.0,
        c_g: float = 0.0,
        p_ref: float = 1e7,
        pc_ow: Callable[[torch.Tensor], torch.Tensor] | None = None,
        pc_og: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth: torch.Tensor | None = None,
        gravity: float = 9.81,
        cell_volumes: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_trans: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.relperm = relperm
        self.pc_ow, self.pc_og = pc_ow, pc_og
        self.rho_w_ref, self.rho_o_ref, self.rho_g_ref = (
            float(rho_w_ref),
            float(rho_o_ref),
            float(rho_g_ref),
        )
        self.mu_w, self.mu_o, self.mu_g = float(mu_w), float(mu_o), float(mu_g)
        self.c_w, self.c_o, self.c_g = float(c_w), float(c_o), float(c_g)
        self.p_ref = float(p_ref)
        self.g = float(gravity)
        n = len(grid.cell_nodes)
        self.n_cells = n
        dtype = perm_tensor.dtype

        if isinstance(porosity, torch.Tensor):
            self.register_buffer("phi", porosity.to(dtype))
        else:
            self.register_buffer("phi", torch.full((n,), float(porosity), dtype=dtype))
        if cell_volumes is None:
            cell_volumes = self._polygon_areas(grid).to(dtype)
        self.register_buffer("V", cell_volumes.to(dtype))
        if depth is not None:
            self.register_buffer("depth", depth.to(dtype))
        else:
            self.depth = None

        stencils = mpfa_o_face_flux_stencils_full(grid, perm_tensor)
        faces = sorted(stencils)
        L = perm_tensor.new_zeros(len(faces), n)
        lr: list[list[int]] = []
        for fi, f in enumerate(faces):
            for c, t in stencils[f].items():
                L[fi, c] = t
            left, right = grid.edge_cells[f]
            lr.append([left, right])
        self.register_buffer("L", L)
        self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long))
        register_nnc(self, nnc_pairs, nnc_trans, dtype)

    @staticmethod
    def _polygon_areas(grid: MPFAGrid2D) -> torch.Tensor:
        return _stable_polygon_areas(grid)

    def state_size(self) -> int:
        return 3 * self.n_cells

    def initial_state(
        self,
        pressure: ScalarInput,
        sw: ScalarInput,
        sg: ScalarInput,
    ) -> torch.Tensor:
        n, dtype, device = self.n_cells, self.V.dtype, self.V.device

        def col(v: ScalarInput) -> torch.Tensor:
            return (
                v.to(device=device, dtype=dtype)
                if isinstance(v, torch.Tensor)
                else torch.full((n,), float(v), dtype=dtype, device=device)
            )

        return torch.cat([col(pressure), col(sw), col(sg)])

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        return {"p": state[:n], "sw": state[n : 2 * n], "sg": state[2 * n : 3 * n]}

    def _rho(self, p: torch.Tensor, ref: float, c: float) -> torch.Tensor:
        return ref * (1.0 + c * (p - self.p_ref))

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: FlowSourceTerms | None = None,
    ) -> torch.Tensor:
        n = self.n_cells
        if state.shape != (3 * n,) or state_old.shape != (3 * n,):
            raise GeoBrainError(
                "MPFAThreePhaseModel state must be a 1D tensor of length 3*n_cells",
                object_name="MPFAThreePhaseModel",
                field="state",
                expected=(3 * n,),
                actual=tuple(state.shape),
            )
        p = state[:n]
        sw = state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        # Gas may be genuinely absent (no connate gas). Use the *raw* Sg for the
        # accumulation so its derivative stays alive at Sg=0: clamping to a floor
        # would zero the gas-equation diagonal wherever gas is absent (singular),
        # and even clamp(min=0) is fragile to round-off pushing Sg slightly < 0.
        # The relperm clamps its own normalized saturation, so mobilities stay
        # valid for the raw Sg.
        sg = state[2 * n : 3 * n]
        so = 1.0 - sw - sg
        p_old = state_old[:n]
        sw_old = state_old[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        sg_old = state_old[2 * n : 3 * n]
        so_old = 1.0 - sw_old - sg_old

        rho_w, rho_o, rho_g = (
            self._rho(p, self.rho_w_ref, self.c_w),
            self._rho(p, self.rho_o_ref, self.c_o),
            self._rho(p, self.rho_g_ref, self.c_g),
        )
        rho_w_o, rho_o_o, rho_g_o = (
            self._rho(p_old, self.rho_w_ref, self.c_w),
            self._rho(p_old, self.rho_o_ref, self.c_o),
            self._rho(p_old, self.rho_g_ref, self.c_g),
        )

        acc_w = self.V * self.phi * (rho_w * sw - rho_w_o * sw_old) / float(dt)
        acc_o = self.V * self.phi * (rho_o * so - rho_o_o * so_old) / float(dt)
        acc_g = self.V * self.phi * (rho_g * sg - rho_g_o * sg_old) / float(dt)

        phi_o = p
        phi_w = p - (self.pc_ow(sw) if self.pc_ow is not None else 0.0)
        phi_g = p + (self.pc_og(sg) if self.pc_og is not None else 0.0)
        if self.depth is not None:  # Φ_α = p − ρ_α g·D (D = depth, +down)
            gz = self.g * self.depth
            phi_o = phi_o - rho_o * gz
            phi_w = phi_w - rho_w * gz
            phi_g = phi_g - rho_g * gz

        G_w, G_o, G_g = self.L @ phi_w, self.L @ phi_o, self.L @ phi_g
        mm_w = rho_w * self.relperm.kr_water(sw) / self.mu_w
        mm_o = rho_o * self.relperm.kr_oil(sw, sg) / self.mu_o
        mm_g = rho_g * self.relperm.kr_gas(sg) / self.mu_g
        F_w = mm_w[upwind_cell(G_w, self.face_lr)] * G_w
        F_o = mm_o[upwind_cell(G_o, self.face_lr)] * G_o
        F_g = mm_g[upwind_cell(G_g, self.face_lr)] * G_g

        R_w = acc_w + scatter_internal_face_flux(F_w, self.face_lr, n)
        R_o = acc_o + scatter_internal_face_flux(F_o, self.face_lr, n)
        R_g = acc_g + scatter_internal_face_flux(F_g, self.face_lr, n)

        if self.nnc_trans is not None:  # fault NNCs (two-point, per phase)
            R_w = nnc_phase_flux(R_w, self.nnc_pairs, self.nnc_trans, phi_w, mm_w)
            R_o = nnc_phase_flux(R_o, self.nnc_pairs, self.nnc_trans, phi_o, mm_o)
            R_g = nnc_phase_flux(R_g, self.nnc_pairs, self.nnc_trans, phi_g, mm_g)

        R_w = R_w - source_block(sources, family="phase", name="water", like=R_w)
        R_o = R_o - source_block(sources, family="phase", name="oil", like=R_o)
        R_g = R_g - source_block(sources, family="phase", name="gas", like=R_g)
        return torch.cat([R_w, R_o, R_g])

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        **kw: FlowSourceTerms | None,
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(
            lambda s: self.residual(s, state_old, dt, **kw),
            state,
            vectorize=True,
        )


class MPFAThreePhaseModel3D(MPFAThreePhaseModel):
    """Oil-water-gas three-phase Darcy flow with a 3-D MPFA-O multi-point flux (SI).

    Args mirror :class:`MPFAThreePhaseModel`, with a :class:`MPFAGrid3D` and
    ``(n_cells, 3, 3)`` permeability tensors.
    """

    schema = _flow_model_schema(
        model_name="MPFAThreePhaseModel3D",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
            ("sg", "1", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
            ("gas_mass", "kg/s", "mass", "sg"),
        ),
        grid_kinds=("mpfa-3d",),
        phases=("water", "oil", "gas"),
        structured_sources=True,
    )

    def __init__(
        self,
        grid: MPFAGrid3D,
        perm_tensor: torch.Tensor,
        porosity: ScalarInput,
        relperm: ThreePhaseRelPerm,
        *,
        rho_w_ref: float = 1000.0,
        rho_o_ref: float = 800.0,
        rho_g_ref: float = 100.0,
        mu_w: float = 1e-3,
        mu_o: float = 2e-3,
        mu_g: float = 2e-5,
        c_w: float = 0.0,
        c_o: float = 0.0,
        c_g: float = 0.0,
        p_ref: float = 1e7,
        pc_ow: Callable[[torch.Tensor], torch.Tensor] | None = None,
        pc_og: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth: torch.Tensor | None = None,
        gravity: float = 9.81,
        cell_volumes: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_trans: torch.Tensor | None = None,
    ) -> None:
        _ModuleBase.__init__(self)
        self.grid = grid
        self.relperm = relperm
        self.pc_ow, self.pc_og = pc_ow, pc_og
        self.rho_w_ref, self.rho_o_ref, self.rho_g_ref = (
            float(rho_w_ref),
            float(rho_o_ref),
            float(rho_g_ref),
        )
        self.mu_w, self.mu_o, self.mu_g = float(mu_w), float(mu_o), float(mu_g)
        self.c_w, self.c_o, self.c_g = float(c_w), float(c_o), float(c_g)
        self.p_ref = float(p_ref)
        self.g = float(gravity)
        n = len(grid.cell_nodes)
        self.n_cells = n
        dtype = perm_tensor.dtype

        if isinstance(porosity, torch.Tensor):
            self.register_buffer("phi", porosity.to(dtype))
        else:
            self.register_buffer("phi", torch.full((n,), float(porosity), dtype=dtype))
        if cell_volumes is None:
            cell_volumes = hex_cell_volumes(grid).to(dtype)
        self.register_buffer("V", cell_volumes.to(dtype))
        if depth is not None:
            self.register_buffer("depth", depth.to(dtype))
        else:
            self.depth = None

        stencils = mpfa_o_face_flux_stencils_3d_full(grid, perm_tensor)
        faces = sorted(stencils)
        L = perm_tensor.new_zeros(len(faces), n)
        lr: list[list[int]] = []
        for fi, f in enumerate(faces):
            for c, t in stencils[f].items():
                L[fi, c] = t
            left, right = grid.face_cells[f]
            lr.append([left, right])
        self.register_buffer("L", L)
        self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long))
        register_nnc(self, nnc_pairs, nnc_trans, dtype)


__all__ = ["MPFAThreePhaseModel", "MPFAThreePhaseModel3D"]
