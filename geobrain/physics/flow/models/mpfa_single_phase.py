"""
Single-phase flow on a general 2-D grid using the MPFA-O multi-point flux.

The two-point models assemble each face flux from just its two neighbours,
``F = mob·T·(p_L − p_R)``: inconsistent on non-K-orthogonal / full-tensor grids.
This model instead assembles the **MPFA-O** flux: each interior face carries a
multi-cell stencil ``{cell: T_c}`` (from :func:`mpfa_o_face_flux_stencils`) and
the face flux is ``F = mob·Σ_c T_c·p_c``, scattered to the face's two neighbours.

Single-phase mass balance on cell ``i``::

    R_i = V_i·(φρ − φρ|old)/Δt + Σ_faces (±ρ_up·μ⁻¹·Σ_c T_c·p_c) − q_i

The MPFA stencils are precomputed from geometry + permeability (the residual is
then differentiable in pressure; recompute the stencils to differentiate in
permeability). Because MPFA reproduces a linear pressure field's flux exactly,
an interior cell whose faces are all fully-assembled has **zero net flux** for a
linear field on any grid, the patch test the tests check (TPFA fails it on
skewed grids). Units are SI.

v1 scope: interior MPFA faces only (boundary half-edges / Dirichlet-MPFA and
multiphase upwinding of the multi-point flux are the natural extensions).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, SupportsFloat, TypeAlias

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from ..discretization.flux import scatter_internal_face_flux, upwind_cell
from ..discretization.mpfa import MPFAGrid2D, _stable_polygon_areas, mpfa_o_face_flux_stencils
from ..wells import FlowSourceTerms, source_block

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


class MPFASinglePhaseModel(_ModuleBase):
    """Single-phase Darcy flow with an MPFA-O multi-point flux (2-D, SI units).

    Args:
        grid: :class:`MPFAGrid2D`.
        perm_tensor: ``(n_cells, 2, 2)`` cell permeability tensors [m²].
        porosity: scalar or ``(n_cells,)``.
        rho_ref, mu, p_ref, fluid_compressibility: single-phase fluid (ρ(p)).
    """

    schema = _flow_model_schema(
        model_name="MPFASinglePhaseModel",
        primary_fields=(("pressure", "Pa", ("cell",), ()),),
        residual_blocks=(("fluid_mass", "kg/s", "mass", "pressure"),),
        grid_kinds=("mpfa-2d",),
        phases=("fluid",),
        structured_sources=True,
    )
    phi: torch.Tensor
    V: torch.Tensor
    L: torch.Tensor
    face_lr: torch.Tensor

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: ScalarInput,
        *,
        rho_ref: float = 1000.0,
        mu: float = 1e-3,
        p_ref: float = 1e7,
        fluid_compressibility: float = 1e-9,
        cell_volumes: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.mu = float(mu)
        self.rho_ref = float(rho_ref)
        self.p_ref = float(p_ref)
        self.c_f = float(fluid_compressibility)
        n = len(grid.cell_nodes)
        self.n_cells = n
        dtype = perm_tensor.dtype
        if isinstance(porosity, torch.Tensor):
            self.register_buffer("phi", porosity.to(dtype))
        else:
            self.register_buffer("phi", torch.full((n,), float(porosity), dtype=dtype))
        if cell_volumes is None:
            cell_volumes = self._polygon_areas(grid)  # 2-D: area·1
        self.register_buffer("V", cell_volumes.to(dtype))

        # Precompute the MPFA face stencils as a (n_faces, n_cells) weight matrix
        # plus the (left, right) neighbour of each assembled face.
        stencils = mpfa_o_face_flux_stencils(grid, perm_tensor)
        faces = sorted(stencils)
        L = perm_tensor.new_zeros(len(faces), n)
        lr: list[list[int]] = []
        for fi, f in enumerate(faces):
            for c, t in stencils[f].items():
                L[fi, c] = t
            left, right = grid.edge_cells[f]
            lr.append([left, right])
        self.register_buffer("L", L)  # face pressure-flux stencil
        self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long))

    @staticmethod
    def _polygon_areas(grid: MPFAGrid2D) -> torch.Tensor:
        return _stable_polygon_areas(grid)

    def state_size(self) -> int:
        return self.n_cells

    def initial_state(self, pressure: ScalarInput) -> torch.Tensor:
        if isinstance(pressure, torch.Tensor):
            return pressure.to(self.V.dtype)
        return torch.full((self.n_cells,), float(pressure), dtype=self.V.dtype)

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"p": state}

    def density(self, p: torch.Tensor) -> torch.Tensor:
        return self.rho_ref * (1.0 + self.c_f * (p - self.p_ref))

    def face_fluxes(self, p: torch.Tensor) -> torch.Tensor:
        """MPFA mass flux per assembled face (left→right), ``ρ_up·μ⁻¹·Σ_c T_c·p_c``."""
        pflux = self.L @ p  # Σ_c T_c·p_c per face
        rho = self.density(p)
        rho_up = rho[upwind_cell(pflux, self.face_lr)]
        return rho_up / self.mu * pflux

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float | torch.Tensor,
        sources: FlowSourceTerms | None = None,
    ) -> torch.Tensor:
        n = self.n_cells
        if state.shape != (n,) or state_old.shape != (n,):
            raise GeoBrainError(
                "MPFASinglePhaseModel state must be (n_cells,)",
                object_name="MPFASinglePhaseModel",
                field="state",
                expected=(n,),
                actual=tuple(state.shape),
            )
        p, p_o = state, state_old
        acc = self.V * (self.phi * self.density(p) - self.phi * self.density(p_o)) / float(dt)
        R = acc
        F = self.face_fluxes(p)
        R = R + scatter_internal_face_flux(F, self.face_lr, n)
        R = R - source_block(sources, family="phase", name="fluid", like=R)
        return R

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float | torch.Tensor,
        **kw: FlowSourceTerms | None,
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(
            lambda s: self.residual(s, state_old, dt, **kw),
            state,
            vectorize=True,
        )


__all__ = ["MPFASinglePhaseModel"]
