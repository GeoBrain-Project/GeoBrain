"""
Single-phase compressible Darcy flow model.

Mass balance per cell (strict SI, in m³/s of surface volume):

    (V · φ · (1/B))^{n+1} − (V · φ · (1/B))^n
    ─────────────────────────────────────────  +  Σ_faces F  −  q_src  +  Σ_bc q_bc  =  0
                dt

where the inter-cell flux is the standard upwinded TPFA Darcy term::

    F_f = mob_upw(f) · T_f · (p_l − p_r),   mob = 1 / (μ · B)

Source terms (``source_rates``, m³/s, positive = injection into the
cell) and Dirichlet-type :class:`FlowBoundary` constraints are
optional. Wells are handled in FLOW-T6.

State vector: ``p`` (oil-phase pressure, one DOF per cell).

The :class:`SinglePhaseModel` is an ``nn.Module`` so that ``rock`` /
``fluid`` / ``boundary`` submodules participate in autograd (perm,
porosity, PVT coefficients and source rates can all be inverted).

A default ``jacobian`` implementation uses
``torch.autograd.functional.jacobian``: dense and slow but exact.
T3 will swap in a sparse colored-FD path; the interface here is
stable.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from ..discretization.boundary import FlowBoundaryGroup
from ..discretization.flux import (
    scatter_boundary_outflow,
    scatter_internal_face_flux,
    upwind_cell,
)
from geobrain.core.constants import STANDARD_GRAVITY as G
from ..properties import SinglePhaseFluid
from ..grid import FlowGrid
from ..solvers import (
    JacobianSparsitySpec,
    compute_sparse_jacobian,
    make_sparsity_spec,
)
from ..properties import Rock


class SinglePhaseModel(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """
    Single-phase compressible Darcy flow residual + Jacobian.

    Args:
        grid: any :class:`FlowGrid` (typically :class:`CartGrid`).
        rock: :class:`Rock` with per-cell perm + porosity.
        fluid: :class:`SinglePhaseFluid` bundling a PVT model.
        boundaries: optional :class:`FlowBoundaryGroup` for Dirichlet-type
            constant-pressure boundary cells.
    """

    schema = _flow_model_schema(
        model_name="SinglePhaseModel",
        primary_fields=(("pressure", "Pa", ("cell",), ()),),
        residual_blocks=(("fluid_surface_volume", "m³/s", "surface-volume", "pressure"),),
        grid_kinds=("cartesian",),
        phases=("fluid",),
        unit_system="SI",
    )

    def __init__(
        self,
        grid: FlowGrid,
        rock: Rock,
        fluid: SinglePhaseFluid,
        boundaries: FlowBoundaryGroup | None = None,
        *,
        gravity: bool = False,
    ) -> None:
        super().__init__()
        if grid.n_cells != int(rock.permeability_m2.shape[0]):
            raise GeoBrainError(
                "Rock.perm must have one entry per grid cell",
                object_name="SinglePhaseModel",
                field="rock.permeability_m2",
                expected=(grid.n_cells,),
                actual=tuple(rock.permeability_m2.shape),
            )
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.boundaries = boundaries if boundaries is not None else FlowBoundaryGroup([])
        self.gravity = bool(gravity)
        self._sparsity_spec: JacobianSparsitySpec | None = None

    @property
    def n_cells(self) -> int:
        return int(self.grid.n_cells)

    def state_size(self) -> int:
        return self.n_cells

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"p": state}

    def accepted_discrete_masks(
        self,
        state: torch.Tensor,
    ) -> dict[str, tuple[bool, ...]]:
        """Return the detached, JSON-ready TPFA fluid upwind choices."""
        with torch.no_grad():
            p = state.detach()
            connection = self.grid._connection_metrics()
            if connection is None or connection.n_faces == 0:
                return {"fluid_upwind_left": ()}
            left, right = connection.neighbors[:, 0], connection.neighbors[:, 1]
            potential_drop = p[left] - p[right]
            if self.gravity:
                depth = self.grid._cell_centers_view()[:, 2]
                density = self.fluid.density(p)
                potential_drop = potential_drop - 0.5 * (
                    density[left] + density[right]
                ) * G * (depth[left] - depth[right])
            return {
                "fluid_upwind_left": tuple(
                    bool(value) for value in (potential_drop >= 0)
                )
            }

    def convergence_inputs(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """``(per-phase residual, reservoir pore volume [RB], per-phase B)`` for the
        single-phase CNV/MB criterion (one phase: the whole residual)."""
        p = state
        pv = self.grid._cell_volumes_view() * self.rock.porosity_at_pressure(p)
        return ([residual], pv, [self.fluid.fvf(p)])

    def initial_state(self, p: float | torch.Tensor) -> torch.Tensor:
        if isinstance(p, torch.Tensor):
            return p.to(dtype=self.grid.dtype, device=self.grid.device)
        return torch.full(
            (self.n_cells,),
            float(p),
            dtype=self.grid.dtype,
            device=self.grid.device,
        )

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        source_rates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Cell-wise mass balance residual.

        Returns ``R[n_cells]`` in m³/s. At the converged ``p`` the
        residual is zero up to ``NEWTON_TOL`` (1e-3 by default).

        ``source_rates`` is an optional ``(n_cells,)`` tensor of
        per-cell injection rates in m³/s (positive = injection,
        negative = production). When ``None`` no source term is added.
        """
        if state.ndim != 1 or state.shape[0] != self.n_cells:
            raise GeoBrainError(
                "SinglePhaseModel state must be a 1D tensor of length n_cells",
                object_name="SinglePhaseModel",
                field="state",
                expected=(self.n_cells,),
                actual=tuple(state.shape),
            )
        p = state
        p_old = state_old

        V = self.grid._cell_volumes_view()
        phi = self.rock.porosity_at_pressure(p)
        phi_old = self.rock.porosity_at_pressure(p_old)
        B = self.fluid.fvf(p)
        B_old = self.fluid.fvf(p_old)
        mu = self.fluid.viscosity(p)

        # Accumulation [m³/s]
        acc = V * (phi / B - phi_old / B_old) / float(dt)

        # Inter-cell flux, upwinded
        c = self.grid._connection_metrics()
        if c is None or c.n_faces == 0:
            R = acc
        else:
            T = self.grid.build_transmissibility(self.rock.permeability_m2)
            T_alpha = T
            cl, cr = c.neighbors[:, 0], c.neighbors[:, 1]
            dp = p[cl] - p[cr]
            mob = 1.0 / (mu * B)
            # Potential difference Φ = p − ρ·g·z (z = cell depth, +down) when
            # gravity is enabled; upwind on the potential.
            if self.gravity:
                z = self.grid._cell_centers_view()[:, 2]
                rho = self.fluid.density(p)
                dphi = dp - 0.5 * (rho[cl] + rho[cr]) * G * (z[cl] - z[cr])
            else:
                dphi = dp
            mob_face = mob[upwind_cell(dphi, c.neighbors)]
            F = mob_face * T_alpha * dphi
            R = acc + scatter_internal_face_flux(F, c.neighbors, self.n_cells)

        # Source terms (positive = injection into the cell, m³/s)
        if source_rates is not None:
            if source_rates.shape != (self.n_cells,):
                raise GeoBrainError(
                    "source_rates must have shape (n_cells,)",
                    object_name="SinglePhaseModel",
                    field="source_rates",
                    expected=(self.n_cells,),
                    actual=tuple(source_rates.shape),
                )
            R = R - source_rates

        # Dirichlet-type boundary flux
        boundary_cells: list[int] = []
        boundary_fluxes: list[torch.Tensor] = []
        for bc in self.boundaries.bcs:
            ci = bc.cell
            q = bc.transmissibility * bc.outward_pressure_drop(p[ci]) / (mu[ci] * B[ci])
            boundary_cells.append(ci)
            boundary_fluxes.append(q)
        if boundary_fluxes:
            R = R + scatter_boundary_outflow(
                torch.stack(boundary_fluxes),
                torch.tensor(boundary_cells, dtype=torch.int64, device=p.device),
                self.n_cells,
            )

        return R

    # ------------------------------------------------------------------
    # Jacobian (default: dense autograd)
    # ------------------------------------------------------------------

    def enable_sparse_jacobian(
        self,
        extra_couplings: list[tuple[int, int]] | None = None,
    ) -> None:
        """
        Switch :meth:`jacobian` to the sparse colored-FD path.

        The sparsity pattern is built once from the TPFA stencil
        (plus any ``extra_couplings`` for wells / multi-perf sources)
        and cached on the model. Subsequent ``jacobian`` calls return
        a coalesced ``torch.sparse_coo_tensor``.
        """
        connection = self.grid._connection_metrics()
        if connection is None:
            raise GeoBrainError(
                "sparse Jacobian requires grid connections",
                object_name="SinglePhaseModel",
                field="grid.conn",
                expected="ConnList",
                actual=None,
            )
        self._sparsity_spec = make_sparsity_spec(
            cell_neighbors=connection.neighbors,
            n_cells=self.n_cells,
            n_vars=1,
            extra_couplings=extra_couplings,
        )

    def disable_sparse_jacobian(self) -> None:
        """Revert to dense autograd Jacobian (default)."""
        self._sparsity_spec = None

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        source_rates: torch.Tensor | None = None,
        *,
        exact: bool = False,
    ) -> torch.Tensor:
        """Newton Jacobian: dense autograd by default, sparse colored
        FD if :meth:`enable_sparse_jacobian` has been called.

        Dense path is exact (autograd) but O(n²) memory, fine for
        moderate grids (< ~10⁴ DOFs). The sparse path uses central FD
        with greedy distance-1 column coloring; one residual evaluation
        per color reconstructs the entire Jacobian.
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            return self.residual(x, state_old, dt, source_rates=source_rates)

        if self._sparsity_spec is None:
            return torch.autograd.functional.jacobian(
                f,
                state,
                create_graph=False,
                vectorize=True,
            )
        return compute_sparse_jacobian(
            f,
            state,
            self._sparsity_spec,
            mode="ad" if exact else "fd",
        )


__all__ = ["SinglePhaseModel"]
