"""
Oil-water two-phase fully-implicit Darcy flow model.

State vector (per-variable block layout, length ``2 · n_cells``)::

    state = [ p_0,  p_1,  …, p_{n-1},  sw_0,  sw_1,  …, sw_{n-1} ]

The "fully implicit" qualifier means pressure and saturation are
solved jointly each Newton step. Strictly-IMPES (implicit pressure,
explicit saturation) is cheaper per step but is CFL-limited and
fights autograd through the sequential split, fully implicit lines
up with the rest of the autograd / sparse-Jacobian infrastructure
and supports adjoint sensitivities without seams.

Per-cell mass balance (one row per phase, units m³/s)::

    (V·φ·Sα/Bα)^{n+1} − (V·φ·Sα/Bα)^n
    ──────────────────────────────────  +  Σ_f F^α  −  q^α  +  Σ_bc q_bc^α  =  0
                dt

with phase flux ``F^α_f = mob_upw^α(f) · T_f · (p_l − p_r)``
and phase mobility ``mob^α = k_rα / (μ_α · Bα)``.

This revision omits gravity, capillary pressure, and wells; sources
are passed as ``source_water_rates`` / ``source_oil_rates`` tensors
(m³/s, positive = injection per phase), wells and Pc / gravity
land in FLOW-T6.

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
from .._defaults import S_MAX, S_MIN
from ..properties import OilWaterFluid
from ..grid import FlowGrid
from ..solvers import (
    JacobianSparsitySpec,
    compute_sparse_jacobian,
    make_sparsity_spec,
)
from ..properties import Rock


class OilWaterModel(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """
    Oil + water two-phase Darcy flow residual + Jacobian.

    Args:
        grid: any :class:`FlowGrid`.
        rock: :class:`Rock` with per-cell perm + porosity.
        fluid: :class:`OilWaterFluid` (PVT for both phases + relperm).
        boundaries: optional Dirichlet-type :class:`FlowBoundaryGroup`.
    """

    schema = _flow_model_schema(
        model_name="OilWaterModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_surface_volume", "m³/s", "surface-volume", "pressure"),
            ("oil_surface_volume", "m³/s", "surface-volume", "sw"),
        ),
        grid_kinds=("cartesian",),
        phases=("water", "oil"),
        unit_system="SI",
    )

    def __init__(
        self,
        grid: FlowGrid,
        rock: Rock,
        fluid: OilWaterFluid,
        boundaries: FlowBoundaryGroup | None = None,
        *,
        gravity: bool = False,
        capillary: bool = False,
    ) -> None:
        super().__init__()
        if grid.n_cells != int(rock.permeability_m2.shape[0]):
            raise GeoBrainError(
                "Rock.perm must have one entry per grid cell",
                object_name="OilWaterModel",
                field="rock.permeability_m2",
                expected=(grid.n_cells,),
                actual=tuple(rock.permeability_m2.shape),
            )
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.boundaries = boundaries if boundaries is not None else FlowBoundaryGroup([])
        self.gravity = bool(gravity)
        self.capillary = bool(capillary)
        self._sparsity_spec: JacobianSparsitySpec | None = None

    @property
    def n_cells(self) -> int:
        return int(self.grid.n_cells)

    def state_size(self) -> int:
        return 2 * self.n_cells

    def initial_state(
        self,
        pressure: float | torch.Tensor,
        sw: float | torch.Tensor,
    ) -> torch.Tensor:
        """Pack ``(p, sw)`` into the canonical state vector."""
        n = self.n_cells
        device, dtype = self.grid.device, self.grid.dtype
        if isinstance(pressure, torch.Tensor):
            p = pressure.to(device=device, dtype=dtype)
        else:
            p = torch.full((n,), float(pressure), device=device, dtype=dtype)
        if isinstance(sw, torch.Tensor):
            s = sw.to(device=device, dtype=dtype)
        else:
            s = torch.full((n,), float(sw), device=device, dtype=dtype)
        if p.shape != (n,) or s.shape != (n,):
            raise GeoBrainError(
                "initial_state(p, sw) needs (n_cells,) tensors",
                object_name="OilWaterModel",
                field="(pressure, sw)",
                expected=((n,), (n,)),
                actual=(tuple(p.shape), tuple(s.shape)),
            )
        return torch.cat([p, s])

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        return {"p": state[:n], "sw": state[n : 2 * n]}

    def accepted_discrete_masks(
        self,
        state: torch.Tensor,
    ) -> dict[str, tuple[bool, ...]]:
        """Return the converged per-phase TPFA upwind choices.

        A true entry means the canonical left cell of that internal face was
        selected.  The comparison exactly matches :func:`upwind_cell`, where a
        zero phase-potential drop deterministically selects the left cell.
        Returned Python booleans are detached and directly JSON serializable.
        """
        with torch.no_grad():
            n = self.n_cells
            p = state[:n].detach()
            sw = state[n : 2 * n].detach().clamp(min=S_MIN, max=S_MAX)
            connection = self.grid._connection_metrics()
            if connection is None or connection.n_faces == 0:
                return {"water_upwind_left": (), "oil_upwind_left": ()}
            left, right = connection.neighbors[:, 0], connection.neighbors[:, 1]
            pressure_drop = p[left] - p[right]
            water_potential_drop = pressure_drop
            oil_potential_drop = pressure_drop
            if self.capillary and self.fluid.capillary is not None:
                capillary_pressure = self.fluid.capillary(sw)
                water_potential_drop = water_potential_drop - (
                    capillary_pressure[left] - capillary_pressure[right]
                )
            if self.gravity:
                depth = self.grid._cell_centers_view()[:, 2]
                depth_drop = depth[left] - depth[right]
                water_density = self.fluid.density_water(p)
                oil_density = self.fluid.density_oil(p)
                water_potential_drop = water_potential_drop - 0.5 * (
                    water_density[left] + water_density[right]
                ) * G * depth_drop
                oil_potential_drop = oil_potential_drop - 0.5 * (
                    oil_density[left] + oil_density[right]
                ) * G * depth_drop
            return {
                "water_upwind_left": tuple(
                    bool(value) for value in (water_potential_drop >= 0)
                ),
                "oil_upwind_left": tuple(
                    bool(value) for value in (oil_potential_drop >= 0)
                ),
            }

    def convergence_inputs(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """``(per-phase residual, reservoir pore volume [RB], per-phase B)`` for the
        CNV/MB convergence criterion; phase order matches the residual ``[R_w, R_o]``."""
        n = self.n_cells
        p = state[:n]
        pv = self.grid._cell_volumes_view() * self.rock.porosity_at_pressure(p)
        return (
            [residual[:n], residual[n : 2 * n]],
            pv,
            [self.fluid.pvt_w.fvf(p), self.fluid.pvt_o.fvf(p)],
        )

    # ------------------------------------------------------------------
    # Sparse Jacobian (opt-in)
    # ------------------------------------------------------------------

    def enable_sparse_jacobian(
        self,
        extra_couplings: list[tuple[int, int]] | None = None,
    ) -> None:
        connection = self.grid._connection_metrics()
        if connection is None:
            raise GeoBrainError(
                "sparse Jacobian requires grid connections",
                object_name="OilWaterModel",
                field="grid.conn",
                expected="ConnList",
                actual=None,
            )
        self._sparsity_spec = make_sparsity_spec(
            cell_neighbors=connection.neighbors,
            n_cells=self.n_cells,
            n_vars=2,
            extra_couplings=extra_couplings,
        )

    def disable_sparse_jacobian(self) -> None:
        self._sparsity_spec = None

    def _transmissibility(self) -> torch.Tensor:
        """TPFA transmissibility with a grad-aware cache.

        ``build_transmissibility`` is a pure function of ``perm``, but the
        Newton FD-Jacobian probes call ``residual`` thousands of times per
        solve with an unchanged ``perm`` (measured: ~6% of the whole flow
        forward rebuilt here). Cache ONLY when autograd is disabled (the
        probe path) and key on the perm tensor's identity/version/device so
        any in-place edit or replacement invalidates. Grad-enabled calls
        (the implicit-FT adjoint's residual) always rebuild through the
        graph, exactly as :meth:`FlowGrid.build_transmissibility`'s
        no-self-caching contract demands.
        """
        perm = self.rock.permeability_m2
        if torch.is_grad_enabled() and (isinstance(perm, torch.Tensor) and perm.requires_grad):
            return self.grid.build_transmissibility(perm)
        key = (perm.data_ptr(), perm._version, perm.dtype, perm.device, tuple(perm.shape))
        cached = getattr(self, "_trans_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        T = self.grid.build_transmissibility(perm)
        self._trans_cache = (key, T)
        return T

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        source_water_rates: torch.Tensor | None = None,
        source_oil_rates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Two-phase cell-wise mass balance.

        Returns ``R[2·n_cells]`` = ``concat(R_w, R_o)`` in m³/s. Source
        rates are per-phase, per-cell (positive = injection). At the
        converged state ``|R|_∞ < NEWTON_TOL``.
        """
        n = self.n_cells
        expected = (2 * n,)
        if state.shape != expected or state_old.shape != expected:
            raise GeoBrainError(
                "OilWaterModel state must be a 1D tensor of length 2*n_cells",
                object_name="OilWaterModel",
                field="state",
                expected=expected,
                actual=tuple(state.shape),
            )

        p = state[:n]
        # Clamp ONLY the saturation that feeds saturation-dependent *properties*
        # (relperm, capillarity): values outside [S_MIN, S_MAX] are unphysical
        # there. The ACCUMULATION (pore mass ∝ φ·S) must use the RAW saturation so
        # its ∂(mass)/∂S never vanishes: a clamped accumulation zeroes the whole
        # saturation column of the Jacobian whenever a cell sits at the bound (an
        # immobile or fully-saturated phase, e.g. a connate / all-resident start),
        # making the Newton system singular even though the physics is fine.
        sw_raw = state[n : 2 * n]
        sw = sw_raw.clamp(min=S_MIN, max=S_MAX)
        p_old = state_old[:n]
        sw_old = state_old[n : 2 * n]
        so_old = 1.0 - sw_old

        V = self.grid._cell_volumes_view()
        Bw = self.fluid.fvf_water(p)
        Bw_old = self.fluid.fvf_water(p_old)
        Bo = self.fluid.fvf_oil(p)
        Bo_old = self.fluid.fvf_oil(p_old)
        mu_w = self.fluid.viscosity_water(p)
        mu_o = self.fluid.viscosity_oil(p)
        phi = self.rock.porosity_at_pressure(p)
        phi_old = self.rock.porosity_at_pressure(p_old)
        kr_w = self.fluid.kr_water(sw)
        kr_o = self.fluid.kr_oil(sw)

        # Accumulation per phase [m³/s]: uses the RAW saturation (see above)
        # so ∂(mass)/∂S stays nonzero even when a cell sits at a saturation bound.
        so_raw = 1.0 - sw_raw
        acc_w = V * (phi * sw_raw / Bw - phi_old * sw_old / Bw_old) / float(dt)
        acc_o = V * (phi * so_raw / Bo - phi_old * so_old / Bo_old) / float(dt)

        # Inter-cell flux per phase (upwinded TPFA)
        c = self.grid._connection_metrics()
        if c is None or c.n_faces == 0:
            R_w = acc_w
            R_o = acc_o
        else:
            T = self._transmissibility()
            T_alpha = T
            cl, cr = c.neighbors[:, 0], c.neighbors[:, 1]
            dp = p[cl] - p[cr]
            mob_w = kr_w / (mu_w * Bw)
            mob_o = kr_o / (mu_o * Bo)
            # Per-phase potential Φ_α = p_α − ρ_α·g·z. The state ``p`` is the oil
            # (reference) pressure; capillarity sets the water pressure
            # p_w = p − Pc_ow(S_w), and gravity adds the buoyancy head ρ·g·z
            # (z = cell depth, +down). The upwind direction is taken on each
            # phase's own potential (it can flip vs the pressure-only sign: how
            # a buoyant phase segregates and how capillarity imbibes).
            dphi_w = dp
            dphi_o = dp
            if self.capillary and self.fluid.capillary is not None:
                pc_ow = self.fluid.capillary(sw)  # p_o − p_w (Pa)
                dphi_w = dphi_w - (pc_ow[cl] - pc_ow[cr])
            if self.gravity:
                z = self.grid._cell_centers_view()[:, 2]
                dz = z[cl] - z[cr]
                rho_w = self.fluid.density_water(p)
                rho_o = self.fluid.density_oil(p)
                dphi_w = dphi_w - 0.5 * (rho_w[cl] + rho_w[cr]) * G * dz
                dphi_o = dphi_o - 0.5 * (rho_o[cl] + rho_o[cr]) * G * dz
            upw_w = mob_w[upwind_cell(dphi_w, c.neighbors)]
            upw_o = mob_o[upwind_cell(dphi_o, c.neighbors)]
            F_w = upw_w * T_alpha * dphi_w
            F_o = upw_o * T_alpha * dphi_o
            R_w = acc_w + scatter_internal_face_flux(F_w, c.neighbors, n)
            R_o = acc_o + scatter_internal_face_flux(F_o, c.neighbors, n)

        # Per-phase source rates (positive = injection of that phase)
        if source_water_rates is not None:
            if source_water_rates.shape != (n,):
                raise GeoBrainError(
                    "source_water_rates must have shape (n_cells,)",
                    object_name="OilWaterModel",
                    field="source_water_rates",
                    expected=(n,),
                    actual=tuple(source_water_rates.shape),
                )
            R_w = R_w - source_water_rates
        if source_oil_rates is not None:
            if source_oil_rates.shape != (n,):
                raise GeoBrainError(
                    "source_oil_rates must have shape (n_cells,)",
                    object_name="OilWaterModel",
                    field="source_oil_rates",
                    expected=(n,),
                    actual=tuple(source_oil_rates.shape),
                )
            R_o = R_o - source_oil_rates

        # Dirichlet-type boundary flux per phase
        boundary_cells: list[int] = []
        boundary_water_flux: list[torch.Tensor] = []
        boundary_oil_flux: list[torch.Tensor] = []
        for bc in self.boundaries.bcs:
            ci = bc.cell
            dp_bc = bc.outward_pressure_drop(p[ci])
            T_bc = bc.transmissibility
            # q_α > 0 when p_cell > p_bc: cell-to-exterior outflow.
            q_w = T_bc * kr_w[ci] / (mu_w[ci] * Bw[ci]) * dp_bc
            q_o = T_bc * kr_o[ci] / (mu_o[ci] * Bo[ci]) * dp_bc
            boundary_cells.append(ci)
            boundary_water_flux.append(q_w)
            boundary_oil_flux.append(q_o)
        if boundary_cells:
            cells = torch.tensor(boundary_cells, dtype=torch.int64, device=p.device)
            R_w = R_w + scatter_boundary_outflow(torch.stack(boundary_water_flux), cells, n)
            R_o = R_o + scatter_boundary_outflow(torch.stack(boundary_oil_flux), cells, n)

        return torch.cat([R_w, R_o])

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        source_water_rates: torch.Tensor | None = None,
        source_oil_rates: torch.Tensor | None = None,
        *,
        exact: bool = False,
    ) -> torch.Tensor:
        """Newton Jacobian: dense autograd by default, sparse colored
        FD via :meth:`enable_sparse_jacobian`."""

        def f(x: torch.Tensor) -> torch.Tensor:
            return self.residual(
                x,
                state_old,
                dt,
                source_water_rates=source_water_rates,
                source_oil_rates=source_oil_rates,
            )

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


__all__ = ["OilWaterModel"]
