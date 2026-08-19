"""
Black-oil three-phase Darcy flow model (dead-oil revision).

State vector (per-variable block layout, length ``3 · n_cells``)::

    state = [ p_0..p_{n-1},  sw_0..sw_{n-1},  sg_0..sg_{n-1} ]

Oil saturation is the closure ``S_o = 1 − S_w − S_g`` (softplus-smoothed
near zero so Newton steps that briefly drive ``S_o`` below zero don't
produce ``S_o = 0`` cliffs in the Jacobian).

Per-cell mass balance, one row per phase, units m³/s for the
liquid phases and m³/s for the gas::

    (V·φ·S_α/B_α)^{n+1} − (V·φ·S_α/B_α)^n
    ──────────────────────────────────────  +  Σ_f F^α  −  q^α  =  0
                dt

with phase flux ``F^α_f = mob_upw^α(f) · T_f · (p_l − p_r)``
and phase mobility ``mob^α = k_rα / (μ_α · B_α)``. Capillary pressure
is ignored (single-pressure black-oil); a future revision will route
``Pc_ow(S_w)`` and ``Pc_og(S_g)`` into the upwinding.

This revision is **dead-oil**: dissolved gas (Rs) is not part of the
state vector. Wells and live-oil PVT land in FLOW-T6/T7.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from ..properties import BlackOilFluid
from ..grid import FlowGrid
from ..solvers import (
    JacobianSparsitySpec,
    compute_sparse_jacobian,
    make_sparsity_spec,
)
from ..properties import Rock


class BlackOilModel(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """
    Dead-oil three-phase fully-implicit Darcy flow model.

    Args:
        grid: any :class:`FlowGrid`.
        rock: :class:`Rock` with per-cell perm + porosity.
        fluid: :class:`BlackOilFluid` bundling 3 PVT models + a 3-phase
            :class:`ThreePhaseRelPerm`.
        boundaries: optional Dirichlet-type :class:`FlowBoundaryGroup`.
    """

    schema = _flow_model_schema(
        model_name="BlackOilModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
            ("sg", "1", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_surface_volume", "m³/s", "surface-volume", "pressure"),
            ("oil_surface_volume", "m³/s", "surface-volume", "sw"),
            ("gas_standard_volume", "m³/s", "surface-volume", "sg"),
        ),
        grid_kinds=("cartesian",),
        phases=("water", "oil", "gas"),
        unit_system="SI",
    )

    def __init__(
        self,
        grid: FlowGrid,
        rock: Rock,
        fluid: BlackOilFluid,
        boundaries: FlowBoundaryGroup | None = None,
        *,
        gravity: bool = False,
        capillary: bool = False,
    ) -> None:
        super().__init__()
        if grid.n_cells != int(rock.permeability_m2.shape[0]):
            raise GeoBrainError(
                "Rock.perm must have one entry per grid cell",
                object_name="BlackOilModel",
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
        return 3 * self.n_cells

    def initial_state(
        self,
        pressure: float | torch.Tensor,
        sw: float | torch.Tensor,
        sg: float | torch.Tensor,
    ) -> torch.Tensor:
        """Pack ``(p, sw, sg)`` into the canonical state vector."""
        n = self.n_cells
        device, dtype = self.grid.device, self.grid.dtype

        def _broadcast(v: float | torch.Tensor, name: str) -> torch.Tensor:
            if isinstance(v, torch.Tensor):
                t = v.to(device=device, dtype=dtype)
            else:
                t = torch.full((n,), float(v), device=device, dtype=dtype)
            if t.shape != (n,):
                raise GeoBrainError(
                    f"initial_state expects ``{name}`` shape (n_cells,)",
                    object_name="BlackOilModel",
                    field=name,
                    expected=(n,),
                    actual=tuple(t.shape),
                )
            return t

        return torch.cat(
            [
                _broadcast(pressure, "pressure"),
                _broadcast(sw, "sw"),
                _broadcast(sg, "sg"),
            ]
        )

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        sw = state[n : 2 * n]
        sg = state[2 * n : 3 * n]
        so = 1.0 - sw - sg
        return {"p": state[:n], "sw": sw, "sg": sg, "so": so}

    def accepted_discrete_masks(
        self,
        state: torch.Tensor,
    ) -> dict[str, tuple[bool, ...]]:
        """Return detached, JSON-ready per-phase TPFA upwind choices."""
        with torch.no_grad():
            n = self.n_cells
            p = state[:n].detach()
            sw = state[n : 2 * n].detach().clamp(min=S_MIN, max=S_MAX)
            sg = state[2 * n : 3 * n].detach().clamp(min=S_MIN, max=S_MAX)
            connection = self.grid._connection_metrics()
            empty: dict[str, tuple[bool, ...]] = {
                "water_upwind_left": (),
                "oil_upwind_left": (),
                "gas_upwind_left": (),
            }
            if connection is None or connection.n_faces == 0:
                return empty
            left, right = connection.neighbors[:, 0], connection.neighbors[:, 1]
            pressure_drop = p[left] - p[right]
            water_potential_drop = pressure_drop
            oil_potential_drop = pressure_drop
            gas_potential_drop = pressure_drop
            if self.capillary:
                if self.fluid.capillary_ow is not None:
                    oil_water_capillary = self.fluid.capillary_ow(sw)
                    water_potential_drop = water_potential_drop - (
                        oil_water_capillary[left] - oil_water_capillary[right]
                    )
                if self.fluid.capillary_og is not None:
                    oil_gas_capillary = self.fluid.capillary_og(sg)
                    gas_potential_drop = gas_potential_drop + (
                        oil_gas_capillary[left] - oil_gas_capillary[right]
                    )
            if self.gravity:
                depth = self.grid._cell_centers_view()[:, 2]
                depth_drop = depth[left] - depth[right]
                for density, name in (
                    (self.fluid.pvt_w.density(p), "water"),
                    (self.fluid.pvt_o.density(p), "oil"),
                    (self.fluid.pvt_g.density(p), "gas"),
                ):
                    gravity_drop = 0.5 * (
                        density[left] + density[right]
                    ) * G * depth_drop
                    if name == "water":
                        water_potential_drop = water_potential_drop - gravity_drop
                    elif name == "oil":
                        oil_potential_drop = oil_potential_drop - gravity_drop
                    else:
                        gas_potential_drop = gas_potential_drop - gravity_drop
            return {
                "water_upwind_left": tuple(
                    bool(value) for value in (water_potential_drop >= 0)
                ),
                "oil_upwind_left": tuple(
                    bool(value) for value in (oil_potential_drop >= 0)
                ),
                "gas_upwind_left": tuple(
                    bool(value) for value in (gas_potential_drop >= 0)
                ),
            }

    def convergence_inputs(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """``(per-phase residual, reservoir pore volume [RB], per-phase B)`` for the
        CNV/MB criterion; phase order matches the residual ``[R_w, R_o, R_g]``."""
        n = self.n_cells
        p = state[:n]
        pv = self.grid._cell_volumes_view() * self.rock.porosity_at_pressure(p)
        return (
            [residual[:n], residual[n : 2 * n], residual[2 * n : 3 * n]],
            pv,
            [self.fluid.pvt_w.fvf(p), self.fluid.pvt_o.fvf(p), self.fluid.pvt_g.fvf(p)],
        )

    # ------------------------------------------------------------------

    def enable_sparse_jacobian(
        self,
        extra_couplings: list[tuple[int, int]] | None = None,
    ) -> None:
        connection = self.grid._connection_metrics()
        if connection is None:
            raise GeoBrainError(
                "sparse Jacobian requires grid connections",
                object_name="BlackOilModel",
                field="grid.conn",
                expected="ConnList",
                actual=None,
            )
        self._sparsity_spec = make_sparsity_spec(
            cell_neighbors=connection.neighbors,
            n_cells=self.n_cells,
            n_vars=3,
            extra_couplings=extra_couplings,
        )

    def disable_sparse_jacobian(self) -> None:
        self._sparsity_spec = None

    # ------------------------------------------------------------------

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        source_water_rates: torch.Tensor | None = None,
        source_oil_rates: torch.Tensor | None = None,
        source_gas_rates: torch.Tensor | None = None,
        sg_turn: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Three-phase cell-wise mass balance.

        Returns ``R[3·n_cells] = concat(R_w, R_o, R_g)``. Source rates
        are per-phase, per-cell (positive = injection of that phase).
        Liquid sources in m³/s, gas source in m³/s.

        ``sg_turn`` is the per-cell turning point (running-max gas saturation)
        for relperm hysteresis, forwarded to a hysteretic ``relperm.kr_gas``.
        The transient driver tracks it as ``max(sg_turn, S_g)`` across accepted
        steps; ``None`` (default) ⇒ the bounding drainage curve (no hysteresis).
        """
        n = self.n_cells
        expected = (3 * n,)
        if state.shape != expected or state_old.shape != expected:
            raise GeoBrainError(
                "BlackOilModel state must be 1D of length 3*n_cells",
                object_name="BlackOilModel",
                field="state",
                expected=expected,
                actual=tuple(state.shape),
            )

        p = state[:n]
        sw = state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        sg = state[2 * n : 3 * n].clamp(min=S_MIN, max=S_MAX)
        # Softplus-smoothed oil saturation closure to keep Jacobian smooth
        # when Newton steps overshoot So < 0.
        so_raw = 1.0 - sw - sg
        so = S_MIN + F.softplus(so_raw - S_MIN, beta=200.0)

        p_old = state_old[:n]
        # Apply the same saturation clamps to the old state so that
        # ``state == state_old`` produces ``R = 0`` identically (otherwise
        # an entry like sg=0 in state_old vs sg=S_MIN in the clamped
        # current state leaks a 1e-6 scale residual into the gas row).
        sw_old = state_old[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        sg_old = state_old[2 * n : 3 * n].clamp(min=S_MIN, max=S_MAX)
        so_old = (1.0 - sw_old - sg_old).clamp(min=S_MIN, max=S_MAX)

        V = self.grid._cell_volumes_view()
        Bw = self.fluid.pvt_w.fvf(p)
        Bo = self.fluid.pvt_o.fvf(p)
        Bg = self.fluid.pvt_g.fvf(p)
        Bw_old = self.fluid.pvt_w.fvf(p_old)
        Bo_old = self.fluid.pvt_o.fvf(p_old)
        Bg_old = self.fluid.pvt_g.fvf(p_old)
        mu_w = self.fluid.pvt_w.viscosity(p)
        mu_o = self.fluid.pvt_o.viscosity(p)
        mu_g = self.fluid.pvt_g.viscosity(p)
        # Solution gas-oil ratio R_s(p) [standard m³/m³]. Zero for a dead-oil PVT, so the
        # dissolved-gas terms below vanish and dead-oil behaviour is exact; a
        # live-oil PVT (PVTLiveOil) makes them carry dissolved gas into the gas
        # balance (accumulation + flux + boundary).
        Rs = self.fluid.pvt_o.rs(p)
        Rs_old = self.fluid.pvt_o.rs(p_old)
        phi = self.rock.porosity_at_pressure(p)
        phi_old = self.rock.porosity_at_pressure(p_old)
        kr_w = self.fluid.relperm.kr_water(sw, sg)
        kr_o = self.fluid.relperm.kr_oil(sw, sg)
        kr_g = (
            self.fluid.relperm.kr_gas(sg, sg_turn)
            if sg_turn is not None
            else self.fluid.relperm.kr_gas(sg)
        )

        # Accumulation per phase (m³/s for w/o, m³/s for g). The gas
        # component is free gas S_g/B_g PLUS gas dissolved in the oil R_s·S_o/B_o.
        acc_w = V * (phi * sw / Bw - phi_old * sw_old / Bw_old) / float(dt)
        acc_o = V * (phi * so / Bo - phi_old * so_old / Bo_old) / float(dt)
        acc_g = (
            V
            * (
                phi * (sg / Bg + Rs * so / Bo)
                - phi_old * (sg_old / Bg_old + Rs_old * so_old / Bo_old)
            )
            / float(dt)
        )

        # Inter-cell flux per phase
        c = self.grid._connection_metrics()
        if c is None or c.n_faces == 0:
            R_w = acc_w
            R_o = acc_o
            R_g = acc_g
        else:
            T = self.grid.build_transmissibility(self.rock.permeability_m2)
            T_alpha = T
            cl, cr = c.neighbors[:, 0], c.neighbors[:, 1]
            dp = p[cl] - p[cr]
            mob_w = kr_w / (mu_w * Bw)
            mob_o = kr_o / (mu_o * Bo)
            mob_g = kr_g / (mu_g * Bg)
            # Per-phase potential Φ_α = p_α − ρ_α·g·z. ``p`` is the oil (reference)
            # pressure; capillarity sets p_w = p − Pc_ow(S_w) and p_g = p + Pc_og(S_g);
            # gravity adds the buoyancy head ρ·g·z (z = cell depth, +down). Upwind
            # on each phase's own potential so gas segregates up while brine sinks
            # and capillarity imbibes.
            dphi_w = dphi_o = dphi_g = dp
            if self.capillary:
                if self.fluid.capillary_ow is not None:
                    pc_ow = self.fluid.capillary_ow(sw)  # p_o − p_w
                    dphi_w = dphi_w - (pc_ow[cl] - pc_ow[cr])
                if self.fluid.capillary_og is not None:
                    pc_og = self.fluid.capillary_og(sg)  # p_g − p_o
                    dphi_g = dphi_g + (pc_og[cl] - pc_og[cr])
            if self.gravity:
                z = self.grid._cell_centers_view()[:, 2]
                dz = z[cl] - z[cr]
                rho_w = self.fluid.pvt_w.density(p)
                rho_o = self.fluid.pvt_o.density(p)
                rho_g = self.fluid.pvt_g.density(p)
                dphi_w = dphi_w - 0.5 * (rho_w[cl] + rho_w[cr]) * G * dz
                dphi_o = dphi_o - 0.5 * (rho_o[cl] + rho_o[cr]) * G * dz
                dphi_g = dphi_g - 0.5 * (rho_g[cl] + rho_g[cr]) * G * dz
            water_upwind = upwind_cell(dphi_w, c.neighbors)
            oil_upwind = upwind_cell(dphi_o, c.neighbors)
            gas_upwind = upwind_cell(dphi_g, c.neighbors)
            upw_w = mob_w[water_upwind]
            upw_o = mob_o[oil_upwind]
            upw_g = mob_g[gas_upwind]
            F_w = upw_w * T_alpha * dphi_w
            F_o = upw_o * T_alpha * dphi_o
            F_g = upw_g * T_alpha * dphi_g
            # Dissolved gas rides with the oil flux, carrying R_s of the
            # upstream oil cell (same upwind direction as the oil phase).
            Rs_face = Rs[oil_upwind]
            F_g = F_g + Rs_face * F_o
            R_w = acc_w + scatter_internal_face_flux(F_w, c.neighbors, n)
            R_o = acc_o + scatter_internal_face_flux(F_o, c.neighbors, n)
            R_g = acc_g + scatter_internal_face_flux(F_g, c.neighbors, n)

        # Per-phase sources (positive = injection)
        for tensor, name, R in (
            (source_water_rates, "source_water_rates", "R_w"),
            (source_oil_rates, "source_oil_rates", "R_o"),
            (source_gas_rates, "source_gas_rates", "R_g"),
        ):
            if tensor is not None and tensor.shape != (n,):
                raise GeoBrainError(
                    f"{name} must have shape (n_cells,)",
                    object_name="BlackOilModel",
                    field=name,
                    expected=(n,),
                    actual=tuple(tensor.shape),
                )
        if source_water_rates is not None:
            R_w = R_w - source_water_rates
        if source_oil_rates is not None:
            R_o = R_o - source_oil_rates
            # Dissolved gas rides with the source oil rate (R_s·oil), exactly as
            # the boundary term below adds R_s·q_o to the gas balance.
            # ``source_gas_rates`` is the *free*-gas source only; without this a
            # producer would withdraw oil yet leave its dissolved gas behind,
            # spuriously super-saturating the cell and liberating gas above p_b.
            R_g = R_g - Rs * source_oil_rates
        if source_gas_rates is not None:
            R_g = R_g - source_gas_rates

        # Dirichlet-type boundary flux per phase
        boundary_cells: list[int] = []
        boundary_water_flux: list[torch.Tensor] = []
        boundary_oil_flux: list[torch.Tensor] = []
        boundary_gas_flux: list[torch.Tensor] = []
        for bc in self.boundaries.bcs:
            ci = bc.cell
            dp_bc = bc.outward_pressure_drop(p[ci])
            T_bc = bc.transmissibility
            # q_α > 0 when p_cell > p_bc: cell-to-exterior outflow.
            q_w = T_bc * kr_w[ci] / (mu_w[ci] * Bw[ci]) * dp_bc
            q_o = T_bc * kr_o[ci] / (mu_o[ci] * Bo[ci]) * dp_bc
            q_g = T_bc * kr_g[ci] / (mu_g[ci] * Bg[ci]) * dp_bc
            q_g = q_g + Rs[ci] * q_o  # dissolved gas leaves/enters with the oil
            boundary_cells.append(ci)
            boundary_water_flux.append(q_w)
            boundary_oil_flux.append(q_o)
            boundary_gas_flux.append(q_g)
        if boundary_cells:
            cells = torch.tensor(boundary_cells, dtype=torch.int64, device=p.device)
            R_w = R_w + scatter_boundary_outflow(torch.stack(boundary_water_flux), cells, n)
            R_o = R_o + scatter_boundary_outflow(torch.stack(boundary_oil_flux), cells, n)
            R_g = R_g + scatter_boundary_outflow(torch.stack(boundary_gas_flux), cells, n)

        return torch.cat([R_w, R_o, R_g])

    # ------------------------------------------------------------------

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        source_water_rates: torch.Tensor | None = None,
        source_oil_rates: torch.Tensor | None = None,
        source_gas_rates: torch.Tensor | None = None,
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
                source_gas_rates=source_gas_rates,
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


__all__ = ["BlackOilModel"]
