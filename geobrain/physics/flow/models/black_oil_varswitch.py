"""
Variable-substitution black-oil model (Coats 1980) with a bubble-point crossing.

The standard :class:`~geobrain.physics.flow.models.black_oil.BlackOilModel` pins
the oil to its *saturated* solution gas-oil ratio ``R_s,sat(p)`` in every cell,
correct for a saturated reservoir, but it cannot represent **undersaturated**
oil (pressure above the bubble point, no free gas, ``R_s`` fixed below
saturation). This model adds variable switching (Coats variable substitution):
the third primary unknown ``X`` per cell switches
between phase states

    * **OilAndGas** (saturated):    ``X = S_g``, ``R_s = R_s,sat(p)`` (free gas present);
    * **OilOnly**  (undersaturated): ``X = R_s < R_s,sat(p)``, ``S_g = 0``.

The switch happens inside the Newton update: free gas that would go negative
disappears (OilAndGas → OilOnly, ``X`` becomes ``R_s,sat(p)``), and dissolved gas
that would exceed saturation liberates a free-gas phase (OilOnly → OilAndGas,
``X`` becomes a small ``S_g``). The oil PVT is evaluated at the *actual* ``R_s``
(``B_o(p, R_s)`` compresses undersaturated oil above its own bubble point), so the
density / swelling coupling is right on both sides of the bubble point.

Gas component balance per cell (free + dissolved), as in the saturated model::

    acc_g = V·(φ·(S_g/B_g + R_s·S_o/B_o) − old) / (dt·β)
    F_g   = F_g,free + R_s,upwind_oil · F_o

This is the disgas formulation (water + oil + dissolved gas). Vaporized oil
(``R_v`` / wet gas, the GasOnly state) is not modelled here.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from geobrain.core.constants import STANDARD_GRAVITY as G
from .._defaults import S_MAX, S_MIN
from ..discretization.boundary import FlowBoundaryGroup
from ..discretization.flux import (
    scatter_boundary_outflow,
    scatter_internal_face_flux,
    upwind_cell,
)
from ..grid import FlowGrid
from ..properties import Rock


class _PVTLike(Protocol):
    def density(self, pressure: torch.Tensor) -> torch.Tensor: ...

    def viscosity(self, pressure: torch.Tensor) -> torch.Tensor: ...

    def fvf(self, pressure: torch.Tensor) -> torch.Tensor: ...


class _LiveOilPVTLike(_PVTLike, Protocol):
    def density(
        self,
        pressure: torch.Tensor,
        rs: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def viscosity(
        self,
        pressure: torch.Tensor,
        rs: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def fvf(
        self,
        pressure: torch.Tensor,
        rs: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def rs(self, pressure: torch.Tensor) -> torch.Tensor: ...


class _ThreePhaseRelPermLike(Protocol):
    def kr_water(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def kr_oil(
        self,
        sw: torch.Tensor,
        sg: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def kr_gas(self, sg: torch.Tensor) -> torch.Tensor: ...


class _VarSwitchFluid(Protocol):
    @property
    def pvt_o(self) -> _LiveOilPVTLike: ...

    @property
    def pvt_w(self) -> _PVTLike: ...

    @property
    def pvt_g(self) -> _PVTLike: ...

    @property
    def relperm(self) -> _ThreePhaseRelPermLike: ...


class _NewtonControls(Protocol):
    max_iter: int
    tol: float


class BlackOilVarSwitchModel(nn.Module):  # type: ignore[misc]  # skipped torch boundary
    """Black-oil with Coats variable substitution (S_g ↔ R_s) for undersaturated oil.

    Args:
        grid:   any :class:`FlowGrid`.
        rock:   :class:`Rock` (per-cell perm + porosity).
        fluid:  a black-oil fluid whose ``pvt_o`` is a
            :class:`~geobrain.physics.flow.properties.PVTLiveOil` (needs ``rs``,
            ``fvf(p, rs)``, ``density(p, rs)``, ``bubble_pressure``).
        boundaries: optional constant-pressure :class:`FlowBoundaryGroup`.
        gravity:    include the buoyancy head in the phase potentials.
        rs_eps:     small offset used when switching at the bubble point.

    .. note::
       **ModelState / operator-path limitation (undersaturated inversion).**
       The flat state is ``[p, S_w, X]`` where the third unknown ``X`` carries
       either ``S_g`` (saturated) or ``R_s`` (undersaturated) per cell. When this
       model is driven through :class:`~geobrain.physics.flow.FlowEvolutionOperator`
       (the Tier-1 ``ModelState`` ``pack_state`` / ``unpack_state`` path), the
       operator-facing state variables are ``(pressure, sw, sg)``; ``R_s`` is
       **not** a ``ModelState`` field (it is not in the ``SPLIT_TO_VARIABLE``
       vocabulary). A ``pack → unpack`` round-trip therefore re-derives ``X`` from
       ``(sw, sg)`` at the *saturated* ``R_s,sat(p)`` and cannot recover the true
       per-cell undersaturated ``R_s``. Consequence: **saturated** forward/inverse
       through the operator path is exact, but **undersaturated** inversion via the
       Tier-1 operator path silently loses the undersaturation degree. For
       undersaturated work drive the model's own ``residual`` / forward-only
       transient path (which keeps ``R_s`` in the flat state ``X``), or extend the
       ``ModelState`` vocabulary with ``rs`` to make it a first-class variable.
    """

    schema = _flow_model_schema(
        model_name="BlackOilVarSwitchModel",
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
        fluid: _VarSwitchFluid,
        boundaries: FlowBoundaryGroup | None = None,
        *,
        gravity: bool = False,
        rs_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        if grid.n_cells != int(rock.permeability_m2.shape[0]):
            raise GeoBrainError(
                "Rock.perm must have one entry per grid cell",
                object_name="BlackOilVarSwitchModel",
                field="rock.permeability_m2",
                expected=(grid.n_cells,),
                actual=tuple(rock.permeability_m2.shape),
            )
        for name in ("rs", "fvf", "bubble_pressure"):
            if not hasattr(fluid.pvt_o, name):
                raise GeoBrainError(
                    "BlackOilVarSwitchModel needs a live-oil pvt_o (PVTLiveOil) "
                    f"exposing {name}(); got {type(fluid.pvt_o).__name__}.",
                    object_name="BlackOilVarSwitchModel",
                    field="fluid.pvt_o",
                    expected="PVTLiveOil",
                    actual=type(fluid.pvt_o).__name__,
                )
        self.grid = grid
        self.rock = rock
        self.fluid = fluid
        self.boundaries = boundaries if boundaries is not None else FlowBoundaryGroup([])
        self.gravity = bool(gravity)
        self.rs_eps = float(rs_eps)
        n = grid.n_cells
        # Phase-state masks: 1 = saturated (OilAndGas, X=Sg), 0 = undersaturated (OilOnly, X=Rs).
        self.register_buffer("sat", torch.ones(n, dtype=torch.bool, device=grid.device))
        self.register_buffer("sat_old", torch.ones(n, dtype=torch.bool, device=grid.device))

    @property
    def n_cells(self) -> int:
        return int(self.grid.n_cells)

    def state_size(self) -> int:
        return 3 * self.n_cells

    # ------------------------------------------------------------------
    # State packing / interpretation
    # ------------------------------------------------------------------
    def initial_state(
        self,
        pressure: float | torch.Tensor,
        sw: float | torch.Tensor,
        sg: float | torch.Tensor,
        rs: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pack ``(p, S_w, X)`` and set the phase-state mask from the initial
        free-gas saturation: cells with ``S_g > 0`` start **saturated**
        (``X = S_g``), cells with ``S_g = 0`` start **undersaturated**
        (``X = R_s``; defaults to the saturated ``R_s,sat(p)`` if ``rs`` omitted,
        i.e. exactly at the bubble point)."""
        n = self.n_cells
        dev, dt = self.grid.device, self.grid.dtype

        def _vec(v: float | torch.Tensor) -> torch.Tensor:
            if isinstance(v, torch.Tensor):
                return v.to(device=dev, dtype=dt)
            return torch.full((n,), float(v), device=dev, dtype=dt)

        p = _vec(pressure)
        s_w = _vec(sw)
        s_g = _vec(sg)
        sat = s_g > 0.0
        rs_val = self.fluid.pvt_o.rs(p) if rs is None else _vec(rs)
        X = torch.where(sat, s_g, rs_val)
        self.sat = sat.clone()
        self.sat_old = sat.clone()
        return torch.cat([p, s_w, X])

    def _interpret(
        self,
        state: torch.Tensor,
        sat: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """``(p, S_w, X)`` + phase mask → ``(p, S_w, S_g, R_s, S_o)``."""
        n = self.n_cells
        p = state[:n]
        s_w = state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        X = state[2 * n : 3 * n]
        rs_sat = self.fluid.pvt_o.rs(p)
        s_g = torch.where(sat, X.clamp(min=0.0), torch.zeros_like(X))
        rs = torch.where(sat, rs_sat, X.clamp(min=0.0))
        s_o = (1.0 - s_w - s_g).clamp(min=S_MIN, max=S_MAX)
        return p, s_w, s_g, rs, s_o

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        p, s_w, s_g, rs, s_o = self._interpret(state, self.sat)
        return {
            "p": p,
            "sw": s_w,
            "sg": s_g,
            "rs": rs,
            "so": s_o,
            "saturated": self.sat.to(state.dtype),
        }

    def accepted_discrete_masks(
        self,
        state: torch.Tensor,
    ) -> dict[str, tuple[bool, ...]]:
        """Return fixed phase-state and per-phase TPFA upwind decisions."""
        with torch.no_grad():
            saturated = self.sat.detach()
            p, _sw, _sg, rs, _so = self._interpret(state.detach(), saturated)
            connection = self.grid._connection_metrics()
            masks: dict[str, tuple[bool, ...]] = {
                "saturated_cell": tuple(bool(value) for value in saturated),
                "undersaturated_cell": tuple(bool(value) for value in ~saturated),
            }
            if connection is None or connection.n_faces == 0:
                return {
                    "water_upwind_left": (),
                    "oil_upwind_left": (),
                    "gas_upwind_left": (),
                    **masks,
                }
            left, right = connection.neighbors[:, 0], connection.neighbors[:, 1]
            pressure_drop = p[left] - p[right]
            water_potential_drop = pressure_drop
            oil_potential_drop = pressure_drop
            gas_potential_drop = pressure_drop
            if self.gravity:
                depth = self.grid._cell_centers_view()[:, 2]
                depth_drop = depth[left] - depth[right]
                for density, name in (
                    (self.fluid.pvt_w.density(p), "water"),
                    (self.fluid.pvt_o.density(p, rs), "oil"),
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
                **masks,
            }

    def convergence_inputs(
        self,
        residual: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """``(per-phase residual, reservoir pore volume [RB], per-phase B)`` for the
        CNV/MB criterion; oil B is evaluated at the actual (undersaturated) R_s."""
        n = self.n_cells
        p, _sw, _sg, rs, _so = self._interpret(state, self.sat)
        pv = self.grid._cell_volumes_view() * self.rock.porosity_at_pressure(p)
        return (
            [residual[:n], residual[n : 2 * n], residual[2 * n : 3 * n]],
            pv,
            [self.fluid.pvt_w.fvf(p), self.fluid.pvt_o.fvf(p, rs), self.fluid.pvt_g.fvf(p)],
        )

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
        source_gas_rates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = self.n_cells
        expected = (3 * n,)
        if state.shape != expected or state_old.shape != expected:
            raise GeoBrainError(
                "BlackOilVarSwitchModel state must be 1D of length 3*n_cells",
                object_name="BlackOilVarSwitchModel",
                field="state",
                expected=expected,
                actual=tuple(state.shape),
            )
        p, sw, sg, rs, so = self._interpret(state, self.sat)
        p_old, sw_old, sg_old, rs_old, so_old = self._interpret(state_old, self.sat_old)

        pvt_o, pvt_w, pvt_g = self.fluid.pvt_o, self.fluid.pvt_w, self.fluid.pvt_g
        V = self.grid._cell_volumes_view()
        Bw, Bo, Bg = pvt_w.fvf(p), pvt_o.fvf(p, rs), pvt_g.fvf(p)
        Bw_o, Bo_o, Bg_o = pvt_w.fvf(p_old), pvt_o.fvf(p_old, rs_old), pvt_g.fvf(p_old)
        mu_w, mu_o, mu_g = pvt_w.viscosity(p), pvt_o.viscosity(p, rs), pvt_g.viscosity(p)
        phi = self.rock.porosity_at_pressure(p)
        phi_old = self.rock.porosity_at_pressure(p_old)
        kr_w = self.fluid.relperm.kr_water(sw, sg)
        kr_o = self.fluid.relperm.kr_oil(sw, sg)
        kr_g = self.fluid.relperm.kr_gas(sg)

        acc_w = V * (phi * sw / Bw - phi_old * sw_old / Bw_o) / float(dt)
        acc_o = V * (phi * so / Bo - phi_old * so_old / Bo_o) / float(dt)
        acc_g = (
            V
            * (phi * (sg / Bg + rs * so / Bo) - phi_old * (sg_old / Bg_o + rs_old * so_old / Bo_o))
            / float(dt)
        )

        c = self.grid._connection_metrics()
        if c is None or c.n_faces == 0:
            R_w, R_o, R_g = acc_w, acc_o, acc_g
        else:
            T_alpha = self.grid.build_transmissibility(
                self.rock.permeability_m2
            )
            cl, cr = c.neighbors[:, 0], c.neighbors[:, 1]
            dp = p[cl] - p[cr]
            mob_w = kr_w / (mu_w * Bw)
            mob_o = kr_o / (mu_o * Bo)
            mob_g = kr_g / (mu_g * Bg)
            dphi_w = dphi_o = dphi_g = dp
            if self.gravity:
                z = self.grid._cell_centers_view()[:, 2]
                dz = z[cl] - z[cr]
                rho_w = pvt_w.density(p)
                rho_o = pvt_o.density(p, rs)
                rho_g = pvt_g.density(p)
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
            # Dissolved gas rides the oil flux, with R_s of the upstream oil cell.
            rs_face = rs[oil_upwind]
            F_g = F_g + rs_face * F_o
            R_w = acc_w + scatter_internal_face_flux(F_w, c.neighbors, n)
            R_o = acc_o + scatter_internal_face_flux(F_o, c.neighbors, n)
            R_g = acc_g + scatter_internal_face_flux(F_g, c.neighbors, n)

        for tensor, name in (
            (source_water_rates, "source_water_rates"),
            (source_oil_rates, "source_oil_rates"),
            (source_gas_rates, "source_gas_rates"),
        ):
            if tensor is not None and tensor.shape != (n,):
                raise GeoBrainError(
                    f"{name} must have shape (n_cells,)",
                    object_name="BlackOilVarSwitchModel",
                    field=name,
                    expected=(n,),
                    actual=tuple(tensor.shape),
                )
        if source_water_rates is not None:
            R_w = R_w - source_water_rates
        if source_oil_rates is not None:
            R_o = R_o - source_oil_rates
            # Dissolved gas rides with the source oil rate (the gas component
            # balance carries R_s·(oil rate)), exactly as the boundary term below
            # adds R_s·q_o to the gas balance. ``source_gas_rates`` is therefore
            # the *free*-gas source only; without this term a producer would
            # withdraw oil while leaving its dissolved gas behind, spuriously
            # super-saturating the cell and liberating free gas above p_b.
            R_g = R_g - rs * source_oil_rates
        if source_gas_rates is not None:
            R_g = R_g - source_gas_rates

        boundary_cells: list[int] = []
        boundary_water_flux: list[torch.Tensor] = []
        boundary_oil_flux: list[torch.Tensor] = []
        boundary_gas_flux: list[torch.Tensor] = []
        for bc in self.boundaries.bcs:
            ci = bc.cell
            dp_bc = bc.outward_pressure_drop(p[ci])
            T_bc = bc.transmissibility
            q_w = T_bc * kr_w[ci] / (mu_w[ci] * Bw[ci]) * dp_bc
            q_o = T_bc * kr_o[ci] / (mu_o[ci] * Bo[ci]) * dp_bc
            q_g = T_bc * kr_g[ci] / (mu_g[ci] * Bg[ci]) * dp_bc
            q_g = q_g + rs[ci] * q_o
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

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        **sources: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(
            lambda x: self.residual(x, state_old, dt, **sources),
            state,
            vectorize=True,
        )

    # ------------------------------------------------------------------
    # Phase switching (the Coats variable substitution)
    # ------------------------------------------------------------------
    def switch_phases(self, state: torch.Tensor) -> torch.Tensor:
        """Update the phase-state mask and reinterpret ``X`` across switches.

        * **OilAndGas → OilOnly**: free gas ``X = S_g`` went non-positive ⇒ all
          gas redissolves; the new unknown is ``R_s = R_s,sat(p)``.
        * **OilOnly → OilAndGas**: dissolved ``X = R_s`` reached saturation
          ``R_s,sat(p)`` ⇒ a free-gas phase appears; the new unknown is a small
          ``S_g = rs_eps``.

        The discrete phase-state MASK is updated under ``no_grad`` (a mask carries
        no gradient), but the X *value* is reinterpreted **differentiably**: cells
        that do NOT switch this step pass ``X`` straight through with its gradient
        intact, so the adjoint of a multi-step run stays correct everywhere except
        at the (zero-measure) switch events, without this, detaching X every step
        severs the parameter gradient through the gas/Rs variable entirely.
        """
        n = self.n_cells
        p = state[:n]
        X = state[2 * n : 3 * n]
        rs_sat = self.fluid.pvt_o.rs(p)
        with torch.no_grad():
            sat0 = self.sat.clone()
            # gas disappears: saturated cell whose Sg fell to/below zero.
            gas_gone = sat0 & (X <= 0.0)
            sat1 = sat0 & ~gas_gone
            # gas appears: undersaturated cell whose Rs reached saturation.
            gas_app = (~sat1) & (X >= rs_sat)
            new_sat = sat1 | gas_app
        # Differentiable reinterpretation. gas_gone → R_s,sat(p)−ε (carries the dp
        # gradient); gas_app → ε (a constant); non-switching cells keep X (and its
        # gradient). The per-range bounds are differentiable clamps (interior
        # pass-through).
        X = torch.where(gas_gone, rs_sat - self.rs_eps, X)
        X = torch.where(gas_app, torch.full_like(X, self.rs_eps), X)
        X = torch.where(
            new_sat,
            X.clamp(min=0.0, max=1.0),
            torch.minimum(X.clamp(min=0.0), rs_sat),
        )
        self.sat = new_sat
        return torch.cat([state[: 2 * n], X])

    def solve_timestep(
        self,
        state_old: torch.Tensor,
        dt: float,
        newton: _NewtonControls,
        **sources: torch.Tensor | None,
    ) -> tuple[torch.Tensor, bool]:
        """One implicit step with phase switching folded into the Newton update.

        Freezes the old-state mask, then Newton-iterates while applying
        :meth:`switch_phases` after each update (so a cell that crosses the
        bubble point flips its primary variable mid-solve).
        Returns ``(state, converged)``.
        """
        self.sat_old = self.sat.clone()  # state_old's mask
        x = state_old.clone()
        r = self.residual(x, state_old, dt, **sources)
        for _ in range(newton.max_iter):
            if float(torch.linalg.vector_norm(r, ord=float("inf"))) < newton.tol:
                return x, True
            J = self.jacobian(x, state_old, dt, **sources)
            dx = torch.linalg.solve(J, r)
            x = x - dx
            x = self.switch_phases(x)
            r = self.residual(x, state_old, dt, **sources)
        return x, float(torch.linalg.vector_norm(r, ord=float("inf"))) < newton.tol
