"""
Compositional (EOS-flash) **thermal** reservoir model on a general 2-D grid using
the MPFA-O multi-point flux, coupled component-mole + energy conservation.

The thermal extension of :class:`MPFACompositionalModel`: temperature ``T`` becomes
a primary variable (after the ``n_c−1`` mole fractions) and the vapor-liquid flash
is run at the *cell* temperature, so K-values, Z-factors and molar densities all
respond to ``T`` (the Wilson estimate and cubic-EOS attraction term are already
``T``-dependent). A single energy-conservation equation is added.

The thermal compositional model uses constant-Cp calorific energy (*not* EOS
departure-function enthalpy), so each phase's molar heat capacity is the
mass-fraction-weighted component heat
capacity expressed per mole, ``cm_p = Σ_i C_i·x_{i,p}·M_i`` [J/(mol·K)], its molar
internal energy is ``U_p = cm_p·T`` and its molar enthalpy ``h_p = cm_p·T + p/ρ_p``
(``ρ_p`` = molar density, so ``p/ρ_p`` is the molar flow-work ``p·v_p``). Per cell::

    comp_i:  V·(N_i − N_i|old)/Δt + Σ_f (±F_{f,i}) − src_i = 0          (i = 1..n_c)
    energy:  V·(E − E|old)/Δt + Σ_f (h_l|_up q_l + h_v|_up q_v + F^cond) − q^e = 0
    E = (1−φ)ρ_r C_r T + φ (S_l ρ_l cm_l + S_v ρ_v cm_v) T

where ``q_l, q_v`` are the *same* upwinded phase molar fluxes the component balance
uses, and conduction follows the parallel-conductance model (rock stencil +
porosity-weighted fluid stencil scaled by ``λ_l S_l,f + λ_v S_v,f``). On a
K-orthogonal grid the whole residual reproduces the TPFA thermal compositional
solution; freezing ``T`` reproduces :class:`MPFACompositionalModel`.

Forcings (the compositional parent has neither, so both are added here):

- **gravity** via a per-cell ``depth_m`` field: per-phase potentials ``Φ_p = p −
  ρ_p,mass·g·D`` (``ρ_p,mass`` = molar density × phase molar mass), so the lighter
  vapor rises (gravity segregation / gas-cap formation);
- **BHP-controlled Peaceman wells** ``(cell, WI, bhp, z_inj, T_inj)``: production
  draws the cell fluid; injection flashes the feed ``z_inj`` at ``(p_cell, T_inj)``
  and its phases flow by mobility (a single-phase injectant injects ``z_inj``).

Two-phase feed, Corey relperm, interior MPFA faces + sources; SI units. (Dirichlet /
Neumann boundary faces are a follow-up; they need boundary feed-flash + augmented
conduction; a rate boundary is already expressible via the per-component molar
``sources``.)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

import torch

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from ..discretization.flux import scatter_boundary_outflow, scatter_internal_face_flux
from ..models._stencil_inversion import StencilInversionMixin
from ..models.mpfa_thermal_single_phase import _register_conduction_3d, _register_conduction_bc_3d
from ..models.mpfa_thermal_two_phase import _register_conduction, _register_conduction_bc
from ..discretization.mpfa import MPFAGrid2D, mpfa_o_face_flux_stencils_bc
from ..discretization.mpfa3d import MPFAGrid3D, mpfa_o_face_flux_stencils_3d_bc
from .._state_validation import (
    cell_scalar_input,
    composition_input,
    nonnegative_real,
    positive_real,
)
from .cubic_eos import CubicEOS
from .flash import flash, require_flash_converged
from .mpfa_model import MPFACompositionalModel, MPFACompositionalModel3D, _R_GAS
from .viscosity import lbc_viscosity


_WellRecord = Sequence[object]
_WellRecords = Sequence[_WellRecord]
_BoundaryRecords = Mapping[int, Sequence[object]]


class _RateWellModel(Protocol):
    rate_well_cells: torch.Tensor | None
    rate_well_WI: torch.Tensor
    rate_well_q: torch.Tensor
    rate_well_z: torch.Tensor
    rate_well_T_inj: torch.Tensor
    rate_well_has_limit: torch.Tensor
    rate_well_bhp_limit: torch.Tensor

    def register_buffer(
        self,
        name: str,
        tensor: torch.Tensor | None,
        persistent: bool = True,
    ) -> None: ...


def _cell_index(value: object) -> int:
    """Normalize the first entry of a legacy well tuple without changing runtime rules."""
    return int(cast(int | str | bytes | bytearray, value))


def register_rate_wells_comp(
    model: _RateWellModel,
    rate_wells: _WellRecords | None,
    dtype: torch.dtype,
) -> None:
    """Register optional **rate-controlled** compositional thermal wells:
    ``[(cell, WI, q_res, z_inj, T_inj[, bhp_limit]), ...]``: ``q_res`` is the
    prescribed total **molar** rate [mol/s] (``> 0`` production, ``< 0`` injection),
    the bhp being *solved* from it (a total-rate well control). Production draws
    the cell fluid (phase molar fluxes split by molar mobility); injection injects the
    specified feed ``z_inj`` at the molar rate; energy is the per-phase molar enthalpy.
    The optional ``bhp_limit`` switches the well to BHP control when the rate-implied bhp
    would violate it (assumed on the physical side of ``p_cell``, producer min below /
    injector max above, so the post-switch flow keeps the rate-target direction; a
    flow-reversing limit is a shut-in well and is not modelled). Shared by the 2-D and
    3-D thermal compositional models."""
    if not rate_wells:
        model.rate_well_cells = None
        return

    def _has(w: _WellRecord) -> bool:
        return len(w) >= 6 and w[5] is not None

    model.register_buffer(
        "rate_well_cells",
        torch.tensor([_cell_index(w[0]) for w in rate_wells], dtype=torch.long),
    )
    model.register_buffer(
        "rate_well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer(
        "rate_well_q", torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer(
        "rate_well_z", torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer(
        "rate_well_T_inj", torch.stack([torch.as_tensor(w[4], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer("rate_well_has_limit", torch.tensor([_has(w) for w in rate_wells]))
    model.register_buffer(
        "rate_well_bhp_limit",
        torch.stack([torch.as_tensor(w[5] if _has(w) else 0.0, dtype=dtype) for w in rate_wells]),
    )


if TYPE_CHECKING:
    class _ThermalCompositionalBase(MPFACompositionalModel):
        """Static compositional interface for the inversion mixin methods."""

        def build_L_perm(self, perm: torch.Tensor) -> torch.Tensor:
            pass

        def build_sparse_L_perm(self, perm: torch.Tensor) -> torch.Tensor:
            pass

        def build_L_rock(
            self,
            lam_rock: float | torch.Tensor,
            porosity: torch.Tensor | None = None,
        ) -> torch.Tensor:
            pass

        def build_sparse_L_rock(
            self,
            lam_rock: float | torch.Tensor,
            porosity: torch.Tensor | None = None,
        ) -> torch.Tensor:
            pass

        def build_L_phi(self, porosity: torch.Tensor) -> torch.Tensor:
            pass

        def build_sparse_L_phi(self, porosity: torch.Tensor) -> torch.Tensor:
            pass
else:
    class _ThermalCompositionalBase(StencilInversionMixin, MPFACompositionalModel):
        pass


class MPFAThermalCompositionalModel(_ThermalCompositionalBase):
    """Multi-component (EOS-flash) thermal flow with an MPFA-O multi-point flux (2-D, SI).

    Args:
        Dimensional public names declare canonical SI units. Component and rock
        heat capacities use J/(kg·K), density uses kg/m³, phase viscosities use
        Pa·s, thermal conductivities use W/(m·K), depth uses m, gravity uses
        m/s², and NNC transmissibility uses m³.
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalCompositionalModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("composition", "1", ("cell", "component"), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("total_molar", "mol/s", "molar", "pressure"),
            ("component_molar", "mol/s", "molar", "composition"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-2d",),
        phases=("liquid", "vapor"),
    )

    rate_well_cells: torch.Tensor | None
    rate_well_WI: torch.Tensor
    rate_well_q: torch.Tensor
    rate_well_z: torch.Tensor
    rate_well_T_inj: torch.Tensor
    rate_well_has_limit: torch.Tensor
    rate_well_bhp_limit: torch.Tensor
    dir_list: list[int] | None

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: object,
        eos: CubicEOS,
        *,
        liquid_viscosity_pa_s: float = 5e-4,
        vapor_viscosity_pa_s: float = 2e-5,
        viscosity: str = "constant",
        residual_liquid_saturation: float = 0.0,
        residual_vapor_saturation: float = 0.0,
        liquid_corey_exponent: float = 2.0,
        vapor_corey_exponent: float = 2.0,
        component_heat_capacities_j_kg_k: object = 2100.0,
        rock_heat_capacity_j_kg_k: float = 1000.0,
        rock_density_kg_m3: float = 2650.0,
        liquid_thermal_conductivity_w_m_k: float = 0.13,
        vapor_thermal_conductivity_w_m_k: float = 0.03,
        rock_thermal_conductivity_w_m_k: float = 3.0,
        cell_volumes_m3: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_transmissibility_m3: torch.Tensor | None = None,
        depth_m: torch.Tensor | None = None,
        gravity_m_s2: float = 9.81,
        bhp_wells: _WellRecords | None = None,
        dirichlet: _BoundaryRecords | None = None,
        neumann: _BoundaryRecords | None = None,
        rate_wells: _WellRecords | None = None,
    ) -> None:
        # the parent stores an isothermal self.T; thermal makes T a state variable
        super().__init__(
            grid,
            perm_tensor,
            porosity,
            eos,
            300.0,
            liquid_viscosity_pa_s=liquid_viscosity_pa_s,
            vapor_viscosity_pa_s=vapor_viscosity_pa_s,
            viscosity=viscosity,
            residual_liquid_saturation=residual_liquid_saturation,
            residual_vapor_saturation=residual_vapor_saturation,
            liquid_corey_exponent=liquid_corey_exponent,
            vapor_corey_exponent=vapor_corey_exponent,
            cell_volumes_m3=cell_volumes_m3,
            nnc_pairs=nnc_pairs,
            nnc_transmissibility_m3=nnc_transmissibility_m3,
        )
        components = tuple(eos.mixture.names)
        self.schema = _flow_model_schema(
            model_name=type(self).__name__,
            primary_fields=(
                ("pressure", "Pa", ("cell",), ()),
                ("composition", "1", ("cell", "component"), components),
                ("temperature", "K", ("cell",), ()),
            ),
            residual_blocks=(
                ("total_molar", "mol/s", "molar", "pressure"),
                ("component_molar", "mol/s", "molar", "composition"),
                ("energy", "W", "energy", "temperature"),
            ),
            grid_kinds=("mpfa-2d",),
            phases=("liquid", "vapor"),
            components=components,
        )
        dtype, device = perm_tensor.dtype, perm_tensor.device
        cp = cell_scalar_input(
            component_heat_capacities_j_kg_k,
            n_cells=self.nc,
            dtype=dtype,
            device=device,
            field="component_heat_capacities_j_kg_k",
            positive=True,
            object_name="MPFAThermalCompositionalModel",
        )
        self.register_buffer("cp_components", cp)
        self.cp_r = positive_real(
            rock_heat_capacity_j_kg_k,
            object_name="MPFAThermalCompositionalModel",
            field="rock_heat_capacity_j_kg_k",
        )
        self.rho_r = positive_real(
            rock_density_kg_m3,
            object_name="MPFAThermalCompositionalModel",
            field="rock_density_kg_m3",
        )
        self.lam_l, self.lam_v, self.lam_rock = (
            positive_real(
                liquid_thermal_conductivity_w_m_k,
                object_name="MPFAThermalCompositionalModel",
                field="liquid_thermal_conductivity_w_m_k",
            ),
            positive_real(
                vapor_thermal_conductivity_w_m_k,
                object_name="MPFAThermalCompositionalModel",
                field="vapor_thermal_conductivity_w_m_k",
            ),
            positive_real(
                rock_thermal_conductivity_w_m_k,
                object_name="MPFAThermalCompositionalModel",
                field="rock_thermal_conductivity_w_m_k",
            ),
        )
        self.g = nonnegative_real(
            gravity_m_s2,
            object_name="MPFAThermalCompositionalModel",
            field="gravity_m_s2",
        )
        if depth_m is not None:
            depth_tensor = cell_scalar_input(
                depth_m,
                n_cells=self.n_cells,
                dtype=dtype,
                device=device,
                field="depth_m",
                positive=False,
                object_name="MPFAThermalCompositionalModel",
            )
            self.register_buffer("depth", depth_tensor)
        else:
            self.depth = None

        # Dirichlet pressure boundary faces ``{e: (p_bc, z_bc, T_bc)}``, a fixed
        # external fluid (feed ``z_bc``) at ``(p_bc, T_bc)``. The compositional parent
        # has no BC machinery, so the ghost-augmented operator (overriding the no-flow
        # L/face_lr) + augmented conduction are built here, and the *constant* ghost
        # phase state is precomputed once (a fixed boundary fluid).
        if dirichlet:
            dir_edges = [int(e) for e in dirichlet]
            L, face_lr, dir_list = mpfa_o_face_flux_stencils_bc(grid, perm_tensor, dir_edges)
            self.register_buffer("L", L)
            self.register_buffer("face_lr", face_lr)
            p_bc = torch.stack([torch.as_tensor(dirichlet[e][0], dtype=dtype) for e in dir_list])
            z_bc = torch.stack([torch.as_tensor(dirichlet[e][1], dtype=dtype) for e in dir_list])
            T_bc = torch.stack([torch.as_tensor(dirichlet[e][2], dtype=dtype) for e in dir_list])
            mm_l, mm_v, x_bc, y_bc, h_l_bc, h_v_bc, S_l_bc = self._ghost_state(p_bc, z_bc, T_bc)
            self.register_buffer("p_bc", p_bc)
            self.register_buffer("ghost_mm_l", mm_l)
            self.register_buffer("ghost_mm_v", mm_v)
            self.register_buffer("ghost_x", x_bc)
            self.register_buffer("ghost_y", y_bc)
            self.register_buffer("ghost_h_l", h_l_bc)
            self.register_buffer("ghost_h_v", h_v_bc)
            self.register_buffer("ghost_S_l", S_l_bc)
            self.register_buffer("T_bc", T_bc)
            _register_conduction_bc(
                self, grid, self.phi, rock_thermal_conductivity_w_m_k, dir_edges, dtype
            )
            self.dir_list = dir_list  # ghost order; lets the inversion mixin
        else:  # rebuild the augmented stencil (perm/λ/φ)
            self.p_bc = None
            self.dir_list = None
            _register_conduction(self, grid, self.phi, rock_thermal_conductivity_w_m_k, dtype)

        # Neumann / rate boundary faces ``{e: (q, z_inj, T_inj)}``: q = outward total
        # molar rate; outflow produces the cell flowing fluid, inflow injects feed z_inj.
        if neumann:
            faces = list(neumann)
            bc = grid.edge_cells
            self.register_buffer(
                "neumann_cells", torch.tensor([bc[f][0] for f in faces], dtype=torch.long)
            )
            self.register_buffer(
                "neumann_q",
                torch.stack([torch.as_tensor(neumann[f][0], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_z",
                torch.stack([torch.as_tensor(neumann[f][1], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_T_inj",
                torch.stack([torch.as_tensor(neumann[f][2], dtype=dtype) for f in faces]),
            )
        else:
            self.neumann_q = None

        # BHP-controlled Peaceman wells (the compositional parent has none): each
        # ``(cell, WI, bhp, z_inj, T_inj)`` produces the cell fluid (component flux
        # q_l·x + q_v·y at the cell enthalpy) or injects the feed ``z_inj`` flashed
        # at the well conditions (p_cell, T_inj).
        if bhp_wells:
            self.register_buffer(
                "well_cells",
                torch.tensor([_cell_index(w[0]) for w in bhp_wells], dtype=torch.long),
            )
            self.register_buffer(
                "well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in bhp_wells])
            )
            self.register_buffer(
                "well_bhp", torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in bhp_wells])
            )
            self.register_buffer(
                "well_z_inj", torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in bhp_wells])
            )
            self.register_buffer(
                "well_T_inj", torch.stack([torch.as_tensor(w[4], dtype=dtype) for w in bhp_wells])
            )
        else:
            self.well_cells = None

        register_rate_wells_comp(self, rate_wells, dtype)

    def _ghost_state(
        self,
        p_bc: torch.Tensor,
        z_bc: torch.Tensor,
        T_bc: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Constant phase state of a Dirichlet boundary fluid (feed ``z_bc`` at
        ``p_bc, T_bc``). Two-phase boundaries flow both equilibrium phases by their
        mobility; a *single-phase* boundary (flash ``V`` outside ``[0,1]``) presents
        the feed composition ``z_bc`` in the present phase with its single-phase
        density (the EOS-extrapolated split would otherwise be ≠ ``z_bc``). Returns
        ``(mm_l, mm_v, x, y, h_l, h_v, S_l)`` (mass-mobilities ρ_p·kr_p/μ_p)."""
        eos = self.eos
        boundary_flash = flash(eos, p_bc, T_bc, z_bc)
        require_flash_converged(boundary_flash, object_name=f"{type(self).__name__}.boundary")
        Vr = boundary_flash.V
        _, x, y, S_l, rho_l, rho_v, _ = self._phase_state_T(p_bc, z_bc, T_bc)
        single_v = (Vr >= 1.0).unsqueeze(-1)
        single_l = (Vr <= 0.0).unsqueeze(-1)
        x = torch.where(single_l, z_bc, x)
        y = torch.where(single_v, z_bc, y)
        A_i, B_i = eos.ab_components(p_bc, T_bc)
        A_z, B_z, _ = eos.mixture_ab(z_bc, A_i, B_i)
        rho_l = torch.where(
            single_l.squeeze(-1),
            p_bc / (eos.compressibility(A_z, B_z, root="liquid") * _R_GAS * T_bc),
            rho_l,
        )
        rho_v = torch.where(
            single_v.squeeze(-1),
            p_bc / (eos.compressibility(A_z, B_z, root="vapor") * _R_GAS * T_bc),
            rho_v,
        )
        S_l = torch.where(
            single_v.squeeze(-1),
            torch.zeros_like(S_l),
            torch.where(single_l.squeeze(-1), torch.ones_like(S_l), S_l),
        )
        mu_l, mu_v = self._phase_viscosities_T(p_bc, T_bc, x, y, rho_l, rho_v)
        mob_l, mob_v = self._mobilities(S_l, mu_l, mu_v)
        mw = self.cp_components * eos.mixture.molar_mass_kg_mol
        h_l = (x * mw).sum(-1) * T_bc + p_bc / rho_l
        h_v = (y * mw).sum(-1) * T_bc + p_bc / rho_v
        return rho_l * mob_l, rho_v * mob_v, x, y, h_l, h_v, S_l

    # ------------------------------------------------------------------
    # State plumbing: [p, z_1..z_{nc-1}, T]
    # ------------------------------------------------------------------
    def state_size(self) -> int:
        return self.n_cells * self.nc + self.n_cells

    def initial_state(  # type: ignore[override]  # thermal state adds temperature
        self,
        pressure: object,
        z: object,
        temperature: object,
    ) -> torch.Tensor:
        n, dtype, device = self.n_cells, self.V_cell.dtype, self.V_cell.device
        p = cell_scalar_input(
            pressure,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="pressure_pa",
            positive=True,
        )
        z_tensor = composition_input(
            z,
            n_cells=n,
            n_components=self.nc,
            dtype=dtype,
            device=device,
        )
        T = cell_scalar_input(
            temperature,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="temperature_k",
            positive=True,
        )
        return torch.cat([p, z_tensor[:, : self.nc - 1].reshape(-1), T])

    def _unpack(  # type: ignore[override]  # thermal state appends temperature
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, nc = self.n_cells, self.nc
        p = state[:n]
        z_red = state[n : n * nc].reshape(n, nc - 1)
        z = torch.cat([z_red, 1.0 - z_red.sum(dim=-1, keepdim=True)], dim=-1)
        T = state[n * nc : n * nc + n]
        return p, z, T

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        p, z, T = self._unpack(state)
        return {"p": p, "z": z, "T": T}

    def accepted_discrete_masks(self, state: torch.Tensor) -> dict[str, tuple[bool, ...]]:
        """Return the converged flash regime selected in each cell."""
        p, z, temperature = self._unpack(state)
        vapor_fraction = self._phase_state_T(p, z, temperature)[0].detach()
        return {
            "liquid_only": tuple(bool(value) for value in (vapor_fraction <= 0.0)),
            "two_phase": tuple(
                bool(value)
                for value in ((vapor_fraction > 0.0) & (vapor_fraction < 1.0))
            ),
            "vapor_only": tuple(bool(value) for value in (vapor_fraction >= 1.0)),
        }

    def _phase_state_T(
        self,
        p: torch.Tensor,
        z: torch.Tensor,
        T: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Flash at the per-cell ``T`` → ``(V, x, y, S_l, rho_l, rho_v, D)`` (molar)."""
        res = flash(self.eos, p, T, z)
        require_flash_converged(res, object_name=type(self).__name__)
        V = res.V
        x, y = res.x, res.y
        A_i, B_i = self.eos.ab_components(p, T)
        A_l, B_l, _ = self.eos.mixture_ab(x, A_i, B_i)
        A_v, B_v, _ = self.eos.mixture_ab(y, A_i, B_i)
        Z_l = self.eos.compressibility(A_l, B_l, root="liquid")
        Z_v = self.eos.compressibility(A_v, B_v, root="vapor")
        v_l = Z_l * _R_GAS * T / p
        v_v = Z_v * _R_GAS * T / p
        rho_l, rho_v = 1.0 / v_l, 1.0 / v_v
        D = (1.0 - V) * v_l + V * v_v
        S_v = (V * v_v) / D
        return V, x, y, 1.0 - S_v, rho_l, rho_v, D

    def _phase_viscosities_T(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rho_l: torch.Tensor,
        rho_v: torch.Tensor,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """``(μ_l, μ_v)`` at the *state* temperature ``T``. The isothermal parent's
        :meth:`_phase_viscosities` hard-codes the frozen ``self.T``; in the thermal
        model ``T`` is a primary variable, so the LBC viscosity and its
        compressibility ``Z = p/(ρ·R·T)`` must use the per-cell / boundary ``T``
        (else the viscosity, and the ``∂μ/∂T`` Jacobian term; are evaluated at the
        stale reference temperature). ``viscosity='constant'`` is unaffected."""
        if self.viscosity != "lbc":
            return self.mu_l, self.mu_v
        Z_l = p / (rho_l * _R_GAS * T)
        Z_v = p / (rho_v * _R_GAS * T)
        mix = self.eos.mixture
        return (lbc_viscosity(mix, p, T, x, Z_l), lbc_viscosity(mix, p, T, y, Z_v))

    # ------------------------------------------------------------------
    # Residual: [R_1..R_nc (components), R_energy]
    # ------------------------------------------------------------------
    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: torch.Tensor | None = None,
        source_energy: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
        lam_rock: float | torch.Tensor | None = None,
        lam_l: float | torch.Tensor | None = None,
        lam_v: float | torch.Tensor | None = None,
        porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, nc = self.n_cells, self.nc
        if state.shape != (n * nc + n,) or state_old.shape != (n * nc + n,):
            raise GeoBrainError(
                "MPFAThermalCompositionalModel state must be length n_cells*(n_components+1)",
                object_name="MPFAThermalCompositionalModel",
                field="state",
                expected=(n * nc + n,),
                actual=tuple(state.shape),
            )
        p, z, T = self._unpack(state)
        p_o, z_o, T_o = self._unpack(state_old)

        V, x, y, S_l, rho_l, rho_v, Dmix = self._phase_state_T(p, z, T)
        _, x_o, y_o, S_l_o, rho_l_o, rho_v_o, Dmix_o = self._phase_state_T(p_o, z_o, T_o)

        phi = self.phi if porosity is None else porosity  # porosity inversion override
        N = (phi / Dmix).unsqueeze(-1) * z
        N_o = (phi / Dmix_o).unsqueeze(-1) * z_o
        acc = self.V_cell.unsqueeze(-1) * (N - N_o) / float(dt)  # (n_cells, nc)
        R = acc

        mw = self.cp_components * self.eos.mixture.molar_mass_kg_mol
        cm_l = (x * mw).sum(-1)  # molar heat capacity of liquid [J/mol/K]
        cm_v = (y * mw).sum(-1)
        h_l = cm_l * T + p / rho_l  # molar enthalpy [J/mol]
        h_v = cm_v * T + p / rho_v
        mu_l, mu_v = self._phase_viscosities_T(p, T, x, y, rho_l, rho_v)
        mob_l, mob_v = self._mobilities(S_l, mu_l, mu_v)

        # per-phase potentials with gravity: Φ_p = p − ρ_p,mass·g·D (mass density =
        # molar density × phase molar mass). Without depth, Φ_l = Φ_v = p (the single
        # Darcy potential of the isothermal compositional model).
        phi_l = phi_v = p
        if self.depth is not None:
            M_l = (x * self.eos.mixture.molar_mass_kg_mol).sum(-1)
            M_v = (y * self.eos.mixture.molar_mass_kg_mol).sum(-1)
            gz = self.g * self.depth
            phi_l = p - rho_l * M_l * gz
            phi_v = p - rho_v * M_v * gz

        mm_l_c, mm_v_c = rho_l * mob_l, rho_v * mob_v  # per-cell mass-mobilities

        # Dirichlet pressure boundaries: augment with the constant ghost state (the
        # operator L / face_lr / conduction stencils span [cells ; ghosts]). The ghost
        # potential is p_bc (no gravity on the ghost: a fixed-potential boundary).
        if self.p_bc is not None:
            phi_l = torch.cat([phi_l, self.p_bc])
            phi_v = torch.cat([phi_v, self.p_bc])
            mm_l_a = torch.cat([mm_l_c, self.ghost_mm_l])
            mm_v_a = torch.cat([mm_v_c, self.ghost_mm_v])
            x_a, y_a = torch.cat([x, self.ghost_x]), torch.cat([y, self.ghost_y])
            h_l_a, h_v_a = torch.cat([h_l, self.ghost_h_l]), torch.cat([h_v, self.ghost_h_v])
            T_aug, Sl_a = torch.cat([T, self.T_bc]), torch.cat([S_l, self.ghost_S_l])
        else:
            mm_l_a, mm_v_a, x_a, y_a = mm_l_c, mm_v_c, x, y
            h_l_a, h_v_a, T_aug, Sl_a = h_l, h_v, T, S_l

        left_cells, right_cells = self.face_lr[:, 0], self.face_lr[:, 1]
        L = self.L if perm is None else self.build_L_perm(perm)
        G_l, G_v = L @ phi_l, L @ phi_v
        up_l, up_v = G_l >= 0, G_v >= 0
        ql = torch.where(up_l, mm_l_a[left_cells], mm_l_a[right_cells]) * G_l
        qv = torch.where(up_v, mm_v_a[left_cells], mm_v_a[right_cells]) * G_v
        x_f = torch.where(up_l.unsqueeze(-1), x_a[left_cells], x_a[right_cells])
        y_f = torch.where(up_v.unsqueeze(-1), y_a[left_cells], y_a[right_cells])
        F = ql.unsqueeze(-1) * x_f + qv.unsqueeze(-1) * y_f  # (n_faces, nc) component flux
        right_is_real = (right_cells < n).unsqueeze(-1)
        right_index = torch.where(right_cells < n, right_cells, left_cells)
        R = R.index_add(0, left_cells, F).index_add(
            0,
            right_index,
            torch.where(right_is_real, -F, F.new_zeros(())),
        )

        if self.nnc_trans is not None:
            a, b = self.nnc_pairs[:, 0], self.nnc_pairs[:, 1]
            Gn = self.nnc_trans * (p[a] - p[b])
            upn = (Gn >= 0).unsqueeze(-1)
            qln = torch.where(Gn >= 0, mm_l_c[a], mm_l_c[b]) * Gn
            qvn = torch.where(Gn >= 0, mm_v_c[a], mm_v_c[b]) * Gn
            Fn = qln.unsqueeze(-1) * torch.where(upn, x[a], x[b]) + qvn.unsqueeze(-1) * torch.where(
                upn, y[a], y[b]
            )
            R = R.index_add(0, a, Fn).index_add(0, b, -Fn)

        # --- energy equation ---
        cm_l_o = (x_o * mw).sum(-1)
        cm_v_o = (y_o * mw).sum(-1)
        S_v_o = 1.0 - S_l_o
        E = (1.0 - phi) * self.rho_r * self.cp_r * T + phi * (
            S_l * rho_l * cm_l + (1.0 - S_l) * rho_v * cm_v
        ) * T
        E_o = (1.0 - phi) * self.rho_r * self.cp_r * T_o + phi * (
            S_l_o * rho_l_o * cm_l_o + S_v_o * rho_v_o * cm_v_o
        ) * T_o
        acc_e = self.V_cell * (E - E_o) / float(dt)

        adv = (
            torch.where(up_l, h_l_a[left_cells], h_l_a[right_cells]) * ql
            + torch.where(up_v, h_v_a[left_cells], h_v_a[right_cells]) * qv
        )
        Sv_a = 1.0 - Sl_a
        ll = self.lam_l if lam_l is None else lam_l
        lv = self.lam_v if lam_v is None else lam_v
        lam_fluid_face = ll * 0.5 * (Sl_a[left_cells] + Sl_a[right_cells]) + lv * 0.5 * (
            Sv_a[left_cells] + Sv_a[right_cells]
        )
        if lam_rock is None and porosity is None:
            L_rock = self.L_rock
        else:
            L_rock = self.build_L_rock(self.lam_rock if lam_rock is None else lam_rock, porosity)
        L_phi = self.L_phi if porosity is None else self.build_L_phi(porosity)
        F_energy = adv + (L_rock @ T_aug) + lam_fluid_face * (L_phi @ T_aug)
        right_is_real = right_cells < n
        R_e = (
            acc_e
            + scatter_internal_face_flux(F_energy[right_is_real], self.face_lr[right_is_real], n)
            + scatter_boundary_outflow(F_energy[~right_is_real], left_cells[~right_is_real], n)
        )

        # BHP-controlled Peaceman wells (component flux + energy).
        #  - PRODUCTION (p_cell ≥ bhp): draws the cell fluid: each phase flows by its
        #    own mobility, F_i = q_l·x_cell + q_v·y_cell, energy h_l·q_l + h_v·q_v.
        #  - INJECTION (p_cell < bhp): injects the *specified feed composition* z_inj
        #    at the injectant's total molar mobility, F_i = q_inj·z_inj (NOT a
        #    mobility-weighted split: the injected stream is a known fluid, so a
        #    single-phase injectant injects z_inj exactly); energy q_inj·h_feed with
        #    the feed molar enthalpy h_feed = Σ_i C_i z_inj_i M_i·T_inj + p/ρ_feed.
        if self.well_cells is not None:
            wc, WI, bhp = self.well_cells, self.well_WI, self.well_bhp
            z_inj, Tinj = self.well_z_inj, self.well_T_inj
            dpw = p[wc] - bhp
            prod = dpw >= 0
            # injectant flash → total molar mobility + feed molar density/enthalpy
            _, xi, yi, Sli, rli, rvi, D_feed = self._phase_state_T(p[wc], z_inj, Tinj)
            mu_li, mu_vi = self._phase_viscosities_T(p[wc], Tinj, xi, yi, rli, rvi)
            mob_li, mob_vi = self._mobilities(Sli, mu_li, mu_vi)
            q_inj = WI * (rli * mob_li + rvi * mob_vi) * dpw  # total injectant molar rate
            cm_feed = (z_inj * mw).sum(-1)
            h_feed = cm_feed * Tinj + p[wc] / (
                1.0 / D_feed
            )  # feed molar enthalpy (ρ_feed = 1/D_feed)
            # production molar phase fluxes from the cell
            q_l_p = WI * (rho_l * mob_l)[wc] * dpw
            q_v_p = WI * (rho_v * mob_v)[wc] * dpw
            pr = prod.unsqueeze(-1)
            F_prod = q_l_p.unsqueeze(-1) * x[wc] + q_v_p.unsqueeze(-1) * y[wc]
            F_inj = q_inj.unsqueeze(-1) * z_inj
            R = R.index_add(0, wc, torch.where(pr, F_prod, F_inj))
            E_prod = h_l[wc] * q_l_p + h_v[wc] * q_v_p
            E_inj = h_feed * q_inj
            R_e = R_e + scatter_boundary_outflow(torch.where(prod, E_prod, E_inj), wc, n)

        # Rate-controlled wells: prescribe the total MOLAR rate q_res (the bhp is solved
        #, a total-rate well control). PRODUCTION (q≥0) draws the cell fluid, phase
        # molar fluxes split by molar mobility (F_i = q_l·x + q_v·y, energy h_l q_l +
        # h_v q_v); INJECTION (q<0) injects the specified feed z_inj at the molar rate
        # (F_i = q·z_inj, energy q·h_feed). With a bhp_limit the well switches to BHP
        # control when the rate-implied bhp would violate it.
        if self.rate_well_cells is not None:
            wc, WI, q = self.rate_well_cells, self.rate_well_WI, self.rate_well_q
            z_inj, Tinj = self.rate_well_z, self.rate_well_T_inj
            prod = q >= 0
            lam_c = (mm_l_c + mm_v_c)[wc].clamp_min(1e-30)  # cell total molar mobility
            _, xi, yi, Sli, rli, rvi, D_feed = self._phase_state_T(p[wc], z_inj, Tinj)
            mu_li, mu_vi = self._phase_viscosities_T(p[wc], Tinj, xi, yi, rli, rvi)
            mob_li, mob_vi = self._mobilities(Sli, mu_li, mu_vi)
            lam_i = (rli * mob_li + rvi * mob_vi).clamp_min(1e-30)  # injectant total molar mobility
            lam = torch.where(prod, lam_c, lam_i)
            bhp_rate = p[wc] - q / (WI * lam)
            lim = self.rate_well_bhp_limit
            viol = self.rate_well_has_limit & torch.where(prod, bhp_rate < lim, bhp_rate > lim)
            q_eff = torch.where(
                viol, WI * lam * (p[wc] - lim), q
            )  # total molar rate (target or bhp-limited)
            q_l_p = q_eff * mm_l_c[wc] / lam_c  # production phase molar fluxes
            q_v_p = q_eff * mm_v_c[wc] / lam_c
            F_prod = q_l_p.unsqueeze(-1) * x[wc] + q_v_p.unsqueeze(-1) * y[wc]
            E_prod = h_l[wc] * q_l_p + h_v[wc] * q_v_p
            cm_feed = (z_inj * mw).sum(-1)
            h_feed = cm_feed * Tinj + p[wc] / (1.0 / D_feed)
            F_inj = q_eff.unsqueeze(-1) * z_inj
            E_inj = q_eff * h_feed
            pr = prod.unsqueeze(-1)
            R = R.index_add(0, wc, torch.where(pr, F_prod, F_inj))
            R_e = R_e + scatter_boundary_outflow(torch.where(prod, E_prod, E_inj), wc, n)

        # Neumann / rate boundary faces: q = outward total molar rate at the boundary
        # cell. Outflow (q ≥ 0) produces the cell's flowing (mobility-weighted) fluid;
        # inflow (q < 0) injects the feed z_inj at its molar enthalpy.
        if self.neumann_q is not None:
            nb, q, z_inj, Tinj = (
                self.neumann_cells,
                self.neumann_q,
                self.neumann_z,
                self.neumann_T_inj,
            )
            out = (q >= 0).unsqueeze(-1)
            mm_t = (mm_l_c[nb] + mm_v_c[nb]).clamp_min(1e-30)
            flow_comp = (
                mm_l_c[nb].unsqueeze(-1) * x[nb] + mm_v_c[nb].unsqueeze(-1) * y[nb]
            ) / mm_t.unsqueeze(-1)
            flow_h = (mm_l_c[nb] * h_l[nb] + mm_v_c[nb] * h_v[nb]) / mm_t
            _, xi, yi, Sli, rli, rvi, D_fi = self._phase_state_T(p[nb], z_inj, Tinj)
            h_fi = (z_inj * mw).sum(-1) * Tinj + p[nb] / (1.0 / D_fi)
            comp = torch.where(out, flow_comp, z_inj)
            h_nb = torch.where(out.squeeze(-1), flow_h, h_fi)
            R = R.index_add(0, nb, q.unsqueeze(-1) * comp)  # q>0 removes, q<0 injects
            R_e = R_e + scatter_boundary_outflow(q * h_nb, nb, n)

        if sources is not None:
            R = R - sources.reshape(n, nc)
        if source_energy is not None:
            R_e = R_e - source_energy
        return torch.cat([R.reshape(-1), R_e])

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        **kw: float | torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(
            lambda s: self.residual(s, state_old, dt, **kw),
            state,
            vectorize=True,
        )


class MPFAThermalCompositionalModel3D(MPFAThermalCompositionalModel):
    """Multi-component (EOS-flash) thermal flow with a 3-D MPFA-O multi-point flux (SI).

    Args mirror :class:`MPFAThermalCompositionalModel`, with a :class:`MPFAGrid3D`
    and ``(n_cells, 3, 3)`` permeability tensors. Gravity is enabled by passing a
    per-cell ``depth_m``.
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalCompositionalModel3D",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("composition", "1", ("cell", "component"), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("total_molar", "mol/s", "molar", "pressure"),
            ("component_molar", "mol/s", "molar", "composition"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-3d",),
        phases=("liquid", "vapor"),
    )

    _MPFA_DIM = 3  # StencilInversionMixin → 3-D stencils

    def __init__(
        self,
        grid: MPFAGrid3D,
        perm_tensor: torch.Tensor,
        porosity: object,
        eos: CubicEOS,
        *,
        liquid_viscosity_pa_s: float = 5e-4,
        vapor_viscosity_pa_s: float = 2e-5,
        viscosity: str = "constant",
        residual_liquid_saturation: float = 0.0,
        residual_vapor_saturation: float = 0.0,
        liquid_corey_exponent: float = 2.0,
        vapor_corey_exponent: float = 2.0,
        component_heat_capacities_j_kg_k: object = 2100.0,
        rock_heat_capacity_j_kg_k: float = 1000.0,
        rock_density_kg_m3: float = 2650.0,
        liquid_thermal_conductivity_w_m_k: float = 0.13,
        vapor_thermal_conductivity_w_m_k: float = 0.03,
        rock_thermal_conductivity_w_m_k: float = 3.0,
        cell_volumes_m3: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_transmissibility_m3: torch.Tensor | None = None,
        depth_m: torch.Tensor | None = None,
        gravity_m_s2: float = 9.81,
        dirichlet: _BoundaryRecords | None = None,
        neumann: _BoundaryRecords | None = None,
        bhp_wells: _WellRecords | None = None,
        rate_wells: _WellRecords | None = None,
    ) -> None:
        # Build the 3-D Darcy machinery (interior L, face_lr, V_cell, phi, eos,
        # mobilities, NNC) via the 3-D compositional model. The isothermal parent
        # stores a frozen self.T; here T is a state variable, so the stored value
        # (300 K) is unused. The compositional parent has no BC/well machinery, so,
        # like the 2-D thermal compositional model: the forcings are built here
        # directly with the 3-D MPFA BC stencils.
        MPFACompositionalModel3D.__init__(
            self,
            grid,
            perm_tensor,
            porosity,
            eos,
            300.0,
            liquid_viscosity_pa_s=liquid_viscosity_pa_s,
            vapor_viscosity_pa_s=vapor_viscosity_pa_s,
            viscosity=viscosity,
            residual_liquid_saturation=residual_liquid_saturation,
            residual_vapor_saturation=residual_vapor_saturation,
            liquid_corey_exponent=liquid_corey_exponent,
            vapor_corey_exponent=vapor_corey_exponent,
            cell_volumes_m3=cell_volumes_m3,
            nnc_pairs=nnc_pairs,
            nnc_transmissibility_m3=nnc_transmissibility_m3,
        )
        components = tuple(eos.mixture.names)
        self.schema = _flow_model_schema(
            model_name=type(self).__name__,
            primary_fields=(
                ("pressure", "Pa", ("cell",), ()),
                ("composition", "1", ("cell", "component"), components),
                ("temperature", "K", ("cell",), ()),
            ),
            residual_blocks=(
                ("total_molar", "mol/s", "molar", "pressure"),
                ("component_molar", "mol/s", "molar", "composition"),
                ("energy", "W", "energy", "temperature"),
            ),
            grid_kinds=("mpfa-3d",),
            phases=("liquid", "vapor"),
            components=components,
        )
        dtype, device = perm_tensor.dtype, perm_tensor.device
        cp = cell_scalar_input(
            component_heat_capacities_j_kg_k,
            n_cells=self.nc,
            dtype=dtype,
            device=device,
            field="component_heat_capacities_j_kg_k",
            positive=True,
            object_name="MPFAThermalCompositionalModel3D",
        )
        self.register_buffer("cp_components", cp)
        self.cp_r = positive_real(
            rock_heat_capacity_j_kg_k,
            object_name="MPFAThermalCompositionalModel3D",
            field="rock_heat_capacity_j_kg_k",
        )
        self.rho_r = positive_real(
            rock_density_kg_m3,
            object_name="MPFAThermalCompositionalModel3D",
            field="rock_density_kg_m3",
        )
        self.lam_l, self.lam_v, self.lam_rock = (
            positive_real(
                liquid_thermal_conductivity_w_m_k,
                object_name="MPFAThermalCompositionalModel3D",
                field="liquid_thermal_conductivity_w_m_k",
            ),
            positive_real(
                vapor_thermal_conductivity_w_m_k,
                object_name="MPFAThermalCompositionalModel3D",
                field="vapor_thermal_conductivity_w_m_k",
            ),
            positive_real(
                rock_thermal_conductivity_w_m_k,
                object_name="MPFAThermalCompositionalModel3D",
                field="rock_thermal_conductivity_w_m_k",
            ),
        )
        self.g = nonnegative_real(
            gravity_m_s2,
            object_name="MPFAThermalCompositionalModel3D",
            field="gravity_m_s2",
        )
        if depth_m is not None:
            depth_tensor = cell_scalar_input(
                depth_m,
                n_cells=self.n_cells,
                dtype=dtype,
                device=device,
                field="depth_m",
                positive=False,
                object_name="MPFAThermalCompositionalModel3D",
            )
            self.register_buffer("depth", depth_tensor)
        else:
            self.depth = None

        # Dirichlet pressure boundary faces ``{f: (p_bc, z_bc, T_bc)}``; build the
        # ghost-augmented 3-D operator (overriding the parent's no-flow L/face_lr) +
        # ghost-augmented conduction, and precompute the constant boundary ghost phase
        # state (the inherited, geometry-agnostic _ghost_state); else interior only.
        if dirichlet:
            dir_faces = [int(f) for f in dirichlet]
            L, face_lr, dir_list = mpfa_o_face_flux_stencils_3d_bc(grid, perm_tensor, dir_faces)
            self.register_buffer("L", L)
            self.register_buffer("face_lr", face_lr)
            p_bc = torch.stack([torch.as_tensor(dirichlet[f][0], dtype=dtype) for f in dir_list])
            z_bc = torch.stack([torch.as_tensor(dirichlet[f][1], dtype=dtype) for f in dir_list])
            T_bc = torch.stack([torch.as_tensor(dirichlet[f][2], dtype=dtype) for f in dir_list])
            mm_l, mm_v, x_bc, y_bc, h_l_bc, h_v_bc, S_l_bc = self._ghost_state(p_bc, z_bc, T_bc)
            self.register_buffer("p_bc", p_bc)
            self.register_buffer("ghost_mm_l", mm_l)
            self.register_buffer("ghost_mm_v", mm_v)
            self.register_buffer("ghost_x", x_bc)
            self.register_buffer("ghost_y", y_bc)
            self.register_buffer("ghost_h_l", h_l_bc)
            self.register_buffer("ghost_h_v", h_v_bc)
            self.register_buffer("ghost_S_l", S_l_bc)
            self.register_buffer("T_bc", T_bc)
            _register_conduction_bc_3d(
                self, grid, self.phi, rock_thermal_conductivity_w_m_k, dir_faces, dtype
            )
            self.dir_list = dir_list  # ghost order; lets the inversion mixin
        else:  # rebuild the augmented stencil (perm/λ/φ)
            self.p_bc = None
            self.dir_list = None
            _register_conduction_3d(self, grid, self.phi, rock_thermal_conductivity_w_m_k, dtype)

        # Neumann / rate boundary faces ``{f: (q, z_inj, T_inj)}`` (q = outward total
        # molar rate; outflow produces the cell flowing fluid, inflow injects z_inj).
        if neumann:
            faces = list(neumann)
            fc = grid.face_cells
            self.register_buffer(
                "neumann_cells", torch.tensor([fc[f][0] for f in faces], dtype=torch.long)
            )
            self.register_buffer(
                "neumann_q",
                torch.stack([torch.as_tensor(neumann[f][0], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_z",
                torch.stack([torch.as_tensor(neumann[f][1], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_T_inj",
                torch.stack([torch.as_tensor(neumann[f][2], dtype=dtype) for f in faces]),
            )
        else:
            self.neumann_q = None

        # BHP-controlled Peaceman wells ``[(cell, WI, bhp, z_inj, T_inj), ...]``
        if bhp_wells:
            self.register_buffer(
                "well_cells",
                torch.tensor([_cell_index(w[0]) for w in bhp_wells], dtype=torch.long),
            )
            self.register_buffer(
                "well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in bhp_wells])
            )
            self.register_buffer(
                "well_bhp", torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in bhp_wells])
            )
            self.register_buffer(
                "well_z_inj", torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in bhp_wells])
            )
            self.register_buffer(
                "well_T_inj", torch.stack([torch.as_tensor(w[4], dtype=dtype) for w in bhp_wells])
            )
        else:
            self.well_cells = None

        register_rate_wells_comp(self, rate_wells, dtype)


__all__ = ["MPFAThermalCompositionalModel", "MPFAThermalCompositionalModel3D"]
