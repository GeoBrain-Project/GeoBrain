"""Single-phase **thermal** reservoir model: coupled mass + energy conservation.

Adds an energy (temperature) equation to single-phase Darcy flow, a total
thermal-energy conservation law. Temperature ``T`` becomes a
primary variable alongside pressure; heat moves both by advection (the flowing
fluid carries its enthalpy) and by conduction through rock + fluid.

Per cell the unknowns are ``(p, T)`` and the two balances are::

    mass:   V·(φρ − φρ|old)/Δt + Σ_faces ρ_up·(1/μ)·T_geom·Δp − q_m = 0
    energy: V·(E − E|old)/Δt + Σ_faces (H_up·F_mass − λ_T·T_geom·ΔT) − q_e = 0
    E = (1−φ)·ρ_r·C_r·T + φ·ρ·C_f·T            (rock + fluid internal energy / volume)
    H = C_f·T + p/ρ                              (specific enthalpy = u + p/ρ)
    λ_T = (1−φ)·λ_rock + φ·λ_fluid               (bulk thermal conductivity)

Internal energy is the linear ``U = C·T`` model;
fluid density ``ρ(p, T) = ρ_ref·(1 + c_f·(p−p_ref) − α_T·(T−T_ref))`` carries the
optional thermal-expansion coupling. Units are SI (Pa, K, m, m², kg, J).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from ..discretization.flux import scatter_internal_face_flux, upwind_cell
from ..grid import FlowGrid
from ..properties import Rock
from .._state_validation import cell_scalar_input, nonnegative_real, positive_real


class ThermalSinglePhaseModel(nn.Module):  # type: ignore[misc]  # skipped torch boundary
    """Single-phase flow with an energy balance (temperature). SI units.

    Args:
        grid, rock: geometry (m, m²) and rock (perm + porosity).
        density_ref_kg_m3, viscosity_pa_s, reference_pressure_pa: reference
            fluid properties in canonical SI.
        fluid_compressibility_pa_inv: ``c_f`` [Pa⁻¹];
            thermal_expansion_k_inv: ``α_T`` [K⁻¹].
        fluid_heat_capacity_j_kg_k, rock_heat_capacity_j_kg_k: specific heat
            capacities [J/(kg·K)].
        rock_density_kg_m3: rock grain density [kg/m³].
        fluid_thermal_conductivity_w_m_k, rock_thermal_conductivity_w_m_k:
            thermal conductivities [W/(m·K)].
        reference_temperature_k: reference temperature [K].
    """

    schema = _flow_model_schema(
        model_name="ThermalSinglePhaseModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("fluid_mass", "kg/s", "mass", "pressure"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("cartesian",),
        phases=("fluid",),
    )

    def __init__(
        self,
        grid: FlowGrid,
        rock: Rock,
        *,
        density_ref_kg_m3: float = 1000.0,
        viscosity_pa_s: float = 1e-3,
        reference_pressure_pa: float = 1e7,
        fluid_compressibility_pa_inv: float = 1e-9,
        thermal_expansion_k_inv: float = 0.0,
        fluid_heat_capacity_j_kg_k: float = 4200.0,
        rock_heat_capacity_j_kg_k: float = 1000.0,
        rock_density_kg_m3: float = 2650.0,
        fluid_thermal_conductivity_w_m_k: float = 0.6,
        rock_thermal_conductivity_w_m_k: float = 3.0,
        reference_temperature_k: float = 300.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.rock = rock
        object_name = "ThermalSinglePhaseModel"
        self.rho_ref = positive_real(
            density_ref_kg_m3, object_name=object_name, field="density_ref_kg_m3"
        )
        self.mu = positive_real(viscosity_pa_s, object_name=object_name, field="viscosity_pa_s")
        self.p_ref = positive_real(
            reference_pressure_pa, object_name=object_name, field="reference_pressure_pa"
        )
        self.c_f = nonnegative_real(
            fluid_compressibility_pa_inv,
            object_name=object_name,
            field="fluid_compressibility_pa_inv",
        )
        self.alpha_T = nonnegative_real(
            thermal_expansion_k_inv,
            object_name=object_name,
            field="thermal_expansion_k_inv",
        )
        self.cp_f = positive_real(
            fluid_heat_capacity_j_kg_k,
            object_name=object_name,
            field="fluid_heat_capacity_j_kg_k",
        )
        self.cp_r = positive_real(
            rock_heat_capacity_j_kg_k,
            object_name=object_name,
            field="rock_heat_capacity_j_kg_k",
        )
        self.rho_r = positive_real(
            rock_density_kg_m3, object_name=object_name, field="rock_density_kg_m3"
        )
        self.lam_f = positive_real(
            fluid_thermal_conductivity_w_m_k,
            object_name=object_name,
            field="fluid_thermal_conductivity_w_m_k",
        )
        self.lam_r = positive_real(
            rock_thermal_conductivity_w_m_k,
            object_name=object_name,
            field="rock_thermal_conductivity_w_m_k",
        )
        self.T_ref = positive_real(
            reference_temperature_k, object_name=object_name, field="reference_temperature_k"
        )

    @property
    def n_cells(self) -> int:
        return int(self.grid.n_cells)

    def state_size(self) -> int:
        return 2 * self.n_cells

    # ------------------------------------------------------------------
    def initial_state(self, pressure: object, temperature: object) -> torch.Tensor:
        n = self.n_cells
        dev, dt = self.grid.device, self.grid.dtype
        p = cell_scalar_input(
            pressure,
            n_cells=n,
            dtype=dt,
            device=dev,
            field="pressure_pa",
            positive=True,
            object_name="ThermalSinglePhaseModel.initial_state",
        )
        T = cell_scalar_input(
            temperature,
            n_cells=n,
            dtype=dt,
            device=dev,
            field="temperature_k",
            positive=True,
            object_name="ThermalSinglePhaseModel.initial_state",
        )
        return torch.cat([p, T])

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        return {"p": state[:n], "T": state[n : 2 * n]}

    def accepted_discrete_masks(
        self,
        state: torch.Tensor,
    ) -> dict[str, tuple[bool, ...]]:
        """Return the detached, JSON-ready advective upwind choices."""
        with torch.no_grad():
            pressure = state[: self.n_cells].detach()
            connection = self.grid._connection_metrics()
            if connection is None or connection.n_faces == 0:
                return {"fluid_upwind_left": ()}
            left, right = connection.neighbors[:, 0], connection.neighbors[:, 1]
            pressure_drop = pressure[left] - pressure[right]
            return {
                "fluid_upwind_left": tuple(
                    bool(value) for value in (pressure_drop >= 0)
                )
            }

    def density(self, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self.rho_ref * (1.0 + self.c_f * (p - self.p_ref) - self.alpha_T * (T - self.T_ref))

    # ------------------------------------------------------------------
    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        source_mass: torch.Tensor | None = None,
        source_energy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``concat(R_mass, R_energy)`` of length ``2·n_cells``."""
        n = self.n_cells
        if state.shape != (2 * n,) or state_old.shape != (2 * n,):
            raise GeoBrainError(
                "ThermalSinglePhaseModel state must be length 2*n_cells",
                object_name="ThermalSinglePhaseModel",
                field="state",
                expected=(2 * n,),
                actual=tuple(state.shape),
            )
        p, T = state[:n], state[n : 2 * n]
        p_o, T_o = state_old[:n], state_old[n : 2 * n]
        Vc = self.grid._cell_volumes_view()
        phi = self.rock.porosity_at_pressure(p)
        phi_o = self.rock.porosity_at_pressure(p_o)
        rho, rho_o = self.density(p, T), self.density(p_o, T_o)

        # internal energy per bulk volume: rock (1−φ)ρ_r C_r T + fluid φ ρ C_f T
        U_f = self.cp_f * T
        E = (1.0 - phi) * self.rho_r * self.cp_r * T + phi * rho * U_f
        U_f_o = self.cp_f * T_o
        E_o = (1.0 - phi_o) * self.rho_r * self.cp_r * T_o + phi_o * rho_o * U_f_o
        H = U_f + p / rho  # specific enthalpy

        acc_m = Vc * (phi * rho - phi_o * rho_o) / float(dt)
        acc_e = Vc * (E - E_o) / float(dt)
        R_m = acc_m
        R_e = acc_e

        c = self.grid._connection_metrics()
        if c is not None and c.n_faces > 0:
            T_geom = self.grid.build_transmissibility(self.rock.permeability_m2)
            cl, cr = c.neighbors[:, 0], c.neighbors[:, 1]
            dp = p[cl] - p[cr]
            dT = T[cl] - T[cr]
            upstream = upwind_cell(dp, c.neighbors)
            rho_up = rho[upstream]
            F_mass = rho_up * (1.0 / self.mu) * T_geom * dp
            H_up = H[upstream]
            lam_bulk = (1.0 - phi) * self.lam_r + phi * self.lam_f
            lam_face = 0.5 * (lam_bulk[cl] + lam_bulk[cr])
            # conduction uses the *geometric* face transmissibility A/d (no
            # permeability): heat conducts through the rock+fluid bulk. The
            # Fourier current cl→cr is +λ·(A/d)·(T_cl−T_cr): heat flows down the
            # gradient (same cl→cr sign convention as the mass flux).
            T_cond = self.grid.build_transmissibility(torch.ones_like(self.rock.permeability_m2))
            F_cond = lam_face * T_cond * dT  # Fourier conduction
            F_energy = H_up * F_mass + F_cond
            R_m = R_m + scatter_internal_face_flux(F_mass, c.neighbors, n)
            R_e = R_e + scatter_internal_face_flux(F_energy, c.neighbors, n)

        if source_mass is not None:
            R_m = R_m - source_mass
        if source_energy is not None:
            R_e = R_e - source_energy
        return torch.cat([R_m, R_e])

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        **kw: torch.Tensor | None,
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(
            lambda s: self.residual(s, state_old, dt, **kw),
            state,
            vectorize=True,
        )


__all__ = ["ThermalSinglePhaseModel"]
