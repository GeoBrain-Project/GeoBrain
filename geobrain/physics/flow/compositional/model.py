"""Compositional reservoir model: EOS flash coupled into the flow residual.

This is the top of the compositional stack: per cell, a vapor-liquid flash maps
the overall composition ``z`` at ``(p, T)`` to the phase split (``V``, liquid
``x``, vapor ``y``) and, via the cubic-EOS compressibility factors, the phase
molar densities and saturations; component **moles** are then conserved on the
grid (accumulation + TPFA inter-cell flux), with component fluxes
``q_i = q_L·x_i + q_V·y_i``.

For ``n_c`` components the unknowns per cell are ``(p, z_1, …, z_{n_c−1})``
(``z_{n_c}=1−Σ``) and the residual is the ``n_c`` component-mole balances::

    R_i = [N_i − N_i^old]/Δt − Σ_faces (q_L·x_{f,i} + q_V·y_{f,i}) − src_i
    N_i = φ · z_i / D ,   D = (1−V)·v_L + V·v_V ,   v_p = Z_p·R·T / p
    q_p = ρ_p · (k_rp/μ_p) · T_geom · Δp     (phase molar flux, upwinded)

Units are **SI** (Pa, K, m, m², mol). The flash is run per cell each residual
evaluation and is end-to-end differentiable (implicit-function-theorem cleanup),
so the whole compositional residual carries gradients.

v1 scope: assumes a two-phase feed (single-phase handling, gravity, an LBC
viscosity correlation and tabulated relperm are deferred); phase viscosities are
constants and relperm is Corey on the liquid saturation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from ..grid import FlowGrid
from ..properties import Rock
from .._state_validation import cell_scalar_input, composition_input, phase_model_config
from .cubic_eos import CubicEOS
from .flash import flash, require_flash_converged

_R_GAS = 8.31446261815324  # J/(mol·K)


class CompositionalModel(nn.Module):  # type: ignore[misc]  # skipped torch boundary
    """Multi-component (EOS-flash) reservoir flow on a grid (SI units).

    Args:
        grid:  any :class:`FlowGrid` (geometry in metres; perm in m²).
        rock:  :class:`Rock` (perm + porosity).
        eos:   a :class:`CubicEOS` (its mixture defines the components).
        temperature: isothermal reservoir temperature [K].
        liquid_viscosity_pa_s / vapor_viscosity_pa_s: phase viscosities [Pa·s].
        swl / sgr / n_l / n_v: Corey liquid/vapor relperm endpoints/exponents
            on the liquid saturation.
    """

    schema = _flow_model_schema(
        model_name="CompositionalModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("composition", "1", ("cell", "component"), ()),
        ),
        residual_blocks=(
            ("total_molar", "mol/s", "molar", "pressure"),
            ("component_molar", "mol/s", "molar", "composition"),
        ),
        grid_kinds=("cartesian",),
        phases=("liquid", "vapor"),
    )

    def __init__(
        self,
        grid: FlowGrid,
        rock: Rock,
        eos: CubicEOS,
        temperature_k: float,
        *,
        liquid_viscosity_pa_s: float = 5e-4,
        vapor_viscosity_pa_s: float = 2e-5,
        residual_liquid_saturation: float = 0.0,
        residual_vapor_saturation: float = 0.0,
        liquid_corey_exponent: float = 2.0,
        vapor_corey_exponent: float = 2.0,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.rock = rock
        self.eos = eos
        (
            self.T,
            self.mu_l,
            self.mu_v,
            self.swl,
            self.sgr,
            self.n_l,
            self.n_v,
        ) = phase_model_config(
            temperature=temperature_k,
            mu_liquid=liquid_viscosity_pa_s,
            mu_vapor=vapor_viscosity_pa_s,
            swl=residual_liquid_saturation,
            sgr=residual_vapor_saturation,
            n_l=liquid_corey_exponent,
            n_v=vapor_corey_exponent,
            object_name="CompositionalModel",
        )
        self.nc = eos.n_components
        components = tuple(eos.mixture.names)
        self.schema = _flow_model_schema(
            model_name=type(self).__name__,
            primary_fields=(
                ("pressure", "Pa", ("cell",), ()),
                ("composition", "1", ("cell", "component"), components),
            ),
            residual_blocks=(
                ("total_molar", "mol/s", "molar", "pressure"),
                ("component_molar", "mol/s", "molar", "composition"),
            ),
            grid_kinds=("cartesian",),
            phases=("liquid", "vapor"),
            components=components,
        )

    @property
    def n_cells(self) -> int:
        return int(self.grid.n_cells)

    def state_size(self) -> int:
        return self.n_cells * self.nc  # p + (nc−1) overall mole fractions

    # ------------------------------------------------------------------
    def initial_state(self, pressure: object, z: object) -> torch.Tensor:
        """Pack ``(p, z_1, …, z_{nc−1})``. ``z`` is the overall composition."""
        n = self.n_cells
        dev, dt = self.grid.device, self.grid.dtype
        p = cell_scalar_input(
            pressure,
            n_cells=n,
            dtype=dt,
            device=dev,
            field="pressure_pa",
            positive=True,
        )
        z_tensor = composition_input(
            z,
            n_cells=n,
            n_components=self.nc,
            dtype=dt,
            device=dev,
        )
        z_red = z_tensor[:, : self.nc - 1].reshape(-1)
        return torch.cat([p, z_red])

    def _unpack(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, nc = self.n_cells, self.nc
        p = state[:n]
        z_red = state[n:].reshape(n, nc - 1)
        z_last = 1.0 - z_red.sum(dim=-1, keepdim=True)
        z = torch.cat([z_red, z_last], dim=-1)
        return p, z

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        p, z = self._unpack(state)
        return {"p": p, "z": z}

    def accepted_discrete_masks(self, state: torch.Tensor) -> dict[str, tuple[bool, ...]]:
        """Return the converged flash regime selected in each cell."""
        p, z = self._unpack(state)
        vapor_fraction = self._phase_state(p, z)[0].detach()
        return {
            "liquid_only": tuple(bool(value) for value in (vapor_fraction <= 0.0)),
            "two_phase": tuple(
                bool(value)
                for value in ((vapor_fraction > 0.0) & (vapor_fraction < 1.0))
            ),
            "vapor_only": tuple(bool(value) for value in (vapor_fraction >= 1.0)),
        }

    # ------------------------------------------------------------------
    def _phase_state(
        self,
        p: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Flash → ``(V, x, y, S_l, rho_l, rho_v)`` per cell (molar densities, sat)."""
        T = p.new_full((), self.T)
        res = flash(self.eos, p, T, z)
        require_flash_converged(res, object_name="CompositionalModel")
        V = res.V
        x, y = res.x, res.y
        A_i, B_i = self.eos.ab_components(p, T)
        A_l, B_l, _ = self.eos.mixture_ab(x, A_i, B_i)
        A_v, B_v, _ = self.eos.mixture_ab(y, A_i, B_i)
        Z_l = self.eos.compressibility(A_l, B_l, root="liquid")
        Z_v = self.eos.compressibility(A_v, B_v, root="vapor")
        v_l = Z_l * _R_GAS * T / p  # liquid molar volume [m³/mol]
        v_v = Z_v * _R_GAS * T / p
        rho_l = 1.0 / v_l  # molar density [mol/m³]
        rho_v = 1.0 / v_v
        D = (1.0 - V) * v_l + V * v_v  # mixture molar volume [m³/mol]
        S_v = (V * v_v) / D  # vapor volume fraction
        S_l = 1.0 - S_v
        return V, x, y, S_l, rho_l, rho_v, D

    def _mobilities(self, S_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Corey liquid/vapor relperm over μ (phase mobility λ = k_r/μ)."""
        se = ((S_l - self.swl) / (1.0 - self.swl - self.sgr + 1e-30)).clamp(0.0, 1.0)
        kr_l = se.pow(self.n_l)
        kr_v = (1.0 - se).pow(self.n_v)
        return kr_l / self.mu_l, kr_v / self.mu_v

    # ------------------------------------------------------------------
    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Component-mole balance ``R[n_cells·n_c]`` = concat over components."""
        n, nc = self.n_cells, self.nc
        if state.shape != (n * nc,) or state_old.shape != (n * nc,):
            raise GeoBrainError(
                "CompositionalModel state must be length n_cells·n_components",
                object_name="CompositionalModel",
                field="state",
                expected=(n * nc,),
                actual=tuple(state.shape),
            )
        p, z = self._unpack(state)
        p_o, z_o = self._unpack(state_old)
        phi = self.rock.porosity_at_pressure(p)
        phi_o = self.rock.porosity_at_pressure(p_o)
        Vc = self.grid._cell_volumes_view()

        V, x, y, S_l, rho_l, rho_v, Dmix = self._phase_state(p, z)
        _, _, _, S_l_o, _, _, Dmix_o = self._phase_state(p_o, z_o)

        # Accumulation: N_i = φ · z_i / D  (component moles per pore volume).
        N = (phi / Dmix).unsqueeze(-1) * z
        N_o = (phi_o / Dmix_o).unsqueeze(-1) * z_o
        acc = Vc.unsqueeze(-1) * (N - N_o) / float(dt)  # (n_cells, nc)

        R = acc
        c = self.grid._connection_metrics()
        if c is not None and c.n_faces > 0:
            T_geom = self.grid.build_transmissibility(self.rock.permeability_m2)
            cl, cr = c.neighbors[:, 0], c.neighbors[:, 1]
            dp = p[cl] - p[cr]
            lam_l, lam_v = self._mobilities(S_l)
            # phase molar flux q_p = ρ_p·λ_p·T_geom·Δp, single-point upwind on Δp
            up = dp >= 0
            q_l = torch.where(up, (rho_l * lam_l)[cl], (rho_l * lam_l)[cr]) * T_geom * dp
            q_v = torch.where(up, (rho_v * lam_v)[cl], (rho_v * lam_v)[cr]) * T_geom * dp
            # component flux: q_L·x_f,i + q_V·y_f,i (upwind compositions on the phase Δp)
            x_f = torch.where(up.unsqueeze(-1), x[cl], x[cr])
            y_f = torch.where(up.unsqueeze(-1), y[cl], y[cr])
            F = q_l.unsqueeze(-1) * x_f + q_v.unsqueeze(-1) * y_f  # (n_faces, nc)
            R = R.index_add(0, cl, F).index_add(0, cr, -F)

        if sources is not None:
            # per-cell, per-component molar source [mol/s] (positive = injection)
            R = R - sources.reshape(n, nc)
        return R.reshape(-1)

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


__all__ = ["CompositionalModel"]
