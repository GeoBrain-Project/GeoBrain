"""
Compositional (EOS-flash) reservoir model on a general 2-D grid using the MPFA-O
multi-point flux.

The MPFA-O counterpart of :class:`CompositionalModel`: the per-cell vapor-liquid
flash, cubic-EOS molar densities, Corey mobilities and the component-mole balance
``q_i = q_L·x_i + q_V·y_i`` are identical, but the two-point geometric flux
``T_geom·Δp`` is replaced by the **MPFA-O multi-point flux** ``G_f = Σ_c T_c·p_c``
(stencils from :func:`mpfa_o_face_flux_stencils_full`, built once from geometry +
absolute permeability), so the discretization is consistent on non-K-orthogonal /
full-tensor grids. For ``n_c`` components the unknowns per cell are
``(p, z_1, …, z_{n_c−1})`` and the residual is the ``n_c`` component-mole
balances (SI units, mol)::

    R_i = V_c·(N_i − N_i^old)/Δt  +  Σ_f (±F_{f,i})  −  src_i ,
    N_i = φ·z_i / D ,   D = (1−V)·v_L + V·v_V ,   v_p = Z_p·R·T/p
    F_{f,i} = q_L·x_{f,i} + q_V·y_{f,i} ,  q_p = (ρ_p·k_rp/μ_p)|_up · G_f

with the upwind cell chosen by the sign of ``G_f``. The flash is run per cell each
residual evaluation and is end-to-end differentiable (implicit-function-theorem
cleanup), so the whole residual carries gradients in pressure / composition /
permeability. On a K-orthogonal grid the stencil collapses to two-point and this
reproduces :class:`CompositionalModel` exactly.

Boundaries are no-flow (interior MPFA faces only); flow is driven by per-cell,
per-component molar sources. v1 scope mirrors :class:`CompositionalModel`:
two-phase feed, constant phase viscosities, Corey relperm on the liquid
saturation, isothermal.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from ..discretization.mpfa import MPFAGrid2D, _stable_polygon_areas, mpfa_o_face_flux_stencils_full
from ..discretization.mpfa3d import MPFAGrid3D, hex_cell_volumes, mpfa_o_face_flux_stencils_3d_full
from ..errors import FlowContractError
from .._state_validation import cell_scalar_input, composition_input, phase_model_config
from .cubic_eos import CubicEOS
from .flash import flash, require_flash_converged
from .viscosity import lbc_viscosity

_R_GAS = 8.31446261815324  # J/(mol·K)


class MPFACompositionalModel(nn.Module):  # type: ignore[misc]  # skipped torch boundary
    """Multi-component (EOS-flash) flow with an MPFA-O multi-point flux (2-D, SI).

    Args:
        grid: :class:`MPFAGrid2D`.
        perm_tensor: ``(n_cells, 2, 2)`` cell permeability tensors [m²].
        porosity: scalar or ``(n_cells,)``.
        eos: a :class:`CubicEOS` (its mixture defines the components).
        temperature: isothermal reservoir temperature [K].
        liquid_viscosity_pa_s / vapor_viscosity_pa_s: phase viscosities [Pa·s].
        swl / sgr / n_l / n_v: Corey liquid/vapor relperm endpoints / exponents.
    """

    schema = _flow_model_schema(
        model_name="MPFACompositionalModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("composition", "1", ("cell", "component"), ()),
        ),
        residual_blocks=(
            ("total_molar", "mol/s", "molar", "pressure"),
            ("component_molar", "mol/s", "molar", "composition"),
        ),
        grid_kinds=("mpfa-2d",),
    phases=("liquid", "vapor"),
    )

    # The 3-D implementation reuses the shared compositional state and schema
    # machinery while supplying its own grid-specific assembly.
    grid: MPFAGrid2D | MPFAGrid3D

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: object,
        eos: CubicEOS,
        temperature_k: float,
        *,
        liquid_viscosity_pa_s: float = 5e-4,
        vapor_viscosity_pa_s: float = 2e-5,
        viscosity: str = "constant",
        residual_liquid_saturation: float = 0.0,
        residual_vapor_saturation: float = 0.0,
        liquid_corey_exponent: float = 2.0,
        vapor_corey_exponent: float = 2.0,
        cell_volumes_m3: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_transmissibility_m3: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if viscosity not in ("constant", "lbc"):
            raise GeoBrainError(
                "viscosity must be 'constant' or 'lbc'",
                object_name="MPFACompositionalModel",
                field="viscosity",
                expected="'constant'|'lbc'",
                actual=viscosity,
            )
        self.grid = grid
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
            object_name="MPFACompositionalModel",
        )
        self.viscosity = viscosity
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
            grid_kinds=("mpfa-2d",),
            phases=("liquid", "vapor"),
            components=components,
        )
        n = len(grid.cell_nodes)
        self._n_cells = n
        if (
            not isinstance(perm_tensor, torch.Tensor)
            or not perm_tensor.is_floating_point()
            or perm_tensor.shape != (n, 2, 2)
            or not bool(torch.isfinite(perm_tensor).all())
        ):
            raise FlowContractError(
                "permeability_m2 must be a finite tensor aligned with the grid",
                object_name="MPFACompositionalModel",
                field="permeability_m2",
                expected=(n, 2, 2),
                actual=(
                    type(perm_tensor).__name__,
                    tuple(getattr(perm_tensor, "shape", ())),
                ),
            )
        dtype = perm_tensor.dtype
        device = perm_tensor.device
        if eos.mixture.dtype != dtype or eos.mixture.device != device:
            raise FlowContractError(
                "EOS metadata must match permeability_m2",
                object_name="MPFACompositionalModel",
                field="eos.dtype/device",
                expected=(str(dtype), str(device)),
                actual=(str(eos.mixture.dtype), str(eos.mixture.device)),
            )
        phi = cell_scalar_input(
            porosity,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="porosity",
            positive=True,
            object_name="MPFACompositionalModel",
        )
        if bool((phi >= 1).any()):
            raise FlowContractError(
                "porosity must be less than one",
                object_name="MPFACompositionalModel",
                field="porosity",
                expected="0 < porosity < 1",
                actual="contains value >= 1",
            )
        self.register_buffer("phi", phi)
        if cell_volumes_m3 is None:
            cell_volumes_m3 = self._polygon_areas(grid)
        volumes = cell_scalar_input(
            cell_volumes_m3,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="cell_volumes_m3",
            positive=True,
            object_name="MPFACompositionalModel",
        )
        self.register_buffer("V_cell", volumes)

        stencils = mpfa_o_face_flux_stencils_full(grid, perm_tensor)
        faces = sorted(stencils)
        L = perm_tensor.new_zeros(len(faces), n)
        lr = []
        for fi, f in enumerate(faces):
            for cc, t in stencils[f].items():
                L[fi, cc] = t
            left, right = grid.edge_cells[f]
            lr.append([left, right])
        self.register_buffer("L", L)
        self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long, device=device))
        from ..models.mpfa_two_phase import register_nnc

        register_nnc(self, nnc_pairs, nnc_transmissibility_m3, dtype)

    @staticmethod
    def _polygon_areas(grid: MPFAGrid2D) -> torch.Tensor:
        return _stable_polygon_areas(grid)

    @property
    def n_cells(self) -> int:
        return self._n_cells

    def state_size(self) -> int:
        return self.n_cells * self.nc

    def initial_state(self, pressure: object, z: object) -> torch.Tensor:
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
        return torch.cat([p, z_tensor[:, : self.nc - 1].reshape(-1)])

    def _unpack(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, nc = self.n_cells, self.nc
        p = state[:n]
        z_red = state[n:].reshape(n, nc - 1)
        z = torch.cat([z_red, 1.0 - z_red.sum(dim=-1, keepdim=True)], dim=-1)
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
        """Flash → ``(V, x, y, S_l, rho_l, rho_v, D)`` per cell (molar densities, sat)."""
        T = p.new_full((), self.T)
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

    def _mobilities(
        self,
        S_l: torch.Tensor,
        mu_l: float | torch.Tensor | None = None,
        mu_v: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Corey liquid/vapor phase mobilities ``k_r/μ``. ``μ`` defaults to the
        constant ``mu_liquid``/``mu_vapor``; pass per-cell viscosities (e.g. LBC)
        to override."""
        se = ((S_l - self.swl) / (1.0 - self.swl - self.sgr + 1e-30)).clamp(0.0, 1.0)
        mu_l = self.mu_l if mu_l is None else mu_l
        mu_v = self.mu_v if mu_v is None else mu_v
        return se.pow(self.n_l) / mu_l, (1.0 - se).pow(self.n_v) / mu_v

    def _phase_viscosities(
        self,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        rho_l: torch.Tensor,
        rho_v: torch.Tensor,
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """``(μ_l, μ_v)``: constant, or per-cell Lohrenz-Bray-Clark from the
        phase composition + compressibility (``Z = p/(ρ·R·T)``)."""
        if self.viscosity != "lbc":
            return self.mu_l, self.mu_v
        Z_l = p / (rho_l * _R_GAS * self.T)
        Z_v = p / (rho_v * _R_GAS * self.T)
        mix = self.eos.mixture
        return (lbc_viscosity(mix, p, self.T, x, Z_l), lbc_viscosity(mix, p, self.T, y, Z_v))

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, nc = self.n_cells, self.nc
        if state.shape != (n * nc,) or state_old.shape != (n * nc,):
            raise GeoBrainError(
                "MPFACompositionalModel state must be length n_cells·n_components",
                object_name="MPFACompositionalModel",
                field="state",
                expected=(n * nc,),
                actual=tuple(state.shape),
            )
        p, z = self._unpack(state)
        p_o, z_o = self._unpack(state_old)

        V, x, y, S_l, rho_l, rho_v, Dmix = self._phase_state(p, z)
        _, _, _, _, _, _, Dmix_o = self._phase_state(p_o, z_o)

        N = (self.phi / Dmix).unsqueeze(-1) * z
        N_o = (self.phi / Dmix_o).unsqueeze(-1) * z_o
        acc = self.V_cell.unsqueeze(-1) * (N - N_o) / float(dt)  # (n_cells, nc)
        R = acc

        G = self.L @ p  # MPFA geometric flux (left→right)
        left_cells, right_cells = self.face_lr[:, 0], self.face_lr[:, 1]
        up = (G >= 0).unsqueeze(-1)
        mu_l, mu_v = self._phase_viscosities(p, x, y, rho_l, rho_v)
        mob_l, mob_v = self._mobilities(S_l, mu_l, mu_v)
        ql = (
            torch.where(
                G >= 0,
                (rho_l * mob_l)[left_cells],
                (rho_l * mob_l)[right_cells],
            )
            * G
        )
        qv = (
            torch.where(
                G >= 0,
                (rho_v * mob_v)[left_cells],
                (rho_v * mob_v)[right_cells],
            )
            * G
        )
        x_f = torch.where(up, x[left_cells], x[right_cells])
        y_f = torch.where(up, y[left_cells], y[right_cells])
        F = ql.unsqueeze(-1) * x_f + qv.unsqueeze(-1) * y_f  # (n_faces, nc)
        R = R.index_add(0, left_cells, F).index_add(0, right_cells, -F)

        if self.nnc_trans is not None:  # fault NNCs (two-point component flux)
            a, b = self.nnc_pairs[:, 0], self.nnc_pairs[:, 1]
            Gn = self.nnc_trans * (p[a] - p[b])
            upn = (Gn >= 0).unsqueeze(-1)
            qln = torch.where(Gn >= 0, (rho_l * mob_l)[a], (rho_l * mob_l)[b]) * Gn
            qvn = torch.where(Gn >= 0, (rho_v * mob_v)[a], (rho_v * mob_v)[b]) * Gn
            Fn = qln.unsqueeze(-1) * torch.where(upn, x[a], x[b]) + qvn.unsqueeze(-1) * torch.where(
                upn, y[a], y[b]
            )
            R = R.index_add(0, a, Fn).index_add(0, b, -Fn)

        if sources is not None:
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


class MPFACompositionalModel3D(MPFACompositionalModel):
    """Multi-component (EOS-flash) flow with a 3-D MPFA-O multi-point flux (SI).

    Args mirror :class:`MPFACompositionalModel`, with a :class:`MPFAGrid3D` and
    ``(n_cells, 3, 3)`` permeability tensors.
    """

    schema = _flow_model_schema(
        model_name="MPFACompositionalModel3D",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("composition", "1", ("cell", "component"), ()),
        ),
        residual_blocks=(
            ("total_molar", "mol/s", "molar", "pressure"),
            ("component_molar", "mol/s", "molar", "composition"),
        ),
        grid_kinds=("mpfa-3d",),
        phases=("liquid", "vapor"),
    )

    def __init__(
        self,
        grid: MPFAGrid3D,
        perm_tensor: torch.Tensor,
        porosity: object,
        eos: CubicEOS,
        temperature_k: float,
        *,
        liquid_viscosity_pa_s: float = 5e-4,
        vapor_viscosity_pa_s: float = 2e-5,
        viscosity: str = "constant",
        residual_liquid_saturation: float = 0.0,
        residual_vapor_saturation: float = 0.0,
        liquid_corey_exponent: float = 2.0,
        vapor_corey_exponent: float = 2.0,
        cell_volumes_m3: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_transmissibility_m3: torch.Tensor | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        if viscosity not in ("constant", "lbc"):
            raise GeoBrainError(
                "viscosity must be 'constant' or 'lbc'",
                object_name="MPFACompositionalModel3D",
                field="viscosity",
                expected="'constant'|'lbc'",
                actual=viscosity,
            )
        self.grid = grid
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
            object_name="MPFACompositionalModel3D",
        )
        self.viscosity = viscosity
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
            grid_kinds=("mpfa-3d",),
            phases=("liquid", "vapor"),
            components=components,
        )
        n = len(grid.cell_nodes)
        self._n_cells = n
        if (
            not isinstance(perm_tensor, torch.Tensor)
            or not perm_tensor.is_floating_point()
            or perm_tensor.shape != (n, 3, 3)
            or not bool(torch.isfinite(perm_tensor).all())
        ):
            raise FlowContractError(
                "permeability_m2 must be a finite tensor aligned with the grid",
                object_name="MPFACompositionalModel3D",
                field="permeability_m2",
                expected=(n, 3, 3),
                actual=(
                    type(perm_tensor).__name__,
                    tuple(getattr(perm_tensor, "shape", ())),
                ),
            )
        dtype = perm_tensor.dtype
        device = perm_tensor.device
        if eos.mixture.dtype != dtype or eos.mixture.device != device:
            raise FlowContractError(
                "EOS metadata must match permeability_m2",
                object_name="MPFACompositionalModel3D",
                field="eos.dtype/device",
                expected=(str(dtype), str(device)),
                actual=(str(eos.mixture.dtype), str(eos.mixture.device)),
            )
        phi = cell_scalar_input(
            porosity,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="porosity",
            positive=True,
            object_name="MPFACompositionalModel3D",
        )
        if bool((phi >= 1).any()):
            raise FlowContractError(
                "porosity must be less than one",
                object_name="MPFACompositionalModel3D",
                field="porosity",
                expected="0 < porosity < 1",
                actual="contains value >= 1",
            )
        self.register_buffer("phi", phi)
        if cell_volumes_m3 is None:
            cell_volumes_m3 = hex_cell_volumes(grid)
        volumes = cell_scalar_input(
            cell_volumes_m3,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="cell_volumes_m3",
            positive=True,
            object_name="MPFACompositionalModel3D",
        )
        self.register_buffer("V_cell", volumes)

        stencils = mpfa_o_face_flux_stencils_3d_full(grid, perm_tensor)
        faces = sorted(stencils)
        L = perm_tensor.new_zeros(len(faces), n)
        lr = []
        for fi, f in enumerate(faces):
            for cc, t in stencils[f].items():
                L[fi, cc] = t
            left, right = grid.face_cells[f]
            lr.append([left, right])
        self.register_buffer("L", L)
        self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long, device=device))
        from ..models.mpfa_two_phase import register_nnc

        register_nnc(self, nnc_pairs, nnc_transmissibility_m3, dtype)


__all__ = ["MPFACompositionalModel", "MPFACompositionalModel3D"]
