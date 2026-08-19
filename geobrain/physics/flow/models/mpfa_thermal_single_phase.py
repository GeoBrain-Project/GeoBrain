"""
Single-phase **thermal** flow on a general 2-D grid using the MPFA-O multi-point
flux, coupled mass + energy conservation.

The MPFA-O counterpart of :class:`ThermalSinglePhaseModel`: temperature ``T`` is a
primary variable alongside pressure ``p``; heat moves by advection (the flowing
fluid carries its enthalpy) and by conduction. Both the Darcy mass flux **and**
the Fourier conduction flux use the multi-point MPFA-O geometric flux ``G_f =
Σ_c T_c·φ_c`` instead of two-point, so each is consistent on non-K-orthogonal /
full-tensor grids. Conduction follows the parallel-conductance model (as
in the multiphase thermal models): a rock stencil ``G^rock = L_rock @ T`` (built
from ``(1−φ)λ_rock·I``) plus a porosity-weighted fluid stencil ``G^φ = L_phi @ T``
(built from ``φ·I``) scaled by the fluid conductivity ``λ_f`` (the single-phase
saturation is ≡ 1). Per cell::

    mass:   V·(φρ − φρ|old)/Δt + Σ_f ρ_up·μ⁻¹·G^p_f − q_m = 0
    energy: V·(E − E|old)/Δt   + Σ_f (H_up·F_mass + F^cond_f) − q_e = 0
    E = (1−φ)ρ_r C_r T + φ ρ C_f T ,  H = C_f T + p/ρ ,  F^cond = G^rock + λ_f·G^φ

where ``G^p = L_perm @ p`` (perm stencil). ``ρ(p,T) = ρ_ref(1 + c_f(p−p_ref) −
α_T(T−T_ref))``. On a K-orthogonal grid this reproduces the TPFA thermal model;
a linear ``T`` field with no flow gives an interior cell zero net conduction flux
on a skewed grid (the conduction patch test). Boundaries are no-flow/adiabatic; SI.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn as nn

from ....core import GeoBrainError
from .._state_validation import cell_scalar_input, nonnegative_real, positive_real
from ..contracts import _flow_model_schema
from ..wells import FlowSourceTerms, source_block
from ..discretization.flux import scatter_internal_face_flux, upwind_cell
from ..discretization.mpfa import MPFAGrid2D, _stable_polygon_areas, mpfa_o_face_flux_stencils_full
from ..discretization.mpfa3d import (
    MPFAGrid3D,
    hex_cell_volumes,
    mpfa_o_face_flux_stencils_3d_bc,
    mpfa_o_face_flux_stencils_3d_full,
)
from ..errors import FlowContractError
from ._stencil_inversion import StencilInversionMixin
from .mpfa_thermal_two_phase import _register_conduction


class _ConductionModel(Protocol):
    n_cells: int
    face_lr: torch.Tensor

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...


if TYPE_CHECKING:
    class _ThermalSingleBase:
        """Static interface for the inversion mixin plus ``torch.nn.Module``."""

        def __init__(self) -> None:
            pass

        def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
            pass

        def build_L_perm(self, perm: torch.Tensor) -> torch.Tensor:
            pass

        def build_L_rock(
            self,
            lam_rock: float | torch.Tensor,
            porosity: torch.Tensor | None = None,
        ) -> torch.Tensor:
            pass

        def build_L_phi(self, porosity: torch.Tensor) -> torch.Tensor:
            pass
else:
    class _ThermalSingleBase(StencilInversionMixin, nn.Module):
        pass


def _thermal_scalar_config(values: dict[str, object], *, object_name: str) -> dict[str, float]:
    """Validate canonical-SI thermal configuration before assembling stencils."""

    nonnegative = {"fluid_compressibility_pa_inv", "thermal_expansion_k_inv"}
    return {
        field: (
            nonnegative_real(value, object_name=object_name, field=field)
            if field in nonnegative
            else positive_real(value, object_name=object_name, field=field)
        )
        for field, value in values.items()
    }


class MPFAThermalSinglePhaseModel(_ThermalSingleBase):
    """Single-phase thermal flow with an MPFA-O multi-point flux (2-D, SI).

    Args:
        grid: :class:`MPFAGrid2D`.
        perm_tensor: ``(n_cells, 2, 2)`` permeability tensors [m²].
        porosity: scalar or ``(n_cells,)``.
        All dimensional configuration names declare their canonical SI units:
        density [kg/m³], viscosity [Pa·s], pressure [Pa], compressibility
        [Pa⁻¹], thermal expansion [K⁻¹], specific heat [J/(kg·K)],
        thermal conductivity [W/(m·K)], and temperature [K].
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalSinglePhaseModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("fluid_mass", "kg/s", "mass", "pressure"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-2d",),
        phases=("fluid",),
        structured_sources=True,
    )
    grid: MPFAGrid2D | MPFAGrid3D
    phi: torch.Tensor
    V: torch.Tensor
    L_perm: torch.Tensor
    L_rock: torch.Tensor
    L_phi: torch.Tensor
    face_lr: torch.Tensor

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: float | torch.Tensor,
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
        cell_volumes_m3: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.grid = grid
        config = _thermal_scalar_config(
            {
                "density_ref_kg_m3": density_ref_kg_m3,
                "viscosity_pa_s": viscosity_pa_s,
                "reference_pressure_pa": reference_pressure_pa,
                "fluid_compressibility_pa_inv": fluid_compressibility_pa_inv,
                "thermal_expansion_k_inv": thermal_expansion_k_inv,
                "fluid_heat_capacity_j_kg_k": fluid_heat_capacity_j_kg_k,
                "rock_heat_capacity_j_kg_k": rock_heat_capacity_j_kg_k,
                "rock_density_kg_m3": rock_density_kg_m3,
                "fluid_thermal_conductivity_w_m_k": fluid_thermal_conductivity_w_m_k,
                "rock_thermal_conductivity_w_m_k": rock_thermal_conductivity_w_m_k,
                "reference_temperature_k": reference_temperature_k,
            },
            object_name="MPFAThermalSinglePhaseModel",
        )
        self.mu = config["viscosity_pa_s"]
        self.rho_ref, self.p_ref = config["density_ref_kg_m3"], config["reference_pressure_pa"]
        self.c_f = config["fluid_compressibility_pa_inv"]
        self.alpha_T = config["thermal_expansion_k_inv"]
        self.cp_f = config["fluid_heat_capacity_j_kg_k"]
        self.cp_r, self.rho_r = config["rock_heat_capacity_j_kg_k"], config["rock_density_kg_m3"]
        self.lam_f = config["fluid_thermal_conductivity_w_m_k"]
        self.lam_r = config["rock_thermal_conductivity_w_m_k"]
        self.T_ref = config["reference_temperature_k"]
        n = len(grid.cell_nodes)
        self.n_cells = n
        if (
            not isinstance(perm_tensor, torch.Tensor)
            or not perm_tensor.is_floating_point()
            or perm_tensor.shape != (n, 2, 2)
            or not bool(torch.isfinite(perm_tensor).all())
        ):
            raise FlowContractError(
                "perm_tensor must be finite floating 2-D permeability tensors",
                object_name="MPFAThermalSinglePhaseModel",
                field="perm_tensor",
                expected=(n, 2, 2),
                actual=(type(perm_tensor).__name__, tuple(getattr(perm_tensor, "shape", ()))),
            )
        dtype = perm_tensor.dtype
        device = perm_tensor.device

        phi = cell_scalar_input(
            porosity,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="porosity",
            positive=False,
            object_name="MPFAThermalSinglePhaseModel",
        )
        if bool(((phi < 0) | (phi > 1)).any()):
            raise FlowContractError(
                "porosity must lie in [0, 1]",
                object_name="MPFAThermalSinglePhaseModel",
                field="porosity",
                expected="[0, 1]",
                actual="contains a value outside [0, 1]",
            )
        self.register_buffer("phi", phi)
        if cell_volumes_m3 is None:
            cell_volumes_m3 = self._polygon_areas(grid).to(dtype=dtype, device=device)
        else:
            cell_volumes_m3 = cell_scalar_input(
                cell_volumes_m3,
                n_cells=n,
                dtype=dtype,
                device=device,
                field="cell_volumes_m3",
                positive=True,
                object_name="MPFAThermalSinglePhaseModel",
            )
        self.register_buffer("V", cell_volumes_m3)

        # MPFA stencils: one for the Darcy flux (perm), and the two
        # parallel-conductance stencils for heat conduction (rock + porosity-weighted
        # fluid), consistent with the multiphase thermal models. Both
        # no-flow-bounded. The conduction face set is asserted to align with the
        # Darcy face_lr (so L_rock @ T / L_phi @ T index the same faces).
        self.register_buffer("L_perm", self._stencil_matrix(grid, perm_tensor))
        lr: list[list[int]] = []
        for f in sorted(mpfa_o_face_flux_stencils_full(grid, perm_tensor)):
            left, right = grid.edge_cells[f]
            lr.append([left, right])
        self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long))
        _register_conduction(self, grid, phi, rock_thermal_conductivity_w_m_k, dtype)

    @staticmethod
    def _stencil_matrix(grid: MPFAGrid2D, tensor: torch.Tensor) -> torch.Tensor:
        st = mpfa_o_face_flux_stencils_full(grid, tensor)
        faces = sorted(st)
        L = tensor.new_zeros(len(faces), len(grid.cell_nodes))
        for fi, f in enumerate(faces):
            for c, t in st[f].items():
                L[fi, c] = t
        return L

    @staticmethod
    def _polygon_areas(grid: MPFAGrid2D) -> torch.Tensor:
        return _stable_polygon_areas(grid)

    def state_size(self) -> int:
        return 2 * self.n_cells

    def initial_state(
        self,
        pressure: float | torch.Tensor,
        temperature: float | torch.Tensor,
    ) -> torch.Tensor:
        pressure_pa = cell_scalar_input(
            pressure,
            n_cells=self.n_cells,
            dtype=self.V.dtype,
            device=self.V.device,
            field="pressure_pa",
            positive=True,
            object_name="MPFAThermalSinglePhaseModel.initial_state",
        )
        temperature_k = cell_scalar_input(
            temperature,
            n_cells=self.n_cells,
            dtype=self.V.dtype,
            device=self.V.device,
            field="temperature_k",
            positive=True,
            object_name="MPFAThermalSinglePhaseModel.initial_state",
        )
        return torch.cat([pressure_pa, temperature_k])

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        return {"p": state[:n], "T": state[n : 2 * n]}

    def density(self, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self.rho_ref * (1.0 + self.c_f * (p - self.p_ref) - self.alpha_T * (T - self.T_ref))

    # build_L_perm(perm) / build_L_rock(lam_rock) and the Dirichlet guard come from
    # StencilInversionMixin (perm / λ_rock inversion via differentiable stencil rebuilds).

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: FlowSourceTerms | None = None,
        perm: torch.Tensor | None = None,
        lam_rock: float | torch.Tensor | None = None,
        lam_fluid: float | torch.Tensor | None = None,
        porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = self.n_cells
        if state.shape != (2 * n,) or state_old.shape != (2 * n,):
            raise GeoBrainError(
                "MPFAThermalSinglePhaseModel state must be length 2*n_cells",
                object_name="MPFAThermalSinglePhaseModel",
                field="state",
                expected=(2 * n,),
                actual=tuple(state.shape),
            )
        p, T = state[:n], state[n : 2 * n]
        p_o, T_o = state_old[:n], state_old[n : 2 * n]
        phi = self.phi if porosity is None else porosity  # porosity inversion override
        rho, rho_o = self.density(p, T), self.density(p_o, T_o)

        U_f = self.cp_f * T
        E = (1.0 - phi) * self.rho_r * self.cp_r * T + phi * rho * U_f
        E_o = (1.0 - phi) * self.rho_r * self.cp_r * T_o + phi * rho_o * (self.cp_f * T_o)
        H = U_f + p / rho  # specific enthalpy

        acc_m = self.V * (phi * rho - phi * rho_o) / float(dt)
        acc_e = self.V * (E - E_o) / float(dt)

        # Darcy stencil: the precomputed buffer (fast forward), or rebuilt from a
        # differentiable ``perm`` field (adjoint / permeability inversion).
        L_perm = self.L_perm if perm is None else self.build_L_perm(perm)
        G_p = L_perm @ p  # MPFA Darcy flux (left→right)
        # parallel-conductance Fourier flux: rock stencil + porosity-weighted fluid
        # stencil (single phase ⇒ saturation ≡ 1, so the fluid weight is just λ_f). The
        # rock stencil / fluid conductivity may be rebuilt from differentiable fields
        # (lam_rock / lam_fluid) for thermal-conductivity inversion.
        # L_rock = (1−φ)λ_rock stencil, L_phi = φ stencil: both depend on φ, so a
        # porosity override rebuilds them (porosity inversion); lam_rock rebuilds L_rock.
        if lam_rock is None and porosity is None:
            L_rock = self.L_rock
        else:
            L_rock = self.build_L_rock(self.lam_r if lam_rock is None else lam_rock, porosity)
        L_phi = self.L_phi if porosity is None else self.build_L_phi(porosity)
        lam_f = self.lam_f if lam_fluid is None else lam_fluid
        F_cond = (L_rock @ T) + lam_f * (L_phi @ T)
        upstream = upwind_cell(G_p, self.face_lr)
        F_mass = rho[upstream] / self.mu * G_p
        H_up = H[upstream]
        F_energy = H_up * F_mass + F_cond

        R_m = acc_m + scatter_internal_face_flux(F_mass, self.face_lr, n)
        R_e = acc_e + scatter_internal_face_flux(F_energy, self.face_lr, n)
        R_m = R_m - source_block(sources, family="phase", name="fluid", like=R_m)
        R_e = R_e - source_block(sources, family="energy", name=None, like=R_e)
        return torch.cat([R_m, R_e])

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: FlowSourceTerms | None = None,
        perm: torch.Tensor | None = None,
        lam_rock: float | torch.Tensor | None = None,
        lam_fluid: float | torch.Tensor | None = None,
        porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(
            lambda s: self.residual(
                s,
                state_old,
                dt,
                sources=sources,
                perm=perm,
                lam_rock=lam_rock,
                lam_fluid=lam_fluid,
                porosity=porosity,
            ),
            state,
            vectorize=True,
        )


def _stencil_matrix_3d(
    grid: MPFAGrid3D, tensor: torch.Tensor, n: int
) -> tuple[torch.Tensor, list[int]]:
    """Assemble a ``(n_faces, n_cells)`` 3-D MPFA-O stencil matrix + its face list."""
    st = mpfa_o_face_flux_stencils_3d_full(grid, tensor)
    faces = sorted(st)
    L = tensor.new_zeros(len(faces), n)
    for fi, f in enumerate(faces):
        for c, t in st[f].items():
            L[fi, c] = t
    return L, faces


def _register_conduction_3d(
    model: _ConductionModel,
    grid: MPFAGrid3D,
    phi: torch.Tensor,
    lambda_rock: float,
    dtype: torch.dtype,
) -> None:
    """Build the 3-D parallel-conductance stencils (rock ``(1−φ)λ_r·I₃`` + porosity-
    weighted fluid ``φ·I₃``) and register them on ``model``, asserting their face set
    aligns with the model's Darcy ``face_lr`` (so the conduction flux indexes the same
    faces). Shared by the 3-D thermal single/two/three-phase/compositional models."""
    eye = torch.eye(3, dtype=dtype)
    rock_tensor = ((1.0 - phi) * float(lambda_rock)).reshape(-1, 1, 1) * eye
    phi_tensor = phi.reshape(-1, 1, 1) * eye
    L_rock, faces_rock = _stencil_matrix_3d(grid, rock_tensor, model.n_cells)
    L_phi, faces_phi = _stencil_matrix_3d(grid, phi_tensor, model.n_cells)
    cond_lr = torch.tensor([list(grid.face_cells[f]) for f in faces_phi], dtype=torch.long)
    if faces_rock != faces_phi or not torch.equal(cond_lr, model.face_lr):
        raise GeoBrainError(
            "3-D conduction stencil face set differs from the Darcy (perm) face set, "
            "non-SPD or degenerate permeability/conductivity?",
            object_name=type(model).__name__,
            field="L_rock",
            expected=tuple(model.face_lr.shape),
            actual=tuple(cond_lr.shape),
        )
    model.register_buffer("L_rock", L_rock)
    model.register_buffer("L_phi", L_phi)


def _register_conduction_bc_3d(
    model: _ConductionModel,
    grid: MPFAGrid3D,
    phi: torch.Tensor,
    lambda_rock: float,
    dir_faces: Sequence[int],
    dtype: torch.dtype,
) -> None:
    """Build the **ghost-cell-augmented** 3-D parallel-conductance stencils for
    Dirichlet boundary faces (so heat conducts to the fixed-``T_bc`` ghost), aligned
    with the model's Darcy ``face_lr`` (built with the same boundary faces). The 3-D
    counterpart of :func:`_register_conduction_bc`; shared by the 3-D thermal
    two/three-phase/compositional models when Dirichlet forcings are present."""
    eye = torch.eye(3, dtype=dtype)
    rock_tensor = ((1.0 - phi) * float(lambda_rock)).reshape(-1, 1, 1) * eye
    phi_tensor = phi.reshape(-1, 1, 1) * eye
    L_rock, lr_rock, _ = mpfa_o_face_flux_stencils_3d_bc(grid, rock_tensor, dir_faces)
    L_phi, lr_phi, _ = mpfa_o_face_flux_stencils_3d_bc(grid, phi_tensor, dir_faces)
    if not (torch.equal(lr_rock, model.face_lr) and torch.equal(lr_phi, model.face_lr)):
        raise GeoBrainError(
            "3-D Dirichlet conduction stencil faces differ from the Darcy faces, "
            "non-SPD or degenerate permeability/conductivity?",
            object_name=type(model).__name__,
            field="L_rock",
            expected=tuple(model.face_lr.shape),
            actual=tuple(lr_rock.shape),
        )
    model.register_buffer("L_rock", L_rock)
    model.register_buffer("L_phi", L_phi)


class MPFAThermalSinglePhaseModel3D(MPFAThermalSinglePhaseModel):
    """Single-phase thermal flow with a 3-D MPFA-O multi-point flux (SI).

    Args:
        grid: :class:`MPFAGrid3D`.
        perm_tensor: ``(n_cells, 3, 3)`` permeability tensors [m²].
        porosity and all canonical-SI thermal scalars: as the 2-D model.
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalSinglePhaseModel3D",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("fluid_mass", "kg/s", "mass", "pressure"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-3d",),
        phases=("fluid",),
        structured_sources=True,
    )

    _MPFA_DIM = 3  # StencilInversionMixin → 3-D stencils

    def __init__(
        self,
        grid: MPFAGrid3D,
        perm_tensor: torch.Tensor,
        porosity: float | torch.Tensor,
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
        cell_volumes_m3: torch.Tensor | None = None,
    ) -> None:
        _ThermalSingleBase.__init__(self)  # skip the 2-D __init__; build 3-D below
        self.grid = grid
        config = _thermal_scalar_config(
            {
                "density_ref_kg_m3": density_ref_kg_m3,
                "viscosity_pa_s": viscosity_pa_s,
                "reference_pressure_pa": reference_pressure_pa,
                "fluid_compressibility_pa_inv": fluid_compressibility_pa_inv,
                "thermal_expansion_k_inv": thermal_expansion_k_inv,
                "fluid_heat_capacity_j_kg_k": fluid_heat_capacity_j_kg_k,
                "rock_heat_capacity_j_kg_k": rock_heat_capacity_j_kg_k,
                "rock_density_kg_m3": rock_density_kg_m3,
                "fluid_thermal_conductivity_w_m_k": fluid_thermal_conductivity_w_m_k,
                "rock_thermal_conductivity_w_m_k": rock_thermal_conductivity_w_m_k,
                "reference_temperature_k": reference_temperature_k,
            },
            object_name="MPFAThermalSinglePhaseModel3D",
        )
        self.mu = config["viscosity_pa_s"]
        self.rho_ref, self.p_ref = config["density_ref_kg_m3"], config["reference_pressure_pa"]
        self.c_f = config["fluid_compressibility_pa_inv"]
        self.alpha_T = config["thermal_expansion_k_inv"]
        self.cp_f = config["fluid_heat_capacity_j_kg_k"]
        self.cp_r, self.rho_r = config["rock_heat_capacity_j_kg_k"], config["rock_density_kg_m3"]
        self.lam_f = config["fluid_thermal_conductivity_w_m_k"]
        self.lam_r = config["rock_thermal_conductivity_w_m_k"]
        self.T_ref = config["reference_temperature_k"]
        n = len(grid.cell_nodes)
        self.n_cells = n
        if (
            not isinstance(perm_tensor, torch.Tensor)
            or not perm_tensor.is_floating_point()
            or perm_tensor.shape != (n, 3, 3)
            or not bool(torch.isfinite(perm_tensor).all())
        ):
            raise FlowContractError(
                "perm_tensor must be finite floating 3-D permeability tensors",
                object_name="MPFAThermalSinglePhaseModel3D",
                field="perm_tensor",
                expected=(n, 3, 3),
                actual=(type(perm_tensor).__name__, tuple(getattr(perm_tensor, "shape", ()))),
            )
        dtype = perm_tensor.dtype
        device = perm_tensor.device

        phi = cell_scalar_input(
            porosity,
            n_cells=n,
            dtype=dtype,
            device=device,
            field="porosity",
            positive=False,
            object_name="MPFAThermalSinglePhaseModel3D",
        )
        if bool(((phi < 0) | (phi > 1)).any()):
            raise FlowContractError(
                "porosity must lie in [0, 1]",
                object_name="MPFAThermalSinglePhaseModel3D",
                field="porosity",
                expected="[0, 1]",
                actual="contains a value outside [0, 1]",
            )
        self.register_buffer("phi", phi)
        if cell_volumes_m3 is None:
            cell_volumes_m3 = hex_cell_volumes(grid).to(dtype=dtype, device=device)
        else:
            cell_volumes_m3 = cell_scalar_input(
                cell_volumes_m3,
                n_cells=n,
                dtype=dtype,
                device=device,
                field="cell_volumes_m3",
                positive=True,
                object_name="MPFAThermalSinglePhaseModel3D",
            )
        self.register_buffer("V", cell_volumes_m3)

        # 3-D Darcy stencil + face cell-pairs
        L_perm, faces = _stencil_matrix_3d(grid, perm_tensor, n)
        self.register_buffer("L_perm", L_perm)
        self.register_buffer(
            "face_lr", torch.tensor([list(grid.face_cells[f]) for f in faces], dtype=torch.long)
        )

        # 3-D parallel-conductance stencils (rock + porosity-weighted fluid), aligned
        # to the Darcy face set (the conduction flux indexes the same faces).
        _register_conduction_3d(self, grid, phi, rock_thermal_conductivity_w_m_k, dtype)

    # build_L_perm / build_L_rock (3-D) come from StencilInversionMixin via _MPFA_DIM=3.


__all__ = ["MPFAThermalSinglePhaseModel", "MPFAThermalSinglePhaseModel3D"]
