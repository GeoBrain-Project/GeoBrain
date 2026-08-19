"""
Oil-water two-phase **thermal** flow on a general 2-D grid using the MPFA-O
multi-point flux, coupled mass (per phase) + energy conservation.

The thermal extension of :class:`MPFATwoPhaseModel`: temperature ``T`` becomes a
primary variable alongside ``[p, S_w]`` and a single energy-conservation equation
is added. Heat moves by advection (each flowing phase carries its specific
enthalpy ``H_α = C_α·T + p/ρ_α``) and by Fourier conduction through the
rock + fluid bulk, a total thermal-energy conservation law, per cell::

    mass_α:  V·φ·(ρ_α S_α − old)/Δt + Σ_f (±F^α_f) − q^α = 0      (α = w, o)
    energy:  V·(E − E|old)/Δt + Σ_f (±F^E_f) − q^e = 0
    E = (1−φ)ρ_r C_r T + φ Σ_α ρ_α S_α C_α T           (internal energy / bulk vol)
    F^E_f = Σ_α H_α|_up · F^α_f  +  F^cond_f               (advective enthalpy + conduction)
    H_α = C_α T + p/ρ_α ,   F^cond_f = G^rock_f + (Σ_α λ_α S̄_α,f)·G^φ_f

The per-phase mass flux ``F^α_f`` (mass-mobility ``ρ_α k_rα/μ_α`` upwinded on the
phase-potential sign) is computed exactly as in :class:`MPFATwoPhaseModel`, and the
advected enthalpy ``H_α`` is upwinded on the *same* sign, so advection stays
consistent with the mass balance. Conduction follows the
parallel-conductance model: a rock MPFA stencil ``G^rock = L_rock @ T`` (built from
``(1−φ)λ_rock·I``) plus a porosity-weighted fluid stencil ``G^φ = L_phi @ T`` (built
from ``φ·I``) scaled by the face-averaged fluid conductivity
``Σ_α λ_α·½(S_α[left_cells]+S_α[right_cells])``, so each conduction path is
MPFA-consistent on
skewed/full-tensor grids and only the saturation weight is state-dependent.

Density carries optional thermal expansion ``ρ_α(p,T) = ρ_ref(1 + c_α(p−p_ref) −
α_T,α(T−T_ref))``. On a K-orthogonal grid this reproduces the TPFA thermal
two-phase solution; with ``α_T = 0`` and a uniform ``T`` the mass rows reproduce
:class:`MPFATwoPhaseModel` exactly.

Forcings: each transports both mass and energy:

- **gravity** via a per-cell ``depth`` field (the phase potentials carry ``−ρ_α g·D``);
- **BHP-controlled Peaceman wells** ``(cell, WI, bhp, inj_sw, T_inj)``: produced
  fluid leaves at the cell enthalpy, injected fluid enters at ``H_α(T_inj)``;
- **Neumann / rate boundary faces** ``{face: (q, sw_inj, T_inj)}``: total
  volumetric rate split by fractional flow, enthalpy upwinded (cell on outflow,
  injection on inflow);
- **Dirichlet / pressure boundary faces** ``{face: (p_bc, sw_bc, T_bc)}``: a
  fixed-pressure ghost; advected enthalpy uses the boundary ``T_bc`` on inflow and
  the boundary also **conducts** heat (augmented conduction stencils to the ghost).

SI units.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import torch

from ....core import GeoBrainError
from .._state_validation import cell_scalar_input, nonnegative_real, positive_real
from ..contracts import _flow_model_schema
from .._defaults import S_MAX, S_MIN
from ..discretization.flux import (
    scatter_boundary_outflow,
    scatter_internal_face_flux,
    upwind_cell,
)
from ..discretization.mpfa import (
    MPFAGrid2D,
    mpfa_o_face_flux_stencils_bc,
    mpfa_o_face_flux_stencils_full,
)
from ..discretization.mpfa3d import MPFAGrid3D
from ..errors import FlowContractError
from ..properties import RelPerm
from ..wells import (
    BHPControl,
    FlowSourceTerms,
    RateControl,
    Well,
    WellRateKind,
    source_block,
)
from ._stencil_inversion import StencilInversionMixin
from .mpfa_two_phase import MPFATwoPhaseModel, MPFATwoPhaseModel3D

ScalarValue: TypeAlias = float | torch.Tensor
MassBoundarySpec: TypeAlias = tuple[ScalarValue, ScalarValue]
ThermalBoundarySpec: TypeAlias = tuple[ScalarValue, ScalarValue, ScalarValue]


class _ConductionModel(Protocol):
    face_lr: torch.Tensor

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...


if TYPE_CHECKING:
    class _ThermalTwoBase:
        """Static MPFA two-phase/inversion interface with skipped imports."""

        _MPFA_DIM: int

        def __init__(
            self,
            grid: MPFAGrid2D,
            perm_tensor: torch.Tensor,
            porosity: torch.Tensor,
            relperm: RelPerm,
            *,
            rho_w_ref: float,
            rho_o_ref: float,
            mu_w: float,
            mu_o: float,
            c_w: float,
            c_o: float,
            p_ref: float,
            capillary: Callable[[torch.Tensor], torch.Tensor] | None,
            depth: torch.Tensor | None,
            gravity: float,
            cell_volumes: torch.Tensor | None,
            dirichlet: Mapping[int, MassBoundarySpec] | None,
            neumann: Mapping[int, MassBoundarySpec] | None,
            wells: list[Well] | None,
        ) -> None:
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
    class _ThermalTwoBase(StencilInversionMixin, MPFATwoPhaseModel):
        pass


def _thermal_two_phase_well_temperatures(
    wells: list[Well] | None,
    *,
    default_temperature_k: float,
) -> tuple[list[float], list[float]]:
    bhp_temperatures: list[float] = []
    rate_temperatures: list[float] = []
    for well in wells or []:
        temperature = well.injection_temperature_k
        if well.well_type == "INJ" and temperature is None:
            raise FlowContractError(
                "Thermal injectors require injection_temperature_k",
                object_name="MPFAThermalTwoPhaseModel",
                field=f"{well.name}.injection_temperature_k",
                expected="> 0 K",
                actual=None,
            )
        resolved = default_temperature_k if temperature is None else temperature
        if isinstance(well.control, BHPControl):
            bhp_temperatures.extend([resolved] * len(well.perforations))
        elif isinstance(well.control, RateControl):
            if well.control.kind is not WellRateKind.RESV:
                raise FlowContractError(
                    "Thermal two-phase wells currently support RESV rate control",
                    object_name="MPFAThermalTwoPhaseModel",
                    field=f"{well.name}.control.kind",
                    expected=WellRateKind.RESV.value,
                    actual=well.control.kind.value,
                )
            if len(well.perforations) != 1:
                raise FlowContractError(
                    "Thermal rate wells currently require one perforation",
                    object_name="MPFAThermalTwoPhaseModel",
                    field=f"{well.name}.perforations",
                    expected="exactly one",
                    actual=len(well.perforations),
                )
            rate_temperatures.append(resolved)
    return bhp_temperatures, rate_temperatures


def _thermal_multiphase_scalar_config(
    values: dict[str, object], *, object_name: str
) -> dict[str, float]:
    """Validate canonical-SI thermal multiphase configuration scalars."""

    nonnegative = {
        "water_compressibility_pa_inv",
        "oil_compressibility_pa_inv",
        "gas_compressibility_pa_inv",
        "water_thermal_expansion_k_inv",
        "oil_thermal_expansion_k_inv",
        "gas_thermal_expansion_k_inv",
        "gravity_m_s2",
    }
    return {
        field: (
            nonnegative_real(value, object_name=object_name, field=field)
            if field in nonnegative
            else positive_real(value, object_name=object_name, field=field)
        )
        for field, value in values.items()
    }


def _validated_spatial_inputs(
    grid: MPFAGrid2D | MPFAGrid3D,
    perm_tensor: torch.Tensor,
    porosity: float | torch.Tensor,
    depth_m: torch.Tensor | None,
    cell_volumes_m3: torch.Tensor | None,
    *,
    dimension: int,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Validate live grid-property tensors without casting or moving them."""

    n_cells = len(grid.cell_nodes)
    expected_shape = (n_cells, dimension, dimension)
    if (
        not isinstance(perm_tensor, torch.Tensor)
        or not perm_tensor.is_floating_point()
        or perm_tensor.shape != expected_shape
        or not bool(torch.isfinite(perm_tensor).all())
    ):
        raise FlowContractError(
            "perm_tensor must contain finite floating permeability tensors",
            object_name=object_name,
            field="perm_tensor",
            expected=expected_shape,
            actual=(type(perm_tensor).__name__, tuple(getattr(perm_tensor, "shape", ()))),
        )
    common = {
        "n_cells": n_cells,
        "dtype": perm_tensor.dtype,
        "device": perm_tensor.device,
        "object_name": object_name,
    }
    phi = cell_scalar_input(
        porosity,
        field="porosity",
        positive=False,
        **common,
    )
    if bool(((phi < 0) | (phi > 1)).any()):
        raise FlowContractError(
            "porosity must lie in [0, 1]",
            object_name=object_name,
            field="porosity",
            expected="[0, 1]",
            actual="contains a value outside [0, 1]",
        )
    validated_depth = (
        None
        if depth_m is None
        else cell_scalar_input(depth_m, field="depth_m", positive=False, **common)
    )
    validated_volumes = (
        None
        if cell_volumes_m3 is None
        else cell_scalar_input(
            cell_volumes_m3,
            field="cell_volumes_m3",
            positive=True,
            **common,
        )
    )
    return phi, validated_depth, validated_volumes


def conduction_stencils(
    grid: MPFAGrid2D,
    phi: torch.Tensor,
    lambda_rock: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute the two parallel-conductance MPFA stencils:
    a rock stencil ``L_rock`` from the porosity-weighted rock conductivity tensor
    ``(1−φ)·λ_rock·I`` and a porosity-weighted geometric fluid stencil ``L_phi``
    from ``φ·I`` (scaled by per-phase ``λ_fluid,α·S̄_α`` in the residual). Both are
    ``(n_faces, n_cells)`` matrices over the no-flow interior faces, MPFA-consistent
    on skewed grids.

    Returns ``(L_rock, L_phi, cond_face_lr)``, the latter the ``(n_faces, 2)``
    cell pairs of the conduction faces, in row order, so the caller can assert it
    aligns with the Darcy (perm) ``face_lr``. For SPD/isotropic conductivity and
    perm the MPFA face set is geometry-determined and identical, but the alignment
    is asserted (not assumed) so a degenerate/non-SPD case fails loudly here rather
    than as a cryptic shape mismatch inside the residual."""
    eye = torch.eye(2, dtype=dtype)
    rock_tensor = ((1.0 - phi) * float(lambda_rock)).reshape(-1, 1, 1) * eye
    phi_tensor = phi.reshape(-1, 1, 1) * eye
    L_rock, faces_rock = _stencil_matrix(grid, rock_tensor)
    L_phi, faces_phi = _stencil_matrix(grid, phi_tensor)
    if faces_rock != faces_phi:
        raise GeoBrainError(
            "rock and fluid conduction stencils cover different MPFA faces "
            "(degenerate conductivity?)",
            object_name="conduction_stencils",
            field="faces",
            expected=faces_phi,
            actual=faces_rock,
        )
    cond_face_lr = torch.tensor([list(grid.edge_cells[f]) for f in faces_phi], dtype=torch.long)
    return L_rock, L_phi, cond_face_lr


def _stencil_matrix(
    grid: MPFAGrid2D, tensor: torch.Tensor
) -> tuple[torch.Tensor, list[int]]:
    st = mpfa_o_face_flux_stencils_full(grid, tensor)
    faces = sorted(st)
    L = tensor.new_zeros(len(faces), len(grid.cell_nodes))
    for fi, f in enumerate(faces):
        for c, t in st[f].items():
            L[fi, c] = t
    return L, faces


def _register_conduction(
    model: _ConductionModel,
    grid: MPFAGrid2D,
    phi: torch.Tensor,
    lambda_rock: float,
    dtype: torch.dtype,
) -> None:
    """Build the two no-flow-interior conduction stencils and register them,
    asserting their face set aligns with the model's Darcy ``face_lr`` (so
    ``L_rock @ T`` / ``L_phi @ T`` index the same faces the advective flux scatters
    to)."""
    L_rock, L_phi, cond_face_lr = conduction_stencils(grid, phi, lambda_rock, dtype)
    if cond_face_lr.shape != model.face_lr.shape or not torch.equal(cond_face_lr, model.face_lr):
        raise GeoBrainError(
            "conduction stencil face set differs from the Darcy (perm) face set, "
            "non-SPD or degenerate permeability/conductivity?",
            object_name=type(model).__name__,
            field="L_rock",
            expected=tuple(model.face_lr.shape),
            actual=tuple(cond_face_lr.shape),
        )
    model.register_buffer("L_rock", L_rock)
    model.register_buffer("L_phi", L_phi)


def _register_conduction_bc(
    model: _ConductionModel,
    grid: MPFAGrid2D,
    phi: torch.Tensor,
    lambda_rock: float,
    dir_edges: Sequence[int],
    dtype: torch.dtype,
) -> None:
    """Build the **ghost-cell-augmented** conduction stencils for Dirichlet
    boundary faces (so heat conducts to the fixed-``T_bc`` ghost), aligned with the
    Darcy ``face_lr`` (which the parent built with the same boundary faces)."""
    eye = torch.eye(2, dtype=dtype)
    rock_tensor = ((1.0 - phi) * float(lambda_rock)).reshape(-1, 1, 1) * eye
    phi_tensor = phi.reshape(-1, 1, 1) * eye
    L_rock, lr_rock, _ = mpfa_o_face_flux_stencils_bc(grid, rock_tensor, dir_edges)
    L_phi, lr_phi, _ = mpfa_o_face_flux_stencils_bc(grid, phi_tensor, dir_edges)
    if not (torch.equal(lr_rock, model.face_lr) and torch.equal(lr_phi, model.face_lr)):
        raise GeoBrainError(
            "Dirichlet conduction stencil faces differ from the Darcy faces",
            object_name=type(model).__name__,
            field="L_rock",
            expected=tuple(model.face_lr.shape),
            actual=tuple(lr_rock.shape),
        )
    model.register_buffer("L_rock", L_rock)
    model.register_buffer("L_phi", L_phi)


class MPFAThermalTwoPhaseModel(_ThermalTwoBase):
    """Oil-water two-phase thermal Darcy flow with an MPFA-O multi-point flux (2-D, SI).

    Args:
        Dimensional parameter names declare canonical SI units: density [kg/m³],
        viscosity [Pa·s], pressure [Pa], compressibility [Pa⁻¹], thermal
        expansion [K⁻¹], heat capacity [J/(kg·K)], thermal conductivity
        [W/(m·K)], depth [m], gravity [m/s²], and cell volume [m³].
        dirichlet: ``{face: (p_bc, sw_bc, T_bc)}`` fixed-pressure boundary faces.
        neumann: ``{face: (q, sw_inj, T_inj)}`` prescribed-rate boundary faces.
        bhp_wells: ``[(cell, WI, bhp, inj_sw, T_inj), ...]`` BHP Peaceman wells.
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalTwoPhaseModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-2d",),
        phases=("water", "oil"),
        structured_sources=True,
    )
    grid: MPFAGrid2D | MPFAGrid3D
    relperm: RelPerm
    capillary: Callable[[torch.Tensor], torch.Tensor] | None
    n_cells: int
    phi: torch.Tensor
    V: torch.Tensor
    depth: torch.Tensor | None
    L: torch.Tensor
    face_lr: torch.Tensor
    L_rock: torch.Tensor
    L_phi: torch.Tensor
    dir_list: list[int] | None
    p_bc: torch.Tensor | None
    sw_bc: torch.Tensor | None
    T_bc: torch.Tensor | None
    neumann_cells: torch.Tensor | None
    neumann_q: torch.Tensor | None
    neumann_sw_inj: torch.Tensor | None
    neumann_T_inj: torch.Tensor | None
    well_cells: torch.Tensor | None
    well_WI: torch.Tensor | None
    well_bhp: torch.Tensor | None
    well_inj_sw: torch.Tensor | None
    well_T_inj: torch.Tensor | None
    well_depth_offset_m: torch.Tensor | None
    rate_well_cells: torch.Tensor | None
    rate_well_WI: torch.Tensor | None
    rate_well_q: torch.Tensor | None
    rate_well_inj_sw: torch.Tensor | None
    rate_well_T_inj: torch.Tensor | None
    rate_well_bhp_limit: torch.Tensor | None
    rate_well_has_limit: torch.Tensor | None
    rate_well_depth_offset_m: torch.Tensor | None
    rho_w_ref: float
    rho_o_ref: float
    c_w: float
    c_o: float
    p_ref: float
    mu_w: float
    mu_o: float
    g: float

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: float | torch.Tensor,
        relperm: RelPerm,
        *,
        water_density_ref_kg_m3: float = 1000.0,
        oil_density_ref_kg_m3: float = 800.0,
        water_viscosity_pa_s: float = 1e-3,
        oil_viscosity_pa_s: float = 2e-3,
        water_compressibility_pa_inv: float = 0.0,
        oil_compressibility_pa_inv: float = 0.0,
        reference_pressure_pa: float = 1e7,
        water_thermal_expansion_k_inv: float = 0.0,
        oil_thermal_expansion_k_inv: float = 0.0,
        water_heat_capacity_j_kg_k: float = 4184.0,
        oil_heat_capacity_j_kg_k: float = 2000.0,
        rock_heat_capacity_j_kg_k: float = 1000.0,
        rock_density_kg_m3: float = 2650.0,
        water_thermal_conductivity_w_m_k: float = 0.6,
        oil_thermal_conductivity_w_m_k: float = 0.15,
        rock_thermal_conductivity_w_m_k: float = 3.0,
        reference_temperature_k: float = 300.0,
        capillary: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth_m: torch.Tensor | None = None,
        gravity_m_s2: float = 9.81,
        cell_volumes_m3: torch.Tensor | None = None,
        dirichlet: Mapping[int, ThermalBoundarySpec] | None = None,
        neumann: Mapping[int, ThermalBoundarySpec] | None = None,
        wells: list[Well] | None = None,
    ) -> None:
        object_name = "MPFAThermalTwoPhaseModel"
        porosity, depth_m, cell_volumes_m3 = _validated_spatial_inputs(
            grid,
            perm_tensor,
            porosity,
            depth_m,
            cell_volumes_m3,
            dimension=2,
            object_name=object_name,
        )
        config = _thermal_multiphase_scalar_config(
            {
                "water_density_ref_kg_m3": water_density_ref_kg_m3,
                "oil_density_ref_kg_m3": oil_density_ref_kg_m3,
                "water_viscosity_pa_s": water_viscosity_pa_s,
                "oil_viscosity_pa_s": oil_viscosity_pa_s,
                "water_compressibility_pa_inv": water_compressibility_pa_inv,
                "oil_compressibility_pa_inv": oil_compressibility_pa_inv,
                "reference_pressure_pa": reference_pressure_pa,
                "water_thermal_expansion_k_inv": water_thermal_expansion_k_inv,
                "oil_thermal_expansion_k_inv": oil_thermal_expansion_k_inv,
                "water_heat_capacity_j_kg_k": water_heat_capacity_j_kg_k,
                "oil_heat_capacity_j_kg_k": oil_heat_capacity_j_kg_k,
                "rock_heat_capacity_j_kg_k": rock_heat_capacity_j_kg_k,
                "rock_density_kg_m3": rock_density_kg_m3,
                "water_thermal_conductivity_w_m_k": water_thermal_conductivity_w_m_k,
                "oil_thermal_conductivity_w_m_k": oil_thermal_conductivity_w_m_k,
                "rock_thermal_conductivity_w_m_k": rock_thermal_conductivity_w_m_k,
                "reference_temperature_k": reference_temperature_k,
                "gravity_m_s2": gravity_m_s2,
            },
            object_name=object_name,
        )
        dtype = perm_tensor.dtype
        # Split boundary metadata; typed wells retain their own SI temperature.
        m_dir = {e: (v[0], v[1]) for e, v in dirichlet.items()} if dirichlet else None
        m_neu = {f: (v[0], v[1]) for f, v in neumann.items()} if neumann else None
        bhp_temperatures, rate_temperatures = _thermal_two_phase_well_temperatures(
            wells,
            default_temperature_k=config["reference_temperature_k"],
        )
        super().__init__(
            grid,
            perm_tensor,
            porosity,
            relperm,
            rho_w_ref=config["water_density_ref_kg_m3"],
            rho_o_ref=config["oil_density_ref_kg_m3"],
            mu_w=config["water_viscosity_pa_s"],
            mu_o=config["oil_viscosity_pa_s"],
            c_w=config["water_compressibility_pa_inv"],
            c_o=config["oil_compressibility_pa_inv"],
            p_ref=config["reference_pressure_pa"],
            capillary=capillary,
            depth=depth_m,
            gravity=config["gravity_m_s2"],
            cell_volumes=cell_volumes_m3,
            dirichlet=m_dir,
            neumann=m_neu,
            wells=wells,
        )
        self.alpha_w = config["water_thermal_expansion_k_inv"]
        self.alpha_o = config["oil_thermal_expansion_k_inv"]
        self.cp_w = config["water_heat_capacity_j_kg_k"]
        self.cp_o = config["oil_heat_capacity_j_kg_k"]
        self.cp_r = config["rock_heat_capacity_j_kg_k"]
        self.rho_r = config["rock_density_kg_m3"]
        self.lam_w = config["water_thermal_conductivity_w_m_k"]
        self.lam_o = config["oil_thermal_conductivity_w_m_k"]
        self.lam_rock = config["rock_thermal_conductivity_w_m_k"]
        self.T_ref = config["reference_temperature_k"]

        # conduction stencils: interior, or ghost-augmented when Dirichlet faces exist
        if dirichlet:
            assert self.dir_list is not None
            _register_conduction_bc(
                self, grid, self.phi, self.lam_rock, [int(e) for e in dirichlet], dtype
            )
            self.register_buffer(
                "T_bc",
                torch.stack([torch.as_tensor(dirichlet[e][2], dtype=dtype) for e in self.dir_list]),
            )
        else:
            _register_conduction(self, grid, self.phi, self.lam_rock, dtype)
            self.T_bc = None
        if neumann:
            self.register_buffer(
                "neumann_T_inj",
                torch.stack([torch.as_tensor(neumann[f][2], dtype=dtype) for f in neumann]),
            )
        else:
            self.neumann_T_inj = None
        if bhp_temperatures:
            self.register_buffer(
                "well_T_inj",
                torch.tensor(bhp_temperatures, dtype=dtype, device=perm_tensor.device),
            )
        else:
            self.well_T_inj = None
        if rate_temperatures:
            self.register_buffer(
                "rate_well_T_inj",
                torch.tensor(rate_temperatures, dtype=dtype, device=perm_tensor.device),
            )
        else:
            self.rate_well_T_inj = None

    # ------------------------------------------------------------------
    # State plumbing: [p, S_w, T]
    # ------------------------------------------------------------------
    def state_size(self) -> int:
        return 3 * self.n_cells

    def initial_state(
        self,
        pressure: float | torch.Tensor,
        sw: float | torch.Tensor,
        temperature: float | torch.Tensor,
    ) -> torch.Tensor:
        common = {
            "n_cells": self.n_cells,
            "dtype": self.V.dtype,
            "device": self.V.device,
            "object_name": "MPFAThermalTwoPhaseModel.initial_state",
        }
        pressure_pa = cell_scalar_input(
            pressure,
            field="pressure_pa",
            positive=True,
            **common,
        )
        water_saturation = cell_scalar_input(
            sw,
            field="water_saturation",
            positive=False,
            **common,
        )
        if bool(((water_saturation < 0) | (water_saturation > 1)).any()):
            raise FlowContractError(
                "water saturation must lie in [0, 1]",
                object_name="MPFAThermalTwoPhaseModel.initial_state",
                field="water_saturation",
                expected="[0, 1]",
                actual="contains a value outside [0, 1]",
            )
        temperature_k = cell_scalar_input(
            temperature,
            field="temperature_k",
            positive=True,
            **common,
        )
        return torch.cat([pressure_pa, water_saturation, temperature_k])

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        return {"p": state[:n], "sw": state[n : 2 * n], "T": state[2 * n : 3 * n]}

    def _rho_w(self, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self.rho_w_ref * (
            1.0 + self.c_w * (p - self.p_ref) - self.alpha_w * (T - self.T_ref)
        )

    def _rho_o(self, p: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return self.rho_o_ref * (
            1.0 + self.c_o * (p - self.p_ref) - self.alpha_o * (T - self.T_ref)
        )

    def rate_well_bhp(self, state: torch.Tensor) -> torch.Tensor | None:
        """Return rate-well BHP using the thermal phase densities at each well."""
        if self.rate_well_cells is None:
            return None
        assert self.rate_well_WI is not None
        assert self.rate_well_q is not None
        assert self.rate_well_inj_sw is not None
        assert self.rate_well_T_inj is not None
        assert self.rate_well_depth_offset_m is not None
        assert self.rate_well_bhp_limit is not None
        assert self.rate_well_has_limit is not None
        n = self.n_cells
        p = state[:n]
        sw = state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        T = state[2 * n : 3 * n]
        wc, WI, q, isw = (
            self.rate_well_cells,
            self.rate_well_WI,
            self.rate_well_q,
            self.rate_well_inj_sw,
        )
        prod = q >= 0
        sw_w = torch.where(prod, sw[wc], isw)
        T_w = torch.where(prod, T[wc], self.rate_well_T_inj)
        lam_w = self.relperm.kr_water(sw_w) / self.mu_w
        lam_o = self.relperm.kr_oil(sw_w) / self.mu_o
        lam_t = (lam_w + lam_o).clamp_min(1e-30)
        rho_w_w = self._rho_w(p[wc], T_w)
        rho_o_w = self._rho_o(p[wc], T_w)
        hydrostatic = self.g * self.rate_well_depth_offset_m * (
            lam_w * rho_w_w + lam_o * rho_o_w
        )
        bhp_rate = p[wc] - (q / WI + hydrostatic) / lam_t
        lim = self.rate_well_bhp_limit
        viol = self.rate_well_has_limit & torch.where(prod, bhp_rate < lim, bhp_rate > lim)
        return torch.where(viol, lim, bhp_rate)

    # build_L_perm(perm) / build_L_rock(lam_rock) and the Dirichlet guard come from
    # StencilInversionMixin (perm / λ_rock inversion via differentiable stencil rebuilds;
    # interior operator only). The residual passes the rebuilt L / L_rock when given.

    # ------------------------------------------------------------------
    # Residual: [R_w, R_o, R_energy]
    # ------------------------------------------------------------------
    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: FlowSourceTerms | None = None,
        perm: torch.Tensor | None = None,
        lam_rock: float | torch.Tensor | None = None,
        lam_w: float | torch.Tensor | None = None,
        lam_o: float | torch.Tensor | None = None,
        porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = self.n_cells
        if state.shape != (3 * n,) or state_old.shape != (3 * n,):
            raise GeoBrainError(
                "MPFAThermalTwoPhaseModel state must be length 3*n_cells",
                object_name="MPFAThermalTwoPhaseModel",
                field="state",
                expected=(3 * n,),
                actual=tuple(state.shape),
            )
        p, T = state[:n], state[2 * n : 3 * n]
        sw = state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        so = 1.0 - sw
        p_o, T_o = state_old[:n], state_old[2 * n : 3 * n]
        sw_o = state_old[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        so_o = 1.0 - sw_o
        phi = self.phi if porosity is None else porosity  # porosity inversion override

        rho_w, rho_o = self._rho_w(p, T), self._rho_o(p, T)
        rho_w_o, rho_o_o = self._rho_w(p_o, T_o), self._rho_o(p_o, T_o)

        acc_w = self.V * phi * (rho_w * sw - rho_w_o * sw_o) / float(dt)
        acc_o = self.V * phi * (rho_o * so - rho_o_o * so_o) / float(dt)

        # internal energy per bulk volume: rock + Σ_α fluid (U_α = C_α·T)
        E = (1.0 - phi) * self.rho_r * self.cp_r * T + phi * (
            rho_w * sw * self.cp_w * T + rho_o * so * self.cp_o * T
        )
        E_o = (1.0 - phi) * self.rho_r * self.cp_r * T_o + phi * (
            rho_w_o * sw_o * self.cp_w * T_o + rho_o_o * so_o * self.cp_o * T_o
        )
        acc_e = self.V * (E - E_o) / float(dt)

        # phase potentials (oil pressure datum), gravity / capillarity
        phi_o = p
        phi_w = p
        if self.capillary is not None:
            phi_w = phi_w - self.capillary(sw)
        if self.depth is not None:
            phi_o = phi_o - rho_o * self.g * self.depth
            phi_w = phi_w - rho_w * self.g * self.depth

        mm_w = rho_w * self.relperm.kr_water(sw) / self.mu_w
        mm_o = rho_o * self.relperm.kr_oil(sw) / self.mu_o
        H_w = self.cp_w * T + p / rho_w
        H_o = self.cp_o * T + p / rho_o

        # Dirichlet pressure boundaries: augment the per-cell arrays with the fixed
        # ghost state (the operator L / face_lr already span [cells ; ghosts]); the
        # conduction stencils L_rock/L_phi are the augmented ones too. T/saturation
        # are augmented with the boundary values for the conduction face-average.
        if self.p_bc is not None:
            assert self.sw_bc is not None
            assert self.T_bc is not None
            rho_w_bc, rho_o_bc = (
                self._rho_w(self.p_bc, self.T_bc),
                self._rho_o(self.p_bc, self.T_bc),
            )
            cap_bc = self.capillary(self.sw_bc) if self.capillary is not None else 0.0
            phi_o = torch.cat([phi_o, self.p_bc])
            phi_w = torch.cat([phi_w, self.p_bc - cap_bc])
            mm_w = torch.cat([mm_w, rho_w_bc * self.relperm.kr_water(self.sw_bc) / self.mu_w])
            mm_o = torch.cat([mm_o, rho_o_bc * self.relperm.kr_oil(self.sw_bc) / self.mu_o])
            H_w = torch.cat([H_w, self.cp_w * self.T_bc + self.p_bc / rho_w_bc])
            H_o = torch.cat([H_o, self.cp_o * self.T_bc + self.p_bc / rho_o_bc])
            T_aug = torch.cat([T, self.T_bc])
            sw_aug = torch.cat([sw, self.sw_bc])
            so_aug = torch.cat([so, 1.0 - self.sw_bc])
        else:
            T_aug, sw_aug, so_aug = T, sw, so

        L = self.L if perm is None else self.build_L_perm(perm)
        G_w, G_o = L @ phi_w, L @ phi_o
        left_cells, right_cells = self.face_lr[:, 0], self.face_lr[:, 1]
        up_w = upwind_cell(G_w, self.face_lr)
        up_o = upwind_cell(G_o, self.face_lr)
        F_w = mm_w[up_w] * G_w
        F_o = mm_o[up_o] * G_o
        right_is_real = right_cells < n  # endpoint 1 may be a ghost

        def scatter(acc: torch.Tensor, face_flux: torch.Tensor) -> torch.Tensor:
            return (
                acc
                + scatter_internal_face_flux(
                    face_flux[right_is_real], self.face_lr[right_is_real], n
                )
                + scatter_boundary_outflow(face_flux[~right_is_real], left_cells[~right_is_real], n)
            )

        R_w = scatter(acc_w, F_w)
        R_o = scatter(acc_o, F_o)

        # energy flux: advected enthalpy (same upwind) + parallel conduction
        adv = H_w[up_w] * F_w + H_o[up_o] * F_o
        sw_face = 0.5 * (sw_aug[left_cells] + sw_aug[right_cells])
        so_face = 0.5 * (so_aug[left_cells] + so_aug[right_cells])
        lw = self.lam_w if lam_w is None else lam_w
        lo = self.lam_o if lam_o is None else lam_o
        lam_fluid_face = lw * sw_face + lo * so_face
        if lam_rock is None and porosity is None:
            L_rock = self.L_rock
        else:
            L_rock = self.build_L_rock(self.lam_rock if lam_rock is None else lam_rock, porosity)
        L_phi = self.L_phi if porosity is None else self.build_L_phi(porosity)
        F_cond = (L_rock @ T_aug) + lam_fluid_face * (L_phi @ T_aug)
        R_e = scatter(acc_e, adv + F_cond)

        # Neumann / rate boundary faces (mass + energy)
        if self.neumann_q is not None:
            assert self.neumann_cells is not None
            assert self.neumann_sw_inj is not None
            assert self.neumann_T_inj is not None
            nb, q, swj, Tinj = (
                self.neumann_cells,
                self.neumann_q,
                self.neumann_sw_inj,
                self.neumann_T_inj,
            )
            out = q >= 0
            sw_b = torch.where(out, sw[nb], swj)
            T_b = torch.where(out, T[nb], Tinj)
            rho_w_b, rho_o_b = self._rho_w(p[nb], T_b), self._rho_o(p[nb], T_b)
            lw = self.relperm.kr_water(sw_b) / self.mu_w
            lo = self.relperm.kr_oil(sw_b) / self.mu_o
            tot = (lw + lo).clamp_min(1e-30)
            Fw_b = rho_w_b * (lw / tot) * q
            Fo_b = rho_o_b * (lo / tot) * q
            H_w_b = self.cp_w * T_b + p[nb] / rho_w_b
            H_o_b = self.cp_o * T_b + p[nb] / rho_o_b
            R_w = R_w + scatter_boundary_outflow(Fw_b, nb, n)
            R_o = R_o + scatter_boundary_outflow(Fo_b, nb, n)
            R_e = R_e + scatter_boundary_outflow(H_w_b * Fw_b + H_o_b * Fo_b, nb, n)

        # BHP-controlled Peaceman wells (mass + energy)
        if self.well_cells is not None:
            assert self.well_WI is not None
            assert self.well_bhp is not None
            assert self.well_inj_sw is not None
            assert self.well_T_inj is not None
            assert self.well_depth_offset_m is not None
            wc, WI, bhp, isw, Tinj = (
                self.well_cells,
                self.well_WI,
                self.well_bhp,
                self.well_inj_sw,
                self.well_T_inj,
            )
            datum_drawdown = p[wc] - bhp
            prod = datum_drawdown >= 0
            sw_w = torch.where(prod, sw[wc], isw)
            T_w = torch.where(prod, T[wc], Tinj)
            rho_w_w, rho_o_w = self._rho_w(p[wc], T_w), self._rho_o(p[wc], T_w)
            lam_w = self.relperm.kr_water(sw_w) / self.mu_w
            lam_o = self.relperm.kr_oil(sw_w) / self.mu_o
            water_drawdown = (
                datum_drawdown - rho_w_w * self.g * self.well_depth_offset_m
            )
            oil_drawdown = datum_drawdown - rho_o_w * self.g * self.well_depth_offset_m
            Fw_w = WI * rho_w_w * lam_w * water_drawdown
            Fo_w = WI * rho_o_w * lam_o * oil_drawdown
            H_w_w = self.cp_w * T_w + p[wc] / rho_w_w
            H_o_w = self.cp_o * T_w + p[wc] / rho_o_w
            R_w = R_w + scatter_boundary_outflow(Fw_w, wc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_w, wc, n)
            R_e = R_e + scatter_boundary_outflow(H_w_w * Fw_w + H_o_w * Fo_w, wc, n)

        # Rate-controlled wells (mass + energy): prescribe the total reservoir-volumetric
        # rate q_res (the bhp is solved); production draws the cell fluid at its enthalpy,
        # injection enters at H_α(T_inj). With a bhp_limit the well switches to BHP control.
        if self.rate_well_cells is not None:
            assert self.rate_well_WI is not None
            assert self.rate_well_q is not None
            assert self.rate_well_inj_sw is not None
            assert self.rate_well_T_inj is not None
            assert self.rate_well_depth_offset_m is not None
            assert self.rate_well_bhp_limit is not None
            assert self.rate_well_has_limit is not None
            wc, WI, q, isw, Tinj = (
                self.rate_well_cells,
                self.rate_well_WI,
                self.rate_well_q,
                self.rate_well_inj_sw,
                self.rate_well_T_inj,
            )
            prod = q >= 0
            sw_w = torch.where(prod, sw[wc], isw)
            T_w = torch.where(prod, T[wc], Tinj)
            rho_w_w, rho_o_w = self._rho_w(p[wc], T_w), self._rho_o(p[wc], T_w)
            mob_w = self.relperm.kr_water(sw_w) / self.mu_w
            mob_o = self.relperm.kr_oil(sw_w) / self.mu_o
            lam_t = (mob_w + mob_o).clamp_min(1e-30)
            hydrostatic = self.g * self.rate_well_depth_offset_m * (
                mob_w * rho_w_w + mob_o * rho_o_w
            )
            bhp_rate = p[wc] - (q / WI + hydrostatic) / lam_t
            water_drawdown = (
                p[wc]
                - bhp_rate
                - rho_w_w * self.g * self.rate_well_depth_offset_m
            )
            oil_drawdown = (
                p[wc]
                - bhp_rate
                - rho_o_w * self.g * self.rate_well_depth_offset_m
            )
            Fw_rate = WI * rho_w_w * mob_w * water_drawdown
            Fo_rate = WI * rho_o_w * mob_o * oil_drawdown
            lim = self.rate_well_bhp_limit
            viol = self.rate_well_has_limit & torch.where(prod, bhp_rate < lim, bhp_rate > lim)
            water_limit_drawdown = (
                p[wc] - lim - rho_w_w * self.g * self.rate_well_depth_offset_m
            )
            oil_limit_drawdown = (
                p[wc] - lim - rho_o_w * self.g * self.rate_well_depth_offset_m
            )
            Fw_w = torch.where(
                viol, WI * rho_w_w * mob_w * water_limit_drawdown, Fw_rate
            )
            Fo_w = torch.where(
                viol, WI * rho_o_w * mob_o * oil_limit_drawdown, Fo_rate
            )
            H_w_w = self.cp_w * T_w + p[wc] / rho_w_w
            H_o_w = self.cp_o * T_w + p[wc] / rho_o_w
            R_w = R_w + scatter_boundary_outflow(Fw_w, wc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_w, wc, n)
            R_e = R_e + scatter_boundary_outflow(H_w_w * Fw_w + H_o_w * Fo_w, wc, n)

        R_w = R_w - source_block(sources, family="phase", name="water", like=R_w)
        R_o = R_o - source_block(sources, family="phase", name="oil", like=R_o)
        R_e = R_e - source_block(sources, family="energy", name=None, like=R_e)
        return torch.cat([R_w, R_o, R_e])

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: FlowSourceTerms | None = None,
        perm: torch.Tensor | None = None,
        lam_rock: float | torch.Tensor | None = None,
        lam_w: float | torch.Tensor | None = None,
        lam_o: float | torch.Tensor | None = None,
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
                lam_w=lam_w,
                lam_o=lam_o,
                porosity=porosity,
            ),
            state,
            vectorize=True,
        )


class MPFAThermalTwoPhaseModel3D(MPFAThermalTwoPhaseModel):
    """Oil-water two-phase thermal Darcy flow with a 3-D MPFA-O multi-point flux (SI).

    Args mirror :class:`MPFAThermalTwoPhaseModel`, with a :class:`MPFAGrid3D` and
    ``(n_cells, 3, 3)`` permeability tensors. Gravity is enabled by passing a
    per-cell ``depth_m``.
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalTwoPhaseModel3D",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-3d",),
        phases=("water", "oil"),
        structured_sources=True,
    )

    _MPFA_DIM = 3  # StencilInversionMixin → 3-D stencils

    def __init__(
        self,
        grid: MPFAGrid3D,
        perm_tensor: torch.Tensor,
        porosity: float | torch.Tensor,
        relperm: RelPerm,
        *,
        water_density_ref_kg_m3: float = 1000.0,
        oil_density_ref_kg_m3: float = 800.0,
        water_viscosity_pa_s: float = 1e-3,
        oil_viscosity_pa_s: float = 2e-3,
        water_compressibility_pa_inv: float = 0.0,
        oil_compressibility_pa_inv: float = 0.0,
        reference_pressure_pa: float = 1e7,
        water_thermal_expansion_k_inv: float = 0.0,
        oil_thermal_expansion_k_inv: float = 0.0,
        water_heat_capacity_j_kg_k: float = 4184.0,
        oil_heat_capacity_j_kg_k: float = 2000.0,
        rock_heat_capacity_j_kg_k: float = 1000.0,
        rock_density_kg_m3: float = 2650.0,
        water_thermal_conductivity_w_m_k: float = 0.6,
        oil_thermal_conductivity_w_m_k: float = 0.15,
        rock_thermal_conductivity_w_m_k: float = 3.0,
        reference_temperature_k: float = 300.0,
        capillary: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth_m: torch.Tensor | None = None,
        gravity_m_s2: float = 9.81,
        cell_volumes_m3: torch.Tensor | None = None,
        dirichlet: Mapping[int, ThermalBoundarySpec] | None = None,
        neumann: Mapping[int, ThermalBoundarySpec] | None = None,
        wells: list[Well] | None = None,
    ) -> None:
        from .mpfa_thermal_single_phase import _register_conduction_3d, _register_conduction_bc_3d

        object_name = "MPFAThermalTwoPhaseModel3D"
        porosity, depth_m, cell_volumes_m3 = _validated_spatial_inputs(
            grid,
            perm_tensor,
            porosity,
            depth_m,
            cell_volumes_m3,
            dimension=3,
            object_name=object_name,
        )
        config = _thermal_multiphase_scalar_config(
            {
                "water_density_ref_kg_m3": water_density_ref_kg_m3,
                "oil_density_ref_kg_m3": oil_density_ref_kg_m3,
                "water_viscosity_pa_s": water_viscosity_pa_s,
                "oil_viscosity_pa_s": oil_viscosity_pa_s,
                "water_compressibility_pa_inv": water_compressibility_pa_inv,
                "oil_compressibility_pa_inv": oil_compressibility_pa_inv,
                "reference_pressure_pa": reference_pressure_pa,
                "water_thermal_expansion_k_inv": water_thermal_expansion_k_inv,
                "oil_thermal_expansion_k_inv": oil_thermal_expansion_k_inv,
                "water_heat_capacity_j_kg_k": water_heat_capacity_j_kg_k,
                "oil_heat_capacity_j_kg_k": oil_heat_capacity_j_kg_k,
                "rock_heat_capacity_j_kg_k": rock_heat_capacity_j_kg_k,
                "rock_density_kg_m3": rock_density_kg_m3,
                "water_thermal_conductivity_w_m_k": water_thermal_conductivity_w_m_k,
                "oil_thermal_conductivity_w_m_k": oil_thermal_conductivity_w_m_k,
                "rock_thermal_conductivity_w_m_k": rock_thermal_conductivity_w_m_k,
                "reference_temperature_k": reference_temperature_k,
                "gravity_m_s2": gravity_m_s2,
            },
            object_name=object_name,
        )
        dtype = perm_tensor.dtype
        # Split boundary metadata; typed wells retain their own SI temperature.
        m_dir = {e: (v[0], v[1]) for e, v in dirichlet.items()} if dirichlet else None
        m_neu = {f: (v[0], v[1]) for f, v in neumann.items()} if neumann else None
        bhp_temperatures, rate_temperatures = _thermal_two_phase_well_temperatures(
            wells,
            default_temperature_k=config["reference_temperature_k"],
        )
        # Build the 3-D Darcy machinery (L/face_lr (ghost-augmented when Dirichlet) ,
        # V, phi, depth, mass PVT, and the mass forcing buffers p_bc/sw_bc/neumann_*/
        # well_*/rate_well_* + the ghost order self.dir_list) via the 3-D two-phase model.
        MPFATwoPhaseModel3D.__init__(
            cast(MPFATwoPhaseModel3D, self),
            grid,
            perm_tensor,
            porosity,
            relperm,
            rho_w_ref=config["water_density_ref_kg_m3"],
            rho_o_ref=config["oil_density_ref_kg_m3"],
            mu_w=config["water_viscosity_pa_s"],
            mu_o=config["oil_viscosity_pa_s"],
            c_w=config["water_compressibility_pa_inv"],
            c_o=config["oil_compressibility_pa_inv"],
            p_ref=config["reference_pressure_pa"],
            capillary=capillary,
            depth=depth_m,
            gravity=config["gravity_m_s2"],
            cell_volumes=cell_volumes_m3,
            dirichlet=m_dir,
            neumann=m_neu,
            wells=wells,
        )
        self.alpha_w = config["water_thermal_expansion_k_inv"]
        self.alpha_o = config["oil_thermal_expansion_k_inv"]
        self.cp_w = config["water_heat_capacity_j_kg_k"]
        self.cp_o = config["oil_heat_capacity_j_kg_k"]
        self.cp_r = config["rock_heat_capacity_j_kg_k"]
        self.rho_r = config["rock_density_kg_m3"]
        self.lam_w = config["water_thermal_conductivity_w_m_k"]
        self.lam_o = config["oil_thermal_conductivity_w_m_k"]
        self.lam_rock = config["rock_thermal_conductivity_w_m_k"]
        self.T_ref = config["reference_temperature_k"]

        # 3-D conduction stencils: interior, or ghost-augmented when Dirichlet faces
        # exist (so the fixed-T_bc boundary also conducts), aligned to the Darcy face set.
        if dirichlet:
            assert self.dir_list is not None
            _register_conduction_bc_3d(
                self, grid, self.phi, self.lam_rock, [int(e) for e in dirichlet], dtype
            )
            self.register_buffer(
                "T_bc",
                torch.stack([torch.as_tensor(dirichlet[e][2], dtype=dtype) for e in self.dir_list]),
            )
        else:
            _register_conduction_3d(self, grid, self.phi, self.lam_rock, dtype)
            self.T_bc = None
        if neumann:
            self.register_buffer(
                "neumann_T_inj",
                torch.stack([torch.as_tensor(neumann[f][2], dtype=dtype) for f in neumann]),
            )
        else:
            self.neumann_T_inj = None
        if bhp_temperatures:
            self.register_buffer(
                "well_T_inj",
                torch.tensor(bhp_temperatures, dtype=dtype, device=perm_tensor.device),
            )
        else:
            self.well_T_inj = None
        if rate_temperatures:
            self.register_buffer(
                "rate_well_T_inj",
                torch.tensor(rate_temperatures, dtype=dtype, device=perm_tensor.device),
            )
        else:
            self.rate_well_T_inj = None

    # build_L_perm / build_L_rock (3-D) come from StencilInversionMixin via _MPFA_DIM=3.


__all__ = ["MPFAThermalTwoPhaseModel", "MPFAThermalTwoPhaseModel3D"]
