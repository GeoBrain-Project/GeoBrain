"""
Oil-water-gas three-phase **thermal** flow on a general 2-D grid using the MPFA-O
multi-point flux, coupled mass (per phase) + energy conservation.

The thermal extension of :class:`MPFAThreePhaseModel`: temperature ``T`` joins
``[p, S_w, S_g]`` as a primary variable and a single energy-conservation equation
is added, a total thermal-energy conservation law. Heat moves by
per-phase advected enthalpy ``H_α = C_α T + p/ρ_α`` (each upwinded on its own phase
mass flux, exactly the phase-potential upwinding of the mass balance) and by
Fourier conduction (parallel-conductance model: a rock MPFA stencil plus a
porosity-weighted fluid stencil scaled by the face-averaged fluid conductivity
``Σ_α λ_α·½(S_α[left_cells]+S_α[right_cells])`` over the three phases).
Per cell::

    mass_α:  V·φ·(ρ_α S_α − old)/Δt + Σ_f (±F^α_f) − q^α = 0      (α = w, o, g)
    energy:  V·(E − E|old)/Δt + Σ_f (Σ_α H_α|_up F^α_f + F^cond_f) − q^e = 0
    E = (1−φ)ρ_r C_r T + φ Σ_α ρ_α S_α C_α T

The accumulation uses the *raw* (unclamped) ``S_g`` exactly as the parent so the
gas-equation diagonal stays alive where gas is absent. With ``α_T = 0`` and a
uniform ``T`` the mass rows reproduce :class:`MPFAThreePhaseModel`; on a
K-orthogonal grid the whole residual reproduces the TPFA thermal three-phase
solution. Forcings (the three-phase parent has none, so the machinery is added
here), each transports mass (3-phase fractional-flow split) and energy:

- **gravity** via a per-cell ``depth`` field (inherited phase potentials);
- **BHP-controlled Peaceman wells** ``(cell, WI, bhp, inj_sw, inj_sg, T_inj)``,
  produced at the cell enthalpy, injected at ``H_α(T_inj)``;
- **Neumann / rate boundary faces** ``{face: (q, sw_inj, sg_inj, T_inj)}``;
- **Dirichlet / pressure boundary faces** ``{face: (p_bc, sw_bc, sg_bc, T_bc)}``;
  a fixed-pressure ghost; advection carries the ghost enthalpy ``H_α(p_bc,T_bc)``
  on inflow and the boundary also conducts (ghost-augmented conduction stencils).

Interior MPFA faces + per-phase mass / energy sources; SI.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import torch

from ....core import GeoBrainError
from .._state_validation import cell_scalar_input
from ..contracts import _flow_model_schema
from .._defaults import S_MAX, S_MIN
from ..discretization.flux import (
    scatter_boundary_outflow,
    scatter_internal_face_flux,
    upwind_cell,
)
from ..discretization.mpfa import MPFAGrid2D, mpfa_o_face_flux_stencils_bc
from ..discretization.mpfa3d import MPFAGrid3D, mpfa_o_face_flux_stencils_3d_bc
from ..errors import FlowContractError
from ..properties import ThreePhaseRelPerm
from ..wells import (
    BHPControl,
    FlowSourceTerms,
    RateControl,
    Well,
    WellRateKind,
    source_block,
    validate_well_control,
)
from ._stencil_inversion import StencilInversionMixin
from .mpfa_thermal_single_phase import _register_conduction_3d, _register_conduction_bc_3d
from .mpfa_thermal_two_phase import (
    _register_conduction,
    _register_conduction_bc,
    _thermal_multiphase_scalar_config,
    _validated_spatial_inputs,
)
from .mpfa_three_phase import MPFAThreePhaseModel, MPFAThreePhaseModel3D

ScalarValue: TypeAlias = float | torch.Tensor
ThermalBoundarySpec: TypeAlias = tuple[
    ScalarValue, ScalarValue, ScalarValue, ScalarValue
]
BHPWellSpec3: TypeAlias = tuple[int, float, float, float, float, float, float]
RateWellSpec3: TypeAlias = (
    tuple[int, float, float, float, float, float, float]
    | tuple[int, float, float, float, float, float, float, float]
)


class _RateWellModel(Protocol):
    rate_well_cells: torch.Tensor | None

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...


if TYPE_CHECKING:
    class _ThermalThreeBase:
        """Static MPFA three-phase/inversion interface with skipped imports."""

        _MPFA_DIM: int

        def __init__(
            self,
            grid: MPFAGrid2D,
            perm_tensor: torch.Tensor,
            porosity: torch.Tensor,
            relperm: ThreePhaseRelPerm,
            *,
            rho_w_ref: float,
            rho_o_ref: float,
            rho_g_ref: float,
            mu_w: float,
            mu_o: float,
            mu_g: float,
            c_w: float,
            c_o: float,
            c_g: float,
            p_ref: float,
            pc_ow: Callable[[torch.Tensor], torch.Tensor] | None,
            pc_og: Callable[[torch.Tensor], torch.Tensor] | None,
            depth: torch.Tensor | None,
            gravity: float,
            cell_volumes: torch.Tensor | None,
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
    class _ThermalThreeBase(StencilInversionMixin, MPFAThreePhaseModel):
        pass


def _thermal_three_phase_well_specs(
    wells: list[Well] | None,
    *,
    default_temperature_k: float,
) -> tuple[list[BHPWellSpec3] | None, list[RateWellSpec3] | None]:
    bhp_specs: list[BHPWellSpec3] = []
    rate_specs: list[RateWellSpec3] = []
    for well in wells or []:
        validate_well_control(well, ("water", "oil", "gas"))
        if len(well.perforations) != 1:
            raise FlowContractError(
                "Thermal three-phase wells currently require one perforation",
                object_name="MPFAThermalThreePhaseModel",
                field=f"{well.name}.perforations",
                expected="exactly one",
                actual=len(well.perforations),
            )
        if well.well_type == "INJ" and well.injection_temperature_k is None:
            raise FlowContractError(
                "Thermal injectors require injection_temperature_k",
                object_name="MPFAThermalThreePhaseModel",
                field=f"{well.name}.injection_temperature_k",
                expected="> 0 K",
                actual=None,
            )
        composition = well.injection_composition or {}
        injection_water_fraction = float(
            composition.get("water", 1.0 if well.injection_phase == "water" else 0.0)
        )
        injection_gas_fraction = float(
            composition.get("gas", 1.0 if well.injection_phase == "gas" else 0.0)
        )
        temperature = (
            default_temperature_k
            if well.injection_temperature_k is None
            else well.injection_temperature_k
        )
        perforation = well.perforations[0]
        prefix = (
            perforation.cell_idx,
            perforation.well_index_m3,
            perforation.depth_offset_m,
        )
        if isinstance(well.control, BHPControl):
            bhp_specs.append(
                prefix
                + (
                    well.control.pressure_pa,
                    injection_water_fraction,
                    injection_gas_fraction,
                    temperature,
                )
            )
            continue
        assert isinstance(well.control, RateControl)
        if well.control.kind is not WellRateKind.RESV:
            raise FlowContractError(
                "Thermal three-phase wells support RESV rate control",
                object_name="MPFAThermalThreePhaseModel",
                field=f"{well.name}.control.kind",
                expected=WellRateKind.RESV.value,
                actual=well.control.kind.value,
            )
        signed_target = (
            well.control.target_m3_s if well.well_type == "PROD" else -well.control.target_m3_s
        )
        spec: RateWellSpec3 = prefix + (
            signed_target,
            injection_water_fraction,
            injection_gas_fraction,
            temperature,
        )
        if well.bhp_limit_pa is not None:
            spec = prefix + (
                signed_target,
                injection_water_fraction,
                injection_gas_fraction,
                temperature,
                well.bhp_limit_pa,
            )
        rate_specs.append(spec)
    return bhp_specs or None, rate_specs or None

def register_rate_wells_3p(
    model: _RateWellModel,
    rate_wells: list[RateWellSpec3] | None,
    dtype: torch.dtype,
) -> None:
    """Register optional **rate-controlled** three-phase thermal wells:
    ``[(cell, WI, q_res, inj_sw, inj_sg, T_inj[, bhp_limit]), ...]``: ``q_res`` is the
    prescribed total reservoir-volumetric rate [m³/s] (``> 0`` production, ``< 0``
    injection), the bhp being *solved* from it (a total-rate well control);
    phases split by three-phase fractional flow, energy by per-phase enthalpy. The
    optional ``bhp_limit`` switches the well to BHP control when the rate-implied bhp
    would violate it (assumed on the physical side of ``p_cell``, producer min below /
    injector max above, so the post-switch flow keeps the rate-target direction; a
    flow-reversing limit is a shut-in well and is not modelled). Shared by the 2-D and
    3-D thermal three-phase models."""
    if not rate_wells:
        model.rate_well_cells = None
        return

    def _has(w: RateWellSpec3) -> bool:
        return len(w) >= 8 and w[-1] is not None

    def _limit(w: RateWellSpec3) -> float:
        return float(w[-1]) if _has(w) else 0.0

    model.register_buffer(
        "rate_well_cells", torch.tensor([int(w[0]) for w in rate_wells], dtype=torch.long)
    )
    model.register_buffer(
        "rate_well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer(
        "rate_well_depth_offset_m",
        torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in rate_wells]),
    )
    model.register_buffer(
        "rate_well_q", torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer(
        "rate_well_inj_sw", torch.stack([torch.as_tensor(w[4], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer(
        "rate_well_inj_sg", torch.stack([torch.as_tensor(w[5], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer(
        "rate_well_T_inj", torch.stack([torch.as_tensor(w[6], dtype=dtype) for w in rate_wells])
    )
    model.register_buffer("rate_well_has_limit", torch.tensor([_has(w) for w in rate_wells]))
    model.register_buffer(
        "rate_well_bhp_limit",
        torch.stack([torch.as_tensor(_limit(w), dtype=dtype) for w in rate_wells]),
    )


class MPFAThermalThreePhaseModel(_ThermalThreeBase):
    """Oil-water-gas three-phase thermal Darcy flow with an MPFA-O multi-point flux (2-D, SI).

    Args:
        Dimensional parameter names declare canonical SI units: density [kg/m³],
        viscosity [Pa·s], pressure [Pa], compressibility [Pa⁻¹], thermal
        expansion [K⁻¹], heat capacity [J/(kg·K)], thermal conductivity
        [W/(m·K)], depth [m], gravity [m/s²], and cell volume [m³].
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalThreePhaseModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
            ("sg", "1", ("cell",), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
            ("gas_mass", "kg/s", "mass", "sg"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-2d",),
        phases=("water", "oil", "gas"),
        structured_sources=True,
    )
    grid: MPFAGrid2D | MPFAGrid3D
    relperm: ThreePhaseRelPerm
    pc_ow: Callable[[torch.Tensor], torch.Tensor] | None
    pc_og: Callable[[torch.Tensor], torch.Tensor] | None
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
    sg_bc: torch.Tensor | None
    T_bc: torch.Tensor | None
    neumann_cells: torch.Tensor | None
    neumann_q: torch.Tensor | None
    neumann_sw_inj: torch.Tensor | None
    neumann_sg_inj: torch.Tensor | None
    neumann_T_inj: torch.Tensor | None
    well_cells: torch.Tensor | None
    well_WI: torch.Tensor | None
    well_bhp: torch.Tensor | None
    well_inj_sw: torch.Tensor | None
    well_inj_sg: torch.Tensor | None
    well_T_inj: torch.Tensor | None
    well_depth_offset_m: torch.Tensor | None
    rate_well_cells: torch.Tensor | None
    rate_well_WI: torch.Tensor | None
    rate_well_q: torch.Tensor | None
    rate_well_inj_sw: torch.Tensor | None
    rate_well_inj_sg: torch.Tensor | None
    rate_well_T_inj: torch.Tensor | None
    rate_well_bhp_limit: torch.Tensor | None
    rate_well_has_limit: torch.Tensor | None
    rate_well_depth_offset_m: torch.Tensor | None
    rho_w_ref: float
    rho_o_ref: float
    rho_g_ref: float
    c_w: float
    c_o: float
    c_g: float
    p_ref: float
    mu_w: float
    mu_o: float
    mu_g: float
    g: float

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: float | torch.Tensor,
        relperm: ThreePhaseRelPerm,
        *,
        water_density_ref_kg_m3: float = 1000.0,
        oil_density_ref_kg_m3: float = 800.0,
        gas_density_ref_kg_m3: float = 100.0,
        water_viscosity_pa_s: float = 1e-3,
        oil_viscosity_pa_s: float = 2e-3,
        gas_viscosity_pa_s: float = 2e-5,
        water_compressibility_pa_inv: float = 0.0,
        oil_compressibility_pa_inv: float = 0.0,
        gas_compressibility_pa_inv: float = 0.0,
        reference_pressure_pa: float = 1e7,
        water_thermal_expansion_k_inv: float = 0.0,
        oil_thermal_expansion_k_inv: float = 0.0,
        gas_thermal_expansion_k_inv: float = 0.0,
        water_heat_capacity_j_kg_k: float = 4184.0,
        oil_heat_capacity_j_kg_k: float = 2000.0,
        gas_heat_capacity_j_kg_k: float = 2200.0,
        rock_heat_capacity_j_kg_k: float = 1000.0,
        rock_density_kg_m3: float = 2650.0,
        water_thermal_conductivity_w_m_k: float = 0.6,
        oil_thermal_conductivity_w_m_k: float = 0.15,
        gas_thermal_conductivity_w_m_k: float = 0.03,
        rock_thermal_conductivity_w_m_k: float = 3.0,
        reference_temperature_k: float = 300.0,
        pc_ow: Callable[[torch.Tensor], torch.Tensor] | None = None,
        pc_og: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth_m: torch.Tensor | None = None,
        gravity_m_s2: float = 9.81,
        cell_volumes_m3: torch.Tensor | None = None,
        dirichlet: Mapping[int, ThermalBoundarySpec] | None = None,
        neumann: Mapping[int, ThermalBoundarySpec] | None = None,
        wells: list[Well] | None = None,
    ) -> None:
        object_name = "MPFAThermalThreePhaseModel"
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
                "gas_density_ref_kg_m3": gas_density_ref_kg_m3,
                "water_viscosity_pa_s": water_viscosity_pa_s,
                "oil_viscosity_pa_s": oil_viscosity_pa_s,
                "gas_viscosity_pa_s": gas_viscosity_pa_s,
                "water_compressibility_pa_inv": water_compressibility_pa_inv,
                "oil_compressibility_pa_inv": oil_compressibility_pa_inv,
                "gas_compressibility_pa_inv": gas_compressibility_pa_inv,
                "reference_pressure_pa": reference_pressure_pa,
                "water_thermal_expansion_k_inv": water_thermal_expansion_k_inv,
                "oil_thermal_expansion_k_inv": oil_thermal_expansion_k_inv,
                "gas_thermal_expansion_k_inv": gas_thermal_expansion_k_inv,
                "water_heat_capacity_j_kg_k": water_heat_capacity_j_kg_k,
                "oil_heat_capacity_j_kg_k": oil_heat_capacity_j_kg_k,
                "gas_heat_capacity_j_kg_k": gas_heat_capacity_j_kg_k,
                "rock_heat_capacity_j_kg_k": rock_heat_capacity_j_kg_k,
                "rock_density_kg_m3": rock_density_kg_m3,
                "water_thermal_conductivity_w_m_k": water_thermal_conductivity_w_m_k,
                "oil_thermal_conductivity_w_m_k": oil_thermal_conductivity_w_m_k,
                "gas_thermal_conductivity_w_m_k": gas_thermal_conductivity_w_m_k,
                "rock_thermal_conductivity_w_m_k": rock_thermal_conductivity_w_m_k,
                "reference_temperature_k": reference_temperature_k,
                "gravity_m_s2": gravity_m_s2,
            },
            object_name=object_name,
        )
        super().__init__(
            grid,
            perm_tensor,
            porosity,
            relperm,
            rho_w_ref=config["water_density_ref_kg_m3"],
            rho_o_ref=config["oil_density_ref_kg_m3"],
            rho_g_ref=config["gas_density_ref_kg_m3"],
            mu_w=config["water_viscosity_pa_s"],
            mu_o=config["oil_viscosity_pa_s"],
            mu_g=config["gas_viscosity_pa_s"],
            c_w=config["water_compressibility_pa_inv"],
            c_o=config["oil_compressibility_pa_inv"],
            c_g=config["gas_compressibility_pa_inv"],
            p_ref=config["reference_pressure_pa"],
            pc_ow=pc_ow,
            pc_og=pc_og,
            depth=depth_m,
            gravity=config["gravity_m_s2"],
            cell_volumes=cell_volumes_m3,
        )
        self.alpha_w = config["water_thermal_expansion_k_inv"]
        self.alpha_o = config["oil_thermal_expansion_k_inv"]
        self.alpha_g = config["gas_thermal_expansion_k_inv"]
        self.cp_w = config["water_heat_capacity_j_kg_k"]
        self.cp_o = config["oil_heat_capacity_j_kg_k"]
        self.cp_g = config["gas_heat_capacity_j_kg_k"]
        self.cp_r = config["rock_heat_capacity_j_kg_k"]
        self.rho_r = config["rock_density_kg_m3"]
        self.lam_w = config["water_thermal_conductivity_w_m_k"]
        self.lam_o = config["oil_thermal_conductivity_w_m_k"]
        self.lam_g = config["gas_thermal_conductivity_w_m_k"]
        self.lam_rock = config["rock_thermal_conductivity_w_m_k"]
        self.T_ref = config["reference_temperature_k"]
        dtype = perm_tensor.dtype

        # Dirichlet pressure boundary faces ``{e: (p_bc, sw_bc, sg_bc, T_bc)}``, the
        # three-phase parent has no BC machinery, so the ghost-augmented operator is
        # built here (overriding the parent's no-flow L/face_lr), and the conduction
        # stencils become ghost-augmented too.
        if dirichlet:
            dir_edges = [int(e) for e in dirichlet]
            L, face_lr, dir_list = mpfa_o_face_flux_stencils_bc(grid, perm_tensor, dir_edges)
            self.register_buffer("L", L)
            self.register_buffer("face_lr", face_lr)
            self.register_buffer(
                "p_bc",
                torch.stack([torch.as_tensor(dirichlet[e][0], dtype=dtype) for e in dir_list]),
            )
            self.register_buffer(
                "sw_bc",
                torch.stack([torch.as_tensor(dirichlet[e][1], dtype=dtype) for e in dir_list]),
            )
            self.register_buffer(
                "sg_bc",
                torch.stack([torch.as_tensor(dirichlet[e][2], dtype=dtype) for e in dir_list]),
            )
            self.register_buffer(
                "T_bc",
                torch.stack([torch.as_tensor(dirichlet[e][3], dtype=dtype) for e in dir_list]),
            )
            _register_conduction_bc(self, grid, self.phi, self.lam_rock, dir_edges, dtype)
            self.dir_list = dir_list  # ghost order; lets the inversion mixin
        else:  # rebuild the augmented stencil (perm/λ/φ)
            self.p_bc = None
            self.dir_list = None
            _register_conduction(self, grid, self.phi, self.lam_rock, dtype)

        # Neumann / rate boundary faces ``{e: (q, sw_inj, sg_inj, T_inj)}``
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
                "neumann_sw_inj",
                torch.stack([torch.as_tensor(neumann[f][1], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_sg_inj",
                torch.stack([torch.as_tensor(neumann[f][2], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_T_inj",
                torch.stack([torch.as_tensor(neumann[f][3], dtype=dtype) for f in faces]),
            )
        else:
            self.neumann_q = None

        # BHP-controlled Peaceman wells (the three-phase parent has none): each
        # ``(cell, WI, bhp, inj_sw, inj_sg, T_inj)`` transports mass (all 3 phases,
        # fractional-flow split) and energy (produced at the cell enthalpy, injected
        # at H_α(T_inj)).
        bhp_specs, rate_specs = _thermal_three_phase_well_specs(
            wells,
            default_temperature_k=self.T_ref,
        )
        if bhp_specs:
            self.register_buffer(
                "well_cells", torch.tensor([int(w[0]) for w in bhp_specs], dtype=torch.long)
            )
            self.register_buffer(
                "well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_depth_offset_m",
                torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in bhp_specs]),
            )
            self.register_buffer(
                "well_bhp", torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_inj_sw", torch.stack([torch.as_tensor(w[4], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_inj_sg", torch.stack([torch.as_tensor(w[5], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_T_inj", torch.stack([torch.as_tensor(w[6], dtype=dtype) for w in bhp_specs])
            )
        else:
            self.well_cells = None

        register_rate_wells_3p(self, rate_specs, dtype)

    # ------------------------------------------------------------------
    # State plumbing: [p, S_w, S_g, T]
    # ------------------------------------------------------------------
    def state_size(self) -> int:
        return 4 * self.n_cells

    def initial_state(
        self,
        pressure: float | torch.Tensor,
        sw: float | torch.Tensor,
        sg: float | torch.Tensor,
        temperature: float | torch.Tensor,
    ) -> torch.Tensor:
        common = {
            "n_cells": self.n_cells,
            "dtype": self.V.dtype,
            "device": self.V.device,
            "object_name": "MPFAThermalThreePhaseModel.initial_state",
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
        gas_saturation = cell_scalar_input(
            sg,
            field="gas_saturation",
            positive=False,
            **common,
        )
        invalid = (
            (water_saturation < 0) | (gas_saturation < 0) | (water_saturation + gas_saturation > 1)
        )
        if bool(invalid.any()):
            raise FlowContractError(
                "phase saturations must leave a non-negative oil saturation",
                object_name="MPFAThermalThreePhaseModel.initial_state",
                field="water_saturation/gas_saturation",
                expected="sw >= 0, sg >= 0, sw + sg <= 1",
                actual="contains an invalid saturation tuple",
            )
        temperature_k = cell_scalar_input(
            temperature,
            field="temperature_k",
            positive=True,
            **common,
        )
        return torch.cat([pressure_pa, water_saturation, gas_saturation, temperature_k])

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        return {
            "p": state[:n],
            "sw": state[n : 2 * n],
            "sg": state[2 * n : 3 * n],
            "T": state[3 * n : 4 * n],
        }

    def _rho_T(
        self,
        p: torch.Tensor,
        T: torch.Tensor,
        ref: float,
        c: float,
        alpha: float,
    ) -> torch.Tensor:
        return ref * (1.0 + c * (p - self.p_ref) - alpha * (T - self.T_ref))

    # ------------------------------------------------------------------
    # Residual: [R_w, R_o, R_g, R_energy]
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
        lam_g: float | torch.Tensor | None = None,
        porosity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = self.n_cells
        if state.shape != (4 * n,) or state_old.shape != (4 * n,):
            raise GeoBrainError(
                "MPFAThermalThreePhaseModel state must be length 4*n_cells",
                object_name="MPFAThermalThreePhaseModel",
                field="state",
                expected=(4 * n,),
                actual=tuple(state.shape),
            )
        p, T = state[:n], state[3 * n : 4 * n]
        sw = state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        sg = state[2 * n : 3 * n]  # raw Sg (keeps gas-eq diagonal alive)
        so = 1.0 - sw - sg
        p_o, T_o = state_old[:n], state_old[3 * n : 4 * n]
        sw_o = state_old[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        sg_o = state_old[2 * n : 3 * n]
        so_o = 1.0 - sw_o - sg_o
        phi = self.phi if porosity is None else porosity  # porosity inversion override

        rho_w = self._rho_T(p, T, self.rho_w_ref, self.c_w, self.alpha_w)
        rho_o = self._rho_T(p, T, self.rho_o_ref, self.c_o, self.alpha_o)
        rho_g = self._rho_T(p, T, self.rho_g_ref, self.c_g, self.alpha_g)
        rho_w_o = self._rho_T(p_o, T_o, self.rho_w_ref, self.c_w, self.alpha_w)
        rho_o_o = self._rho_T(p_o, T_o, self.rho_o_ref, self.c_o, self.alpha_o)
        rho_g_o = self._rho_T(p_o, T_o, self.rho_g_ref, self.c_g, self.alpha_g)

        acc_w = self.V * phi * (rho_w * sw - rho_w_o * sw_o) / float(dt)
        acc_o = self.V * phi * (rho_o * so - rho_o_o * so_o) / float(dt)
        acc_g = self.V * phi * (rho_g * sg - rho_g_o * sg_o) / float(dt)

        E = (1.0 - phi) * self.rho_r * self.cp_r * T + phi * (
            rho_w * sw * self.cp_w * T + rho_o * so * self.cp_o * T + rho_g * sg * self.cp_g * T
        )
        E_o = (1.0 - phi) * self.rho_r * self.cp_r * T_o + phi * (
            rho_w_o * sw_o * self.cp_w * T_o
            + rho_o_o * so_o * self.cp_o * T_o
            + rho_g_o * sg_o * self.cp_g * T_o
        )
        acc_e = self.V * (E - E_o) / float(dt)

        phi_o = p
        phi_w = p - (self.pc_ow(sw) if self.pc_ow is not None else 0.0)
        phi_g = p + (self.pc_og(sg) if self.pc_og is not None else 0.0)
        if self.depth is not None:
            gz = self.g * self.depth
            phi_o = phi_o - rho_o * gz
            phi_w = phi_w - rho_w * gz
            phi_g = phi_g - rho_g * gz

        mm_w = rho_w * self.relperm.kr_water(sw) / self.mu_w
        mm_o = rho_o * self.relperm.kr_oil(sw, sg) / self.mu_o
        mm_g = rho_g * self.relperm.kr_gas(sg) / self.mu_g
        H_w = self.cp_w * T + p / rho_w
        H_o = self.cp_o * T + p / rho_o
        H_g = self.cp_g * T + p / rho_g

        # Dirichlet pressure boundaries: augment the per-cell arrays with the fixed
        # ghost state (L / face_lr / the conduction stencils already span the ghosts).
        if self.p_bc is not None:
            assert self.sw_bc is not None
            assert self.sg_bc is not None
            assert self.T_bc is not None
            rho_w_bc = self._rho_T(self.p_bc, self.T_bc, self.rho_w_ref, self.c_w, self.alpha_w)
            rho_o_bc = self._rho_T(self.p_bc, self.T_bc, self.rho_o_ref, self.c_o, self.alpha_o)
            rho_g_bc = self._rho_T(self.p_bc, self.T_bc, self.rho_g_ref, self.c_g, self.alpha_g)
            phi_o = torch.cat([phi_o, self.p_bc])
            phi_w = torch.cat(
                [phi_w, self.p_bc - (self.pc_ow(self.sw_bc) if self.pc_ow is not None else 0.0)]
            )
            phi_g = torch.cat(
                [phi_g, self.p_bc + (self.pc_og(self.sg_bc) if self.pc_og is not None else 0.0)]
            )
            mm_w = torch.cat([mm_w, rho_w_bc * self.relperm.kr_water(self.sw_bc) / self.mu_w])
            mm_o = torch.cat(
                [mm_o, rho_o_bc * self.relperm.kr_oil(self.sw_bc, self.sg_bc) / self.mu_o]
            )
            mm_g = torch.cat([mm_g, rho_g_bc * self.relperm.kr_gas(self.sg_bc) / self.mu_g])
            H_w = torch.cat([H_w, self.cp_w * self.T_bc + self.p_bc / rho_w_bc])
            H_o = torch.cat([H_o, self.cp_o * self.T_bc + self.p_bc / rho_o_bc])
            H_g = torch.cat([H_g, self.cp_g * self.T_bc + self.p_bc / rho_g_bc])
            T_aug = torch.cat([T, self.T_bc])
            sw_aug = torch.cat([sw, self.sw_bc])
            sg_aug = torch.cat([sg, self.sg_bc])
            so_aug = torch.cat([so, 1.0 - self.sw_bc - self.sg_bc])
        else:
            T_aug, sw_aug, sg_aug, so_aug = T, sw, sg, so

        left_cells, right_cells = self.face_lr[:, 0], self.face_lr[:, 1]
        L = self.L if perm is None else self.build_L_perm(perm)
        G_w, G_o, G_g = L @ phi_w, L @ phi_o, L @ phi_g
        up_w = upwind_cell(G_w, self.face_lr)
        up_o = upwind_cell(G_o, self.face_lr)
        up_g = upwind_cell(G_g, self.face_lr)
        F_w = mm_w[up_w] * G_w
        F_o = mm_o[up_o] * G_o
        F_g = mm_g[up_g] * G_g
        right_is_real = right_cells < n

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
        R_g = scatter(acc_g, F_g)

        # energy flux: per-phase advected enthalpy (same upwind) + parallel conduction
        adv = H_w[up_w] * F_w + H_o[up_o] * F_o + H_g[up_g] * F_g
        sw_f = 0.5 * (sw_aug[left_cells] + sw_aug[right_cells])
        so_f = 0.5 * (so_aug[left_cells] + so_aug[right_cells])
        sg_f = 0.5 * (sg_aug[left_cells] + sg_aug[right_cells])
        lw = self.lam_w if lam_w is None else lam_w
        lo = self.lam_o if lam_o is None else lam_o
        lg = self.lam_g if lam_g is None else lam_g
        lam_fluid_face = lw * sw_f + lo * so_f + lg * sg_f
        if lam_rock is None and porosity is None:
            L_rock = self.L_rock
        else:
            L_rock = self.build_L_rock(self.lam_rock if lam_rock is None else lam_rock, porosity)
        L_phi = self.L_phi if porosity is None else self.build_L_phi(porosity)
        F_cond = (L_rock @ T_aug) + lam_fluid_face * (L_phi @ T_aug)
        R_e = scatter(acc_e, adv + F_cond)

        # Neumann / rate boundary faces (mass + energy, three-phase fractional flow)
        if self.neumann_q is not None:
            assert self.neumann_cells is not None
            assert self.neumann_sw_inj is not None
            assert self.neumann_sg_inj is not None
            assert self.neumann_T_inj is not None
            nb, q = self.neumann_cells, self.neumann_q
            swj, sgj, Tinj = self.neumann_sw_inj, self.neumann_sg_inj, self.neumann_T_inj
            out = q >= 0
            sw_b = torch.where(out, sw[nb], swj)
            sg_b = torch.where(out, sg[nb], sgj)
            T_b = torch.where(out, T[nb], Tinj)
            rho_w_b = self._rho_T(p[nb], T_b, self.rho_w_ref, self.c_w, self.alpha_w)
            rho_o_b = self._rho_T(p[nb], T_b, self.rho_o_ref, self.c_o, self.alpha_o)
            rho_g_b = self._rho_T(p[nb], T_b, self.rho_g_ref, self.c_g, self.alpha_g)
            lw = self.relperm.kr_water(sw_b) / self.mu_w
            lo = self.relperm.kr_oil(sw_b, sg_b) / self.mu_o
            lg = self.relperm.kr_gas(sg_b) / self.mu_g
            tot = (lw + lo + lg).clamp_min(1e-30)
            Fw_b = rho_w_b * (lw / tot) * q
            Fo_b = rho_o_b * (lo / tot) * q
            Fg_b = rho_g_b * (lg / tot) * q
            H_w_b = self.cp_w * T_b + p[nb] / rho_w_b
            H_o_b = self.cp_o * T_b + p[nb] / rho_o_b
            H_g_b = self.cp_g * T_b + p[nb] / rho_g_b
            R_w = R_w + scatter_boundary_outflow(Fw_b, nb, n)
            R_o = R_o + scatter_boundary_outflow(Fo_b, nb, n)
            R_g = R_g + scatter_boundary_outflow(Fg_b, nb, n)
            R_e = R_e + scatter_boundary_outflow(H_w_b * Fw_b + H_o_b * Fo_b + H_g_b * Fg_b, nb, n)

        # BHP-controlled Peaceman wells (mass + energy, all three phases)
        if self.well_cells is not None:
            assert self.well_WI is not None
            assert self.well_bhp is not None
            assert self.well_inj_sw is not None
            assert self.well_inj_sg is not None
            assert self.well_T_inj is not None
            assert self.well_depth_offset_m is not None
            wc, WI, bhp = self.well_cells, self.well_WI, self.well_bhp
            isw, isg, Tinj = self.well_inj_sw, self.well_inj_sg, self.well_T_inj
            datum_drawdown = p[wc] - bhp
            prod = datum_drawdown >= 0
            sw_w = torch.where(prod, sw[wc], isw)
            sg_w = torch.where(prod, sg[wc], isg)
            T_w = torch.where(prod, T[wc], Tinj)
            rho_w_w = self._rho_T(p[wc], T_w, self.rho_w_ref, self.c_w, self.alpha_w)
            rho_o_w = self._rho_T(p[wc], T_w, self.rho_o_ref, self.c_o, self.alpha_o)
            rho_g_w = self._rho_T(p[wc], T_w, self.rho_g_ref, self.c_g, self.alpha_g)
            water_drawdown = (
                datum_drawdown - rho_w_w * self.g * self.well_depth_offset_m
            )
            oil_drawdown = datum_drawdown - rho_o_w * self.g * self.well_depth_offset_m
            gas_drawdown = datum_drawdown - rho_g_w * self.g * self.well_depth_offset_m
            Fw_w = (
                WI
                * rho_w_w
                * self.relperm.kr_water(sw_w)
                / self.mu_w
                * water_drawdown
            )
            Fo_w = (
                WI
                * rho_o_w
                * self.relperm.kr_oil(sw_w, sg_w)
                / self.mu_o
                * oil_drawdown
            )
            Fg_w = (
                WI
                * rho_g_w
                * self.relperm.kr_gas(sg_w)
                / self.mu_g
                * gas_drawdown
            )
            R_w = R_w + scatter_boundary_outflow(Fw_w, wc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_w, wc, n)
            R_g = R_g + scatter_boundary_outflow(Fg_w, wc, n)
            H_w_w = self.cp_w * T_w + p[wc] / rho_w_w
            H_o_w = self.cp_o * T_w + p[wc] / rho_o_w
            H_g_w = self.cp_g * T_w + p[wc] / rho_g_w
            R_e = R_e + scatter_boundary_outflow(H_w_w * Fw_w + H_o_w * Fo_w + H_g_w * Fg_w, wc, n)

        # Rate-controlled wells (mass + energy): prescribe the total reservoir-volumetric
        # rate q_res (bhp solved); 3-phase fractional-flow split, per-phase enthalpy
        # (cell on production, T_inj on injection); optional BHP-limit switch.
        if self.rate_well_cells is not None:
            assert self.rate_well_WI is not None
            assert self.rate_well_q is not None
            assert self.rate_well_inj_sw is not None
            assert self.rate_well_inj_sg is not None
            assert self.rate_well_T_inj is not None
            assert self.rate_well_depth_offset_m is not None
            assert self.rate_well_bhp_limit is not None
            assert self.rate_well_has_limit is not None
            wc, WI, q = self.rate_well_cells, self.rate_well_WI, self.rate_well_q
            isw, isg, Tinj = self.rate_well_inj_sw, self.rate_well_inj_sg, self.rate_well_T_inj
            prod = q >= 0
            sw_w = torch.where(prod, sw[wc], isw)
            sg_w = torch.where(prod, sg[wc], isg)
            T_w = torch.where(prod, T[wc], Tinj)
            rho_w_w = self._rho_T(p[wc], T_w, self.rho_w_ref, self.c_w, self.alpha_w)
            rho_o_w = self._rho_T(p[wc], T_w, self.rho_o_ref, self.c_o, self.alpha_o)
            rho_g_w = self._rho_T(p[wc], T_w, self.rho_g_ref, self.c_g, self.alpha_g)
            mob_w = self.relperm.kr_water(sw_w) / self.mu_w
            mob_o = self.relperm.kr_oil(sw_w, sg_w) / self.mu_o
            mob_g = self.relperm.kr_gas(sg_w) / self.mu_g
            lam_t = (mob_w + mob_o + mob_g).clamp_min(1e-30)
            hydrostatic = self.g * self.rate_well_depth_offset_m * (
                mob_w * rho_w_w + mob_o * rho_o_w + mob_g * rho_g_w
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
            gas_drawdown = (
                p[wc]
                - bhp_rate
                - rho_g_w * self.g * self.rate_well_depth_offset_m
            )
            Fw_rate = WI * rho_w_w * mob_w * water_drawdown
            Fo_rate = WI * rho_o_w * mob_o * oil_drawdown
            Fg_rate = WI * rho_g_w * mob_g * gas_drawdown
            lim = self.rate_well_bhp_limit
            viol = self.rate_well_has_limit & torch.where(prod, bhp_rate < lim, bhp_rate > lim)
            water_limit_drawdown = (
                p[wc] - lim - rho_w_w * self.g * self.rate_well_depth_offset_m
            )
            oil_limit_drawdown = (
                p[wc] - lim - rho_o_w * self.g * self.rate_well_depth_offset_m
            )
            gas_limit_drawdown = (
                p[wc] - lim - rho_g_w * self.g * self.rate_well_depth_offset_m
            )
            Fw_w = torch.where(
                viol, WI * rho_w_w * mob_w * water_limit_drawdown, Fw_rate
            )
            Fo_w = torch.where(
                viol, WI * rho_o_w * mob_o * oil_limit_drawdown, Fo_rate
            )
            Fg_w = torch.where(
                viol, WI * rho_g_w * mob_g * gas_limit_drawdown, Fg_rate
            )
            R_w = R_w + scatter_boundary_outflow(Fw_w, wc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_w, wc, n)
            R_g = R_g + scatter_boundary_outflow(Fg_w, wc, n)
            H_w_w = self.cp_w * T_w + p[wc] / rho_w_w
            H_o_w = self.cp_o * T_w + p[wc] / rho_o_w
            H_g_w = self.cp_g * T_w + p[wc] / rho_g_w
            R_e = R_e + scatter_boundary_outflow(H_w_w * Fw_w + H_o_w * Fo_w + H_g_w * Fg_w, wc, n)

        R_w = R_w - source_block(sources, family="phase", name="water", like=R_w)
        R_o = R_o - source_block(sources, family="phase", name="oil", like=R_o)
        R_g = R_g - source_block(sources, family="phase", name="gas", like=R_g)
        R_e = R_e - source_block(sources, family="energy", name=None, like=R_e)
        return torch.cat([R_w, R_o, R_g, R_e])

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
        lam_g: float | torch.Tensor | None = None,
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
                lam_g=lam_g,
                porosity=porosity,
            ),
            state,
            vectorize=True,
        )


class MPFAThermalThreePhaseModel3D(MPFAThermalThreePhaseModel):
    """Oil-water-gas three-phase thermal Darcy flow with a 3-D MPFA-O multi-point flux (SI).

    Args mirror :class:`MPFAThermalThreePhaseModel`, with a :class:`MPFAGrid3D`
    and ``(n_cells, 3, 3)`` permeability tensors. Gravity is enabled by passing a
    per-cell ``depth_m``.
    """

    schema = _flow_model_schema(
        model_name="MPFAThermalThreePhaseModel3D",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
            ("sg", "1", ("cell",), ()),
            ("temperature", "K", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
            ("gas_mass", "kg/s", "mass", "sg"),
            ("energy", "W", "energy", "temperature"),
        ),
        grid_kinds=("mpfa-3d",),
        phases=("water", "oil", "gas"),
        structured_sources=True,
    )

    _MPFA_DIM = 3  # StencilInversionMixin → 3-D stencils

    def __init__(
        self,
        grid: MPFAGrid3D,
        perm_tensor: torch.Tensor,
        porosity: float | torch.Tensor,
        relperm: ThreePhaseRelPerm,
        *,
        water_density_ref_kg_m3: float = 1000.0,
        oil_density_ref_kg_m3: float = 800.0,
        gas_density_ref_kg_m3: float = 100.0,
        water_viscosity_pa_s: float = 1e-3,
        oil_viscosity_pa_s: float = 2e-3,
        gas_viscosity_pa_s: float = 2e-5,
        water_compressibility_pa_inv: float = 0.0,
        oil_compressibility_pa_inv: float = 0.0,
        gas_compressibility_pa_inv: float = 0.0,
        reference_pressure_pa: float = 1e7,
        water_thermal_expansion_k_inv: float = 0.0,
        oil_thermal_expansion_k_inv: float = 0.0,
        gas_thermal_expansion_k_inv: float = 0.0,
        water_heat_capacity_j_kg_k: float = 4184.0,
        oil_heat_capacity_j_kg_k: float = 2000.0,
        gas_heat_capacity_j_kg_k: float = 2200.0,
        rock_heat_capacity_j_kg_k: float = 1000.0,
        rock_density_kg_m3: float = 2650.0,
        water_thermal_conductivity_w_m_k: float = 0.6,
        oil_thermal_conductivity_w_m_k: float = 0.15,
        gas_thermal_conductivity_w_m_k: float = 0.03,
        rock_thermal_conductivity_w_m_k: float = 3.0,
        reference_temperature_k: float = 300.0,
        pc_ow: Callable[[torch.Tensor], torch.Tensor] | None = None,
        pc_og: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth_m: torch.Tensor | None = None,
        gravity_m_s2: float = 9.81,
        cell_volumes_m3: torch.Tensor | None = None,
        dirichlet: Mapping[int, ThermalBoundarySpec] | None = None,
        neumann: Mapping[int, ThermalBoundarySpec] | None = None,
        wells: list[Well] | None = None,
    ) -> None:
        object_name = "MPFAThermalThreePhaseModel3D"
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
                "gas_density_ref_kg_m3": gas_density_ref_kg_m3,
                "water_viscosity_pa_s": water_viscosity_pa_s,
                "oil_viscosity_pa_s": oil_viscosity_pa_s,
                "gas_viscosity_pa_s": gas_viscosity_pa_s,
                "water_compressibility_pa_inv": water_compressibility_pa_inv,
                "oil_compressibility_pa_inv": oil_compressibility_pa_inv,
                "gas_compressibility_pa_inv": gas_compressibility_pa_inv,
                "reference_pressure_pa": reference_pressure_pa,
                "water_thermal_expansion_k_inv": water_thermal_expansion_k_inv,
                "oil_thermal_expansion_k_inv": oil_thermal_expansion_k_inv,
                "gas_thermal_expansion_k_inv": gas_thermal_expansion_k_inv,
                "water_heat_capacity_j_kg_k": water_heat_capacity_j_kg_k,
                "oil_heat_capacity_j_kg_k": oil_heat_capacity_j_kg_k,
                "gas_heat_capacity_j_kg_k": gas_heat_capacity_j_kg_k,
                "rock_heat_capacity_j_kg_k": rock_heat_capacity_j_kg_k,
                "rock_density_kg_m3": rock_density_kg_m3,
                "water_thermal_conductivity_w_m_k": water_thermal_conductivity_w_m_k,
                "oil_thermal_conductivity_w_m_k": oil_thermal_conductivity_w_m_k,
                "gas_thermal_conductivity_w_m_k": gas_thermal_conductivity_w_m_k,
                "rock_thermal_conductivity_w_m_k": rock_thermal_conductivity_w_m_k,
                "reference_temperature_k": reference_temperature_k,
                "gravity_m_s2": gravity_m_s2,
            },
            object_name=object_name,
        )
        # Build the 3-D Darcy machinery (interior L, face_lr, V, phi, depth, mass PVT,
        # NNC) via the 3-D three-phase model. The three-phase parent has no BC/well
        # machinery, so (like the 2-D thermal three-phase model) the forcings are
        # built here directly with the 3-D MPFA BC stencils.
        MPFAThreePhaseModel3D.__init__(
            cast(MPFAThreePhaseModel3D, self),
            grid,
            perm_tensor,
            porosity,
            relperm,
            rho_w_ref=config["water_density_ref_kg_m3"],
            rho_o_ref=config["oil_density_ref_kg_m3"],
            rho_g_ref=config["gas_density_ref_kg_m3"],
            mu_w=config["water_viscosity_pa_s"],
            mu_o=config["oil_viscosity_pa_s"],
            mu_g=config["gas_viscosity_pa_s"],
            c_w=config["water_compressibility_pa_inv"],
            c_o=config["oil_compressibility_pa_inv"],
            c_g=config["gas_compressibility_pa_inv"],
            p_ref=config["reference_pressure_pa"],
            pc_ow=pc_ow,
            pc_og=pc_og,
            depth=depth_m,
            gravity=config["gravity_m_s2"],
            cell_volumes=cell_volumes_m3,
        )
        dtype = perm_tensor.dtype
        self.alpha_w = config["water_thermal_expansion_k_inv"]
        self.alpha_o = config["oil_thermal_expansion_k_inv"]
        self.alpha_g = config["gas_thermal_expansion_k_inv"]
        self.cp_w = config["water_heat_capacity_j_kg_k"]
        self.cp_o = config["oil_heat_capacity_j_kg_k"]
        self.cp_g = config["gas_heat_capacity_j_kg_k"]
        self.cp_r = config["rock_heat_capacity_j_kg_k"]
        self.rho_r = config["rock_density_kg_m3"]
        self.lam_w = config["water_thermal_conductivity_w_m_k"]
        self.lam_o = config["oil_thermal_conductivity_w_m_k"]
        self.lam_g = config["gas_thermal_conductivity_w_m_k"]
        self.lam_rock = config["rock_thermal_conductivity_w_m_k"]
        self.T_ref = config["reference_temperature_k"]

        # Dirichlet pressure boundary faces ``{f: (p_bc, sw_bc, sg_bc, T_bc)}``; build
        # the ghost-augmented 3-D operator (overriding the parent's no-flow L/face_lr)
        # + ghost-augmented conduction; else interior conduction only.
        if dirichlet:
            dir_faces = [int(f) for f in dirichlet]
            L, face_lr, dir_list = mpfa_o_face_flux_stencils_3d_bc(grid, perm_tensor, dir_faces)
            self.register_buffer("L", L)
            self.register_buffer("face_lr", face_lr)
            self.register_buffer(
                "p_bc",
                torch.stack([torch.as_tensor(dirichlet[f][0], dtype=dtype) for f in dir_list]),
            )
            self.register_buffer(
                "sw_bc",
                torch.stack([torch.as_tensor(dirichlet[f][1], dtype=dtype) for f in dir_list]),
            )
            self.register_buffer(
                "sg_bc",
                torch.stack([torch.as_tensor(dirichlet[f][2], dtype=dtype) for f in dir_list]),
            )
            self.register_buffer(
                "T_bc",
                torch.stack([torch.as_tensor(dirichlet[f][3], dtype=dtype) for f in dir_list]),
            )
            _register_conduction_bc_3d(self, grid, self.phi, self.lam_rock, dir_faces, dtype)
            self.dir_list = dir_list  # ghost order; lets the inversion mixin
        else:  # rebuild the augmented stencil (perm/λ/φ)
            self.p_bc = None
            self.dir_list = None
            _register_conduction_3d(self, grid, self.phi, self.lam_rock, dtype)

        # Neumann / rate boundary faces ``{f: (q, sw_inj, sg_inj, T_inj)}``
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
                "neumann_sw_inj",
                torch.stack([torch.as_tensor(neumann[f][1], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_sg_inj",
                torch.stack([torch.as_tensor(neumann[f][2], dtype=dtype) for f in faces]),
            )
            self.register_buffer(
                "neumann_T_inj",
                torch.stack([torch.as_tensor(neumann[f][3], dtype=dtype) for f in faces]),
            )
        else:
            self.neumann_q = None

        # BHP-controlled Peaceman wells ``[(cell, WI, bhp, inj_sw, inj_sg, T_inj), ...]``
        bhp_specs, rate_specs = _thermal_three_phase_well_specs(
            wells,
            default_temperature_k=self.T_ref,
        )
        if bhp_specs:
            self.register_buffer(
                "well_cells", torch.tensor([int(w[0]) for w in bhp_specs], dtype=torch.long)
            )
            self.register_buffer(
                "well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_depth_offset_m",
                torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in bhp_specs]),
            )
            self.register_buffer(
                "well_bhp", torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_inj_sw", torch.stack([torch.as_tensor(w[4], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_inj_sg", torch.stack([torch.as_tensor(w[5], dtype=dtype) for w in bhp_specs])
            )
            self.register_buffer(
                "well_T_inj", torch.stack([torch.as_tensor(w[6], dtype=dtype) for w in bhp_specs])
            )
        else:
            self.well_cells = None

        register_rate_wells_3p(self, rate_specs, dtype)


__all__ = ["MPFAThermalThreePhaseModel", "MPFAThermalThreePhaseModel3D"]
