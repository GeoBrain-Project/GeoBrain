"""
Oil-water two-phase transient flow on a general 2-D grid using the MPFA-O
multi-point flux.

The two-point multiphase models build each phase flux from just the face's two
neighbours, ``F^α = ρ_α·mob_α(S_up)·T·(Φ^α_L − Φ^α_R)``, inconsistent on
non-K-orthogonal / full-tensor grids. This model keeps the *same* fully-implicit
two-phase mass balance and phase-potential upwinding but replaces the two-point
geometric flux with the **MPFA-O multi-point flux**: each interior face carries a
multi-cell stencil ``L_f = {cell: T_c}`` (from :func:`mpfa_o_face_flux_stencils`,
built once from geometry + absolute permeability), and the *geometric* flux of a
phase potential is ``G^α_f = Σ_c T_c·Φ^α_c``. The phase mass flux is then

    F^α_f = (ρ_α·k_rα/μ_α)|_up · G^α_f ,

with the upwind cell chosen by the sign of ``G^α_f`` (phase-potential upwinding,
exactly as the TPFA model). Per-cell mass balance (one row per phase, SI)::

    V·φ·(ρ_α S_α − ρ_α S_α|old)/Δt  +  Σ_f (±F^α_f)  −  q^α  =  0

State layout matches :class:`OilWaterModel`, ``[p (oil pressure), S_w]`` of
length ``2·n_cells``. Boundaries are no-flow (the reservoir default: interior
faces carry the only inter-cell flux); flow is driven by per-phase sources / a
well rate. Optional capillarity ``Pc(S_w) = p_o − p_w`` and gravity (a per-cell
depth field) enter each phase potential, both routed through the same MPFA
stencil so they stay consistent on skewed full-tensor grids.

On a K-orthogonal grid the MPFA stencil collapses to two-point and this model
reproduces the TPFA two-phase solution exactly; on a skewed grid a linear
pressure field with uniform mobility gives an interior cell zero net flux for
both phases (the multiphase patch test), which TPFA cannot.

v1 scope: interior MPFA faces (no-flow boundaries) + per-phase sources. Pressure
(Dirichlet) / prescribed-flux (Neumann) boundary faces in the multiphase flux,
and three-phase / dissolved-gas coupling, are the natural extensions.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, SupportsFloat, TypeAlias

import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..contracts import _flow_model_schema
from .._defaults import S_MAX, S_MIN
from ..errors import FlowContractError
from ..discretization.flux import (
    scatter_boundary_outflow,
    scatter_internal_face_flux,
    upwind_cell,
)
from ..discretization.mpfa import MPFAGrid2D, _stable_polygon_areas, mpfa_o_face_flux_stencils_full
from ..discretization.mpfa3d import MPFAGrid3D, hex_cell_volumes, mpfa_o_face_flux_stencils_3d_full
from ..properties import RelPerm
from ..wells import (
    BHPControl,
    FlowSourceTerms,
    RateControl,
    Well,
    WellRateKind,
    source_block,
    validate_well_control,
)

ScalarValue: TypeAlias = float | torch.Tensor
ScalarInput: TypeAlias = SupportsFloat | torch.Tensor
BoundarySpec: TypeAlias = tuple[ScalarValue, ScalarValue]
BHPWellSpec: TypeAlias = tuple[int, float, float, float, float | None, float]
RateWellSpec: TypeAlias = tuple[int, float, float, float, float | None, float]
PerforationSpec: TypeAlias = tuple[int, float, float]
MultiPerfWellSpec: TypeAlias = tuple[
    list[PerforationSpec], float, float, WellRateKind, float, float
]
MultiPerfFluxes: TypeAlias = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class _NNCModule(Protocol):
    nnc_pairs: torch.Tensor | None
    nnc_trans: torch.Tensor | None

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None: ...


if TYPE_CHECKING:
    class _ModuleBase:
        """Static interface for ``torch.nn.Module`` when imports are skipped."""

        def __init__(self) -> None:
            pass

        def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
            pass
else:
    _ModuleBase = nn.Module


def _injection_water_fraction(well: Well) -> float:
    if well.well_type == "PROD":
        return 0.0
    if well.injection_composition is not None:
        return float(well.injection_composition.get("water", 0.0))
    return 1.0 if well.injection_phase == "water" else 0.0


def _two_phase_well_specs(
    wells: list[Well] | None,
) -> tuple[
    list[BHPWellSpec] | None,
    list[RateWellSpec] | None,
    list[MultiPerfWellSpec] | None,
]:
    """Lower typed public wells into the model's private tensor registries."""

    bhp_specs: list[BHPWellSpec] = []
    rate_specs: list[RateWellSpec] = []
    multiperf_specs: list[MultiPerfWellSpec] = []
    for well in wells or []:
        validate_well_control(well, ("water", "oil"))
        injection_water_fraction = _injection_water_fraction(well)
        if isinstance(well.control, BHPControl):
            if well.rate_limit is not None and well.rate_limit.kind is not WellRateKind.RESV:
                raise FlowContractError(
                    "A BHP well rate limit must use RESV in this model",
                    object_name="MPFATwoPhaseModel",
                    field=f"{well.name}.rate_limit.kind",
                    expected=WellRateKind.RESV.value,
                    actual=well.rate_limit.kind.value,
                )
            for perforation in well.perforations:
                spec = (
                    perforation.cell_idx,
                    perforation.well_index_m3,
                    well.control.pressure_pa,
                    injection_water_fraction,
                    None
                    if well.rate_limit is None
                    else well.rate_limit.target_m3_s,
                    perforation.depth_offset_m,
                )
                bhp_specs.append(spec)
            continue
        assert isinstance(well.control, RateControl)
        if well.control.kind not in {
            WellRateKind.ORAT,
            WellRateKind.WRAT,
            WellRateKind.LRAT,
            WellRateKind.RESV,
        }:
            raise FlowContractError(
                "MPFA oil-water wells support ORAT, WRAT, LRAT, and RESV controls",
                object_name="MPFATwoPhaseModel",
                field=f"{well.name}.control.kind",
                expected=(
                    WellRateKind.ORAT.value,
                    WellRateKind.WRAT.value,
                    WellRateKind.LRAT.value,
                    WellRateKind.RESV.value,
                ),
                actual=well.control.kind.value,
            )
        signed_target = (
            well.control.target_m3_s if well.well_type == "PROD" else -well.control.target_m3_s
        )
        perforations = [
            (item.cell_idx, item.well_index_m3, item.depth_offset_m)
            for item in well.perforations
        ]
        if len(perforations) == 1 and well.control.kind is WellRateKind.RESV:
            cell_idx, well_index_m3, depth_offset_m = perforations[0]
            spec = (
                cell_idx,
                well_index_m3,
                signed_target,
                injection_water_fraction,
                well.bhp_limit_pa,
                depth_offset_m,
            )
            rate_specs.append(spec)
        else:
            if well.bhp_limit_pa is not None:
                raise FlowContractError(
                    "Multi-perforation rate wells do not yet support BHP limits",
                    object_name="MPFATwoPhaseModel",
                    field=f"{well.name}.bhp_limit_pa",
                    expected=None,
                    actual=well.bhp_limit_pa,
                )
            standards = well.standard_densities_kg_m3 or {}
            multiperf_specs.append(
                (
                    perforations,
                    signed_target,
                    injection_water_fraction,
                    well.control.kind,
                    float(standards.get("water", 1.0)),
                    float(standards.get("oil", 1.0)),
                )
            )
    return bhp_specs or None, rate_specs or None, multiperf_specs or None


def register_nnc(
    module: _NNCModule,
    nnc_pairs: torch.Tensor | None,
    nnc_trans: torch.Tensor | None,
    dtype: torch.dtype,
) -> None:
    """Register optional non-neighbour connections (e.g. fault NNCs from
    :func:`~geobrain.physics.flow.grid.compute_fault_nnc`) as buffers on ``module``:
    two-point connections ``(cellA, cellB)`` with transmissibility ``T`` added to
    the residual on top of the multi-point face flux. Shared by the two-/three-
    phase and compositional MPFA models."""
    if nnc_pairs is None or nnc_trans is None or len(nnc_trans) == 0:
        module.nnc_pairs = None
        module.nnc_trans = None
    else:
        module.register_buffer("nnc_pairs", nnc_pairs.to(torch.long))
        module.register_buffer("nnc_trans", nnc_trans.to(dtype))


def nnc_phase_flux(
    R: torch.Tensor,
    pairs: torch.Tensor,
    trans: torch.Tensor,
    phi: torch.Tensor,
    massmob: torch.Tensor,
) -> torch.Tensor:
    """Add the canonical upwinded NNC divergence of one phase to ``R``: for
    each pair ``(a, b)``, ``G = T·(Φ_a − Φ_b)`` with phase-potential upwinding of
    the mass mobility, scattered conservatively (``+F`` to ``a``, ``-F`` to
    ``b``)."""
    a, b = pairs[:, 0], pairs[:, 1]
    G = trans * (phi[a] - phi[b])
    F = torch.where(G >= 0, massmob[a], massmob[b]) * G
    return R + scatter_internal_face_flux(F, pairs, R.numel())


class MPFATwoPhaseModel(_ModuleBase):
    """Oil-water two-phase Darcy flow with an MPFA-O multi-point flux (2-D, SI).

    Args:
        grid: :class:`MPFAGrid2D`.
        perm_tensor: ``(n_cells, 2, 2)`` cell permeability tensors [m²].
        porosity: scalar or ``(n_cells,)``.
        relperm: a :class:`RelPerm` (``kr_water``/``kr_oil`` of ``S_w``).
        rho_w_ref, rho_o_ref: phase reference densities [kg/m³].
        mu_w, mu_o: phase viscosities [Pa·s].
        c_w, c_o: phase compressibilities [1/Pa] (``ρ_α = ρ_ref(1+c_α(p−p_ref))``).
        p_ref: reference pressure [Pa].
        capillary: optional ``Pc(S_w) → p_o − p_w`` [Pa].
        depth: optional ``(n_cells,)`` depth (+down) [m] enabling gravity.
        gravity: gravitational acceleration [m/s²] (used iff ``depth`` given).
    """

    schema = _flow_model_schema(
        model_name="MPFATwoPhaseModel",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
        ),
        grid_kinds=("mpfa-2d",),
        phases=("water", "oil"),
        structured_sources=True,
    )
    grid: MPFAGrid2D | MPFAGrid3D
    relperm: RelPerm
    capillary: Callable[[torch.Tensor], torch.Tensor] | None
    phi: torch.Tensor
    V: torch.Tensor
    depth: torch.Tensor | None
    L: torch.Tensor
    face_lr: torch.Tensor
    dir_list: list[int] | None
    p_bc: torch.Tensor | None
    sw_bc: torch.Tensor | None
    dir_bc_cells: torch.Tensor | None
    nnc_pairs: torch.Tensor | None
    nnc_trans: torch.Tensor | None
    neumann_cells: torch.Tensor | None
    neumann_q: torch.Tensor | None
    neumann_sw_inj: torch.Tensor | None
    well_cells: torch.Tensor | None
    well_WI: torch.Tensor | None
    well_bhp: torch.Tensor | None
    well_inj_sw: torch.Tensor | None
    well_rate_limit: torch.Tensor | None
    well_has_rate_limit: torch.Tensor | None
    well_depth_offset_m: torch.Tensor | None
    rate_well_cells: torch.Tensor | None
    rate_well_WI: torch.Tensor | None
    rate_well_q: torch.Tensor | None
    rate_well_inj_sw: torch.Tensor | None
    rate_well_bhp_limit: torch.Tensor | None
    rate_well_has_limit: torch.Tensor | None
    rate_well_depth_offset_m: torch.Tensor | None
    mp_perf_cell: torch.Tensor | None
    mp_perf_WI: torch.Tensor | None
    mp_perf_well: torch.Tensor | None
    mp_depth_offset_m: torch.Tensor | None
    mp_q: torch.Tensor | None
    mp_inj_sw: torch.Tensor | None
    mp_rate_kind: torch.Tensor | None
    mp_standard_water_density: torch.Tensor | None
    mp_standard_oil_density: torch.Tensor | None
    mp_n_wells: int

    def __init__(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        porosity: ScalarInput,
        relperm: RelPerm,
        *,
        rho_w_ref: float = 1000.0,
        rho_o_ref: float = 800.0,
        mu_w: float = 1e-3,
        mu_o: float = 2e-3,
        c_w: float = 0.0,
        c_o: float = 0.0,
        p_ref: float = 1e7,
        capillary: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth: torch.Tensor | None = None,
        gravity: float = 9.81,
        cell_volumes: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_trans: torch.Tensor | None = None,
        dirichlet: Mapping[int, BoundarySpec] | None = None,
        neumann: Mapping[int, BoundarySpec] | None = None,
        wells: list[Well] | None = None,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.relperm = relperm
        self.capillary = capillary
        self.rho_w_ref, self.rho_o_ref = float(rho_w_ref), float(rho_o_ref)
        self.mu_w, self.mu_o = float(mu_w), float(mu_o)
        self.c_w, self.c_o = float(c_w), float(c_o)
        self.p_ref = float(p_ref)
        self.g = float(gravity)
        n = len(grid.cell_nodes)
        self.n_cells = n
        dtype = perm_tensor.dtype

        if isinstance(porosity, torch.Tensor):
            self.register_buffer("phi", porosity.to(dtype))
        else:
            self.register_buffer("phi", torch.full((n,), float(porosity), dtype=dtype))
        if cell_volumes is None:
            cell_volumes = self._polygon_areas(grid).to(dtype)
        self.register_buffer("V", cell_volumes.to(dtype))
        if depth is not None:
            self.register_buffer("depth", depth.to(dtype))
        else:
            self.depth = None

        # MPFA face stencils. Without Dirichlet boundaries: no-flow per-face
        # stencils (n_faces, n_cells). With Dirichlet pressure boundaries: the
        # ghost-cell augmented operator (n_faces, n_cells + n_dirichlet), where
        # each Dirichlet boundary face connects its cell to a fixed-pressure ghost.
        self._register_dirichlet(grid, perm_tensor, dirichlet, dtype)
        self._register_nnc(nnc_pairs, nnc_trans, dtype)
        self._register_neumann(grid, neumann, dtype)
        bhp_specs, rate_specs, multiperf_specs = _two_phase_well_specs(wells)
        self._register_wells(bhp_specs, dtype)
        self._register_rate_wells(rate_specs, dtype)
        self._register_multiperf_wells(multiperf_specs, dtype)

    def _register_neumann(
        self,
        grid: MPFAGrid2D | MPFAGrid3D,
        neumann: Mapping[int, BoundarySpec] | None,
        dtype: torch.dtype,
    ) -> None:
        """Register optional prescribed-flux (Neumann / rate) boundary faces:
        ``{face: (q, sw_inj)}`` with ``q`` the prescribed total *volumetric*
        outward flux [m³/s] and ``sw_inj`` the inflow water saturation (used when
        ``q < 0``). The phases split by fractional flow, the cell's on outflow,
        ``sw_inj``'s on inflow, so the water cut emerges from the solution."""
        if not neumann:
            self.neumann_cells = None
            self.neumann_q = None
            self.neumann_sw_inj = None
            return
        bc = grid.edge_cells if hasattr(grid, "edge_cells") else grid.face_cells
        faces = list(neumann)
        self.register_buffer(
            "neumann_cells", torch.tensor([bc[f][0] for f in faces], dtype=torch.long)
        )
        self.register_buffer(
            "neumann_q", torch.stack([torch.as_tensor(neumann[f][0], dtype=dtype) for f in faces])
        )
        self.register_buffer(
            "neumann_sw_inj",
            torch.stack([torch.as_tensor(neumann[f][1], dtype=dtype) for f in faces]),
        )

    def _register_wells(
        self, bhp_wells: list[BHPWellSpec] | None, dtype: torch.dtype
    ) -> None:
        """Register optional BHP-controlled Peaceman wells: ``[(cell, WI, bhp,
        inj_sw[, rate_limit]), ...]`` with ``WI`` the well index (e.g. from
        :func:`~geobrain.physics.flow.wells.compute_well_index`), ``bhp`` the
        bottom-hole pressure and ``inj_sw`` the injected water saturation (used
        when the well injects, ``p_cell < bhp``). The phase rate
        ``q_α = WI·(ρ_α k_rα/μ_α)·(p_cell − bhp)`` splits by fractional flow, the
        cell's on production, ``inj_sw``'s on injection. The optional ``rate_limit``
        caps the total *reservoir* volumetric rate ``|WI·λ_t·(p−bhp)|``, switching the
        well to **rate control** at the cap (a rate limit on a BHP target,
        the dual of the rate well's ``bhp_limit``)."""
        if not bhp_wells:
            self.well_cells = None
            self.well_WI = None
            self.well_bhp = None
            self.well_inj_sw = None
            self.well_rate_limit = None
            self.well_has_rate_limit = None
            self.well_depth_offset_m = None
            return

        def _has(w: BHPWellSpec) -> bool:
            return len(w) >= 5 and w[4] is not None

        self.register_buffer(
            "well_cells", torch.tensor([int(w[0]) for w in bhp_wells], dtype=torch.long)
        )
        self.register_buffer(
            "well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in bhp_wells])
        )
        self.register_buffer(
            "well_bhp", torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in bhp_wells])
        )
        self.register_buffer(
            "well_inj_sw", torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in bhp_wells])
        )
        self.register_buffer(
            "well_depth_offset_m",
            torch.stack([torch.as_tensor(w[5], dtype=dtype) for w in bhp_wells]),
        )
        self.register_buffer("well_has_rate_limit", torch.tensor([_has(w) for w in bhp_wells]))
        self.register_buffer(
            "well_rate_limit",
            torch.stack(  # finite placeholder where no limit (masked off)
                [torch.as_tensor(w[4] if _has(w) else 1e30, dtype=dtype) for w in bhp_wells]
            ),
        )

    def _register_rate_wells(
        self, rate_wells: list[RateWellSpec] | None, dtype: torch.dtype
    ) -> None:
        """Register optional **rate-controlled** Peaceman wells: ``[(cell, WI, q_res,
        inj_sw[, bhp_limit]), ...]`` with ``q_res`` the prescribed total *reservoir*
        volumetric rate [m³/s] (``> 0`` production, ``< 0`` injection) and ``inj_sw``
        the injected water saturation (used when ``q_res < 0``). Unlike a BHP well, the
        bottom-hole pressure is *solved* from the rate (single-perforation closed form
        ``bhp = p_cell − q_res/(WI·λ_t)``, a total-rate well control); the phase
        rates split by fractional flow. The optional ``bhp_limit`` is a constraint that
        switches the well to BHP control when the rate-implied bhp would violate it (a
        producer's *min* bhp / an injector's *max* bhp, limit switching).
        ``WI`` is needed only for the implied bhp / limit check (not the rate split).

        The ``bhp_limit`` is assumed to lie on the physical side of the cell pressure,
        a producer's *min* bhp below ``p_cell``, an injector's *max* bhp above it, so
        the post-switch flow keeps the rate target's direction and the upwind saturation
        (``sign(q_res)``) stays correct. A limit on the wrong side describes a well that
        cannot operate (it would reverse flow): that shut-in configuration is not modelled."""
        if not rate_wells:
            self.rate_well_cells = None
            self.rate_well_WI = None
            self.rate_well_q = None
            self.rate_well_inj_sw = None
            self.rate_well_bhp_limit = None
            self.rate_well_has_limit = None
            self.rate_well_depth_offset_m = None
            return

        def _has(w: RateWellSpec) -> bool:
            return len(w) >= 5 and w[4] is not None

        self.register_buffer(
            "rate_well_cells", torch.tensor([int(w[0]) for w in rate_wells], dtype=torch.long)
        )
        self.register_buffer(
            "rate_well_WI", torch.stack([torch.as_tensor(w[1], dtype=dtype) for w in rate_wells])
        )
        self.register_buffer(
            "rate_well_q", torch.stack([torch.as_tensor(w[2], dtype=dtype) for w in rate_wells])
        )
        self.register_buffer(
            "rate_well_inj_sw",
            torch.stack([torch.as_tensor(w[3], dtype=dtype) for w in rate_wells]),
        )
        self.register_buffer(
            "rate_well_depth_offset_m",
            torch.stack([torch.as_tensor(w[5], dtype=dtype) for w in rate_wells]),
        )
        self.register_buffer("rate_well_has_limit", torch.tensor([_has(w) for w in rate_wells]))
        # finite placeholder where no limit (the switch is masked off by has_limit, so
        # the unselected bhp-control branch stays finite: no NaN gradient through where)
        self.register_buffer(
            "rate_well_bhp_limit",
            torch.stack(
                [torch.as_tensor(w[4] if _has(w) else 0.0, dtype=dtype) for w in rate_wells]
            ),
        )

    def rate_well_bhp(self, state: torch.Tensor) -> torch.Tensor | None:
        """Effective bottom-hole pressure of each rate-controlled well: the bhp that
        delivers the target rate (``p_cell − q_res/(WI·λ_t)``), or the limit where the
        well switched to BHP control. ``None`` if there are no rate wells. Uses the
        ``[p, S_w, …]`` state prefix, so it also serves the thermal ``[p, S_w, T]``."""
        if self.rate_well_cells is None:
            return None
        assert self.rate_well_WI is not None
        assert self.rate_well_q is not None
        assert self.rate_well_inj_sw is not None
        assert self.rate_well_depth_offset_m is not None
        assert self.rate_well_bhp_limit is not None
        assert self.rate_well_has_limit is not None
        n = self.n_cells
        p = state[:n]
        sw = state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        wc, WI, q, isw = (
            self.rate_well_cells,
            self.rate_well_WI,
            self.rate_well_q,
            self.rate_well_inj_sw,
        )
        prod = q >= 0
        sw_w = torch.where(prod, sw[wc], isw)
        lam_t = (
            self.relperm.kr_water(sw_w) / self.mu_w + self.relperm.kr_oil(sw_w) / self.mu_o
        ).clamp_min(1e-30)
        lam_w = self.relperm.kr_water(sw_w) / self.mu_w
        lam_o = self.relperm.kr_oil(sw_w) / self.mu_o
        hydrostatic = self.g * self.rate_well_depth_offset_m * (
            lam_w * self.rho_w(p)[wc] + lam_o * self.rho_o(p)[wc]
        )
        bhp_rate = p[wc] - (q / WI + hydrostatic) / lam_t
        lim = self.rate_well_bhp_limit
        viol = self.rate_well_has_limit & torch.where(prod, bhp_rate < lim, bhp_rate > lim)
        return torch.where(viol, lim, bhp_rate)

    def _register_multiperf_wells(
        self,
        multiperf_wells: list[MultiPerfWellSpec] | None,
        dtype: torch.dtype,
    ) -> None:
        """Register **multi-perforation** rate-controlled wells: ``[(perfs, q_target,
        inj_sw, kind, rho_w_std, rho_o_std), ...]`` where ``perfs = [(cell, WI), ...]``
        are the perforations that
        **share one bottom-hole pressure**, ``q_target`` the total well rate (``< 0`` =
        injection, injecting water saturation ``inj_sw``), and ``kind`` one of the
        typed ORAT/WRAT/LRAT/RESV meanings. Surface rates use each well's explicitly
        declared standard density. The shared bhp is solved closed-form from the rate
        constraint ``Σ_k WI_k Λ_k (p_k − bhp) = q_target``; ``Λ_k`` is the total volumetric
        mobility ``λ_w+λ_o`` (RESV) or the surface-weighted mobility
        selected phase surface-weighted mobility ``ρ_α λ_α/ρ_{α,std}`` at perforation
        ``k``, so a single
        perforation under RESV reduces exactly to :meth:`_register_rate_wells`.
        Each perforation's phase **mass** flux is ``F_{α,k} = WI_k ρ_α λ_{α,k} (p_k − bhp)``,
        with ``λ`` from the perforation cell's saturation (producer) or ``inj_sw`` (injector)."""
        if not multiperf_wells:
            self.mp_perf_cell = None
            self.mp_perf_WI = self.mp_perf_well = self.mp_q = None
            self.mp_inj_sw = self.mp_rate_kind = None
            self.mp_standard_water_density = self.mp_standard_oil_density = None
            self.mp_depth_offset_m = None
            self.mp_n_wells = 0
            return
        perf_cell: list[int] = []
        perf_WI: list[float] = []
        perf_depth: list[float] = []
        perf_well: list[int] = []
        q: list[float] = []
        isw: list[float] = []
        kinds: list[int] = []
        standard_water: list[float] = []
        standard_oil: list[float] = []
        kind_codes = {
            WellRateKind.RESV: 0,
            WellRateKind.ORAT: 1,
            WellRateKind.WRAT: 2,
            WellRateKind.LRAT: 3,
        }
        for wid, (perfs, qt, iw, kind, rho_w_std, rho_o_std) in enumerate(
            multiperf_wells
        ):
            for cell, WI, depth_offset_m in perfs:
                perf_cell.append(int(cell))
                perf_WI.append(WI)
                perf_depth.append(depth_offset_m)
                perf_well.append(wid)
            q.append(qt)
            isw.append(iw)
            kinds.append(kind_codes[kind])
            standard_water.append(rho_w_std)
            standard_oil.append(rho_o_std)
        self.register_buffer("mp_perf_cell", torch.tensor(perf_cell, dtype=torch.long))
        self.register_buffer(
            "mp_perf_WI", torch.stack([torch.as_tensor(w, dtype=dtype) for w in perf_WI])
        )
        self.register_buffer("mp_perf_well", torch.tensor(perf_well, dtype=torch.long))
        self.register_buffer(
            "mp_depth_offset_m",
            torch.stack([torch.as_tensor(value, dtype=dtype) for value in perf_depth]),
        )
        self.register_buffer("mp_q", torch.stack([torch.as_tensor(v, dtype=dtype) for v in q]))
        self.register_buffer(
            "mp_inj_sw", torch.stack([torch.as_tensor(v, dtype=dtype) for v in isw])
        )
        self.register_buffer("mp_rate_kind", torch.tensor(kinds, dtype=torch.long))
        self.register_buffer(
            "mp_standard_water_density",
            torch.tensor(standard_water, dtype=dtype),
        )
        self.register_buffer(
            "mp_standard_oil_density",
            torch.tensor(standard_oil, dtype=dtype),
        )
        self.mp_n_wells = len(multiperf_wells)

    def _multiperf_fluxes(
        self,
        p: torch.Tensor,
        sw: torch.Tensor,
        rho_w: torch.Tensor,
        rho_o: torch.Tensor,
    ) -> MultiPerfFluxes | None:
        """``(F_w_perf, F_o_perf, perf_cell, bhp)`` for the multi-perforation rate wells
        (``None`` if there are none). ``F`` are per-perforation phase mass fluxes [kg/s]
        (production +, into the cell residual as ``−F``); ``bhp`` ``(n_wells,)`` the shared
        bottom-hole pressure of each well. Uses the ``[p, S_w]`` state prefix (serves the
        thermal ``[p, S_w, T]`` layout too)."""
        if self.mp_perf_cell is None:  # subclasses that don't register it (3-D /
            return None  # thermal, own residuals) ⇒ multiperf inactive
        assert self.mp_perf_well is not None
        assert self.mp_perf_WI is not None
        assert self.mp_inj_sw is not None
        assert self.mp_rate_kind is not None
        assert self.mp_q is not None
        assert self.mp_standard_water_density is not None
        assert self.mp_standard_oil_density is not None
        assert self.mp_depth_offset_m is not None
        pc, pw, pWI = self.mp_perf_cell, self.mp_perf_well, self.mp_perf_WI
        rw, ro, inj_sw = rho_w[pc], rho_o[pc], self.mp_inj_sw[pw]
        kind, q = self.mp_rate_kind[pw], self.mp_q
        rho_w_standard = self.mp_standard_water_density[pw]
        rho_o_standard = self.mp_standard_oil_density[pw]

        def lam_Lam(
            sw_w: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            lam_w = self.relperm.kr_water(sw_w) / self.mu_w
            lam_o = self.relperm.kr_oil(sw_w) / self.mu_o
            water_weight = torch.where(
                (kind == 0) | (kind == 2) | (kind == 3),
                torch.where(kind == 0, torch.ones_like(rw), rw / rho_w_standard),
                torch.zeros_like(rw),
            )
            oil_weight = torch.where(
                (kind == 0) | (kind == 1) | (kind == 3),
                torch.where(kind == 0, torch.ones_like(ro), ro / rho_o_standard),
                torch.zeros_like(ro),
            )
            return lam_w, lam_o, water_weight, oil_weight

        def solve(sw_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return the closed-form shared BHP and raw well mobility."""
            lam_w, lam_o, water_weight, oil_weight = lam_Lam(sw_w)
            water_coefficient = pWI * water_weight * lam_w
            oil_coefficient = pWI * oil_weight * lam_o
            water_datum_pressure = p[pc] - rw * self.g * self.mp_depth_offset_m
            oil_datum_pressure = p[pc] - ro * self.g * self.mp_depth_offset_m
            num = p.new_zeros(self.mp_n_wells).scatter_add(
                0,
                pw,
                water_coefficient * water_datum_pressure
                + oil_coefficient * oil_datum_pressure,
            )
            raw = p.new_zeros(self.mp_n_wells).scatter_add(
                0, pw, water_coefficient + oil_coefficient
            )
            return (num - q) / raw.clamp_min(1e-30), raw

        # Per-perforation upwind: a PRODUCING perforation (p_k > bhp) draws the CELL composition,
        # an INJECTING one (p_k < bhp) the injected inj_sw. Under CROSS-FLOW (the shared bhp lies
        # between the perforation pressures) some perforations flow opposite to the well's overall
        # sign, so keying the upwind on the well sign mis-splits the phases. The bhp depends on the
        # upwind (through the mobilities), so iterate the *detached* upwind pattern to a fixed point;
        # the final pass is then differentiable with the pattern frozen (as the advective upwind is
        # elsewhere). A single perforation cannot cross-flow ⇒ this reduces to the well-sign form.
        with torch.no_grad():
            up = (q >= 0)[pw]  # well-sign initial guess
            for _ in range(12):
                bhp, _ = solve(torch.where(up, sw[pc], inj_sw))
                lam_w, lam_o, _, _ = lam_Lam(torch.where(up, sw[pc], inj_sw))
                water_drawdown = (
                    p[pc] - bhp[pw] - rw * self.g * self.mp_depth_offset_m
                )
                oil_drawdown = (
                    p[pc] - bhp[pw] - ro * self.g * self.mp_depth_offset_m
                )
                up_new = (lam_w * water_drawdown + lam_o * oil_drawdown) >= 0
                if bool(torch.equal(up_new, up)):
                    break
                up = up_new
        sw_w = torch.where(up, sw[pc], inj_sw)
        lam_w, lam_o, _, _ = lam_Lam(sw_w)
        bhp, raw_den = solve(sw_w)
        water_drawdown = p[pc] - bhp[pw] - rw * self.g * self.mp_depth_offset_m
        oil_drawdown = p[pc] - bhp[pw] - ro * self.g * self.mp_depth_offset_m
        return (
            pWI * rw * lam_w * water_drawdown,
            pWI * ro * lam_o * oil_drawdown,
            pc,
            bhp,
            raw_den,
        )

    def multiperf_well_bhp(self, state: torch.Tensor) -> torch.Tensor | None:
        """Shared bottom-hole pressure ``(n_wells,)`` of each multi-perforation rate well
        delivering its target rate (``NaN`` for a degenerate zero-total-mobility well whose
        bhp is undefined), or ``None`` if there are none (a diagnostic)."""
        if self.mp_perf_cell is None:
            return None
        n = self.n_cells
        p, sw = state[:n], state[n : 2 * n].clamp(min=S_MIN, max=S_MAX)
        fluxes = self._multiperf_fluxes(p, sw, self.rho_w(p), self.rho_o(p))
        assert fluxes is not None
        _, _, _, bhp, raw_den = fluxes
        return torch.where(raw_den > 1e-20, bhp, torch.full_like(bhp, float("nan")))

    def _register_dirichlet(
        self,
        grid: MPFAGrid2D,
        perm_tensor: torch.Tensor,
        dirichlet: Mapping[int, BoundarySpec] | None,
        dtype: torch.dtype,
    ) -> None:
        from ..discretization.mpfa import mpfa_o_face_flux_stencils_bc

        n = self.n_cells
        if dirichlet:
            dir_edges = [int(e) for e in dirichlet]
            L, face_lr, dir_list = mpfa_o_face_flux_stencils_bc(grid, perm_tensor, dir_edges)
            self.register_buffer("L", L)
            self.register_buffer("face_lr", face_lr)
            self.dir_list = dir_list  # ghost order, subclasses align extra BC data (e.g. T_bc)
            self.register_buffer(
                "p_bc",
                torch.stack([torch.as_tensor(dirichlet[e][0], dtype=dtype) for e in dir_list]),
            )
            self.register_buffer(
                "sw_bc",
                torch.stack([torch.as_tensor(dirichlet[e][1], dtype=dtype) for e in dir_list]),
            )
            # Cell adjacent to each Dirichlet ghost (dir_list order): its depth is the
            # ghost's hydrostatic reference D_bc, so the boundary potential carries the
            # same gravity head as the interior (see residual). Mirrors the NFVM depth_bc.
            self.register_buffer(
                "dir_bc_cells",
                torch.tensor([int(grid.edge_cells[e][0]) for e in dir_list], dtype=torch.long),
            )
        else:
            self.dir_list = None
            self.dir_bc_cells = None
            stencils = mpfa_o_face_flux_stencils_full(grid, perm_tensor)
            faces = sorted(stencils)
            L = perm_tensor.new_zeros(len(faces), n)
            lr: list[list[int]] = []
            for fi, f in enumerate(faces):
                for c, t in stencils[f].items():
                    L[fi, c] = t
                left, right = grid.edge_cells[f]
                lr.append([left, right])
            self.register_buffer("L", L)
            self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long))
            self.p_bc = None
            self.sw_bc = None

    def _register_nnc(
        self,
        nnc_pairs: torch.Tensor | None,
        nnc_trans: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        register_nnc(self, nnc_pairs, nnc_trans, dtype)

    @staticmethod
    def _polygon_areas(grid: MPFAGrid2D) -> torch.Tensor:
        return _stable_polygon_areas(grid)

    # ------------------------------------------------------------------
    # State plumbing (mirrors OilWaterModel: [p, S_w])
    # ------------------------------------------------------------------
    def state_size(self) -> int:
        return 2 * self.n_cells

    def initial_state(
        self,
        pressure: ScalarInput,
        sw: ScalarInput,
    ) -> torch.Tensor:
        n, dtype, device = self.n_cells, self.V.dtype, self.V.device
        p = (
            pressure.to(device=device, dtype=dtype)
            if isinstance(pressure, torch.Tensor)
            else torch.full((n,), float(pressure), dtype=dtype, device=device)
        )
        s = (
            sw.to(device=device, dtype=dtype)
            if isinstance(sw, torch.Tensor)
            else torch.full((n,), float(sw), dtype=dtype, device=device)
        )
        return torch.cat([p, s])

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.n_cells
        return {"p": state[:n], "sw": state[n : 2 * n]}

    def rho_w(self, p: torch.Tensor) -> torch.Tensor:
        return self.rho_w_ref * (1.0 + self.c_w * (p - self.p_ref))

    def rho_o(self, p: torch.Tensor) -> torch.Tensor:
        return self.rho_o_ref * (1.0 + self.c_o * (p - self.p_ref))

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------
    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        sources: FlowSourceTerms | None = None,
    ) -> torch.Tensor:
        n = self.n_cells
        if state.shape != (2 * n,) or state_old.shape != (2 * n,):
            raise GeoBrainError(
                "MPFATwoPhaseModel state must be a 1D tensor of length 2*n_cells",
                object_name="MPFATwoPhaseModel",
                field="state",
                expected=(2 * n,),
                actual=tuple(state.shape),
            )
        p, sw = state[:n], state[n:].clamp(min=S_MIN, max=S_MAX)
        p_old, sw_old = state_old[:n], state_old[n:].clamp(min=S_MIN, max=S_MAX)
        so, so_old = 1.0 - sw, 1.0 - sw_old

        rho_w, rho_o = self.rho_w(p), self.rho_o(p)
        rho_w_old, rho_o_old = self.rho_w(p_old), self.rho_o(p_old)

        # Accumulation per phase [kg/s]
        acc_w = self.V * self.phi * (rho_w * sw - rho_w_old * sw_old) / float(dt)
        acc_o = self.V * self.phi * (rho_o * so - rho_o_old * so_old) / float(dt)

        # Per-cell phase potentials: Φ_o = p (+ρ_o g z); Φ_w = p − Pc (+ρ_w g z)
        phi_o = p
        phi_w = p
        if self.capillary is not None:
            phi_w = phi_w - self.capillary(sw)
        if self.depth is not None:  # Φ_α = p − ρ_α g·D (D = depth, +down)
            phi_o = phi_o - rho_o * self.g * self.depth
            phi_w = phi_w - rho_w * self.g * self.depth

        massmob_w = rho_w * self.relperm.kr_water(sw) / self.mu_w
        massmob_o = rho_o * self.relperm.kr_oil(sw) / self.mu_o

        # Dirichlet pressure boundaries enter as fixed-pressure ghost cells: the
        # face operator L is over [cells ; ghosts], so augment the phase
        # potentials and mobilities with the boundary state (upwinding then picks
        # the cell on outflow, the ghost / boundary state on inflow).
        if self.p_bc is not None:
            rho_w_bc, rho_o_bc = self.rho_w(self.p_bc), self.rho_o(self.p_bc)
            phi_o_bc = self.p_bc
            phi_w_bc = self.p_bc - (
                self.capillary(self.sw_bc) if self.capillary is not None else 0.0
            )
            if self.depth is not None:  # Φ_α,bc = p_bc − ρ_α g·D_bc, the boundary
                D_bc = self.depth[
                    self.dir_bc_cells
                ]  # ghost carries the SAME hydrostatic head as the
                phi_o_bc = (
                    phi_o_bc - rho_o_bc * self.g * D_bc
                )  # interior (D_bc = adjacent-cell depth), so a
                phi_w_bc = (
                    phi_w_bc - rho_w_bc * self.g * D_bc
                )  # hydrostatic column with a lateral Dirichlet BC
                # at its equilibrium pressure has NO spurious boundary flux (was ∝ ρgD).
            phi_o = torch.cat([phi_o, phi_o_bc])
            phi_w = torch.cat([phi_w, phi_w_bc])
            massmob_w = torch.cat(
                [massmob_w, rho_w_bc * self.relperm.kr_water(self.sw_bc) / self.mu_w]
            )
            massmob_o = torch.cat(
                [massmob_o, rho_o_bc * self.relperm.kr_oil(self.sw_bc) / self.mu_o]
            )

        # MPFA geometric flux per face (endpoint0→endpoint1), phase-potential upwind
        G_o = self.L @ phi_o
        G_w = self.L @ phi_w
        left_cells, right_cells = self.face_lr[:, 0], self.face_lr[:, 1]
        F_w = massmob_w[upwind_cell(G_w, self.face_lr)] * G_w
        F_o = massmob_o[upwind_cell(G_o, self.face_lr)] * G_o

        # scatter to real cells only: endpoint 0 is always a cell; endpoint 1 is
        # a ghost on Dirichlet faces (mass leaving/entering the domain there).
        right_is_real = right_cells < n
        internal_cells = self.face_lr[right_is_real]
        boundary_cells = left_cells[~right_is_real]

        def face_divergence(face_flux: torch.Tensor) -> torch.Tensor:
            return scatter_internal_face_flux(
                face_flux[right_is_real], internal_cells, n
            ) + scatter_boundary_outflow(face_flux[~right_is_real], boundary_cells, n)

        R_w = acc_w + face_divergence(F_w)
        R_o = acc_o + face_divergence(F_o)

        # Non-neighbour connections (faults): two-point flux added on top of the
        # multi-point face flux (a, b are cell indices ⇒ index the cell part of
        # the possibly-augmented potential / mobility arrays).
        if self.nnc_trans is not None:
            R_w = nnc_phase_flux(R_w, self.nnc_pairs, self.nnc_trans, phi_w, massmob_w)
            R_o = nnc_phase_flux(R_o, self.nnc_pairs, self.nnc_trans, phi_o, massmob_o)

        # Prescribed-flux (Neumann / rate) boundary faces: total volumetric flux q
        # (outward +), split by fractional flow: cell's on outflow, sw_inj's on
        # inflow, so the water cut is solution-dependent (a rate-controlled BC).
        if self.neumann_q is not None:
            nb, q, swj = self.neumann_cells, self.neumann_q, self.neumann_sw_inj
            lam_w_c = self.relperm.kr_water(sw)[nb] / self.mu_w
            lam_o_c = self.relperm.kr_oil(sw)[nb] / self.mu_o
            lam_w_j = self.relperm.kr_water(swj) / self.mu_w
            lam_o_j = self.relperm.kr_oil(swj) / self.mu_o
            out = q >= 0
            lw = torch.where(out, lam_w_c, lam_w_j)
            lo = torch.where(out, lam_o_c, lam_o_j)
            tot = (lw + lo).clamp_min(1e-30)
            Fw_b = rho_w[nb] * (lw / tot) * q  # mass outflow (q>0) / inflow (q<0)
            Fo_b = rho_o[nb] * (lo / tot) * q
            R_w = R_w + scatter_boundary_outflow(Fw_b, nb, n)
            R_o = R_o + scatter_boundary_outflow(Fo_b, nb, n)

        # BHP-controlled Peaceman wells: q_α = WI·(ρ_α k_rα/μ_α)·(p_cell − bhp),
        # phase split by fractional flow: cell's on production (p_cell>bhp),
        # inj_sw's on injection. The well rate emerges from the solution pressure.
        if self.well_cells is not None:
            wc, WI, bhp, isw = self.well_cells, self.well_WI, self.well_bhp, self.well_inj_sw
            datum_drawdown = p[wc] - bhp
            sw_w = torch.where(
                datum_drawdown >= 0, sw[wc], isw
            )  # production uses the cell, injection the well
            lam_w = self.relperm.kr_water(sw_w) / self.mu_w
            lam_o = self.relperm.kr_oil(sw_w) / self.mu_o
            water_drawdown = (
                datum_drawdown - rho_w[wc] * self.g * self.well_depth_offset_m
            )
            oil_drawdown = datum_drawdown - rho_o[wc] * self.g * self.well_depth_offset_m
            water_volume = WI * lam_w * water_drawdown
            oil_volume = WI * lam_o * oil_drawdown
            Fw_w = WI * rho_w[wc] * lam_w * water_drawdown
            Fo_w = WI * rho_o[wc] * lam_o * oil_drawdown
            q_reservoir = water_volume + oil_volume
            cap = self.well_has_rate_limit & (
                q_reservoir.abs() > self.well_rate_limit
            )
            cap_scale = self.well_rate_limit / q_reservoir.abs().clamp_min(1e-30)
            Fw_w = torch.where(cap, Fw_w * cap_scale, Fw_w)
            Fo_w = torch.where(cap, Fo_w * cap_scale, Fo_w)
            R_w = R_w + scatter_boundary_outflow(Fw_w, wc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_w, wc, n)

        # Rate-controlled Peaceman wells: prescribe the total reservoir-volumetric rate
        # q_res (the bhp is solved); the phases split by fractional flow, cell's on
        # production (q_res>0), inj_sw's on injection (q_res<0). With a bhp_limit the
        # well switches to BHP control when the rate-implied bhp would violate it.
        if self.rate_well_cells is not None:
            assert self.rate_well_WI is not None
            assert self.rate_well_q is not None
            assert self.rate_well_inj_sw is not None
            assert self.rate_well_depth_offset_m is not None
            assert self.rate_well_bhp_limit is not None
            assert self.rate_well_has_limit is not None
            wc, WI, q, isw = (
                self.rate_well_cells,
                self.rate_well_WI,
                self.rate_well_q,
                self.rate_well_inj_sw,
            )
            prod = q >= 0
            sw_w = torch.where(prod, sw[wc], isw)
            lam_w = self.relperm.kr_water(sw_w) / self.mu_w
            lam_o = self.relperm.kr_oil(sw_w) / self.mu_o
            lam_t = (lam_w + lam_o).clamp_min(1e-30)
            hydrostatic = self.g * self.rate_well_depth_offset_m * (
                lam_w * rho_w[wc] + lam_o * rho_o[wc]
            )
            bhp_rate = p[wc] - (q / WI + hydrostatic) / lam_t
            water_drawdown = (
                p[wc]
                - bhp_rate
                - rho_w[wc] * self.g * self.rate_well_depth_offset_m
            )
            oil_drawdown = (
                p[wc]
                - bhp_rate
                - rho_o[wc] * self.g * self.rate_well_depth_offset_m
            )
            Fw_rate = WI * rho_w[wc] * lam_w * water_drawdown
            Fo_rate = WI * rho_o[wc] * lam_o * oil_drawdown
            lim = self.rate_well_bhp_limit
            viol = self.rate_well_has_limit & torch.where(prod, bhp_rate < lim, bhp_rate > lim)
            water_limit_drawdown = (
                p[wc] - lim - rho_w[wc] * self.g * self.rate_well_depth_offset_m
            )
            oil_limit_drawdown = (
                p[wc] - lim - rho_o[wc] * self.g * self.rate_well_depth_offset_m
            )
            Fw_w = torch.where(
                viol, WI * rho_w[wc] * lam_w * water_limit_drawdown, Fw_rate
            )
            Fo_w = torch.where(
                viol, WI * rho_o[wc] * lam_o * oil_limit_drawdown, Fo_rate
            )
            R_w = R_w + scatter_boundary_outflow(Fw_w, wc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_w, wc, n)

        mp = self._multiperf_fluxes(
            p, sw, rho_w, rho_o
        )  # multi-perforation rate / surface-rate wells
        if mp is not None:
            Fw_p, Fo_p, pc = mp[0], mp[1], mp[2]
            R_w = R_w + scatter_boundary_outflow(Fw_p, pc, n)
            R_o = R_o + scatter_boundary_outflow(Fo_p, pc, n)

        R_w = R_w - source_block(sources, family="phase", name="water", like=R_w)
        R_o = R_o - source_block(sources, family="phase", name="oil", like=R_o)
        return torch.cat([R_w, R_o])

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
        **kw: FlowSourceTerms | None,
    ) -> torch.Tensor:
        return torch.autograd.functional.jacobian(
            lambda s: self.residual(s, state_old, dt, **kw),
            state,
            vectorize=True,
        )


class MPFATwoPhaseModel3D(MPFATwoPhaseModel):
    """Oil-water two-phase Darcy flow with a 3-D MPFA-O multi-point flux (SI).

    Args mirror :class:`MPFATwoPhaseModel`, with a :class:`MPFAGrid3D` and
    ``(n_cells, 3, 3)`` permeability tensors; ``depth`` (for gravity) defaults to
    the cell-centroid z if gravity is enabled by passing it explicitly.
    """

    schema = _flow_model_schema(
        model_name="MPFATwoPhaseModel3D",
        primary_fields=(
            ("pressure", "Pa", ("cell",), ()),
            ("sw", "1", ("cell",), ()),
        ),
        residual_blocks=(
            ("water_mass", "kg/s", "mass", "pressure"),
            ("oil_mass", "kg/s", "mass", "sw"),
        ),
        grid_kinds=("mpfa-3d",),
        phases=("water", "oil"),
        structured_sources=True,
    )

    def __init__(
        self,
        grid: MPFAGrid3D,
        perm_tensor: torch.Tensor,
        porosity: ScalarInput,
        relperm: RelPerm,
        *,
        rho_w_ref: float = 1000.0,
        rho_o_ref: float = 800.0,
        mu_w: float = 1e-3,
        mu_o: float = 2e-3,
        c_w: float = 0.0,
        c_o: float = 0.0,
        p_ref: float = 1e7,
        capillary: Callable[[torch.Tensor], torch.Tensor] | None = None,
        depth: torch.Tensor | None = None,
        gravity: float = 9.81,
        cell_volumes: torch.Tensor | None = None,
        nnc_pairs: torch.Tensor | None = None,
        nnc_trans: torch.Tensor | None = None,
        dirichlet: Mapping[int, BoundarySpec] | None = None,
        neumann: Mapping[int, BoundarySpec] | None = None,
        wells: list[Well] | None = None,
    ) -> None:
        _ModuleBase.__init__(self)
        self.grid = grid
        self.relperm = relperm
        self.capillary = capillary
        self.rho_w_ref, self.rho_o_ref = float(rho_w_ref), float(rho_o_ref)
        self.mu_w, self.mu_o = float(mu_w), float(mu_o)
        self.c_w, self.c_o = float(c_w), float(c_o)
        self.p_ref = float(p_ref)
        self.g = float(gravity)
        n = len(grid.cell_nodes)
        self.n_cells = n
        dtype = perm_tensor.dtype

        if isinstance(porosity, torch.Tensor):
            self.register_buffer("phi", porosity.to(dtype))
        else:
            self.register_buffer("phi", torch.full((n,), float(porosity), dtype=dtype))
        if cell_volumes is None:
            cell_volumes = hex_cell_volumes(grid).to(dtype)
        self.register_buffer("V", cell_volumes.to(dtype))
        if depth is not None:
            self.register_buffer("depth", depth.to(dtype))
        else:
            self.depth = None

        if dirichlet:  # ghost-cell augmented operator
            from ..discretization.mpfa3d import mpfa_o_face_flux_stencils_3d_bc

            dir_faces = [int(f) for f in dirichlet]
            L, face_lr, dir_list = mpfa_o_face_flux_stencils_3d_bc(grid, perm_tensor, dir_faces)
            self.register_buffer("L", L)
            self.register_buffer("face_lr", face_lr)
            self.dir_list = dir_list  # ghost order, subclasses align extra BC data (e.g. T_bc)
            self.register_buffer(
                "p_bc",
                torch.stack([torch.as_tensor(dirichlet[f][0], dtype=dtype) for f in dir_list]),
            )
            self.register_buffer(
                "sw_bc",
                torch.stack([torch.as_tensor(dirichlet[f][1], dtype=dtype) for f in dir_list]),
            )
            # Cell adjacent to each Dirichlet ghost (dir_list order) → its depth is the
            # ghost's hydrostatic reference D_bc (see residual).
            self.register_buffer(
                "dir_bc_cells",
                torch.tensor([int(grid.face_cells[f][0]) for f in dir_list], dtype=torch.long),
            )
        else:
            self.dir_bc_cells = None
            stencils = mpfa_o_face_flux_stencils_3d_full(grid, perm_tensor)
            faces = sorted(stencils)
            L = perm_tensor.new_zeros(len(faces), n)
            lr: list[list[int]] = []
            for fi, f in enumerate(faces):
                for c, t in stencils[f].items():
                    L[fi, c] = t
                left, right = grid.face_cells[f]
                lr.append([left, right])
            self.register_buffer("L", L)
            self.register_buffer("face_lr", torch.tensor(lr, dtype=torch.long))
            self.dir_list = None
            self.p_bc = None
            self.sw_bc = None
        self._register_nnc(nnc_pairs, nnc_trans, dtype)
        self._register_neumann(grid, neumann, dtype)
        bhp_specs, rate_specs, multiperf_specs = _two_phase_well_specs(wells)
        self._register_wells(bhp_specs, dtype)
        self._register_rate_wells(rate_specs, dtype)
        self._register_multiperf_wells(
            multiperf_specs, dtype
        )  # dimension-agnostic shared-bhp wells (3-D)


__all__ = ["MPFATwoPhaseModel", "MPFATwoPhaseModel3D"]
