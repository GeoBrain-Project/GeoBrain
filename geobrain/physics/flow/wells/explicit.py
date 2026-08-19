"""
Well models for the flow module.

Three primitives:

- :class:`Perforation`: one well/cell connection with an SI Peaceman
  geometric well index (m³). Multi-perforation wells stitch several cells into one
  bottomhole pressure (BHP).
- :class:`WellControl` (``BHPControl`` / ``RateControl``): what the
  reservoir engineer wants the well to do.
- :class:`Well` + :class:`WellGroup`: collection that produces sparse,
  schema-declared :class:`FlowSourceTerms` phase-mass blocks in kg/s
  (positive = injection into the cell).

Sign convention throughout: rates are signed per phase, positive =
into the reservoir cell. Producers contribute negative rates; gas
injectors contribute positive ``source_gas_rates``.

Wells deliberately do **not** live inside the model residual. Pulling wells out
preserves three properties:

1. The kernel residual stays a pure function of ``state`` plus typed source
   blocks, autograd-friendly and easy to differentiate.
2. Wells are unit-testable in isolation (Peaceman index, BHP control,
   RATE allocation) without spinning up a Newton solve.
3. The same Well types work across single-phase / oil-water /
   black-oil models; phase dispatch is handled here, not duplicated
   inside three residual functions.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Literal

import torch
import torch.nn as nn

from ....core import GeoBrainError
from .._defaults import DEVICE, DTYPE, EPS
from ..errors import FlowContractError


_STANDARD_GRAVITY_M_S2 = 9.80665


class WellRateKind(str, Enum):
    """Unambiguous petroleum well-rate meanings, all expressed in m³/s."""

    ORAT = "ORAT"
    WRAT = "WRAT"
    GRAT = "GRAT"
    LRAT = "LRAT"
    RESV = "RESV"


@dataclass(frozen=True, slots=True)
class RateControl:
    """Positive SI well-rate magnitude with an explicit rate meaning.

    Attributes:
        kind: which phase/total rate is controlled
            (:class:`~geobrain.physics.flow.WellRateKind`).
        target_m3_s: target volumetric rate [m^3/s].
    """

    kind: WellRateKind
    target_m3_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WellRateKind):
            raise FlowContractError(
                "RateControl.kind must be a WellRateKind",
                object_name="RateControl",
                field="kind",
                expected=tuple(item.value for item in WellRateKind),
                actual=self.kind,
            )
        if not math.isfinite(self.target_m3_s) or self.target_m3_s <= 0.0:
            raise FlowContractError(
                "RateControl target must be finite and positive",
                object_name="RateControl",
                field="target_m3_s",
                expected="> 0 m³/s",
                actual=self.target_m3_s,
            )


@dataclass(frozen=True, slots=True)
class BHPControl:
    """Positive finite bottom-hole pressure in TRUE pascals.

    The well layer computes drawdown in SI against the adapter-converted
    reservoir pressure. A model built on the native field-unit fixture
    convention (state in psi) must therefore convert its BHP targets with
    :func:`geobrain.physics.flow.adapters.field_units.pressure_psi_to_pa`,
    passing the native psi magnitude here makes a producer see an enormous
    drawdown and clamps an injector inert.
    """

    pressure_pa: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.pressure_pa) or self.pressure_pa <= 0.0:
            raise FlowContractError(
                "BHPControl pressure must be finite and positive",
                object_name="BHPControl",
                field="pressure_pa",
                expected="> 0 Pa",
                actual=self.pressure_pa,
            )


@dataclass(frozen=True, slots=True)
class WellRateReport:
    """Four distinct positive-magnitude well-rate observations in SI."""

    oil_surface_m3_s: torch.Tensor
    water_surface_m3_s: torch.Tensor
    gas_standard_m3_s: torch.Tensor
    reservoir_m3_s: torch.Tensor

    def __post_init__(self) -> None:
        blocks = (
            self.oil_surface_m3_s,
            self.water_surface_m3_s,
            self.gas_standard_m3_s,
            self.reservoir_m3_s,
        )
        reference = blocks[0]
        if not torch.is_tensor(reference) or not reference.is_floating_point():
            raise FlowContractError(
                "WellRateReport entries must be floating tensors",
                object_name="WellRateReport",
                field="oil_surface_m3_s",
                expected="floating torch.Tensor",
                actual=type(reference).__name__,
            )
        for block in blocks[1:]:
            if (
                not torch.is_tensor(block)
                or not block.is_floating_point()
                or block.shape != reference.shape
                or block.dtype != reference.dtype
                or block.device != reference.device
            ):
                raise FlowContractError(
                    "WellRateReport entries must share shape, dtype, and device",
                    object_name="WellRateReport",
                    field="rate blocks",
                    expected=(tuple(reference.shape), str(reference.dtype), str(reference.device)),
                    actual=(
                        tuple(block.shape) if torch.is_tensor(block) else None,
                        str(block.dtype) if torch.is_tensor(block) else type(block).__name__,
                        str(block.device) if torch.is_tensor(block) else None,
                    ),
                )


def controlled_rate(report: WellRateReport, kind: WellRateKind) -> torch.Tensor:
    """Select one typed rate; LRAT is oil + water and never includes gas."""

    if kind is WellRateKind.ORAT:
        return report.oil_surface_m3_s
    if kind is WellRateKind.WRAT:
        return report.water_surface_m3_s
    if kind is WellRateKind.GRAT:
        return report.gas_standard_m3_s
    if kind is WellRateKind.LRAT:
        return report.oil_surface_m3_s + report.water_surface_m3_s
    if kind is WellRateKind.RESV:
        return report.reservoir_m3_s
    raise FlowContractError(
        "Unsupported well-rate kind",
        object_name="controlled_rate",
        field="kind",
        expected=tuple(item.value for item in WellRateKind),
        actual=kind,
    )


def _validate_source_block(
    tensor: torch.Tensor,
    *,
    object_name: str,
    field: str,
) -> tuple[tuple[int, ...], torch.dtype, torch.device]:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point() or tensor.ndim != 1:
        raise FlowContractError(
            "Flow source block must be a one-dimensional floating tensor",
            object_name=object_name,
            field=field,
            expected="floating Tensor[cell]",
            actual=(type(tensor).__name__, getattr(tensor, "shape", None)),
        )
    values = tensor.coalesce().values() if tensor.is_sparse else tensor
    if not bool(torch.isfinite(values).all()):
        raise FlowContractError(
            "Flow source block must be finite",
            object_name=object_name,
            field=field,
            expected="all finite",
            actual="contains NaN or infinity",
        )
    return tuple(tensor.shape), tensor.dtype, tensor.device


@dataclass(frozen=True, slots=True)
class FlowSourceTerms:
    """Schema-declared sparse or dense SI source blocks.

    Positive values inject into a cell; negative values produce from it.
    Phase blocks are kg/s, component blocks are mol/s, and energy is W.
    """

    phase_mass_kg_s: Mapping[str, torch.Tensor]
    component_mol_s: Mapping[str, torch.Tensor]
    energy_w: torch.Tensor | None = None

    def __post_init__(self) -> None:
        references: list[tuple[tuple[int, ...], torch.dtype, torch.device]] = []
        for map_name, blocks in (
            ("phase_mass_kg_s", self.phase_mass_kg_s),
            ("component_mol_s", self.component_mol_s),
        ):
            if not isinstance(blocks, Mapping):
                raise FlowContractError(
                    "Flow source blocks must be mappings",
                    object_name="FlowSourceTerms",
                    field=map_name,
                    expected="Mapping[str, Tensor[cell]]",
                    actual=type(blocks).__name__,
                )
            copied: dict[str, torch.Tensor] = {}
            for name, block in blocks.items():
                if not isinstance(name, str) or not name:
                    raise FlowContractError(
                        "Flow source block names must be non-empty strings",
                        object_name="FlowSourceTerms",
                        field=map_name,
                        expected="non-empty string keys",
                        actual=name,
                    )
                copied[name] = block
                references.append(
                    _validate_source_block(
                        block,
                        object_name="FlowSourceTerms",
                        field=f"{map_name}.{name}",
                    )
                )
            object.__setattr__(self, map_name, copied)
        if self.energy_w is not None:
            references.append(
                _validate_source_block(
                    self.energy_w,
                    object_name="FlowSourceTerms",
                    field="energy_w",
                )
            )
        if references and any(item != references[0] for item in references[1:]):
            raise FlowContractError(
                "Flow source blocks must share shape, dtype, and device",
                object_name="FlowSourceTerms",
                field="source blocks",
                expected=references[0],
                actual=tuple(references),
            )


def source_block(
    sources: FlowSourceTerms | None,
    *,
    family: Literal["phase", "component", "energy"],
    name: str | None,
    like: torch.Tensor,
) -> torch.Tensor:
    """Return one validated dense source block or a zero block like ``like``."""

    if sources is None:
        return torch.zeros_like(like)
    if not isinstance(sources, FlowSourceTerms):
        raise FlowContractError(
            "Model sources must be FlowSourceTerms",
            object_name="source_block",
            field="sources",
            expected="FlowSourceTerms | None",
            actual=type(sources).__name__,
        )
    if family == "phase":
        block = sources.phase_mass_kg_s.get(name or "")
    elif family == "component":
        block = sources.component_mol_s.get(name or "")
    else:
        block = sources.energy_w
    if block is None:
        return torch.zeros_like(like)
    if block.shape != like.shape or block.dtype != like.dtype or block.device != like.device:
        raise FlowContractError(
            "Flow source block does not match model state",
            object_name="source_block",
            field=name or family,
            expected=(tuple(like.shape), str(like.dtype), str(like.device)),
            actual=(tuple(block.shape), str(block.dtype), str(block.device)),
        )
    if not block.is_sparse:
        return block
    sparse = block.coalesce()
    return torch.zeros_like(like).index_add(
        0,
        sparse.indices()[0],
        sparse.values(),
    )


# ---------------------------------------------------------------------------
# Peaceman well index
# ---------------------------------------------------------------------------


def compute_well_index(
    dx_m: float,
    dy_m: float,
    dz_m: float,
    kx_m2: float,
    ky_m2: float | None = None,
    well_radius_m: float = 0.1,
    skin: float = 0.0,
) -> float:
    """
    Peaceman (1983) well index for a vertical well in a Cartesian block.

    Args:
        dx_m, dy_m:    block extent perpendicular to the well [m].
        dz_m:          block thickness in the well direction [m].
        kx_m2, ky_m2:  per-axis permeability [m²]. ``ky_m2`` defaults to ``kx_m2``
            (isotropic).
        well_radius_m: wellbore radius ``r_w`` [m].
        skin:          dimensionless skin factor (positive = damaged).

    Returns the SI geometric well index [m³]. Multiplication by mobility
    [1/(Pa·s)] and pressure drawdown [Pa] yields reservoir volume [m³/s].
    """
    if ky_m2 is None:
        ky_m2 = kx_m2
    if dx_m <= 0 or dy_m <= 0 or dz_m <= 0:
        raise FlowContractError(
            "compute_well_index requires positive block dimensions",
            object_name="compute_well_index", field="(dx_m, dy_m, dz_m)",
            expected="all > 0 m", actual=(dx_m, dy_m, dz_m),
        )
    if kx_m2 <= 0 or ky_m2 <= 0:
        raise FlowContractError(
            "compute_well_index requires positive permeabilities",
            object_name="compute_well_index", field="(kx_m2, ky_m2)",
            expected="all > 0 m²", actual=(kx_m2, ky_m2),
        )
    if well_radius_m <= 0:
        raise FlowContractError(
            "compute_well_index requires positive well_radius_m",
            object_name="compute_well_index", field="well_radius_m",
            expected="> 0 m", actual=well_radius_m,
        )
    ratio = ky_m2 / kx_m2
    r_o_num = math.sqrt(
        math.sqrt(ratio) * dx_m**2 + math.sqrt(1.0 / ratio) * dy_m**2
    )
    r_o_den = ratio ** 0.25 + (1.0 / ratio) ** 0.25
    r_o = 0.28 * r_o_num / r_o_den
    denominator = math.log(r_o / well_radius_m) + skin
    if denominator <= 0.0:
        raise FlowContractError(
            "Peaceman equivalent radius must exceed the effective well radius",
            object_name="compute_well_index",
            field="well_radius_m/skin",
            expected="log(r_e / r_w) + skin > 0",
            actual=denominator,
        )
    return 2.0 * math.pi * math.sqrt(kx_m2 * ky_m2) * dz_m / denominator


# ---------------------------------------------------------------------------
# Well objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Perforation:
    """One well/cell connection.

    ``depth_offset_m`` is the depth offset below the well's BHP datum,
    ``z_cell − z_ref`` [m] (``z`` is +down). The hydrostatic pressure
    ``ρ_mix · g · depth_offset_m`` adds to the perforation
    drawdown so a multi-perforation well referenced to one BHP sees the
    correct per-connection pressure. Default ``0.0`` = single-node well
    (BHP datum at the perforation), reproducing the gravity-free flow law.

    Attributes:
        cell_idx: connected grid cell index.
        well_index_m3: Peaceman well index [m^3].
        depth_offset_m: completion depth offset from the well datum [m].
    """

    cell_idx: int
    well_index_m3: float
    depth_offset_m: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.cell_idx, bool) or not isinstance(self.cell_idx, int):
            raise FlowContractError(
                "Perforation cell_idx must be an integer",
                object_name="Perforation",
                field="cell_idx",
                expected="int",
                actual=self.cell_idx,
            )
        if not math.isfinite(self.well_index_m3) or self.well_index_m3 <= 0.0:
            raise FlowContractError(
                "Perforation well index must be finite and positive",
                object_name="Perforation",
                field="well_index_m3",
                expected="> 0 m³",
                actual=self.well_index_m3,
            )
        if not math.isfinite(self.depth_offset_m):
            raise FlowContractError(
                "Perforation depth offset must be finite",
                object_name="Perforation",
                field="depth_offset_m",
                expected="finite m",
                actual=self.depth_offset_m,
            )


WellType = Literal["INJ", "PROD"]
InjectionPhase = Literal["water", "oil", "gas", "fluid"]


@dataclass(frozen=True, slots=True)
class WellStandardConditions:
    """Declared standard conditions used by ORAT/WRAT/GRAT/LRAT."""

    pressure_pa: float
    temperature_k: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.pressure_pa)
            or self.pressure_pa <= 0.0
            or not math.isfinite(self.temperature_k)
            or self.temperature_k <= 0.0
        ):
            raise FlowContractError(
                "Well standard conditions must be finite and positive",
                object_name="WellStandardConditions",
                field="pressure_pa/temperature_k",
                expected="> 0 Pa and > 0 K",
                actual=(self.pressure_pa, self.temperature_k),
            )


def _immutable_phase_scalars(
    values: Mapping[str, float],
    *,
    field: str,
    strictly_positive: bool,
) -> Mapping[str, float]:
    """Copy phase metadata into an immutable ``str -> float`` mapping."""

    if not isinstance(values, Mapping):
        raise FlowContractError(
            "Well phase metadata must be a mapping",
            object_name="Well",
            field=field,
            expected="Mapping[str, finite float]",
            actual=type(values).__qualname__,
        )
    normalized: dict[str, float] = {}
    for phase, raw_value in values.items():
        if not isinstance(phase, str) or not phase:
            raise FlowContractError(
                "Well phase metadata keys must be non-empty strings",
                object_name="Well",
                field=field,
                expected="non-empty str keys",
                actual=phase,
            )
        if isinstance(raw_value, bool):
            valid_value = False
        elif isinstance(raw_value, torch.Tensor):
            valid_value = (
                raw_value.numel() == 1
                and raw_value.dtype is not torch.bool
                and not raw_value.is_complex()
                and bool(torch.isfinite(raw_value.detach()).all())
            )
        else:
            valid_value = isinstance(raw_value, Real) and math.isfinite(float(raw_value))
        if not valid_value:
            raise FlowContractError(
                "Well phase metadata values must be finite real scalars",
                object_name="Well",
                field=field,
                expected="finite real scalar (bool and non-scalar Tensor are invalid)",
                actual=(phase, raw_value),
            )
        scalar = (
            float(raw_value.detach().item())
            if isinstance(raw_value, torch.Tensor)
            else float(raw_value)
        )
        if scalar < 0.0 or (strictly_positive and scalar == 0.0):
            raise FlowContractError(
                "Well phase metadata value is outside its physical domain",
                object_name="Well",
                field=field,
                expected="> 0" if strictly_positive else ">= 0",
                actual=(phase, scalar),
            )
        normalized[str(phase)] = scalar
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class Well:
    """
    A reservoir well.

    Args:
        name:             user-facing name.
        well_type:        ``"INJ"`` or ``"PROD"``.
        control:          a :class:`BHPControl` or :class:`RateControl`.
        perforations:     one or more :class:`Perforation` entries.
        injection_phase:  explicit injected phase; only consulted for injectors.
    """

    name: str
    well_type: WellType
    control: BHPControl | RateControl
    perforations: tuple[Perforation, ...]
    injection_phase: InjectionPhase | None = None
    injection_composition: Mapping[str, float] | None = None
    injection_temperature_k: float | None = None
    standard_conditions: WellStandardConditions | None = None
    standard_densities_kg_m3: Mapping[str, float] | None = None
    # Optional operating limits used by the implicit BHP well model
    # (:mod:`well_system`) for between-solve control switching:
    #   bhp_limit_pa: producer BHP floor / injector BHP ceiling [Pa].
    #   rate_limit:    typed volumetric cap [m³/s] for a BHP-target well.
    #   datum_depth_m: depth of the BHP reference datum [m, +down];
    #                   per-perforation ``depth_offset_m`` carries the head.
    bhp_limit_pa: float | None = None
    rate_limit: RateControl | None = None
    datum_depth_m: float | None = None

    def __post_init__(self) -> None:
        if self.well_type not in ("INJ", "PROD"):
            raise GeoBrainError(
                "Well.well_type must be 'INJ' or 'PROD'",
                object_name="Well", field="well_type",
                expected="INJ or PROD", actual=self.well_type,
            )
        if self.well_type == "INJ" and self.injection_phase not in (
            "water",
            "oil",
            "gas",
            "fluid",
        ):
            raise GeoBrainError(
                "Injector well needs injection_phase in {'water', 'gas'}",
                object_name="Well", field="injection_phase",
                expected="'water', 'oil', 'gas', or 'fluid'", actual=self.injection_phase,
            )
        if not self.perforations:
            raise GeoBrainError(
                "Well must have at least one perforation",
                object_name="Well", field="perforations",
                expected="non-empty", actual=(),
            )
        if self.injection_composition is not None:
            composition = _immutable_phase_scalars(
                self.injection_composition,
                field="injection_composition",
                strictly_positive=False,
            )
            if not math.isclose(
                sum(composition.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise FlowContractError(
                    "Well injection composition must be nonnegative and sum to one",
                    object_name="Well",
                    field="injection_composition",
                    expected="nonnegative fractions summing to 1",
                    actual=composition,
                )
            object.__setattr__(self, "injection_composition", composition)
        if self.injection_temperature_k is not None and (
            not math.isfinite(self.injection_temperature_k)
            or self.injection_temperature_k <= 0.0
        ):
            raise FlowContractError(
                "Injection temperature must be finite and positive",
                object_name="Well",
                field="injection_temperature_k",
                expected="> 0 K",
                actual=self.injection_temperature_k,
            )
        if self.standard_densities_kg_m3 is not None:
            object.__setattr__(
                self,
                "standard_densities_kg_m3",
                _immutable_phase_scalars(
                    self.standard_densities_kg_m3,
                    field="standard_densities_kg_m3",
                    strictly_positive=True,
                ),
            )
        for name, value in (
            ("bhp_limit_pa", self.bhp_limit_pa),
            ("datum_depth_m", self.datum_depth_m),
        ):
            if value is not None and not math.isfinite(value):
                raise FlowContractError(
                    "Well SI metadata must be finite",
                    object_name="Well",
                    field=name,
                    expected="finite",
                    actual=value,
                )


# ---------------------------------------------------------------------------
# WellGroup: sparse, canonical-SI source assembly
# ---------------------------------------------------------------------------


def _validate_phase_inputs(
    pressure_pa: torch.Tensor,
    mobilities_pa_s_inv: Mapping[str, torch.Tensor],
    densities_kg_m3: Mapping[str, torch.Tensor],
    n_cells: int,
) -> None:
    if pressure_pa.shape != (n_cells,) or not pressure_pa.is_floating_point():
        raise FlowContractError(
            "Well pressure must be a floating Tensor[cell]",
            object_name="WellGroup",
            field="pressure_pa",
            expected=(n_cells,),
            actual=tuple(pressure_pa.shape),
        )
    if not mobilities_pa_s_inv:
        raise FlowContractError(
            "Well assembly requires at least one phase mobility",
            object_name="WellGroup",
            field="mobilities_pa_s_inv",
            expected="non-empty phase mapping",
            actual=(),
        )
    for phase, mobility in mobilities_pa_s_inv.items():
        density = densities_kg_m3.get(phase)
        for field, tensor in (("mobility", mobility), ("density", density)):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != (n_cells,)
                or tensor.dtype != pressure_pa.dtype
                or tensor.device != pressure_pa.device
            ):
                raise FlowContractError(
                    "Well phase inputs must match pressure shape, dtype, and device",
                    object_name="WellGroup",
                    field=f"{phase}.{field}",
                    expected=((n_cells,), str(pressure_pa.dtype), str(pressure_pa.device)),
                    actual=(
                        tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None,
                        str(tensor.dtype)
                        if isinstance(tensor, torch.Tensor)
                        else type(tensor).__name__,
                        str(tensor.device) if isinstance(tensor, torch.Tensor) else None,
                    ),
                )


def validate_well_control(well: Well, phases: tuple[str, ...]) -> None:
    """Fail before assembly when a control cannot be represented by a model."""

    phase_set = set(phases)
    if well.well_type == "INJ" and well.injection_phase not in phase_set:
        raise FlowContractError(
            "Injector phase is unsupported by the target model",
            object_name="Well",
            field="injection_phase",
            expected=tuple(sorted(phase_set)),
            actual=well.injection_phase,
        )
    if not isinstance(well.control, RateControl):
        return
    required_phase = {
        WellRateKind.ORAT: "oil",
        WellRateKind.WRAT: "water",
        WellRateKind.GRAT: "gas",
    }.get(well.control.kind)
    if required_phase is not None and required_phase not in phase_set:
        raise FlowContractError(
            "Rate kind is unsupported by the target model",
            object_name="Well",
            field="control.kind",
            expected=tuple(
                kind.value
                for kind, phase in (
                    (WellRateKind.ORAT, "oil"),
                    (WellRateKind.WRAT, "water"),
                    (WellRateKind.GRAT, "gas"),
                )
                if phase in phase_set
            ) + (WellRateKind.RESV.value,),
            actual=well.control.kind.value,
        )
    if well.control.kind is WellRateKind.LRAT and not ({"oil", "water"} & phase_set):
        raise FlowContractError(
            "LRAT requires an oil or water phase",
            object_name="Well",
            field="control.kind",
            expected="oil and/or water model",
            actual=tuple(sorted(phase_set)),
        )
    if well.well_type == "INJ":
        injection_phase = well.injection_phase
        assert injection_phase is not None
        allowed = {
            "water": {WellRateKind.WRAT, WellRateKind.LRAT, WellRateKind.RESV},
            "oil": {WellRateKind.ORAT, WellRateKind.LRAT, WellRateKind.RESV},
            "gas": {WellRateKind.GRAT, WellRateKind.RESV},
            "fluid": {WellRateKind.RESV},
        }[injection_phase]
        if well.control.kind not in allowed:
            raise FlowContractError(
                "Injector rate kind is incompatible with its injection phase",
                object_name="Well",
                field="control.kind",
                expected=tuple(kind.value for kind in sorted(allowed, key=lambda item: item.value)),
                actual=well.control.kind.value,
            )
    if well.control.kind is not WellRateKind.RESV:
        required = {
            WellRateKind.ORAT: ("oil",),
            WellRateKind.WRAT: ("water",),
            WellRateKind.GRAT: ("gas",),
            WellRateKind.LRAT: tuple(phase for phase in ("oil", "water") if phase in phase_set),
        }[well.control.kind]
        if well.standard_conditions is None or well.standard_densities_kg_m3 is None:
            raise FlowContractError(
                "Surface/standard rate controls require declared standard conditions and densities",
                object_name="Well",
                field="standard_conditions/standard_densities_kg_m3",
                expected=f"explicit metadata for {required}",
                actual=(well.standard_conditions, well.standard_densities_kg_m3),
            )
        missing = tuple(phase for phase in required if phase not in well.standard_densities_kg_m3)
        if missing:
            raise FlowContractError(
                "Surface/standard rate control is missing a standard phase density",
                object_name="Well",
                field="standard_densities_kg_m3",
                expected=required,
                actual=missing,
            )


def _total_mobility(
    cell_idx: int,
    mobilities_pa_s_inv: Mapping[str, torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    total = torch.zeros((), dtype=like.dtype, device=like.device)
    for mobility in mobilities_pa_s_inv.values():
        total = total + mobility[cell_idx]
    return total


def _hydrostatic_pressure_pa(
    perforation: Perforation,
    well: Well,
    mobilities_pa_s_inv: Mapping[str, torch.Tensor],
    densities_kg_m3: Mapping[str, torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    if perforation.depth_offset_m == 0.0:
        return torch.zeros((), dtype=like.dtype, device=like.device)
    cell_idx = perforation.cell_idx
    if well.well_type == "INJ":
        injection_phase = well.injection_phase
        assert injection_phase is not None
        density = densities_kg_m3[injection_phase][cell_idx]
    else:
        total = _total_mobility(cell_idx, mobilities_pa_s_inv, like).clamp_min(EPS)
        density = torch.zeros((), dtype=like.dtype, device=like.device)
        for phase, mobility in mobilities_pa_s_inv.items():
            density = density + mobility[cell_idx] * densities_kg_m3[phase][cell_idx] / total
    return density * _STANDARD_GRAVITY_M_S2 * perforation.depth_offset_m


def _empty_rate_report(like: torch.Tensor) -> WellRateReport:
    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    return WellRateReport(zero, zero.clone(), zero.clone(), zero.clone())


def _validate_rate_report_standard_metadata(
    well: Well,
    phases: tuple[str, ...],
) -> None:
    """Require a complete declared basis for every standard-volume report."""

    standard_phases = tuple(
        phase for phase in ("oil", "water", "gas") if phase in phases
    )
    if not standard_phases:
        return
    if well.standard_conditions is None:
        raise FlowContractError(
            "Surface/standard rate reporting requires declared standard conditions",
            object_name="WellRateReport",
            field=f"{well.name}.standard_conditions",
            expected="WellStandardConditions(pressure_pa, temperature_k)",
            actual=None,
        )
    standards = well.standard_densities_kg_m3 or {}
    missing = tuple(phase for phase in standard_phases if phase not in standards)
    if missing:
        raise FlowContractError(
            "Surface/standard rate reporting requires an explicit standard density",
            object_name="WellRateReport",
            field=f"{well.name}.standard_densities_kg_m3",
            expected=standard_phases,
            actual=missing,
        )


def _report_from_perforation_rates(
    well: Well,
    rates: list[tuple[str, int, torch.Tensor]],
    densities_kg_m3: Mapping[str, torch.Tensor],
    like: torch.Tensor,
) -> WellRateReport:
    report = _empty_rate_report(like)
    oil = report.oil_surface_m3_s
    water = report.water_surface_m3_s
    gas = report.gas_standard_m3_s
    reservoir = report.reservoir_m3_s
    standards = well.standard_densities_kg_m3 or {}
    for phase, cell_idx, reservoir_rate in rates:
        reservoir = reservoir + reservoir_rate
        standard_density = standards.get(phase)
        if standard_density is None:
            continue
        standard_rate = reservoir_rate * densities_kg_m3[phase][cell_idx] / standard_density
        if phase == "oil":
            oil = oil + standard_rate
        elif phase == "water":
            water = water + standard_rate
        elif phase == "gas":
            gas = gas + standard_rate
    return WellRateReport(oil, water, gas, reservoir)


def _sparse_cell_block(
    cells: list[int],
    values: list[torch.Tensor],
    *,
    n_cells: int,
    like: torch.Tensor,
) -> torch.Tensor:
    if not cells:
        indices = torch.empty((1, 0), dtype=torch.int64, device=like.device)
        data = torch.empty((0,), dtype=like.dtype, device=like.device)
    else:
        indices = torch.tensor([cells], dtype=torch.int64, device=like.device)
        data = torch.stack(values)
    return torch.sparse_coo_tensor(
        indices,
        data,
        (n_cells,),
        dtype=like.dtype,
        device=like.device,
    ).coalesce()


class WellGroup(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """Typed wells assembled as sparse, positive-injection SI source blocks.

    Args:
        wells: member :class:`Well` list (may start empty).
        n_cells: reservoir cell count for source-term assembly.
        device / dtype: tensor placement of assembled source terms.
    """

    def __init__(
        self,
        wells: list[Well] | None = None,
        n_cells: int | None = None,
        device: str | torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
    ) -> None:
        super().__init__()
        self.wells: list[Well] = list(wells) if wells else []
        self.n_cells = n_cells
        self.device = torch.device(device)
        self.dtype = dtype

    def add(self, well: Well) -> WellGroup:
        self.wells.append(well)
        return self

    def __len__(self) -> int:
        return len(self.wells)

    def __iter__(self) -> Iterator[Well]:
        return iter(self.wells)

    def compute_source_terms(
        self,
        pressure_pa: torch.Tensor,
        mobilities_pa_s_inv: Mapping[str, torch.Tensor],
        densities_kg_m3: Mapping[str, torch.Tensor],
        *,
        bhp_pa: Mapping[int, torch.Tensor] | None = None,
    ) -> FlowSourceTerms:
        """Aggregate all perforations into sparse phase-mass blocks [kg/s]."""

        if self.n_cells is None:
            raise FlowContractError(
                "WellGroup.n_cells must be set before source assembly",
                object_name="WellGroup",
                field="n_cells",
                expected="positive int",
                actual=None,
            )
        _validate_phase_inputs(
            pressure_pa,
            mobilities_pa_s_inv,
            densities_kg_m3,
            self.n_cells,
        )
        phase_cells: dict[str, list[int]] = {phase: [] for phase in mobilities_pa_s_inv}
        phase_values: dict[str, list[torch.Tensor]] = {
            phase: [] for phase in mobilities_pa_s_inv
        }
        phases = tuple(mobilities_pa_s_inv)
        for well_idx, well in enumerate(self.wells):
            validate_well_control(well, phases)
            for perforation in well.perforations:
                if not 0 <= perforation.cell_idx < self.n_cells:
                    raise FlowContractError(
                        "Perforation cell_idx is outside the model grid",
                        object_name="Perforation",
                        field="cell_idx",
                        expected=f"0..{self.n_cells - 1}",
                        actual=perforation.cell_idx,
                    )
            bhp_value = None if bhp_pa is None else bhp_pa.get(well_idx)
            bhp_operated = bhp_value is not None or isinstance(well.control, BHPControl)
            perforation_rates: list[tuple[str, int, torch.Tensor]] = []
            if bhp_operated:
                if bhp_value is None:
                    assert isinstance(well.control, BHPControl)
                    bhp_value = pressure_pa.new_tensor(well.control.pressure_pa)
                for perforation in well.perforations:
                    cell_idx = perforation.cell_idx
                    head = _hydrostatic_pressure_pa(
                        perforation,
                        well,
                        mobilities_pa_s_inv,
                        densities_kg_m3,
                        pressure_pa,
                    )
                    # At a datum above the perforation (depth offset > 0), the
                    # wellbore pressure at the perforation is BHP + rho*g*dz.
                    # Reservoir-to-well drawdown is therefore p - BHP - rho*g*dz.
                    drawdown = pressure_pa[cell_idx] - bhp_value - head
                    if well.well_type == "INJ":
                        injection_phase = well.injection_phase
                        assert injection_phase is not None
                        reservoir_rate = (
                            perforation.well_index_m3
                            * _total_mobility(cell_idx, mobilities_pa_s_inv, pressure_pa)
                            * (-drawdown).clamp_min(0.0)
                        )
                        perforation_rates.append(
                            (injection_phase, cell_idx, reservoir_rate)
                        )
                    else:
                        producing_drawdown = drawdown.clamp_min(0.0)
                        for phase, mobility in mobilities_pa_s_inv.items():
                            perforation_rates.append(
                                (
                                    phase,
                                    cell_idx,
                                    perforation.well_index_m3
                                    * mobility[cell_idx]
                                    * producing_drawdown,
                                )
                            )
            else:
                assert isinstance(well.control, RateControl)
                for perforation in well.perforations:
                    cell_idx = perforation.cell_idx
                    if well.well_type == "INJ":
                        injection_phase = well.injection_phase
                        assert injection_phase is not None
                        perforation_rates.append(
                            (
                                injection_phase,
                                cell_idx,
                                perforation.well_index_m3
                                * _total_mobility(cell_idx, mobilities_pa_s_inv, pressure_pa),
                            )
                        )
                    else:
                        for phase, mobility in mobilities_pa_s_inv.items():
                            perforation_rates.append(
                                (
                                    phase,
                                    cell_idx,
                                    perforation.well_index_m3 * mobility[cell_idx],
                                )
                            )
                report = _report_from_perforation_rates(
                    well,
                    perforation_rates,
                    densities_kg_m3,
                    pressure_pa,
                )
                denominator = controlled_rate(report, well.control.kind)
                if not bool(torch.isfinite(denominator).all()) or float(denominator.detach()) <= 0.0:
                    raise FlowContractError(
                        "Rate control has zero or non-finite deliverability",
                        object_name="Well",
                        field="control",
                        expected="positive controlled-rate mobility",
                        actual=float(denominator.detach()),
                    )
                scale = well.control.target_m3_s / denominator
                perforation_rates = [
                    (phase, cell_idx, rate * scale)
                    for phase, cell_idx, rate in perforation_rates
                ]

            sign = 1.0 if well.well_type == "INJ" else -1.0
            for phase, cell_idx, reservoir_rate in perforation_rates:
                phase_cells[phase].append(cell_idx)
                phase_values[phase].append(
                    sign * densities_kg_m3[phase][cell_idx] * reservoir_rate
                )

        return FlowSourceTerms(
            phase_mass_kg_s={
                phase: _sparse_cell_block(
                    phase_cells[phase],
                    phase_values[phase],
                    n_cells=self.n_cells,
                    like=pressure_pa,
                )
                for phase in phases
            },
            component_mol_s={},
        )

    def compute_rate_reports(
        self,
        pressure_pa: torch.Tensor,
        mobilities_pa_s_inv: Mapping[str, torch.Tensor],
        densities_kg_m3: Mapping[str, torch.Tensor],
        *,
        bhp_pa: Mapping[int, torch.Tensor] | None = None,
    ) -> tuple[WellRateReport, ...]:
        """Return one positive-magnitude typed SI rate report per well.

        Oil/water/gas surface entries require explicit standard pressure,
        temperature, and phase densities on every well. Incomplete metadata
        fails before any well-rate assembly instead of reporting a false zero.
        ``reservoir_m3_s`` is reconstructed from the SI phase-mass blocks.
        """

        if self.n_cells is None:
            raise FlowContractError(
                "WellGroup.n_cells must be set before rate reporting",
                object_name="WellGroup",
                field="n_cells",
                expected="positive int",
                actual=None,
            )
        _validate_phase_inputs(
            pressure_pa,
            mobilities_pa_s_inv,
            densities_kg_m3,
            self.n_cells,
        )
        phases = tuple(mobilities_pa_s_inv)
        for well in self.wells:
            _validate_rate_report_standard_metadata(well, phases)
        reports: list[WellRateReport] = []
        for well_idx, well in enumerate(self.wells):
            local_bhp = None
            if bhp_pa is not None and well_idx in bhp_pa:
                local_bhp = {0: bhp_pa[well_idx]}
            sources = WellGroup(
                [well],
                n_cells=self.n_cells,
                device=self.device,
                dtype=self.dtype,
            ).compute_source_terms(
                pressure_pa,
                mobilities_pa_s_inv,
                densities_kg_m3,
                bhp_pa=local_bhp,
            )
            direction = 1.0 if well.well_type == "INJ" else -1.0
            report = _empty_rate_report(pressure_pa)
            oil = report.oil_surface_m3_s
            water = report.water_surface_m3_s
            gas = report.gas_standard_m3_s
            reservoir = report.reservoir_m3_s
            standards = well.standard_densities_kg_m3 or {}
            for phase, mass_block in sources.phase_mass_kg_s.items():
                if mass_block.is_sparse:
                    sparse_mass = mass_block.coalesce()
                    cells = sparse_mass.indices()[0]
                    mass_magnitude = direction * sparse_mass.values()
                    phase_density = densities_kg_m3[phase].index_select(0, cells)
                else:
                    mass_magnitude = direction * mass_block
                    phase_density = densities_kg_m3[phase]
                reservoir = reservoir + (mass_magnitude / phase_density).sum()
                standard_density = standards.get(phase)
                if standard_density is None:
                    if phase in {"oil", "water", "gas"} and bool(
                        (mass_magnitude != 0.0).any()
                    ):
                        raise FlowContractError(
                            "Surface rate reporting requires an explicit standard density",
                            object_name="WellRateReport",
                            field=f"{well.name}.standard_densities_kg_m3.{phase}",
                            expected="> 0 kg/m³ at the declared standard conditions",
                            actual=None,
                        )
                    continue
                standard_rate = mass_magnitude.sum() / standard_density
                if phase == "oil":
                    oil = oil + standard_rate
                elif phase == "water":
                    water = water + standard_rate
                elif phase == "gas":
                    gas = gas + standard_rate
            reports.append(WellRateReport(oil, water, gas, reservoir))
        return tuple(reports)


def well_control_residual(
    pressure_pa: torch.Tensor,
    mobilities_pa_s_inv: Mapping[str, torch.Tensor],
    bhp_pa: Mapping[int, torch.Tensor],
    operating_controls: list[BHPControl | RateControl],
    wells: list[Well],
    *,
    bhp_scale: float = 1.0,
    rate_scale: float = 1.0,
    densities_kg_m3: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return one SI BHP/rate closure row per well."""

    phases = tuple(mobilities_pa_s_inv)
    rows: list[torch.Tensor] = []
    for well_idx, (well, control) in enumerate(zip(wells, operating_controls, strict=True)):
        validate_well_control(
            Well(
                name=well.name,
                well_type=well.well_type,
                control=control,
                perforations=well.perforations,
                injection_phase=well.injection_phase,
                injection_composition=well.injection_composition,
                injection_temperature_k=well.injection_temperature_k,
                standard_conditions=well.standard_conditions,
                standard_densities_kg_m3=well.standard_densities_kg_m3,
                bhp_limit_pa=well.bhp_limit_pa,
                rate_limit=well.rate_limit,
                datum_depth_m=well.datum_depth_m,
            ),
            phases,
        )
        bhp_value = bhp_pa[well_idx]
        if isinstance(control, BHPControl):
            rows.append((bhp_value - control.pressure_pa) * bhp_scale)
            continue
        perforation_rates: list[tuple[str, int, torch.Tensor]] = []
        for perforation in well.perforations:
            cell_idx = perforation.cell_idx
            head = _hydrostatic_pressure_pa(
                perforation,
                well,
                mobilities_pa_s_inv,
                densities_kg_m3,
                pressure_pa,
            )
            drawdown = pressure_pa[cell_idx] - bhp_value - head
            if well.well_type == "INJ":
                injection_phase = well.injection_phase
                assert injection_phase is not None
                perforation_rates.append(
                    (
                        injection_phase,
                        cell_idx,
                        perforation.well_index_m3
                        * _total_mobility(cell_idx, mobilities_pa_s_inv, pressure_pa)
                        * (-drawdown).clamp_min(0.0),
                    )
                )
            else:
                producing_drawdown = drawdown.clamp_min(0.0)
                for phase, mobility in mobilities_pa_s_inv.items():
                    perforation_rates.append(
                        (
                            phase,
                            cell_idx,
                            perforation.well_index_m3
                            * mobility[cell_idx]
                            * producing_drawdown,
                        )
                    )
        report = _report_from_perforation_rates(
            well,
            perforation_rates,
            densities_kg_m3,
            pressure_pa,
        )
        rows.append(
            (controlled_rate(report, control.kind) - control.target_m3_s) * rate_scale
        )
    return torch.stack(rows)


__all__ = [
    "BHPControl",
    "FlowSourceTerms",
    "Perforation",
    "RateControl",
    "Well",
    "WellGroup",
    "WellRateKind",
    "WellRateReport",
    "WellStandardConditions",
    "compute_well_index",
    "controlled_rate",
    "source_block",
    "validate_well_control",
    "well_control_residual",
]
