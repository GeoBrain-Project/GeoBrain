"""Implicit typed-SI wells coupled to the SI TPFA reservoir models.

The well equations and all public well controls are canonical SI, and the
TPFA models declare SI residual schemas (state pressure Pa, residual
blocks m³/s of surface volume, dt in seconds). The adapter seam that remains is the CURRENCY conversion
between the wells layer (phase mass, kg/s) and the models (surface-volume
rates, m³/s), a standard-density division, no unit-system crossing.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import torch
import torch.nn as nn

from ....core import GeoBrainError
from ..errors import FlowCapabilityError, FlowContractError
from ..solvers.jacobian import (
    JacobianSparsitySpec,
    compute_coloring,
    compute_sparsity_pattern,
)
from .explicit import (
    BHPControl,
    FlowSourceTerms,
    RateControl,
    Well,
    WellGroup,
    well_control_residual,
)

logger = logging.getLogger(__name__)

_SECONDS_PER_DAY = 86_400.0
_SPARSE_LAYOUTS = {torch.sparse_coo, torch.sparse_csr}

_PressureAdapter = Callable[[torch.Tensor], torch.Tensor]
_PhaseAdapter = Callable[[torch.Tensor], Mapping[str, torch.Tensor]]
_ResidualAdapter = Callable[
    [torch.Tensor, torch.Tensor, float, FlowSourceTerms],
    torch.Tensor,
]
_ModelAdapter = tuple[
    _PressureAdapter,
    _PhaseAdapter,
    _PhaseAdapter,
    _ResidualAdapter,
    int,
]


class _ConnectionLike(Protocol):
    neighbors: np.ndarray


class _GridLike(Protocol):
    device: torch.device
    dtype: torch.dtype

    def _connection_metrics(self) -> _ConnectionLike | None: ...


class _WellModelLike(Protocol):
    n_cells: int
    grid: _GridLike


class _SingleFluidLike(Protocol):
    pvt: object

    def viscosity(self, pressure: torch.Tensor) -> torch.Tensor: ...

    def density(self, pressure: torch.Tensor) -> torch.Tensor: ...


class _SingleModelLike(_WellModelLike, Protocol):
    fluid: _SingleFluidLike

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt_days: float,
        *,
        source_rates: torch.Tensor,
    ) -> torch.Tensor: ...


class _OilWaterFluidLike(Protocol):
    pvt_w: object
    pvt_o: object

    def kr_water(self, saturation: torch.Tensor) -> torch.Tensor: ...

    def kr_oil(self, saturation: torch.Tensor) -> torch.Tensor: ...

    def viscosity_water(self, pressure: torch.Tensor) -> torch.Tensor: ...

    def viscosity_oil(self, pressure: torch.Tensor) -> torch.Tensor: ...

    def density_water(self, pressure: torch.Tensor) -> torch.Tensor: ...

    def density_oil(self, pressure: torch.Tensor) -> torch.Tensor: ...


class _OilWaterModelLike(_WellModelLike, Protocol):
    fluid: _OilWaterFluidLike

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt_days: float,
        *,
        source_water_rates: torch.Tensor,
        source_oil_rates: torch.Tensor,
    ) -> torch.Tensor: ...


class _PVTLike(Protocol):
    def viscosity(self, pressure: torch.Tensor) -> torch.Tensor: ...

    def density(self, pressure: torch.Tensor) -> torch.Tensor: ...


class _ThreePhaseRelPermLike(Protocol):
    def kr_water(self, saturation: torch.Tensor) -> torch.Tensor: ...

    def kr_oil(
        self,
        water_saturation: torch.Tensor,
        gas_saturation: torch.Tensor,
    ) -> torch.Tensor: ...

    def kr_gas(self, saturation: torch.Tensor) -> torch.Tensor: ...


class _BlackOilFluidLike(Protocol):
    pvt_w: _PVTLike
    pvt_o: _PVTLike
    pvt_g: _PVTLike
    relperm: _ThreePhaseRelPermLike


class _BlackOilModelLike(_WellModelLike, Protocol):
    fluid: _BlackOilFluidLike

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt_days: float,
        *,
        source_water_rates: torch.Tensor,
        source_oil_rates: torch.Tensor,
        source_gas_rates: torch.Tensor,
    ) -> torch.Tensor: ...


def _scatter_sparse_vector(block: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    """Materialise a sparse source vector by differentiable indexed scatter.

    Reservoir kernels consume a dense residual vector, but a well source has
    one entry per perforated cell.  Indexed scatter preserves that O(cells)
    vector contract without routing the sparse well path through
    :meth:`torch.Tensor.to_dense`.
    """

    if block.layout == torch.strided:
        return block
    if block.layout != torch.sparse_coo:
        raise FlowCapabilityError(
            "Well source adaptation supports strided or sparse COO vectors",
            object_name="WellSystem",
            field="source.layout",
            expected=(str(torch.strided), str(torch.sparse_coo)),
            actual=str(block.layout),
        )
    source = block.coalesce()
    return torch.zeros_like(like).index_add(
        0,
        source.indices()[0],
        source.values(),
    )


def _reference_density(pvt: object, *, phase: str) -> float:
    """Read an explicit standard/reference density from a PVT object."""

    candidates = {
        "oil": ("surface_oil_density_kg_m3", "density_ref_kg_m3"),
        "gas": ("surface_gas_density_kg_m3", "density_ref_kg_m3"),
        "water": ("density_ref_kg_m3",),
        "fluid": ("density_ref_kg_m3",),
    }.get(phase, ("density_ref_kg_m3",))
    for name in candidates:
        value = getattr(pvt, name, None)
        if value is not None:
            scalar = float(value)
            if scalar > 0.0:
                return scalar
    raise FlowCapabilityError(
        "TPFA source adaptation requires an explicit standard density",
        object_name="WellSystem",
        field=f"{phase}.standard_density",
        expected=f"one of {candidates}",
        actual=type(pvt).__name__,
    )


def _volume_rate_from_mass(
    sources: FlowSourceTerms,
    *,
    phase: str,
    standard_density_kg_m3: float,
    gas: bool,
    like: torch.Tensor,
) -> torch.Tensor:
    block = sources.phase_mass_kg_s.get(phase)
    if block is None:
        return torch.zeros_like(like)
    dense = _scatter_sparse_vector(block, like=like)
    return dense / standard_density_kg_m3


def _mass_from_volume_residual(
    residual: torch.Tensor,
    *,
    standard_density_kg_m3: float,
    gas: bool,
) -> torch.Tensor:
    """Convert one surface-volume residual block [m³/s] to canonical kg/s."""

    return residual * standard_density_kg_m3


def _require_field_schema(model: object, expected_units: tuple[str, ...]) -> None:
    schema = getattr(model, "schema", None)
    actual_units = tuple(
        getattr(block, "unit", None)
        for block in getattr(schema, "residual_blocks", ())
    )
    if getattr(schema, "unit_system", None) != "SI" or actual_units != expected_units:
        raise FlowCapabilityError(
            "WellSystem adapter requires the exact declared SI residual basis",
            object_name="WellSystem",
            field="model.schema",
            expected={"unit_system": "SI", "residual_units": expected_units},
            actual={
                "unit_system": getattr(schema, "unit_system", None),
                "residual_units": actual_units,
            },
            hint="use a supported SI TPFA model or provide a complete adapter",
        )


def _single_phase_adapter(model: _SingleModelLike) -> _ModelAdapter:
    n = int(model.n_cells)
    _require_field_schema(model, ("m³/s",))
    standard_density = _reference_density(model.fluid.pvt, phase="water")

    def pressure(x_res: torch.Tensor) -> torch.Tensor:
        return x_res[:n]

    def mobilities(x_res: torch.Tensor) -> Mapping[str, torch.Tensor]:
        pressure_pa = pressure(x_res)
        return {"water": 1.0 / model.fluid.viscosity(pressure_pa)}

    def densities(x_res: torch.Tensor) -> Mapping[str, torch.Tensor]:
        pressure_pa = pressure(x_res)
        return {"water": model.fluid.density(pressure_pa)}

    def residual(
        x_res: torch.Tensor,
        x_res_old: torch.Tensor,
        dt_s: float,
        sources: FlowSourceTerms,
    ) -> torch.Tensor:
        si_residual = model.residual(
            x_res,
            x_res_old,
            dt_s,
            source_rates=_volume_rate_from_mass(
                sources,
                phase="water",
                standard_density_kg_m3=standard_density,
                gas=False,
                like=x_res[:n],
            ),
        )
        return _mass_from_volume_residual(
            si_residual,
            standard_density_kg_m3=standard_density,
            gas=False,
        )

    return pressure, mobilities, densities, residual, 1


def _oilwater_adapter(model: _OilWaterModelLike) -> _ModelAdapter:
    n = int(model.n_cells)
    _require_field_schema(model, ("m³/s", "m³/s"))
    rho_sc = {
        "water": _reference_density(model.fluid.pvt_w, phase="water"),
        "oil": _reference_density(model.fluid.pvt_o, phase="oil"),
    }

    def pressure(x_res: torch.Tensor) -> torch.Tensor:
        return x_res[:n]

    def mobilities(x_res: torch.Tensor) -> Mapping[str, torch.Tensor]:
        p = pressure(x_res)
        sw = x_res[n : 2 * n].clamp(min=1e-6, max=1.0 - 1e-6)
        fluid = model.fluid
        return {
            "water": fluid.kr_water(sw) / fluid.viscosity_water(p),
            "oil": fluid.kr_oil(sw) / fluid.viscosity_oil(p),
        }

    def densities(x_res: torch.Tensor) -> Mapping[str, torch.Tensor]:
        p = pressure(x_res)
        return {
            "water": model.fluid.density_water(p),
            "oil": model.fluid.density_oil(p),
        }

    def residual(
        x_res: torch.Tensor,
        x_res_old: torch.Tensor,
        dt_s: float,
        sources: FlowSourceTerms,
    ) -> torch.Tensor:
        si_residual = model.residual(
            x_res,
            x_res_old,
            dt_s,
            source_water_rates=_volume_rate_from_mass(
                sources,
                phase="water",
                standard_density_kg_m3=rho_sc["water"],
                gas=False,
                like=x_res[:n],
            ),
            source_oil_rates=_volume_rate_from_mass(
                sources,
                phase="oil",
                standard_density_kg_m3=rho_sc["oil"],
                gas=False,
                like=x_res[:n],
            ),
        )
        return torch.cat(
            (
                _mass_from_volume_residual(
                    si_residual[:n],
                    standard_density_kg_m3=rho_sc["water"],
                    gas=False,
                ),
                _mass_from_volume_residual(
                    si_residual[n:],
                    standard_density_kg_m3=rho_sc["oil"],
                    gas=False,
                ),
            )
        )

    return pressure, mobilities, densities, residual, 2


def _blackoil_adapter(model: _BlackOilModelLike) -> _ModelAdapter:
    n = int(model.n_cells)
    _require_field_schema(model, ("m³/s", "m³/s", "m³/s"))
    fluid = model.fluid
    rho_sc = {
        "water": _reference_density(fluid.pvt_w, phase="water"),
        "oil": _reference_density(fluid.pvt_o, phase="oil"),
        "gas": _reference_density(fluid.pvt_g, phase="gas"),
    }

    def pressure(x_res: torch.Tensor) -> torch.Tensor:
        return x_res[:n]

    def mobilities(x_res: torch.Tensor) -> Mapping[str, torch.Tensor]:
        p = pressure(x_res)
        sw = x_res[n : 2 * n].clamp(min=1e-6, max=1.0 - 1e-6)
        sg = x_res[2 * n : 3 * n].clamp(min=1e-6, max=1.0 - 1e-6)
        return {
            "water": fluid.relperm.kr_water(sw) / fluid.pvt_w.viscosity(p),
            "oil": fluid.relperm.kr_oil(sw, sg) / fluid.pvt_o.viscosity(p),
            "gas": fluid.relperm.kr_gas(sg) / fluid.pvt_g.viscosity(p),
        }

    def densities(x_res: torch.Tensor) -> Mapping[str, torch.Tensor]:
        p = pressure(x_res)
        return {
            "water": fluid.pvt_w.density(p),
            "oil": fluid.pvt_o.density(p),
            "gas": fluid.pvt_g.density(p),
        }

    def residual(
        x_res: torch.Tensor,
        x_res_old: torch.Tensor,
        dt_s: float,
        sources: FlowSourceTerms,
    ) -> torch.Tensor:
        si_residual = model.residual(
            x_res,
            x_res_old,
            dt_s,
            source_water_rates=_volume_rate_from_mass(
                sources,
                phase="water",
                standard_density_kg_m3=rho_sc["water"],
                gas=False,
                like=x_res[:n],
            ),
            source_oil_rates=_volume_rate_from_mass(
                sources,
                phase="oil",
                standard_density_kg_m3=rho_sc["oil"],
                gas=False,
                like=x_res[:n],
            ),
            source_gas_rates=_volume_rate_from_mass(
                sources,
                phase="gas",
                standard_density_kg_m3=rho_sc["gas"],
                gas=True,
                like=x_res[:n],
            ),
        )
        return torch.cat(
            (
                _mass_from_volume_residual(
                    si_residual[:n],
                    standard_density_kg_m3=rho_sc["water"],
                    gas=False,
                ),
                _mass_from_volume_residual(
                    si_residual[n : 2 * n],
                    standard_density_kg_m3=rho_sc["oil"],
                    gas=False,
                ),
                _mass_from_volume_residual(
                    si_residual[2 * n :],
                    standard_density_kg_m3=rho_sc["gas"],
                    gas=True,
                ),
            )
        )

    return pressure, mobilities, densities, residual, 3


def _auto_adapter(model: _WellModelLike) -> _ModelAdapter:
    cls = type(model).__name__
    if cls == "OilWaterModel":
        return _oilwater_adapter(cast(_OilWaterModelLike, model))
    if cls in {"BlackOilModel", "BlackOilVarSwitchModel"}:
        return _blackoil_adapter(cast(_BlackOilModelLike, model))
    raise GeoBrainError(
        f"WellSystem has no built-in adapter for model '{cls}'",
        object_name="WellSystem",
        field="model",
        expected="OilWaterModel, BlackOilModel, BlackOilVarSwitchModel, or explicit adapters",
        actual=cls,
    )


def _as_coalesced_coo(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.layout == torch.sparse_coo:
        return matrix.coalesce()
    if matrix.layout == torch.sparse_csr:
        return matrix.to_sparse_coo().coalesce()
    raise FlowContractError(
        "Bordered Jacobian blocks must be sparse COO or CSR tensors",
        object_name="SparseBorderedJacobian",
        field="block.layout",
        expected=tuple(str(layout) for layout in _SPARSE_LAYOUTS),
        actual=str(matrix.layout),
    )


def _sparse_submatrix(
    matrix: torch.Tensor,
    *,
    row_start: int,
    row_stop: int,
    column_start: int,
    column_stop: int,
) -> torch.Tensor:
    """Slice a sparse matrix by filtering and rebasing COO coordinates."""

    source = _as_coalesced_coo(matrix)
    source_indices = source.indices()
    mask = (
        (source_indices[0] >= row_start)
        & (source_indices[0] < row_stop)
        & (source_indices[1] >= column_start)
        & (source_indices[1] < column_stop)
    )
    indices = source_indices[:, mask].clone()
    indices[0] -= row_start
    indices[1] -= column_start
    values = source.values()[mask]
    nonzero = values != 0.0
    return torch.sparse_coo_tensor(
        indices[:, nonzero],
        values[nonzero],
        (row_stop - row_start, column_stop - column_start),
        dtype=source.dtype,
        device=source.device,
    ).coalesce()


def _compute_sparse_vjp_jacobian(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    spec: JacobianSparsitySpec,
) -> torch.Tensor:
    """Differentiate structural rows by reverse-mode coloring.

    Sparse COO construction currently drops forward-mode tangents in PyTorch,
    while its reverse-mode derivative through ``values`` is supported.  Row
    coloring groups residual rows that share no structural column, so each VJP
    recovers one exact derivative per stored coordinate.  The largest dense
    temporary is a single state-sized VJP; no square dense Jacobian is formed.
    """

    row_colors, color_count = compute_coloring(
        spec.cols,
        spec.rows,
        spec.n_dof,
    )
    entry_colors = row_colors[spec.rows]
    rows = torch.from_numpy(spec.rows).to(device=state.device)
    columns = torch.from_numpy(spec.cols).to(device=state.device)
    values = torch.empty(
        spec.rows.size,
        dtype=state.dtype,
        device=state.device,
    )
    with torch.enable_grad():
        differentiated_state = state.detach().requires_grad_(True)
        residual = residual_fn(differentiated_state)
        for color in range(color_count):
            entry_mask_np = entry_colors == color
            if not bool(entry_mask_np.any()):
                continue
            entry_indices_np = np.flatnonzero(entry_mask_np)
            entry_indices = torch.from_numpy(entry_indices_np).to(device=state.device)
            probe_rows_np = np.unique(spec.rows[entry_mask_np])
            probe_rows = torch.from_numpy(probe_rows_np).to(device=state.device)
            probe = torch.zeros_like(residual)
            probe.index_fill_(0, probe_rows, 1.0)
            gradient = torch.autograd.grad(
                residual,
                differentiated_state,
                grad_outputs=probe,
                retain_graph=color + 1 < color_count,
                create_graph=False,
            )[0]
            values[entry_indices] = gradient[columns[entry_indices]].detach()
    return torch.sparse_coo_tensor(
        torch.stack((rows, columns)),
        values,
        (spec.n_dof, spec.n_dof),
        dtype=state.dtype,
        device=state.device,
    ).coalesce()


@dataclass(frozen=True, slots=True)
class SparseBorderedJacobian:
    """Four sparse blocks of a reservoir/well bordered Jacobian.

    The canonical augmented ordering is ``[reservoir dofs, well BHPs]``.
    :meth:`assemble` shifts COO coordinates and concatenates values directly;
    the sparse execution path never creates an augmented dense matrix.
    """

    reservoir: torch.Tensor
    reservoir_to_well: torch.Tensor
    well_to_reservoir: torch.Tensor
    controls: torch.Tensor

    def __post_init__(self) -> None:
        blocks = (
            self.reservoir,
            self.reservoir_to_well,
            self.well_to_reservoir,
            self.controls,
        )
        if any(block.layout not in _SPARSE_LAYOUTS for block in blocks):
            raise FlowContractError(
                "Bordered Jacobian blocks must be sparse COO or CSR tensors",
                object_name="SparseBorderedJacobian",
                field="blocks.layout",
                expected=tuple(str(layout) for layout in _SPARSE_LAYOUTS),
                actual=tuple(str(block.layout) for block in blocks),
            )
        reservoir_dofs = self.reservoir.shape[0]
        wells = self.controls.shape[0]
        expected_shapes = (
            (reservoir_dofs, reservoir_dofs),
            (reservoir_dofs, wells),
            (wells, reservoir_dofs),
            (wells, wells),
        )
        if tuple(block.shape for block in blocks) != expected_shapes:
            raise FlowContractError(
                "Bordered Jacobian block shapes are inconsistent",
                object_name="SparseBorderedJacobian",
                field="blocks.shape",
                expected=expected_shapes,
                actual=tuple(tuple(block.shape) for block in blocks),
            )
        reference = self.reservoir
        if any(
            block.dtype != reference.dtype or block.device != reference.device
            for block in blocks[1:]
        ):
            raise FlowContractError(
                "Bordered Jacobian blocks must share dtype and device",
                object_name="SparseBorderedJacobian",
                field="blocks",
                expected=(str(reference.dtype), str(reference.device)),
                actual=tuple((str(block.dtype), str(block.device)) for block in blocks),
            )

    @property
    def nnz(self) -> int:
        """Stored nonzeros across all four blocks."""

        return sum(_as_coalesced_coo(block)._nnz() for block in self._blocks())

    def _blocks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.reservoir,
            self.reservoir_to_well,
            self.well_to_reservoir,
            self.controls,
        )

    def assemble(self) -> torch.Tensor:
        """Assemble the augmented sparse COO matrix without densification."""

        reservoir, reservoir_to_well, well_to_reservoir, controls = (
            _as_coalesced_coo(block) for block in self._blocks()
        )
        reservoir_dofs = reservoir.shape[0]
        wells = controls.shape[0]
        offsets = (
            (0, 0),
            (0, reservoir_dofs),
            (reservoir_dofs, 0),
            (reservoir_dofs, reservoir_dofs),
        )
        indices: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for block, (row_offset, column_offset) in zip(
            (reservoir, reservoir_to_well, well_to_reservoir, controls),
            offsets,
            strict=True,
        ):
            offset = torch.tensor(
                [[row_offset], [column_offset]],
                dtype=torch.int64,
                device=block.device,
            )
            indices.append(block.indices() + offset)
            values.append(block.values())
        return torch.sparse_coo_tensor(
            torch.cat(indices, dim=1),
            torch.cat(values),
            (reservoir_dofs + wells, reservoir_dofs + wells),
            dtype=reservoir.dtype,
            device=reservoir.device,
        ).coalesce()

    def to_dense_compatibility(self) -> torch.Tensor:
        """Explicit dense-only adapter for small legacy callers and tests."""

        return self.assemble().to_dense()


class WellSystem(nn.Module):  # type: ignore[misc]  # skipped torch import boundary
    """One implicit BHP degree of freedom per typed-SI well."""

    def __init__(
        self,
        model: _WellModelLike,
        well_group: WellGroup,
        *,
        bhp_scale: float = 1.0,
        rate_scale: float = 1.0,
        bhp_init_pa: torch.Tensor | None = None,
        default_producer_bhp_floor_pa: float | None = None,
        mobility_fn: _PhaseAdapter | None = None,
        density_fn: _PhaseAdapter | None = None,
        residual_fn: _ResidualAdapter | None = None,
        n_prim: int | None = None,
        jacobian_sparsity: JacobianSparsitySpec | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.well_group = well_group
        self.wells = list(well_group.wells)
        self.W = len(self.wells)
        if self.W == 0:
            raise FlowContractError(
                "WellSystem requires at least one well",
                object_name="WellSystem",
                field="well_group",
                expected="non-empty",
                actual=0,
            )
        self.n_cells = int(model.n_cells)
        if well_group.n_cells != self.n_cells:
            raise FlowContractError(
                "WellGroup cell count must match its reservoir model",
                object_name="WellSystem",
                field="well_group.n_cells",
                expected=self.n_cells,
                actual=well_group.n_cells,
            )
        self.bhp_scale = float(bhp_scale)
        self.rate_scale = float(rate_scale)
        self.default_producer_bhp_floor_pa = default_producer_bhp_floor_pa
        explicit_adapters = all(
            item is not None for item in (mobility_fn, density_fn, residual_fn, n_prim)
        )
        if explicit_adapters:
            assert mobility_fn is not None
            assert density_fn is not None
            assert residual_fn is not None
            assert n_prim is not None
            if not isinstance(jacobian_sparsity, JacobianSparsitySpec):
                raise FlowCapabilityError(
                    "Custom WellSystem residual adapters require an explicit "
                    "JacobianSparsitySpec",
                    object_name="WellSystem",
                    field="jacobian_sparsity",
                    expected=JacobianSparsitySpec,
                    actual=type(jacobian_sparsity),
                )
            self._pressure_pa: _PressureAdapter = lambda state: state[: self.n_cells]
            self._mobilities: _PhaseAdapter = mobility_fn
            self._densities: _PhaseAdapter = density_fn
            self._residual_adapter: _ResidualAdapter = residual_fn
            self.n_prim = int(n_prim)
        elif any(item is not None for item in (mobility_fn, density_fn, residual_fn, n_prim)):
            raise FlowContractError(
                "Explicit WellSystem adapters must be supplied together",
                object_name="WellSystem",
                field="mobility_fn/density_fn/residual_fn/n_prim",
                expected="all or none",
                actual="partial",
            )
        else:
            if jacobian_sparsity is not None:
                raise FlowContractError(
                    "Jacobian sparsity is only accepted with explicit WellSystem adapters",
                    object_name="WellSystem",
                    field="jacobian_sparsity",
                    expected="None for built-in model adapters",
                    actual=type(jacobian_sparsity),
                )
            (
                self._pressure_pa,
                self._mobilities,
                self._densities,
                self._residual_adapter,
                self.n_prim,
            ) = _auto_adapter(model)
        self.res_size = self.n_prim * self.n_cells
        self.device = model.grid.device
        self.dtype = model.grid.dtype
        if bhp_init_pa is not None and (
            bhp_init_pa.shape != (self.W,)
            or bhp_init_pa.dtype != self.dtype
            or bhp_init_pa.device != self.device
        ):
            raise FlowContractError(
                "Initial BHP tensor must match the model",
                object_name="WellSystem",
                field="bhp_init_pa",
                expected=((self.W,), str(self.dtype), str(self.device)),
                actual=(tuple(bhp_init_pa.shape), str(bhp_init_pa.dtype), str(bhp_init_pa.device)),
            )
        self._bhp_init_pa = bhp_init_pa
        self.operating_controls: list[BHPControl | RateControl] = [
            well.control for well in self.wells
        ]
        if jacobian_sparsity is None:
            self._jacobian_sparsity = self._build_augmented_sparsity()
        else:
            expected_dofs = self.res_size + self.W
            if jacobian_sparsity.n_dof != expected_dofs:
                raise FlowContractError(
                    "Custom WellSystem sparsity size must match the augmented state",
                    object_name="WellSystem",
                    field="jacobian_sparsity.n_dof",
                    expected=expected_dofs,
                    actual=jacobian_sparsity.n_dof,
                )
            self._jacobian_sparsity = jacobian_sparsity

    def _build_augmented_sparsity(self) -> JacobianSparsitySpec:
        """Build the fixed TPFA/perforation sparsity of the augmented system."""

        connection = self.model.grid._connection_metrics()
        if connection is None:
            raise FlowCapabilityError(
                "Sparse WellSystem Jacobians require declared grid connections",
                object_name="WellSystem",
                field="model.grid.connections",
                expected="cell-neighbor topology",
                actual=None,
            )
        reservoir_rows, reservoir_columns = compute_sparsity_pattern(
            connection.neighbors,
            self.n_cells,
            self.n_prim,
        )
        rows = [reservoir_rows]
        columns = [reservoir_columns]

        reservoir_to_well: set[tuple[int, int]] = set()
        well_to_reservoir: set[tuple[int, int]] = set()
        for well_index, well in enumerate(self.wells):
            for perforation in well.perforations:
                cell = perforation.cell_idx
                for variable in range(self.n_prim):
                    reservoir_dof = variable * self.n_cells + cell
                    reservoir_to_well.add((reservoir_dof, self.res_size + well_index))
                    well_to_reservoir.add((self.res_size + well_index, reservoir_dof))
        border_pairs = sorted(reservoir_to_well | well_to_reservoir)
        if border_pairs:
            border = np.asarray(border_pairs, dtype=np.int64)
            rows.append(border[:, 0])
            columns.append(border[:, 1])

        control_indices: np.ndarray = np.arange(
            self.res_size,
            self.res_size + self.W,
            dtype=np.int64,
        )
        rows.append(control_indices)
        columns.append(control_indices)
        all_rows = np.concatenate(rows)
        all_columns = np.concatenate(columns)
        n_dof = self.res_size + self.W
        colors, n_colors = compute_coloring(all_rows, all_columns, n_dof)
        return JacobianSparsitySpec(
            n_dof=n_dof,
            rows=all_rows,
            cols=all_columns,
            colors=colors,
            n_colors=n_colors,
        )

    def _bhp_limit_pa(self, well: Well) -> float | None:
        if well.bhp_limit_pa is not None:
            return float(well.bhp_limit_pa)
        if (
            well.well_type == "PROD"
            and isinstance(well.control, RateControl)
            and self.default_producer_bhp_floor_pa is not None
        ):
            return self.default_producer_bhp_floor_pa
        return None

    def initial_bhp(self, x_res: torch.Tensor | None = None) -> torch.Tensor:
        if self._bhp_init_pa is not None:
            return self._bhp_init_pa.clone()
        pressure = None if x_res is None else self._pressure_pa(x_res)
        values: list[torch.Tensor] = []
        for well in self.wells:
            if isinstance(well.control, BHPControl):
                values.append(
                    torch.tensor(well.control.pressure_pa, dtype=self.dtype, device=self.device)
                )
            elif pressure is not None:
                values.append(pressure[well.perforations[0].cell_idx])
            elif well.bhp_limit_pa is not None:
                values.append(torch.tensor(well.bhp_limit_pa, dtype=self.dtype, device=self.device))
            else:
                raise FlowContractError(
                    "Rate-controlled wells need reservoir state, explicit BHP initialisation, or limit",
                    object_name="WellSystem",
                    field=f"{well.name}.bhp_init_pa",
                    expected="x_res, bhp_init_pa, or bhp_limit_pa",
                    actual=None,
                )
        return torch.stack(values)

    def augment(self, x_res: torch.Tensor, bhp_pa: torch.Tensor | None = None) -> torch.Tensor:
        return torch.cat([x_res, self.initial_bhp(x_res) if bhp_pa is None else bhp_pa])

    def split(self, x_aug: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x_aug[: self.res_size], x_aug[self.res_size :]

    def source_terms(self, x_aug: torch.Tensor) -> FlowSourceTerms:
        x_res, bhp_pa = self.split(x_aug)
        pressure_pa = self._pressure_pa(x_res)
        return self.well_group.compute_source_terms(
            pressure_pa,
            self._mobilities(x_res),
            self._densities(x_res),
            bhp_pa={index: bhp_pa[index] for index in range(self.W)},
        )

    def residual(self, x_aug: torch.Tensor, x_aug_old: torch.Tensor, dt_s: float) -> torch.Tensor:
        x_res, bhp_pa = self.split(x_aug)
        x_res_old, _ = self.split(x_aug_old)
        pressure_pa = self._pressure_pa(x_res)
        mobilities = self._mobilities(x_res)
        densities = self._densities(x_res)
        bhp_map = {index: bhp_pa[index] for index in range(self.W)}
        sources = self.well_group.compute_source_terms(
            pressure_pa,
            mobilities,
            densities,
            bhp_pa=bhp_map,
        )
        reservoir = self._residual_adapter(x_res, x_res_old, dt_s, sources)
        control = well_control_residual(
            pressure_pa,
            mobilities,
            bhp_map,
            self.operating_controls,
            self.wells,
            bhp_scale=self.bhp_scale,
            rate_scale=self.rate_scale,
            densities_kg_m3=densities,
        )
        return torch.cat([reservoir, control])

    def jacobian_blocks(
        self,
        x_aug: torch.Tensor,
        x_aug_old: torch.Tensor,
        dt_s: float,
    ) -> SparseBorderedJacobian:
        augmented = _compute_sparse_vjp_jacobian(
            lambda value: self.residual(value, x_aug_old, dt_s),
            x_aug,
            self._jacobian_sparsity,
        )
        return SparseBorderedJacobian(
            reservoir=_sparse_submatrix(
                augmented,
                row_start=0,
                row_stop=self.res_size,
                column_start=0,
                column_stop=self.res_size,
            ),
            reservoir_to_well=_sparse_submatrix(
                augmented,
                row_start=0,
                row_stop=self.res_size,
                column_start=self.res_size,
                column_stop=self.res_size + self.W,
            ),
            well_to_reservoir=_sparse_submatrix(
                augmented,
                row_start=self.res_size,
                row_stop=self.res_size + self.W,
                column_start=0,
                column_stop=self.res_size,
            ),
            controls=_sparse_submatrix(
                augmented,
                row_start=self.res_size,
                row_stop=self.res_size + self.W,
                column_start=self.res_size,
                column_stop=self.res_size + self.W,
            ),
        )

    def jacobian(self, x_aug: torch.Tensor, x_aug_old: torch.Tensor, dt_s: float) -> torch.Tensor:
        """Return the canonical sparse augmented reservoir/well Jacobian."""

        return self.jacobian_blocks(x_aug, x_aug_old, dt_s).assemble()

    def jacobian_dense_compatibility(
        self,
        x_aug: torch.Tensor,
        x_aug_old: torch.Tensor,
        dt_s: float,
    ) -> torch.Tensor:
        """Explicit dense-only adapter for small legacy callers."""

        return self.jacobian_blocks(
            x_aug,
            x_aug_old,
            dt_s,
        ).to_dense_compatibility()

    def residual_fn(
        self,
        x_aug_old: torch.Tensor,
        dt_s: float,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x_aug: self.residual(x_aug, x_aug_old, dt_s)

    def jacobian_fn(
        self,
        x_aug_old: torch.Tensor,
        dt_s: float,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda x_aug: self.jacobian(x_aug, x_aug_old, dt_s)

    def well_bhp(self, x_aug: torch.Tensor) -> torch.Tensor:
        return self.split(x_aug)[1].detach()

    def check_limits(self, x_aug: torch.Tensor) -> bool:
        bhp_pa = self.well_bhp(x_aug)
        switched = False
        for index, well in enumerate(self.wells):
            if not isinstance(self.operating_controls[index], RateControl):
                continue
            limit = self._bhp_limit_pa(well)
            if limit is None:
                continue
            solved = float(bhp_pa[index])
            violated = (well.well_type == "PROD" and solved < limit) or (
                well.well_type == "INJ" and solved > limit
            )
            if violated:
                self.operating_controls[index] = BHPControl(limit)
                switched = True
                logger.info("well %s switched to BHP control at %.6g Pa", well.name, limit)
        return switched

    def reset_controls(self) -> None:
        self.operating_controls = [well.control for well in self.wells]

    def well_rates(self, x_aug: torch.Tensor) -> FlowSourceTerms:
        """Return sparse SI phase-mass source blocks; no ambiguous total exists."""

        return self.source_terms(x_aug)


__all__ = ["SparseBorderedJacobian", "WellSystem"]
