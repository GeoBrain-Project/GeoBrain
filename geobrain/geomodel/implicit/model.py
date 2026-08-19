"""
ImplicitModel: multi-series implicit geological model with fault support.

Orchestrates Cokriging interpolation across multiple geological series with
ERODE/ONLAP stacking and simple fault displacement. Fully differentiable
via PyTorch autograd.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import math

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from ..capabilities import GeomodelCapabilityReport, GeomodelUnsupportedCombination
from ..errors import GeomodelContractError, GeomodelResourceError
from ..resources import GeomodelResourceEstimate

from .data import (
    ImplicitModelConfig,
    SeriesDefinition,
    FaultDefinition,
    InterpolationInput,
    StackRelation,
)
from .kernel import CubicKernel, GaussianKernel, CovarianceKernel
from .interpolator import CokrigingInterpolator


class ImplicitModel(nn.Module):
    """
    Differentiable implicit geological model.

    Builds a 3D (or 2D) geological block model from surface contact points
    and orientation measurements using Universal Cokriging. Supports:

    - Multiple geological series with ERODE / ONLAP stacking
    - Simple fault displacement (differentiable tanh-based sign function)
    - End-to-end differentiability for gradient-based optimization

    Args:
        config: Model configuration (extent, resolution, kernel, etc.).
        series: List of geological series definitions (oldest first).
        faults: Optional list of fault definitions.

    Example:
        >>> model = ImplicitModel(config, series=[series1, series2])
        >>> result = model(soft=True)
        >>> block = result['block']  # (nx*ny[*nz],) soft lithology ids
    """

    def __init__(
        self,
        config: ImplicitModelConfig,
        series: List[SeriesDefinition],
        faults: Optional[List[FaultDefinition]] = None,
    ):
        super().__init__()
        if not isinstance(config, ImplicitModelConfig):
            raise GeomodelContractError(
                "ImplicitModel requires ImplicitModelConfig",
                object_name=type(self).__name__, field="config",
                expected="ImplicitModelConfig", actual=type(config).__name__,
            )
        self.config = config
        self.series = tuple(series)
        self.faults = tuple(faults or ())
        if not self.series or any(not isinstance(item, SeriesDefinition) for item in self.series):
            raise GeomodelContractError(
                "ImplicitModel requires at least one SeriesDefinition",
                object_name=type(self).__name__, field="series",
                expected="non-empty sequence of SeriesDefinition", actual=type(series).__name__,
            )
        self._validate_tensor_contracts()
        estimate = self.estimate_resources(autograd=True)
        if config.budget_bytes is not None and estimate.total_bytes > config.budget_bytes:
            raise GeomodelResourceError(
                "implicit-model resource estimate exceeds the configured budget",
                object_name=type(self).__name__, field="budget_bytes",
                expected=f">= {estimate.total_bytes}", actual=config.budget_bytes,
            )

        # Build kernel
        self.kernel = self._build_kernel(config)

        # Build interpolator
        self.interpolator = CokrigingInterpolator(
            kernel=self.kernel,
            drift_degree=config.drift_degree,
        )

        # Fault displacements as learnable parameters
        if self.faults:
            self.fault_displacements = nn.ParameterList([
                nn.Parameter(torch.tensor(
                    f.displacement,
                    dtype=config.torch_dtype,
                    device=config.torch_device,
                ))
                for f in self.faults
            ])
        else:
            self.fault_displacements = nn.ParameterList()

    @staticmethod
    def _build_kernel(config: ImplicitModelConfig) -> CovarianceKernel:
        """Instantiate the covariance kernel from config."""
        c_o = config.computed_c_o
        a = config.range
        if config.anisotropy is None:
            aniso = None
        elif isinstance(config.anisotropy, torch.Tensor):
            if config.anisotropy.dtype != config.torch_dtype or config.anisotropy.device != config.torch_device:
                raise GeomodelContractError(
                    "anisotropy tensor must already match configuration dtype/device",
                    object_name="ImplicitModel", field="config.anisotropy",
                    expected=(config.torch_dtype, config.torch_device),
                    actual=(config.anisotropy.dtype, config.anisotropy.device),
                )
            aniso = config.anisotropy
        else:
            aniso = torch.as_tensor(
                config.anisotropy,
                dtype=config.torch_dtype,
                device=config.torch_device,
            )
        if config.kernel == "cubic":
            return CubicKernel(a=a, c_o=c_o, aniso=aniso)
        elif config.kernel == "gaussian":
            return GaussianKernel(a=a, c_o=c_o, aniso=aniso)
        else:
            raise GeomodelContractError(
                "implicit covariance kernel is unsupported",
                object_name="ImplicitModel", field="config.kernel",
                expected="cubic or gaussian", actual=config.kernel,
            )

    # ------------------------------------------------------------------
    # Fault processing
    # ------------------------------------------------------------------

    def _interpolate_fault(
        self,
        fault: FaultDefinition,
        displacement_param: nn.Parameter,
        grid: torch.Tensor,
    ) -> dict:
        """
        Interpolate a single fault and compute displacement field.

        Returns a dict with fault scalar field, sign field, displacement vector,
        and the fault normal direction.
        """
        inp = InterpolationInput.from_series(
            SeriesDefinition(
                name=fault.name,
                surface_points=fault.surface_points,
                orientations=fault.orientations,
                relation=StackRelation.FAULT,
            )
        )
        # Solve and evaluate
        weights = self.interpolator.solve(inp)
        Z_fault = self.interpolator.evaluate(grid, inp, weights)

        # Iso-value: mean at surface points
        Z_at_sp = self.interpolator.evaluate(inp.sp_coords, inp, weights)
        iso = Z_at_sp.mean()

        # Smooth sign function (differentiable)
        sign = torch.tanh(50.0 * (Z_fault - iso))

        # Fault normal: average orientation direction
        fault_normal = inp.ori_gradients.mean(dim=0)
        fault_normal = fault_normal / fault_normal.norm().clamp(min=1e-12)

        return {
            'scalar_field': Z_fault,
            'sign': sign,
            'displacement': displacement_param,
            'normal': fault_normal,
            'affected': fault.affected_series_indices,
        }

    def _apply_fault_displacement(
        self,
        grid: torch.Tensor,
        fault_info: dict,
    ) -> torch.Tensor:
        """
        Apply fault displacement to grid coordinates.

        x' = x + sign * displacement * fault_normal
        """
        offset = (
            fault_info['sign'].unsqueeze(-1)
            * fault_info['displacement']
            * fault_info['normal'].unsqueeze(0)
        )
        return grid + offset

    # ------------------------------------------------------------------
    # Series interpolation
    # ------------------------------------------------------------------

    def _interpolate_series(
        self,
        series: SeriesDefinition,
        grid: torch.Tensor,
        soft: bool,
        temperature: float,
    ) -> dict:
        """
        Interpolate a single geological series on the given grid.

        Returns scalar field, iso values, and block (hard or soft).
        """
        inp = InterpolationInput.from_series(series)
        weights = self.interpolator.solve(inp)
        Z = self.interpolator.evaluate(grid, inp, weights)

        # Iso-values from surface points
        Z_at_sp = self.interpolator.evaluate(inp.sp_coords, inp, weights)
        iso_vals = self.interpolator.compute_iso_values(
            Z_at_sp, inp.sp_surface_id, inp.n_surfaces,
        )

        if soft:
            block = self.interpolator.scalar_field_to_block_soft(Z, iso_vals, temperature)
        else:
            block = self.interpolator.scalar_field_to_block(Z, iso_vals)

        return {
            'scalar_field': Z,
            'iso_vals': iso_vals,
            'block': block,
        }

    # ------------------------------------------------------------------
    # Multi-series stacking
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_erode(
        final: torch.Tensor,
        block_new: torch.Tensor,
        block_raw: torch.Tensor,
        soft: bool,
        temperature: float,
    ) -> torch.Tensor:
        """ERODE stacking: new series overwrites where it HAS lithology.

        Presence is decided on the *pre-offset* block ``block_raw`` (id 0 is the
        series' basement / not-deposited region, ids > 0 are its layers). The id
        actually written is the offset ``block_new`` so series ids never clash.
        Deciding presence on ``block_new`` would be wrong: every cell of series
        ≥ 1 carries a positive offset id, so the new series, including its empty
        basement; would erase the entire older stack.
        """
        if soft:
            mask = torch.sigmoid(temperature * (block_raw - 0.5))
            return mask * block_new + (1.0 - mask) * final
        else:
            has_lith = block_raw > 0
            return torch.where(has_lith, block_new, final)

    @staticmethod
    def _stack_onlap(
        final: torch.Tensor,
        block_new: torch.Tensor,
        block_raw: torch.Tensor,
        soft: bool,
        temperature: float,
    ) -> torch.Tensor:
        """ONLAP stacking: new series fills only previously-unassigned regions.

        New-series presence uses the *pre-offset* ``block_raw`` (id 0 = basement);
        the "unassigned" test uses ``final`` (cells never written by any earlier
        series stay exactly 0 through the offset stacking).
        """
        if soft:
            mask_empty = torch.sigmoid(temperature * (0.5 - final))
            has_lith = torch.sigmoid(temperature * (block_raw - 0.5))
            fill = mask_empty * has_lith
            return fill * block_new + (1.0 - fill) * final
        else:
            is_empty = final == 0
            has_lith = block_raw > 0
            fill = is_empty & has_lith
            return torch.where(fill, block_new, final)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_tensor_contracts(self) -> None:
        """Reject tensor mismatches without moving or detaching caller tensors."""
        expected_dtype = self.config.torch_dtype
        expected_device = self.config.torch_device
        for prefix, definition in (
            *((f"series[{index}]", item) for index, item in enumerate(self.series)),
            *((f"faults[{index}]", item) for index, item in enumerate(self.faults)),
        ):
            for name, tensor in (
                ("surface_points.coords", definition.surface_points.coords),
                ("orientations.coords", definition.orientations.coords),
                ("orientations.gradients", definition.orientations.gradients),
            ):
                if tensor.dtype != expected_dtype or tensor.device != expected_device:
                    raise GeomodelContractError(
                        "implicit input tensor must exactly match configuration dtype/device",
                        object_name=type(self).__name__, field=f"{prefix}.{name}",
                        expected=(expected_dtype, expected_device),
                        actual=(tensor.dtype, tensor.device),
                    )
                if tensor.shape[-1] != self.config.ndim:
                    raise GeomodelContractError(
                        "implicit input coordinate dimension does not match configuration",
                        object_name=type(self).__name__, field=f"{prefix}.{name}",
                        expected=self.config.ndim, actual=tensor.shape[-1],
                    )
            ids = definition.surface_points.surface_id
            if ids.device != expected_device:
                raise GeomodelContractError(
                    "surface identifiers must be on the configured device",
                    object_name=type(self).__name__, field=f"{prefix}.surface_points.surface_id",
                    expected=expected_device, actual=ids.device,
                )

    @classmethod
    def capabilities(cls) -> GeomodelCapabilityReport:
        return GeomodelCapabilityReport(
            name="implicit_model",
            domain="geomodel",
            maturity="production",
            algorithm="universal cokriging implicit geology",
            dimensions=(2, 3),
            property_kinds=("categorical",),
            property_units=(None,),
            coordinate_unit="m",
            dtypes=("float32", "float64"),
            devices=("cpu", "cuda", "mps"),
            backends=("torch",),
            conditioning=("surface_points", "orientations", "faults"),
            deterministic_scope="fixed inputs, dtype, and device",
            differentiability="scalar fields, soft blocks, coordinates, orientations, anisotropy, and fault displacement",
            optional_dependencies=(),
            checkpoint_required=False,
            resource_estimate_supported=True,
            unsupported=(GeomodelUnsupportedCombination(
                (("output", "hard_block"), ("autograd", "required")),
                "hard categorical classification is non-differentiable",
                "request soft=True for differentiable output",
            ),),
        )

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "geobrain.geomodel.input/1.0",
            "type": "object",
            "properties": {
                "soft": {"type": "boolean"},
                "temperature": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["soft", "temperature"],
            "additionalProperties": False,
        }

    def estimate_resources(self, *, autograd: bool = True) -> GeomodelResourceEstimate:
        nodes = math.prod(self.config.resolution)
        observations = sum(
            item.surface_points.coords.shape[0] + item.orientations.coords.shape[0] * self.config.ndim
            for item in (*self.series, *self.faults)
        )
        order = observations + self.config.ndim + 1
        itemsize = 8 if self.config.dtype == "float64" else 4
        components = (
            ("grid", nodes * self.config.ndim * itemsize),
            ("augmented_covariance", order * order * itemsize),
            ("scalar_fields", nodes * (len(self.series) + len(self.faults)) * itemsize),
            ("output", nodes * itemsize),
            ("autograd_retention", nodes * itemsize * 4 if autograd else 0),
        )
        total = sum(value for _, value in components)
        return GeomodelResourceEstimate(
            total,
            components,
            0,
            0,
            order * order,
            order,
            order**3 // 3,
            0,
            1,
            ("dense augmented cokriging solve", "one evaluation grid"),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        soft: bool = True,
        temperature: float = 50.0,
    ) -> Dict[str, object]:
        """
        Run the full implicit geological model.

        Steps:
            1. Interpolate fault scalar fields and compute displacements.
            2. For each series (oldest to newest): apply the relevant
               fault displacements to the grid, then cokriging
               interpolation → scalar field → block.
            3. Stack series according to ERODE / ONLAP rules.

        Args:
            soft: If True, use differentiable sigmoid classification.
            temperature: Sigmoid sharpness for soft classification.

        Returns:
            Dict with keys ``'block'``, (n_points,) final lithology ids
            (soft or hard); ``'scalar_fields'``, per-series scalar field
            tensors; ``'iso_vals'``, per-series iso-surface values (pass
            with the matching scalar field to
            ``interpolator.scalar_field_to_property`` to obtain a
            differentiable property field for the physics); ``'grid'``,
            (n_points, D) evaluation grid coordinates.
        """
        grid = self.config.make_grid()

        # Step 1: Process faults
        fault_infos = []
        for i, fault in enumerate(self.faults):
            info = self._interpolate_fault(fault, self.fault_displacements[i], grid)
            fault_infos.append(info)

        # Step 2 & 3: Process series and stack
        final_block = torch.zeros(grid.shape[0], dtype=grid.dtype, device=grid.device)
        scalar_fields = []
        iso_values = []  # per-series iso-surface values (needed to map Z → property)
        series_offset = 0  # offset lithology ids so series don't clash

        for s_idx, series_def in enumerate(self.series):
            # Apply relevant faults to grid
            displaced_grid = grid.clone()
            for fi in fault_infos:
                affected = fi['affected']
                if affected is None or s_idx in affected:
                    displaced_grid = self._apply_fault_displacement(displaced_grid, fi)

            # Interpolate this series
            result = self._interpolate_series(series_def, displaced_grid, soft, temperature)
            scalar_fields.append(result['scalar_field'])
            iso_values.append(result['iso_vals'])

            # Offset block ids by previous series count so series ids never
            # clash. The masks (presence of this series) must be decided on the
            # PRE-offset block ``block_raw``: id 0 is this series' empty basement
            #, while the offset ``block_new`` is the id actually written.
            block_raw = result['block']
            block_new = block_raw + series_offset

            # Stack
            relation = series_def.relation
            if s_idx == 0 or relation == StackRelation.BASEMENT:
                final_block = block_new
            elif relation == StackRelation.ERODE:
                final_block = self._stack_erode(
                    final_block, block_new, block_raw, soft, temperature)
            elif relation == StackRelation.ONLAP:
                final_block = self._stack_onlap(
                    final_block, block_new, block_raw, soft, temperature)

            # Accumulate offset (n_surfaces defines the number of iso-surfaces,
            # which gives n_surfaces + 1 distinct regions, but we use n_surfaces
            # as the offset to keep ids contiguous)
            inp = InterpolationInput.from_series(series_def)
            series_offset += inp.n_surfaces + 1

        return {
            'block': final_block if soft else final_block.detach(),
            'scalar_fields': scalar_fields,
            'iso_vals': iso_values,
            'grid': grid,
        }
