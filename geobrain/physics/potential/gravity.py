"""Strict-SI gravity forward operators.

The 3-D operator compiles core ``TensorMesh`` geometry through the single
depth-down to elevation-up bridge and delegates every component to the
shared potential-field sensitivity executor.  The 2-D operator retains the exact Talwani
infinite-strike kernel while adopting the same immutable survey, explicit
execution, validation, and SI output contracts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, ClassVar, Literal, cast

import torch

from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ErrorCode,
    ForwardContext,
    ForwardOperator,
    ForwardOutput,
    ModelState,
)
from geobrain.mesh import canonicalize_cell_field, require_capable_mesh
from geobrain.mesh.capabilities import PrismGeometryMesh, StructuredMesh
from geobrain.mesh.tensor import TensorMesh

from ._engine import build_sensitivity
from ._engine.execute import PotentialSensitivity
from ._engine.plan import PrismKernelPlan, tensor_mesh_cell_bounds_m
from .capabilities import (
    PotentialCapabilityReport,
    potential_capability_report,
    potential_input_schema,
)
from .resources import (
    PotentialResourceEstimate,
    estimate_prism_resources,
    infeasible_resource_estimate,
)
from .config import PotentialExecutionConfig
from .errors import (
    PotentialCapabilityError,
    PotentialContractError,
    PotentialNumericsError,
    PotentialResourceError,
)
from .surveys import PotentialSurvey2D, PotentialSurvey3D


GravityComponent = Literal["gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
_GRAVITY_COMPONENTS = frozenset({"gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"})
_ACCELERATION_COMPONENTS = frozenset({"gx", "gy", "gz"})
from .helpers import G_SI as _GRAVITATIONAL_CONSTANT_SI  # single-sourced (identical value)
_TWO_DIMENSIONAL_FIXED_OVERHEAD_BYTES = 262_144


def _execution_or_default(execution: PotentialExecutionConfig | None) -> PotentialExecutionConfig:
    if execution is None:
        return PotentialExecutionConfig()
    if not isinstance(execution, PotentialExecutionConfig):
        raise PotentialContractError(
            "Gravity execution must be PotentialExecutionConfig or None.",
            object_name="Gravity",
            field="execution",
            expected="PotentialExecutionConfig or None",
            actual=execution,
            hint="Provide an explicit validated Potential execution policy.",
        )
    return execution


def _differentiability(
    execution: PotentialExecutionConfig,
    output_keys: tuple[str, ...],
) -> DifferentiabilitySpec:
    level = (
        DifferentiabilityLevel.FULL_AUTOGRAD
        if execution.strategy in {"dense", "store"}
        else DifferentiabilityLevel.CUSTOM_VJP
    )
    return DifferentiabilitySpec(
        level=level,
        trainable_inputs=("rho",),
        output_keys=output_keys,
        input_units={"rho": "kg/m^3"},
    )


def _validate_components(components: object) -> tuple[GravityComponent, ...]:
    if not isinstance(components, tuple) or not components:
        raise PotentialContractError(
            "Gravity components must be a non-empty tuple.",
            object_name="Gravity3D",
            field="components",
            expected="non-empty ordered tuple of unique gravity component names",
            actual=components,
            hint="Request one or more SI gravity acceleration or tensor components.",
        )
    if len(set(components)) != len(components):
        raise PotentialContractError(
            "Gravity components must be unique.",
            object_name="Gravity3D",
            field="components",
            expected="unique ordered gravity component names",
            actual=components,
            hint="Remove duplicate components while preserving order.",
        )
    unsupported = tuple(
        component
        for component in components
        if not isinstance(component, str) or component not in _GRAVITY_COMPONENTS
    )
    if unsupported:
        raise PotentialCapabilityError(
            "Gravity components contain unsupported names.",
            object_name="Gravity3D",
            field="components",
            expected=sorted(_GRAVITY_COMPONENTS),
            actual=unsupported,
            hint="Use only supported Cartesian acceleration and tensor names.",
        )
    return cast(tuple[GravityComponent, ...], components)


def _require_prism_mesh(ctx: ForwardContext, *, dimensions: int, owner: str):
    # Capability-declared requirement; no concrete-class guard. The 3-D
    # prism engine consumes only PrismGeometryMesh.cell_bounds(), so graded
    # octrees qualify; the 2-D Talwani path still needs the structured
    # section layout.
    if dimensions == 3:
        mesh = require_capable_mesh(ctx, PrismGeometryMesh, owner=owner)
    else:
        mesh = require_capable_mesh(ctx, StructuredMesh, PrismGeometryMesh, owner=owner)
    if mesh.n_dim != dimensions:
        raise PotentialContractError(
            f"{owner} requires a {dimensions}-D TensorMesh.",
            object_name=owner,
            field="mesh",
            expected=f"{dimensions}-D TensorMesh",
            actual={"type": type(mesh).__name__, "n_dim": getattr(mesh, "n_dim", None)},
            hint=f"Provide a {dimensions}-D core TensorMesh in ForwardContext.",
        )
    return mesh


def _validate_density(value: torch.Tensor, *, owner: str) -> torch.Tensor:
    if value.layout != torch.strided:
        raise PotentialCapabilityError(
            "Gravity density requires a strided tensor layout.",
            object_name=owner,
            field="rho",
            expected="torch.strided",
            actual=value.layout,
            hint="Materialize density as a strided tensor before execution.",
        )
    if value.dtype not in {torch.float32, torch.float64}:
        raise PotentialContractError(
            "Gravity density dtype is unsupported.",
            object_name=owner,
            field="rho",
            expected=["torch.float32", "torch.float64"],
            actual=value.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Use float32 or float64 SI density contrast.",
        )
    if value.device.type not in {"cpu", "cuda"}:
        raise PotentialContractError(
            "Gravity density device is unsupported.",
            object_name=owner,
            field="rho",
            expected=["cpu", "cuda"],
            actual=value.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Move density explicitly to CPU or an available CUDA device.",
        )
    if not bool(torch.isfinite(value).all()):
        raise PotentialContractError(
            "Gravity density must be finite.",
            object_name=owner,
            field="rho",
            expected="finite kg/m^3 density contrast",
            actual="contains non-finite values",
            hint="Replace NaN or infinite density values before execution.",
        )
    return value


def _talwani_antiderivative(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return w * torch.atan2(u, w) + 0.5 * u * torch.log(u.square() + w.square())


class _Gravity2DTiledFunction(torch.autograd.Function):  # type: ignore[misc, unused-ignore]
    @staticmethod
    def forward(
        ctx: Any,
        operator: Gravity2D,
        observations: torch.Tensor,
        bounds: torch.Tensor,
        model: torch.Tensor,
        observation_tile_size: int,
        cell_tile_size: int,
    ) -> torch.Tensor:
        ctx.operator = operator
        ctx.observation_tile_size = observation_tile_size
        ctx.cell_tile_size = cell_tile_size
        ctx.save_for_backward(observations, bounds)
        return operator._apply_tiled(
            observations,
            bounds,
            model,
            observation_tile_size=observation_tile_size,
            cell_tile_size=cell_tile_size,
        )

    @staticmethod
    def backward(
        ctx: Any,
        cotangent: torch.Tensor,
    ) -> tuple[None, None, None, torch.Tensor, None, None]:
        if torch.is_grad_enabled():
            raise PotentialCapabilityError(
                "Tiled Gravity2D supports first-order differentiation only.",
                object_name="Gravity2D",
                field="differentiation_order",
                expected="first-order model VJP",
                actual="higher-order derivative requested",
                hint="Use dense/store execution for full autograd.",
            )
        operator = cast(Gravity2D, ctx.operator)
        observations, bounds = ctx.saved_tensors
        gradient = operator._transpose_tiled(
            observations,
            bounds,
            cotangent,
            observation_tile_size=cast(int, ctx.observation_tile_size),
            cell_tile_size=cast(int, ctx.cell_tile_size),
        )
        return None, None, None, gradient, None, None


class Gravity2D(ForwardOperator):  # type: ignore[misc, unused-ignore]
    """Talwani infinite-strike gravity with elevation-positive-up stations.

    Args:
        survey: 2-D station geometry.
        execution: optional strategy/tiling/budget policy.
    """

    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (
        StructuredMesh,
        PrismGeometryMesh,
    )

    def __init__(
        self,
        survey: PotentialSurvey2D,
        *,
        execution: PotentialExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, PotentialSurvey2D):
            raise PotentialContractError(
                "Gravity2D requires PotentialSurvey2D.",
                object_name=type(self).__name__,
                field="survey",
                expected="PotentialSurvey2D with (x, z) metre columns",
                actual=survey,
                hint="Construct an immutable PotentialSurvey2D first.",
            )
        self.survey = survey
        self.execution = _execution_or_default(execution)
        self.differentiability = _differentiability(self.execution, ("gz",))
        self._stored_key: tuple[object, ...] | None = None
        self._stored_kernel: torch.Tensor | None = None

    @classmethod
    def capabilities(cls) -> PotentialCapabilityReport:
        return potential_capability_report(
            operator_id="gravity-2d",
            components=("gz",),
            output_units={"gz": "m/s^2"},
            state_fields={"rho": "kg/m^3"},
            survey_width=2,
        )

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        return potential_input_schema(
            operator_id="gravity-2d",
            components=("gz",),
            tensor_name="rho",
            survey_width=2,
        )

    def estimate_resources(
        self,
        state: ModelState,
        ctx: ForwardContext,
    ) -> PotentialResourceEstimate:
        (rho,) = state.fetch("rho")
        _validate_density(rho, owner=type(self).__name__)
        mesh = _require_prism_mesh(ctx, dimensions=2, owner=type(self).__name__)
        itemsize = rho.element_size()
        input_bytes = (self.survey.positions_m.numel() + mesh.n_cells * 5) * itemsize
        output_bytes = self.survey.positions_m.shape[0] * itemsize
        working_set_bytes = 65_536 + self.survey.positions_m.shape[0] * mesh.n_cells * itemsize
        persistent_bytes = working_set_bytes - 65_536 if self.execution.strategy == "store" else 0
        peak_bytes = input_bytes + output_bytes + working_set_bytes + persistent_bytes
        if peak_bytes > self.execution.budget_bytes:
            return infeasible_resource_estimate(self.execution)
        selected = "dense" if self.execution.strategy == "auto" else self.execution.strategy
        if selected not in {"dense", "tiled", "store"}:  # pragma: no cover
            raise AssertionError(f"resolved strategy {selected!r} escaped validation")
        return PotentialResourceEstimate(
            schema_version="geobrain.potential.resource/1.0",
            requested_strategy=self.execution.strategy,
            selected_strategy=cast(Literal["dense", "tiled", "store"], selected),
            budget_bytes=self.execution.budget_bytes,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            working_set_bytes=working_set_bytes,
            persistent_bytes=persistent_bytes,
            peak_bytes=peak_bytes,
            observation_tile_size=(
                self.execution.observation_tile_size if selected == "tiled" else None
            ),
            cell_tile_size=self.execution.cell_tile_size if selected == "tiled" else None,
            feasible=True,
            reason_code=None,
        )

    def _kernel(
        self,
        observations: torch.Tensor,
        bounds: torch.Tensor,
        *,
        observation_slice: slice,
        cell_slice: slice,
    ) -> torch.Tensor:
        selected_observations = observations[observation_slice]
        selected_bounds = bounds[cell_slice]
        station_x = selected_observations[:, 0, None]
        station_depth = -selected_observations[:, 1, None]
        z_lo = selected_bounds[None, :, 0]
        z_hi = selected_bounds[None, :, 1]
        x_lo = selected_bounds[None, :, 2]
        x_hi = selected_bounds[None, :, 3]
        inside = (
            (station_depth >= z_lo)
            & (station_depth <= z_hi)
            & (station_x >= x_lo)
            & (station_x <= x_hi)
        )
        if bool(inside.any()):
            raise PotentialContractError(
                "Gravity2D observations must be outside every source cell.",
                object_name=type(self).__name__,
                field="survey.positions_m",
                expected="stations outside all closed 2-D cells",
                actual="contains a station on or inside a source cell",
                hint="Move the station above the mesh in elevation-positive-up coordinates.",
            )
        u_left = x_lo - station_x
        u_right = x_hi - station_x
        w_top = z_lo - station_depth
        w_bottom = z_hi - station_depth
        geometric = (
            _talwani_antiderivative(u_right, w_bottom)
            - _talwani_antiderivative(u_right, w_top)
            - _talwani_antiderivative(u_left, w_bottom)
            + _talwani_antiderivative(u_left, w_top)
        )
        # The Talwani formula is depth-positive-down; public gz is elevation-up.
        return -2.0 * _GRAVITATIONAL_CONSTANT_SI * geometric

    def _apply_tiled(
        self,
        observations: torch.Tensor,
        bounds: torch.Tensor,
        model: torch.Tensor,
        *,
        observation_tile_size: int,
        cell_tile_size: int,
    ) -> torch.Tensor:
        result = torch.zeros(
            observations.shape[0],
            dtype=model.dtype,
            device=model.device,
        )
        for observation_start in range(0, observations.shape[0], observation_tile_size):
            observation_stop = min(
                observation_start + observation_tile_size,
                observations.shape[0],
            )
            observation_slice = slice(observation_start, observation_stop)
            for cell_start in range(0, bounds.shape[0], cell_tile_size):
                cell_stop = min(cell_start + cell_tile_size, bounds.shape[0])
                cell_slice = slice(cell_start, cell_stop)
                result[observation_slice] = (
                    result[observation_slice]
                    + self._kernel(
                        observations,
                        bounds,
                        observation_slice=observation_slice,
                        cell_slice=cell_slice,
                    )
                    @ model[cell_slice]
                )
        return result

    def _transpose_tiled(
        self,
        observations: torch.Tensor,
        bounds: torch.Tensor,
        cotangent: torch.Tensor,
        *,
        observation_tile_size: int,
        cell_tile_size: int,
    ) -> torch.Tensor:
        result = torch.zeros(
            bounds.shape[0],
            dtype=cotangent.dtype,
            device=cotangent.device,
        )
        for observation_start in range(0, observations.shape[0], observation_tile_size):
            observation_stop = min(
                observation_start + observation_tile_size,
                observations.shape[0],
            )
            observation_slice = slice(observation_start, observation_stop)
            for cell_start in range(0, bounds.shape[0], cell_tile_size):
                cell_stop = min(cell_start + cell_tile_size, bounds.shape[0])
                cell_slice = slice(cell_start, cell_stop)
                kernel = self._kernel(
                    observations,
                    bounds,
                    observation_slice=observation_slice,
                    cell_slice=cell_slice,
                )
                result[cell_slice] = result[cell_slice] + (kernel.mT @ cotangent[observation_slice])
        return result

    def _resolve_execution(
        self,
        *,
        n_observations: int,
        n_cells: int,
        itemsize: int,
        input_output_bytes: int,
    ) -> tuple[Literal["dense", "tiled", "store"], int | None, int | None, int]:
        dense_peak = (
            _TWO_DIMENSIONAL_FIXED_OVERHEAD_BYTES
            + input_output_bytes
            + n_observations * n_cells * 8 * itemsize
        )
        requested = self.execution.strategy
        dense_selected = requested in {"dense", "store"} or (
            requested == "auto" and dense_peak <= self.execution.budget_bytes
        )
        if dense_selected:
            if dense_peak > self.execution.budget_bytes:
                raise PotentialResourceError(
                    "Gravity2D full execution exceeds the configured budget.",
                    object_name=type(self).__name__,
                    field="budget_bytes",
                    expected=f">= {dense_peak}",
                    actual=self.execution.budget_bytes,
                    hint="Increase the budget or select tiled execution.",
                )
            selected = "dense" if requested == "auto" else requested
            return cast(Literal["dense", "store"], selected), None, None, dense_peak

        base_bytes = _TWO_DIMENSIONAL_FIXED_OVERHEAD_BYTES + input_output_bytes
        bytes_per_pair = 8 * itemsize
        max_pairs = (self.execution.budget_bytes - base_bytes) // bytes_per_pair
        if max_pairs < 1:
            raise PotentialResourceError(
                "Gravity2D tiled execution cannot fit one observation-cell pair.",
                object_name=type(self).__name__,
                field="budget_bytes",
                expected=f">= {base_bytes + bytes_per_pair}",
                actual=self.execution.budget_bytes,
                hint="Increase budget_bytes enough for fixed inputs and one bounded tile.",
            )
        requested_observation_tile = self.execution.observation_tile_size
        requested_cell_tile = self.execution.cell_tile_size
        if requested_observation_tile is None and requested_cell_tile is None:
            observation_tile = min(n_observations, max_pairs)
            cell_tile = min(n_cells, max(1, max_pairs // observation_tile))
        elif requested_observation_tile is None:
            if requested_cell_tile is None:  # pragma: no cover - planner invariant
                raise AssertionError("tile resolution lost the requested cell tile")
            cell_tile = min(n_cells, requested_cell_tile)
            observation_tile = min(n_observations, max(1, max_pairs // cell_tile))
        elif requested_cell_tile is None:
            observation_tile = min(n_observations, requested_observation_tile)
            cell_tile = min(n_cells, max(1, max_pairs // observation_tile))
        else:
            observation_tile = min(n_observations, requested_observation_tile)
            cell_tile = min(n_cells, requested_cell_tile)
        tiled_peak = base_bytes + observation_tile * cell_tile * bytes_per_pair
        if tiled_peak > self.execution.budget_bytes:
            raise PotentialResourceError(
                "Gravity2D requested tile sizes exceed the configured budget.",
                object_name=type(self).__name__,
                field="budget_bytes",
                expected=f">= {tiled_peak}",
                actual=self.execution.budget_bytes,
                hint="Increase the budget or reduce the explicit tile sizes.",
            )
        return "tiled", observation_tile, cell_tile, tiled_peak

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (rho_input,) = state.fetch("rho")
        _validate_density(rho_input, owner=type(self).__name__)
        mesh = _require_prism_mesh(ctx, dimensions=2, owner=type(self).__name__)
        rho = canonicalize_cell_field(mesh, rho_input, name="rho", owner=type(self).__name__)
        observations = self.survey.positions_m.to(device=rho.device, dtype=rho.dtype)
        bounds = mesh.cell_bounds().to(device=rho.device, dtype=rho.dtype)
        n_observations = observations.shape[0]
        n_cells = bounds.shape[0]
        itemsize = rho.element_size()
        input_output_bytes = (
            observations.numel() + bounds.numel() + rho.numel() + n_observations
        ) * itemsize
        selected, observation_tile, cell_tile, peak_bytes = self._resolve_execution(
            n_observations=n_observations,
            n_cells=n_cells,
            itemsize=itemsize,
            input_output_bytes=input_output_bytes,
        )
        requested = self.execution.strategy
        flat_rho = rho.reshape(-1)
        cache_event = "none"
        if selected == "store":
            key = (
                observations.detach().cpu().numpy().tobytes(),
                bounds.detach().cpu().numpy().tobytes(),
                rho.dtype,
                rho.device,
            )
            if self._stored_key == key and self._stored_kernel is not None:
                kernel = self._stored_kernel
                cache_event = "hit"
            else:
                kernel = self._kernel(
                    observations,
                    bounds,
                    observation_slice=slice(None),
                    cell_slice=slice(None),
                ).detach()
                self._stored_key = key
                self._stored_kernel = kernel
                cache_event = "miss"
            gz = kernel @ flat_rho
        elif selected == "dense":
            kernel = self._kernel(
                observations,
                bounds,
                observation_slice=slice(None),
                cell_slice=slice(None),
            )
            gz = kernel @ flat_rho
        else:
            if observation_tile is None or cell_tile is None:  # pragma: no cover
                raise AssertionError("tiled execution reached without resolved tiles")
            gz = _Gravity2DTiledFunction.apply(  # type: ignore[no-untyped-call, unused-ignore]
                self,
                observations,
                bounds,
                flat_rho,
                observation_tile,
                cell_tile,
            )
        if not bool(torch.isfinite(gz).all()):
            raise PotentialNumericsError(
                "Gravity2D produced a non-finite result.",
                object_name=type(self).__name__,
                field="gz",
                expected="finite m/s^2 output",
                actual={"shape": list(gz.shape), "dtype": gz.dtype, "device": gz.device},
                code=ErrorCode.EXECUTION_FAILED,
                hint="Use float64 or rescale the SI geometry.",
            )
        return ForwardOutput(
            data={"gz": gz},
            metadata={
                "units": {"gz": "m/s^2"},
                "kernel": "talwani_1959",
                "requested_strategy": requested,
                "selected_strategy": selected,
                "observation_tile_size": observation_tile,
                "cell_tile_size": cell_tile,
                "peak_bytes": peak_bytes,
                "cache_event": cache_event,
            },
        )


class Gravity3D(ForwardOperator):  # type: ignore[misc, unused-ignore]
    """Bounded rectangular-prism gravity in strict SI units.

    Args:
        survey: 3-D station geometry.
        components: gravity components to emit (``'gz'``, ``'gx'``, ...).
        execution: optional strategy/tiling/budget policy.
    """

    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (PrismGeometryMesh,)

    def __init__(
        self,
        survey: PotentialSurvey3D,
        *,
        components: tuple[GravityComponent, ...] = ("gz",),
        execution: PotentialExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, PotentialSurvey3D):
            raise PotentialContractError(
                "Gravity3D requires PotentialSurvey3D.",
                object_name=type(self).__name__,
                field="survey",
                expected="PotentialSurvey3D with (x, y, z) metre columns",
                actual=survey,
                hint="Construct an immutable PotentialSurvey3D first.",
            )
        self.survey = survey
        self.components = _validate_components(components)
        self.execution = _execution_or_default(execution)
        self.differentiability = _differentiability(self.execution, self.components)
        self._sensitivity_cache: OrderedDict[str, PotentialSensitivity] = OrderedDict()

    @classmethod
    def capabilities(cls) -> PotentialCapabilityReport:
        components = ("gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz")
        return potential_capability_report(
            operator_id="gravity-3d",
            components=components,
            output_units={
                component: "m/s^2" if len(component) == 2 else "s^-2"
                for component in components
            },
            state_fields={"rho": "kg/m^3"},
            survey_width=3,
        )

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        components = ("gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz")
        return potential_input_schema(
            operator_id="gravity-3d",
            components=components,
            tensor_name="rho",
            survey_width=3,
        )

    def estimate_resources(
        self,
        state: ModelState,
        ctx: ForwardContext,
    ) -> PotentialResourceEstimate:
        (rho_input,) = state.fetch("rho")
        _validate_density(rho_input, owner=type(self).__name__)
        mesh = _require_prism_mesh(ctx, dimensions=3, owner=type(self).__name__)
        rho = canonicalize_cell_field(
            mesh,
            rho_input,
            name="rho",
            owner=type(self).__name__,
        )
        plan = PrismKernelPlan.build(
            observations_m=self.survey.positions_m.to(dtype=rho.dtype, device=rho.device),
            cell_bounds_m=tensor_mesh_cell_bounds_m(mesh).to(
                dtype=rho.dtype,
                device=rho.device,
            ),
            components=self.components,
            dtype=rho.dtype,
            device=rho.device,
        )
        try:
            return estimate_prism_resources(
                plan=plan,
                execution=self.execution,
                model_field_count=1,
            )
        except PotentialResourceError:
            return infeasible_resource_estimate(self.execution)

    def _compile_sensitivity(
        self,
        *,
        mesh: TensorMesh,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[PotentialSensitivity, str]:
        observations = self.survey.positions_m.to(dtype=dtype, device=device)
        bounds = tensor_mesh_cell_bounds_m(mesh).to(dtype=dtype, device=device)
        plan = PrismKernelPlan.build(
            observations_m=observations,
            cell_bounds_m=bounds,
            components=self.components,
            dtype=dtype,
            device=device,
        )
        cached = self._sensitivity_cache.pop(plan.fingerprint, None)
        if cached is not None:
            self._sensitivity_cache[plan.fingerprint] = cached
            return cached, "hit"
        sensitivity = build_sensitivity(
            plan=plan,
            execution=self.execution,
            model_contract="rho",
            projection=None,
        )
        self._sensitivity_cache[plan.fingerprint] = sensitivity
        if len(self._sensitivity_cache) > 8:
            self._sensitivity_cache.popitem(last=False)
        return sensitivity, "miss"

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (rho_input,) = state.fetch("rho")
        _validate_density(rho_input, owner=type(self).__name__)
        mesh = _require_prism_mesh(ctx, dimensions=3, owner=type(self).__name__)
        rho = canonicalize_cell_field(mesh, rho_input, name="rho", owner=type(self).__name__)
        sensitivity, cache_event = self._compile_sensitivity(
            mesh=mesh,
            dtype=rho.dtype,
            device=rho.device,
        )
        data = sensitivity.apply({"rho": rho.reshape(-1)})
        units = {
            component: "m/s^2" if component in _ACCELERATION_COMPONENTS else "s^-2"
            for component in self.components
        }
        estimate = sensitivity.estimate
        return ForwardOutput(
            data=data,
            fields={},
            metadata={
                "units": units,
                "kernel": "rectangular_prism_si",
                "requested_strategy": self.execution.strategy,
                "selected_strategy": sensitivity.selected_strategy,
                "plan_fingerprint": sensitivity.plan.fingerprint,
                "observation_tile_size": estimate.observation_tile_size,
                "cell_tile_size": estimate.cell_tile_size,
                "cache_event": cache_event,
                "resource_estimate": estimate.to_dict(),
            },
        )


__all__ = ["Gravity2D", "Gravity3D", "GravityComponent"]
