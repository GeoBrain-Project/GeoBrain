"""Unified dimension-neutral eager propagation for packed Wave requests.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from typing import Callable, Mapping, Sequence, cast

import torch
from torch import Tensor

from geobrain.core.runtime import maybe_compile

from ...errors import WaveCapabilityError, WaveContractError, WaveNumericsError
from ..boundaries.cpml import CPML, build_cpml
from ..boundaries.cpml3d import CPML3D, build_cpml_3d
from ..contracts import (
    ExecutionTelemetry,
    PropagationRequest,
    PropagationResult,
    WaveEquationProtocol,
)


def _numerics_error(
    equation: WaveEquationProtocol,
    field: str,
    expected: object,
    actual: object,
) -> WaveNumericsError:
    """Build a consistently attributed eager numerical failure."""
    return WaveNumericsError(
        "Wave eager execution could not produce a complete finite result",
        object_name=type(equation).__name__,
        field=field,
        expected=expected,
        actual=actual,
    )


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    equation: WaveEquationProtocol
    advance: Callable[..., Sequence[Tensor]]
    coefficients: Mapping[str, Tensor]
    boundary: object
    dt: float
    spacing: tuple[float, ...]
    source_indices: Tensor
    source_shot_index: Tensor
    receiver_indices: Tensor
    receiver_shot_index: Tensor
    components: tuple[str, ...]
    interior_slices: tuple[slice, ...]
    illumination: tuple[str, ...]
    snapshot_policy: str
    snapshot_indices: tuple[int, ...]
    telemetry: ExecutionTelemetry


def _all_finite(tensors: Sequence[Tensor]) -> bool:
    """Check state finiteness without changing its autograd graph."""
    with torch.no_grad():
        return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _sample_components(
    context: _ExecutionContext, state: Sequence[Tensor]
) -> Tensor:
    """Return one packed receiver sample with exact declared component order."""
    try:
        sampled = dict(
            context.equation.sample_receivers(
                state,
                context.receiver_indices,
                context.receiver_shot_index,
                context.components,
            )
        )
    except (KeyError, IndexError, TypeError, RuntimeError) as exc:
        raise _numerics_error(
            context.equation,
            "components",
            context.components,
            "receiver sampling failed",
        ) from exc
    if tuple(sampled) != context.components:
        raise _numerics_error(
            context.equation,
            "components",
            context.components,
            tuple(sampled),
        )
    values = tuple(sampled[name] for name in context.components)
    expected_shape = (context.receiver_indices.shape[0],)
    if any(
        not isinstance(value, Tensor) or tuple(value.shape) != expected_shape
        for value in values
    ):
        raise _numerics_error(
            context.equation,
            "components",
            f"one vector of shape {expected_shape} per component",
            tuple(
                tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
                for value in values
            ),
        )
    packed = torch.stack(values, dim=-1)
    if not _all_finite((packed,)):
        raise _numerics_error(
            context.equation, "traces", "finite receiver samples", "non-finite"
        )
    return packed


def _accumulate_illumination(
    context: _ExecutionContext,
    state: Sequence[Tensor],
    accumulated: dict[str, Tensor],
) -> None:
    """Accumulate requested energy maps inside the shared traversal."""
    if not context.illumination:
        return
    try:
        available = dict(context.equation.illumination_fields(state))
    except (KeyError, IndexError, TypeError, RuntimeError) as exc:
        raise _numerics_error(
            context.equation,
            "illumination",
            context.illumination,
            "illumination collection failed",
        ) from exc
    for name in context.illumination:
        if name not in available:
            raise _numerics_error(
                context.equation,
                "illumination",
                context.illumination,
                tuple(available),
            )
        field = available[name]
        reference = state[0]
        if (
            not isinstance(field, Tensor)
            or tuple(field.shape) != tuple(reference.shape)
            or field.dtype is not reference.dtype
            or field.device != reference.device
        ):
            raise _numerics_error(
                context.equation,
                "illumination",
                (
                    f"{name!r} tensor with shape={tuple(reference.shape)}, "
                    f"dtype={reference.dtype}, device={reference.device}"
                ),
                (
                    type(field).__name__
                    if not isinstance(field, Tensor)
                    else (
                        f"shape={tuple(field.shape)}, dtype={field.dtype}, "
                        f"device={field.device}"
                    )
                ),
            )
        if not _all_finite((field,)):
            raise _numerics_error(
                context.equation,
                "illumination",
                f"finite {name!r} field",
                "non-finite",
            )
        try:
            interior = field[(slice(None), 0, *context.interior_slices)]
        except (IndexError, RuntimeError, TypeError) as exc:
            raise _numerics_error(
                context.equation,
                "illumination",
                f"sliceable {name!r} execution field",
                "interior extraction failed",
            ) from exc
        energy = interior.pow(2).sum(dim=0)
        accumulated[name] = (
            energy if name not in accumulated else accumulated[name] + energy
        )


def _collect_final_snapshot(
    context: _ExecutionContext,
    state: tuple[Tensor, ...],
) -> dict[str, Tensor]:
    """Validate and crop the one complete final wavefield snapshot."""
    try:
        snapshots = dict(context.equation.snapshot_fields(state))
    except (KeyError, IndexError, TypeError, RuntimeError, ValueError) as exc:
        raise _numerics_error(
            context.equation,
            "fields",
            "complete final snapshot mapping",
            "snapshot collection failed",
        ) from exc
    if tuple(snapshots) != ("wavefield",):
        raise _numerics_error(
            context.equation,
            "fields",
            ("wavefield",),
            tuple(snapshots),
        )
    field = snapshots["wavefield"]
    reference = state[0]
    if (
        not isinstance(field, Tensor)
        or tuple(field.shape) != tuple(reference.shape)
        or field.dtype is not reference.dtype
        or field.device != reference.device
    ):
        raise _numerics_error(
            context.equation,
            "fields",
            (
                f"wavefield shape={tuple(reference.shape)}, "
                f"dtype={reference.dtype}, device={reference.device}"
            ),
            (
                type(field).__name__
                if not isinstance(field, Tensor)
                else (
                    f"shape={tuple(field.shape)}, dtype={field.dtype}, "
                    f"device={field.device}"
                )
            ),
        )
    if not _all_finite((field,)):
        raise _numerics_error(
            context.equation,
            "fields",
            "finite final wavefield",
            "non-finite",
        )
    try:
        interior = field[(slice(None), slice(None), *context.interior_slices)]
    except (IndexError, RuntimeError, TypeError) as exc:
        raise _numerics_error(
            context.equation,
            "fields",
            "sliceable final wavefield",
            "interior extraction failed",
        ) from exc
    expected_shape = (
        int(reference.shape[0]),
        int(reference.shape[1]),
        *tuple(
            int(spatial_slice.stop) - int(spatial_slice.start)
            for spatial_slice in context.interior_slices
        ),
    )
    if tuple(interior.shape) != expected_shape or not _all_finite((interior,)):
        raise _numerics_error(
            context.equation,
            "fields",
            f"finite interior wavefield shape={expected_shape}",
            f"shape={tuple(interior.shape)}",
        )
    return {"wavefield": interior}


def _collect_illumination_results(
    context: _ExecutionContext,
    illumination: Mapping[str, Tensor],
    reference: Tensor,
) -> dict[str, Tensor]:
    """Validate complete full-memory illumination maps before publication."""
    if tuple(illumination) != context.illumination:
        raise _numerics_error(
            context.equation,
            "illumination",
            context.illumination,
            tuple(illumination),
        )
    expected_shape = tuple(
        int(spatial_slice.stop) - int(spatial_slice.start)
        for spatial_slice in context.interior_slices
    )
    result: dict[str, Tensor] = {}
    for name in context.illumination:
        value = illumination[name]
        if (
            not isinstance(value, Tensor)
            or tuple(value.shape) != expected_shape
            or value.dtype is not reference.dtype
            or value.device != reference.device
            or not _all_finite((value,))
        ):
            raise _numerics_error(
                context.equation,
                "illumination",
                (
                    f"finite {name!r} map with shape={expected_shape}, "
                    f"dtype={reference.dtype}, device={reference.device}"
                ),
                (
                    type(value).__name__
                    if not isinstance(value, Tensor)
                    else (
                        f"shape={tuple(value.shape)}, dtype={value.dtype}, "
                        f"device={value.device}"
                    )
                ),
            )
        result[f"forward_wavefield_{name}"] = value
    return result


def _run_segment(
    context: _ExecutionContext,
    state: Sequence[Tensor],
    wavelet_chunk: Tensor,
    *,
    collect_illumination: bool,
    time_start: int = 0,
) -> tuple[tuple[Tensor, ...], Tensor, dict[str, Tensor]]:
    """The single time-traversal primitive used by both 2-D and 3-D execution."""
    records: list[Tensor] = []
    retained: dict[str, Tensor] = {}
    illumination: dict[str, Tensor] = {}
    wavefield_energy: Tensor | None = None
    current = tuple(state)
    for time_index in range(int(wavelet_chunk.shape[1])):
        try:
            current = tuple(
                context.advance(
                    current,
                    context.coefficients,
                    boundary=context.boundary,
                    dt=context.dt,
                    spacing=context.spacing,
                )
            )
            current = tuple(
                context.equation.inject_sources(
                    current,
                    context.source_indices,
                    context.source_shot_index,
                    wavelet_chunk[:, time_index],
                    dt=context.dt,
                )
            )
            context.telemetry.record_advance()
        except (KeyError, IndexError, TypeError, RuntimeError) as exc:
            raise _numerics_error(
                context.equation,
                "state",
                "complete equation state",
                "step or source injection failed",
            ) from exc
        if len(current) != len(
            context.equation.declaration.state_fields
        ) + len(context.equation.declaration.cpml_fields):
            raise _numerics_error(
                context.equation,
                "state",
                (
                    len(context.equation.declaration.state_fields)
                    + len(context.equation.declaration.cpml_fields)
                ),
                len(current),
            )
        if not _all_finite(current):
            raise _numerics_error(
                context.equation, "state", "finite state tensors", "non-finite"
            )
        context.telemetry.observe_live_state(current)
        records.append(_sample_components(context, current))
        global_time_index = time_start + time_index
        if (
            context.snapshot_policy == "selected"
            and global_time_index in context.snapshot_indices
        ):
            retained[f"snapshot:{global_time_index}"] = _collect_final_snapshot(
                context, current
            )["wavefield"]
        if context.snapshot_policy == "energy":
            wavefield = _collect_final_snapshot(context, current)["wavefield"]
            energy = wavefield.pow(2).sum(dim=(0, 1))
            wavefield_energy = (
                energy
                if wavefield_energy is None
                else wavefield_energy + energy
            )
        if collect_illumination:
            _accumulate_illumination(context, current, illumination)
    if wavefield_energy is not None:
        retained["wavefield_energy"] = wavefield_energy
    retained.update(illumination)
    return current, torch.stack(records, dim=1), retained


def _pad_model(
    tensor: Tensor, padding: tuple[tuple[int, int], ...]
) -> Tensor:
    """Replicate-pad a live model without moving or casting it."""
    torch_padding = tuple(value for pair in reversed(padding) for value in pair)
    wrapped = tensor[(None, None, *([slice(None)] * tensor.ndim))]
    return torch.nn.functional.pad(
        wrapped, torch_padding, mode="replicate"
    )[0, 0]


def _null_cpml_2d(
    shape: tuple[int, int], *, device: torch.device, dtype: torch.dtype
) -> CPML:
    nz, nx = shape
    zx = torch.zeros((1, 1, 1, nx), device=device, dtype=dtype)
    zz = torch.zeros((1, 1, nz, 1), device=device, dtype=dtype)
    onex = torch.ones_like(zx)
    onez = torch.ones_like(zz)
    return CPML(
        bx_int=zx,
        ax_int=zx,
        kx_int=onex,
        bz_int=zz,
        az_int=zz,
        kz_int=onez,
        bx_half=zx,
        ax_half=zx,
        kx_half=onex,
        bz_half=zz,
        az_half=zz,
        kz_half=onez,
    )


def _null_cpml_3d(
    shape: tuple[int, int, int], *, device: torch.device, dtype: torch.dtype
) -> CPML3D:
    nz, ny, nx = shape

    def zeros(extent: tuple[int, ...]) -> Tensor:
        return torch.zeros(extent, device=device, dtype=dtype)

    sx = (1, 1, 1, 1, nx)
    sy = (1, 1, 1, ny, 1)
    sz = (1, 1, nz, 1, 1)
    return CPML3D(
        bx_int=zeros(sx),
        ax_int=zeros(sx),
        kx_int=torch.ones(sx, device=device, dtype=dtype),
        by_int=zeros(sy),
        ay_int=zeros(sy),
        ky_int=torch.ones(sy, device=device, dtype=dtype),
        bz_int=zeros(sz),
        az_int=zeros(sz),
        kz_int=torch.ones(sz, device=device, dtype=dtype),
        bx_half=zeros(sx),
        ax_half=zeros(sx),
        kx_half=torch.ones(sx, device=device, dtype=dtype),
        by_half=zeros(sy),
        ay_half=zeros(sy),
        ky_half=torch.ones(sy, device=device, dtype=dtype),
        bz_half=zeros(sz),
        az_half=zeros(sz),
        kz_half=torch.ones(sz, device=device, dtype=dtype),
    )


def _prepare_execution(
    request: PropagationRequest,
    telemetry: ExecutionTelemetry | None = None,
) -> tuple[_ExecutionContext, tuple[Tensor, ...], tuple[int, ...]]:
    """Build padded coefficients, CPML, state, and packed execution indices."""
    acquisition = request.acquisition
    equation = request.equation
    model_reference = request.model[equation.declaration.required_model_fields[0]]
    boundary_config = request.config.boundary
    layers = boundary_config.layers if boundary_config.kind == "cpml" else 0
    padding = (
        (0 if boundary_config.free_surface else layers, layers),
        *((layers, layers),) * (equation.declaration.dimension - 1),
    )
    padded_shape = tuple(
        size + low + high
        for size, (low, high) in zip(acquisition.mesh_shape, padding)
    )
    offsets = tuple(low for low, _ in padding)
    source_indices = acquisition.source_indices + acquisition.source_indices.new_tensor(
        offsets
    )
    receiver_indices = (
        acquisition.receiver_indices
        + acquisition.receiver_indices.new_tensor(offsets)
    )
    padded_model = {
        name: _pad_model(request.model[name], padding)
        for name in equation.declaration.required_model_fields
    }

    configure_surface = getattr(equation, "configure_free_surface", None)
    if callable(configure_surface):
        configure_surface(boundary_config.free_surface, offsets[0])
    configure_source = getattr(equation, "configure_source_normalization", None)
    if callable(configure_source):
        cell_volume = 1.0
        for spacing in acquisition.spacing:
            cell_volume *= spacing
        configure_source(cell_volume)

    try:
        dt_max = float(equation.cfl_limit(padded_model, acquisition.spacing))
    except (
        KeyError,
        IndexError,
        TypeError,
        RuntimeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise WaveContractError(
            "Wave equation CFL evaluation failed",
            object_name=type(equation).__name__,
            field="cfl",
            expected="positive finite stability limit",
            actual="evaluation failed",
        ) from exc
    if not math.isfinite(dt_max) or dt_max <= 0.0:
        raise WaveContractError(
            "Wave equation CFL limit is invalid",
            object_name=type(equation).__name__,
            field="cfl",
            expected="positive finite stability limit",
            actual=dt_max,
        )
    if acquisition.dt > dt_max:
        message = (
            f"CFL violated: dt={acquisition.dt:.4e} > dt_max={dt_max:.4e}"
        )
        if request.config.discretization.strict_cfl:
            raise WaveNumericsError(
                message,
                object_name=type(equation).__name__,
                field="dt",
                expected=f"<= {dt_max}",
                actual=acquisition.dt,
            )
        warnings.warn(message + " (strict_cfl=False; continuing.)", stacklevel=3)

    vmax_method = getattr(equation, "max_velocity", None)
    try:
        vmax = float(vmax_method(padded_model)) if callable(vmax_method) else 1.0
    except (
        KeyError,
        IndexError,
        TypeError,
        RuntimeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise WaveContractError(
            "Wave equation maximum-velocity evaluation failed",
            object_name=type(equation).__name__,
            field="max_velocity",
            expected="positive finite velocity",
            actual="evaluation failed",
        ) from exc
    if not math.isfinite(vmax) or vmax <= 0.0:
        raise WaveContractError(
            "Wave equation maximum velocity is invalid",
            object_name=type(equation).__name__,
            field="max_velocity",
            expected="positive finite velocity",
            actual=vmax,
        )
    boundary: CPML | CPML3D
    if equation.declaration.dimension == 2:
        if boundary_config.kind == "cpml":
            boundary = build_cpml(
                padded_shape[0],
                padded_shape[1],
                layers,
                acquisition.spacing[0],
                acquisition.spacing[1],
                acquisition.dt,
                vmax,
                reflection=boundary_config.target_reflection,
                npower=boundary_config.profile_order,
                kappa_max=boundary_config.kappa_max,
                alpha_max=boundary_config.alpha_max,
                free_surface=boundary_config.free_surface,
                device=model_reference.device,
                dtype=model_reference.dtype,
            )
        else:
            boundary = _null_cpml_2d(
                cast(tuple[int, int], padded_shape),
                device=model_reference.device,
                dtype=model_reference.dtype,
            )
    else:
        if boundary_config.kind == "cpml":
            boundary = build_cpml_3d(
                padded_shape[0],
                padded_shape[1],
                padded_shape[2],
                layers,
                acquisition.spacing[0],
                acquisition.spacing[1],
                acquisition.spacing[2],
                acquisition.dt,
                vmax,
                reflection=boundary_config.target_reflection,
                npower=boundary_config.profile_order,
                kappa_max=boundary_config.kappa_max,
                alpha_max=boundary_config.alpha_max,
                free_surface=boundary_config.free_surface,
                device=model_reference.device,
                dtype=model_reference.dtype,
            )
        else:
            boundary = _null_cpml_3d(
                cast(tuple[int, int, int], padded_shape),
                device=model_reference.device,
                dtype=model_reference.dtype,
            )
    try:
        coefficients = dict(
            equation.prepare_model(
                padded_model,
                dt=acquisition.dt,
                spacing=acquisition.spacing,
            )
        )
        state = tuple(
            equation.initialize_state(
                acquisition.n_shot,
                padded_shape,
                device=model_reference.device,
                dtype=model_reference.dtype,
            )
        )
    except (KeyError, IndexError, TypeError, RuntimeError) as exc:
        raise _numerics_error(
            equation, "state", "initialized equation state", "setup failed"
        ) from exc
    expected_state = len(equation.declaration.state_fields) + len(
        equation.declaration.cpml_fields
    )
    if len(state) != expected_state:
        raise _numerics_error(equation, "state", expected_state, len(state))
    requested_illumination = request.config.output.illumination
    if requested_illumination:
        try:
            available_illumination = dict(
                equation.illumination_fields(state)
            )
        except (KeyError, IndexError, TypeError, RuntimeError) as exc:
            raise WaveCapabilityError(
                "Wave equation illumination capabilities are unavailable",
                object_name=type(equation).__name__,
                field="illumination",
                expected=requested_illumination,
                actual="capability inspection failed",
            ) from exc
        missing_illumination = tuple(
            name
            for name in requested_illumination
            if name not in available_illumination
        )
        if missing_illumination:
            raise WaveCapabilityError(
                "Wave equation does not provide requested illumination",
                object_name=type(equation).__name__,
                field="illumination",
                expected=tuple(available_illumination),
                actual=missing_illumination,
            )
    interior_slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, acquisition.mesh_shape)
    )
    context = _ExecutionContext(
        equation=equation,
        advance=maybe_compile(equation.advance),
        coefficients=coefficients,
        boundary=boundary,
        dt=acquisition.dt,
        spacing=acquisition.spacing,
        source_indices=source_indices,
        source_shot_index=acquisition.source_shot_index,
        receiver_indices=receiver_indices,
        receiver_shot_index=acquisition.receiver_shot_index,
        components=request.components,
        interior_slices=interior_slices,
        illumination=request.config.output.illumination,
        snapshot_policy=request.config.output.snapshot_policy,
        snapshot_indices=request.config.output.snapshot_indices,
        telemetry=telemetry or ExecutionTelemetry(request.acquisition.nt),
    )
    common_context_tensors = (
        *coefficients.values(),
        source_indices,
        acquisition.source_shot_index,
        receiver_indices,
        acquisition.receiver_shot_index,
        *(
            value
            for value in vars(boundary).values()
            if isinstance(value, Tensor)
        ),
    )
    for tensor in common_context_tensors:
        context.telemetry.observe_saved_tensor(tensor)
    context.telemetry.observe_live_state(state)
    return context, state, padded_shape


def _merge_collections(
    destination: dict[str, Tensor],
    source: Mapping[str, Tensor],
) -> None:
    """Merge segment outputs without changing selected-snapshot identity."""
    for name, value in source.items():
        if name.startswith("snapshot:"):
            if name in destination:
                raise WaveContractError(
                    "duplicate selected Wave snapshot",
                    object_name="EagerWaveBackend",
                    field="snapshot_indices",
                    expected="each requested time retained once",
                    actual=name,
                )
            destination[name] = value
        else:
            destination[name] = (
                value if name not in destination else destination[name] + value
            )


def _assemble_result(
    request: PropagationRequest,
    context: _ExecutionContext,
    state: tuple[Tensor, ...],
    records: Tensor,
    collections: Mapping[str, Tensor],
    telemetry: ExecutionTelemetry,
    *,
    diagnostics: Mapping[str, object] | None = None,
) -> PropagationResult:
    """Validate and publish one strategy's shared eager traversal outputs."""
    if not _all_finite((records, *state)):
        raise _numerics_error(
            request.equation,
            "result",
            "finite complete propagation result",
            "non-finite",
        )
    policy = request.config.output.snapshot_policy
    fields: dict[str, Tensor] = {}
    if policy == "final":
        fields.update(_collect_final_snapshot(context, state))
    elif policy == "selected":
        missing = tuple(
            index
            for index in request.config.output.snapshot_indices
            if f"snapshot:{index}" not in collections
        )
        if missing:
            raise _numerics_error(
                request.equation,
                "snapshot_indices",
                request.config.output.snapshot_indices,
                f"missing {missing}",
            )
        fields["wavefield"] = torch.stack(
            tuple(
                collections[f"snapshot:{index}"]
                for index in request.config.output.snapshot_indices
            )
        )
    elif policy == "energy":
        if "wavefield_energy" not in collections:
            raise _numerics_error(
                request.equation,
                "snapshot_policy",
                "complete wavefield energy",
                "missing",
            )
        fields["wavefield_energy"] = collections["wavefield_energy"]
    illumination = {
        name: collections[name]
        for name in context.illumination
        if name in collections
    }
    fields.update(_collect_illumination_results(context, illumination, state[0]))
    if not request.config.output.retain_field_gradients:
        fields = {name: value.detach() for name, value in fields.items()}
    result_diagnostics: dict[str, object] = {
        "backend": "eager",
        "dimension": request.equation.declaration.dimension,
        "survey_fingerprint": request.acquisition.survey_fingerprint,
        "memory_strategy": request.config.memory.strategy,
        "peak_live_state_scope": (
            "exact strategy-owned retained and explicitly observed live "
            "tensor storage; excludes framework-internal backward-replay "
            "temporaries"
        ),
    }
    if diagnostics is not None:
        result_diagnostics.update(diagnostics)
    return PropagationResult(
        traces=records,
        fields=fields,
        diagnostics=result_diagnostics,
        accounting=telemetry.snapshot(),
        complete=True,
        _telemetry=telemetry,
    )


class EagerWaveBackend:
    """Execute validated packed Wave requests with one eager time traversal."""

    name = "eager"

    def execute(self, request: PropagationRequest) -> PropagationResult:
        """Dispatch through the selected memory object exactly once."""
        if not isinstance(request, PropagationRequest):
            raise WaveContractError(
                "invalid eager Wave request",
                object_name=type(self).__name__,
                field="request",
                expected="PropagationRequest",
                actual=type(request).__name__,
            )
        return request.memory.execute(request, self)

    def prepare(
        self,
        request: PropagationRequest,
        telemetry: ExecutionTelemetry,
    ) -> tuple[_ExecutionContext, tuple[Tensor, ...], tuple[int, ...]]:
        """Prepare one already capability-validated memory execution."""
        return _prepare_execution(request, telemetry)

    def run_segment(
        self,
        context: _ExecutionContext,
        state: Sequence[Tensor],
        wavelets: Tensor,
        *,
        time_start: int,
    ) -> tuple[tuple[Tensor, ...], Tensor, dict[str, Tensor]]:
        """Expose the one dimension-neutral traversal primitive to strategies."""
        return _run_segment(
            context,
            state,
            wavelets,
            collect_illumination=True,
            time_start=time_start,
        )

    def assemble(
        self,
        request: PropagationRequest,
        context: _ExecutionContext,
        state: tuple[Tensor, ...],
        records: Tensor,
        collections: Mapping[str, Tensor],
        telemetry: ExecutionTelemetry,
        *,
        diagnostics: Mapping[str, object] | None = None,
    ) -> PropagationResult:
        """Assemble one complete strategy result with common validation."""
        return _assemble_result(
            request,
            context,
            state,
            records,
            collections,
            telemetry,
            diagnostics=diagnostics,
        )


__all__ = ["EagerWaveBackend"]
