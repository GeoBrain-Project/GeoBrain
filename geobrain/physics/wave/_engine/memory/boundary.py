"""Boundary-saving custom VJP for supported reversible Wave equations.

The forward stores the complete exterior state band at every time and the
backward reconstructs the physical interior in reverse.  Coefficient gradients
in the outer replicated model rings remain an explicit approximation contract.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, cast

import torch
from torch import Tensor

from ...errors import WaveCapabilityError
from ..backends.eager import _ExecutionContext
from ..boundaries.rim import assemble, save_rim, set_rim
from ..contracts import (
    ExecutionTelemetry,
    EagerStrategyBackendProtocol,
    InverseStep2D,
    InverseStep3D,
    PropagationRequest,
    PropagationResult,
    RimSetter,
    WaveBackendProtocol,
    WaveEquationProtocol,
    WaveMemoryGuarantees,
)


_SUPPORTED_EQUATIONS = frozenset(
    {
        "acoustic-2d",
        "acoustic-3d",
        "elastic-2d",
        "elastic-3d",
        "elastic-vti-2d",
        "elastic-vti-3d",
        # This row has a dedicated gradient comparison in
        # test_boundary_memory_gradients.py; the 2-D TTI inverse remains unproven.
        "elastic-tti-3d",
    }
)


class _BoundaryEquationProtocol(WaveEquationProtocol, Protocol):
    """Reversible equation members consumed only by boundary reconstruction."""

    halo_width: int
    wavefield_names: tuple[str, ...]
    memory_names: tuple[str, ...]
    source_field: str
    snapshot_field: str

    def field_index(self, name: str) -> int: ...


def _unsupported(
    request: PropagationRequest,
    field: str,
    expected: object,
    actual: object,
) -> WaveCapabilityError:
    return WaveCapabilityError(
        "unsupported Wave boundary-saving combination",
        object_name=type(request.equation).__name__,
        field=field,
        expected=expected,
        actual=actual,
        hint="select full, checkpoint, or recursive memory",
    )


def _validate_boundary_request(request: PropagationRequest) -> None:
    """Reject every knowable unsupported row before eager preparation."""
    declaration = request.equation.declaration
    boundary = request.config.boundary
    output = request.config.output
    if declaration.identifier not in _SUPPORTED_EQUATIONS:
        raise _unsupported(
            request,
            "equation",
            tuple(sorted(_SUPPORTED_EQUATIONS)),
            declaration.identifier,
        )
    if boundary.kind != "cpml":
        raise _unsupported(request, "boundary.kind", "cpml", boundary.kind)
    if boundary.free_surface:
        raise _unsupported(request, "free_surface", False, True)
    halo_width = getattr(request.equation, "halo_width", None)
    if (
        isinstance(halo_width, bool)
        or not isinstance(halo_width, int)
        or boundary.layers < halo_width
    ):
        raise _unsupported(
            request,
            "boundary.layers",
            f">= equation halo width {halo_width}",
            boundary.layers,
        )
    if request.components != ("pressure",):
        raise _unsupported(
            request, "components", ("pressure",), request.components
        )
    if output.snapshot_policy not in ("none", "final"):
        raise _unsupported(
            request,
            "snapshot_policy",
            "'none' or 'final'",
            output.snapshot_policy,
        )
    if output.illumination:
        raise _unsupported(request, "illumination", (), output.illumination)
    equation: Any = request.equation
    try:
        inverse_step = equation.inverse_step
        wavefield_names = equation.wavefield_names
        memory_names = equation.memory_names
        source_field = equation.source_field
        snapshot_field = equation.snapshot_field
        field_index = equation.field_index
    except Exception as exc:
        raise _unsupported(
            request,
            "equation",
            "accessible reversible boundary-saving members",
            type(equation).__name__,
        ) from exc
    name_containers_valid = (
        type(wavefield_names) is tuple
        and type(memory_names) is tuple
    )
    all_names = (
        (*wavefield_names, *memory_names)
        if name_containers_valid
        else ()
    )
    structurally_valid = (
        callable(inverse_step)
        and name_containers_valid
        and bool(wavefield_names)
        and all(type(name) is str and name for name in wavefield_names)
        and all(type(name) is str and name for name in memory_names)
        and len(set(all_names)) == len(all_names)
        and wavefield_names == declaration.state_fields
        and memory_names == declaration.cpml_fields
        and type(source_field) is str
        and source_field in wavefield_names
        and type(snapshot_field) is str
        and snapshot_field in wavefield_names
        and callable(field_index)
    )
    if not structurally_valid:
        raise _unsupported(
            request,
            "equation",
            (
                "callable inverse_step and coherent wavefield, memory, "
                "source, snapshot, and field-index members"
            ),
            type(equation).__name__,
        )
    try:
        indices = tuple(field_index(name) for name in all_names)
    except Exception as exc:
        raise _unsupported(
            request,
            "equation",
            "field_index resolves every declared state member",
            type(equation).__name__,
        ) from exc
    if (
        any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(all_names)
            for index in indices
        )
        or len(set(indices)) != len(indices)
    ):
        raise _unsupported(
            request,
            "equation",
            f"unique field indices within [0, {len(all_names)})",
            indices,
        )


@dataclass(slots=True)
class _BoundaryExecution:
    """Tensor-free custom-autograd configuration and reconstruction metadata."""

    context: _ExecutionContext
    telemetry: ExecutionTelemetry
    coefficient_names: tuple[str, ...]
    wavefield_names: tuple[str, ...]
    wf_indices: tuple[int, ...]
    memory_names: tuple[str, ...]
    mem_indices: tuple[int, ...]
    n_state: int
    prepared_state: tuple[Tensor, ...]
    frame_mask: Tensor
    nt: int
    n_shot: int
    padded_shape: tuple[int, ...]


def _packed_samples(
    context: _ExecutionContext, state: Sequence[Tensor]
) -> Tensor:
    sampled = context.equation.sample_receivers(
        state,
        context.receiver_indices,
        context.receiver_shot_index,
        context.components,
    )
    return torch.stack(tuple(sampled[name] for name in context.components), dim=-1)


def _inverse(
    config: _BoundaryExecution,
    state: Sequence[Tensor],
    coefficients: Mapping[str, Tensor],
    set_saved_rim: RimSetter,
) -> list[Tensor]:
    equation = cast(_BoundaryEquationProtocol, config.context.equation)
    spacing = tuple(reversed(config.context.spacing))
    inverse_step = equation.inverse_step
    if inverse_step is None:
        raise RuntimeError("validated boundary equation has no inverse step")
    if len(spacing) == 2:
        inverse_step_2d = cast(InverseStep2D, inverse_step)
        return list(
            inverse_step_2d(
                state,
                coefficients,
                config.context.dt,
                spacing[0],
                spacing[1],
                set_saved_rim,
            )
        )
    inverse_step_3d = cast(InverseStep3D, inverse_step)
    return list(
        inverse_step_3d(
            state,
            coefficients,
            config.context.dt,
            spacing[0],
            spacing[1],
            spacing[2],
            set_saved_rim,
        )
    )


class _BoundarySaved(torch.autograd.Function):  # type: ignore[misc]
    @staticmethod
    def forward(
        ctx: Any,
        wavelets: Tensor,
        config: _BoundaryExecution,
        *coefficient_tensors: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context = config.context
        equation = cast(_BoundaryEquationProtocol, context.equation)
        coefficients = dict(
            zip(config.coefficient_names, coefficient_tensors)
        )
        with torch.no_grad():
            state = tuple(
                equation.initialize_state(
                    config.n_shot,
                    config.padded_shape,
                    device=wavelets.device,
                    dtype=wavelets.dtype,
                )
            )
            rims = {
                name: [save_rim(state[index], config.frame_mask)]
                for name, index in zip(
                    config.wavefield_names, config.wf_indices
                )
            }
            memories = {
                name: [save_rim(state[index], config.frame_mask)]
                for name, index in zip(
                    config.memory_names, config.mem_indices
                )
            }
            records: list[Tensor] = []
            for time_index in range(config.nt):
                state = tuple(
                    equation.advance(
                        state,
                        coefficients,
                        boundary=context.boundary,
                        dt=context.dt,
                        spacing=context.spacing,
                    )
                )
                config.telemetry.record_advance()
                state = tuple(
                    equation.inject_sources(
                        state,
                        context.source_indices,
                        context.source_shot_index,
                        wavelets[:, time_index],
                        dt=context.dt,
                    )
                )
                records.append(_packed_samples(context, state))
                for name, index in zip(
                    config.wavefield_names, config.wf_indices
                ):
                    rims[name].append(
                        save_rim(state[index], config.frame_mask)
                    )
                for name, index in zip(
                    config.memory_names, config.mem_indices
                ):
                    memories[name].append(
                        save_rim(state[index], config.frame_mask)
                    )
            final_wavefields = [
                state[index].clone() for index in config.wf_indices
            ]
            rim_stacks = [
                torch.stack(rims[name]) for name in config.wavefield_names
            ]
            memory_stacks = [
                torch.stack(memories[name]) for name in config.memory_names
            ]
            packed_records = torch.stack(records, dim=1)
            snapshot = equation.snapshot_fields(state)["wavefield"].detach()
        saved = (
            wavelets,
            *coefficient_tensors,
            *rim_stacks,
            *memory_stacks,
            *final_wavefields,
        )
        for tensor in saved:
            config.telemetry.observe_saved_tensor(tensor)
        config.telemetry.observe_live_state(
            (*config.prepared_state, *state)
        )
        ctx.boundary_execution = config
        ctx.save_for_backward(*saved)
        return packed_records, snapshot

    @staticmethod
    def backward(
        ctx: Any,
        gradient_records: Tensor,
        gradient_snapshot: Tensor | None,
    ) -> tuple[object, ...]:
        config: _BoundaryExecution = ctx.boundary_execution
        context = config.context
        equation = cast(_BoundaryEquationProtocol, context.equation)
        coefficient_count = len(config.coefficient_names)
        wavefield_count = len(config.wavefield_names)
        memory_count = len(config.memory_names)
        saved = ctx.saved_tensors
        wavelets = saved[0]
        coefficient_tensors = list(saved[1 : 1 + coefficient_count])
        rim_start = 1 + coefficient_count
        rim_stacks = saved[rim_start : rim_start + wavefield_count]
        memory_start = rim_start + wavefield_count
        memory_stacks = saved[memory_start : memory_start + memory_count]
        final_wavefields = list(saved[memory_start + memory_count :])
        rim_map = dict(zip(config.wavefield_names, rim_stacks))
        memory_map = dict(zip(config.memory_names, memory_stacks))
        coefficients = dict(
            zip(config.coefficient_names, coefficient_tensors)
        )

        coefficient_gradients = [
            torch.zeros_like(coefficient) for coefficient in coefficient_tensors
        ]
        needs_wavelets = ctx.needs_input_grad[0]
        wavelet_gradient = (
            torch.zeros_like(wavelets) if needs_wavelets else None
        )
        wavefield_cotangents = [
            torch.zeros_like(field) for field in final_wavefields
        ]
        memory_cotangents = [
            torch.zeros_like(final_wavefields[0])
            for _ in range(memory_count)
        ]
        if gradient_snapshot is not None:
            snapshot_name = getattr(
                equation, "snapshot_field", equation.source_field
            )
            wavefield_cotangents[
                config.wf_indices.index(
                    equation.field_index(snapshot_name)
                )
            ] = gradient_snapshot
        current = assemble(config, final_wavefields)

        for time_index in range(config.nt - 1, -1, -1):
            with torch.no_grad():
                without_source = equation.inject_sources(
                    current,
                    context.source_indices,
                    context.source_shot_index,
                    -wavelets[:, time_index],
                    dt=context.dt,
                )

                def restore(
                    field: Tensor,
                    name: str,
                    index: int = time_index,
                ) -> Tensor:
                    return set_rim(
                        field,
                        config.frame_mask,
                        rim_map[name][index],
                    )

                previous = _inverse(
                    config, without_source, coefficients, restore
                )

            with torch.enable_grad():
                wavefield_inputs = [
                    previous[index].detach().requires_grad_(True)
                    for index in config.wf_indices
                ]
                memory_inputs = [
                    set_rim(
                        torch.zeros_like(previous[config.wf_indices[0]]),
                        config.frame_mask,
                        memory_map[name][time_index],
                    ).requires_grad_(True)
                    for name in config.memory_names
                ]
                full_input = assemble(
                    config, wavefield_inputs, memory_inputs
                )
                coefficient_inputs = [
                    value.detach().requires_grad_(True)
                    for value in coefficient_tensors
                ]
                local_coefficients = dict(
                    zip(config.coefficient_names, coefficient_inputs)
                )
                wavelet_input = wavelets[:, time_index].detach()
                if needs_wavelets:
                    wavelet_input.requires_grad_(True)
                output = equation.advance(
                    full_input,
                    local_coefficients,
                    boundary=context.boundary,
                    dt=context.dt,
                    spacing=context.spacing,
                )
                config.telemetry.record_recomputed_advance()
                output = equation.inject_sources(
                    output,
                    context.source_indices,
                    context.source_shot_index,
                    wavelet_input,
                    dt=context.dt,
                )
                record = _packed_samples(context, output)
                wavefield_outputs = [
                    output[index] for index in config.wf_indices
                ]
                memory_outputs = [
                    output[index] for index in config.mem_indices
                ]
                inputs = (
                    *wavefield_inputs,
                    *memory_inputs,
                    *coefficient_inputs,
                    *((wavelet_input,) if needs_wavelets else ()),
                )
                gradients = torch.autograd.grad(
                    (*wavefield_outputs, *memory_outputs, record),
                    inputs,
                    grad_outputs=(
                        *wavefield_cotangents,
                        *memory_cotangents,
                        gradient_records[:, time_index],
                    ),
                    allow_unused=True,
                )
            references = list(inputs)
            gradients = tuple(
                torch.zeros_like(reference)
                if gradient is None
                else gradient
                for gradient, reference in zip(gradients, references)
            )
            wavefield_cotangents = list(gradients[:wavefield_count])
            memory_cotangents = list(
                gradients[wavefield_count : wavefield_count + memory_count]
            )
            coefficient_start = wavefield_count + memory_count
            for index in range(coefficient_count):
                coefficient_gradients[index] = (
                    coefficient_gradients[index]
                    + gradients[coefficient_start + index]
                )
            if needs_wavelets:
                assert wavelet_gradient is not None
                wavelet_gradient[:, time_index] = gradients[-1]
            current = previous

        return wavelet_gradient, None, *coefficient_gradients


class BoundaryMemory:
    """Custom-VJP exterior-band boundary saving with an explicit support table."""

    guarantees = WaveMemoryGuarantees(
        strategy="boundary",
        supports_autograd=True,
        preserves_forward_values=True,
    )

    def execute(
        self,
        request: PropagationRequest,
        backend: WaveBackendProtocol,
    ) -> PropagationResult:
        """Validate capabilities, run custom VJP, and publish approximation metadata."""
        _validate_boundary_request(request)
        eager = cast(EagerStrategyBackendProtocol, backend)
        telemetry = ExecutionTelemetry(request.acquisition.nt)
        context, initial_state, padded_shape = eager.prepare(
            request, telemetry
        )
        equation = cast(_BoundaryEquationProtocol, context.equation)
        frame_mask = torch.ones(
            padded_shape,
            dtype=torch.bool,
            device=request.wavelets.device,
        )
        frame_mask[context.interior_slices] = False
        config = _BoundaryExecution(
            context=context,
            telemetry=telemetry,
            coefficient_names=tuple(context.coefficients),
            wavefield_names=tuple(equation.wavefield_names),
            wf_indices=tuple(
                equation.field_index(name) for name in equation.wavefield_names
            ),
            memory_names=tuple(equation.memory_names),
            mem_indices=tuple(
                equation.field_index(name) for name in equation.memory_names
            ),
            n_state=len(initial_state),
            prepared_state=initial_state,
            frame_mask=frame_mask,
            nt=request.acquisition.nt,
            n_shot=request.acquisition.n_shot,
            padded_shape=padded_shape,
        )
        telemetry.observe_saved_tensor(frame_mask)
        coefficient_tensors = tuple(
            context.coefficients[name] for name in config.coefficient_names
        )
        try:
            records, snapshot = _BoundarySaved.apply(
                request.wavelets,
                config,
                *coefficient_tensors,
            )
        finally:
            config.prepared_state = ()
        final_state = list(initial_state)
        snapshot_name = getattr(
            equation, "snapshot_field", equation.source_field
        )
        final_state[equation.field_index(snapshot_name)] = snapshot
        return eager.assemble(
            request,
            context,
            tuple(final_state),
            records,
            {},
            telemetry,
            diagnostics={
                "outer_model_gradient_rings": "approximate",
                "boundary_reconstruction": "custom_vjp",
            },
        )


__all__ = ["BoundaryMemory"]
