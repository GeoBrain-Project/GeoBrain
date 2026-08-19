"""Native Wave request preparation and result assembly.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ....errors import WaveCapabilityError
from ...contracts import (
    ExecutionTelemetry,
    PropagationRequest,
    PropagationResult,
)
from ..eager import _assemble_result, _prepare_execution
from .loader import NativeExtensionName


def prepare_native_execution(
    request: PropagationRequest,
) -> tuple[Any, tuple[torch.Tensor, ...], ExecutionTelemetry]:
    """Prepare one supported request without running the eager traversal."""
    telemetry = ExecutionTelemetry(request.acquisition.nt)
    context, state, _ = _prepare_execution(request, telemetry)
    return context, state, telemetry


def native_receiver_coordinates(context: Any) -> tuple[torch.Tensor, ...]:
    """Return the one shared receiver grid accepted by native capability checks."""
    n_shot = int(context.source_shot_index.numel())
    receivers_per_shot = int(context.receiver_indices.shape[0]) // n_shot
    return tuple(
        context.receiver_indices[:receivers_per_shot, axis]
        for axis in range(context.receiver_indices.shape[1])
    )


def assemble_native_result(
    request: PropagationRequest,
    context: Any,
    state: Sequence[torch.Tensor],
    dense_records: torch.Tensor,
    telemetry: ExecutionTelemetry,
    *,
    native_extension: NativeExtensionName,
) -> PropagationResult:
    """Pack native dense records and publish one complete native result."""
    n_shot = request.acquisition.n_shot
    nt = request.acquisition.nt
    n_component = len(request.components)
    n_trace = request.acquisition.n_trace
    receivers_per_shot = n_trace // n_shot
    dense_shape: tuple[int, ...] = (n_shot, nt, receivers_per_shot)
    if n_component != 1:
        dense_shape = (*dense_shape, n_component)
    reference = request.wavelets
    if (
        not isinstance(dense_records, torch.Tensor)
        or tuple(dense_records.shape) != dense_shape
        or dense_records.dtype is not reference.dtype
        or dense_records.device != reference.device
        or not bool(torch.isfinite(dense_records).all())
    ):
        raise WaveCapabilityError(
            "native CUDA extension returned invalid dense records",
            object_name="NativeWaveBackend",
            field="records",
            expected=(dense_shape, reference.dtype, reference.device, "finite"),
            actual=(
                tuple(dense_records.shape),
                dense_records.dtype,
                dense_records.device,
            ),
            hint="verify the native extension ABI or select backend='eager'",
        )
    if n_component == 1:
        packed = (
            dense_records.permute(0, 2, 1)
            .reshape(n_trace, nt)
            .unsqueeze(-1)
        )
    else:
        packed = dense_records.permute(0, 2, 1, 3).reshape(
            n_trace, nt, n_component
        )
    packed_shape = (n_trace, nt, n_component)
    if (
        tuple(packed.shape) != packed_shape
        or packed.dtype is not reference.dtype
        or packed.device != reference.device
        or not bool(torch.isfinite(packed).all())
    ):
        raise WaveCapabilityError(
            "native CUDA records could not be packed safely",
            object_name="NativeWaveBackend",
            field="records",
            expected=(packed_shape, reference.dtype, reference.device, "finite"),
            actual=(tuple(packed.shape), packed.dtype, packed.device),
            hint="verify the native extension ABI or select backend='eager'",
        )
    for _ in range(request.acquisition.nt):
        telemetry.record_advance()
    final_state = tuple(state)
    telemetry.observe_live_state(final_state)
    return _assemble_result(
        request,
        context,
        final_state,
        packed,
        {},
        telemetry,
        diagnostics={
            "backend": "native",
            "native_extension": native_extension,
            "native_accounting_complete": False,
            "native_accounting_scope": (
                "forward work plus shared preparation and published state; "
                "excludes native autograd histories, checkpoints, backward "
                "scratch, and replay"
            ),
            "peak_live_state_scope": (
                "shared preparation and published state only; excludes native "
                "autograd histories, checkpoints, and backward scratch"
            ),
        },
    )


__all__ = [
    "assemble_native_result",
    "native_receiver_coordinates",
    "prepare_native_execution",
]
