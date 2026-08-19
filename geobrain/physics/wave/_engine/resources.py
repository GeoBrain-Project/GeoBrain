"""Deterministic Wave resource estimation and budget enforcement.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch

from ..errors import WaveContractError, WaveResourceError
from .contracts import PropagationRequest


_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _RuntimeEnvironmentKey:
    """Exact execution environment addressed by one runtime calibration."""

    platform_system: str
    platform_version: str
    machine_architecture: str
    torch_version: str
    device_type: str

    def to_dict(self) -> dict[str, str]:
        """Return ordered JSON data for diagnostics and registry discovery."""
        return {
            "platform_system": self.platform_system,
            "platform_version": self.platform_version,
            "machine_architecture": self.machine_architecture,
            "torch_version": self.torch_version,
            "device_type": self.device_type,
        }


@dataclass(frozen=True, slots=True)
class _RuntimeCalibration:
    """Measured autograd runtime envelope for one exact environment."""

    environment: _RuntimeEnvironmentKey
    autograd_runtime_envelope_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Return ordered JSON data without exposing the private record type."""
        return {
            **self.environment.to_dict(),
            "autograd_runtime_envelope_bytes": self.autograd_runtime_envelope_bytes,
        }


# Each row is the 256 MiB power-of-two ceiling above an independent six-row
# fresh-process measurement grid taken on that exact environment (the wave
# memory-calibration record keeps the procedure and every row's parameters,
# including per-row CPML width for the torch 2.13 re-measurement). CUDA and
# every other OS/architecture/Torch tuple remain an external
# device-benchmark evidence item.
_V0_2_0_RUNTIME_CALIBRATIONS = (
    _RuntimeCalibration(
        environment=_RuntimeEnvironmentKey(
            platform_system="Darwin",
            platform_version="26.5.1",
            machine_architecture="arm64",
            torch_version="2.10.0",
            device_type="cpu",
        ),
        autograd_runtime_envelope_bytes=256 * 1024 * 1024,
    ),
    # Re-measured on the same host after the torch 2.10 -> 2.13 upgrade;
    # grid max incremental 198,017,024 bytes -> same 256 MiB ceiling.
    _RuntimeCalibration(
        environment=_RuntimeEnvironmentKey(
            platform_system="Darwin",
            platform_version="26.5.1",
            machine_architecture="arm64",
            torch_version="2.13.0",
            device_type="cpu",
        ),
        autograd_runtime_envelope_bytes=256 * 1024 * 1024,
    ),
)


def _local_runtime_environment(device_type: str) -> _RuntimeEnvironmentKey:
    """Identify the exact local runtime tuple used by estimator discovery."""
    system = platform.system()
    version = platform.mac_ver()[0] if system == "Darwin" else platform.release()
    return _RuntimeEnvironmentKey(
        platform_system=system,
        platform_version=version,
        machine_architecture=platform.machine(),
        torch_version=str(torch.__version__),
        device_type=device_type,
    )


def _runtime_calibration(device_type: str) -> _RuntimeCalibration | None:
    """Return the exact local calibration, never a cross-environment fallback."""
    environment = _local_runtime_environment(device_type)
    return next(
        (
            calibration
            for calibration in _V0_2_0_RUNTIME_CALIBRATIONS
            if calibration.environment == environment
        ),
        None,
    )


def runtime_calibration_registry() -> tuple[dict[str, object], ...]:
    """Return deterministic JSON-ready 0.2.0 runtime calibration records."""
    return tuple(calibration.to_dict() for calibration in _V0_2_0_RUNTIME_CALIBRATIONS)


def autograd_resource_estimate_supported(*, device_type: str = "cpu") -> bool:
    """Report whether the exact local autograd runtime has measured evidence."""
    return _runtime_calibration(device_type) is not None


def runtime_calibration_remediation(*, budget_enforcement: bool) -> str:
    """Return the exact execution or direct-estimation calibration remedy."""
    if budget_enforcement:
        return (
            "remove memory.budget_bytes or register the exact measured runtime calibration "
            "for this environment"
        )
    return (
        "direct autograd resource estimation is unavailable until the exact "
        "measured runtime calibration for this environment is registered; ordinary "
        "execution remains available separately with memory.budget_bytes=None"
    )


def _require_runtime_calibration(
    device_type: str,
    *,
    object_name: str = "estimate_resources",
    budget_enforcement: bool = False,
) -> _RuntimeCalibration:
    """Fail loudly instead of returning an uncalibrated structural-only bound."""
    calibration = _runtime_calibration(device_type)
    if calibration is None:
        raise WaveResourceError(
            (
                "Wave budget runtime calibration is unavailable"
                if budget_enforcement
                else "Wave direct autograd runtime calibration is unavailable"
            ),
            object_name=object_name,
            field="runtime_calibration",
            expected="an exact registered OS, architecture, Torch, and device calibration",
            actual=_local_runtime_environment(device_type).to_dict(),
            hint=runtime_calibration_remediation(
                budget_enforcement=budget_enforcement,
            ),
        )
    return calibration


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Conservative byte estimate partitioned by allocation purpose."""

    model_coefficients_bytes: int
    live_fields_bytes: int
    cpml_memories_bytes: int
    saved_history_bytes: int
    output_bytes: int
    snapshots_bytes: int
    autograd_overhead_bytes: int
    runtime_overhead_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        """Require exact non-negative integer partition accounting."""
        parts = (
            self.model_coefficients_bytes,
            self.live_fields_bytes,
            self.cpml_memories_bytes,
            self.saved_history_bytes,
            self.output_bytes,
            self.snapshots_bytes,
            self.autograd_overhead_bytes,
            self.runtime_overhead_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*parts, self.total_bytes)
        ) or self.total_bytes != sum(parts):
            raise WaveContractError(
                "invalid Wave resource estimate",
                object_name=type(self).__name__,
                field="total_bytes",
                expected="sum of non-negative integer byte partitions",
                actual=self.total_bytes,
            )


def estimate_resources(request: PropagationRequest, *, autograd_enabled: bool) -> ResourceEstimate:
    """Estimate a conservative strategy-specific execution high-water bound."""
    if type(autograd_enabled) is not bool:
        raise WaveContractError(
            "invalid Wave resource estimation flag",
            object_name="estimate_resources",
            field="autograd_enabled",
            expected="bool",
            actual=type(autograd_enabled).__name__,
        )
    reference = request.model[request.equation.declaration.required_model_fields[0]]
    runtime_calibration = (
        _require_runtime_calibration(reference.device.type) if autograd_enabled else None
    )
    itemsize = reference.element_size()
    cells = reference.numel()
    n_shot = request.acquisition.n_shot
    declaration = request.equation.declaration
    boundary = request.config.boundary
    layers = boundary.layers if boundary.kind == "cpml" else 0
    padding = (
        (0 if boundary.free_surface else layers, layers),
        *((layers, layers),) * (declaration.dimension - 1),
    )
    padded_shape = tuple(
        size + low + high for size, (low, high) in zip(request.acquisition.mesh_shape, padding)
    )
    padded_cells = math.prod(padded_shape)
    state_field_count = len(declaration.state_fields)
    cpml_field_count = len(declaration.cpml_fields)
    persistent_field_count = state_field_count + cpml_field_count
    required_model_count = len(declaration.required_model_fields)

    live_model = (
        sum(request.model[name].numel() for name in declaration.required_model_fields) * itemsize
    )
    padded_models = required_model_count * padded_cells * itemsize
    # Every supported equation prepares no more dense coefficient tensors than
    # its declared persistent state-plus-CPML field set.  Using that explicit
    # declaration-derived bound covers the wider TTI and viscoelastic maps
    # without an equation-name table or an empirical multiplier.
    prepared_coefficient_count = persistent_field_count
    prepared_coefficients = prepared_coefficient_count * padded_cells * itemsize
    boundary_coefficients = 6 * sum(padded_shape) * itemsize
    acquisition_indices = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            request.acquisition.source_indices,
            request.acquisition.receiver_indices,
            request.acquisition.source_shot_index,
            request.acquisition.receiver_shot_index,
        )
    )
    padded_coordinate_indices = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            request.acquisition.source_indices,
            request.acquisition.receiver_indices,
        )
    )
    wavelet_storage = request.wavelets.numel() * request.wavelets.element_size()
    model_coefficients = (
        live_model
        + padded_models
        + prepared_coefficients
        + boundary_coefficients
        + acquisition_indices
        + padded_coordinate_indices
        + wavelet_storage
    )
    live_fields = state_field_count * n_shot * padded_cells * itemsize
    cpml_memories = cpml_field_count * n_shot * padded_cells * itemsize
    persistent_fields = live_fields + cpml_memories
    nt = request.acquisition.nt
    full_traversal_storage = persistent_fields * nt
    strategy = request.config.memory.strategy
    if strategy == "full":
        saved_history = full_traversal_storage
    elif strategy == "checkpoint":
        checkpoint_count = min(request.config.memory.checkpoint_segments, nt)
        saved_history = (checkpoint_count + 1) * persistent_fields
    elif strategy == "recursive":
        leaf_count = (
            nt + request.config.memory.recursive_leaf_steps - 1
        ) // request.config.memory.recursive_leaf_steps
        recursion_depth = max(0, (leaf_count - 1).bit_length())
        # This tier counts persistent recursion checkpoints only.  The
        # autograd reserve below separately bounds the retained graph inside a
        # leaf, including the degenerate leaf_steps == nt traversal.
        saved_history = (2 + recursion_depth) * persistent_fields
    else:
        rim_cells = padded_cells - cells
        saved_rims = persistent_field_count * n_shot * rim_cells * (nt + 1) * itemsize
        final_wavefield_clones = state_field_count * n_shot * padded_cells * itemsize
        saved_history = saved_rims + final_wavefield_clones + persistent_fields
    output = (
        request.acquisition.n_trace * request.acquisition.nt * len(request.components) * itemsize
    )
    policy = request.config.output.snapshot_policy
    if policy == "none":
        snapshot_count = 0
    elif policy == "selected":
        snapshot_count = len(request.config.output.snapshot_indices)
    else:
        snapshot_count = 1
    snapshots = snapshot_count * n_shot * cells * itemsize
    snapshots += len(request.config.output.illumination) * cells * itemsize
    if strategy == "boundary":
        snapshots += padded_cells
    # One differentiated step can retain one complete state-width of graph
    # intermediates plus one conservative prepared-coefficient width.  Full
    # history already owns the state-width at every time, so its additional
    # graph tier needs only the coefficient width.  Replay strategies instead
    # reserve the complete step-graph width for their largest simultaneously
    # differentiated interval.  Boundary reconstruction differentiates one
    # reconstructed step at a time.
    step_graph_storage = persistent_fields + prepared_coefficients
    if not autograd_enabled:
        autograd_overhead = 0
    elif strategy == "full":
        autograd_overhead = nt * prepared_coefficients
    elif strategy == "checkpoint":
        segment_count = min(
            request.config.memory.checkpoint_segments,
            nt,
        )
        maximum_segment_steps = (nt + segment_count - 1) // segment_count
        autograd_overhead = maximum_segment_steps * step_graph_storage
    elif strategy == "recursive":
        leaf_steps = min(
            request.config.memory.recursive_leaf_steps,
            nt,
        )
        autograd_overhead = leaf_steps * step_graph_storage
    else:
        autograd_overhead = step_graph_storage
    structural_parts = (
        model_coefficients,
        live_fields,
        cpml_memories,
        saved_history,
        output,
        snapshots,
        autograd_overhead,
    )
    structural_total = sum(structural_parts)
    runtime_overhead = 0
    if runtime_calibration is not None:
        runtime_overhead = max(
            0,
            runtime_calibration.autograd_runtime_envelope_bytes - structural_total,
        )
    parts = (*structural_parts, runtime_overhead)
    return ResourceEstimate(
        model_coefficients_bytes=model_coefficients,
        live_fields_bytes=live_fields,
        cpml_memories_bytes=cpml_memories,
        saved_history_bytes=saved_history,
        output_bytes=output,
        snapshots_bytes=snapshots,
        autograd_overhead_bytes=autograd_overhead,
        runtime_overhead_bytes=runtime_overhead,
        total_bytes=sum(parts),
    )


def allocate_with_budget(
    request: PropagationRequest,
    allocation_hook: Callable[[], _T],
    *,
    autograd_enabled: bool,
) -> _T:
    """Invoke an allocator only after the request's resource budget passes."""
    budget = request.config.memory.budget_bytes
    if budget is None:
        return allocation_hook()
    reference = request.model[request.equation.declaration.required_model_fields[0]]
    _require_runtime_calibration(
        reference.device.type,
        object_name="allocate_with_budget",
        budget_enforcement=True,
    )
    estimate = estimate_resources(request, autograd_enabled=autograd_enabled)
    if estimate.total_bytes > budget:
        raise WaveResourceError(
            "Wave resource estimate exceeds configured budget",
            object_name="allocate_with_budget",
            field="budget_bytes",
            expected=f">= {estimate.total_bytes}",
            actual=budget,
        )
    return allocation_hook()


__all__ = [
    "ResourceEstimate",
    "allocate_with_budget",
    "autograd_resource_estimate_supported",
    "estimate_resources",
    "runtime_calibration_remediation",
    "runtime_calibration_registry",
]
