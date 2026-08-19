"""Pure capability decisions for the experimental Wave native backend.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch

from ...contracts import PropagationRequest
from ...equations.acoustic import AcousticVelocityStress
from ...equations.acoustic3d import AcousticVelocityStress3D
from ...equations.elastic import ElasticVelocityStress
from ...equations.elastic3d import ElasticVelocityStress3D
from . import loader


@dataclass(frozen=True, slots=True)
class NativeCapabilityDecision:
    """One immutable native-support decision with structured error context."""

    supported: bool
    reason: str
    remediation: str
    field: str
    expected: object
    actual: object

    def __post_init__(self) -> None:
        """Reject decisions that cannot explain or remediate their outcome."""
        if (
            type(self.supported) is not bool
            or type(self.reason) is not str
            or not self.reason
            or type(self.remediation) is not str
            or not self.remediation
            or type(self.field) is not str
            or not self.field
        ):
            raise ValueError("invalid native capability decision")


class _NativeEquationProtocol(Protocol):
    """Additional equation member consumed only by native capability checks."""

    fd_order: int


_SUPPORTED_EQUATIONS: dict[type[object], tuple[int, tuple[tuple[str, ...], ...]]] = {
    AcousticVelocityStress: (
        2,
        (("pressure",), ("pressure", "vx", "vz")),
    ),
    ElasticVelocityStress: (
        2,
        (("pressure",), ("pressure", "vx", "vz")),
    ),
    AcousticVelocityStress3D: (3, (("pressure",),)),
}


def _unsupported(
    reason: str,
    remediation: str,
    *,
    field: str,
    expected: object,
    actual: object,
) -> NativeCapabilityDecision:
    return NativeCapabilityDecision(
        supported=False,
        reason=reason,
        remediation=remediation,
        field=field,
        expected=expected,
        actual=actual,
    )


def _source_layout_supported(request: PropagationRequest) -> bool:
    acquisition = request.acquisition
    if tuple(acquisition.source_shot_index.shape) != (acquisition.n_shot,):
        return False
    if any(
        int(actual) != expected
        for expected, actual in enumerate(acquisition.source_shot_index)
    ):
        return False
    if acquisition.n_trace % acquisition.n_shot:
        return False
    receivers_per_shot = acquisition.n_trace // acquisition.n_shot
    if tuple(acquisition.receiver_shot_index.shape) != (
        acquisition.n_trace,
    ):
        return False
    for shot in range(acquisition.n_shot):
        start = shot * receivers_per_shot
        stop = start + receivers_per_shot
        if any(
            int(actual) != shot
            for actual in acquisition.receiver_shot_index[start:stop]
        ):
            return False
    try:
        receivers = acquisition.receiver_indices.view(
            acquisition.n_shot,
            receivers_per_shot,
            acquisition.receiver_indices.shape[1],
        )
    except RuntimeError:
        return False
    base = receivers[0]
    return all(
        torch.equal(receivers[shot], base)
        for shot in range(1, acquisition.n_shot)
    )


def probe_native_capability(request: object) -> NativeCapabilityDecision:
    """Decide support without compiling, allocating state, or executing a step."""
    if not isinstance(request, PropagationRequest):
        return _unsupported(
            "native execution requires a validated PropagationRequest",
            "construct a PropagationRequest before selecting the native backend",
            field="request",
            expected="PropagationRequest",
            actual=type(request).__name__,
        )
    from .dispatch import NativeWaveBackend

    backend_type = type(request.backend)
    if backend_type is not NativeWaveBackend:
        return _unsupported(
            "request does not own the native backend",
            "construct the request with NativeWaveBackend",
            field="backend",
            expected="NativeWaveBackend",
            actual=backend_type.__name__,
        )
    equation_type: type[object] = type(request.equation)
    if equation_type is ElasticVelocityStress3D:
        return _unsupported(
            "native Elastic3D public-axis ABI has not been scientifically verified",
            "select backend='eager' until a native axis adapter is validated",
            field="equation",
            expected="verified public z-x-y component ABI",
            actual="legacy native z-y-x component ABI",
        )
    support = _SUPPORTED_EQUATIONS.get(equation_type)
    if support is None:
        return _unsupported(
            "equation has no complete native CUDA implementation",
            "select backend='eager' or use an isotropic acoustic/elastic equation",
            field="equation",
            expected="exact supported isotropic equation type",
            actual=type(request.equation).__name__,
        )
    dimension, component_sets = support
    native_equation = cast(_NativeEquationProtocol, request.equation)
    actual_dimension = request.equation.declaration.dimension
    if actual_dimension != dimension:
        return _unsupported(
            "equation dimension does not match its native implementation",
            "correct the equation declaration or select backend='eager'",
            field="dimension",
            expected=dimension,
            actual=actual_dimension,
        )
    if request.config.memory.strategy != "full":
        return _unsupported(
            "native CUDA execution supports only full memory",
            "set memory.strategy='full' or select backend='eager'",
            field="memory",
            expected="full",
            actual=request.config.memory.strategy,
        )
    if (
        request.config.discretization.fd_order
        != native_equation.fd_order
        or native_equation.fd_order > 16
    ):
        return _unsupported(
            "native finite-difference stencil exceeds or disagrees with its contract",
            "use a matching even fd_order no greater than 16",
            field="fd_order",
            expected="matching even order <= 16",
            actual=(
                request.config.discretization.fd_order,
                native_equation.fd_order,
            ),
        )
    if (
        equation_type is AcousticVelocityStress
        and request.config.boundary.free_surface
    ):
        return _unsupported(
            "acoustic 2-D native CUDA does not implement a free surface",
            "disable free_surface or select backend='eager'",
            field="free_surface",
            expected=False,
            actual=True,
        )
    if request.components not in component_sets:
        return _unsupported(
            "receiver component set has no complete native implementation",
            "request pressure only or the supported pressure/vx/vz tuple",
            field="components",
            expected=component_sets,
            actual=request.components,
        )
    if bool(
        getattr(request.equation, "_normalize_source_by_cell_volume", False)
    ):
        return _unsupported(
            "native source injection does not implement cell-volume normalization",
            "disable source normalization or select backend='eager'",
            field="source_normalization",
            expected=False,
            actual=True,
        )
    output = request.config.output
    if output.snapshot_policy not in ("none", "final"):
        return _unsupported(
            "native CUDA does not implement requested snapshot retention",
            "request no snapshots/final snapshot or select backend='eager'",
            field="snapshot_policy",
            expected=("none", "final"),
            actual=output.snapshot_policy,
        )
    if output.illumination:
        return _unsupported(
            "native CUDA does not implement illumination accumulation",
            "disable illumination or select backend='eager'",
            field="illumination",
            expected=(),
            actual=output.illumination,
        )
    if output.retain_field_gradients:
        return _unsupported(
            "native final wavefields are diagnostic and non-differentiable",
            "set retain_field_gradients=False or select backend='eager'",
            field="retain_field_gradients",
            expected=False,
            actual=True,
        )
    if not loader.is_available():
        return _unsupported(
            "CUDA runtime is unavailable",
            "select backend='eager' or run on a CUDA worker",
            field="device",
            expected="available CUDA runtime",
            actual="unavailable",
        )
    live_tensors = (
        *request.model.values(),
        request.wavelets,
        request.acquisition.source_indices,
        request.acquisition.receiver_indices,
    )
    if not all(loader.is_cuda_tensor(tensor) for tensor in live_tensors):
        return _unsupported(
            "native CUDA inputs are not all live on CUDA",
            "place model, wavelets, and acquisition indices on one CUDA device",
            field="device",
            expected="all native live tensors on CUDA",
            actual="non-CUDA live tensor",
        )
    reference = request.model[
        request.equation.declaration.required_model_fields[0]
    ]
    if reference.dtype not in (torch.float32, torch.float64):
        return _unsupported(
            "native CUDA kernels support float32 and float64 only",
            "use matching float32/float64 model and wavelet tensors",
            field="dtype",
            expected=("torch.float32", "torch.float64"),
            actual=str(reference.dtype),
        )
    if any(
        tensor.dtype is not reference.dtype
        for tensor in (*request.model.values(), request.wavelets)
    ):
        return _unsupported(
            "native model and wavelet dtypes disagree",
            "use one matching floating dtype for every live tensor",
            field="dtype",
            expected=str(reference.dtype),
            actual="mixed",
        )
    if not _source_layout_supported(request):
        return _unsupported(
            "native CUDA requires one source per shot and shared receivers",
            "use one source per shot with an identical receiver grid or select eager",
            field="source_layout",
            expected="one source per shot and shared receiver geometry",
            actual="packed irregular acquisition",
        )
    return NativeCapabilityDecision(
        supported=True,
        reason="native CUDA request is supported",
        remediation="none required",
        field="capability",
        expected="supported native request",
        actual="supported",
    )


__all__ = ["NativeCapabilityDecision", "probe_native_capability"]
