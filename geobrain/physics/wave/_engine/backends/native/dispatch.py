"""Exhaustive dispatch and fail-loud execution for Wave native CUDA.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ....errors import WaveCapabilityError
from ...contracts import PropagationRequest, PropagationResult
from ...equations.acoustic import AcousticVelocityStress
from ...equations.acoustic3d import AcousticVelocityStress3D
from ...equations.elastic import ElasticVelocityStress
from ...equations.elastic3d import ElasticVelocityStress3D
from . import loader
from .acoustic import execute_acoustic2d, execute_acoustic3d
from .capabilities import NativeCapabilityDecision, probe_native_capability
from .elastic import execute_elastic2d, execute_elastic3d


NativeExecutor = Callable[[PropagationRequest, Any], PropagationResult]


@dataclass(frozen=True, slots=True)
class NativeDispatchEntry:
    """One exact equation/dimension native handler and extension."""

    extension: loader.NativeExtensionName
    execute: NativeExecutor


NATIVE_DISPATCH_TABLE: dict[
    tuple[int, type[object]], NativeDispatchEntry
] = {
    (2, AcousticVelocityStress): NativeDispatchEntry(
        "acoustic2d", execute_acoustic2d
    ),
    (3, AcousticVelocityStress3D): NativeDispatchEntry(
        "acoustic3d", execute_acoustic3d
    ),
    (2, ElasticVelocityStress): NativeDispatchEntry(
        "elastic2d", execute_elastic2d
    ),
    (3, ElasticVelocityStress3D): NativeDispatchEntry(
        "elastic3d", execute_elastic3d
    ),
}


def _capability_error(
    decision: NativeCapabilityDecision,
) -> WaveCapabilityError:
    return WaveCapabilityError(
        decision.reason,
        object_name="NativeWaveBackend",
        field=decision.field,
        expected=decision.expected,
        actual=decision.actual,
        hint=decision.remediation,
    )


class NativeWaveBackend:
    """Execute only complete, explicitly supported native CUDA requests."""

    name = "native"

    def execute(self, request: PropagationRequest) -> PropagationResult:
        """Return one complete native result or raise a structured capability error."""
        decision = probe_native_capability(request)
        if not decision.supported:
            raise _capability_error(decision)
        key = (
            request.equation.declaration.dimension,
            type(request.equation),
        )
        entry = NATIVE_DISPATCH_TABLE.get(key)
        if entry is None:
            raise WaveCapabilityError(
                "native dispatch has no exact equation handler",
                object_name=type(self).__name__,
                field="dispatch",
                expected=tuple(
                    (dimension, equation.__name__)
                    for dimension, equation in NATIVE_DISPATCH_TABLE
                ),
                actual=(key[0], key[1].__name__),
                hint="select backend='eager' or install a matching native handler",
            )
        try:
            extension = loader.load_native_extension(entry.extension)
        except Exception as exc:
            raise WaveCapabilityError(
                "native CUDA extension could not be loaded",
                object_name=type(self).__name__,
                field="extension",
                expected=f"loadable {entry.extension} extension",
                actual=type(exc).__name__,
                hint="verify CUDA/PyTorch compiler compatibility or select eager",
            ) from exc
        try:
            result = entry.execute(request, extension)
        except WaveCapabilityError:
            raise
        except Exception as exc:
            raise WaveCapabilityError(
                "native CUDA execution did not complete",
                object_name=type(self).__name__,
                field="runtime",
                expected="one complete native PropagationResult",
                actual=type(exc).__name__,
                hint="inspect the chained native failure or select backend='eager'",
            ) from exc
        if (
            not isinstance(result, PropagationResult)
            or not result.complete
            or result.diagnostics.get("backend") != "native"
        ):
            raise WaveCapabilityError(
                "native CUDA handler returned an incomplete result",
                object_name=type(self).__name__,
                field="result",
                expected="complete PropagationResult with backend='native'",
                actual=type(result).__name__,
            )
        return result


__all__ = [
    "NATIVE_DISPATCH_TABLE",
    "NativeDispatchEntry",
    "NativeWaveBackend",
]
