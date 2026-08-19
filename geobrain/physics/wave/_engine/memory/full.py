"""Full-autograd Wave memory execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from typing import cast

from ..contracts import (
    EagerStrategyBackendProtocol,
    ExecutionTelemetry,
    PropagationRequest,
    PropagationResult,
    WaveBackendProtocol,
    WaveMemoryGuarantees,
)


class FullMemory:
    """Retain the ordinary eager autograd graph for the complete traversal."""

    guarantees = WaveMemoryGuarantees(
        strategy="full",
        supports_autograd=True,
        preserves_forward_values=True,
    )

    def execute(
        self,
        request: PropagationRequest,
        backend: WaveBackendProtocol,
    ) -> PropagationResult:
        """Run one full graph while measuring actual saved tensor storages."""
        eager = cast(EagerStrategyBackendProtocol, backend)
        telemetry = ExecutionTelemetry(request.acquisition.nt)
        context, state, _ = eager.prepare(request, telemetry)
        with torch.autograd.graph.saved_tensors_hooks(
            telemetry.observe_saved_tensor,
            lambda tensor: tensor,
        ):
            state, records, collections = eager.run_segment(
                context,
                state,
                request.wavelets,
                time_start=0,
            )
        return eager.assemble(
            request,
            context,
            state,
            records,
            collections,
            telemetry,
        )


__all__ = ["FullMemory"]
