"""Uniform non-reentrant checkpoint Wave memory execution.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from typing import cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from ..backends.eager import _ExecutionContext, _merge_collections
from ..contracts import (
    ExecutionTelemetry,
    EagerStrategyBackendProtocol,
    PropagationRequest,
    PropagationResult,
    WaveBackendProtocol,
    WaveMemoryGuarantees,
)


def _collection_names(
    context: _ExecutionContext,
    *,
    start: int,
    stop: int,
) -> tuple[str, ...]:
    names: list[str] = []
    if context.snapshot_policy == "selected":
        names.extend(
            f"snapshot:{index}"
            for index in context.snapshot_indices
            if start <= index < stop
        )
    if context.snapshot_policy == "energy":
        names.append("wavefield_energy")
    names.extend(context.illumination)
    return tuple(names)


def _run_checkpoint_segment(
    backend: EagerStrategyBackendProtocol,
    context: _ExecutionContext,
    state: tuple[Tensor, ...],
    wavelets: Tensor,
    *,
    start: int,
) -> tuple[tuple[Tensor, ...], Tensor, dict[str, Tensor]]:
    """Checkpoint one interval and return its exact retained outputs."""
    coefficient_names = tuple(context.coefficients)
    coefficient_tensors = tuple(
        context.coefficients[name] for name in coefficient_names
    )
    n_state = len(state)
    names = _collection_names(
        context,
        start=start,
        stop=start + int(wavelets.shape[1]),
    )

    def segment(*tensors: Tensor) -> tuple[Tensor, ...]:
        local_coefficients = dict(
            zip(
                coefficient_names,
                tensors[n_state : n_state + len(coefficient_names)],
            )
        )
        local_context = replace(context, coefficients=local_coefficients)
        new_state, records, collections = backend.run_segment(
            local_context,
            tensors[:n_state],
            tensors[-1],
            time_start=start,
        )
        return (*new_state, records, *(collections[name] for name in names))

    def checkpoint_contexts() -> tuple[
        AbstractContextManager[None],
        AbstractContextManager[None],
    ]:
        return nullcontext(), context.telemetry.recompute_region()

    with set_checkpoint_early_stop(False):
        output = checkpoint(
            segment,
            *state,
            *coefficient_tensors,
            wavelets,
            use_reentrant=False,
            context_fn=checkpoint_contexts,
        )
    collection_start = n_state + 1
    collections = dict(zip(names, output[collection_start:]))
    return tuple(output[:n_state]), output[n_state], collections


class CheckpointMemory:
    """Retain uniform segment boundaries and replay every segment in backward."""

    guarantees = WaveMemoryGuarantees(
        strategy="checkpoint",
        supports_autograd=True,
        preserves_forward_values=True,
    )

    def execute(
        self,
        request: PropagationRequest,
        backend: WaveBackendProtocol,
    ) -> PropagationResult:
        """Run configured uniform segments through the shared eager primitive."""
        eager = cast(EagerStrategyBackendProtocol, backend)
        telemetry = ExecutionTelemetry(request.acquisition.nt)
        context, state, _ = eager.prepare(request, telemetry)
        segment_count = min(
            request.config.memory.checkpoint_segments,
            request.acquisition.nt,
        )
        bounds = (
            torch.linspace(0, request.acquisition.nt, segment_count + 1)
            .round()
            .long()
            .tolist()
        )
        records: list[Tensor] = []
        collections: dict[str, Tensor] = {}
        with torch.autograd.graph.saved_tensors_hooks(
            telemetry.observe_saved_tensor,
            lambda tensor: tensor,
        ):
            for start, stop in zip(bounds[:-1], bounds[1:]):
                if stop <= start:
                    continue
                state, chunk, retained = _run_checkpoint_segment(
                    eager,
                    context,
                    state,
                    request.wavelets[:, start:stop],
                    start=start,
                )
                records.append(chunk)
                _merge_collections(collections, retained)
        return eager.assemble(
            request,
            context,
            state,
            torch.cat(records, dim=1),
            collections,
            telemetry,
        )


__all__ = ["CheckpointMemory"]
