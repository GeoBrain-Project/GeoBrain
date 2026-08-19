"""Treeverse-style recursive checkpoint Wave memory execution.

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

from ..contracts import (
    ExecutionTelemetry,
    EagerStrategyBackendProtocol,
    PropagationRequest,
    PropagationResult,
    WaveBackendProtocol,
    WaveMemoryGuarantees,
)
from .checkpoint import _collection_names


class RecursiveMemory:
    """Retain one leaf graph plus nested bisection boundary states."""

    guarantees = WaveMemoryGuarantees(
        strategy="recursive",
        supports_autograd=True,
        preserves_forward_values=True,
    )

    def execute(
        self,
        request: PropagationRequest,
        backend: WaveBackendProtocol,
    ) -> PropagationResult:
        """Run nested non-reentrant checkpoints and observe actual replay work."""
        eager = cast(EagerStrategyBackendProtocol, backend)
        telemetry = ExecutionTelemetry(request.acquisition.nt)
        context, initial_state, _ = eager.prepare(request, telemetry)
        coefficient_names = tuple(context.coefficients)
        coefficient_tensors = tuple(
            context.coefficients[name] for name in coefficient_names
        )
        n_state = len(initial_state)
        leaf_steps = request.config.memory.recursive_leaf_steps

        def checkpoint_contexts() -> tuple[
            AbstractContextManager[None],
            AbstractContextManager[None],
        ]:
            return nullcontext(), telemetry.recompute_region()

        def run_interval(
            start: int,
            *tensors: Tensor,
        ) -> tuple[Tensor, ...]:
            state = tensors[:n_state]
            local_coefficients = dict(
                zip(
                    coefficient_names,
                    tensors[n_state : n_state + len(coefficient_names)],
                )
            )
            wavelets = tensors[-1]
            stop = start + int(wavelets.shape[1])
            names = _collection_names(context, start=start, stop=stop)
            local_context = replace(context, coefficients=local_coefficients)
            if wavelets.shape[1] <= leaf_steps:
                final_state, records, collections = eager.run_segment(
                    local_context,
                    state,
                    wavelets,
                    time_start=start,
                )
                return (
                    *final_state,
                    records,
                    *(collections[name] for name in names),
                )

            midpoint = int(wavelets.shape[1]) // 2
            middle_time = start + midpoint

            def left(*left_tensors: Tensor) -> tuple[Tensor, ...]:
                return run_interval(start, *left_tensors)

            left_names = _collection_names(
                context, start=start, stop=middle_time
            )
            with set_checkpoint_early_stop(False):
                left_output = checkpoint(
                    left,
                    *state,
                    *tuple(local_coefficients.values()),
                    wavelets[:, :midpoint],
                    use_reentrant=False,
                    context_fn=checkpoint_contexts,
                )

            def right(*right_tensors: Tensor) -> tuple[Tensor, ...]:
                return run_interval(middle_time, *right_tensors)

            right_names = _collection_names(
                context, start=middle_time, stop=stop
            )
            with set_checkpoint_early_stop(False):
                right_output = checkpoint(
                    right,
                    *left_output[:n_state],
                    *tuple(local_coefficients.values()),
                    wavelets[:, midpoint:],
                    use_reentrant=False,
                    context_fn=checkpoint_contexts,
                )
            left_collections = dict(
                zip(left_names, left_output[n_state + 1 :])
            )
            right_collections = dict(
                zip(right_names, right_output[n_state + 1 :])
            )
            retained = {
                name: (
                    left_collections[name] + right_collections[name]
                    if name in left_collections and name in right_collections
                    else (
                        left_collections[name]
                        if name in left_collections
                        else right_collections[name]
                    )
                )
                for name in names
            }
            records = torch.cat(
                (left_output[n_state], right_output[n_state]), dim=1
            )
            return (
                *right_output[:n_state],
                records,
                *(retained[name] for name in names),
            )

        with torch.autograd.graph.saved_tensors_hooks(
            telemetry.observe_saved_tensor,
            lambda tensor: tensor,
        ):
            output = run_interval(
                0,
                *initial_state,
                *coefficient_tensors,
                request.wavelets,
            )
        names = _collection_names(
            context, start=0, stop=request.acquisition.nt
        )
        state = tuple(output[:n_state])
        records = output[n_state]
        collections = dict(zip(names, output[n_state + 1 :]))
        return eager.assemble(
            request,
            context,
            state,
            records,
            collections,
            telemetry,
        )


__all__ = ["RecursiveMemory"]
