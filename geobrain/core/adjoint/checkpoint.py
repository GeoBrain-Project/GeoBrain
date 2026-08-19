"""Operator-level gradient checkpointing.

Wraps an arbitrary :class:`~geobrain.core.Operator` so its forward is
recomputed during backward via ``torch.utils.checkpoint.checkpoint``.
Trades compute for memory; primarily aimed at the time-marching wave
kernels (long acoustic/elastic FDTD).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils.checkpoint as _ck

from ..errors import GeoBrainError
from ..operator import ForwardOperator, Operator, PropertyTransform
from ..containers import ModelState, ForwardOutput
from ..context import ForwardContext


def _ckpt_run_state(inner: Operator, state: ModelState, ctx: ForwardContext) -> ModelState:
    names = sorted(state.tensors)
    tensors = [state.tensors[n] for n in names]
    meta_state = state.metadata
    capture: dict[str, Any] = {}

    def _run(*ts: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rebuilt = ModelState(
            tensors={n: t for n, t in zip(names, ts)},
            metadata=meta_state,
        )
        out_state = inner._forward(rebuilt, ctx)
        if not isinstance(out_state, ModelState):
            raise GeoBrainError(
                "checkpoint()-wrapped PropertyTransform must return a ModelState",
                object_name="checkpoint",
                field="inner._forward",
                expected="ModelState",
                actual=type(out_state).__name__,
            )
        out_names = sorted(out_state.tensors)
        capture["names"] = out_names
        capture["metadata"] = out_state.metadata
        return tuple(out_state.tensors[n] for n in out_names)

    out = _ck.checkpoint(_run, *tensors, use_reentrant=False)
    if not isinstance(out, tuple):
        out = (out,)
    out_tensors: dict[str, torch.Tensor] = {
        n: t for n, t in zip(capture["names"], out)
    }
    return ModelState(tensors=out_tensors, metadata=capture["metadata"])


def _ckpt_run_obs(inner: Operator, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
    names = sorted(state.tensors)
    tensors = [state.tensors[n] for n in names]
    meta_state = state.metadata
    capture: dict[str, Any] = {}

    def _run(*ts: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rebuilt = ModelState(
            tensors={n: t for n, t in zip(names, ts)},
            metadata=meta_state,
        )
        pred = inner._forward(rebuilt, ctx)
        if not isinstance(pred, ForwardOutput):
            raise GeoBrainError(
                "checkpoint()-wrapped ForwardOperator must return a ForwardOutput",
                object_name="checkpoint",
                field="inner._forward",
                expected="ForwardOutput",
                actual=type(pred).__name__,
            )
        data_names = sorted(pred.data)
        field_names = sorted(pred.fields)
        capture["data_names"] = data_names
        capture["field_names"] = field_names
        capture["metadata"] = pred.metadata
        return tuple(
            [pred.data[n] for n in data_names]
            + [pred.fields[n] for n in field_names]
        )

    out = _ck.checkpoint(_run, *tensors, use_reentrant=False)
    if not isinstance(out, tuple):
        out = (out,)
    n_data = len(capture["data_names"])
    data_dict: dict[str, torch.Tensor] = {
        n: t for n, t in zip(capture["data_names"], out[:n_data])
    }
    fields_dict: dict[str, torch.Tensor] = {
        n: t for n, t in zip(capture["field_names"], out[n_data:])
    }
    return ForwardOutput(data=data_dict, fields=fields_dict, metadata=capture["metadata"])


class _CheckpointedTransform(PropertyTransform):
    def __init__(self, inner: PropertyTransform) -> None:
        super().__init__()
        self.inner = inner
        # Pass-through spec: ``inner`` is fixed at construction, so its
        # differentiability / mesh-capability metadata is read once here
        # rather than re-read via a property on every access: a checkpoint wrapper never changes what it wraps.
        self.differentiability = inner.differentiability
        self.requires_mesh_capabilities = inner.requires_mesh_capabilities

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        return _ckpt_run_state(self.inner, state, ctx)


class _CheckpointedObservation(ForwardOperator):
    def __init__(self, inner: ForwardOperator) -> None:
        super().__init__()
        self.inner = inner
        # Pass-through spec: see _CheckpointedTransform.__init__.
        self.differentiability = inner.differentiability
        self.requires_mesh_capabilities = inner.requires_mesh_capabilities

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        return _ckpt_run_obs(self.inner, state, ctx)


def checkpoint(inner: Operator) -> Operator:
    """Wrap ``inner`` so its forward is recomputed during backward.

    Returns a wrapper of the same flavour as ``inner``:

        - ``PropertyTransform`` → ``_CheckpointedTransform`` (yields a
          ``ModelState``).
        - ``ForwardOperator`` → ``_CheckpointedObservation`` (yields a
          ``ForwardOutput``).

    The wrapped op runs inside ``torch.utils.checkpoint.checkpoint`` with
    ``use_reentrant=False``: forward stores only the inputs / final output,
    and backward re-runs the forward under autograd. Net effect, peak
    memory drops from ``O(activations)`` to ``O(inputs + output)`` at the
    cost of one extra forward pass.

    Most useful around long time-marched physics (``Acoustic2D``,
    ``Acoustic3D``).  Differentiability of the wrapped operator is
    preserved verbatim, the wrapper copies ``inner.differentiability``
    once at construction (``inner`` is fixed for the wrapper's lifetime).

    Example::

        wave   = Acoustic2D(survey=..., wavelet=...)
        cheap  = checkpoint(wave)
        chain  = cheap @ Gardner()             # standard composition
    """
    if isinstance(inner, PropertyTransform):
        return _CheckpointedTransform(inner)
    if isinstance(inner, ForwardOperator):
        return _CheckpointedObservation(inner)
    raise GeoBrainError(
        "checkpoint() expects a PropertyTransform or ForwardOperator",
        object_name="checkpoint",
        field="inner",
        expected="PropertyTransform | ForwardOperator",
        actual=type(inner),
    )
