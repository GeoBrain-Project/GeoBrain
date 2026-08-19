"""Finite-difference vs autograd gradient cross-check: the per-entry DIAGNOSTIC.

Answers "*which entries* of this field have a wrong gradient?": probes
``n_probes`` random entries with per-entry central differences (2 forwards
per probe) and returns the raw ``{auto, fd, indices, max_rel_err}`` for
inspection, a localization instrument. You choose the scalar reduction and
the tolerance policy. For FDTD/Krylov-scale operators, where per-entry
probing is unaffordable, prefer a directional-derivative check (a couple of
random unit directions costs O(1) forwards regardless of grid size).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable

import torch

from ..errors import GeoBrainError
from ..operator import Operator
from ..containers import ModelState, ForwardOutput
from ..context import ForwardContext


def gradient_check(
    operator: Operator,
    state: ModelState,
    ctx: ForwardContext,
    *,
    field_name: str,
    scalar_fn: Callable[[ForwardOutput | ModelState], torch.Tensor],
    eps: float = 1e-3,
    n_probes: int = 16,
    seed: int = 0,
) -> dict[str, torch.Tensor | float]:
    """
    Compare central-difference and autograd gradients on random entries.

    Args:
        operator: the ``Operator`` under test.
        state: a ``ModelState`` containing ``field_name`` plus any other
            inputs the operator needs.
        ctx: ``ForwardContext`` for the forward.
        field_name: the input field to differentiate.
        scalar_fn: maps the operator's output (``ForwardOutput`` or
            ``ModelState``) to a 0-d ``Tensor``. Typically a sum of squared
            residuals; do not detach inside.
        eps: central-difference step in absolute units of ``field_name``.
        n_probes: how many random entries of ``field_name`` to probe.
        seed: RNG seed for the probe indices.

    Returns:
        ``{"auto": ..., "fd": ..., "indices": ..., "max_rel_err": float}``;
        both arrays have shape ``(n_probes,)``. ``max_rel_err`` is the
        usual ``max |fd − auto| / max(|auto|, 1e-12)``.
    """
    if field_name not in state.tensors:
        raise GeoBrainError(
            "gradient_check missing input field",
            object_name="gradient_check",
            field="field_name",
            expected=f"present in {sorted(state.tensors)}",
            actual=field_name,
        )

    if state.tensors[field_name].is_complex():
        # A real ``eps`` step (below) perturbs only the real part, so ``fd`` would
        # measure the real part of the Wirtinger derivative and could never match
        # the full autograd complex gradient: the comparison is meaningless.
        # Fail loud instead of returning a misleading ``max_rel_err``.
        raise GeoBrainError(
            "gradient_check does not support complex fields: a real "
            "finite-difference step only probes the real part of the Wirtinger "
            "derivative and cannot be compared to the autograd gradient.",
            object_name="gradient_check",
            field="field_name",
            expected="a real-valued field",
            actual=str(state.tensors[field_name].dtype),
        )

    # --- autograd reference ---
    leaf = state.tensors[field_name].detach().clone().requires_grad_(True)
    new_tensors = dict(state.tensors)
    new_tensors[field_name] = leaf
    new_state = ModelState(tensors=new_tensors, metadata=state.metadata)
    out = operator(new_state, ctx)
    scalar = scalar_fn(out)
    (grad_auto,) = torch.autograd.grad(scalar, leaf)
    grad_auto_flat = grad_auto.detach().flatten()

    # --- probe indices ---
    n = leaf.numel()
    n_probes = min(n_probes, n)
    g = torch.Generator(device=leaf.device).manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=leaf.device)[:n_probes]

    base = leaf.detach().clone()
    # Reuse ONE working tensor across every probe: perturb a single scalar
    # entry in place, evaluate, then restore the original value. This is
    # numerically identical to cloning ``base`` per probe (the plus/minus
    # inputs are the same as ``base`` everywhere except entry ``k``), but it
    # allocates one tensor instead of two per probe.
    work = base.clone()
    work_flat = work.view(-1)
    base_flat = base.view(-1)
    fd = torch.zeros(n_probes, dtype=base.dtype, device=base.device)
    for i, k in enumerate(idx.tolist()):
        orig = base_flat[k]
        work_flat[k] = orig + eps
        s_plus = scalar_fn(operator(
            ModelState(tensors={**new_tensors, field_name: work},
                       metadata=state.metadata), ctx))
        work_flat[k] = orig - eps
        s_minus = scalar_fn(operator(
            ModelState(tensors={**new_tensors, field_name: work},
                       metadata=state.metadata), ctx))
        work_flat[k] = orig
        fd[i] = (s_plus - s_minus) / (2 * eps)

    auto = grad_auto_flat[idx]
    denom = auto.abs().clamp(min=1e-12)
    rel = (fd - auto).abs() / denom
    return {
        "auto": auto.detach(),
        "fd": fd.detach(),
        "indices": idx.detach(),
        "max_rel_err": float(rel.max()),
    }
