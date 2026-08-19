"""
Hamiltonian-dynamics primitives for the Hamiltonian samplers.

Owns target-gradient evaluation plus the mass-matrix primitives shared by the
NUTS trajectory: kinetic energy, momentum sampling, ``M⁻¹·p`` drift, and the
momentum dot product. HMC keeps its optimised full-kick unit-mass integrator
inlined because it is not bit-identical to composing symmetric single steps.

``mass_info`` is either ``None`` (identity mass) or a per-field dict
``{field_name: entry}`` where each entry is one of:

    {"kind": "diagonal", "diag": Tensor (field_shape)}
    {"kind": "dense",    "M_inv": Tensor (n, n),
                         "chol_M": Tensor (n, n),
                         "field_shape": tuple}

Mixing kinds across fields is supported (small dense, large diagonal).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Any, Mapping, TypeAlias

import torch

from ...core import ForwardContext, ModelState
from ..base import (
    LogPosteriorTarget,
    _evaluate_log_posterior,
    _requires_grad_leaves,
)


MassEntry: TypeAlias = dict[str, Any]
MassInfo: TypeAlias = Mapping[str, MassEntry]


def _log_prob_and_gradient(
    target: LogPosteriorTarget,
    position: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    ctx: ForwardContext,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate a detached log-density and its full per-field gradient."""
    leaves = _requires_grad_leaves(position)
    log_prob = _evaluate_log_posterior(
        target,
        ModelState(tensors=leaves, metadata=metadata),
        ctx,
    )
    values = torch.autograd.grad(
        log_prob,
        list(leaves.values()),
        allow_unused=True,
    )
    gradient = {
        name: (
            value.detach()
            if value is not None
            else torch.zeros_like(tensor)
        )
        for (name, tensor), value in zip(leaves.items(), values)
    }
    return log_prob.detach(), gradient


def _dot_dict(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    return sum(float((a[name] * b[name]).sum().item()) for name in a)


def _kinetic(
    momentum: Mapping[str, torch.Tensor],
    mass_info: MassInfo | None,
) -> torch.Tensor:
    if mass_info is None:
        return 0.5 * sum(p.pow(2).sum() for p in momentum.values())
    ke = torch.zeros((), dtype=next(iter(momentum.values())).dtype,
                      device=next(iter(momentum.values())).device)
    for name, p in momentum.items():
        entry = mass_info[name]
        if entry["kind"] == "diagonal":
            ke = ke + 0.5 * (p.pow(2) / entry["diag"]).sum()
        else:   # dense
            p_flat = p.reshape(-1)
            ke = ke + 0.5 * p_flat @ entry["M_inv"] @ p_flat
    return ke


def _sample_momentum(
    theta: Mapping[str, torch.Tensor],
    mass_info: MassInfo | None,
    generator: torch.Generator | None,
) -> dict[str, torch.Tensor]:
    """Draw momentum ``p ~ N(0, M)`` under the chosen mass kind."""
    out: dict[str, torch.Tensor] = {}
    for name, t in theta.items():
        z = torch.randn(t.shape, generator=generator, dtype=t.dtype, device=t.device)
        if mass_info is None:
            out[name] = z
            continue
        entry = mass_info[name]
        if entry["kind"] == "diagonal":
            out[name] = z * entry["diag"].sqrt()
        else:   # dense
            z_flat = z.reshape(-1)
            out[name] = (entry["chol_M"] @ z_flat).reshape(t.shape)
    return out


def _drift(
    p_half: Mapping[str, torch.Tensor],
    mass_info: MassInfo | None,
) -> dict[str, torch.Tensor]:
    """Compute M⁻¹·p for each field, returning tensors of matching shape."""
    out: dict[str, torch.Tensor] = {}
    for name, p in p_half.items():
        if mass_info is None:
            out[name] = p
            continue
        entry = mass_info[name]
        if entry["kind"] == "diagonal":
            out[name] = p / entry["diag"]
        else:
            shape = entry["field_shape"]
            p_flat = p.reshape(-1)
            out[name] = (entry["M_inv"] @ p_flat).reshape(shape)
    return out
