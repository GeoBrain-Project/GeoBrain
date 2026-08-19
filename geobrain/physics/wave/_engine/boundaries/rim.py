"""
Shared rim-save / rim-restore / state-assemble helpers for the boundary-saving
custom VJP in :mod:`wave._engine.memory.boundary`.

These three operations are dimension-agnostic; the ``mask`` carries the spatial
selection and the wavefields carry their own shape, so both adjoints share one
definition. Single-sourcing matters here specifically because these run inside
the custom-VJP path: a silent drift between a 2-D and 3-D copy would corrupt the
reconstructed gradient rather than raise, exactly the failure class hardest to
catch.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch
from torch import Tensor


class _RimConfig(Protocol):
    """Minimal dimension-neutral config surface consumed by :func:`assemble`."""

    n_state: int
    wf_indices: tuple[int, ...]
    mem_indices: tuple[int, ...]


def save_rim(field: Tensor, mask: Tensor) -> Tensor:
    """Clone the rim cells (channel 0) selected by ``mask``: the saved truth."""
    return field[:, 0][:, mask].clone()


def set_rim(field: Tensor, mask: Tensor, vals: Tensor) -> Tensor:
    """Return ``field`` with its ``mask`` rim cells overwritten by ``vals``."""
    f = field.clone()
    f[:, 0][:, mask] = vals
    return f


def assemble(
    cfg: _RimConfig,
    wavefields: Sequence[Tensor],
    memory: Sequence[Tensor] | None = None,
) -> list[Tensor]:
    """Full state list with ``wavefields`` at ``cfg.wf_indices`` and, when
    given, ``memory`` (CPML ψ) at ``cfg.mem_indices``; unfilled slots zeroed.

    ``memory=None`` zero-fills the ψ slots. That is exact only where CPML is
    the identity (the physical interior); the W2 fix threads the SAVED ψ band
    history through here for the per-step VJP inputs instead.
    """
    state: list[Tensor] = [torch.zeros_like(wavefields[0]) for _ in range(cfg.n_state)]
    for idx, f in zip(cfg.wf_indices, wavefields):
        state[idx] = f
    if memory is not None:
        for idx, m in zip(cfg.mem_indices, memory):
            state[idx] = m
    return state
