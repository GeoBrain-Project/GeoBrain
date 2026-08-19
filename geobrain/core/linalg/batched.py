"""
Frequency-batching primitives.

The natural-source and controlled-source 2D/3D EM forwards solve one INDEPENDENT sparse
system per frequency, ``A(ω_i) x_i = b_i``, in a Python loop. On the GPU that loop is
dispatch-bound: n_freq small sparse matvecs per Krylov iteration, each paying launch
overhead. :func:`block_diag` assembles the per-frequency operators into a single
block-diagonal system so the whole set solves in ONE Krylov call, one large matvec per
iteration instead of n_freq small ones, while a block-diagonal (per-block) preconditioner
keeps each frequency's convergence independent.

The assembly is differentiable (indices are offset, values are concatenated), so σ
gradients flow back to each frequency's block exactly as in the per-frequency loop.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..errors import GeoBrainError


def block_diag(mats: Sequence[torch.Tensor]) -> torch.Tensor:
    """Assemble square sparse matrices into one block-diagonal sparse-COO tensor.

    Differentiable through the block values, so a downstream solve's gradient reaches each
    block. Off-diagonal blocks are structurally zero (never materialized).
    Dense blocks are accepted and converted via ``to_sparse_coo()`` (the
    conversion is differentiable); each block must be square.

    All blocks must share one dtype and one device (the assembled system is a
    single tensor; a silent cast/copy would detach gradients or hide a device
    mistake), violations raise a structured :class:`GeoBrainError`.
    """
    mats = list(mats)
    if not mats:
        raise GeoBrainError(
            "block_diag requires at least one block",
            object_name="block_diag", field="mats",
            expected="non-empty sequence of sparse matrices", actual=[],
        )
    dtype0, device0 = mats[0].dtype, mats[0].device
    for i, A in enumerate(mats):
        if not isinstance(A, torch.Tensor):
            raise GeoBrainError(
                "block_diag blocks must be torch tensors",
                object_name="block_diag", field=f"mats[{i}]",
                expected="torch.Tensor", actual=type(A).__name__,
            )
        if A.dtype != dtype0:
            raise GeoBrainError(
                "block_diag blocks must share one dtype",
                object_name="block_diag", field=f"mats[{i}].dtype",
                expected=str(dtype0), actual=str(A.dtype),
            )
        if A.device != device0:
            raise GeoBrainError(
                "block_diag blocks must share one device",
                object_name="block_diag", field=f"mats[{i}].device",
                expected=str(device0), actual=str(A.device),
            )
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise GeoBrainError(
                "block_diag blocks must be square 2-D matrices",
                object_name="block_diag", field=f"mats[{i}].shape",
                expected="(n, n)", actual=tuple(A.shape),
            )
    idx_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    row_off = 0
    col_off = 0
    for A in mats:
        A = (A.coalesce() if A.is_sparse else A.to_sparse_coo().coalesce()) \
            if A.layout is not torch.strided else A.to_sparse_coo().coalesce()
        idx = A.indices().clone()
        idx[0] = idx[0] + row_off
        idx[1] = idx[1] + col_off
        idx_parts.append(idx)
        val_parts.append(A.values())
        row_off += A.shape[0]
        col_off += A.shape[1]
    indices = torch.cat(idx_parts, dim=1)
    values = torch.cat(val_parts)
    return torch.sparse_coo_tensor(indices, values, (row_off, col_off)).coalesce()
