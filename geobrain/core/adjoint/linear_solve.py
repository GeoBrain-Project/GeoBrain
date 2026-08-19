"""Implicit-VJP linear solves: dense and sparse variants.

Both forward solves ``A x = b``; backward propagates ``∂L/∂x`` to ``∂L/∂A``
and ``∂L/∂b`` via the adjoint of the solve, the implicit function theorem.

The two variants differ in *who* implements that backward:

- The dense :func:`linear_solve_with_adjoint` is a thin wrapper over
  ``torch.linalg.solve``. It adds **no** custom backward, PyTorch's own autograd
  for ``solve`` already is the implicit-function-theorem adjoint, reusing the LU
  internally. The wrapper exists for input validation only. There is **no**
  caller-visible factor handle on the dense path (no ``factor=`` parameter):
  explicit factor reuse is a sparse-only contract (see the sparse function).
- The sparse :func:`sparse_linear_solve_with_adjoint` *does* hand-write a
  ``torch.autograd.Function`` (see ``_sparse_impl``), because its forward
  goes through scipy's ``splu``, which is opaque to autograd.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..errors import GeoBrainError
from ._sparse_impl import sparse_solve_with_adjoint as _sparse_solve_impl

if TYPE_CHECKING:
    from ._sparse_impl import SparseFactor
    from ..linalg.seam import SparseFactorSolver


def linear_solve_with_adjoint(
    A: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Solve ``A x = b`` with gradients flowing through both ``A`` and ``b``.

    Args:
        A: square 2D tensor (``n × n``); discretised operator. May depend on
            trainable parameters.
        b: right-hand side, ``(n,)`` or ``(n, k)``.

    Returns:
        ``x`` such that ``A x = b``. Gradients flow through PyTorch's own
        autograd for ``torch.linalg.solve``, which already is the
        implicit-function-theorem adjoint (``∂L/∂x`` → ``∂L/∂A``, ``∂L/∂b``).
        This function adds no custom backward; see the module docstring.

    Note:
        Finiteness of ``A`` / ``b`` is **deliberately not checked**: this is a
        pure-PyTorch path, so a non-finite input propagates as ``NaN`` per IEEE
        (the correct, downstream-``isfinite``-catchable behaviour). The sparse
        sibling :func:`sparse_linear_solve_with_adjoint` *does* guard finiteness,
        only because scipy's ``splu`` reports a non-finite matrix as a
        misleading "singular" error at a foreign boundary; the dense torch path
        has no such boundary, hence the asymmetry is intentional.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise GeoBrainError(
            "linear_solve_with_adjoint expects a square 2D A",
            object_name="linear_solve_with_adjoint",
            field="A",
            expected="square 2D Tensor",
            actual=tuple(A.shape),
        )
    if b.ndim not in (1, 2) or b.shape[0] != A.shape[0]:
        raise GeoBrainError(
            "b must be 1D or 2D with leading dim matching A",
            object_name="linear_solve_with_adjoint",
            field="b",
            expected=f"({A.shape[0]},) or ({A.shape[0]}, k)",
            actual=tuple(b.shape),
        )

    # Dense solve; backward reuses the factorisation under the hood.
    if b.ndim == 1:
        return torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    return torch.linalg.solve(A, b)


def sparse_linear_solve_with_adjoint(
    A_sparse: torch.Tensor,
    b: torch.Tensor,
    *,
    factor: "SparseFactor | None" = None,
    solver: "SparseFactorSolver | None" = None,
) -> torch.Tensor:
    """
    Sparse ``A x = b`` with implicit-VJP backward.

    Forward: factor ``A`` (default scipy splu) + solve. Backward: the factor is
    reused for the adjoint solve; ``∂L/∂A`` is a rank-1 outer product on each
    non-zero entry.

    Args:
        A_sparse: torch sparse CSR/COO tensor; values may carry autograd.
        b:        right-hand side, ``(n,)`` or ``(n, k)``.
        factor:   optional :class:`~geobrain.core.adjoint.SparseFactor` (from
            :func:`factorize_sparse`) to REUSE instead of re-factorizing, for
            repeated-matrix solves such as a time loop's many equal-``dt``
            substeps. Must factor the same numeric ``A``. The σ-gradient is
            still computed per call, so reuse changes only the numeric solves.
        solver:   optional :class:`~geobrain.core.adjoint.SparseFactorSolver`
            backend (default :class:`ScipySpluSolver`) used when no ``factor`` is
            supplied, the injection seam for a future GPU/iterative backend.

    Returns:
        ``x`` such that ``A x = b``.

    Note:
        Gradients are **first-order only**. The backward reuses the scipy
        ``splu`` factor, which is opaque to autograd, so double-backward /
        Hessian-vector products are **not** supported through the sparse path,
        a second ``torch.autograd.grad`` raises. Use the dense
        :func:`linear_solve_with_adjoint` when second-order gradients are needed.
    """
    return _sparse_solve_impl(A_sparse, b, factor=factor, solver=solver)
