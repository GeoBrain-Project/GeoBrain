"""Hermitian sparse adjoint operations for Helmholtz solves.

The CPU SciPy forward is connected to PyTorch through GeoBrain's implicit
linear-solve VJP. Complex derivatives use the Hermitian, never plain-transpose,
adjoint.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core.adjoint import ScipySpluSolver, sparse_linear_solve_with_adjoint


_DIRECT_SOLVER = ScipySpluSolver()


def solve_with_implicit_adjoint(
    matrix: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Solve ``matrix @ field = rhs`` with the reusable Hermitian VJP."""
    return sparse_linear_solve_with_adjoint(matrix, rhs, solver=_DIRECT_SOLVER)


def hermitian_matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply the conjugate transpose of a sparse COO matrix."""
    coalesced = matrix.coalesce()
    indices = coalesced.indices()
    adjoint = torch.sparse_coo_tensor(
        torch.stack((indices[1], indices[0])),
        coalesced.values().conj(),
        size=(matrix.shape[1], matrix.shape[0]),
        dtype=matrix.dtype,
        device=matrix.device,
    ).coalesce()
    column = vector.unsqueeze(1) if vector.ndim == 1 else vector
    result = torch.sparse.mm(adjoint, column)
    return result[:, 0] if vector.ndim == 1 else result


__all__ = ["hermitian_matvec", "solve_with_implicit_adjoint"]
