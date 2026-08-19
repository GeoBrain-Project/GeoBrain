"""Frequency-domain multi-right-hand-side execution primitives.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core.adjoint import factorize_sparse, sparse_linear_solve_with_adjoint
from geobrain.core.linalg import SparseFactorSolver
from geobrain.physics.em.errors import EMContractError

from .cache import EMExecutionCache
from .contracts import EMExecutionDiagnostics, FactorCacheKey
from .time import resolve_engine_solver, solver_factor_key


def solve_compatible_rhs(
    matrix_key: FactorCacheKey,
    rhs: torch.Tensor,
    *,
    cache: EMExecutionCache,
    solver: SparseFactorSolver | None = None,
) -> tuple[torch.Tensor, EMExecutionDiagnostics]:
    """Solve one exact matrix against one or many compatible RHS columns."""
    if not isinstance(rhs, torch.Tensor) or rhs.ndim not in (1, 2):
        raise EMContractError(
            "inductive RHS must have shape (n,) or (n, k)",
            details={"shape": list(getattr(rhs, "shape", ()))},
            object_name="solve_compatible_rhs",
            field="rhs",
            expected="torch.Tensor with rank 1 or 2",
            actual=type(rhs).__qualname__,
        )
    resolved_solver = resolve_engine_solver() if solver is None else solver
    expected_key = solver_factor_key(matrix_key.assembly, resolved_solver)
    if matrix_key != expected_key:
        raise EMContractError(
            "factor key does not match the resolved sparse solver",
            details={
                "received_backend": matrix_key.assembly.backend,
                "expected_backend": expected_key.assembly.backend,
            },
            object_name="solve_compatible_rhs",
            field="matrix_key",
            expected="key derived from the resolved solver",
            actual="mismatched solver controls",
        )
    matrix = cache.require_assembly(matrix_key.assembly)
    if not isinstance(matrix, torch.Tensor) or matrix.layout not in (
        torch.sparse_coo,
        torch.sparse_csr,
    ):
        raise EMContractError(
            "registered inductive assembly must be a sparse tensor",
            details={"received_type": type(matrix).__qualname__},
            object_name="solve_compatible_rhs",
            field="matrix",
            expected="sparse COO or CSR tensor",
            actual=type(matrix).__qualname__,
        )
    factor = cache.get_or_factor(
        matrix_key,
        lambda: factorize_sparse(matrix, solver=resolved_solver),
    )
    result = sparse_linear_solve_with_adjoint(matrix, rhs, factor=factor)
    cache.diagnostics.solve_count += 1
    cache.diagnostics.rhs_count += 1 if rhs.ndim == 1 else int(rhs.shape[1])
    return result, cache.diagnostics


__all__ = ["solve_compatible_rhs"]
