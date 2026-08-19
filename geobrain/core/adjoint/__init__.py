"""Adjoint utilities: implicit-VJP solvers, checkpointing, VJP and gradient checks.

Cross-cutting differentiation helpers that several physics families rely on:

- :func:`linear_solve_with_adjoint`: canonical implicit-VJP solve used by every
  elliptic forward (DC / MT 2D/3D, single-phase flow, Helmholtz dense).
- :func:`sparse_linear_solve_with_adjoint`: sparse variant for complex-CSR
  forwards (Helmholtz / MT3D / DC3D).
- :func:`solve_adjoint`: the MATRIX-FREE / iterative regime of the same idea:
  wraps any :class:`~geobrain.core.linalg.LinearOperator` (or callable) and a
  Krylov solver into a differentiable solve. All three differentiable-solve
  regimes (dense direct, sparse direct, matrix-free iterative) are
  importable from this one package.
- :func:`checkpoint`: operator wrapper that re-runs the wrapped forward during
  backward via ``torch.utils.checkpoint``. Trades compute for memory; aimed at
  long time-marching wave kernels.
- :func:`vjp`: explicit vector-Jacobian product (first-order: it does not
  retain the graph, so it is not itself a Hessian-vector product). Useful for
  Bayesian sensitivity and adjoint inspection.
- :func:`gradient_check`: central-difference vs autograd comparison.

Module layout:

- ``linear_solve.py``: :func:`linear_solve_with_adjoint`,
  :func:`sparse_linear_solve_with_adjoint`.
- ``matrix_free.py``: :func:`solve_adjoint`.
- ``checkpoint.py``: :func:`checkpoint` and per-flavour wrappers.
- ``vjp.py``: :func:`vjp`.
- ``gradient_check.py``: :func:`gradient_check`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from ._sparse_impl import (
    SparseFactor,
    factorize_sparse,
    set_default_sparse_solver,
    sparse_solver,
    default_sparse_solver_is_cpu_only,
)
from ..linalg.backend_pcg import PcgGpuSolver
from ..linalg.seam import ScipySpluSolver, SparseFactorSolver
from .checkpoint import checkpoint
from .gradient_check import gradient_check
from .linear_solve import linear_solve_with_adjoint, sparse_linear_solve_with_adjoint
from .matrix_free import solve_adjoint
from .vjp import vjp

__all__ = [
    "checkpoint",
    "gradient_check",
    "linear_solve_with_adjoint",
    "sparse_linear_solve_with_adjoint",
    "SparseFactor",
    "factorize_sparse",
    "SparseFactorSolver",
    "solve_adjoint",
    "ScipySpluSolver",
    "PcgGpuSolver",
    "set_default_sparse_solver",
    "sparse_solver",
    "default_sparse_solver_is_cpu_only",
    "vjp",
]
