"""``geobrain.core.linalg``: the unified matrix-free, complex-correct, differentiable linear
algebra stack.

Named ``linalg`` (not ``solvers``) to avoid collision with :mod:`geobrain.optim.solvers`
(the outer optimizers/samplers). One ``LinearOperator`` + Krylov + preconditioner
interface, complex-correct throughout (conjugate inner products; ``rmatvec = A^H``).

This package is pure solve-machinery, HOW to solve ``Ax = b``. Differentiating
THROUGH a solve is the sibling :mod:`geobrain.core.adjoint`'s axis: the
matrix-free implicit-differentiation entry point ``solve_adjoint`` (verified
against ``torch.linalg.solve``'s gradient convention) lives there as
``adjoint/matrix_free.py``, so every differentiable-solve seam is
importable from that one package.

Consumers, by seam:

- the frequency-batched EM solves use ``block_diag`` (MT2D/MT3D/FDEM3D);
- :class:`~geobrain.core.linalg.backend.KrylovSolver` and
  :class:`~geobrain.core.linalg.backend_pcg.PcgGpuSolver` implement the
  ``core.adjoint`` SparseFactorSolver seam (opt-in complex / real-SPD GPU
  paths; all three backends live here, adjoint re-exports the public path);
- the flow linear solvers delegate their recurrences here, keeping flow's
  convergence POLICY local (x0-continuation restarts, per-iteration GMRES
  stopping via ``check_every=1``, tol*||b|| criteria via the atol channel).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Any

from .seam import (
    ScipySpluSolver,
    SparseFactorSolver,
    default_sparse_solver_is_cpu_only,
    set_default_sparse_solver,
    sparse_solver,
)
from .backend import KrylovSolver
from .backend_pcg import PcgGpuSolver
from .linear_operator import LinearOperator, aslinearoperator
from .krylov import cg, bicgstab, gmres, SolveStats
from .precond import Jacobi, ILU0
from .batched import block_diag

__all__ = [
    "PcgGpuSolver",
    "KrylovSolver",
    "SparseFactorSolver", "ScipySpluSolver", "sparse_solver",
    "set_default_sparse_solver", "default_sparse_solver_is_cpu_only",
    "LinearOperator",
    "aslinearoperator",
    "cg",
    "bicgstab",
    "gmres",
    "SolveStats",
    "Jacobi",
    "ILU0",
    "block_diag",
    # lazy (see __getattr__ below): CuDSSSolver, cudss_available
]

# CuDSSSolver / cudss_available: lazy,
# guarded re-export of the optional NVIDIA cuDSS backend
# (geobrain.core.linalg.backend_cudss: needs the nvidia-cudss-cu12 wheel +
# a CUDA device at USE time, not at import time). ``backend_cudss`` itself
# never eagerly loads libcudss.so at module scope, so a straight top-level
# import would already be safe today, but resolving it through
# ``__getattr__`` keeps that safety explicit and future-proof (the absence of
# the wheel must never break ``import geobrain.core.linalg``), and avoids
# paying the ctypes/glob import cost for callers who never touch cuDSS.
_LAZY_EXPORTS: dict[str, str] = {
    "CuDSSSolver": "backend_cudss",
    "cudss_available": "backend_cudss",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
