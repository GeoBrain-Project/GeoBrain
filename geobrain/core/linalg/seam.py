"""
The sparse factor-and-solve SEAM: protocol, default backend, and scoping.

Owned by :mod:`geobrain.core.linalg`: the protocol a factor-reusing sparse backend implements
(:class:`SparseFactorSolver`), the built-in CPU default
(:class:`ScipySpluSolver`), the process-global / ContextVar-scoped selection
machinery (:func:`set_default_sparse_solver` / :func:`sparse_solver`), and
the CPU-only gate operators consult. ``core.adjoint`` CONSUMES this seam for
its implicit-VJP bridge and re-exports the names for backward compatibility;
the dependency is unidirectional: linalg defines the linear-algebra contract,
adjoint (and the physics operators behind it) use it.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import contextlib
import contextvars
from abc import ABC, abstractmethod
from typing import Any

import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from ..errors import GeoBrainError

class SparseFactorSolver(ABC):
    """Backend contract for the sparse adjoint solve: factor + (adjoint) solve.

    A backend implements two methods; both work in torch tensors so the bridge
    stays backend-agnostic (the backend does any host/device conversion):

    - ``factor(indices, values, shape) -> handle``: factorize the sparse matrix
      given as ``(2, nnz)`` ``(row, col)`` indices, ``(nnz,)`` values, and an
      ``(n, n)`` shape; returns an opaque, reusable handle.
    - ``solve(handle, b, *, adjoint) -> Tensor``: solve ``A x = b`` (or the
      conjugate-transpose system ``Aᴴ x = b`` when ``adjoint=True``) for a dense
      ``b`` of shape ``(n,)`` or ``(n, k)``, returning ``x`` like ``b``.
    """

    @abstractmethod
    def factor(self, indices: torch.Tensor, values: torch.Tensor, shape) -> Any:
        ...

    @abstractmethod
    def solve(self, handle: Any, b: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
        ...


class ScipySpluSolver(SparseFactorSolver):
    """Default CPU backend: ``scipy.sparse.linalg.splu`` (float64 / complex128)."""

    def factor(self, indices: torch.Tensor, values: torch.Tensor, shape) -> Any:
        # detach() first: the raw `.values()` view of a complex CSR tensor
        # rejects several pointwise ops ("Sparse CSR tensors do not have
        # strides"); the detached view behaves like a normal strided tensor.
        if not bool(torch.isfinite(values.detach()).all()):
            raise GeoBrainError(
                "sparse matrix values are not finite (NaN/Inf); check the "
                "property model (e.g. σ) used to assemble the system",
                object_name="ScipySpluSolver",
                field="values",
                expected="all entries finite",
                actual="NaN/Inf present",
            )
        rows = indices[0].cpu().numpy()
        cols = indices[1].cpu().numpy()
        A = sp.csc_matrix((values.detach().cpu().numpy(), (rows, cols)), shape=shape)
        try:
            return spla.splu(A)
        except RuntimeError as exc:
            raise GeoBrainError(
                f"sparse LU factorization failed ({exc}); common causes: "
                "non-positive σ entries, a missing Dirichlet pin, or a "
                "pure-Neumann (singular) system",
                object_name="ScipySpluSolver",
                field="A",
                expected="non-singular sparse matrix",
                actual=str(exc),
            ) from exc

    def solve(self, handle: Any, b: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
        b_np = b.detach().cpu().numpy()
        if adjoint:
            # Aᴴ solve: 'H' (conjugate-transpose) for complex, 'T' for real,
            # matches torch.linalg.solve's autograd backward for the dense case.
            # PRECONDITION (seam contract): b dtype matches the factored
            # matrix dtype: the adjoint bridge enforces it at entry. A direct
            # seam consumer passing a real b over a complex factor would get
            # a plain transpose where the conjugate transpose is meant.
            trans = "H" if torch.is_complex(b) else "T"
            x_np = handle.solve(b_np, trans=trans)
        else:
            x_np = handle.solve(b_np)
        return torch.from_numpy(x_np).to(dtype=b.dtype, device=b.device)


# * ``_SCOPED_SOLVER``: a ``ContextVar`` holding a per-context override
#   installed by the scoped :func:`sparse_solver` context manager. Because it
#   is a ``ContextVar``, the scope is isolated per thread / per asyncio task,
#   so two threads each entering their own ``sparse_solver(...)`` block no
#   longer clobber a shared global and corrupt each other's restore. ``None``
#   means "no scoped override → fall through to the process default".
#
# Every call site reads the effective backend via :func:`_effective_default`
# when ``solver=`` is not given, so both layers are honored with zero operator
# changes and the built-in default keeps the historical behaviour exactly.
_DEFAULT_SOLVER: SparseFactorSolver = ScipySpluSolver()
_SCOPED_SOLVER: contextvars.ContextVar["SparseFactorSolver | None"] = (
    contextvars.ContextVar("geobrain_scoped_sparse_solver", default=None)
)


def _effective_default() -> SparseFactorSolver:
    """The backend a ``solver=``-less call resolves to: the scoped override if
    one is active in this context, else the process-global default."""
    scoped = _SCOPED_SOLVER.get()
    return scoped if scoped is not None else _DEFAULT_SOLVER


def set_default_sparse_solver(solver: SparseFactorSolver | None) -> None:
    """Set the process-global default backend for every implicit-sparse adjoint
    solve. Pass None to reset to the built-in ScipySpluSolver. Because every
    call site falls through to the default when solver= is not given, this is
    honored everywhere with zero operator changes.

    This is the PROCESS-wide setting (visible across threads). For a temporary,
    thread-isolated override use the scoped :func:`sparse_solver` context
    manager instead."""
    global _DEFAULT_SOLVER
    if solver is None:
        _DEFAULT_SOLVER = ScipySpluSolver()
        return
    if not isinstance(solver, SparseFactorSolver):
        raise GeoBrainError(
            "set_default_sparse_solver: solver must be a SparseFactorSolver "
            "(or None to reset)",
            object_name="set_default_sparse_solver",
            field="solver",
            expected="SparseFactorSolver or None",
            actual=type(solver).__name__,
        )
    _DEFAULT_SOLVER = solver


@contextlib.contextmanager
def sparse_solver(solver: SparseFactorSolver | None):
    """Scoped override of the default sparse solver; restores the prior on exit.

    Thread/async-safe: the override lives in a :class:`contextvars.ContextVar`,
    so concurrent threads (or asyncio tasks) each entering their own
    ``sparse_solver(...)`` block get an isolated override and never clobber one
    another's restore. Passing ``None`` scopes a fresh CPU
    :class:`ScipySpluSolver` for the block."""
    if solver is None:
        solver = ScipySpluSolver()
    elif not isinstance(solver, SparseFactorSolver):
        raise GeoBrainError(
            "sparse_solver: solver must be a SparseFactorSolver (or None)",
            object_name="sparse_solver",
            field="solver",
            expected="SparseFactorSolver or None",
            actual=type(solver).__name__,
        )
    token = _SCOPED_SOLVER.set(solver)
    try:
        yield solver
    finally:
        _SCOPED_SOLVER.reset(token)


def _is_cpu_only_solver(solver: SparseFactorSolver | None) -> bool:
    """True if ``solver`` is the CPU-only scipy backend (or a subclass of it).

    Used to gate the device guard: a non-CPU matrix is rejected ONLY when the
    effective backend is :class:`ScipySpluSolver` (scipy splu factors on the
    host). Any other :class:`SparseFactorSolver` (e.g. a GPU CG backend) is
    trusted to handle its own device. ``None`` means "fall through to the
    effective default" (scoped override if active, else the process default).
    """
    if solver is None:
        return isinstance(_effective_default(), ScipySpluSolver)
    return isinstance(solver, ScipySpluSolver)


def default_sparse_solver_is_cpu_only() -> bool:
    """True if the process-global default sparse backend is the CPU-only scipy splu.

    Operators gate their device guards on this: a complex/sparse EM forward may accept
    GPU input ONLY when a GPU-capable backend (e.g. ``geobrain.core.linalg.KrylovSolver``) is
    active via :func:`set_default_sparse_solver` / :func:`sparse_solver`. With the default
    scipy backend it stays True, so the historical CPU-only behavior is unchanged."""
    return _is_cpu_only_solver(None)
