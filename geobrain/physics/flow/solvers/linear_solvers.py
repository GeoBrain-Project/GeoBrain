# pyright: reportPrivateImportUsage=false
"""
Linear solvers for the Newton system ``J · dx = -r``.

Pure-torch core (``DirectSolver`` / ``BiCGSTABSolver`` / ``GMRESSolver``)
plus a scipy-bridged extension family for reservoir-scale problems:

- :class:`DirectSolver`: dense ``torch.linalg.solve``. ``n_dof < ~5e3``.
- :class:`BiCGSTABSolver`: Krylov, non-symmetric flow Jacobians.
- :class:`GMRESSolver`: restarted GMRES; more memory but more robust.
- :class:`SparseDirectSolver`: scipy ``spsolve`` for ``n_dof < ~100k``.
- :class:`ILU0Preconditioner`: scipy ``spilu`` ILU(0).
- :class:`BILU0Preconditioner`: block-ILU(0) for multi-phase Jacobians
  whose state vector is per-variable ordered.
- :class:`MultigridSolver`: thin Krylov-style wrapper around
  :class:`GeometricMultigrid` (Cartesian only).
- :class:`CPRSolver`: Constrained Pressure Residual: GMRES + BILU0 +
  geometric MG on the pressure sub-block.

Choosing a solver: the forward Newton path (:class:`~geobrain.physics.flow.
newton.NewtonSolver`, used by the flow operators) defaults to a *direct*
solve, :class:`SparseDirectSolver` (scipy ``spsolve``) for sparse Jacobians,
:class:`DirectSolver` for dense, which is the production path. The pure-torch
Krylov solvers (:class:`BiCGSTABSolver`, :class:`GMRESSolver`) are **opt-in**:
their Arnoldi / Gram–Schmidt / Givens iterations are intrinsically sequential
Python loops (the nature of Krylov recurrences, not a missed vectorisation),
so they target testing, teaching, and small or GPU-resident problems where the
scipy bridge is undesirable. For reservoir-scale runs prefer
``SparseDirectSolver`` or :class:`CPRSolver` (which delegates its GMRES
iteration to compiled scipy).

All solvers expose ``solve(J, r) -> dx`` with ``J @ dx ≈ -r``. ``J``
may be a dense ``torch.Tensor`` or a sparse COO/CSR tensor; the iterative
solvers only call ``J @ v`` (matvec). The scipy-bridged solvers accept CPU
tensors only and expose no implicit device-to-host fallback.

Preconditioners expose ``setup(J)`` and ``apply(v) -> M⁻¹·v``. The
default :class:`JacobiPreconditioner` (diagonal scaling) is cheap and
effective for diagonally-dominant flow Jacobians. The scipy-bridged
preconditioners (``ILU0``, ``BILU0``) accept either torch or numpy
input on ``apply`` so they can be plugged into scipy's GMRES via a
``LinearOperator`` while still being usable from a torch Krylov.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from ....core import GeoBrainError

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NoReturn, Protocol

import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]
import torch

from .._defaults import EPS
from ..errors import FlowCapabilityError, FlowConvergenceError
from .config import KrylovConfig
from .diagnostics import convergence_diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TORCH_SPARSE_LAYOUTS = {
    torch.sparse_coo,
    torch.sparse_csr,
    torch.sparse_csc,
    torch.sparse_bsr,
    torch.sparse_bsc,
}

_FLOW_REAL_DTYPES = frozenset({torch.float32, torch.float64})
_TORCH_KRYLOV_LAYOUTS = frozenset(
    {torch.strided, torch.sparse_coo, torch.sparse_csr}
)


class _Preconditioner(Protocol):
    def setup(self, matrix: torch.Tensor) -> None: ...

    def apply(self, vector: torch.Tensor) -> torch.Tensor: ...


def _preflight_linear_system(
    matrix: object,
    residual: object,
    *,
    object_name: str,
    supported_layouts: frozenset[torch.layout],
    supported_devices: frozenset[str] | None = None,
    layout_description: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate a Flow linear system before assembly or numerical kernels.

    Flow solves are real, single-right-hand-side systems.  Keeping this gate
    shared prevents one solver from leaking a raw Torch/SciPy exception for a
    combination that another solver already reports as an unavailable
    capability.
    """
    if not isinstance(matrix, torch.Tensor):
        raise FlowCapabilityError(
            f"{object_name} matrix must be a torch tensor",
            object_name=object_name,
            field="matrix",
            expected="torch.Tensor",
            actual=type(matrix).__name__,
        )
    if not isinstance(residual, torch.Tensor):
        raise FlowCapabilityError(
            f"{object_name} residual must be a torch tensor",
            object_name=object_name,
            field="residual",
            expected="torch.Tensor",
            actual=type(residual).__name__,
        )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise FlowCapabilityError(
            f"{object_name} matrix must be two-dimensional and square",
            object_name=object_name,
            field="matrix.shape",
            expected="(n, n)",
            actual=tuple(matrix.shape),
        )
    if residual.ndim != 1 or residual.shape[0] != matrix.shape[0]:
        raise FlowCapabilityError(
            f"{object_name} residual must be a vector matching the matrix",
            object_name=object_name,
            field="residual.shape",
            expected=(matrix.shape[0],),
            actual=tuple(residual.shape),
        )
    if residual.layout != torch.strided:
        raise FlowCapabilityError(
            f"{object_name} residual must use the dense strided layout",
            object_name=object_name,
            field="residual.layout",
            expected=str(torch.strided),
            actual=str(residual.layout),
        )
    if matrix.layout not in supported_layouts:
        expected_layouts = tuple(sorted(str(layout) for layout in supported_layouts))
        description = layout_description or "the declared matrix"
        raise FlowCapabilityError(
            f"{object_name} accepts {description} layouts only",
            object_name=object_name,
            field="matrix.layout",
            expected=expected_layouts,
            actual=str(matrix.layout),
        )
    if matrix.device != residual.device:
        raise FlowCapabilityError(
            f"{object_name} matrix and residual devices must match",
            object_name=object_name,
            field="matrix/residual.device",
            expected=str(matrix.device),
            actual=str(residual.device),
        )
    if matrix.dtype != residual.dtype:
        raise FlowCapabilityError(
            f"{object_name} matrix and residual dtypes must match",
            object_name=object_name,
            field="matrix/residual.dtype",
            expected=str(matrix.dtype),
            actual=str(residual.dtype),
        )
    if matrix.dtype not in _FLOW_REAL_DTYPES:
        raise FlowCapabilityError(
            f"{object_name} supports real float32 and float64 systems only",
            object_name=object_name,
            field="matrix.dtype",
            expected=(str(torch.float32), str(torch.float64)),
            actual=str(matrix.dtype),
        )
    if supported_devices is not None and matrix.device.type not in supported_devices:
        device_message = (
            f"{object_name} is CPU-only and never copies from another device"
            if supported_devices == frozenset({"cpu"})
            else f"{object_name} does not support the requested device"
        )
        raise FlowCapabilityError(
            device_message,
            object_name=object_name,
            field="matrix/residual.device",
            expected=tuple(sorted(supported_devices)),
            actual=str(matrix.device),
        )
    return matrix, residual


def _is_torch_sparse(matrix: torch.Tensor) -> bool:
    return matrix.layout in _TORCH_SPARSE_LAYOUTS


def _matvec(J: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sparse / dense matrix-vector product on the same device."""
    if _is_torch_sparse(J):
        return torch.sparse.mm(J, v.unsqueeze(1)).squeeze(1)
    return J @ v


def _diag(J: torch.Tensor) -> torch.Tensor:
    """Extract the diagonal of a dense or sparse tensor."""
    if _is_torch_sparse(J):
        J = J if J.layout == torch.sparse_coo else J.to_sparse_coo()
        J = J.coalesce()
        idx = J.indices()
        is_diag = idx[0] == idx[1]
        n = J.shape[0]
        d = torch.zeros(n, device=J.device, dtype=J.values().dtype)
        d.index_add_(0, idx[0][is_diag], J.values()[is_diag])
        return d
    return torch.diagonal(J)


def _torch_matrix_is_finite(J: torch.Tensor) -> bool:
    if J.layout == torch.sparse_coo:
        values = J._values()
    elif J.layout != torch.strided:
        values = J.values()
    else:
        values = J
    return bool(torch.isfinite(values).all())


def _apply_1d(
    precond: _Preconditioner,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapt a flow preconditioner (1-D ``apply``) to linalg's column-block
    calling convention: the linalg Krylovs promote a 1-D rhs to an ``(n, 1)``
    column internally, but flow preconditioners (Jacobi/ILU0/MG/...) contract
    on 1-D vectors, a Jacobi ``(n,1) * (n,)`` would silently broadcast to
    ``(n, n)``."""

    def apply(v: torch.Tensor) -> torch.Tensor:
        if v.ndim == 2 and v.shape[1] == 1:
            return precond.apply(v.squeeze(1)).unsqueeze(1)
        return precond.apply(v)

    return apply


# ---------------------------------------------------------------------------
# Preconditioners
# ---------------------------------------------------------------------------


class IdentityPreconditioner:
    """No-op preconditioner: ``M⁻¹·v = v``."""

    def setup(self, J: torch.Tensor) -> None:
        return None

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        return v


class JacobiPreconditioner:
    """Diagonal scaling: ``M = diag(J)``.

    Cheap, robust on diagonally-dominant matrices (typical of TPFA
    flow Jacobians after row scaling). Falls back to identity-scaling
    if any diagonal entry is below ``eps``.
    """

    def __init__(self, eps: float = EPS) -> None:
        self.eps = float(eps)
        self._inv_diag: torch.Tensor | None = None

    def setup(self, J: torch.Tensor) -> None:
        d = _diag(J)
        safe = torch.where(d.abs() > self.eps, d, torch.ones_like(d))
        self._inv_diag = 1.0 / safe

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        if self._inv_diag is None:
            raise GeoBrainError("JacobiPreconditioner.apply called before setup")
        return self._inv_diag * v


# ---------------------------------------------------------------------------
# Linear-solver result container
# ---------------------------------------------------------------------------


@dataclass
class LinearSolveStats:
    """Diagnostic info from an iterative linear solve."""

    converged: bool = False
    iterations: int = 0
    residual_norm: float = float("inf")
    history: list[float] = field(default_factory=list)
    breakdown: bool = False


def _raise_linear_failure(
    *,
    reason: str,
    iterations: int,
    max_iterations: int,
    initial_residual_norm: float,
    residual_norm: float,
    history: list[float] | tuple[float, ...],
    object_name: str,
    cause: Exception | None = None,
) -> NoReturn:
    record = convergence_diagnostics(
        stage="linear",
        converged=False,
        reason=reason,  # type: ignore[arg-type]
        iterations=iterations,
        max_iterations=max_iterations,
        initial_residual_norm=initial_residual_norm,
        residual_norm=residual_norm,
        residual_history=history,
    )
    error = FlowConvergenceError(
        f"{object_name} did not produce a converged finite update",
        object_name=object_name,
        field="linear_solve",
        expected="converged linear update",
        actual=record.to_dict(),
        diagnostics=record,
    )
    if cause is None:
        raise error
    raise error from cause


# ---------------------------------------------------------------------------
# Direct (dense) solver
# ---------------------------------------------------------------------------


class DirectSolver:
    """Dense direct solver via ``torch.linalg.solve``. For ``n_dof < ~5e3``."""

    last_stats: LinearSolveStats | None = None

    def solve(self, J: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        J, residual = _preflight_linear_system(
            J,
            residual,
            object_name=type(self).__name__,
            supported_layouts=frozenset({torch.strided}),
            layout_description="dense strided matrix",
        )
        initial = float(torch.linalg.vector_norm(residual.detach()))
        try:
            if not bool(torch.isfinite(J).all()) or not bool(torch.isfinite(residual).all()):
                _raise_linear_failure(
                    reason="nonfinite",
                    iterations=0,
                    max_iterations=1,
                    initial_residual_norm=initial,
                    residual_norm=initial,
                    history=(initial,),
                    object_name=type(self).__name__,
                )
            if initial == 0.0:
                dx = torch.zeros_like(residual)
                self.last_stats = LinearSolveStats(
                    converged=True,
                    iterations=0,
                    residual_norm=0.0,
                    history=[0.0],
                )
                return dx
            dx = torch.linalg.solve(J, -residual)
        except FlowConvergenceError:
            raise
        except torch.linalg.LinAlgError as error:
            _raise_linear_failure(
                reason="breakdown",
                iterations=0,
                max_iterations=1,
                initial_residual_norm=initial,
                residual_norm=initial,
                history=(initial,),
                object_name=type(self).__name__,
                cause=error,
            )
        if not bool(torch.isfinite(dx).all()):
            _raise_linear_failure(
                reason="nonfinite",
                iterations=1,
                max_iterations=1,
                initial_residual_norm=initial,
                residual_norm=float("inf"),
                history=(initial, float("inf")),
                object_name=type(self).__name__,
            )
        residual_norm = float(
            torch.linalg.vector_norm((_matvec(J, dx) + residual).detach())
        )
        if not np.isfinite(residual_norm):
            _raise_linear_failure(
                reason="nonfinite",
                iterations=1,
                max_iterations=1,
                initial_residual_norm=initial,
                residual_norm=residual_norm,
                history=(initial, residual_norm),
                object_name=type(self).__name__,
            )
        self.last_stats = LinearSolveStats(
            converged=True, iterations=1,
            residual_norm=residual_norm,
            history=[],
        )
        return dx


# ---------------------------------------------------------------------------
# BiCGSTAB
# ---------------------------------------------------------------------------


class BiCGSTABSolver:
    """Biconjugate Gradient Stabilized: pure torch.

    Well-suited to non-symmetric reservoir Jacobians. Optional left
    preconditioner ``M⁻¹``. The defaults are tuned for forward solves
    in Newton iterations; pass a tighter ``tol`` if used as the inner
    solver in a quasi-Newton scheme.
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 500,
        precond: _Preconditioner | None = None,
    ) -> None:
        validated = KrylovConfig(
            method="bicgstab",
            max_iterations=max_iter,
            tolerance=tol,
        )
        self.tol = validated.tolerance
        self.max_iter = validated.max_iterations
        self.precond = precond if precond is not None else JacobiPreconditioner()
        self.last_stats: LinearSolveStats | None = None

    def _raise_failure(
        self,
        *,
        reason: str,
        iterations: int,
        initial_residual_norm: float,
        residual_norm: float,
        history: list[float] | tuple[float, ...],
        breakdown: bool = False,
        cause: Exception | None = None,
    ) -> NoReturn:
        """Publish failure stats before raising the matching typed diagnostic."""
        self.last_stats = LinearSolveStats(
            converged=False,
            iterations=iterations,
            residual_norm=residual_norm,
            history=list(history),
            breakdown=breakdown,
        )
        _raise_linear_failure(
            reason=reason,
            iterations=iterations,
            max_iterations=self.max_iter,
            initial_residual_norm=initial_residual_norm,
            residual_norm=residual_norm,
            history=history,
            object_name=type(self).__name__,
            cause=cause,
        )

    def solve(self, J: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Delegates the BiCGStab recurrence to :func:`geobrain.core.linalg.bicgstab`
        while keeping the flow-specific convergence
        POLICY here: the historical r0-refresh restart on bi-orthogonality
        breakdown becomes up to ``restarts_max`` re-invocations continuing from
        the current iterate (a restart from x is exactly an r0 refresh), and
        the stopping criterion stays the true-residual ``tol * ||b||``.
        """
        from geobrain.core import linalg as _gl

        J, residual = _preflight_linear_system(
            J,
            residual,
            object_name=type(self).__name__,
            supported_layouts=_TORCH_KRYLOV_LAYOUTS,
            layout_description="dense strided, sparse COO, or sparse CSR matrix",
        )
        b = -residual
        b_norm = float(torch.linalg.vector_norm(b))
        if not _torch_matrix_is_finite(J) or not bool(torch.isfinite(b).all()):
            self._raise_failure(
                reason="nonfinite",
                iterations=0,
                initial_residual_norm=b_norm,
                residual_norm=b_norm,
                history=(b_norm,),
            )
        if b_norm == 0.0:
            self.last_stats = LinearSolveStats(
                converged=True,
                iterations=0,
                residual_norm=0.0,
                history=[0.0],
            )
            return torch.zeros_like(b)
        try:
            self.precond.setup(J)
        except RuntimeError as error:
            self._raise_failure(
                reason="breakdown",
                iterations=0,
                initial_residual_norm=b_norm,
                residual_norm=b_norm,
                history=(b_norm,),
                breakdown=True,
                cause=error,
            )

        restarts_max = 5
        x = None
        total_iters = 0
        stats = None
        budget = self.max_iter
        for _ in range(restarts_max + 1):
            x, stats = _gl.bicgstab(
                J, b, M=_apply_1d(self.precond),
                rtol=self.tol, atol=0.0, maxiter=budget, x0=x,
            )
            total_iters += stats.iterations
            budget = self.max_iter - total_iters
            # Converged, budget exhausted, or a clean (non-breakdown) stall:
            # stop. Only a flagged breakdown earns an r0-refresh restart.
            if stats.converged or budget <= 0 or not stats.breakdown:
                break

        if x is None or stats is None or not bool(torch.isfinite(x).all()):
            self._raise_failure(
                reason="nonfinite",
                iterations=total_iters,
                initial_residual_norm=b_norm,
                residual_norm=float("inf"),
                history=(b_norm, float("inf")),
                breakdown=bool(getattr(stats, "breakdown", False)),
            )
        true_residual_norm = float(
            torch.linalg.vector_norm((_matvec(J, x) - b).detach())
        )
        true_history = (
            [b_norm]
            if total_iters == 0 and true_residual_norm == b_norm
            else [b_norm, true_residual_norm]
        )
        certified = (
            np.isfinite(true_residual_norm)
            and true_residual_norm <= self.tol * b_norm
        )
        self.last_stats = LinearSolveStats(
            converged=certified,
            iterations=total_iters,
            residual_norm=true_residual_norm,
            history=true_history,
            breakdown=bool(getattr(stats, "breakdown", False)),
        )
        if not np.isfinite(true_residual_norm):
            self._raise_failure(
                reason="nonfinite",
                iterations=total_iters,
                initial_residual_norm=b_norm,
                residual_norm=true_residual_norm,
                history=true_history,
                breakdown=self.last_stats.breakdown,
            )
        if not self.last_stats.converged:
            recurrence_breakdown = self.last_stats.breakdown or bool(stats.converged)
            self._raise_failure(
                reason="breakdown" if recurrence_breakdown else "max_iterations",
                iterations=total_iters,
                initial_residual_norm=b_norm,
                residual_norm=self.last_stats.residual_norm,
                history=true_history,
                breakdown=recurrence_breakdown,
            )
        return x


# ---------------------------------------------------------------------------
# Restarted GMRES
# ---------------------------------------------------------------------------


class GMRESSolver:
    """Restarted GMRES(m): pure torch.

    Memory cost scales as ``O(restart · n_dof)``. Robust on
    ill-conditioned and indefinite systems where BiCGSTAB can stall.
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 500,
        restart: int = 50,
        precond: _Preconditioner | None = None,
    ) -> None:
        validated = KrylovConfig(
            method="gmres",
            max_iterations=max_iter,
            tolerance=tol,
            restart=restart,
        )
        self.tol = validated.tolerance
        self.max_iter = validated.max_iterations
        self.restart = validated.restart or 1
        self.precond = precond if precond is not None else JacobiPreconditioner()
        self.last_stats: LinearSolveStats | None = None

    def solve(self, J: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Delegates the Arnoldi/Givens recurrence to :func:`geobrain.core.linalg.gmres`.
        The inner iteration uses its natural
        left-preconditioned relative norm, but Flow accepts the update only
        after independently certifying the original system:
        ``||b - J x|| <= tol * ||b||``. This final check prevents a strongly
        scaling preconditioner from declaring a small nonzero right-hand side
        converged at the zero iterate.
        """
        from geobrain.core import linalg as _gl

        J, residual = _preflight_linear_system(
            J,
            residual,
            object_name=type(self).__name__,
            supported_layouts=_TORCH_KRYLOV_LAYOUTS,
            layout_description="dense strided, sparse COO, or sparse CSR matrix",
        )
        b = -residual
        b_norm = float(torch.linalg.vector_norm(b))
        if not _torch_matrix_is_finite(J) or not bool(torch.isfinite(b).all()):
            _raise_linear_failure(
                reason="nonfinite",
                iterations=0,
                max_iterations=self.max_iter,
                initial_residual_norm=b_norm,
                residual_norm=b_norm,
                history=(b_norm,),
                object_name=type(self).__name__,
            )
        if b_norm == 0.0:
            self.last_stats = LinearSolveStats(
                converged=True, iterations=0, residual_norm=0.0,
            )
            return torch.zeros_like(b)
        try:
            self.precond.setup(J)
        except RuntimeError as error:
            _raise_linear_failure(
                reason="breakdown",
                iterations=0,
                max_iterations=self.max_iter,
                initial_residual_norm=b_norm,
                residual_norm=b_norm,
                history=(b_norm,),
                object_name=type(self).__name__,
                cause=error,
            )

        x, stats = _gl.gmres(
            J, b, M=_apply_1d(self.precond), restart=self.restart,
            rtol=self.tol, atol=0.0, maxiter=self.max_iter,
            check_every=1,  # historical flow semantics: test every iteration
        )
        if not bool(torch.isfinite(x).all()):
            _raise_linear_failure(
                reason="nonfinite",
                iterations=int(stats.iterations),
                max_iterations=self.max_iter,
                initial_residual_norm=b_norm,
                residual_norm=float("inf"),
                history=(b_norm, float("inf")),
                object_name=type(self).__name__,
            )
        true_residual_norm = float(
            torch.linalg.vector_norm((_matvec(J, x) - b).detach())
        )
        true_history = (
            [b_norm]
            if int(stats.iterations) == 0 and true_residual_norm == b_norm
            else [b_norm, true_residual_norm]
        )
        certified = (
            bool(stats.converged)
            and np.isfinite(true_residual_norm)
            and true_residual_norm <= self.tol * b_norm
        )
        self.last_stats = LinearSolveStats(
            converged=certified,
            iterations=int(stats.iterations),
            residual_norm=true_residual_norm,
            history=true_history,
            breakdown=bool(getattr(stats, "breakdown", False)),
        )
        if not np.isfinite(float(stats.residual_norm)) or not np.isfinite(
            true_residual_norm
        ):
            _raise_linear_failure(
                reason="nonfinite",
                iterations=self.last_stats.iterations,
                max_iterations=self.max_iter,
                initial_residual_norm=b_norm,
                residual_norm=self.last_stats.residual_norm,
                history=self.last_stats.history,
                object_name=type(self).__name__,
            )
        if not self.last_stats.converged:
            _raise_linear_failure(
                reason=(
                    "breakdown"
                    if self.last_stats.breakdown or bool(stats.converged)
                    else "max_iterations"
                ),
                iterations=self.last_stats.iterations,
                max_iterations=self.max_iter,
                initial_residual_norm=b_norm,
                residual_norm=self.last_stats.residual_norm,
                history=self.last_stats.history,
                object_name=type(self).__name__,
            )
        return x


# ---------------------------------------------------------------------------
# scipy bridge helpers (FLOW2 advanced solvers)
# ---------------------------------------------------------------------------


def _to_scipy_csr(J: object) -> sp.csr_matrix:
    """
    Materialise ``J`` as a scipy ``csr_matrix``.

    Accepts: scipy sparse, torch sparse_coo, dense torch. Used by the
    scipy-bridged solvers / preconditioners below.
    """
    if isinstance(J, sp.spmatrix):
        return J.tocsr()
    if not isinstance(J, torch.Tensor):
        raise FlowCapabilityError(
            "scipy bridge expects a scipy matrix or CPU torch tensor",
            object_name="_to_scipy_csr",
            field="matrix",
            expected="scipy sparse or CPU torch.Tensor",
            actual=type(J).__name__,
        )
    if J.device.type != "cpu":
        raise FlowCapabilityError(
            "scipy sparse operations are CPU-only",
            object_name="_to_scipy_csr",
            field="matrix.device",
            expected="cpu",
            actual=str(J.device),
            hint="use a pure-torch Krylov solver on this device",
        )
    tensor: torch.Tensor = J
    if _is_torch_sparse(tensor):
        coo: torch.Tensor = (
            tensor if tensor.layout == torch.sparse_coo else tensor.to_sparse_coo()
        )
        coo = coo.coalesce()
        idx = coo.indices().detach().numpy()
        val = coo.values().detach().numpy()
        shape = tuple(coo.shape)
        return sp.coo_matrix((val, (idx[0], idx[1])), shape=shape).tocsr()
    return sp.csr_matrix(tensor.detach().numpy())


# ---------------------------------------------------------------------------
# SparseDirectSolver: scipy spsolve
# ---------------------------------------------------------------------------


class SparseDirectSolver:
    """
    Scipy sparse LU. Forward-only (no autograd through the solve).

    Suitable for ``n_dof < ~100k`` reservoir problems. The forward
    Newton path of :class:`ImplicitFlowEvolutionOperator` runs under
    ``torch.no_grad``, so the lack of autograd here is by design, the
    implicit-FT backward handles gradients separately.
    """

    last_stats: LinearSolveStats | None = None

    def solve(self, J: object, residual: torch.Tensor) -> torch.Tensor:
        J, residual = _preflight_linear_system(
            J,
            residual,
            object_name=type(self).__name__,
            supported_layouts=frozenset({torch.sparse_coo, torch.sparse_csr}),
            supported_devices=frozenset({"cpu"}),
            layout_description="sparse COO or sparse CSR matrix",
        )
        device, dtype = residual.device, residual.dtype
        initial = float(torch.linalg.vector_norm(residual.detach()))
        J_csc = _to_scipy_csr(J).tocsc()
        b = -residual.detach().numpy()
        if not np.isfinite(J_csc.data).all() or not np.isfinite(b).all():
            _raise_linear_failure(
                reason="nonfinite",
                iterations=0,
                max_iterations=1,
                initial_residual_norm=initial,
                residual_norm=initial,
                history=(initial,),
                object_name=type(self).__name__,
            )
        if initial == 0.0:
            dx = torch.zeros_like(residual)
            self.last_stats = LinearSolveStats(
                converged=True,
                iterations=0,
                residual_norm=0.0,
                history=[0.0],
            )
            return dx
        try:
            x = spla.spsolve(J_csc, b)
        except RuntimeError as error:
            _raise_linear_failure(
                reason="breakdown",
                iterations=0,
                max_iterations=1,
                initial_residual_norm=initial,
                residual_norm=initial,
                history=(initial,),
                object_name=type(self).__name__,
                cause=error,
            )
        if not np.isfinite(x).all():
            _raise_linear_failure(
                reason="nonfinite",
                iterations=1,
                max_iterations=1,
                initial_residual_norm=initial,
                residual_norm=float("inf"),
                history=(initial, float("inf")),
                object_name=type(self).__name__,
            )
        dx = torch.tensor(x, device=device, dtype=dtype)
        # Residual norm via numpy to avoid an unnecessary torch matvec.
        r_after = J_csc @ x - b
        residual_norm = float(np.linalg.norm(r_after))
        if not np.isfinite(residual_norm):
            _raise_linear_failure(
                reason="nonfinite",
                iterations=1,
                max_iterations=1,
                initial_residual_norm=initial,
                residual_norm=residual_norm,
                history=(initial, residual_norm),
                object_name=type(self).__name__,
            )
        self.last_stats = LinearSolveStats(
            converged=True,
            iterations=1,
            residual_norm=residual_norm,
            history=[],
        )
        return dx


# ---------------------------------------------------------------------------
# ILU0 / BILU0 preconditioners (scipy-bridged)
# ---------------------------------------------------------------------------


class ILU0Preconditioner:
    """
    Scipy ``spilu`` ILU(0) preconditioner.

    Used either inside a torch Krylov (``.apply`` returns torch) or
    through scipy's ``LinearOperator`` (``.apply_numpy`` returns numpy).
    """

    def __init__(self, fill_factor: float = 1.0, drop_tol: float = 1e-4) -> None:
        self.fill_factor = float(fill_factor)
        self.drop_tol = float(drop_tol)
        self._lu = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def setup(self, J: object) -> None:
        J_csc = _to_scipy_csr(J).tocsc()
        self._lu = spla.spilu(
            J_csc, fill_factor=self.fill_factor, drop_tol=self.drop_tol,
        )
        if isinstance(J, torch.Tensor):
            self._device = J.device
            self._dtype = J.values().dtype if J.is_sparse else J.dtype

    def apply_numpy(self, r: np.ndarray) -> np.ndarray:
        if self._lu is None:
            raise GeoBrainError("ILU0Preconditioner.apply called before setup")
        return self._lu.solve(r)

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        if self._lu is None:
            raise GeoBrainError("ILU0Preconditioner.apply called before setup")
        if v.device.type != "cpu":
            raise FlowCapabilityError(
                "ILU0Preconditioner is CPU-only and never copies from another device",
                object_name="ILU0Preconditioner",
                field="vector.device",
                expected="cpu",
                actual=str(v.device),
            )
        v_np = v.detach().numpy()
        x_np = self._lu.solve(v_np)
        dtype = self._dtype or v.dtype
        return torch.tensor(x_np, device="cpu", dtype=dtype)


class BILU0Preconditioner:
    """
    Block ILU(0) on per-variable ordered block-structured Jacobians.

    State convention: ``[var_0_c0..var_0_c{n-1}, var_1_c0..var_1_c{n-1}, ...]``.
    Internally permutes to per-cell ordering and applies scipy's spilu;
    point-ILU on the permuted matrix is exactly block-ILU on the
    original block pattern.
    """

    def __init__(self, n_cells: int, n_vars: int) -> None:
        self.n_cells = int(n_cells)
        self.n_vars = int(n_vars)
        n = self.n_cells * self.n_vars
        perm: np.ndarray = np.zeros(n, dtype=np.int64)
        for c in range(self.n_cells):
            for v in range(self.n_vars):
                perm[self.n_vars * c + v] = v * self.n_cells + c
        self._perm = perm
        self._inv_perm = np.argsort(perm)
        self._lu = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def setup(self, J: object) -> None:
        Jcsr = _to_scipy_csr(J)
        P = sp.eye(Jcsr.shape[0], format="csr")[self._perm]
        Jp = P @ Jcsr @ P.T
        self._lu = spla.spilu(Jp.tocsc(), fill_factor=1.0, drop_tol=1e-4)
        if isinstance(J, torch.Tensor):
            self._device = J.device
            self._dtype = J.values().dtype if J.is_sparse else J.dtype

    def apply_numpy(self, r: np.ndarray) -> np.ndarray:
        if self._lu is None:
            raise GeoBrainError("BILU0Preconditioner.apply called before setup")
        r_p = r[self._perm]
        x_p = self._lu.solve(r_p)
        return x_p[self._inv_perm]

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        if self._lu is None:
            raise GeoBrainError("BILU0Preconditioner.apply called before setup")
        if v.device.type != "cpu":
            raise FlowCapabilityError(
                "BILU0Preconditioner is CPU-only and never copies from another device",
                object_name="BILU0Preconditioner",
                field="vector.device",
                expected="cpu",
                actual=str(v.device),
            )
        v_np = v.detach().numpy()
        x_np = self.apply_numpy(v_np)
        dtype = self._dtype or v.dtype
        return torch.tensor(x_np, device="cpu", dtype=dtype)


# ---------------------------------------------------------------------------
# MultigridSolver: Krylov-style wrapper around GeometricMultigrid
# ---------------------------------------------------------------------------


class MultigridSolver:
    """
    Geometric MG outer solver (Cartesian only).

    Thin wrapper exposing the standard ``solve(J, residual)`` API by
    delegating to :class:`~geobrain.physics.flow.solvers.GeometricMultigrid`.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        n_pre: int = 2,
        n_post: int = 2,
        omega: float = 2.0 / 3.0,
        n_coarsest: int = 50,
        tol: float = 1e-8,
        max_iter: int = 50,
    ) -> None:
        # Lazy import to keep the linear_solvers ↔ multigrid edge one-way
        # (multigrid never imports linear_solvers).
        from .multigrid import GeometricMultigrid
        self.mg = GeometricMultigrid(
            nx,
            ny,
            nz,
            n_pre=n_pre,
            n_post=n_post,
            omega=omega,
            n_coarsest=n_coarsest,
            tol=tol,
            max_iter=max_iter,
        )
        self.last_stats: LinearSolveStats | None = None

    def solve(self, J: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        J, residual = _preflight_linear_system(
            J,
            residual,
            object_name=type(self).__name__,
            supported_layouts=_TORCH_KRYLOV_LAYOUTS,
            supported_devices=frozenset({"cpu", "cuda"}),
            layout_description="dense strided, sparse COO, or sparse CSR matrix",
        )
        dx = self.mg.solve(J, residual)
        # Residual norm at the returned iterate.
        if _is_torch_sparse(J):
            r = -residual - torch.sparse.mm(J, dx.view(-1, 1)).view(-1)
        else:
            r = -residual - J @ dx
        r_norm = float(torch.linalg.vector_norm(r))
        b_norm = float(torch.linalg.vector_norm(residual).clamp_min(EPS))
        diagnostics = self.mg.last_diagnostics
        self.last_stats = LinearSolveStats(
            converged=(r_norm / b_norm) < self.mg.effective_tolerance,
            iterations=(
                diagnostics.iterations if diagnostics is not None else self.mg.max_iter
            ),
            residual_norm=r_norm,
            history=(
                list(diagnostics.residual_history) if diagnostics is not None else []
            ),
        )
        if not self.last_stats.converged:
            _raise_linear_failure(
                reason="max_iterations",
                iterations=self.last_stats.iterations,
                max_iterations=self.mg.max_iter,
                initial_residual_norm=b_norm,
                residual_norm=r_norm,
                history=(),
                object_name=type(self).__name__,
            )
        return dx


# ---------------------------------------------------------------------------
# CPRSolver: Constrained Pressure Residual
# ---------------------------------------------------------------------------


class CPRSolver:
    """
    CPR: outer scipy-GMRES with two-stage preconditioner.

    Stage 1: pressure restriction → MG solve on the pressure sub-block.
    Stage 2: BILU0 on the full block-structured Jacobian.

    Pressure restriction here uses the True-IMPES trivial form: take
    rows of variable 0 in the per-variable ordered state vector. The
    layout convention is ``[p_0..p_{n-1}, sw_0..sw_{n-1}, ...]``;
    pressure is variable 0, saturation(s) variable 1+.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        n_cells: int,
        n_vars: int,
        tol: float = 1e-6,
        max_iter: int = 100,
        restart: int = 30,
    ) -> None:
        from .multigrid import GeometricMultigrid
        validated = KrylovConfig(
            method="gmres",
            max_iterations=max_iter,
            tolerance=tol,
            restart=restart,
        )
        self.n_cells = int(n_cells)
        self.n_vars = int(n_vars)
        self.bilu0 = BILU0Preconditioner(n_cells, n_vars)
        self.mg = GeometricMultigrid(nx, ny, nz)
        self.tol = validated.tolerance
        self.max_iter = validated.max_iterations
        self.restart = validated.restart or 1
        self.last_stats: LinearSolveStats | None = None

    def _press_restrict(self, A_full: sp.csr_matrix) -> sp.csr_matrix:
        """Extract the cell-by-cell pressure sub-matrix (variable 0)."""
        n = self.n_cells
        expected = n * self.n_vars
        if A_full.shape != (expected, expected):
            raise GeoBrainError(
                f"CPR expected Jacobian shape ({expected}, {expected}) "
                f"for n_cells={n}, n_vars={self.n_vars}; got {A_full.shape}.",
            )
        return A_full[:n, :n]

    def solve(self, J: object, residual: torch.Tensor) -> torch.Tensor:
        J, residual = _preflight_linear_system(
            J,
            residual,
            object_name=type(self).__name__,
            supported_layouts=_TORCH_KRYLOV_LAYOUTS,
            supported_devices=frozenset({"cpu"}),
            layout_description="dense strided, sparse COO, or sparse CSR matrix",
        )
        device, dtype = residual.device, residual.dtype
        initial = float(torch.linalg.vector_norm(residual.detach()))
        try:
            J_csc = _to_scipy_csr(J).tocsc()
            b = -residual.detach().numpy()
            if not np.isfinite(J_csc.data).all() or not np.isfinite(b).all():
                _raise_linear_failure(
                    reason="nonfinite",
                    iterations=0,
                    max_iterations=self.max_iter,
                    initial_residual_norm=initial,
                    residual_norm=initial,
                    history=(initial,),
                    object_name=type(self).__name__,
                )
            if initial == 0.0:
                dx = torch.zeros_like(residual)
                self.last_stats = LinearSolveStats(
                    converged=True,
                    iterations=0,
                    residual_norm=0.0,
                    history=[0.0],
                )
                return dx
            n_p = self.n_cells

            # Pressure submatrix → torch sparse for GeometricMultigrid.
            A_pp = self._press_restrict(J_csc.tocsr()).tocsr().tocoo()
            idx = torch.tensor(
                np.vstack([A_pp.row, A_pp.col]), dtype=torch.int64, device=device,
            )
            vals = torch.tensor(A_pp.data, dtype=dtype, device=device)
            A_pp_t = torch.sparse_coo_tensor(
                idx, vals, size=A_pp.shape, device=device, dtype=dtype,
            ).coalesce()
            self.mg.setup(A_pp_t)
            self.bilu0.setup(J_csc)

            def precond_apply(r: np.ndarray) -> np.ndarray:
                # Pressure pre-correction.
                r_p = r[:n_p]
                r_p_t = torch.tensor(r_p, device=device, dtype=dtype)
                dp_t = self.mg.apply(r_p_t)
                dp = dp_t.detach().numpy()
                dx = np.zeros_like(r)
                dx[:n_p] = dp
                # Residual after pressure correction; BILU0 on the remainder.
                r_after = r - J_csc @ dx
                dx2 = self.bilu0.apply_numpy(r_after)
                result: np.ndarray = dx + dx2
                return result

            M = spla.LinearOperator(J_csc.shape, matvec=precond_apply)
            x, info = spla.gmres(
                J_csc, b,
                rtol=self.tol, maxiter=self.max_iter, restart=self.restart, M=M,
            )
        except FlowConvergenceError:
            raise
        except RuntimeError as error:
            _raise_linear_failure(
                reason="breakdown",
                iterations=0,
                max_iterations=self.max_iter,
                initial_residual_norm=initial,
                residual_norm=initial,
                history=(initial,),
                object_name=type(self).__name__,
                cause=error,
            )
        if not np.isfinite(x).all():
            _raise_linear_failure(
                reason="nonfinite",
                iterations=0,
                max_iterations=self.max_iter,
                initial_residual_norm=initial,
                residual_norm=float("inf"),
                history=(initial, float("inf")),
                object_name=type(self).__name__,
            )
        dx = torch.tensor(x, device=device, dtype=dtype)
        r_after = J_csc @ x - b
        residual_norm = float(np.linalg.norm(r_after))
        if not np.isfinite(residual_norm):
            _raise_linear_failure(
                reason="nonfinite",
                iterations=0,
                max_iterations=self.max_iter,
                initial_residual_norm=initial,
                residual_norm=residual_norm,
                history=(initial, residual_norm),
                object_name=type(self).__name__,
            )
        converged = info == 0 and residual_norm <= self.tol * initial
        self.last_stats = LinearSolveStats(
            converged=converged,
            iterations=int(info) if info > 0 else 1,
            residual_norm=residual_norm,
            history=[initial, residual_norm],
        )
        if not self.last_stats.converged:
            _raise_linear_failure(
                reason="breakdown" if info <= 0 else "max_iterations",
                iterations=self.last_stats.iterations,
                max_iterations=self.max_iter,
                initial_residual_norm=float(np.linalg.norm(b)),
                residual_norm=self.last_stats.residual_norm,
                history=self.last_stats.history,
                object_name=type(self).__name__,
            )
        return dx


__all__ = [
    "BILU0Preconditioner",
    "BiCGSTABSolver",
    "CPRSolver",
    "DirectSolver",
    "GMRESSolver",
    "ILU0Preconditioner",
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "LinearSolveStats",
    "MultigridSolver",
    "SparseDirectSolver",
]
