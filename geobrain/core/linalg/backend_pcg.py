"""
GPU Jacobi-preconditioned Conjugate-Gradient backend for the sparse implicit-VJP
bridge (:class:`~geobrain.core.adjoint.SparseFactorSolver`).

:class:`PcgGpuSolver` implements the two primitives the sparse adjoint machinery
needs, ``factor(indices, values, shape) -> handle`` and
``solve(handle, b, *, adjoint) -> x``: entirely in torch on CUDA sparse
tensors. It is the GPU counterpart to the default CPU
:class:`~geobrain.core.adjoint.ScipySpluSolver`: install it via
:func:`~geobrain.core.adjoint.set_default_sparse_solver` (or the scoped
:func:`~geobrain.core.adjoint.sparse_solver`) and every sparse forward that
routes through :func:`sparse_linear_solve_with_adjoint` (DC/IP/SIP Poisson) runs
its solve on the GPU with zero operator changes.

This class is a THIN DELEGATE over the unified
:mod:`geobrain.core.linalg` stack: the CG loop is :func:`geobrain.core.linalg.cg` (the
same complex-correct, per-column-batched Jacobi-PCG the ``KrylovSolver`` backend
uses), so the platform has ONE CG implementation. What this class adds on top is
the DC-specific policy: the forced-CUDA device home, the factor-time random-probe
symmetry check, the configurable ``diag_floor`` Jacobi clamp (sign-preserving,
distinct from ``linalg.precond.Jacobi``'s identity fallback), and HARD errors on
non-convergence / breakdown (an SPD-contract violation raises "not positive
definite" at the first bad iteration instead of grinding to ``maxiter``). The
:mod:`geobrain.core.linalg` is imported at module level, post-option-A the
seam lives in ``core.linalg`` and linalg no longer imports ``core.adjoint``,
so the import cycle that once forced function-local imports here is gone.

Method: Jacobi preconditioner ``M_inv = 1/diag(A)`` (diagonal clamped away from
zero), CG matvec via ``torch.sparse.mm``, relative-residual stopping
``||r||/||b|| < tol``, a ``maxiter`` cap, and a HARD error on non-convergence.
All ``k`` RHS columns are solved simultaneously (batched CG with per-column
scalars).

Applicability: REAL SYMMETRIC POSITIVE DEFINITE systems ONLY. The DC/IP/SIP
Poisson operator is real SPD once the Dirichlet pin (or a Robin boundary)
removes the pure-Neumann nullspace, so CG applies and the adjoint solve
``Aᵀ y = b`` equals the forward solve (implemented against a stored transpose
for generality; a no-op for the symmetric case). A cheap random-probe symmetry
check at factor time refuses a non-symmetric matrix, and complex inputs are
rejected outright; this is NOT for complex EM / Helmholtz systems, whose
indefinite complex operators need a direct or Krylov (GMRES/BiCGStab) backend.

CONVERGENCE CAVEAT. Jacobi is a WEAK preconditioner: the CG iteration count
grows with the conditioning of ``A``. High resistivity-contrast models, very
fine meshes, or badly graded cells can drive the iteration count up (or, past
``maxiter``, trigger the non-convergence error). For those regimes raise
``maxiter``/relax ``tol``, or use a stronger preconditioner (e.g. algebraic
multigrid), not provided here. On the well-conditioned DC systems it targets it
converges in a few hundred iterations and is dramatically faster than a CPU
sparse LU as the mesh grows (the LU factorization cost is super-quadratic).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any

import torch

from ..errors import GeoBrainError
# Module-level: the seam lives in core.linalg,
# so linalg no longer imports core.adjoint and the historical cycle is gone.
from .krylov import cg
from .linear_operator import LinearOperator, _spmv
from .seam import SparseFactorSolver


class _PcgHandle:
    """Opaque factor handle: the CUDA operator, its transpose, and M_inv."""

    __slots__ = ("A", "A_t", "M_inv", "n", "dtype", "device")

    def __init__(self, A, A_t, M_inv, n, dtype, device):
        self.A = A          # CSR (n, n) on CUDA, forward matvec operator
        self.A_t = A_t      # CSR (n, n) on CUDA, transpose (adjoint matvec)
        self.M_inv = M_inv  # (n,) dense, 1/diag(A), clamped
        self.n = n
        self.dtype = dtype
        self.device = device


class PcgGpuSolver(SparseFactorSolver):
    """Jacobi-PCG sparse solver on CUDA (real SPD systems, float64).

    A drop-in :class:`SparseFactorSolver` backend for the sparse implicit-VJP
    bridge; the CG loop delegates to :func:`geobrain.core.linalg.cg`. See the module
    docstring for the SPD-only applicability and the Jacobi convergence caveat.

    Args:
        tol: relative-residual stopping tolerance ``||r||/||b|| < tol``.
        maxiter: maximum CG iterations before raising a non-convergence error.
        device: CUDA device for the operator and solves (default ``"cuda"``).
        diag_floor: minimum |diag| used to build ``M_inv`` (avoids /0).
        check_symmetry: run a cheap random-probe symmetry test at factor time
            and raise if the matrix is not symmetric (CG needs SPD).
        store_iterations: keep the last solve's iteration count / achieved
            relative residual on ``.last_iters`` / ``.last_residual`` (for a
            benchmark / diagnostic harness). Does not affect numerics.
    """

    def __init__(
        self,
        *,
        tol: float = 1e-7,
        maxiter: int = 5000,
        device: str | torch.device = "cuda",
        diag_floor: float = 1e-30,
        check_symmetry: bool = True,
        store_iterations: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise GeoBrainError(
                "PcgGpuSolver requires CUDA but torch.cuda.is_available() is False",
                object_name="PcgGpuSolver", field="device",
                expected="a CUDA device", actual="no CUDA",
            )
        self.tol = float(tol)
        self.maxiter = int(maxiter)
        self.device = torch.device(device)
        self.diag_floor = float(diag_floor)
        self.check_symmetry = bool(check_symmetry)
        self.store_iterations = bool(store_iterations)
        # Diagnostics from the most recent solve (forward or adjoint).
        self.last_iters: int | None = None
        self.last_residual: float | None = None

    # ------------------------------------------------------------------ factor
    def factor(self, indices: torch.Tensor, values: torch.Tensor, shape) -> Any:
        """Build the CUDA operator and Jacobi preconditioner.

        ``indices`` is ``(2, nnz)`` ``(row, col)`` (COO-style; the bridge has
        already coalesced COO / expanded CSR before calling us). ``values`` is
        ``(nnz,)``. We store a coalesced CUDA CSR for fast repeated matvecs, its
        transpose for the adjoint solve, and ``M_inv = 1/diag(A)``.
        """
        n_rows, n_cols = int(shape[0]), int(shape[1])
        if n_rows != n_cols:
            raise GeoBrainError(
                "PcgGpuSolver: A must be square",
                object_name="PcgGpuSolver", field="shape",
                expected="(n, n)", actual=(n_rows, n_cols),
            )
        n = n_rows

        vals = values.detach()
        if torch.is_complex(vals):
            raise GeoBrainError(
                "PcgGpuSolver is a CG solver; it only supports REAL symmetric "
                "positive-definite matrices (float64), not complex systems "
                "(use a direct/Krylov backend for complex EM/Helmholtz)",
                object_name="PcgGpuSolver", field="values.dtype",
                expected="float64", actual=str(vals.dtype),
            )
        if not bool(torch.isfinite(vals).all()):
            raise GeoBrainError(
                "PcgGpuSolver: sparse matrix values are not finite (NaN/Inf), "
                "check the property model (e.g. σ) used to assemble the system",
                object_name="PcgGpuSolver", field="values",
                expected="all entries finite", actual="NaN/Inf present",
            )

        idx = indices.detach().to(device=self.device, dtype=torch.long)
        vals = vals.to(device=self.device, dtype=torch.float64)

        # Coalesced COO on CUDA (sum any duplicate (row, col) once).
        A_coo = torch.sparse_coo_tensor(idx, vals, (n, n)).coalesce()
        ci = A_coo.indices()
        cv = A_coo.values()

        # --- diagonal → Jacobi preconditioner ---------------------------------
        # Extract diag by scattering the on-diagonal nnz into a length-n vector.
        # Clamp magnitude away from 0 while PRESERVING SIGN, then invert,
        # deliberately different from linalg.precond.Jacobi (which falls back to
        # the identity on a zero diagonal): ``diag_floor`` is a public knob here.
        diag = torch.zeros(n, dtype=torch.float64, device=self.device)
        on_diag = ci[0] == ci[1]
        diag = diag.scatter_add(0, ci[0][on_diag], cv[on_diag])
        sign = torch.where(diag >= 0, 1.0, -1.0)
        mag = diag.abs().clamp(min=self.diag_floor)
        M_inv = 1.0 / (sign * mag)

        # Store CSR for fast matvec; build the transpose operator for adjoint.
        A_csr = A_coo.to_sparse_csr()
        A_t_coo = torch.sparse_coo_tensor(
            torch.stack([ci[1], ci[0]]), cv, (n, n)
        ).coalesce()
        A_t_csr = A_t_coo.to_sparse_csr()

        if self.check_symmetry:
            self._assert_symmetric(A_csr, A_t_csr, n)

        return _PcgHandle(A_csr, A_t_csr, M_inv, n, torch.float64, self.device)

    def _assert_symmetric(self, A, A_t, n: int) -> None:
        """Cheap random-probe symmetry check: ``A u == Aᵀ u`` for random ``u``.

        CG requires an SPD (hence symmetric) operator; a silently non-symmetric
        matrix would make CG converge to the wrong vector or stall. Two random
        probes catch any asymmetry with probability 1 (up to rounding).
        """
        g = torch.Generator(device=self.device).manual_seed(0)
        u = torch.randn(n, 2, generator=g, dtype=torch.float64, device=self.device)
        au = torch.sparse.mm(A, u)
        atu = torch.sparse.mm(A_t, u)
        num = (au - atu).abs().max()
        den = au.abs().max().clamp(min=1e-300)
        rel = float(num / den)
        if rel > 1e-9:
            raise GeoBrainError(
                "PcgGpuSolver: matrix is not symmetric (relative A−Aᵀ probe "
                f"residual {rel:.3e}); CG needs a symmetric positive-definite "
                "operator. Use a direct/GMRES backend for non-symmetric systems.",
                object_name="PcgGpuSolver", field="A",
                expected="symmetric (Aᵀ == A)", actual=f"asymmetry {rel:.3e}",
            )

    # ------------------------------------------------------------------- solve
    def solve(self, handle: Any, b: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
        """Solve ``A x = b`` (or ``Aᵀ x = b`` when ``adjoint``) by Jacobi-PCG.

        Returns ``x`` on ``b.device`` with ``b``'s original shape. For the
        symmetric DC operator the adjoint branch is numerically identical; it is
        implemented against the stored transpose for correctness on any
        (symmetric) system. Raises on breakdown (not positive definite, flagged
        by the delegated :func:`geobrain.core.linalg.cg` at the first non-positive-
        curvature iteration) and on non-convergence.
        """
        h: _PcgHandle = handle
        out_device = b.device
        out_shape = b.shape

        b_work = b.detach().to(device=h.device, dtype=torch.float64)
        squeeze = b_work.ndim == 1
        if squeeze:
            b_work = b_work.unsqueeze(1)
        if b_work.shape[0] != h.n:
            raise GeoBrainError(
                "PcgGpuSolver: b leading dim must match A",
                object_name="PcgGpuSolver", field="b.shape",
                expected=f"({h.n}, k)", actual=tuple(b.shape),
            )

        A_mat = h.A_t if adjoint else h.A
        op = LinearOperator(
            shape=(h.n, h.n),
            matvec=lambda v: _spmv(A_mat, v),
            dtype=h.dtype,
            device=h.device,
            hermitian=True,   # real symmetric; rmatvec is never used by cg
        )
        M_inv = h.M_inv

        def jacobi(v: torch.Tensor) -> torch.Tensor:
            return (M_inv.unsqueeze(-1) * v) if v.ndim > 1 else M_inv * v

        x, stats = cg(
            op, b_work, M=jacobi, rtol=self.tol, atol=0.0, maxiter=self.maxiter,
        )

        if stats.breakdown:
            raise GeoBrainError(
                "PcgGpuSolver: non-positive curvature pᵀAp <= 0 on an active "
                "column; the operator is not positive definite (CG requires "
                "SPD). Use a direct or GMRES/BiCGStab backend for indefinite "
                "systems.",
                object_name="PcgGpuSolver", field="A",
                expected="positive definite (pᵀAp > 0)",
                actual=f"breakdown after {stats.iterations} iterations",
            )

        # Achieved TRUE relative residual, max over RHS columns (one extra
        # matvec; also feeds .last_residual diagnostics below).
        r = b_work - _spmv(A_mat, x)
        b_norm_safe = torch.linalg.vector_norm(b_work, dim=0).clamp(min=1e-300)
        rel = float((torch.linalg.vector_norm(r, dim=0) / b_norm_safe).max())

        if self.store_iterations:
            self.last_iters = stats.iterations
            self.last_residual = rel

        if not stats.converged:
            raise GeoBrainError(
                f"PcgGpuSolver: CG failed to converge in {self.maxiter} iters "
                f"(achieved relative residual {rel:.3e}, tol {self.tol:.1e}). "
                "The system may be ill-conditioned or not SPD; raise maxiter, "
                "relax tol, or use a stronger preconditioner (e.g. AMG).",
                object_name="PcgGpuSolver", field="convergence",
                expected=f"||r||/||b|| < {self.tol:.1e}",
                actual=f"{rel:.3e} after {self.maxiter} iters",
            )

        x = x.squeeze(1) if squeeze else x
        return x.to(device=out_device, dtype=b.dtype).reshape(out_shape)
