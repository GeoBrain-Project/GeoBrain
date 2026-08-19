"""
Implicit-differentiation adjoint for the Krylov solve.

``solve_adjoint(A, b)`` returns ``x = A^{-1} b`` and is differentiable w.r.t. both ``A``
(dense or sparse assembled matrix) and ``b`` WITHOUT unrolling autograd through the solver
iterations. The backward solves the adjoint system ``A^H λ = ∂L/∂x`` with the same Krylov
method and forms

    grad_b = λ,     grad_A = -λ xᴴ   (restricted to A's sparsity pattern when A is sparse)

which is exactly ``torch.linalg.solve``'s own gradient convention, so the complex
conjugation is verified against PyTorch rather than hand-argued. This is the single VJP node of the
matrix-free regime, the sparse bridge's stored-factor node generalized to
any matvec-defined operator.



DESIGN BOUNDARY: the
platform deliberately keeps TWO differentiable-solve families,

- THIS module: matrix-free / iterative regime. ``solve_adjoint`` wraps ANY
  LinearOperator + Krylov solve with the adjoint rule; nothing is factored,
  the backward re-solves with ``A^H``.
- :mod:`geobrain.core.adjoint`: factor-reuse direct regime for ASSEMBLED
  sparse/dense systems (``sparse_linear_solve_with_adjoint`` /
  ``linear_solve_with_adjoint``): factor once behind the
  :mod:`~geobrain.core.linalg.seam`, reuse the handle across the forward AND
  the backward (and across polarizations / frequencies via fingerprinting).

They are not duplicates: unifying them would force the direct bridge's
factor-handle lifecycle onto matrix-free callers (or vice versa) and churn
every EM/flow call site for zero measured gain. If a THIRD family ever
appears, revisit.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.autograd.function import once_differentiable

from ..errors import GeoBrainError

from ..linalg.linear_operator import LinearOperator, aslinearoperator
from ..linalg.krylov import cg


def _grad_A(lam: torch.Tensor, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """grad_A = -λ xᴴ, summed over RHS columns; sparse A -> grad only at its nonzeros."""
    lam2 = lam.unsqueeze(1) if lam.ndim == 1 else lam  # (n, k)
    x2 = x.unsqueeze(1) if x.ndim == 1 else x
    if A.layout is not torch.strided:  # sparse (COO or CSR/...)
        Acoo = A.coalesce() if A.is_sparse else A.to_sparse_coo().coalesce()
        idx = Acoo.indices()
        rows, cols = idx[0], idx[1]
        vals = -(lam2[rows] * x2[cols].conj()).sum(dim=1)
        return torch.sparse_coo_tensor(idx, vals, A.shape, dtype=A.dtype, device=A.device)
    return -(lam2 @ x2.conj().transpose(-2, -1))


def _check_stats(stats, *, which: str, check: bool) -> None:
    """Raise a structured error when a checked solve did not converge.

    ``solve_adjoint`` is the DIFFERENTIABLE entry point: a silently
    non-converged forward returns a wrong ``x``, and a silently non-converged
    adjoint solve returns a wrong ``λ``, i.e. wrong GRADIENTS with no signal.
    The krylov layer's "return stats, never raise" contract pushes this policy
    to the caller; here we ARE the caller, so enforce it (mirror of
    ``KrylovSolver.raise_on_nonconvergence``). ``check=False`` restores the
    old silent behaviour for callers that inspect residuals themselves.
    """
    if not check or stats.converged:
        return
    detail = (
        "Krylov breakdown (operator violates the solver's contract, e.g. a "
        "non-HPD matrix given to cg)"
        if stats.breakdown
        else f"residual {stats.residual_norm:.3e} after {stats.iterations} iterations"
    )
    raise GeoBrainError(
        f"solve_adjoint: the {which} solve did not converge, {detail}. "
        "Tighten maxiter/rtol, add a preconditioner (M= / M_adjoint=), pick a "
        "suitable method (gmres/bicgstab for indefinite systems), or pass "
        "check=False to accept a non-converged iterate.",
        object_name="solve_adjoint",
        field=which,
        expected="converged Krylov solve",
        actual=detail,
    )


class _SolveAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b, solver, hermitian, check, kwargs, adj_kwargs):
        with torch.no_grad():
            x, stats = solver(aslinearoperator(A), b, **kwargs)
        _check_stats(stats, which="forward", check=check)
        ctx.save_for_backward(A, x)
        ctx.solver = solver
        ctx.hermitian = hermitian
        ctx.check = check
        ctx.adj_kwargs = adj_kwargs
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_x):
        A, x = ctx.saved_tensors
        op = aslinearoperator(A)
        if ctx.hermitian:
            adj = op
        else:  # solve A^H λ = grad_x
            adj = LinearOperator(
                shape=(op.shape[1], op.shape[0]),
                matvec=op.rmatvec,
                rmatvec=op.matvec,
                dtype=op.dtype,
                device=op.device,
            )
        with torch.no_grad():
            lam, stats = ctx.solver(adj, grad_x, **ctx.adj_kwargs)
        _check_stats(stats, which="adjoint (backward)", check=ctx.check)

        grad_A = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_b = lam
        if ctx.needs_input_grad[0]:
            grad_A = _grad_A(lam, x, A)
        return grad_A, grad_b, None, None, None, None, None


def solve_adjoint(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    solver: Callable = cg,
    hermitian: bool = False,
    check: bool = True,
    M_adjoint: object | None = None,
    **solver_kwargs: object,
) -> torch.Tensor:
    """Differentiable linear solve ``A^{-1} b`` via implicit differentiation.

    Parameters
    ----------
    A : assembled matrix (dense or sparse COO), differentiable
    b : right-hand side, ``(n,)`` or ``(n, k)``, differentiable
    solver : a stack Krylov solver (:func:`cg`, :func:`bicgstab`, :func:`gmres`)
    hermitian : if True, the adjoint solve reuses ``A`` (A == A^H); else it solves ``A^H``
    check : raise a structured :class:`GeoBrainError` when the forward or the
        backward (adjoint) Krylov solve does not converge (default True, a
        silently non-converged solve yields a wrong ``x`` and wrong gradients;
        mirrors ``KrylovSolver.raise_on_nonconvergence``). Pass ``False`` to
        accept non-converged iterates and inspect residuals yourself.
    M_adjoint : optional preconditioner for the BACKWARD ``A^H λ = g`` solve.
        A forward preconditioner built from ``A`` (e.g. ``Jacobi(A)``, ``ILU0(A)``)
        is generally NOT a good preconditioner for ``A^H`` unless ``A`` is
        Hermitian: for the diag family use the conjugated diagonal, for ILU0
        factor ``A^H`` (this is exactly what ``KrylovSolver`` does internally).
        Default ``None``: the backward reuses the forward ``M`` (correct answer
        when converged either way; only the convergence RATE differs).
    solver_kwargs : forwarded to the solver (``rtol``, ``maxiter``, ``M``, ``restart`` …)

    Note:
        Gradients are **first-order only**, exactly as on the sparse-direct
        path: the backward runs its Krylov solve under ``no_grad`` and builds
        ``grad_A = -λ xᴴ`` from constants, so double-backward / Hessian-vector
        products are not supported, a second ``torch.autograd.grad`` raises
        (the backward is marked ``once_differentiable``). Use the dense
        :func:`~geobrain.core.adjoint.linear_solve_with_adjoint` when
        second-order gradients are needed.
    """
    # Normalize any non-COO sparse layout (e.g. the CSR that DC3D assembles) to COO so the
    # backward's grad_A is a well-formed sparse-COO gradient; the conversion is
    # differentiable, so grad still flows to whatever built A.
    if isinstance(A, torch.Tensor) and A.layout is not torch.strided and not A.is_sparse:
        A = A.to_sparse_coo()
    adj_kwargs = dict(solver_kwargs)
    if M_adjoint is not None:
        adj_kwargs["M"] = M_adjoint
    return _SolveAdjoint.apply(A, b, solver, hermitian, check, solver_kwargs,
                               adj_kwargs)
