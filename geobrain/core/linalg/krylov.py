"""
Complex-correct Krylov solvers for the unified stack.

Every inner product here is a CONJUGATE inner product (``sum(a.conj() * b)``), computed
per right-hand-side column so a block of RHS solves simultaneously. This is what makes
the solvers correct on complex-Hermitian systems (curl-curl, Helmholtz), a
real-oriented ``torch.dot`` recurrence silently diverges on complex input
(the historical flow implementation did exactly that before delegating here).

Solvers return ``(x, SolveStats)`` and do NOT raise on non-convergence, the caller
inspects ``stats.converged`` and decides (a forward operator may raise a structured
GeoBrainError; an inversion may accept a loose inner solve).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch

from ..errors import GeoBrainError
from .linear_operator import LinearOperator, aslinearoperator

Preconditioner = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class SolveStats:
    """Diagnostics from an iterative solve.

    ``residual_norm`` / ``history`` are the max over RHS columns per iteration.
    NORM CONVENTION: for :func:`cg` and :func:`bicgstab` these are TRUE residual
    norms ``||b - A x||``; for the left-preconditioned :func:`gmres` they are
    PRECONDITIONED residuals ``||M^{-1}(b - A x)||`` (the same norm its
    ``rtol * ||M^{-1} b||`` tolerance lives in, with a strong ``M`` the true
    residual can differ noticeably).

    ``breakdown``: the Krylov recurrence hit a numerical breakdown, for
    :func:`cg` a non-positive curvature ``p^H A p <= 0`` on an unconverged
    column (the operator is not Hermitian positive-definite, so CG theory is
    void and the solve early-breaks); for :func:`bicgstab` a (clamped)
    near-zero denominator (``rho``/``omega`` stagnation; the iterate stays
    finite and iteration continues, but treat the result with suspicion).
    ``breakdown`` implies nothing about ``converged`` for bicgstab; for cg a
    breakdown always returns ``converged=False``.
    """

    converged: bool = False
    iterations: int = 0
    residual_norm: float = float("inf")
    history: List[float] = field(default_factory=list)
    breakdown: bool = False


def _as_apply(M: Optional[Union[Preconditioner, object]]) -> Preconditioner:
    """Normalize a preconditioner (callable ``v -> M^{-1} v``, an object with ``.apply``,
    or None) into a callable. None is the identity."""
    if M is None:
        return lambda v: v
    if callable(M):
        return M
    apply = getattr(M, "apply", None)
    if callable(apply):
        return apply
    raise GeoBrainError(
        "preconditioner must be callable or expose .apply(v)",
        object_name="krylov", field="M",
        expected="callable | object with .apply | None", actual=type(M),
    )


def _cdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Column-wise conjugate inner product: returns shape (k,) for (n, k) inputs."""
    return (a.conj() * b).sum(dim=0)


def _safe(z: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Clamp a (possibly complex) scalar-per-column tensor away from zero magnitude so a
    Krylov breakdown yields a finite (non-converged) iterate instead of NaNs."""
    mag = z.abs()
    return torch.where(mag > eps, z, torch.full_like(z, eps))


def cg(
    A: Union[LinearOperator, torch.Tensor],
    b: torch.Tensor,
    *,
    M: Optional[Union[Preconditioner, object]] = None,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    x0: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, SolveStats]:
    """Preconditioned Conjugate Gradient for Hermitian positive-definite ``A``.

    Supports real and complex systems and a block of right-hand sides (``b`` shape ``(n,)``
    or ``(n, k)``). Convergence is per-column: it stops when every column's residual norm
    is ``<= rtol * ||b_col|| + atol``. Returns the (possibly non-converged) iterate and
    :class:`SolveStats`.
    """
    A = aslinearoperator(A)
    apply_M = _as_apply(M)

    one_d = b.ndim == 1
    B = b.unsqueeze(1) if one_d else b
    n, _ = B.shape
    if maxiter is None:
        maxiter = 10 * n

    X = torch.zeros_like(B) if x0 is None else (x0.unsqueeze(1) if one_d else x0).clone()

    R = B - A.matvec(X)
    Z = apply_M(R)
    P = Z.clone()
    rz_old = _cdot(R, Z)  # (k,)

    bnorm = torch.linalg.vector_norm(B, dim=0)
    bnorm = torch.where(bnorm > 0, bnorm, torch.ones_like(bnorm))
    tol = rtol * bnorm + atol

    history: List[float] = []
    rnorm = torch.linalg.vector_norm(R, dim=0)
    converged = bool(torch.all(rnorm <= tol))  # x0 (or a zero b) already good
    breakdown = False
    it = 0
    while not converged and it < maxiter:
        it += 1
        AP = A.matvec(P)
        pAp = _cdot(P, AP)  # (k,)
        # HPD guard: for a Hermitian positive-definite A and a nonzero search
        # direction, p^H A p is strictly positive (real). A non-positive (or
        # vanishing) curvature on a column that has NOT yet converged means the
        # operator violates CG's contract: the recurrence is void from here,
        # so early-break with a finite iterate and a breakdown flag instead of
        # NaN-cascading to maxiter. Converged columns legitimately have p ~ 0
        # and are masked out.
        active = rnorm > tol
        curvature = pAp.real if pAp.is_complex() else pAp
        if bool((((curvature <= 0) | (pAp.abs() <= 1e-30)) & active).any()):
            breakdown = True
            break
        alpha = rz_old / pAp
        X = X + alpha * P
        R = R - alpha * AP
        rnorm = torch.linalg.vector_norm(R, dim=0)
        history.append(float(rnorm.max()))
        if bool(torch.all(rnorm <= tol)):
            converged = True
            break
        Z = apply_M(R)
        rz_new = _cdot(R, Z)
        beta = rz_new / _safe(rz_old)
        P = Z + beta * P
        rz_old = rz_new

    x = X.squeeze(1) if one_d else X
    return x, SolveStats(
        converged=converged,
        iterations=it,
        residual_norm=float(rnorm.max()),
        history=history,
        breakdown=breakdown,
    )


def _gmres_single(A, b, apply_M, rtol, atol, restart, maxiter, check_every=None,
                  x0=None):
    """Left-preconditioned restarted GMRES(restart) for a single RHS. The per-cycle
    least-squares ``min ||beta e1 - H y||`` is solved by reduced QR (complex-correct,
    avoids hand-rolled complex Givens). Returns (x, converged, total_iters, residual).

    ``check_every=k`` additionally solves the small LS every ``k`` Arnoldi steps
    and exits the cycle as soon as the projected residual meets the tolerance,
    one extra tiny-QR dispatch per check. Worth it when the system converges
    well inside a cycle (the flow Newton Jacobians); ``None`` keeps the
    dispatch-minimal per-cycle-only check (the EM default)."""
    n = b.shape[0]
    dtype, device = b.dtype, b.device
    x = torch.zeros(n, dtype=dtype, device=device) if x0 is None else x0.clone()
    bnorm = float(torch.linalg.vector_norm(apply_M(b)))
    if bnorm == 0.0:
        bnorm = 1.0
    tol = rtol * bnorm + atol

    total = 0
    converged = False
    resid = float("inf")
    cycle_hist: List[float] = []
    while total < maxiter and not converged:
        r = apply_M(b - A.matvec(x))
        beta = torch.linalg.vector_norm(r)
        resid = float(beta)
        if resid <= tol:
            converged = True
            break
        m = min(restart, maxiter - total)
        V = torch.zeros(n, m + 1, dtype=dtype, device=device)
        H = torch.zeros(m + 1, m, dtype=dtype, device=device)
        V[:, 0] = r / beta
        # Minimal-dispatch cycle: run the Arnoldi/MGS steps, then solve the small
        # least-squares ONCE at the cycle end via one reduced QR. Pure-torch Krylov is
        # dispatch-bound (each tiny op ~tens of microseconds), so solving the LS once per
        # cycle beats both a QR-every-step (dispatch AND ~O(restart^4) flops) and a scalar
        # Givens loop (~O(restart^2) tiny ops). Convergence is checked per cycle; the true
        # iteration-count lever is preconditioning, not the LS bookkeeping.
        kk = 0
        for j in range(m):
            w = apply_M(A.matvec(V[:, j]))
            for i in range(j + 1):  # modified Gram-Schmidt, conjugate inner product
                H[i, j] = torch.vdot(V[:, i], w)
                w = w - H[i, j] * V[:, i]
            hj = torch.linalg.vector_norm(w)
            H[j + 1, j] = hj.to(dtype)
            kk = j + 1
            if float(hj) > 1e-14:
                V[:, j + 1] = w / hj
            else:
                break  # happy breakdown: the Krylov subspace is A-invariant
            if check_every is not None and kk % check_every == 0 and kk < m:
                # Opt-in inner check: one small reduced-QR on the current
                # (kk+1, kk) Hessenberg: exit the cycle early once the
                # projected residual meets the tolerance.
                Hj = H[: kk + 1, :kk]
                ej = torch.zeros(kk + 1, dtype=dtype, device=device)
                ej[0] = beta.to(dtype)
                Qj, Rj = torch.linalg.qr(Hj, mode="reduced")
                yj = torch.linalg.solve_triangular(
                    Rj, (Qj.conj().transpose(-2, -1) @ ej).unsqueeze(-1),
                    upper=True).squeeze(-1)
                if float(torch.linalg.vector_norm(ej - Hj @ yj)) <= tol:
                    break
        total += kk
        Hk = H[: kk + 1, :kk]
        e1 = torch.zeros(kk + 1, dtype=dtype, device=device)
        e1[0] = beta.to(dtype)
        Q, R = torch.linalg.qr(Hk, mode="reduced")
        y = torch.linalg.solve_triangular(
            R, (Q.conj().transpose(-2, -1) @ e1).unsqueeze(-1), upper=True).squeeze(-1)
        x = x + V[:, :kk] @ y
        resid = float(torch.linalg.vector_norm(e1 - Hk @ y))
        cycle_hist.append(resid)
        if resid <= tol:
            converged = True
    return x, converged, total, resid, cycle_hist


def gmres(
    A: Union[LinearOperator, torch.Tensor],
    b: torch.Tensor,
    *,
    M: Optional[Union[Preconditioner, object]] = None,
    restart: Optional[int] = None,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    x0: Optional[torch.Tensor] = None,
    check_every: Optional[int] = None,
) -> tuple[torch.Tensor, SolveStats]:
    """Left-preconditioned restarted GMRES(``restart``) for general complex/real ``A``.

    ``x0`` warm-starts every column (an exact ``x0`` converges in zero
    iterations). ``check_every=k`` (opt-in) tests convergence every ``k`` Arnoldi steps inside a
    cycle (one small extra QR per check) instead of only at cycle boundaries;
    pass ``1`` to reproduce classical per-iteration stopping when the system is
    expected to converge well inside a restart cycle.

    The robust choice for strongly-indefinite complex-symmetric systems (curl-curl,
    Helmholtz). Block-RHS (``b`` shape ``(n,)`` or ``(n, k)``) is handled by solving each
    column independently (a block-GMRES optimization is future work). Returns the iterate
    and :class:`SolveStats` (``converged`` = all columns; ``iterations`` = max over columns).

    NORM CONVENTION: with LEFT preconditioning, convergence is measured in the
    PRECONDITIONED residual norm: the stopping rule is
    ``||M^{-1}(b - A x)|| <= rtol * ||M^{-1} b|| + atol``, and
    ``stats.residual_norm`` / ``stats.history`` report that same preconditioned
    norm. With a strong ``M`` (e.g. ILU0) the TRUE residual ``||b - A x||`` can
    differ from it by roughly ``cond(M)``; recompute it explicitly if you need a
    guarantee in the unpreconditioned norm. ``stats.history`` holds per-cycle
    residuals for a single RHS; it is empty for block RHS (per-column histories
    are ragged, inspect per-column solves individually if needed).
    """
    A = aslinearoperator(A)
    apply_M = _as_apply(M)

    one_d = b.ndim == 1
    B = b.unsqueeze(1) if one_d else b
    n, k = B.shape
    if maxiter is None:
        maxiter = 10 * n
    if restart is None:
        restart = min(n, 30)  # modest restart: fewer MGS ops/cycle when dispatch-bound

    cols: List[torch.Tensor] = []
    all_conv = True
    max_iters = 0
    max_resid = 0.0
    single_hist: List[float] = []
    X0 = None if x0 is None else (x0.unsqueeze(1) if one_d else x0)
    for c in range(k):
        xc, conv, iters, resid, cycle_hist = _gmres_single(
            A, B[:, c], apply_M, rtol, atol, restart, maxiter,
            check_every=check_every,
            x0=None if X0 is None else X0[:, c],
        )
        cols.append(xc)
        all_conv = all_conv and conv
        max_iters = max(max_iters, iters)
        max_resid = max(max_resid, resid)
        if k == 1:
            single_hist = cycle_hist

    X = torch.stack(cols, dim=1)
    x = X.squeeze(1) if one_d else X
    return x, SolveStats(
        converged=all_conv, iterations=max_iters, residual_norm=max_resid,
        history=single_hist,
    )


def bicgstab(
    A: Union[LinearOperator, torch.Tensor],
    b: torch.Tensor,
    *,
    M: Optional[Union[Preconditioner, object]] = None,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    x0: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, SolveStats]:
    """Preconditioned BiCGStab for general (non-Hermitian / indefinite) ``A``.

    Handles the complex-symmetric indefinite systems (curl-curl, Helmholtz) that CG
    cannot. Complex-correct (conjugate inner products), block-RHS (``b`` shape ``(n,)`` or
    ``(n, k)``), right-preconditioned. Breakdowns are clamped so a stalled solve returns a
    finite non-converged iterate rather than NaNs.
    """
    A = aslinearoperator(A)
    apply_M = _as_apply(M)

    one_d = b.ndim == 1
    B = b.unsqueeze(1) if one_d else b
    n, k = B.shape
    if maxiter is None:
        maxiter = 10 * n

    X = torch.zeros_like(B) if x0 is None else (x0.unsqueeze(1) if one_d else x0).clone()

    r = B - A.matvec(X)
    r0 = r.clone()  # fixed shadow residual
    v = torch.zeros_like(B)
    p = torch.zeros_like(B)
    ones = torch.ones(k, dtype=B.dtype, device=B.device)
    rho_old = ones.clone()
    alpha = ones.clone()
    omega = ones.clone()

    bnorm = torch.linalg.vector_norm(B, dim=0)
    bnorm = torch.where(bnorm > 0, bnorm, torch.ones_like(bnorm))
    tol = rtol * bnorm + atol

    history: List[float] = []
    rnorm = torch.linalg.vector_norm(r, dim=0)
    converged = bool(torch.all(rnorm <= tol))  # x0 (or a zero b) already good
    breakdown = False
    it = 0
    for it in range(1, (0 if converged else maxiter) + 1):
        rho_new = _cdot(r0, r)
        # Diagnostic: a clamp that actually FIRES below means rho/omega/r0·v
        # hit the breakdown threshold: the iterate stays finite and iteration
        # continues (the documented clamp contract), but flag it so callers
        # can treat the result with suspicion.
        if bool((rho_old.abs() <= 1e-30).any()) or bool((omega.abs() <= 1e-30).any()):
            breakdown = True
        beta = (rho_new / _safe(rho_old)) * (alpha / _safe(omega))
        p = r + beta * (p - omega * v)
        phat = apply_M(p)
        v = A.matvec(phat)
        r0v = _cdot(r0, v)
        if bool((r0v.abs() <= 1e-30).any()):
            breakdown = True
        alpha = rho_new / _safe(r0v)
        s = r - alpha * v
        snorm = torch.linalg.vector_norm(s, dim=0)
        X_half = X + alpha * phat
        if bool(torch.all(snorm <= tol)):
            X = X_half
            rnorm = snorm
            history.append(float(snorm.max()))
            converged = True
            break
        shat = apply_M(s)
        t = A.matvec(shat)
        omega = _cdot(t, s) / _safe(_cdot(t, t))
        X = X_half + omega * shat
        r = s - omega * t
        rnorm = torch.linalg.vector_norm(r, dim=0)
        history.append(float(rnorm.max()))
        if bool(torch.all(rnorm <= tol)):
            converged = True
            break
        rho_old = rho_new

    x = X.squeeze(1) if one_d else X
    return x, SolveStats(
        converged=converged,
        iterations=it,
        residual_norm=float(rnorm.max()),
        history=history,
        breakdown=breakdown,
    )
