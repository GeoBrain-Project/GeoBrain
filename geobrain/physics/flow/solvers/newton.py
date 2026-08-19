"""
Damped Newton-Raphson nonlinear solver.

Generic over the model: solver only needs ``residual_fn(x) → r`` and
``jacobian_fn(x) → J`` callables (J may be dense or sparse; see the
``linear_solver`` plug-in below). Convergence is judged by the
reservoir-engineering absolute-max norm ``|r|_∞ < tol`` with a
relative-decrease safeguard.

Line search uses back-tracking with step bisection. Every exhausted or invalid
solve raises :class:`FlowConvergenceError`; a finite non-converged state is
never returned as if it were usable output.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NoReturn, Protocol

import torch

from .._defaults import MAX_NEWTON_ITER, NEWTON_TOL, NEWTON_TOL_REL
from ..errors import FlowContractError, FlowConvergenceError
from .config import NewtonConfig
from .diagnostics import FlowConvergenceDiagnostics, convergence_diagnostics

if TYPE_CHECKING:
    from .linear_solvers import SparseDirectSolver

logger = logging.getLogger(__name__)

_ConvergenceReason = Literal[
    "tolerance",
    "max_iterations",
    "line_search",
    "breakdown",
    "nonfinite",
    "invalid_state",
]


class _LinearSolver(Protocol):
    def solve(self, J: torch.Tensor, r: torch.Tensor) -> torch.Tensor: ...


class _DirectLinearSolver:
    """
    Default direct linear solver, storage-aware.

    Dense Jacobians use ``torch.linalg.solve``. Sparse Jacobians are routed
    to scipy sparse LU (:class:`~geobrain.physics.flow.solvers.SparseDirectSolver`,
    ``spsolve``) WITHOUT densifying, so a tridiagonal/banded reservoir
    Jacobian is solved in ~O(n) instead of being expanded to an O(n²) dense
    matrix and solved in O(n³). The Newton forward solve runs under
    ``torch.no_grad`` (the implicit-FT backward handles gradients separately),
    so the sparse path's lack of autograd is by design. For very large
    problems pass an explicit iterative solver (BiCGSTAB / GMRES) from
    ``linear_solvers`` instead.
    """

    def __init__(self) -> None:
        # Lazily created on first sparse solve (avoids an import cycle and the
        # cost when only dense Jacobians are ever seen).
        self._sparse: SparseDirectSolver | None = None

    def solve(self, J: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if J.is_sparse:
            if self._sparse is None:
                from .linear_solvers import SparseDirectSolver

                self._sparse = SparseDirectSolver()
            # SparseDirectSolver.solve returns dx solving ``J dx = -r``.
            return self._sparse.solve(J, r)
        return -torch.linalg.solve(J, r)


@dataclass
class NewtonResult:
    """Converged outcome of a single Newton solve."""

    converged: bool = False
    iterations: int = 0
    state: torch.Tensor | None = None
    residual_norm: float = float("inf")
    history: list[float] = field(default_factory=list)
    jacobian: torch.Tensor | None = None
    # All step halvings failed to decrease |r|; caller should cut dt.
    no_progress: bool = False
    diagnostics: FlowConvergenceDiagnostics | None = None


class NewtonSolver:
    """
    Damped Newton with back-tracking line search.

    Args:
        linear_solver: object exposing ``.solve(J, r) → dx`` such that
            ``J @ dx ≈ -r``. Default :class:`_DirectLinearSolver`.
        tol: max-residual absolute tolerance.
        tol_rel: relative-residual tolerance ``|r| / |r_0| < tol_rel``.
        max_iter: maximum Newton iterations.
        line_search: enable back-tracking line search (default True).
        line_search_max_halvings: how many step halvings before bailing
            out with ``no_progress=True``.
        keep_jacobian: store the final Jacobian on the result (needed
            for adjoint-state assembly).
        verbose: log per-iteration residual norm.
    """

    def __init__(
        self,
        linear_solver: _LinearSolver | None = None,
        tol: float = NEWTON_TOL,
        tol_rel: float = NEWTON_TOL_REL,
        max_iter: int = MAX_NEWTON_ITER,
        line_search: bool = True,
        line_search_max_halvings: int = 10,
        keep_jacobian: bool = False,
        verbose: bool = False,
    ) -> None:
        validated = NewtonConfig(
            max_iterations=max_iter,
            residual_tolerance=tol,
            update_tolerance=tol_rel,
            line_search_max_iterations=line_search_max_halvings,
        )
        self.linear_solver = linear_solver or _DirectLinearSolver()
        self.tol: float = float(validated.residual_tolerance)
        self.tol_rel: float = float(validated.update_tolerance)
        self.max_iter = validated.max_iterations
        self.line_search = bool(line_search)
        self.line_search_max_halvings = validated.line_search_max_iterations
        self.keep_jacobian = bool(keep_jacobian)
        self.verbose = bool(verbose)

    def solve(
        self,
        residual_fn: Callable[[torch.Tensor], torch.Tensor],
        jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
        state0: torch.Tensor,
        converged_fn: Callable[[torch.Tensor, torch.Tensor], bool] | None = None,
    ) -> NewtonResult:
        """Solve ``R(x)=0`` from ``state0``.

        ``converged_fn(r, x) → bool`` overrides the default ``|r|_∞`` test with a
        custom convergence criterion (e.g. a :class:`CNVMBCriterion` closure);
        the back-tracking line search still measures progress by ``|r|_∞``.
        """
        if not isinstance(state0, torch.Tensor) or not state0.is_floating_point():
            raise FlowContractError(
                "Newton state must be a floating tensor",
                object_name="NewtonSolver",
                field="state0",
                expected="floating torch.Tensor",
                actual=type(state0).__name__,
            )

        def norm(value: torch.Tensor) -> float:
            return float(torch.linalg.vector_norm(value.detach(), ord=float("inf")))

        def diagnostics(
            *,
            converged: bool,
            reason: _ConvergenceReason,
            iterations: int,
            initial: float,
            current: float,
            history_values: list[float],
        ) -> FlowConvergenceDiagnostics:
            criterion = getattr(converged_fn, "flow_criterion", None)
            errors = None if criterion is None else criterion.last_errors
            return convergence_diagnostics(
                stage="nonlinear",
                converged=converged,
                reason=reason,
                iterations=iterations,
                max_iterations=self.max_iter,
                initial_residual_norm=initial,
                residual_norm=current,
                residual_history=history_values,
                cnv=() if errors is None else tuple(errors["cnv"]),
                mb=() if errors is None else tuple(errors["mb"]),
            )

        def fail(
            *,
            reason: _ConvergenceReason,
            iterations: int,
            initial: float,
            current: float,
            history_values: list[float],
            message: str,
        ) -> NoReturn:
            record = diagnostics(
                converged=False,
                reason=reason,
                iterations=iterations,
                initial=initial,
                current=current,
                history_values=history_values,
            )
            raise FlowConvergenceError(
                message,
                object_name="NewtonSolver",
                field="convergence",
                expected="converged nonlinear state",
                actual=record.to_dict(),
                diagnostics=record,
            )

        x = state0.clone()
        if not bool(torch.isfinite(x).all()):
            fail(
                reason="nonfinite",
                iterations=0,
                initial=float("inf"),
                current=float("inf"),
                history_values=(),  # type: ignore[arg-type]
                message="Newton initial state contains NaN or infinity",
            )
        r = residual_fn(x)
        if not isinstance(r, torch.Tensor) or not bool(torch.isfinite(r).all()):
            fail(
                reason="nonfinite",
                iterations=0,
                initial=float("inf"),
                current=float("inf"),
                history_values=[],
                message="Newton residual contains NaN or infinity",
            )
        r_norm0 = norm(r)
        history: list[float] = [r_norm0]

        def _converged(res: torch.Tensor, state: torch.Tensor, r_norm: float) -> bool:
            if converged_fn is not None:
                return bool(converged_fn(res, state))
            return r_norm < self.tol or r_norm / max(r_norm0, 1e-30) < self.tol_rel

        if _converged(r, x, r_norm0):
            record = diagnostics(
                converged=True,
                reason="tolerance",
                iterations=0,
                initial=r_norm0,
                current=r_norm0,
                history_values=history,
            )
            return NewtonResult(
                converged=True,
                iterations=0,
                state=x,
                residual_norm=r_norm0,
                history=history,
                diagnostics=record,
            )

        for it in range(1, self.max_iter + 1):
            J = jacobian_fn(x)
            values = J.values() if J.layout in {torch.sparse_coo, torch.sparse_csr} else J
            if not isinstance(J, torch.Tensor) or not bool(torch.isfinite(values).all()):
                fail(
                    reason="nonfinite",
                    iterations=it,
                    initial=r_norm0,
                    current=history[-1],
                    history_values=history,
                    message="Newton Jacobian contains NaN or infinity",
                )
            try:
                dx = self.linear_solver.solve(J, r)
            except FlowConvergenceError:
                raise
            except RuntimeError as error:
                record = convergence_diagnostics(
                    stage="linear",
                    converged=False,
                    reason="breakdown",
                    iterations=0,
                    max_iterations=1,
                    initial_residual_norm=history[-1],
                    residual_norm=history[-1],
                    residual_history=(history[-1],),
                )
                raise FlowConvergenceError(
                    "Newton linear solve broke down",
                    object_name="NewtonSolver",
                    field="linear_solve",
                    expected="finite converged update",
                    actual=type(error).__name__,
                    diagnostics=record,
                ) from error
            stats = getattr(self.linear_solver, "last_stats", None)
            if stats is not None and not bool(stats.converged):
                record = convergence_diagnostics(
                    stage="linear",
                    converged=False,
                    reason="max_iterations",
                    iterations=int(stats.iterations),
                    max_iterations=int(getattr(self.linear_solver, "max_iter", stats.iterations)),
                    initial_residual_norm=history[-1],
                    residual_norm=float(stats.residual_norm),
                    residual_history=tuple(stats.history),
                )
                raise FlowConvergenceError(
                    "Newton linear solve exhausted its iteration budget",
                    object_name="NewtonSolver",
                    field="linear_solve",
                    expected="converged linear update",
                    actual=record.to_dict(),
                    diagnostics=record,
                )
            if not bool(torch.isfinite(dx).all()):
                fail(
                    reason="nonfinite",
                    iterations=it,
                    initial=r_norm0,
                    current=history[-1],
                    history_values=history,
                    message="Newton update contains NaN or infinity",
                )

            if self.line_search:
                alpha = 1.0
                r_norm_old = float(torch.linalg.vector_norm(r.detach(), ord=float("inf")))
                ls_progress = False
                x_try = x
                r_try = r
                r_norm_try = r_norm_old
                for _ in range(self.line_search_max_halvings):
                    candidate = x + alpha * dx
                    try:
                        candidate_residual = residual_fn(candidate)
                    except FlowContractError:
                        # Constitutive kernels reject non-physical states. Such
                        # a trial is expected during damping; keep the last
                        # valid state and reduce the step instead of bypassing
                        # the kernel's domain contract.
                        alpha *= 0.5
                        continue
                    x_try = candidate
                    r_try = candidate_residual
                    if not bool(torch.isfinite(r_try).all()):
                        alpha *= 0.5
                        continue
                    r_norm_try = norm(r_try)
                    if r_norm_try <= r_norm_old:
                        ls_progress = True
                        break
                    alpha *= 0.5
                x = x_try
                r = r_try
                r_norm = r_norm_try
                if not ls_progress:
                    history.append(r_norm)
                    fail(
                        reason="line_search",
                        iterations=it,
                        initial=r_norm0,
                        current=r_norm,
                        history_values=history,
                        message="Newton line search could not find a finite decreasing step",
                    )
            else:
                x = x + dx
                if not bool(torch.isfinite(x).all()):
                    fail(
                        reason="nonfinite",
                        iterations=it,
                        initial=r_norm0,
                        current=history[-1],
                        history_values=history,
                        message="Newton state became NaN or infinity",
                    )
                r = residual_fn(x)
                if not bool(torch.isfinite(r).all()):
                    fail(
                        reason="nonfinite",
                        iterations=it,
                        initial=r_norm0,
                        current=float("inf"),
                        history_values=history,
                        message="Newton residual became NaN or infinity",
                    )
                r_norm = norm(r)

            history.append(r_norm)
            if self.verbose:
                logger.info("  Newton iter %3d: |r|_inf = %.3e", it, r_norm)

            if _converged(r, x, r_norm):
                J_final = jacobian_fn(x) if self.keep_jacobian else None
                record = diagnostics(
                    converged=True,
                    reason="tolerance",
                    iterations=it,
                    initial=r_norm0,
                    current=r_norm,
                    history_values=history,
                )
                return NewtonResult(
                    converged=True,
                    iterations=it,
                    state=x,
                    residual_norm=r_norm,
                    history=history,
                    jacobian=J_final,
                    diagnostics=record,
                )

        fail(
            reason="max_iterations",
            iterations=self.max_iter,
            initial=r_norm0,
            current=history[-1],
            history_values=history,
            message="Newton solve exhausted its iteration budget",
        )


def solve_newton(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    state0: torch.Tensor,
    *,
    config: NewtonConfig,
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> NewtonResult:
    """Solve a nonlinear system under the canonical fail-loud API."""
    jacobian = (
        jacobian_fn
        if jacobian_fn is not None
        else lambda state: torch.autograd.functional.jacobian(residual_fn, state, vectorize=True)
    )
    return NewtonSolver(
        tol=config.residual_tolerance,
        tol_rel=config.update_tolerance,
        max_iter=config.max_iterations,
        line_search_max_halvings=config.line_search_max_iterations,
    ).solve(residual_fn, jacobian, state0)


__all__ = ["NewtonResult", "NewtonSolver", "solve_newton"]
