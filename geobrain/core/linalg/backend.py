"""
Krylov backend for the sparse implicit-VJP bridge.

:class:`KrylovSolver` implements the :class:`~geobrain.core.adjoint.SparseFactorSolver`
contract using the :mod:`geobrain.core.linalg` stack, so it drops straight into the existing
seam::

    from geobrain.core.linalg import sparse_solver
    with sparse_solver(KrylovSolver(method="gmres")):
        ...  # every sparse EM/Helmholtz forward now solves on GPU via complex Krylov

Unlike the default CPU ``ScipySpluSolver``, this backend never factorizes; it keeps the
assembled operator and runs a matrix-free Krylov solve (with a Jacobi preconditioner by
default) on whatever device the matrix lives on. That is the GPU path the complex EM /
Helmholtz families lack today; the bridge's device guard already trusts any non-scipy
backend to manage its own device.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ..errors import GeoBrainError
from .seam import SparseFactorSolver

from .linear_operator import LinearOperator, aslinearoperator
from .krylov import cg, bicgstab, gmres
from .precond import Jacobi, ILU0

_METHODS = {"cg": cg, "bicgstab": bicgstab, "gmres": gmres}


def _hermitian_coo(A: torch.Tensor) -> torch.Tensor:
    """A^H as a coalesced sparse-COO tensor (swap indices, conjugate values)."""
    idx = A.indices()
    return torch.sparse_coo_tensor(
        torch.stack([idx[1], idx[0]]), A.values().conj(),
        (A.shape[1], A.shape[0]), dtype=A.dtype, device=A.device,
    ).coalesce()


class _KrylovHandle:
    __slots__ = ("op", "adj_op", "precond_fwd", "precond_adj")

    def __init__(self, op, adj_op, precond_fwd, precond_adj):
        self.op = op
        self.adj_op = adj_op
        self.precond_fwd = precond_fwd
        self.precond_adj = precond_adj


class KrylovSolver(SparseFactorSolver):
    """SparseFactorSolver backed by the complex-correct :mod:`geobrain.core.linalg` Krylov stack.

    Parameters
    ----------
    method : "cg" (Hermitian PD) | "bicgstab" | "gmres" (general/indefinite; default)
    rtol, atol, maxiter : Krylov tolerances / cap
    restart : GMRES restart length (ignored by cg/bicgstab)
    preconditioner : "jacobi" (default), "ilu0", or None
    raise_on_nonconvergence : raise a GeoBrainError if the solve does not converge
        (default True, a non-converged inner solve yields wrong gradients)
    """

    def __init__(
        self,
        *,
        method: str = "gmres",
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
        restart: Optional[int] = None,
        preconditioner: Optional[str] = "jacobi",
        raise_on_nonconvergence: bool = True,
    ) -> None:
        if method not in _METHODS:
            raise GeoBrainError(
                f"unknown Krylov method {method!r}; choose from {sorted(_METHODS)}",
                object_name="KrylovSolver", field="method",
            )
        if preconditioner not in (None, "jacobi", "ilu0"):
            raise GeoBrainError(
                f"unknown preconditioner {preconditioner!r}; use 'jacobi', 'ilu0', or None",
                object_name="KrylovSolver", field="preconditioner",
            )
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter
        self.restart = restart
        self.preconditioner = preconditioner
        self.raise_on_nonconvergence = raise_on_nonconvergence

    def factor(self, indices: torch.Tensor, values: torch.Tensor, shape) -> Any:
        if not bool(torch.isfinite(values.detach()).all()):
            raise GeoBrainError(
                "sparse matrix values are not finite (NaN/Inf); check the property "
                "model (e.g. σ) used to assemble the system",
                object_name="KrylovSolver", field="values",
            )
        A = torch.sparse_coo_tensor(indices, values.detach(), tuple(shape)).coalesce()
        op = aslinearoperator(A)
        adj_op = LinearOperator(
            shape=(A.shape[1], A.shape[0]),
            matvec=op.rmatvec,   # A^H action
            rmatvec=op.matvec,
            dtype=A.dtype,
            device=A.device,
        )
        precond_fwd = precond_adj = None
        if self.preconditioner == "jacobi":
            precond_fwd = Jacobi(A)
            inv_conj = precond_fwd.inv_diag.conj()  # diag(A^H) = conj(diag(A))
            def precond_adj(value, inv_conj=inv_conj):
                if value.ndim > 1:
                    return inv_conj.unsqueeze(-1) * value
                return inv_conj * value
        elif self.preconditioner == "ilu0":
            precond_fwd = ILU0(A)
            precond_adj = ILU0(_hermitian_coo(A))
        return _KrylovHandle(op, adj_op, precond_fwd, precond_adj)

    def solve(self, handle: Any, b: torch.Tensor, *, adjoint: bool) -> torch.Tensor:
        op = handle.adj_op if adjoint else handle.op
        M = handle.precond_adj if adjoint else handle.precond_fwd
        solver = _METHODS[self.method]
        kwargs = dict(rtol=self.rtol, atol=self.atol, maxiter=self.maxiter, M=M)
        if self.method == "gmres" and self.restart is not None:
            kwargs["restart"] = self.restart
        x, stats = solver(op, b, **kwargs)
        if self.raise_on_nonconvergence and not stats.converged:
            raise GeoBrainError(
                f"{self.method} did not converge in {stats.iterations} iterations "
                f"(residual {stats.residual_norm:.3e}); tighten maxiter, add a stronger "
                f"preconditioner, or use a direct backend",
                object_name="KrylovSolver", field="convergence",
                expected=f"residual <= rtol={self.rtol}",
                actual=f"{stats.residual_norm:.3e}",
            )
        return x
