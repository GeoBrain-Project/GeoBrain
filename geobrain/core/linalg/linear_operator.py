"""
Matrix-free linear operators for the unified solver stack.

A :class:`LinearOperator` exposes a system ``A`` purely through its action
(``matvec`` = ``A @ x``) and its conjugate-transpose action (``rmatvec`` = ``A^H @ x``).
The Krylov solvers and the implicit-differentiation adjoint consume only these two
callables, so the same solver works whether ``A`` is an assembled sparse tensor, a dense
tensor, or a never-materialized closure (e.g. a matrix-free curl-curl or FFT operator).

``rmatvec`` is the CONJUGATE transpose, not the plain transpose, the adjoint linear
solve for a complex system solves ``A^H λ = g``, and getting the conjugation wrong is a
silent, hard-to-spot error on complex EM/Helmholtz problems.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch

from ..errors import GeoBrainError

MatVec = Callable[[torch.Tensor], torch.Tensor]


def _spmv(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Sparse (coalesced COO) times dense, supporting v shape (n,) or (n, k)."""
    if v.ndim == 1:
        return torch.sparse.mm(M, v.unsqueeze(1)).squeeze(1)
    return torch.sparse.mm(M, v)


def _hermitian_coo(A: torch.Tensor) -> torch.Tensor:
    """Build A^H (conjugate transpose) of a sparse tensor as a coalesced COO tensor."""
    A = A.coalesce()
    idx = A.indices()
    vals = A.values()
    idx_h = torch.stack([idx[1], idx[0]])
    return torch.sparse_coo_tensor(
        idx_h, vals.conj(), (A.shape[1], A.shape[0]), device=A.device, dtype=A.dtype
    ).coalesce()


class LinearOperator:
    """A linear map defined by its ``matvec`` (and optionally ``rmatvec``) action.

    Parameters
    ----------
    shape : (m, n)
    matvec : callable ``x -> A @ x`` (must accept ``(n,)`` and ``(n, k)``)
    rmatvec : callable ``x -> A^H @ x``; omit only if ``hermitian`` is True
    hermitian : if True, ``A == A^H`` so ``rmatvec`` falls back to ``matvec``
    dtype : keyword-only; the operator's scalar dtype (drives solver
        workspace allocation)
    device : keyword-only; the device the matvec runs on
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        matvec: MatVec,
        rmatvec: Optional[MatVec] = None,
        *,
        dtype: torch.dtype,
        device: torch.device,
        hermitian: bool = False,
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.dtype = dtype
        self.device = device
        self.hermitian = hermitian
        self._matvec = matvec
        self._rmatvec = rmatvec

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        return self._matvec(x)

    def rmatvec(self, x: torch.Tensor) -> torch.Tensor:
        if self.hermitian:
            return self._matvec(x)
        if self._rmatvec is None:
            raise GeoBrainError(
                "LinearOperator has no rmatvec (A^H action); supply one or set hermitian=True",
                object_name="LinearOperator",
            )
        return self._rmatvec(x)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self._matvec(x)


def aslinearoperator(
    A: Union[LinearOperator, torch.Tensor],
) -> LinearOperator:
    """Coerce a dense or sparse tensor (or an existing LinearOperator) into a
    LinearOperator. Callables must be wrapped in :class:`LinearOperator` directly (they
    carry no shape/dtype)."""
    if isinstance(A, LinearOperator):
        return A
    if isinstance(A, torch.Tensor):
        if A.layout is not torch.strided:  # any sparse layout (COO/CSR/...)
            Acoo = A.coalesce() if A.is_sparse else A.to_sparse_coo().coalesce()

            # Build A^H LAZILY on the first rmatvec and cache it: a forward-only
            # consumer (one Krylov solve, no adjoint) never pays the transpose's
            # construction/coalesce cost or its extra copy of the matrix.
            _cache: dict = {}

            def _rmv(v: torch.Tensor) -> torch.Tensor:
                Ah = _cache.get("Ah")
                if Ah is None:
                    Ah = _hermitian_coo(Acoo)
                    _cache["Ah"] = Ah
                return _spmv(Ah, v)

            return LinearOperator(
                shape=A.shape,
                matvec=lambda v: _spmv(Acoo, v),
                rmatvec=_rmv,
                dtype=A.dtype,
                device=A.device,
            )
        return LinearOperator(
            shape=A.shape,
            matvec=lambda v: A @ v,
            rmatvec=lambda v: A.conj().transpose(-2, -1) @ v,
            dtype=A.dtype,
            device=A.device,
        )
    raise GeoBrainError(
        f"cannot coerce {type(A)!r} to LinearOperator; wrap callables in LinearOperator(...)",
        object_name="aslinearoperator",
    )
