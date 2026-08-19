"""
Preconditioners for the Krylov stack.

A preconditioner is any callable ``v -> M^{-1} v`` (accepting ``(n,)`` or ``(n, k)``), so
the solvers consume them uniformly. Preconditioning is the real lever on iteration count;
Jacobi is cheap and GPU-friendly but weak; :class:`ILU0` (scipy incomplete-LU) is strong
and typically cuts iterations by an order of magnitude on the ill-conditioned complex EM /
Helmholtz systems, at the cost of a CPU factorization + per-apply host round-trip (so it is
best on the CPU path; a GPU-native strong preconditioner, multigrid / auxiliary-space
Maxwell: is the roadmap's separate stretch item).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from ..errors import GeoBrainError


def _diagonal(A: torch.Tensor) -> torch.Tensor:
    if A.layout is torch.strided:
        return torch.diagonal(A)
    Acoo = A.coalesce() if A.is_sparse else A.to_sparse_coo().coalesce()
    idx, vals = Acoo.indices(), Acoo.values()
    d = torch.zeros(A.shape[0], dtype=vals.dtype, device=vals.device)
    on_diag = idx[0] == idx[1]
    d.index_add_(0, idx[0][on_diag], vals[on_diag])
    return d


class Jacobi:
    """Diagonal (Jacobi) preconditioner ``M^{-1} = diag(A)^{-1}``. Cheap, GPU-friendly.

    Zero (or ``|d| <= eps``) diagonal entries are mapped to ``1``, the
    preconditioner acts as identity on those rows rather than dividing by
    zero. Fine as a preconditioner (any SPD ``M`` is valid); it does mean a
    structurally-zero-diagonal system gets no help on those rows.
    """

    def __init__(self, A: torch.Tensor, eps: float = 1e-30) -> None:
        d = _diagonal(A)
        safe = torch.where(d.abs() > eps, d, torch.ones_like(d))
        self.inv_diag = 1.0 / safe

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return (self.inv_diag.unsqueeze(-1) * v) if v.ndim > 1 else self.inv_diag * v


class ILU0:
    """Incomplete-LU preconditioner via ``scipy.sparse.linalg.spilu``.

    Strong general-purpose preconditioner (real or complex). The factorization and each
    apply run on the CPU (host round-trip for GPU inputs), so this is the CPU-path lever;
    it typically reduces Krylov iterations by 10x+ on ill-conditioned EM systems.
    """

    def __init__(self, A: torch.Tensor, **spilu_kwargs) -> None:
        Acoo = (A.coalesce() if A.is_sparse else A.to_sparse_coo().coalesce()) \
            if A.layout is not torch.strided else A.to_sparse_coo().coalesce()
        idx = Acoo.indices().cpu().numpy()
        vals = Acoo.values().detach().cpu().numpy()
        n = int(A.shape[0])
        csc = sp.csc_matrix((vals, (idx[0], idx[1])), shape=(n, n))
        try:
            self._ilu = spla.spilu(csc, **spilu_kwargs)
        except RuntimeError as exc:
            raise GeoBrainError(
                "ILU0 factorization failed (singular or badly scaled matrix)",
                object_name="ILU0", field="A",
                expected="spilu-factorizable sparse matrix", actual=str(exc),
                hint="try Jacobi, a shifted system, or scipy spilu options "
                     "(drop_tol/fill_factor)",
            ) from exc

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        v_np = v.detach().cpu().numpy()
        x_np = self._ilu.solve(v_np)  # scipy handles (n,) and (n, k)
        return torch.from_numpy(np.ascontiguousarray(x_np)).to(dtype=v.dtype, device=v.device)
