"""
Geometric multigrid (V-cycle) for Cartesian-grid TPFA Poisson systems.

Pure torch: no scipy.sparse. Hierarchy:

- **Restriction** ``R``: 2:1 full-weighting on every axis that still has
  > 4 cells. Built as a ``torch.sparse_coo_tensor``.
- **Prolongation** ``P = R.T``: transpose-injection.
- **Galerkin coarse operator** ``A_c = R·A·P``: implemented by mapping
  the fine COO entries to coarse cells and letting ``coalesce`` sum
  duplicates; avoids the need for a general sparse-sparse mat-mul.
- **Smoother**: weighted Jacobi (textbook ω = 2/3).
- **Coarse solver**: dense ``torch.linalg.solve`` once the level falls
  below ``n_coarsest``.
- **Outer**: Richardson iteration of V-cycles until
  ``|r|/|r₀| < tol`` or ``max_iter`` is hit.

Limitations: Cartesian only, TPFA-shaped sparsity (≤ 7 nonzeros / row
in 3D), single-equation (no block-MG). For unstructured grids or
saturated multi-phase systems, use BiCGSTAB / GMRES with a Jacobi
preconditioner.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ....core import GeoBrainError
from .._defaults import EPS
from ..errors import FlowConvergenceError
from .config import KrylovConfig
from .diagnostics import FlowConvergenceDiagnostics, convergence_diagnostics


def _coarsen_dim(n: int) -> int:
    """2:1 coarsen unless already <= 4."""
    return max(1, n // 2) if n > 4 else n


def _effective_relative_tolerance(dtype: torch.dtype, requested: float) -> float:
    """Return a relative tolerance resolvable by the requested real dtype."""
    machine_epsilon = float(torch.finfo(dtype).eps)
    return max(float(requested), 32.0 * machine_epsilon)


def _build_restriction(
    nx: int,
    ny: int,
    nz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """
    Full-weighting restriction ``R`` (n_coarse × n_fine).

    Each row of ``R`` averages the fine cells covered by one coarse cell;
    ``P = R.T`` injects coarse values back. Returns ``(R_sparse_coo, (nx_c, ny_c, nz_c))``.
    """
    nx_c, ny_c, nz_c = _coarsen_dim(nx), _coarsen_dim(ny), _coarsen_dim(nz)
    n_fine = nx * ny * nz
    n_coarse = nx_c * ny_c * nz_c

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for kc in range(nz_c):
        kfs = [2 * kc, 2 * kc + 1] if nz_c < nz else [kc]
        kfs = [k for k in kfs if k < nz]
        for jc in range(ny_c):
            jfs = [2 * jc, 2 * jc + 1] if ny_c < ny else [jc]
            jfs = [j for j in jfs if j < ny]
            for ic in range(nx_c):
                ifs = [2 * ic, 2 * ic + 1] if nx_c < nx else [ic]
                ifs = [i for i in ifs if i < nx]
                n_kids = len(ifs) * len(jfs) * len(kfs)
                if n_kids == 0:
                    continue
                w = 1.0 / n_kids
                rc = ic + jc * nx_c + kc * nx_c * ny_c
                for kf in kfs:
                    for jf in jfs:
                        for if_ in ifs:
                            rf = if_ + jf * nx + kf * nx * ny
                            rows.append(rc)
                            cols.append(rf)
                            vals.append(w)

    idx = torch.tensor([rows, cols], dtype=torch.int64, device=device)
    v = torch.tensor(vals, dtype=dtype, device=device)
    R = torch.sparse_coo_tensor(idx, v, size=(n_coarse, n_fine)).coalesce()
    return R, (nx_c, ny_c, nz_c)


def _galerkin(A: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """``A_c = R · A · R.T`` via COO coalesce: pure-torch alternative
    to sparse-sparse mat-mul.

    For full-weighting restriction with weight ``w = 1/k`` (k children
    per coarse cell), an entry ``A[i, j] = a`` contributes to
    ``A_c[c1(i), c2(j)]`` with weight ``w(c1) · w(c2) · a``, equivalent
    to scattering each fine COO entry into the coarse pair and letting
    coalesce add duplicates.
    """
    A = A.coalesce()
    R = R.coalesce()
    n_coarse = R.shape[0]

    # Maps fine cell → coarse cell + weight.
    r_idx = R.indices()  # (2, nnz_R)
    r_val = R.values()
    n_fine = A.shape[0]
    cell_to_coarse = torch.full((n_fine,), -1, dtype=torch.int64, device=A.device)
    cell_weight = torch.zeros(n_fine, dtype=A.dtype, device=A.device)
    cell_to_coarse[r_idx[1]] = r_idx[0]
    cell_weight[r_idx[1]] = r_val

    a_idx = A.indices()  # (2, nnz_A)
    a_val = A.values()
    c_rows = cell_to_coarse[a_idx[0]]
    c_cols = cell_to_coarse[a_idx[1]]
    keep = (c_rows >= 0) & (c_cols >= 0)
    c_rows = c_rows[keep]
    c_cols = c_cols[keep]
    weights = cell_weight[a_idx[0][keep]] * cell_weight[a_idx[1][keep]]
    coarse_vals = a_val[keep] * weights

    coarse_idx = torch.stack([c_rows, c_cols], dim=0)
    return torch.sparse_coo_tensor(
        coarse_idx,
        coarse_vals,
        size=(n_coarse, n_coarse),
    ).coalesce()


def _diag(A: torch.Tensor) -> torch.Tensor:
    A = A.coalesce()
    idx = A.indices()
    is_diag = idx[0] == idx[1]
    n = A.shape[0]
    d = torch.zeros(n, device=A.device, dtype=A.values().dtype)
    d.index_add_(0, idx[0][is_diag], A.values()[is_diag])
    return d


def _sparse_conjugate_gradient(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    relative_tolerance: float,
    max_iterations: int,
) -> torch.Tensor:
    """Solve one small SPD coarse system without materialising a dense matrix."""
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    residual_sq = torch.dot(residual, residual)
    rhs_norm = float(torch.linalg.vector_norm(rhs.detach()))
    if rhs_norm == 0.0:
        return solution

    threshold = relative_tolerance * rhs_norm
    for _ in range(max_iterations):
        product = torch.sparse.mm(matrix, direction.unsqueeze(1)).squeeze(1)
        curvature = torch.dot(direction, product)
        curvature_value = float(curvature.detach())
        if not math.isfinite(curvature_value) or curvature_value <= 0.0:
            raise torch.linalg.LinAlgError("sparse coarse operator is not finite positive-definite")
        alpha = residual_sq / curvature
        solution = solution + alpha * direction
        next_residual = residual - alpha * product
        next_residual_norm = float(torch.linalg.vector_norm(next_residual.detach()))
        if not math.isfinite(next_residual_norm):
            raise torch.linalg.LinAlgError("sparse coarse residual became non-finite")
        if next_residual_norm <= threshold:
            return solution
        next_residual_sq = torch.dot(next_residual, next_residual)
        beta = next_residual_sq / residual_sq
        direction = next_residual + beta * direction
        residual = next_residual
        residual_sq = next_residual_sq
    raise torch.linalg.LinAlgError("sparse coarse conjugate-gradient solve did not converge")


@dataclass
class _Level:
    A: torch.Tensor  # sparse (n_l × n_l)
    R: torch.Tensor | None  # sparse (n_{l+1} × n_l), None on coarsest
    diag: torch.Tensor  # (n_l,) for Jacobi smoother
    grid_shape: tuple[int, int, int]


class GeometricMultigrid:
    """
    V-cycle GMG for Cartesian TPFA Poisson-type systems.

    Args:
        nx, ny, nz: fine-grid dims (cells, x-fastest ordering).
        n_pre, n_post: pre/post smoothing iterations per level.
        omega: Jacobi relaxation factor (2/3 textbook optimum).
        n_coarsest: switch to direct dense solve when level n_dof ≤ this.
        tol: outer relative-residual tolerance.
        max_iter: outer V-cycle iterations.
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
        validated = KrylovConfig(
            method="gmres",
            max_iterations=max_iter,
            tolerance=tol,
            restart=1,
        )
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.n_pre, self.n_post = int(n_pre), int(n_post)
        self.omega = float(omega)
        self.n_coarsest = int(n_coarsest)
        self.tol = validated.tolerance
        self.effective_tolerance = self.tol
        self.max_iter = validated.max_iterations
        self.last_diagnostics: FlowConvergenceDiagnostics | None = None
        self.levels: list[_Level] = []
        self._A_shape: tuple[int, int] | None = None
        # Restriction operators depend ONLY on the grid shape (nx, ny, nz), not on the
        # operator VALUES, so they are built once and reused across solves. The
        # value-dependent operator hierarchy (level A's, Galerkin coarse A's, Jacobi
        # diagonals) is reassembled on every setup so a same-shape A with new VALUES
        # (e.g. the Jacobian inside a Newton loop) is never solved against a stale
        # operator.
        self._restrictions: list[tuple[torch.Tensor | None, tuple[int, int, int]]] | None = None
        self._restr_key: tuple[torch.device, torch.dtype] | None = None

    def _restriction_hierarchy(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[tuple[torch.Tensor | None, tuple[int, int, int]]]:
        """Grid-shape-only restriction operators: one ``(R, grid_shape)`` per level,
        with ``R is None`` on the coarsest. Value-independent, so cached across solves."""
        restr: list[tuple[torch.Tensor | None, tuple[int, int, int]]] = []
        cur_nx, cur_ny, cur_nz = self.nx, self.ny, self.nz
        while True:
            n = cur_nx * cur_ny * cur_nz
            if n <= self.n_coarsest:
                restr.append((None, (cur_nx, cur_ny, cur_nz)))
                break
            R, (nx_c, ny_c, nz_c) = _build_restriction(
                cur_nx,
                cur_ny,
                cur_nz,
                device,
                dtype,
            )
            if R.shape[0] == n:
                # No further coarsening possible: stop.
                restr.append((None, (cur_nx, cur_ny, cur_nz)))
                break
            restr.append((R, (cur_nx, cur_ny, cur_nz)))
            cur_nx, cur_ny, cur_nz = nx_c, ny_c, nz_c
        return restr

    def setup(self, A: torch.Tensor) -> None:
        """(Re)assemble the operator hierarchy from a fine-level sparse Jacobian.

        The restriction operators are grid-shape-only and cached; the value-dependent
        operators (each level's ``A``/``diag`` and the Galerkin coarse ``A``) are
        rebuilt every call so an updated ``A`` (same shape, new values) is honoured."""
        if A.shape[0] != self.nx * self.ny * self.nz:
            raise GeoBrainError(
                f"GeometricMultigrid: A.shape[0]={A.shape[0]} but "
                f"nx*ny*nz={self.nx * self.ny * self.nz}",
            )
        if not A.is_sparse:
            A = A.to_sparse()
        A = A.coalesce()
        key = (A.device, A.dtype)
        if self._restrictions is None or self._restr_key != key:
            self._restrictions = self._restriction_hierarchy(A.device, A.dtype)
            self._restr_key = key
        levels: list[_Level] = []
        cur_A = A
        for R, grid_shape in self._restrictions:
            diag = _diag(cur_A)
            levels.append(_Level(A=cur_A, R=R, diag=diag, grid_shape=grid_shape))
            if R is None:
                break
            cur_A = _galerkin(cur_A, R)
        self.levels = levels
        self._A_shape = tuple(A.shape)

    def _smooth(
        self, A: torch.Tensor, b: torch.Tensor, diag: torch.Tensor, x: torch.Tensor, n_iter: int
    ) -> torch.Tensor:
        """Weighted Jacobi: ``x ← x + ω · diag⁻¹ · (b − A·x)``."""
        for _ in range(n_iter):
            r = b - torch.sparse.mm(A, x.view(-1, 1)).view(-1)
            x = x + self.omega * (r / (diag + EPS))
        return x

    def _vcycle(self, level_idx: int, b: torch.Tensor) -> torch.Tensor:
        L = self.levels[level_idx]
        if L.R is None:
            coarse_tolerance = _effective_relative_tolerance(
                b.dtype,
                min(1.0e-12, self.tol * 0.1),
            )
            return _sparse_conjugate_gradient(
                L.A,
                b,
                relative_tolerance=coarse_tolerance,
                max_iterations=max(16, 4 * int(L.A.shape[0])),
            )
        x = torch.zeros_like(b)
        x = self._smooth(L.A, b, L.diag, x, self.n_pre)
        r_l = b - torch.sparse.mm(L.A, x.view(-1, 1)).view(-1)
        r_c = torch.sparse.mm(L.R, r_l.view(-1, 1)).view(-1)
        e_c = self._vcycle(level_idx + 1, r_c)
        # Prolongation = R.T (transpose-injection).
        P = L.R.t()
        e_l = torch.sparse.mm(P, e_c.view(-1, 1)).view(-1)
        x = x + e_l
        x = self._smooth(L.A, b, L.diag, x, self.n_post)
        return x

    def solve(self, A: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Outer V-cycle iteration. Solves ``A · dx = −residual``."""
        # Reassemble the operator hierarchy from THIS A every call. The old code
        # rebuilt only on a shape change, so a same-shape update (a Newton Jacobian
        # that changes value but not size) kept iterating against the STALE operator
        # and converged to the wrong system. Restrictions are cached inside setup(),
        # so the reassembly is a handful of cheap sparse ops, not a full rebuild.
        self.last_diagnostics = None
        self.effective_tolerance = _effective_relative_tolerance(A.dtype, self.tol)
        if A.layout == torch.sparse_coo:
            values = A._values()
        elif A.layout != torch.strided:
            values = A.values()
        else:
            values = A
        residual_norm0 = float(torch.linalg.vector_norm(residual.detach()))
        if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(residual).all()):
            record = convergence_diagnostics(
                stage="linear",
                converged=False,
                reason="nonfinite",
                iterations=0,
                max_iterations=self.max_iter,
                initial_residual_norm=residual_norm0,
                residual_norm=residual_norm0,
                residual_history=(residual_norm0,),
            )
            self.last_diagnostics = record
            raise FlowConvergenceError(
                "geometric multigrid received a non-finite system",
                object_name="GeometricMultigrid",
                field="linear_system",
                expected="finite matrix and residual",
                actual=record.to_dict(),
                diagnostics=record,
            )
        self.setup(A)
        A0 = self.levels[0].A
        b = -residual
        x = torch.zeros_like(b)
        b_norm = float(torch.linalg.vector_norm(b))
        if b_norm == 0.0:
            self.last_diagnostics = convergence_diagnostics(
                stage="linear",
                converged=True,
                reason="tolerance",
                iterations=0,
                max_iterations=self.max_iter,
                initial_residual_norm=0.0,
                residual_norm=0.0,
                residual_history=(0.0,),
            )
            return x
        history: list[float] = []
        for iteration in range(self.max_iter):
            r = b - torch.sparse.mm(A0, x.view(-1, 1)).view(-1)
            residual_norm = float(torch.linalg.vector_norm(r.detach()))
            history.append(residual_norm)
            if not math.isfinite(residual_norm):
                record = convergence_diagnostics(
                    stage="linear",
                    converged=False,
                    reason="nonfinite",
                    iterations=iteration,
                    max_iterations=self.max_iter,
                    initial_residual_norm=history[0],
                    residual_norm=residual_norm,
                    residual_history=history,
                )
                self.last_diagnostics = record
                raise FlowConvergenceError(
                    "geometric multigrid residual became non-finite",
                    object_name="GeometricMultigrid",
                    field="convergence",
                    expected="finite decreasing residual",
                    actual=record.to_dict(),
                    diagnostics=record,
                )
            if residual_norm / b_norm < self.effective_tolerance:
                self.last_diagnostics = convergence_diagnostics(
                    stage="linear",
                    converged=True,
                    reason="tolerance",
                    iterations=iteration,
                    max_iterations=self.max_iter,
                    initial_residual_norm=history[0],
                    residual_norm=residual_norm,
                    residual_history=history,
                )
                return x
            try:
                correction = self._vcycle(0, r)
            except torch.linalg.LinAlgError as error:
                record = convergence_diagnostics(
                    stage="linear",
                    converged=False,
                    reason="breakdown",
                    iterations=iteration,
                    max_iterations=self.max_iter,
                    initial_residual_norm=history[0],
                    residual_norm=residual_norm,
                    residual_history=history,
                )
                self.last_diagnostics = record
                raise FlowConvergenceError(
                    "geometric multigrid coarse solve broke down",
                    object_name="GeometricMultigrid",
                    field="linear_solve",
                    expected="finite nonsingular coarse correction",
                    actual=record.to_dict(),
                    diagnostics=record,
                ) from error
            x = x + correction
        r = b - torch.sparse.mm(A0, x.view(-1, 1)).view(-1)
        residual_norm = float(torch.linalg.vector_norm(r.detach()))
        history.append(residual_norm)
        if residual_norm / b_norm < self.effective_tolerance:
            self.last_diagnostics = convergence_diagnostics(
                stage="linear",
                converged=True,
                reason="tolerance",
                iterations=self.max_iter,
                max_iterations=self.max_iter,
                initial_residual_norm=history[0],
                residual_norm=residual_norm,
                residual_history=history,
            )
            return x
        record = convergence_diagnostics(
            stage="linear",
            converged=False,
            reason="max_iterations",
            iterations=self.max_iter,
            max_iterations=self.max_iter,
            initial_residual_norm=history[0],
            residual_norm=residual_norm,
            residual_history=history,
        )
        self.last_diagnostics = record
        raise FlowConvergenceError(
            "geometric multigrid exhausted its V-cycle budget",
            object_name="GeometricMultigrid",
            field="convergence",
            expected=(
                f"relative residual < {self.effective_tolerance} "
                f"(requested {self.tol})"
            ),
            actual=record.to_dict(),
            diagnostics=record,
        )

    def apply(self, r: torch.Tensor) -> torch.Tensor:
        """Single V-cycle preconditioner application: ``z ≈ A⁻¹ · r``."""
        return self._vcycle(0, r)


__all__ = ["GeometricMultigrid"]
