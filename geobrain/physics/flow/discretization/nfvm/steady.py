"""NFVM steady single-phase drivers: structured Cartesian (2-D / 3-D) and unstructured
2-D Dirichlet BVP solvers built on the :mod:`.kernel` core.

``solve_nfvm_steady`` / ``solve_nfvm_steady_3d`` build the half-face geometry for a
structured grid with mirror ghost cells for Dirichlet boundaries;
``solve_nfvm_unstructured`` does the same from a polygonal :class:`MPFAGrid2D`. All
delegate the nonlinear Newton solve to :func:`.kernel.solve_nfvm` (except the
Cartesian 2-D driver, which inlines an equivalent damped Newton for legacy parity).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Protocol

import torch

from .....core import GeoBrainError
from ...errors import FlowContractError, FlowConvergenceError
from ...solvers.diagnostics import convergence_diagnostics
from ...solvers.linear_solvers import DirectSolver
from ..flux import scatter_boundary_outflow, scatter_internal_face_flux
from .kernel import (
    FaceRecord,
    LinearDiscretization,
    NFVMGeometry,
    _onesided,
    _oriented_normal,
    decompose_half_face,
    linear_discretization,
    nfvm_flux,
    solve_nfvm,
)


class _NFVMGrid2D(Protocol):
    """Structural interface required by the unstructured NFVM driver."""

    @property
    def cell_nodes(self) -> tuple[tuple[int, ...], ...]: ...

    @property
    def edge_cells(self) -> tuple[tuple[int, ...], ...]: ...

    @property
    def n_edges(self) -> int: ...

    def _cell_centroids_view(self) -> torch.Tensor: ...

    def edge_normal_area(self, edge: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def edge_midpoint(self, edge: int) -> torch.Tensor: ...


def solve_nfvm_unstructured(
    grid: _NFVMGrid2D,
    perm: torch.Tensor,
    dirichlet: Mapping[int, float | torch.Tensor],
    *,
    scheme: str = "ntpfa",
    tol: float = 1e-9,
    max_iter: int = 60,
) -> torch.Tensor:
    """NFVM steady BVP on a general **unstructured** 2-D :class:`MPFAGrid2D`.

    ``perm`` ``(n_cells, 2, 2)``; ``dirichlet`` maps a boundary edge index → its
    fixed pressure. Builds the half-face geometry from the polygonal grid (cell
    centroids, edge normals / lengths / midpoints) and solves with :func:`solve_nfvm`.
    Differentiable in ``perm`` / boundary data."""
    dtype = perm.dtype
    n = len(grid.cell_nodes)
    centroids = grid._cell_centroids_view()
    xc = [centroids[c] for c in range(n)]
    Kl = [perm[c] for c in range(n)]
    ghost_p: list[torch.Tensor] = []
    fpc: list[list[FaceRecord]] = [[] for _ in range(n)]
    gid = n
    for e in range(grid.n_edges):
        cells = grid.edge_cells[e]
        nrm, length = grid.edge_normal_area(e)
        xf = grid.edge_midpoint(e)
        if len(cells) == 2:
            a, b = cells
            fpc[a].append((b, _oriented_normal(xc[a], xf, nrm), float(length), xf))
            fpc[b].append((a, _oriented_normal(xc[b], xf, nrm), float(length), xf))
        else:
            (a,) = cells
            if e not in dirichlet:
                continue  # no-flow boundary: skip
            fpc[a].append((gid, _oriented_normal(xc[a], xf, nrm), float(length), xf))
            xc.append(2.0 * xf - xc[a])  # mirror ghost
            Kl.append(perm[a])
            ghost_p.append(torch.as_tensor(dirichlet[e], dtype=dtype))
            gid += 1
    geom = NFVMGeometry(torch.stack(xc), torch.stack(Kl), fpc, ghost_p, n)
    return solve_nfvm(geom, scheme=scheme, tol=tol, max_iter=max_iter)


def solve_nfvm_steady_3d(
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    perm: torch.Tensor,
    pbc: Callable[[int, int, int], float | torch.Tensor],
    *,
    scheme: str = "ntpfa",
    tol: float = 1e-9,
    max_iter: int = 60,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """NFVM steady BVP on a structured **3-D** hex grid (the 3-D triplet basis).

    ``perm`` ``(n_cells, 3, 3)``; ``pbc(i, j, k)`` the Dirichlet pressure for an
    out-of-domain neighbour. Each cell has 6 faces; the co-normal decomposes over a
    non-negative **triplet** of the 6 face HAP directions."""
    if perm.shape[-2:] != (3, 3):
        raise GeoBrainError(
            "3-D NFVM needs (n_cells, 3, 3) permeability",
            object_name="solve_nfvm_steady_3d",
            field="perm",
            expected="(n_cells, 3, 3)",
            actual=tuple(perm.shape),
        )
    n = nx * ny * nz
    d = torch.tensor([dx, dy, dz], dtype=dtype)

    def cid(i: int, j: int, k: int) -> int:
        return (k * ny + j) * nx + i

    xc = [
        torch.tensor([(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz], dtype=dtype)
        for k in range(nz)
        for j in range(ny)
        for i in range(nx)
    ]
    Kl = [perm[c] for c in range(n)]
    ghost_p: list[torch.Tensor] = []
    fpc: list[list[FaceRecord]] = [[] for _ in range(n)]
    dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    gid = n
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                c = cid(i, j, k)
                for di, dj, dk in dirs:
                    ii, jj, kk = i + di, j + dj, k + dk
                    no = torch.tensor([float(di), float(dj), float(dk)], dtype=dtype)
                    A = float(
                        dx * dy * dz / d[0 if di else (1 if dj else 2)]
                    )  # face area ⟂ to the axis
                    xf = xc[c] + 0.5 * no * d
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        fpc[c].append((cid(ii, jj, kk), no, A, xf))
                    else:
                        fpc[c].append((gid, no, A, xf))
                        xc.append(2.0 * xf - xc[c])
                        Kl.append(perm[c])
                        ghost_p.append(torch.as_tensor(pbc(ii, jj, kk), dtype=dtype))
                        gid += 1
    geom = NFVMGeometry(torch.stack(xc), torch.stack(Kl), fpc, ghost_p, n)
    return solve_nfvm(geom, scheme=scheme, tol=tol, max_iter=max_iter)


# --------------------------------------------------------------------------
# Structured Cartesian driver with ghost cells for Dirichlet
# --------------------------------------------------------------------------
class _CartNFVM:
    """Builds the NFVM half-face data for a structured Cartesian grid + ghosts."""

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        perm: torch.Tensor,
        pbc: Callable[[int, int], float | torch.Tensor],
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        n = nx * ny
        self.n = n
        # cell centroids + perm, then ghosts (one per boundary face)
        x = [
            torch.tensor([(i + 0.5) * dx, (j + 0.5) * dy], dtype=dtype)
            for j in range(ny)
            for i in range(nx)
        ]
        Kl = [perm[c] for c in range(n)]
        self.ghost_p: list[torch.Tensor] = []  # Dirichlet pressure per ghost
        nbr: dict[tuple[int, tuple[int, int]], int] = {}  # (c, dir) -> neighbor index
        face_centroid: dict[tuple[int, tuple[int, int]], torch.Tensor] = {}
        nrm: dict[tuple[int, tuple[int, int]], torch.Tensor] = {}
        dirs: list[tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        gid = n
        for j in range(ny):
            for i in range(nx):
                c = j * nx + i
                for di, dj in dirs:
                    ii, jj = i + di, j + dj
                    xf = torch.tensor(
                        [(i + 0.5 + 0.5 * di) * dx, (j + 0.5 + 0.5 * dj) * dy], dtype=dtype
                    )
                    no = torch.tensor([float(di), float(dj)], dtype=dtype)
                    face_centroid[(c, (di, dj))] = xf
                    nrm[(c, (di, dj))] = no
                    if 0 <= ii < nx and 0 <= jj < ny:
                        nbr[(c, (di, dj))] = jj * nx + ii
                    else:  # ghost: mirror, same perm, Dirichlet value
                        nbr[(c, (di, dj))] = gid
                        x.append(2.0 * xf - x[c])
                        Kl.append(perm[c])
                        self.ghost_p.append(torch.as_tensor(pbc(ii, jj), dtype=dtype))
                        gid += 1
        self.x = torch.stack(x)
        self.K = torch.stack(Kl)
        self.dirs = dirs
        self.nbr, self.face_centroid, self.nrm = nbr, face_centroid, nrm

    def faces_of_cell(self, c: int) -> list[FaceRecord]:
        out: list[FaceRecord] = []
        for di, dj in self.dirs:
            A = self.dy if di != 0 else self.dx
            out.append(
                (
                    self.nbr[(c, (di, dj))],
                    self.nrm[(c, (di, dj))],
                    A,
                    self.face_centroid[(c, (di, dj))],
                )
            )
        return out

    def build(
        self, scheme: str
    ) -> tuple[
        list[tuple[LinearDiscretization, LinearDiscretization]],
        list[LinearDiscretization],
    ]:
        """Per interior face → (L_disc, R_disc); per boundary face → one-sided L_disc."""
        n = self.n
        interior: list[tuple[LinearDiscretization, LinearDiscretization]] = []
        boundary: list[LinearDiscretization] = []
        seen: set[tuple[int, int]] = set()
        for c in range(n):
            fc = self.faces_of_cell(c)
            for k, (other, no, A, xf) in enumerate(fc):
                key = (min(c, other), max(c, other)) if other < n else (c, other)
                if key in seen:
                    continue
                seen.add(key)
                target = (other, no, A, xf)
                dL = decompose_half_face(c, target, fc, self.K, self.x)
                if other < n:  # interior face
                    fo = self.faces_of_cell(other)
                    # the matching half face from the other cell (normal flips)
                    tgt_o = next(t for t in fo if t[0] == c)
                    dR = decompose_half_face(other, tgt_o, fo, self.K, self.x)
                    if dL is None or dR is None:
                        T = float(A * (no @ (self.K[c] @ no))) / (
                            self.dx if no[0] != 0 else self.dy
                        )
                        L_disc = LinearDiscretization(
                            left=c, right=other, t_l=-T, t_r=T, mpfa=[]
                        )
                        R_disc = LinearDiscretization(
                            left=c, right=other, t_l=-T, t_r=T, mpfa=[]
                        )
                    else:
                        L_disc = linear_discretization(dL, c, other)
                        R_disc = linear_discretization(dR, c, other)
                    interior.append((L_disc, R_disc))
                else:  # boundary face (Dirichlet ghost)
                    if dL is None:
                        T = float(A * (no @ (self.K[c] @ no))) / (
                            0.5 * (self.dx if no[0] != 0 else self.dy)
                        )
                        L_disc = LinearDiscretization(
                            left=c, right=other, t_l=-T, t_r=T, mpfa=[]
                        )
                    else:
                        L_disc = linear_discretization(dL, c, other)
                    boundary.append(L_disc)
        return interior, boundary


def solve_nfvm_steady(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    perm: torch.Tensor,
    pbc: Callable[[int, int], float | torch.Tensor],
    *,
    scheme: str = "ntpfa",
    tol: float = 1e-9,
    max_iter: int = 60,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Steady single-phase NFVM (NTPFA/NMPFA) Dirichlet BVP on a Cartesian grid.

    ``perm`` ``(n_cells, 2, 2)``; ``pbc(i, j)`` the Dirichlet pressure for an
    out-of-domain neighbour. Returns the ``(n_cells,)`` cell pressures. The flux is
    the monotone nonlinear two-point form, solved by Newton (initialised from the
    boundary mean). Differentiable in ``perm`` / boundary data."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise FlowContractError(
            "max_iter must be a positive integer",
            object_name="solve_nfvm_steady",
            field="max_iter",
            expected="positive integer",
            actual=max_iter,
        )
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not math.isfinite(float(tol))
        or float(tol) <= 0
    ):
        raise FlowContractError(
            "tol must be positive and finite",
            object_name="solve_nfvm_steady",
            field="tol",
            expected="finite value > 0",
            actual=tol,
        )
    if perm.shape[-2:] != (2, 2):
        raise GeoBrainError(
            "NFVM needs (n_cells, 2, 2) permeability",
            object_name="solve_nfvm_steady",
            field="perm",
            expected="(n_cells, 2, 2)",
            actual=tuple(perm.shape),
        )
    g = _CartNFVM(nx, ny, dx, dy, perm, pbc, dtype=dtype)
    interior, boundary = g.build(scheme)
    p_aug_const = float(torch.stack(g.ghost_p).mean()) if g.ghost_p else 0.0

    def residual(pc: torch.Tensor, mu_half: bool = False) -> torch.Tensor:
        p = torch.cat([pc, torch.stack(g.ghost_p)]) if g.ghost_p else pc
        R = pc.new_zeros(g.n)
        for L_disc, R_disc in interior:
            F = nfvm_flux(p, L_disc, R_disc, scheme, mu_half=mu_half)
            face_cells = torch.tensor(
                [[L_disc["left"], L_disc["right"]]], dtype=torch.long, device=F.device
            )
            R = R + scatter_internal_face_flux(F.reshape(1), face_cells, g.n)
        for L_disc in boundary:
            q = -_onesided(p, L_disc)[0]
            cells = torch.tensor([L_disc["left"]], dtype=torch.long, device=q.device)
            R = R + scatter_boundary_outflow(q.reshape(1), cells, g.n)
        return R

    # initial guess: solve the linear avgMPFA system (μ frozen at ½)
    p = perm.new_full((g.n,), p_aug_const)
    r0 = residual(p, mu_half=True)
    if float(torch.linalg.vector_norm(r0.detach())) != 0.0:
        J0 = torch.autograd.functional.jacobian(
            lambda q: residual(q, mu_half=True),
            p,
            vectorize=True,
        )
        p = p + DirectSolver().solve(J0, r0)

    # damped Newton on the nonlinear system
    history: list[float] = []
    for iteration in range(max_iter):
        r = residual(p)
        rn = float(torch.linalg.vector_norm(r.detach(), ord=float("inf")))
        history.append(rn)
        if not math.isfinite(rn):
            record = convergence_diagnostics(
                stage="nonlinear",
                converged=False,
                reason="nonfinite",
                iterations=iteration,
                max_iterations=max_iter,
                initial_residual_norm=history[0],
                residual_norm=rn,
                residual_history=history,
            )
            raise FlowConvergenceError(
                "NFVM residual became non-finite",
                object_name="solve_nfvm_steady",
                field="residual",
                expected="finite convergent residual",
                actual=record.to_dict(),
                diagnostics=record,
            )
        if rn < tol:
            return p
        J = torch.autograd.functional.jacobian(residual, p, vectorize=True)
        update = DirectSolver().solve(J, r)
        alpha = 1.0
        found = False
        for _ls in range(25):  # back-tracking line search on |r|∞
            r_try = residual(p + alpha * update)
            if float(torch.linalg.vector_norm(r_try.detach(), ord=float("inf"))) < rn:
                found = True
                break
            alpha *= 0.5
        if not found:
            record = convergence_diagnostics(
                stage="nonlinear",
                converged=False,
                reason="line_search",
                iterations=iteration + 1,
                max_iterations=max_iter,
                initial_residual_norm=history[0],
                residual_norm=rn,
                residual_history=history,
            )
            raise FlowConvergenceError(
                "NFVM line search could not find a decreasing step",
                object_name="solve_nfvm_steady",
                field="line_search",
                expected="finite decreasing residual",
                actual=record.to_dict(),
                diagnostics=record,
            )
        p = p + alpha * update
    final_norm = float(torch.linalg.vector_norm(residual(p).detach(), ord=float("inf")))
    history.append(final_norm)
    if final_norm < tol:
        return p
    record = convergence_diagnostics(
        stage="nonlinear",
        converged=False,
        reason="max_iterations",
        iterations=max_iter,
        max_iterations=max_iter,
        initial_residual_norm=history[0],
        residual_norm=final_norm,
        residual_history=history,
    )
    raise FlowConvergenceError(
        "NFVM steady solve exhausted its Newton iteration budget",
        object_name="solve_nfvm_steady",
        field="convergence",
        expected=f"residual norm < {tol}",
        actual=record.to_dict(),
        diagnostics=record,
    )
