"""NFVM discretization core: the reusable kernel island.

Harmonic averaging points, positive basis / conical decomposition,
one-sided half-face decomposition and the nonlinear flux evaluation, plus the
geometry-generic container :class:`NFVMGeometry`, the assembler ``_build_nfvm``
and the steady single-phase Dirichlet driver ``solve_nfvm``. This is the
grid-agnostic, dimension-generic part that the structured / unstructured drivers
and the physics solvers build on.

See the package docstring (``__init__``) for the scheme overview.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias, TypedDict

import torch

from ...errors import FlowContractError, FlowConvergenceError
from ...solvers.diagnostics import convergence_diagnostics
from ...solvers.linear_solvers import DirectSolver
from ..flux import scatter_boundary_outflow, scatter_internal_face_flux


FaceArea: TypeAlias = float | torch.Tensor
FaceRecord: TypeAlias = tuple[int, torch.Tensor, FaceArea, torch.Tensor]
BasisResult: TypeAlias = tuple[tuple[int, ...], tuple[torch.Tensor, ...]]


class HalfFaceDecomposition(TypedDict):
    """Typed coefficients for one cell-side face decomposition."""

    self: int
    other_cells: list[int]
    self_weights: list[torch.Tensor]
    other_cells_weights: list[torch.Tensor]
    triplet_weights: list[torch.Tensor]


class LinearDiscretization(TypedDict):
    """Typed one-sided linear flux stencil consumed by :func:`nfvm_flux`."""

    left: int
    right: int
    t_l: float | torch.Tensor
    t_r: float | torch.Tensor
    mpfa: list[tuple[int, torch.Tensor]]


# --------------------------------------------------------------------------
# Harmonic averaging point
# --------------------------------------------------------------------------
def harmonic_average_point(
    K1: torch.Tensor,
    x1: torch.Tensor,
    K2: torch.Tensor,
    x2: torch.Tensor,
    xf: torch.Tensor,
    nf: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """HAP between two cells across a face + interpolation weights ``(w1, w2)``."""
    lam1 = nf @ (K1 @ nf)
    g1 = K1 @ nf - lam1 * nf
    lam2 = nf @ (K2 @ nf)
    g2 = K2 @ nf - lam2 * nf
    d1s = (xf - x1) @ nf
    d2s = (xf - x2) @ nf
    y1 = x1 + d1s * nf
    y2 = x2 + d2s * nf
    d1, d2 = torch.abs(d1s), torch.abs(d2s)
    w1 = lam1 * d2
    w2 = lam2 * d1
    wt = (w1 + w2).clamp_min(1e-300)
    hp = (w1 * y1 + w2 * y2 + d1 * d2 * (g1 - g2)) / wt
    return hp, (w1 / wt, w2 / wt)


# --------------------------------------------------------------------------
# Positive basis: dimension-generic: 2-D duo / 3-D triplet
# --------------------------------------------------------------------------
def _basis_coefficients(
    cols: Sequence[torch.Tensor], conormal: torch.Tensor
) -> torch.Tensor | None:
    """Solve ``M·c = conormal`` with ``M`` the columns ``cols`` (a duo or triplet);
    ``None`` if degenerate."""
    M = torch.stack(cols, dim=1)
    if float(torch.abs(torch.linalg.det(M)).detach()) < 1e-8:
        return None
    return torch.linalg.solve(M, conormal)


def _duo_coefficients(
    ti: torch.Tensor, tj: torch.Tensor, conormal: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:  # kept for backward compat
    c = _basis_coefficients([ti, tj], conormal)
    return None if c is None else (c[0], c[1])


def find_minimizing_basis(
    x_t: torch.Tensor,
    conormal: torch.Tensor,
    points: Sequence[torch.Tensor],
) -> BasisResult | None:
    """Conical decomposition of ``conormal`` over ``(points[i] − x_t)``.

    The decomposition works in 2-D
    (a **duo**, 2 vectors) and 3-D (a **triplet**, 3 vectors), keyed off
    ``len(conormal)``.

    Candidates are normalised and sorted by their angle to ``conormal`` (so the
    most aligned basis is found first); the inner loop returns the first
    non-negative basis with ``max(weights) ≤ 1`` (else the best non-negative one).
    Returns ``(idx_tuple, weights)`` with
    ``conormal ≈ Σ w·(points[idx] − x_t)``, ``w ≥ 0``, or ``None`` if no
    non-negative basis exists (caller falls back to two-point)."""
    dim = int(conormal.shape[0])
    k = dim  # basis size = dimension
    t_raw = [pt - x_t for pt in points]
    tnorm = [v.norm().clamp_min(1e-300) for v in t_raw]
    t_unit = [v / nrm for v, nrm in zip(t_raw, tnorm)]
    conormal_norm = conormal.norm().clamp_min(1e-300)
    normalized_conormal = conormal / conormal_norm

    def angle(i: int) -> float:
        c = (t_unit[i] @ normalized_conormal).clamp(0.0, 1.0)
        return float(torch.arccos(c).detach())

    order = sorted(range(len(points)), key=angle)
    N = len(order)
    best: BasisResult | None = None
    best_val = float("inf")
    import itertools

    for combo in itertools.combinations(range(N), k):  # ordered candidate combinations
        idx = tuple(order[a] for a in combo)
        coef = _basis_coefficients([t_unit[i] for i in idx], normalized_conormal)
        if coef is None:
            continue
        cd = [float(coef[m].detach()) for m in range(k)]
        if all(c >= 0.0 for c in cd):
            val = max(cd)
            w = tuple(conormal_norm * coef[m] / tnorm[idx[m]] for m in range(k))
            if val <= 1.0:
                return idx, w
            if val < best_val:
                best_val = val
                best = (idx, w)
    return best


def find_minimizing_basis_2d(
    x_t: torch.Tensor,
    conormal: torch.Tensor,
    points: Sequence[torch.Tensor],
) -> BasisResult | None:
    """Backward-compatible 2-D entry point (see :func:`find_minimizing_basis`)."""
    return find_minimizing_basis(x_t, conormal, points)


# --------------------------------------------------------------------------
# Half-face decomposition
# --------------------------------------------------------------------------
def decompose_half_face(
    cell: int,
    target: FaceRecord,
    faces_of_cell: Sequence[FaceRecord],
    K: torch.Tensor,
    x_cells: torch.Tensor,
) -> HalfFaceDecomposition | None:
    """One-cell co-normal decomposition for the half face ``(cell, target)``.

    ``target`` / each entry of ``faces_of_cell`` is ``(other_cell, n_out, area,
    x_face)``. Returns a decomposition dict, or ``None`` if no non-negative basis
    exists (caller falls back to two-point)."""
    other_t, n_t, A_t, _ = target
    AKn = K[cell] @ (A_t * n_t)
    others, haps, self_w, other_w = [], [], [], []
    for other, n_f, _, x_f in faces_of_cell:
        hp, (sw, ow) = harmonic_average_point(
            K[cell], x_cells[cell], K[other], x_cells[other], x_f, n_f
        )
        others.append(other)
        haps.append(hp)
        self_w.append(sw)
        other_w.append(ow)
    res = find_minimizing_basis(x_cells[cell], AKn, haps)
    if res is None:
        return None
    idxs, ws = res  # 2 (2-D) or 3 (3-D) basis directions
    return {
        "self": cell,
        "other_cells": [others[i] for i in idxs],
        "self_weights": [self_w[i] for i in idxs],
        "other_cells_weights": [other_w[i] for i in idxs],
        "triplet_weights": list(ws),
    }


def _two_point_trans(
    decomp: HalfFaceDecomposition, cell: int
) -> float | torch.Tensor:
    T: float | torch.Tensor = 0.0
    if decomp["self"] == cell:
        for sw, tw in zip(decomp["self_weights"], decomp["triplet_weights"]):
            T = T + sw * tw
    for i, c in enumerate(decomp["other_cells"]):
        if c == cell:
            T = T + decomp["triplet_weights"][i] * decomp["other_cells_weights"][i]
    return T


def linear_discretization(
    decomp: HalfFaceDecomposition, left: int, right: int
) -> LinearDiscretization:
    """Collapse a half-face decomposition to a one-sided flux
    ``t_l·p_l + t_r·p_r + Σ_c t_c·p_c`` (NFVMLinearDiscretization)."""
    t_l = _two_point_trans(decomp, left)
    t_r = _two_point_trans(decomp, right)
    w_tot = -sum(decomp["triplet_weights"])
    if decomp["self"] == left:
        sgn = 1.0
        t_l = t_l + w_tot
    else:
        sgn = -1.0
        t_r = t_r + w_tot
    mpfa: list[tuple[int, torch.Tensor]] = []
    for i, c in enumerate(decomp["other_cells"]):
        if c != left and c != right:
            mpfa.append((c, sgn * decomp["triplet_weights"][i] * decomp["other_cells_weights"][i]))
    return {"left": left, "right": right, "t_l": sgn * t_l, "t_r": sgn * t_r, "mpfa": mpfa}


# --------------------------------------------------------------------------
# Nonlinear flux evaluation
# --------------------------------------------------------------------------
def _onesided(
    p: torch.Tensor, disc: LinearDiscretization
) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = disc["left"], disc["right"]
    rem = p.new_zeros(())
    for c, v in disc["mpfa"]:
        rem = rem + v * p[c]
    q = disc["t_l"] * p[left] + disc["t_r"] * p[right] + rem
    return q, rem


def nfvm_flux(
    p: torch.Tensor,
    L_disc: LinearDiscretization,
    R_disc: LinearDiscretization,
    scheme: str = "ntpfa",
    mu_half: bool = False,
) -> torch.Tensor:
    """Return the canonical nonlinear NFVM flux from left to right.

    Positive flux travels from ``L_disc['left']`` to ``L_disc['right']``.
    The half-face algebra historically returns the opposite sign, so the
    orientation is normalised exactly once at this kernel boundary.

    ``mu_half=True`` freezes ``μ_L = μ_R = ½``, the *linear* avgMPFA flux, used as
    a robust initial guess for the nonlinear Newton iteration."""
    q_l, r_l = _onesided(p, L_disc)
    q_r0, r_r0 = _onesided(p, R_disc)
    q_r, r_r = -q_r0, -r_r0
    if mu_half:
        mu_l = mu_r = p.new_tensor(0.5)
    else:
        if scheme == "nmpfa":
            r_lw, r_rw = torch.abs(r_l), torch.abs(r_r)
        else:
            r_lw, r_rw = r_l, r_r
        r_tot = r_lw + r_rw
        if float(torch.abs(r_tot).detach()) < 1e-10:
            mu_l = mu_r = p.new_tensor(0.5)
        else:
            mu_l, mu_r = r_rw / r_tot, r_lw / r_tot
    return -(mu_l * q_l - mu_r * q_r)


# --------------------------------------------------------------------------
# Generic geometry-driven NFVM (unstructured 2-D / 3-D)
# --------------------------------------------------------------------------
class NFVMGeometry:
    """Grid-agnostic container of the per-cell half-face data the NFVM operators
    need. ``x`` ``(n_total, dim)`` cell + ghost centroids; ``K`` ``(n_total, dim,
    dim)`` perm tensors (ghosts copy the adjacent cell); ``faces_per_cell[c]`` a list
    of ``(neighbor, n_out, area, x_face)`` for each real cell ``c`` (``neighbor ≥
    n_real`` is a Dirichlet ghost); ``ghost_p`` the ghost pressures; ``n_real`` the
    number of real cells. Works for any dimension / connectivity."""

    def __init__(
        self,
        x: torch.Tensor,
        K: torch.Tensor,
        faces_per_cell: list[list[FaceRecord]],
        ghost_p: list[torch.Tensor],
        n_real: int,
    ) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] not in {2, 3}:
            raise FlowContractError(
                "NFVM coordinates must be a floating [cell, xy|xyz] tensor",
                object_name="NFVMGeometry",
                field="x",
                expected="[cell, 2] or [cell, 3]",
                actual=type(x).__name__ if not isinstance(x, torch.Tensor) else tuple(x.shape),
            )
        if x.dtype not in {torch.float32, torch.float64}:
            raise FlowContractError(
                "NFVM coordinates require a supported floating dtype",
                object_name="NFVMGeometry",
                field="x.dtype",
                expected=(str(torch.float32), str(torch.float64)),
                actual=str(x.dtype),
            )
        if not bool(torch.isfinite(x).all()):
            raise FlowContractError(
                "NFVM coordinates must be finite",
                object_name="NFVMGeometry",
                field="x",
                expected="all finite",
                actual="non-finite entries present",
            )
        dimension = int(x.shape[1])
        if (
            not isinstance(K, torch.Tensor)
            or K.ndim != 3
            or K.shape != (x.shape[0], dimension, dimension)
        ):
            raise FlowContractError(
                "NFVM permeability must align with all real and ghost coordinates",
                object_name="NFVMGeometry",
                field="K",
                expected=(int(x.shape[0]), dimension, dimension),
                actual=type(K).__name__ if not isinstance(K, torch.Tensor) else tuple(K.shape),
            )
        if K.dtype != x.dtype or K.device != x.device:
            raise FlowContractError(
                "NFVM permeability dtype/device must match coordinates",
                object_name="NFVMGeometry",
                field="K",
                expected={"dtype": str(x.dtype), "device": str(x.device)},
                actual={"dtype": str(K.dtype), "device": str(K.device)},
            )
        if not isinstance(n_real, int) or isinstance(n_real, bool) or not 0 < n_real <= x.shape[0]:
            raise FlowContractError(
                "NFVM real-cell count is outside the coordinate layout",
                object_name="NFVMGeometry",
                field="n_real",
                expected=f"1..{x.shape[0]}",
                actual=n_real,
            )
        if not isinstance(faces_per_cell, list) or len(faces_per_cell) != n_real:
            raise FlowContractError(
                "NFVM face records must have one row per real cell",
                object_name="NFVMGeometry",
                field="faces_per_cell",
                expected=n_real,
                actual=(
                    len(faces_per_cell)
                    if isinstance(faces_per_cell, list)
                    else type(faces_per_cell).__name__
                ),
            )
        for cell, records in enumerate(faces_per_cell):
            if not isinstance(records, list):
                raise FlowContractError(
                    "NFVM face records must be lists",
                    object_name="NFVMGeometry",
                    field=f"faces_per_cell[{cell}]",
                    expected="list[face record]",
                    actual=type(records).__name__,
                )
            for face, record in enumerate(records):
                if not isinstance(record, tuple) or len(record) != 4:
                    raise FlowContractError(
                        "NFVM face record must contain neighbor, normal, area, and centroid",
                        object_name="NFVMGeometry",
                        field=f"faces_per_cell[{cell}][{face}]",
                        expected="tuple[neighbor, normal, area, centroid]",
                        actual=record,
                    )
                neighbor, normal, area, centroid = record
                if (
                    isinstance(neighbor, bool)
                    or not isinstance(neighbor, int)
                    or not 0 <= neighbor < x.shape[0]
                ):
                    raise FlowContractError(
                        "NFVM face neighbor is outside the coordinate layout",
                        object_name="NFVMGeometry",
                        field=f"faces_per_cell[{cell}][{face}].neighbor",
                        expected=f"0..{x.shape[0] - 1}",
                        actual=neighbor,
                    )
                for label, tensor, expected_shape in (
                    ("normal", normal, (dimension,)),
                    ("centroid", centroid, (dimension,)),
                ):
                    if not isinstance(tensor, torch.Tensor) or tensor.shape != expected_shape:
                        raise FlowContractError(
                            f"NFVM face {label} has an invalid shape",
                            object_name="NFVMGeometry",
                            field=f"faces_per_cell[{cell}][{face}].{label}",
                            expected=expected_shape,
                            actual=(
                                type(tensor).__name__
                                if not isinstance(tensor, torch.Tensor)
                                else tuple(tensor.shape)
                            ),
                        )
                    if tensor.dtype != x.dtype or tensor.device != x.device:
                        raise FlowContractError(
                            f"NFVM face {label} dtype/device must match coordinates",
                            object_name="NFVMGeometry",
                            field=f"faces_per_cell[{cell}][{face}].{label}",
                            expected={"dtype": str(x.dtype), "device": str(x.device)},
                            actual={"dtype": str(tensor.dtype), "device": str(tensor.device)},
                        )
                    if not bool(torch.isfinite(tensor).all()):
                        raise FlowContractError(
                            f"NFVM face {label} must be finite",
                            object_name="NFVMGeometry",
                            field=f"faces_per_cell[{cell}][{face}].{label}",
                            expected="all finite",
                            actual="non-finite entries present",
                        )
                if isinstance(area, torch.Tensor):
                    if area.ndim != 0 or area.dtype != x.dtype or area.device != x.device:
                        raise FlowContractError(
                            "NFVM face area dtype/device must match coordinates",
                            object_name="NFVMGeometry",
                            field=f"faces_per_cell[{cell}][{face}].area",
                            expected={"shape": (), "dtype": str(x.dtype), "device": str(x.device)},
                            actual={
                                "shape": tuple(area.shape),
                                "dtype": str(area.dtype),
                                "device": str(area.device),
                            },
                        )
                    valid_area = bool(torch.isfinite(area)) and bool(area > 0)
                else:
                    try:
                        area_value = float(area)
                        valid_area = math.isfinite(area_value) and area_value > 0
                    except (TypeError, ValueError, OverflowError):
                        valid_area = False
                if not valid_area:
                    raise FlowContractError(
                        "NFVM face area must be finite and positive",
                        object_name="NFVMGeometry",
                        field=f"faces_per_cell[{cell}][{face}].area",
                        expected="finite and > 0",
                        actual=area,
                    )
        if not isinstance(ghost_p, list) or len(ghost_p) > x.shape[0] - n_real:
            raise FlowContractError(
                "NFVM ghost pressures cannot exceed the ghost coordinate count",
                object_name="NFVMGeometry",
                field="ghost_p",
                expected=f"0..{int(x.shape[0]) - n_real}",
                actual=len(ghost_p) if isinstance(ghost_p, list) else type(ghost_p).__name__,
            )
        for ghost, pressure in enumerate(ghost_p):
            if (
                not isinstance(pressure, torch.Tensor)
                or pressure.numel() != 1
                or pressure.dtype != x.dtype
                or pressure.device != x.device
            ):
                raise FlowContractError(
                    "NFVM ghost pressure dtype/device must match coordinates",
                    object_name="NFVMGeometry",
                    field=f"ghost_p[{ghost}]",
                    expected={"numel": 1, "dtype": str(x.dtype), "device": str(x.device)},
                    actual=(
                        type(pressure).__name__
                        if not isinstance(pressure, torch.Tensor)
                        else {
                            "numel": pressure.numel(),
                            "dtype": str(pressure.dtype),
                            "device": str(pressure.device),
                        }
                    ),
                )
        self._x = x.clone()
        self._K = K.clone()
        self._fpc = tuple(
            tuple(
                (
                    neighbor,
                    normal.clone(),
                    area.clone() if isinstance(area, torch.Tensor) else area,
                    centroid.clone(),
                )
                for neighbor, normal, area, centroid in records
            )
            for records in faces_per_cell
        )
        self._ghost_p = tuple(pressure.clone() for pressure in ghost_p)
        self._n = n_real
        self._coordinate_columns = ("x", "y") if dimension == 2 else ("x", "y", "z")
        minimum = self._x.amin(dim=0)
        self._origin_m = tuple(float(value.detach()) for value in minimum)

    @property
    def x(self) -> torch.Tensor:
        """Storage-independent coordinate snapshot."""

        return self._x.clone()

    def _coordinates_view(self) -> torch.Tensor:
        return self._x

    @property
    def K(self) -> torch.Tensor:
        """Storage-independent permeability snapshot."""

        return self._K.clone()

    def _permeability_view(self) -> torch.Tensor:
        return self._K

    @property
    def ghost_p(self) -> tuple[torch.Tensor, ...]:
        return tuple(pressure.clone() for pressure in self._ghost_p)

    def _ghost_pressures_view(self) -> tuple[torch.Tensor, ...]:
        return self._ghost_p

    @property
    def n(self) -> int:
        return self._n

    @property
    def coordinate_columns(self) -> tuple[str, ...]:
        return self._coordinate_columns

    @property
    def z_positive_down(self) -> bool:
        return True

    @property
    def origin_m(self) -> tuple[float, ...]:
        return self._origin_m

    @property
    def dtype(self) -> torch.dtype:
        return self._x.dtype

    @property
    def device(self) -> torch.device:
        return self._x.device

    def faces_of_cell(self, c: int) -> tuple[FaceRecord, ...]:
        return tuple(
            (
                neighbor,
                normal.clone(),
                area.clone() if isinstance(area, torch.Tensor) else area,
                centroid.clone(),
            )
            for neighbor, normal, area, centroid in self._fpc[c]
        )

    def _faces_of_cell_view(self, c: int) -> tuple[FaceRecord, ...]:
        return self._fpc[c]

    def _face_records_view(self) -> tuple[tuple[FaceRecord, ...], ...]:
        return self._fpc


def _build_nfvm(
    geom: NFVMGeometry, scheme: str
) -> tuple[
    list[tuple[LinearDiscretization, LinearDiscretization]],
    list[LinearDiscretization],
]:
    """Per interior face → ``(L_disc, R_disc)``; per boundary face → one-sided
    ``L_disc``. Falls back to a two-point flux when a non-negative basis is missing."""
    n = geom.n
    coordinates = geom._coordinates_view()
    permeability = geom._permeability_view()
    interior: list[tuple[LinearDiscretization, LinearDiscretization]] = []
    boundary: list[LinearDiscretization] = []
    seen: set[tuple[int, int]] = set()
    for c in range(n):
        fc = geom._faces_of_cell_view(c)
        for other, no, A, xf in fc:
            key = (min(c, other), max(c, other)) if other < n else (c, other)
            if key in seen:
                continue
            seen.add(key)
            dL = decompose_half_face(c, (other, no, A, xf), fc, permeability, coordinates)
            if other < n:  # interior face
                fo = geom._faces_of_cell_view(other)
                tgt_o = next(t for t in fo if t[0] == c)
                dR = decompose_half_face(other, tgt_o, fo, permeability, coordinates)
                if dL is None or dR is None:
                    dist = float(torch.linalg.vector_norm(coordinates[c] - coordinates[other]))
                    T = A * (no @ (permeability[c] @ no)) / dist
                    # The raw half-face form is −T·(f[left]−f[right]);
                    # :func:`nfvm_flux` normalises it to canonical left-to-right
                    # +T·(f[left]−f[right]).
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
            else:  # boundary (Dirichlet ghost)
                if dL is None:
                    dist = float(torch.linalg.vector_norm(coordinates[c] - xf))
                    T = A * (no @ (permeability[c] @ no)) / dist
                    L_disc = LinearDiscretization(
                        left=c, right=other, t_l=-T, t_r=T, mpfa=[]
                    )
                else:
                    L_disc = linear_discretization(dL, c, other)
                boundary.append(L_disc)
    return interior, boundary


def solve_nfvm(
    geom: NFVMGeometry,
    *,
    scheme: str = "ntpfa",
    tol: float = 1e-9,
    max_iter: int = 60,
) -> torch.Tensor:
    """Steady single-phase NFVM Dirichlet BVP on a :class:`NFVMGeometry` (any grid /
    dimension). Returns the ``(n_real,)`` cell pressures, differentiable in
    ``perm`` / boundary data. avgMPFA-initialised, damped Newton on the monotone
    nonlinear two-point flux."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise FlowContractError(
            "max_iter must be a positive integer",
            object_name="solve_nfvm",
            field="max_iter",
            expected="positive integer",
            actual=max_iter,
        )
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not math.isfinite(float(tol)) or float(tol) <= 0:
        raise FlowContractError(
            "tol must be positive and finite",
            object_name="solve_nfvm",
            field="tol",
            expected="finite value > 0",
            actual=tol,
        )
    interior, boundary = _build_nfvm(geom, scheme)
    ghost_p = geom._ghost_pressures_view()
    ghost = torch.stack(ghost_p) if ghost_p else None
    p_const = float(ghost.mean()) if ghost is not None else 0.0

    def residual(pc: torch.Tensor, mu_half: bool = False) -> torch.Tensor:
        p = torch.cat([pc, ghost]) if ghost is not None else pc
        R = pc.new_zeros(geom.n)
        for L_disc, R_disc in interior:
            F = nfvm_flux(p, L_disc, R_disc, scheme, mu_half=mu_half)
            face_cells = torch.tensor(
                [[L_disc["left"], L_disc["right"]]], dtype=torch.long, device=F.device
            )
            R = R + scatter_internal_face_flux(F.reshape(1), face_cells, geom.n)
        for L_disc in boundary:
            q = -_onesided(p, L_disc)[0]
            cells = torch.tensor([L_disc["left"]], dtype=torch.long, device=q.device)
            R = R + scatter_boundary_outflow(q.reshape(1), cells, geom.n)
        return R

    p = geom._permeability_view().new_full((geom.n,), p_const)
    r0 = residual(p, mu_half=True)
    if float(torch.linalg.vector_norm(r0.detach())) != 0.0:
        J0 = torch.autograd.functional.jacobian(
            lambda q: residual(q, mu_half=True),
            p,
            vectorize=True,
        )
        p = p + DirectSolver().solve(J0, r0)
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
                object_name="solve_nfvm",
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
        for _ls in range(25):
            trial = residual(p + alpha * update)
            if float(torch.linalg.vector_norm(trial.detach(), ord=float("inf"))) < rn:
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
                object_name="solve_nfvm",
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
        object_name="solve_nfvm",
        field="convergence",
        expected=f"residual norm < {tol}",
        actual=record.to_dict(),
        diagnostics=record,
    )


def _oriented_normal(
    x_cell: torch.Tensor, x_face: torch.Tensor, normal: torch.Tensor
) -> torch.Tensor:
    """Normal oriented to point *out* of the cell (``(x_face − x_cell)·n > 0``)."""
    return normal if float((x_face - x_cell) @ normal) >= 0 else -normal


def _extend_to_ghosts(vals: torch.Tensor, geom: NFVMGeometry) -> torch.Tensor:
    """Broadcast a per-real-cell field ``vals`` ``(n_real,)`` to the full ``(n_total,)``
    cell+ghost layout a conduction :class:`NFVMGeometry` needs, each Dirichlet ghost
    copies its adjacent real cell's value (matching the perm-ghost convention)."""
    n_total = geom._coordinates_view().shape[0]
    if n_total == geom.n:
        return vals
    ext = vals.new_empty(n_total)
    ext[: geom.n] = vals
    for c in range(geom.n):
        for face in geom._faces_of_cell_view(c):
            if face[0] >= geom.n:  # face = (neighbor, n_out, area, x_face)
                ext[face[0]] = vals[c]
    return ext
