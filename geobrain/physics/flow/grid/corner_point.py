"""Corner-point grid geometry and non-neighbour connectivity.

Consumes canonical-SI ``dims``/``COORD``/``ZCORN`` records after the explicit
GRDECL adapter has assigned and converted source units, then turns them into the
node + hexahedral-cell mesh the MPFA / flow machinery consumes:

  * ``SPECGRID`` / ``DIMENS``: grid dimensions ``(NX, NY, NZ)``.
  * ``COORD``: ``6·(NX+1)·(NY+1)`` values: per pillar a top point ``(x,y,z)`` and
    a bottom point, defining the (generally slanted) coordinate line.
  * ``ZCORN``: ``8·NX·NY·NZ`` corner depths on the doubled ``(2NX, 2NY, 2NZ)``
    lattice; the depth of cell ``(i,j,k)``'s corner ``(a,b,c)``
    (``a,b,c ∈ {0,1}``) is ``ZCORN[(2i+a) + (2j+b)·2NX + (2k+c)·4·NX·NY]``
    (the standard corner-indexing convention).
  * ``ACTNUM``: optional active-cell flags.

Each cell corner's ``(x,y)`` is interpolated along its pillar at the ZCORN depth;
coincident corners are merged so a conforming (non-faulted) grid yields a shared-
node hex mesh suitable for :class:`~geobrain.physics.flow.discretization.mpfa3d.MPFAGrid3D`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TypeAlias

import torch

from ....core import GeoBrainError
from ..discretization.transmissibility import expand_perm, full_tensor_face_transmissibility

# GRDECL corner (a,b,c) for each VTK hexahedron node (a=x-,b=y-,c=z-side).
_VTK_TO_ABC = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]

Point2D: TypeAlias = tuple[float, float]
Pillar: TypeAlias = tuple[int, int]
DepthQuad: TypeAlias = tuple[float, float, float, float]


def _pillar_xy_at_depth(
    coord: Sequence[float], pi: int, pj: int, nx: int, z: float
) -> Point2D:
    """Interpolate ``(x, y)`` along pillar ``(pi, pj)`` at depth ``z``."""
    base = 6 * (pj * (nx + 1) + pi)
    xt, yt, zt, xb, yb, zb = coord[base : base + 6]
    dz = zb - zt
    if abs(dz) < 1e-300:
        return xt, yt
    t = (z - zt) / dz
    return xt + t * (xb - xt), yt + t * (yb - yt)


def corner_point_to_hex(
    dims: tuple[int, int, int],
    coord: Sequence[float],
    zcorn: Sequence[float],
    *,
    tol: float = 1e-6,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, list[list[int]]]:
    """Corner-point ``(dims, COORD, ZCORN)`` → ``(nodes (n,3), cell_nodes)``.

    Cell corners are interpolated along pillars and coincident corners merged
    (rounding to ``tol``), giving a conforming hex mesh (VTK node order per cell).
    """
    nx, ny, nz = dims
    if len(coord) != 6 * (nx + 1) * (ny + 1):
        raise GeoBrainError(
            "COORD length must be 6·(NX+1)·(NY+1)",
            object_name="corner_point_to_hex",
            field="COORD",
            expected=6 * (nx + 1) * (ny + 1),
            actual=len(coord),
        )
    if len(zcorn) != 8 * nx * ny * nz:
        raise GeoBrainError(
            "ZCORN length must be 8·NX·NY·NZ",
            object_name="corner_point_to_hex",
            field="ZCORN",
            expected=8 * nx * ny * nz,
            actual=len(zcorn),
        )

    def zc(i: int, j: int, k: int, a: int, b: int, c: int) -> float:
        ii, jj, kk = 2 * i + a, 2 * j + b, 2 * k + c
        return zcorn[ii + jj * (2 * nx) + kk * (4 * nx * ny)]

    node_of: dict[tuple[int, int, int], int] = {}
    nodes: list[tuple[float, float, float]] = []
    cell_nodes: list[list[int]] = []

    def node_id(x: float, y: float, z: float) -> int:
        key = (round(x / tol), round(y / tol), round(z / tol))
        idx = node_of.get(key)
        if idx is None:
            idx = len(nodes)
            node_of[key] = idx
            nodes.append((x, y, z))
        return idx

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cn: list[int] = []
                for a, b, c in _VTK_TO_ABC:
                    z = zc(i, j, k, a, b, c)
                    x, y = _pillar_xy_at_depth(coord, i + a, j + b, nx, z)
                    cn.append(node_id(x, y, z))
                cell_nodes.append(cn)

    return torch.tensor(nodes, dtype=dtype), cell_nodes


def _polygon_signed_area_2d(poly: Sequence[Point2D]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def _clip_convex_2d(subject: Sequence[Point2D], clip: Sequence[Point2D]) -> list[Point2D]:
    """Sutherland-Hodgman clip of ``subject`` against the convex CCW ``clip``
    polygon (lists of ``(s, z)`` points). Returns the intersection polygon."""

    def inside(p: Point2D, a: Point2D, b: Point2D) -> bool:  # left of directed edge a→b
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-12

    def intersect(p: Point2D, q: Point2D, a: Point2D, b: Point2D) -> Point2D:
        """Return the intersection of segment ``p-q`` and line ``a-b``."""
        x1, y1 = p
        x2, y2 = q
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-300:
            return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    out = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        inp, out = out, []
        if not inp:
            break
        prv = inp[-1]
        for cur in inp:
            if inside(cur, a, b):
                if not inside(prv, a, b):
                    out.append(intersect(prv, cur, a, b))
                out.append(cur)
            elif inside(prv, a, b):
                out.append(intersect(prv, cur, a, b))
            prv = cur
    return out


@dataclass
class CornerPointNNC:
    """Non-neighbour connections across faults in a corner-point grid.

    ``pairs`` ``(M, 2)`` cell-index pairs (``(k·NY+j)·NX+i`` ordering, matching
    :func:`corner_point_to_hex`), ``trans`` ``(M,)`` connection transmissibilities
    [SI], ``areas`` ``(M,)`` juxtaposition (overlap) areas, and the synthesized
    fault-face ``centroids``/``normals`` ``(M, 3)``. ``trans`` is differentiable
    in the permeability passed to :func:`compute_fault_nnc`.
    """

    pairs: torch.Tensor
    trans: torch.Tensor
    areas: torch.Tensor
    centroids: torch.Tensor
    normals: torch.Tensor


def compute_fault_nnc(
    dims: tuple[int, int, int],
    coord: Sequence[float],
    zcorn: Sequence[float],
    perm: torch.Tensor,
    *,
    actnum: Sequence[int] | None = None,
    tol: float = 1e-6,
    min_overlap: float = 1e-9,
    dtype: torch.dtype = torch.float64,
) -> CornerPointNNC:
    """Fault non-neighbour connections (NNC) for a (faulted) corner-point grid.

    Across a fault two laterally-adjacent columns are offset in depth, so their
    cell faces do not match and the conforming hex mesh (shared-node) carries no
    connection there. This finds, for every I- and J-direction column interface,
    the cell pairs whose faces *juxtapose* (their depth ranges overlap) but are
    **not** conforming, builds the overlap quad on the fault plane, and computes
    the connection transmissibility as the half-transmissibility harmonic average
    (the industry-standard form) over the **overlap area**,
    :func:`~geobrain.physics.flow.discretization.transmissibility.full_tensor_face_transmissibility`.

    A throw smaller than a cell links each cell to two neighbours across the
    fault (areas summing to the full face); a throw exceeding the column height
    juxtaposes nothing (a sealing fault). Conforming interfaces yield no NNC (the
    mesh already connects them).

    Args:
        dims, coord, zcorn: corner-point deck (see :func:`corner_point_to_hex`).
        perm: ``(n_cells, 3, 3)`` tensors or ``(n_cells, m)`` entries
            (``m∈{1,3,6}``, expanded via
            :func:`~geobrain.physics.flow.discretization.transmissibility.expand_perm`).
        actnum: optional active-cell flags; inactive cells are skipped.
    """
    nx, ny, nz = dims
    nodes, cell_nodes = corner_point_to_hex(dims, coord, zcorn, tol=tol, dtype=dtype)
    cc = torch.stack([nodes[cn].mean(0) for cn in cell_nodes])  # (n_cells, 3)
    n_cells = nx * ny * nz
    if perm.shape[-2:] != (3, 3):
        perm = expand_perm(perm.reshape(n_cells, -1), 3)
    coord = [float(x) for x in coord]
    zcorn = [float(x) for x in zcorn]

    def cidx(i: int, j: int, k: int) -> int:
        return (k * ny + j) * nx + i

    def zc(i: int, j: int, k: int, a: int, b: int, c: int) -> float:
        return zcorn[(2 * i + a) + (2 * j + b) * (2 * nx) + (2 * k + c) * (4 * nx * ny)]

    def active(c: int) -> bool:
        return actnum is None or int(actnum[c]) != 0

    def map3d(pP: Pillar, pQ: Pillar, s: float, z: float) -> list[float]:
        """Map ``(s, z)`` on the fault to a three-dimensional point."""
        xa, ya = _pillar_xy_at_depth(coord, pP[0], pP[1], nx, z)
        xb, yb = _pillar_xy_at_depth(coord, pQ[0], pQ[1], nx, z)
        return [(1.0 - s) * xa + s * xb, (1.0 - s) * ya + s * yb, z]

    fc_l: list[torch.Tensor] = []
    fn_l: list[torch.Tensor] = []
    fa_l: list[torch.Tensor] = []
    pair_l: list[list[int]] = []

    def emit(
        cA: int,
        cB: int,
        pP: Pillar,
        pQ: Pillar,
        Lz: DepthQuad,
        Rz: DepthQuad,
    ) -> None:
        """Exact overlap of two fault faces by polygon clipping in ``(s, z)``.

        ``Lz``/``Rz`` are ``(top@s0, bot@s0, top@s1, bot@s1)`` depths of the left
        and right faces on the shared pillars ``pP`` (s=0) and ``pQ`` (s=1)."""
        lt0, lb0, lt1, lb1 = Lz
        rt0, rb0, rt1, rb1 = Rz
        if max(lb0, lb1) <= min(rt0, rt1) or max(rb0, rb1) <= min(lt0, lt1):
            return  # depth ranges disjoint
        if (
            abs(lt0 - rt0) < tol
            and abs(lb0 - rb0) < tol
            and abs(lt1 - rt1) < tol
            and abs(lb1 - rb1) < tol
        ):
            return  # conforming face ⇒ mesh handles it
        subj = [(0.0, lt0), (1.0, lt1), (1.0, lb1), (0.0, lb0)]
        clip = [(0.0, rt0), (1.0, rt1), (1.0, rb1), (0.0, rb0)]
        if _polygon_signed_area_2d(subj) < 0:
            subj = subj[::-1]
        if _polygon_signed_area_2d(clip) < 0:
            clip = clip[::-1]
        poly = _clip_convex_2d(subj, clip)
        if len(poly) < 3:
            return
        pts = [torch.tensor(map3d(pP, pQ, s, z), dtype=dtype) for s, z in poly]
        c0 = torch.stack(pts).mean(0)
        m = len(pts)
        area = c0.new_zeros(())
        nrm = c0.new_zeros(3)
        cen = c0.new_zeros(3)
        for t in range(m):
            a = pts[t] - c0
            b = pts[(t + 1) % m] - c0
            crx = torch.linalg.cross(a, b)
            ta = 0.5 * crx.norm()
            area = area + ta
            nrm = nrm + crx
            cen = cen + ta * ((c0 + pts[t] + pts[(t + 1) % m]) / 3.0)
        if float(area) < min_overlap:
            return
        fc_l.append(cen / area.clamp_min(1e-300))
        fn_l.append(nrm / nrm.norm().clamp_min(1e-300))
        fa_l.append(area)
        pair_l.append([cA, cB])

    # I-direction interfaces: left cell x⁺ (a=1) vs right cell x⁻ (a=0); pillars (i+1,j),(i+1,j+1)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx - 1):
                cA = cidx(i, j, k)
                if not active(cA):
                    continue
                Lz = (
                    zc(i, j, k, 1, 0, 0),
                    zc(i, j, k, 1, 0, 1),
                    zc(i, j, k, 1, 1, 0),
                    zc(i, j, k, 1, 1, 1),
                )
                for kR in range(nz):
                    cB = cidx(i + 1, j, kR)
                    if not active(cB):
                        continue
                    Rz = (
                        zc(i + 1, j, kR, 0, 0, 0),
                        zc(i + 1, j, kR, 0, 0, 1),
                        zc(i + 1, j, kR, 0, 1, 0),
                        zc(i + 1, j, kR, 0, 1, 1),
                    )
                    emit(cA, cB, (i + 1, j), (i + 1, j + 1), Lz, Rz)

    # J-direction interfaces: left cell y⁺ (b=1) vs right cell y⁻ (b=0); pillars (i,j+1),(i+1,j+1)
    for k in range(nz):
        for j in range(ny - 1):
            for i in range(nx):
                cA = cidx(i, j, k)
                if not active(cA):
                    continue
                Lz = (
                    zc(i, j, k, 0, 1, 0),
                    zc(i, j, k, 0, 1, 1),
                    zc(i, j, k, 1, 1, 0),
                    zc(i, j, k, 1, 1, 1),
                )
                for kR in range(nz):
                    cB = cidx(i, j + 1, kR)
                    if not active(cB):
                        continue
                    Rz = (
                        zc(i, j + 1, kR, 0, 0, 0),
                        zc(i, j + 1, kR, 0, 0, 1),
                        zc(i, j + 1, kR, 1, 0, 0),
                        zc(i, j + 1, kR, 1, 0, 1),
                    )
                    emit(cA, cB, (i, j + 1), (i + 1, j + 1), Lz, Rz)

    if not pair_l:
        z = torch.zeros
        return CornerPointNNC(
            z(0, 2, dtype=torch.long),
            z(0, dtype=dtype),
            z(0, dtype=dtype),
            z(0, 3, dtype=dtype),
            z(0, 3, dtype=dtype),
        )
    pairs = torch.tensor(pair_l, dtype=torch.long)
    fc, fn, fa = torch.stack(fc_l), torch.stack(fn_l), torch.stack(fa_l)
    trans = full_tensor_face_transmissibility(cc, fc, fn, fa, perm, pairs)
    return CornerPointNNC(pairs, trans, fa, fc, fn)


__all__ = [
    "corner_point_to_hex",
    "CornerPointNNC",
    "compute_fault_nnc",
]
