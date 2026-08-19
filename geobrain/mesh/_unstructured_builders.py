"""Pure geometry kernels for :class:`UnstructuredMesh`'s convenience builders.

These four functions carry the dimension-specific geometry math that
``UnstructuredMesh.from_polygons`` / ``from_simplices`` lean on, circumcentre
solves and the face-derivation engines that turn raw connectivity into
TPFA-ready ``FaceRecords`` / ``BoundaryRecords``. They are split out of
``unstructured.py`` so that file holds the mesh *type* (the SoA invariants and
the capability surface) while the bulky, self-contained geometry lives here.

They are deliberately kept as TWO independent face-derivation engines
(``derive_faces_2d`` over polygon-loop edges, ``derive_faces_3d`` over
simplex-combination triangles). The shared dispatch skeleton is superficial;
the topology enumeration and per-face geometry genuinely differ, and these feed
TPFA transmissibility, where a clear separately-verifiable kernel per dimension
is worth more than one parametrised one.

Scaling: the 3-D tet path is fully vectorized (``derive_faces_3d``;
~200k tets in seconds) with the per-face scalar kernel retained as
``derive_faces_3d_scalar``: the byte-level oracle the test suite pins
the vectorized path against. The 2-D polygon path
(``derive_faces_2d``) deliberately stays scalar: it must handle arbitrary
variable-length polygon loops (not just triangles), and 2-D problem sizes
are orders of magnitude below the 3-D inductive-EM scale that motivated the
vectorization. Revisit only if a 2-D workload actually hits it.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import combinations

import torch

from ..core.errors import GeoBrainError
from .capabilities import BoundaryRecords, FaceRecords


def _tpfa_face_distances(
    s_i: torch.Tensor, s_j: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """TPFA cell_i / cell_j face distances ``(dist_l, dist_r)`` from the two
    SIGNED perpendicular offsets of the cell points from the face plane.

    ``s_i = (c_i - p)·n`` and ``s_j = (c_j - p)·n`` are the signed distances of
    the two cell points from the face plane (``p`` a point on the face, ``n`` the
    unit normal oriented ``cell_i → cell_j``, so ``s_j >= s_i``). The consistent
    TPFA transmissibility distance is the projected cell-point separation
    ``d = |(c_j - c_i)·n| = |s_j - s_i|`` (the gap the two-point flux differences
    across the face). We must return ``dist_l, dist_r > 0`` with
    ``dist_l + dist_r == d``. Two regimes:

    - **Straddle** (``s_i``/``s_j`` OPPOSITE signs: the self-centered /
      Delaunay-orthogonal case, and ALWAYS true for centroid cell points, which
      lie strictly inside their own cell): ``dist_l = |s_i|``, ``dist_r = |s_j|``.
      Opposite signs make ``|s_i| + |s_j| = |s_j - s_i| = d`` already, so this is
      returned UNCHANGED; the common well-behaved case is byte-identical to the
      historical unsigned ``abs`` computation.

    - **Same side** (SAME sign: non-self-centered: a circumcentre of an obtuse /
      non-Delaunay-orthogonal simplex can fall on the wrong side of a shared
      face, so both cell points sit on one side). Here the raw unsigned
      perpendicular distances do NOT sum to ``d`` (they overshoot by ``2·min|s|``),
      which silently corrupts the transmissibility. We instead RESCALE: split the
      true projected separation ``d`` in the ratio ``|s_i| : |s_j|``. This keeps
      ``dist_l + dist_r == d`` exactly (``dist_r = d - dist_l``), stays strictly
      positive, preserves the relative perpendicular weighting the
      heterogeneous-σ harmonic mean consumes, and is C0-continuous with the
      straddle branch (at ``s_i = 0`` both give ``dist_l = 0, dist_r = d``). The
      caller emits a one-time non-self-centered warning.

    Returns ``(dist_l, dist_r, straddle)``; ``straddle=False`` flags the
    rescaled non-self-centered face so the caller can warn once.
    """
    abs_i = torch.abs(s_i)
    abs_j = torch.abs(s_j)
    if float(s_i) * float(s_j) <= 0.0:
        # Opposite signs (or a cell point on the plane, caught by the degenerate
        # guard downstream): |s_i| + |s_j| == d already: leave byte-identical.
        return abs_i, abs_j, True
    # Same side: rescale so dist_l + dist_r == |(c_j - c_i)·n| exactly.
    d = torch.abs(s_j - s_i)                     # = |(c_j - c_i)·n|
    dist_l = d * (abs_i / (abs_i + abs_j))
    dist_r = d - dist_l
    return dist_l, dist_r, False


def triangle_circumcenter(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, ci: int,
) -> torch.Tensor:
    """Circumcentre of triangle (a, b, c) via the 2×2 perpendicular-bisector
    system ``2(B−A)·x = |B|²−|A|²``, ``2(C−A)·x = |C|²−|A|²``."""
    m = torch.stack([2.0 * (b - a), 2.0 * (c - a)])  # (2, 2)
    rhs = torch.stack([
        (b @ b) - (a @ a),
        (c @ c) - (a @ a),
    ])
    if float(torch.abs(torch.det(m))) <= 1e-30:
        raise GeoBrainError(
            "degenerate (collinear) triangle, no circumcentre",
            object_name="UnstructuredMesh.from_polygons", field="cells",
            expected="non-collinear triangle", actual=f"cell {ci}",
        )
    return torch.linalg.solve(m, rhs)


def tet_circumcenter(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
    ci: int,
) -> torch.Tensor:
    """Circumsphere centre of tet (a, b, c, d) via the 3×3 system
    ``2(B−A)·x = |B|²−|A|²`` (and the B→C, B→D rows)."""
    m = torch.stack([2.0 * (b - a), 2.0 * (c - a), 2.0 * (d - a)])  # (3, 3)
    rhs = torch.stack([
        (b @ b) - (a @ a),
        (c @ c) - (a @ a),
        (d @ d) - (a @ a),
    ])
    if float(torch.abs(torch.det(m))) <= 1e-30:
        raise GeoBrainError(
            "degenerate (coplanar) tetrahedron, no circumsphere centre",
            object_name="UnstructuredMesh.from_simplices", field="simplices",
            expected="non-coplanar tetrahedron", actual=f"cell {ci}",
        )
    return torch.linalg.solve(m, rhs)


def derive_faces_2d(verts, cells, centers):
    edges = defaultdict(list)
    cell_ccw: list[float] = []  # +1 if the loop winds CCW (shoelace >= 0), else -1
    for ci, loop in enumerate(cells):
        k = len(loop)
        a2 = 0.0
        for e in range(k):
            va, vb = int(loop[e]), int(loop[(e + 1) % k])
            edges[frozenset((va, vb))].append((ci, va, vb))
            pa_, pb_ = verts[va], verts[vb]
            a2 += float(pa_[0] * pb_[1] - pb_[0] * pa_[1])
        cell_ccw.append(1.0 if a2 >= 0.0 else -1.0)

    fi, fj, f_area, f_dl, f_dr, f_norm, f_cent = [], [], [], [], [], [], []
    b_cell, b_area, b_norm, b_cent = [], [], [], []
    non_self_centered = False  # set once a same-side (rescaled) face is seen

    def edge_geom(va, vb):
        pa, pb = verts[va], verts[vb]
        mid = 0.5 * (pa + pb)
        length = torch.linalg.norm(pb - pa)
        return pa, pb, mid, length

    for key, owners in edges.items():
        if len(owners) == 2:
            (ca, va, vb), (cb, _, _) = owners
            ci_, cj_ = (ca, cb) if ca < cb else (cb, ca)
            pa, pb, mid, length = edge_geom(va, vb)
            ci_c, cj_c = centers[ci_], centers[cj_]
            d = pb - pa
            nrm = torch.tensor([-d[1], d[0]], dtype=torch.float64)
            nrm = nrm / torch.linalg.norm(nrm)
            if float(torch.dot(nrm, cj_c - ci_c)) < 0:
                nrm = -nrm
            # Signed perpendicular offsets of each cell point from the face line
            # along the oriented normal. |s| reproduces the historical unsigned
            # perp_dist; the sign tells us whether the cell points straddle the
            # face (self-centered) so dist_l/dist_r stay TPFA-consistent.
            s_i = torch.dot(ci_c - pa, nrm)
            s_j = torch.dot(cj_c - pa, nrm)
            dl, dr, straddle = _tpfa_face_distances(s_i, s_j)
            if not straddle:
                non_self_centered = True
            if float(dl) <= 1e-12 or float(dr) <= 1e-12:
                raise GeoBrainError(
                    "degenerate FV geometry, a cell point lies on a face "
                    "plane (dist_l/dist_r ~ 0). This happens with "
                    "circumcentres of sliver/obtuse triangles. Supply a "
                    "Delaunay-quality mesh or use cell_center='centroid'.",
                    object_name="UnstructuredMesh.from_polygons",
                    field="dist_l/dist_r",
                    expected="> 1e-12",
                    actual=min(float(dl), float(dr)),
                )
            fi.append(ci_)
            fj.append(cj_)
            f_area.append(length)
            f_dl.append(dl)
            f_dr.append(dr)
            f_norm.append(nrm)
            f_cent.append(mid)
        elif len(owners) == 1:
            (ca, va, vb) = owners[0]
            pa, pb, mid, length = edge_geom(va, vb)
            d = pb - pa
            # Outward normal from the polygon loop winding (topological): the
            # right-perpendicular of a CCW-traversed edge points outward. This is
            # robust even when the cell point lies OUTSIDE a non-convex cell,
            # where a midpoint-minus-cell-point test would flip the normal inward.
            nrm = torch.tensor([d[1], -d[0]], dtype=torch.float64)
            if cell_ccw[ca] < 0.0:
                nrm = -nrm
            nrm = nrm / torch.linalg.norm(nrm)
            b_cell.append(ca)
            b_area.append(length)
            b_norm.append(nrm)
            b_cent.append(mid)
        else:
            raise GeoBrainError(
                "non-manifold polygon mesh edge shared by more than two cells",
                object_name="UnstructuredMesh.from_polygons", field="cells",
                expected="each edge owned by one or two cells",
                actual=f"edge {tuple(sorted(key))} has {len(owners)} owners",
            )

    if non_self_centered:
        # One-time, non-destructive: at least one internal face had both cell
        # points on the SAME side of it (a non-self-centered triangulation, e.g.
        # circumcentres of obtuse triangles). dist_l/dist_r were rescaled to the
        # projected separation so they stay TPFA-consistent, but the two-point
        # flux there is only approximate: mirror the degenerate-guard advice.
        warnings.warn(
            "non-self-centered FV geometry; an internal face has both cell "
            "points on the SAME side of it (circumcentres of non-Delaunay-"
            "orthogonal / obtuse triangles). The unsigned perpendicular "
            "distances no longer sum to the cell-point separation, so "
            "dist_l/dist_r were rescaled to split the projected separation "
            "|(c_j-c_i)·n| (dist_l+dist_r stays exact, but the TPFA two-point "
            "flux on those faces is only approximate). Supply a self-centered "
            "(Delaunay-orthogonal) mesh or use cell_center='centroid' for a "
            "consistent TPFA.",
            UserWarning,
            stacklevel=2,
        )

    faces = FaceRecords(
        cell_i=torch.tensor(fi, dtype=torch.long),
        cell_j=torch.tensor(fj, dtype=torch.long),
        area=torch.stack(f_area) if f_area else torch.empty(0, dtype=torch.float64),
        dist_l=torch.stack(f_dl) if f_dl else torch.empty(0, dtype=torch.float64),
        dist_r=torch.stack(f_dr) if f_dr else torch.empty(0, dtype=torch.float64),
        normal=torch.stack(f_norm) if f_norm else torch.empty(0, 2, dtype=torch.float64),
        centroid=torch.stack(f_cent) if f_cent else torch.empty(0, 2, dtype=torch.float64),
    )
    boundary = BoundaryRecords(
        cell=torch.tensor(b_cell, dtype=torch.long),
        area=torch.stack(b_area) if b_area else torch.empty(0, dtype=torch.float64),
        normal=torch.stack(b_norm) if b_norm else torch.empty(0, 2, dtype=torch.float64),
        centroid=torch.stack(b_cent) if b_cent else torch.empty(0, 2, dtype=torch.float64),
    )
    return faces, boundary


def derive_faces_3d_scalar(verts, simplices, centers):
    """Derive 3-D internal FaceRecords + BoundaryRecords from tets.

    The four triangular faces of tet ``(a, b, c, d)`` are
    ``(b, c, d)``, ``(a, c, d)``, ``(a, b, d)``, ``(a, b, c)``, each the
    three vertices left after dropping one. Faces are matched across tets by
    the ``frozenset`` of their vertex ids.
    """
    tri_faces: dict = defaultdict(list)
    for ci, tet in enumerate(simplices):
        tet_i = [int(v) for v in tet]
        for tri in combinations(tet_i, 3):
            tri_faces[frozenset(tri)].append((ci, tri))

    fi, fj, f_area, f_dl, f_dr, f_norm, f_cent = [], [], [], [], [], [], []
    b_cell, b_area, b_norm, b_cent = [], [], [], []
    non_self_centered = False  # set once a same-side (rescaled) face is seen

    def tri_geom(tri):
        pa, pb, pc = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        cross = torch.linalg.cross(pb - pa, pc - pa)
        nrm_len = torch.linalg.norm(cross)
        area = 0.5 * nrm_len
        unit = cross / nrm_len
        cent = (pa + pb + pc) / 3.0
        return pa, area, unit, cent

    for key, owners in tri_faces.items():
        if len(owners) == 2:
            (ca, tri_a), (cb, _tri_b) = owners
            ci_, cj_ = (ca, cb) if ca < cb else (cb, ca)
            pa, area, unit, cent = tri_geom(tri_a)
            ci_c, cj_c = centers[ci_], centers[cj_]
            # Orient the unit normal from cell_i toward cell_j.
            if float(torch.dot(unit, cj_c - ci_c)) < 0:
                unit = -unit
            # Signed perpendicular offsets of each cell point from the face plane
            # along the oriented normal. |s| reproduces the historical unsigned
            # perpendicular distance; the sign tells us whether the cell points
            # straddle the face (self-centered) so dist_l/dist_r stay
            # TPFA-consistent (see _tpfa_face_distances).
            s_i = torch.dot(ci_c - pa, unit)
            s_j = torch.dot(cj_c - pa, unit)
            dl, dr, straddle = _tpfa_face_distances(s_i, s_j)
            if not straddle:
                non_self_centered = True
            if float(dl) <= 1e-12 or float(dr) <= 1e-12:
                raise GeoBrainError(
                    "degenerate FV geometry, a cell point lies on a face "
                    "plane (dist_l/dist_r ~ 0). This happens with "
                    "circumcentres of sliver/obtuse tetrahedra. Supply a "
                    "Delaunay-quality mesh or use cell_center='centroid'.",
                    object_name="UnstructuredMesh.from_simplices",
                    field="dist_l/dist_r",
                    expected="> 1e-12",
                    actual=min(float(dl), float(dr)),
                )
            fi.append(ci_)
            fj.append(cj_)
            f_area.append(area)
            f_dl.append(dl)
            f_dr.append(dr)
            f_norm.append(unit)
            f_cent.append(cent)
        elif len(owners) == 1:
            ca, tri_a = owners[0]
            pa, area, unit, cent = tri_geom(tri_a)
            # Outward normal points away from the tet's opposite (4th) vertex
            # (topological): robust even when the cell point: e.g. an obtuse
            # tet's circumcentre: lies outside the tet, where an away-from-cell-
            # point test would flip the normal inward.
            tet_v = [int(v) for v in simplices[ca]]
            opp = next(v for v in tet_v if v not in tri_a)
            if float(torch.dot(unit, cent - verts[opp])) < 0:
                unit = -unit
            b_cell.append(ca)
            b_area.append(area)
            b_norm.append(unit)
            b_cent.append(cent)
        else:
            raise GeoBrainError(
                "non-manifold tetrahedral mesh face shared by more than two cells",
                object_name="UnstructuredMesh.from_simplices", field="simplices",
                expected="each triangular face owned by one or two cells",
                actual=f"face {tuple(sorted(key))} has {len(owners)} owners",
            )

    if non_self_centered:
        # One-time, non-destructive: at least one internal face had both cell
        # points on the SAME side of it (a non-self-centered tetrahedralisation,
        # e.g. circumcentres of obtuse tets). dist_l/dist_r were rescaled to the
        # projected separation so they stay TPFA-consistent, but the two-point
        # flux there is only approximate: mirror the degenerate-guard advice.
        warnings.warn(
            "non-self-centered FV geometry; an internal face has both cell "
            "points on the SAME side of it (circumcentres of non-Delaunay-"
            "orthogonal / obtuse tetrahedra). The unsigned perpendicular "
            "distances no longer sum to the cell-point separation, so "
            "dist_l/dist_r were rescaled to split the projected separation "
            "|(c_j-c_i)·n| (dist_l+dist_r stays exact, but the TPFA two-point "
            "flux on those faces is only approximate). Supply a self-centered "
            "(Delaunay-orthogonal) mesh or use cell_center='centroid' for a "
            "consistent TPFA.",
            UserWarning,
            stacklevel=2,
        )

    faces = FaceRecords(
        cell_i=torch.tensor(fi, dtype=torch.long),
        cell_j=torch.tensor(fj, dtype=torch.long),
        area=torch.stack(f_area) if f_area else torch.empty(0, dtype=torch.float64),
        dist_l=torch.stack(f_dl) if f_dl else torch.empty(0, dtype=torch.float64),
        dist_r=torch.stack(f_dr) if f_dr else torch.empty(0, dtype=torch.float64),
        normal=torch.stack(f_norm) if f_norm else torch.empty(0, 3, dtype=torch.float64),
        centroid=torch.stack(f_cent) if f_cent else torch.empty(0, 3, dtype=torch.float64),
    )
    boundary = BoundaryRecords(
        cell=torch.tensor(b_cell, dtype=torch.long),
        area=torch.stack(b_area) if b_area else torch.empty(0, dtype=torch.float64),
        normal=torch.stack(b_norm) if b_norm else torch.empty(0, 3, dtype=torch.float64),
        centroid=torch.stack(b_cent) if b_cent else torch.empty(0, 3, dtype=torch.float64),
    )
    return faces, boundary


# Local tet faces in ``itertools.combinations(range(4), 3)`` order, the SAME
# enumeration the scalar kernel used, so the vectorized path reproduces its
# first-encounter emission order exactly. Row k drops local vertex ``3 - k``.
_TET_FACE_COMBS = torch.tensor(
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long,
)


def derive_faces_3d(verts, simplices, centers):
    """Vectorized 3-D internal FaceRecords + BoundaryRecords from tets.

    Drop-in replacement for :func:`derive_faces_3d_scalar` (retained as the
    test oracle): same face matching (sorted vertex triples), same geometry,
    and the same **first-encounter emission order** as the historical
    frozenset-dict implementation, no Python loop over faces, so the tet
    path scales to the 1e5-1e6 cells inductive EM needs.
    """
    verts = verts.to(torch.float64)
    centers = centers.to(torch.float64)
    device = verts.device
    if isinstance(simplices, torch.Tensor):
        # Normalize the connectivity onto the vertex device (the historical
        # ``.tolist()`` path was device-agnostic; mixed-device inputs must
        # keep working).
        cn = simplices.to(dtype=torch.long, device=device)
    else:
        cn = torch.tensor([[int(v) for v in tet] for tet in simplices],
                          dtype=torch.long, device=device)
    nc = int(cn.shape[0])
    if nc == 0:
        e = torch.empty(0, dtype=torch.float64)
        e3 = torch.empty(0, 3, dtype=torch.float64)
        el = torch.empty(0, dtype=torch.long)
        return (FaceRecords(cell_i=el, cell_j=el.clone(), area=e,
                            dist_l=e.clone(), dist_r=e.clone(),
                            normal=e3, centroid=e3.clone()),
                BoundaryRecords(cell=el.clone(), area=e.clone(),
                                normal=e3.clone(), centroid=e3.clone()))

    tris = cn[:, _TET_FACE_COMBS]                    # (nc, 4, 3) enum order
    flat = tris.reshape(-1, 3)                       # flat idx f = ci*4 + k
    keys = flat.sort(dim=1).values                   # canonical triples
    uniq, inv, counts = torch.unique(
        keys, dim=0, return_inverse=True, return_counts=True)
    n_grp = int(uniq.shape[0])
    if bool((counts > 2).any()):
        g_bad = int((counts > 2).nonzero()[0, 0])
        raise GeoBrainError(
            "non-manifold tetrahedral mesh face shared by more than two cells",
            object_name="UnstructuredMesh.from_simplices", field="simplices",
            expected="each triangular face owned by one or two cells",
            actual=(f"face {tuple(int(v) for v in uniq[g_bad])} has "
                    f"{int(counts[g_bad])} owners"),
        )

    n_flat = int(flat.shape[0])
    flat_idx = torch.arange(n_flat, dtype=torch.long, device=device)
    first = torch.full((n_grp,), n_flat, dtype=torch.long,
                       device=device).scatter_reduce(
        0, inv, flat_idx, reduce="amin", include_self=True)
    last = torch.full((n_grp,), -1, dtype=torch.long,
                      device=device).scatter_reduce(
        0, inv, flat_idx, reduce="amax", include_self=True)
    order = torch.argsort(first)                     # first-encounter order

    def _tri_geom(fa):
        """(area, unit, cent, pa) of the FIRST owner's triangle, enum order."""
        t = flat[fa]                                 # (g, 3)
        pa = verts[t[:, 0]]
        pb = verts[t[:, 1]]
        pc = verts[t[:, 2]]
        cross = torch.linalg.cross(pb - pa, pc - pa)
        nrm_len = torch.linalg.norm(cross, dim=1)
        area = 0.5 * nrm_len
        unit = cross / nrm_len.unsqueeze(1)
        cent = (pa + pb + pc) / 3.0
        return area, unit, cent, pa

    # ---- internal faces (two owners) ----
    ig = order[counts[order] == 2]
    if int(ig.numel()):
        fa = first[ig]
        fb = last[ig]
        ci_ = fa // 4                                # fa < fb  ⇒  ci_ <= cj_
        cj_ = fb // 4
        area, unit, cent, pa = _tri_geom(fa)
        ci_c = centers[ci_]
        cj_c = centers[cj_]
        flip = ((unit * (cj_c - ci_c)).sum(dim=1) < 0)
        unit = torch.where(flip.unsqueeze(1), -unit, unit)
        s_i = ((ci_c - pa) * unit).sum(dim=1)
        s_j = ((cj_c - pa) * unit).sum(dim=1)
        abs_i = torch.abs(s_i)
        abs_j = torch.abs(s_j)
        straddle = (s_i * s_j) <= 0.0
        d = torch.abs(s_j - s_i)
        dl = torch.where(straddle, abs_i, d * (abs_i / (abs_i + abs_j)))
        dr = torch.where(straddle, abs_j, d - dl)
        bad = (dl <= 1e-12) | (dr <= 1e-12)
        if bool(bad.any()):
            b0 = int(bad.nonzero()[0, 0])
            raise GeoBrainError(
                "degenerate FV geometry, a cell point lies on a face "
                "plane (dist_l/dist_r ~ 0). This happens with "
                "circumcentres of sliver/obtuse tetrahedra. Supply a "
                "Delaunay-quality mesh or use cell_center='centroid'.",
                object_name="UnstructuredMesh.from_simplices",
                field="dist_l/dist_r",
                expected="> 1e-12",
                actual=min(float(dl[b0]), float(dr[b0])),
            )
        if bool((~straddle).any()):
            warnings.warn(
                "non-self-centered FV geometry; an internal face has both cell "
                "points on the SAME side of it (circumcentres of non-Delaunay-"
                "orthogonal / obtuse tetrahedra). The unsigned perpendicular "
                "distances no longer sum to the cell-point separation, so "
                "dist_l/dist_r were rescaled to split the projected separation "
                "|(c_j-c_i)·n| (dist_l+dist_r stays exact, but the TPFA two-point "
                "flux on those faces is only approximate). Supply a self-centered "
                "(Delaunay-orthogonal) mesh or use cell_center='centroid' for a "
                "consistent TPFA.",
                UserWarning,
                stacklevel=2,
            )
        faces = FaceRecords(
            cell_i=ci_, cell_j=cj_, area=area, dist_l=dl, dist_r=dr,
            normal=unit, centroid=cent,
        )
    else:
        e = torch.empty(0, dtype=torch.float64)
        el = torch.empty(0, dtype=torch.long)
        faces = FaceRecords(cell_i=el, cell_j=el.clone(), area=e,
                            dist_l=e.clone(), dist_r=e.clone(),
                            normal=torch.empty(0, 3, dtype=torch.float64),
                            centroid=torch.empty(0, 3, dtype=torch.float64))

    # ---- boundary faces (one owner) ----
    bg = order[counts[order] == 1]
    if int(bg.numel()):
        fa = first[bg]
        ca = fa // 4
        k = fa % 4
        area, unit, cent, _pa = _tri_geom(fa)
        # combination row k drops local vertex 3-k: the tet's opposite vertex
        opp = cn[ca, 3 - k]
        flip = ((unit * (cent - verts[opp])).sum(dim=1) < 0)
        unit = torch.where(flip.unsqueeze(1), -unit, unit)
        boundary = BoundaryRecords(cell=ca, area=area, normal=unit,
                                   centroid=cent)
    else:
        boundary = BoundaryRecords(
            cell=torch.empty(0, dtype=torch.long),
            area=torch.empty(0, dtype=torch.float64),
            normal=torch.empty(0, 3, dtype=torch.float64),
            centroid=torch.empty(0, 3, dtype=torch.float64),
        )
    return faces, boundary
