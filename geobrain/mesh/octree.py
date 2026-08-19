"""OctreeMesh: flat-leaf-list quadtree (2-D) / octree (3-D) with adaptive refinement.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Callable, ClassVar

import torch

from ..core.errors import GeoBrainError
from .base import Mesh
from .capabilities import (
    BoundaryRecords, ConnectivityMesh, FaceRecords, GeometryMesh,
    PrismGeometryMesh,
)
from .tensor import TensorMesh

# Public aliases for ``OctreeMesh.from_tensor_mesh``'s two predicate forms.
# The forms are EXPLICIT keywords (no probe/auto-detection: a scalar predicate
# of the shape ``bool(...) and level < N`` evaluated on batch tensors can
# yield a correctly-shaped bool tensor and be silently misread, so the
# caller says which form it is):
#   refine_fn:         scalar, ``(centre (n_dim,), level: int) -> bool``
#                      (the historical per-cell form; Python-rate).
#   refine_batch_fn:   vectorized, ``(centers (n, n_dim), levels (n,)) ->
#                      (n,) bool tensor`` (the preferred form).
_RefinePredicate = Callable[[torch.Tensor, int], bool]
_BatchRefinePredicate = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _child_sign_grid(n_dim: int) -> torch.Tensor:
    """Sign-offset grid for octree refinement children.

    For ``n_dim = 2`` returns the 4 sign tuples ``(±1, ±1)``;
    for ``n_dim = 3`` returns the 8 sign tuples ``(±1, ±1, ±1)``.
    Cached on the type since it never depends on data.
    """
    if n_dim == 2:
        return torch.tensor(
            [[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=torch.float64,
        )
    if n_dim == 3:
        return torch.tensor(
            [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)],
            dtype=torch.float64,
        )
    raise GeoBrainError(
        "child sign grid only defined for n_dim in {2, 3}",
        object_name="OctreeMesh", field="n_dim",
        expected="2 or 3", actual=n_dim,
    )


class OctreeMesh(Mesh):
    """
    Flat-leaf-list quadtree (2-D) / octree (3-D) mesh.

    Internally stores leaves only: no parent/child pointers. Each leaf cell
    carries its centre, per-axis half-width, and an integer refinement level
    (0 = root, +1 per subdivision). A field on the octree is a 1-D tensor of
    length :attr:`n_cells` indexed by leaf-cell ID.

    Construction is normally through :meth:`from_tensor_mesh`, which starts from a
    uniform :class:`TensorMesh` and applies one or more refinement passes flagged
    by a boolean mask (or a callable predicate).

    **Finite-volume caveats.**

    - *Fine-coarse TPFA is first-order-inconsistent.* At every fine-coarse
      (hanging) interface the face centroid handed out by :meth:`face_neighbors`
      is offset from the straight line joining the two cell centres, so a
      two-point flux across it is *skewed*, geometrically inconsistent by
      ``O(1)`` (a measurement showed ~1/3 of the face flux error for an exact
      linear field). For accuracy either (a) route through the MPFA path
      (:mod:`geobrain.physics.em.numerics.finite_volume.poisson_octree_mpfa`),
      (b) use a 2:1-:meth:`balance`\\ d or otherwise conforming mesh, or
      (c) apply a deferred skewness correction using the offset exposed by
      :meth:`face_skewness` (face centroid minus the cell-centre midpoint).

    - *Partially-covered coarse faces close.* A coarse leaf face only partially
      covered by finer neighbours (a stepped domain / a >1-level jump) emits its
      uncovered remainder as a boundary record, so the discrete divergence
      ``sum_faces area * n_outward`` closes on every leaf (see
      :meth:`boundary_faces`).
    """

    # ClassVar is correct HERE (contrast TensorMesh's plain annotation):
    # octree capabilities never refine per instance: every OctreeMesh
    # carries exactly this set.
    mesh_capabilities: ClassVar[frozenset[type]] = frozenset({
        GeometryMesh, ConnectivityMesh, PrismGeometryMesh,
    })

    def __init__(
        self,
        centers: torch.Tensor,
        half_widths: torch.Tensor,
        *,
        n_dim: int | None = None,
        levels: torch.Tensor | None = None,
    ) -> None:
        """
        Build an octree directly from leaf geometry.

        Args:
            centers: ``(n_cells, n_dim)`` leaf-centre coordinates.
            half_widths: ``(n_cells, n_dim)`` per-axis half-widths.
            n_dim: Spatial dimension, 2 or 3. ``None`` (default) infers it
                from ``centers.shape[1]``; passing it explicitly acts as a
                cross-check against the geometry (a required keyword would
                restate what the tensor already carries).
            levels: Optional ``(n_cells,)`` refinement levels (default all 0).

        Raises:
            GeoBrainError: On bad ``n_dim``, shape mismatch, or non-positive
                half-widths.
        """
        if n_dim is None:
            if centers.ndim != 2:
                raise GeoBrainError(
                    "centers must be (n_cells, n_dim)",
                    object_name="OctreeMesh", field="centers",
                    expected="2-D tensor", actual=tuple(centers.shape),
                )
            n_dim = int(centers.shape[1])
        if n_dim not in (2, 3):
            raise GeoBrainError(
                "OctreeMesh supports only 2D and 3D",
                object_name="OctreeMesh", field="n_dim",
                expected="2 or 3", actual=n_dim,
            )
        if centers.ndim != 2 or centers.shape[1] != n_dim:
            raise GeoBrainError(
                "centers must be (n_cells, n_dim)",
                object_name="OctreeMesh", field="centers",
                expected=f"(n_cells, {n_dim})",
                actual=tuple(centers.shape),
            )
        if half_widths.shape != centers.shape:
            raise GeoBrainError(
                "half_widths must match centers shape",
                object_name="OctreeMesh", field="half_widths",
                expected=tuple(centers.shape),
                actual=tuple(half_widths.shape),
            )
        if (half_widths <= 0).any():
            raise GeoBrainError(
                "half_widths must all be positive",
                object_name="OctreeMesh", field="half_widths",
                expected="> 0", actual="<= 0 entries present",
            )
        n_cells = centers.shape[0]
        if levels is None:
            levels = torch.zeros(n_cells, dtype=torch.long)
        if levels.shape != (n_cells,):
            raise GeoBrainError(
                "levels must be (n_cells,)",
                object_name="OctreeMesh", field="levels",
                expected=(n_cells,), actual=tuple(levels.shape),
            )
        # Detach to avoid carrying graph state on geometry metadata, and
        # canonicalize geometry to CPU (device is a per-field/consumer concern,
        # matching uniform TensorMesh): a projection built from CUDA-resident
        # meshes then lands its precomputed geometry on a consistent device.
        self._centers = centers.detach().cpu().to(dtype=torch.float64)
        self._half_widths = half_widths.detach().cpu().to(dtype=torch.float64)
        self._levels = levels.detach().cpu().to(dtype=torch.long)
        self._n_dim = int(n_dim)
        self._n_cells = int(n_cells)

        # Lazy geometry/topology caches. An OctreeMesh is immutable after
        # construction (refine() returns a NEW mesh), so these are loop-invariant
        # across an inversion. Caching face_neighbors/boundary_faces is the big
        # win: they are O(n_cells²) plane-matching loops that were otherwise
        # rebuilt every forward. Compute once, hand out a fresh clone each call.
        self._cv_cache: torch.Tensor | None = None
        self._cb_cache: torch.Tensor | None = None
        self._fn_cache: FaceRecords | None = None
        self._bf_cache: BoundaryRecords | None = None
        # Per-axis matched internal-face cell pairs (numpy int64 (i, j) arrays,
        # already in the legacy i-outer/j-inner row order). Shared by
        # face_neighbors() and boundary_faces() so the O(n log n) face-plane
        # bucketing runs once per mesh.
        self._axis_pairs_cache: list[tuple] | None = None

    # ------------------------------------------------------------------

    def min_cell_widths(self) -> tuple[float, ...]:
        """Per-axis smallest leaf width (= the finest refinement level's size).

        Lossy for a refined octree (a single scalar per axis cannot describe a
        mixed-resolution mesh), so this is for reporting/diagnostics only, never
        treat it as a grid spacing. For per-cell geometry use
        :meth:`cell_centers` / :attr:`half_widths`.
        """
        finest = self._half_widths.min(dim=0).values * 2.0
        return tuple(float(v) for v in finest.tolist())

    @property
    def n_dim(self) -> int:                          # type: ignore[override]
        return self._n_dim

    def cell_centers(self) -> torch.Tensor:          # type: ignore[override]
        """``(n_cells, n_dim)`` per-leaf centre coordinates."""
        return self._centers.clone()

    @property
    def half_widths(self) -> torch.Tensor:
        """``(n_cells, n_dim)`` per-cell half-widths."""
        return self._half_widths.clone()

    @property
    def levels(self) -> torch.Tensor:
        """``(n_cells,)`` integer refinement levels (0 = unrefined)."""
        return self._levels.clone()

    def cell_widths(self) -> torch.Tensor:
        """Full per-axis cell widths ``(n_cells, n_dim)``: ``2 * half_widths``.

        NAME-SHARING NOTE: the three meshes expose ``cell_widths`` in three
        deliberately different shapes, each native to its geometry,
        TensorMesh: a PROPERTY of per-axis 1-D width vectors;
        OctreeMesh (this): a METHOD returning per-cell ``(n_cells, n_dim)``
        widths (AMR cells have no shared per-axis vector);
        CylindricalMesh: a property ``(wz, wr)`` pair. Generic code should
        branch on capability, not probe the attribute shape.
        """
        return 2.0 * self.half_widths

    def cell_volumes(self) -> torch.Tensor:
        """``(n_cells,)`` cell volumes (area in 2-D, volume in 3-D)."""
        if self._cv_cache is None:
            self._cv_cache = torch.prod(self._half_widths * 2.0, dim=1)
        return self._cv_cache.clone()

    def cell_bounds(self) -> torch.Tensor:
        """``(n_cells, 2*n_dim)`` axis-aligned cell extents ``[lo, hi]`` per axis.

        Layout ``[lo_0, hi_0, lo_1, hi_1, ...]`` (one lo/hi pair per axis),
        cell order = leaf order (matches :meth:`cell_centers`). General over
        refinement levels: ``lo = centre - half_width``, ``hi = centre +
        half_width`` per cell, per axis.
        """
        if self._cb_cache is None:
            lo = self._centers - self._half_widths    # (n_cells, n_dim)
            hi = self._centers + self._half_widths
            cols: list[torch.Tensor] = []
            for axis in range(self._n_dim):
                cols.append(lo[:, axis])
                cols.append(hi[:, axis])
            self._cb_cache = torch.stack(cols, dim=-1)
        return self._cb_cache.clone()

    def _internal_face_pairs(self) -> list[tuple]:
        """Per-axis matched internal-face cell pairs, in legacy row order.

        Returns one ``(ii, jj)`` tuple of numpy ``int64`` arrays per axis, where
        ``ii`` is the cell on the ``-axis`` side of the shared face (its ``+``
        face lies on the plane) and ``jj`` the cell on the ``+axis`` side (its
        ``-`` face lies on the plane). Within each axis the pairs are sorted
        lexicographically by ``(ii, jj)``, byte-for-byte the same order the
        legacy ``for i: for j in sorted candidates`` double loop emitted.

        Algorithm (O(n log n), pure numpy): for each axis the two face planes of
        every leaf (``lo``/``hi`` on that axis) are clustered by coordinate, a
        sorted sweep grouping values within ``eps``, so all faces on one physical
        plane share a bucket id regardless of which leaf produced them. A
        ``+`` face and a ``-`` face can only be a neighbour pair if they land in
        the same bucket, so matching happens WITHIN a bucket and the cross-section
        overlap test is a single broadcast, replacing the legacy O(n_cells²)
        per-cell plane scan. Buckets are visited in ascending-plane order and the
        emitted pairs re-sorted per axis, so the record order is reproducible and
        identical to the legacy function.
        """
        if self._axis_pairs_cache is not None:
            return self._axis_pairs_cache
        import numpy as np

        c_np = self._centers.detach().cpu().numpy()
        h_np = self._half_widths.detach().cpu().numpy()
        lo = c_np - h_np
        hi = c_np + h_np
        n_dim = self._n_dim
        n_cells = c_np.shape[0]
        # Same tolerance the legacy scan used for face-plane coincidence and for
        # the cross-section overlap rejection.
        eps = 1e-9 * float(h_np.min() + 1e-300)

        axis_pairs: list[tuple] = []
        for axis in range(n_dim):
            other_axes = [a for a in range(n_dim) if a != axis]
            if n_cells == 0:
                axis_pairs.append(
                    (np.empty(0, np.int64), np.empty(0, np.int64))
                )
                continue
            # --- cluster the 2*n face planes on this axis into bucket ids ----
            # A leaf contributes its lo-face (index i in [0, n)) and its hi-face
            # (index i+n). Clustering lo and hi TOGETHER makes a hi-face and a
            # lo-face at the same coordinate share a bucket, which is exactly the
            # legacy ``|lo[j,axis] - hi[i,axis]| < eps`` neighbour condition.
            planes = np.concatenate([lo[:, axis], hi[:, axis]])
            order = np.argsort(planes, kind="stable")
            sp = planes[order]
            # New bucket whenever the gap to the previous sorted plane exceeds
            # eps (dyadic octree planes are either identical or a finest-cell
            # width apart, so this reproduces the legacy match exactly).
            brk = np.empty(sp.shape[0], dtype=np.int64)
            brk[0] = 0
            if sp.shape[0] > 1:
                brk[1:] = (np.diff(sp) > eps).astype(np.int64)
            bucket_sorted = np.cumsum(brk)
            bucket = np.empty(sp.shape[0], dtype=np.int64)
            bucket[order] = bucket_sorted
            key_lo = bucket[:n_cells]     # bucket of each leaf's -face
            key_hi = bucket[n_cells:]     # bucket of each leaf's +face

            # --- group leaves by their +face bucket and by their -face bucket --
            def _group(keys: "np.ndarray") -> dict:
                o = np.argsort(keys, kind="stable")
                sk = keys[o]
                uniq, first = np.unique(sk, return_index=True)
                edges = np.append(first, sk.shape[0])
                return {
                    int(uniq[g]): o[edges[g]:edges[g + 1]]
                    for g in range(uniq.shape[0])
                }

            hi_groups = _group(key_hi)    # bucket -> cells whose +face is there
            lo_groups = _group(key_lo)    # bucket -> cells whose -face is there

            ii_parts: list[np.ndarray] = []
            jj_parts: list[np.ndarray] = []
            for p, high_cells in hi_groups.items():
                low_cells = lo_groups.get(p)
                if low_cells is None:
                    continue
                # Broadcast cross-section overlap on the non-normal axes. Two
                # leaves are neighbours iff the intersection width is > eps on
                # every other axis (product of those widths is the face area).
                valid = np.ones((high_cells.shape[0], low_cells.shape[0]), dtype=bool)
                for a in other_axes:
                    o_lo = np.maximum(
                        lo[high_cells, a][:, None], lo[low_cells, a][None, :]
                    )
                    o_hi = np.minimum(
                        hi[high_cells, a][:, None], hi[low_cells, a][None, :]
                    )
                    valid &= (o_hi - o_lo) > eps
                if not valid.any():
                    continue
                ri, cj = np.nonzero(valid)
                ii_parts.append(high_cells[ri])
                jj_parts.append(low_cells[cj])

            if ii_parts:
                ii = np.concatenate(ii_parts)
                jj = np.concatenate(jj_parts)
                # Legacy emission order: cell-i ascending (outer loop), then
                # cell-j ascending within each i. A lexsort on (jj, ii): primary
                # ii, secondary jj: reproduces it exactly.
                srt = np.lexsort((jj, ii))
                ii = np.ascontiguousarray(ii[srt])
                jj = np.ascontiguousarray(jj[srt])
            else:
                ii = np.empty(0, np.int64)
                jj = np.empty(0, np.int64)
            axis_pairs.append((ii, jj))

        self._axis_pairs_cache = axis_pairs
        return axis_pairs

    def face_neighbors(self) -> FaceRecords:
        """Internal-face cell-pair records as a :class:`FaceRecords`.

        Promotes the face-discovery logic from
        the retired legacy per-pair scan into the mesh-agnostic
        :class:`FaceRecords` struct. The legacy full centre-to-centre ``dist``
        is split into per-side half distances ``dist_l``/``dist_r`` whose sum
        recovers ``dist``, so assemblers using ``T = σ·area/dist`` are
        behaviour-preserving.

        Iteration order mirrors the legacy function exactly (axis-outer,
        cell-i inner) so that record index ``k`` corresponds 1-to-1 with
        the legacy scan's record ``k`` (the scan itself is retired; its
        arithmetic survives as the byte-level oracle in the octree test
        suite, and
        ``poisson_octree._find_octree_face_neighbors`` is now a thin adapter
        over these records).

        Cached on the (immutable) mesh; returns a fresh clone each call. The
        heavy lifting is the O(n log n) :meth:`_internal_face_pairs` face-plane
        bucketing (was an O(n_cells²) plane scan rebuilt every forward).

        .. warning::
           **Fine-coarse TPFA is first-order-inconsistent.** For a hanging
           (fine-coarse) face the ``centroid`` returned here is cell ``i``'s
           centre projected onto the shared plane, which does NOT lie on the
           line joining the two cell centres, a two-point flux across it is
           *skewed* (geometrically inconsistent by ``O(1)``; ~1/3 of the face
           flux error for an exact linear field). Conforming faces are exact.
           For a consistent discretisation steer to the MPFA path
           (:mod:`~geobrain.physics.em.numerics.finite_volume.poisson_octree_mpfa`),
           to a 2:1-:meth:`balance`\\ d / conforming mesh, or apply a deferred
           skewness correction from :meth:`face_skewness` (the face-centroid vs
           cell-centre-midpoint offset). This method deliberately does NOT add a
           skewness field to :class:`FaceRecords` (a serialized, physics-consumed
           schema); the correction is left to the downstream assembler.
        """
        if self._fn_cache is not None:
            return self._fn_cache.clone()

        centers = self._centers          # CPU float64
        half = self._half_widths
        n_dim = self._n_dim
        axis_pairs = self._internal_face_pairs()

        cell_i_parts: list[torch.Tensor] = []
        cell_j_parts: list[torch.Tensor] = []
        area_parts: list[torch.Tensor] = []
        dl_parts: list[torch.Tensor] = []
        dr_parts: list[torch.Tensor] = []
        normal_parts: list[torch.Tensor] = []
        centroid_parts: list[torch.Tensor] = []

        lo = centers - half
        hi = centers + half
        for axis in range(n_dim):
            ii_np, jj_np = axis_pairs[axis]
            nf = int(ii_np.shape[0])
            ii = torch.from_numpy(ii_np).to(torch.long)
            jj = torch.from_numpy(jj_np).to(torch.long)
            # Face area = product of the per-axis overlap widths on every axis
            # OTHER than the face axis (edge length in 2-D, rectangle in 3-D).
            area = torch.ones(nf, dtype=torch.float64)
            for a in range(n_dim):
                if a == axis:
                    continue
                o_lo = torch.maximum(lo[ii, a], lo[jj, a])
                o_hi = torch.minimum(hi[ii, a], hi[jj, a])
                area = area * (o_hi - o_lo)
            # Per-side half distances: dl = cell i's half-width on the face axis
            # (its centre-to-face distance); dr = cell j's centre to the plane.
            # Their sum equals the legacy full centre-to-centre dist on axis.
            dl = half[ii, axis].clone()
            dr = torch.abs(centers[jj, axis] - hi[ii, axis])
            normal = torch.zeros(nf, n_dim, dtype=torch.float64)
            normal[:, axis] = 1.0
            # Face centroid: cell i's centre shifted to its +axis face plane.
            centroid = centers[ii].clone()
            centroid[:, axis] = centers[ii, axis] + half[ii, axis]

            cell_i_parts.append(ii)
            cell_j_parts.append(jj)
            area_parts.append(area)
            dl_parts.append(dl)
            dr_parts.append(dr)
            normal_parts.append(normal)
            centroid_parts.append(centroid)

        self._fn_cache = FaceRecords(
            cell_i=torch.cat(cell_i_parts) if cell_i_parts else torch.empty(0, dtype=torch.long),
            cell_j=torch.cat(cell_j_parts) if cell_j_parts else torch.empty(0, dtype=torch.long),
            area=torch.cat(area_parts) if area_parts else torch.empty(0, dtype=torch.float64),
            dist_l=torch.cat(dl_parts) if dl_parts else torch.empty(0, dtype=torch.float64),
            dist_r=torch.cat(dr_parts) if dr_parts else torch.empty(0, dtype=torch.float64),
            normal=torch.cat(normal_parts) if normal_parts else torch.empty(0, n_dim, dtype=torch.float64),
            centroid=torch.cat(centroid_parts) if centroid_parts else torch.empty(0, n_dim, dtype=torch.float64),
        )
        return self._fn_cache.clone()

    def boundary_faces(self) -> BoundaryRecords:
        """Outer (domain-boundary) leaf faces as a :class:`BoundaryRecords`.

        For each leaf and each of its ``2·n_dim`` axis-aligned faces, the face
        is a domain boundary iff no other leaf abuts it (no neighbour shares
        that face plane with overlapping cross-section). Detection mirrors the
        plane-matching in :meth:`face_neighbors` (an internal face has a
        neighbour whose opposite plane coincides within ``eps`` and whose
        cross-section overlaps); the complement is the boundary set.

        The outward normal points away from the owning leaf centre (``±axis``),
        matching :meth:`TensorMesh.boundary_faces` /
        :meth:`UnstructuredMesh.boundary_faces`. The face centroid is the leaf
        centre projected onto the face plane.

        A *fully-unshared* leaf face emits one record carrying the FULL leaf face
        area (a coarse leaf adjacent to finer leaves on the *interior* keeps that
        face interior). A *partially-covered* coarse face, one that finer
        neighbours cover over only part of its area, leaving an uncovered strip
        (a stepped domain or a >1-level jump), additionally emits the positive
        **uncovered remainder** as a boundary record (area = full face area minus
        the summed internal-overlap area, centroid approximated by the leaf-centre
        projection). Without it that strip would be neither an internal face nor a
        boundary face and the discrete divergence ``sum_faces area * n_outward``
        on that leaf would not close (residual ~1.0). Fully-covered and
        fully-uncovered faces are byte-identical to the pre-remainder behaviour;
        the remainder records are appended after them and are empty for any mesh
        that tiles a box without partial coverage.

        Cached on the (immutable) mesh; returns a fresh clone each call. A leaf
        face is a full boundary iff it is NOT one of the internal faces discovered
        by :meth:`_internal_face_pairs` (a leaf's ``+`` face is internal iff the
        leaf appears as ``cell_i``; its ``-`` face iff it appears as ``cell_j``),
        so the O(n_cells²) plane scan is replaced by set complement over the
        O(n log n) bucketing result.
        """
        if self._bf_cache is not None:
            return self._bf_cache.clone()
        import numpy as np

        centers = self._centers          # CPU float64
        half = self._half_widths
        n_dim = self._n_dim
        n_cells = self._n_cells
        axis_pairs = self._internal_face_pairs()
        lo = centers - half
        hi = centers + half

        cell_parts: list[torch.Tensor] = []
        area_parts: list[torch.Tensor] = []
        normal_parts: list[torch.Tensor] = []
        centroid_parts: list[torch.Tensor] = []
        # Partial-coverage remainder records, appended AFTER the full-face
        # records so a non-partial mesh (empty remainder) stays byte-identical.
        rem_cell_parts: list[torch.Tensor] = []
        rem_area_parts: list[torch.Tensor] = []
        rem_normal_parts: list[torch.Tensor] = []
        rem_centroid_parts: list[torch.Tensor] = []

        for axis in range(n_dim):
            ii_np, jj_np = axis_pairs[axis]
            # A leaf's +axis face is internal iff it is an i (lo-side owner) of
            # some internal face; its -axis face is internal iff it is a j.
            has_plus = np.zeros(n_cells, dtype=bool)
            has_minus = np.zeros(n_cells, dtype=bool)
            has_plus[ii_np] = True
            has_minus[jj_np] = True
            minus_cells = np.nonzero(~has_minus)[0]     # -side (lo) boundary
            plus_cells = np.nonzero(~has_plus)[0]       # +side (hi) boundary
            # Legacy emission order: for each cell i ascending, its lo face
            # (-side) before its hi face (+side). lexsort with cells primary and
            # a side rank (0 for lo, 1 for hi) secondary reproduces it exactly.
            cells = np.concatenate([minus_cells, plus_cells])
            sides = np.concatenate([
                np.full(minus_cells.shape[0], -1.0),
                np.full(plus_cells.shape[0], 1.0),
            ])
            rank = np.concatenate([
                np.zeros(minus_cells.shape[0], dtype=np.int64),
                np.ones(plus_cells.shape[0], dtype=np.int64),
            ])
            srt = np.lexsort((rank, cells))
            cells_t = torch.from_numpy(np.ascontiguousarray(cells[srt])).to(torch.long)
            sides_t = torch.from_numpy(np.ascontiguousarray(sides[srt])).to(torch.float64)
            nb = int(cells_t.shape[0])

            # Boundary face area = FULL leaf cross-section (product of the full
            # widths, 2*half, on every axis other than the face axis).
            area = torch.ones(nb, dtype=torch.float64)
            for a in range(n_dim):
                if a == axis:
                    continue
                area = area * (2.0 * half[cells_t, a])
            normal = torch.zeros(nb, n_dim, dtype=torch.float64)
            normal[:, axis] = sides_t
            centroid = centers[cells_t].clone()
            centroid[:, axis] = centers[cells_t, axis] + sides_t * half[cells_t, axis]

            cell_parts.append(cells_t)
            area_parts.append(area)
            normal_parts.append(normal)
            centroid_parts.append(centroid)

            # --- partial-coverage remainder on this axis ---------------------
            # A leaf whose +face (or -face) HAS an internal neighbour but whose
            # neighbours cover less than its full face area has an uncovered
            # strip. Emit that positive remainder as a boundary record so the
            # per-cell divergence closes. Covered area per side = sum of the
            # internal-overlap areas the leaf participates in (as cell_i on its
            # +face, as cell_j on its -face). Full-face leaves fall out with a
            # zero remainder, so tiled meshes emit nothing here.
            ii_t = torch.from_numpy(ii_np).to(torch.long)
            jj_t = torch.from_numpy(jj_np).to(torch.long)
            pair_area = torch.ones(ii_t.shape[0], dtype=torch.float64)
            for a in range(n_dim):
                if a == axis:
                    continue
                o_lo = torch.maximum(lo[ii_t, a], lo[jj_t, a])
                o_hi = torch.minimum(hi[ii_t, a], hi[jj_t, a])
                pair_area = pair_area * (o_hi - o_lo)
            covered_plus = torch.zeros(n_cells, dtype=torch.float64)
            covered_minus = torch.zeros(n_cells, dtype=torch.float64)
            covered_plus.index_add_(0, ii_t, pair_area)   # cell_i's +face
            covered_minus.index_add_(0, jj_t, pair_area)  # cell_j's -face
            full_face = torch.ones(n_cells, dtype=torch.float64)
            for a in range(n_dim):
                if a == axis:
                    continue
                full_face = full_face * (2.0 * half[:, a])
            has_plus_t = torch.from_numpy(has_plus)
            has_minus_t = torch.from_numpy(has_minus)
            # Relative tolerance: a genuine strip is an O(1) fraction of the
            # coarse face area; float noise on a fully-tiled face is ~1e-15.
            thr = full_face * 1e-9
            for side, has_side, covered in (
                (1.0, has_plus_t, covered_plus),
                (-1.0, has_minus_t, covered_minus),
            ):
                remainder = full_face - covered
                sel = torch.from_numpy(np.asarray(has_side)) & (remainder > thr)
                idx = sel.nonzero().flatten()
                if idx.numel() == 0:
                    continue
                rc = idx.to(torch.long)
                rem_cell_parts.append(rc)
                rem_area_parts.append(remainder[rc])
                rn = torch.zeros(rc.shape[0], n_dim, dtype=torch.float64)
                rn[:, axis] = side
                rem_normal_parts.append(rn)
                rcen = centers[rc].clone()
                rcen[:, axis] = centers[rc, axis] + side * half[rc, axis]
                rem_centroid_parts.append(rcen)

        all_cell = cell_parts + rem_cell_parts
        all_area = area_parts + rem_area_parts
        all_normal = normal_parts + rem_normal_parts
        all_centroid = centroid_parts + rem_centroid_parts
        self._bf_cache = BoundaryRecords(
            cell=torch.cat(all_cell) if all_cell else torch.empty(0, dtype=torch.long),
            area=torch.cat(all_area) if all_area else torch.empty(0, dtype=torch.float64),
            normal=torch.cat(all_normal) if all_normal else torch.empty(0, n_dim, dtype=torch.float64),
            centroid=torch.cat(all_centroid) if all_centroid else torch.empty(0, n_dim, dtype=torch.float64),
        )
        return self._bf_cache.clone()

    def face_skewness(self) -> torch.Tensor:
        """Per-internal-face TPFA skewness offset ``(nf, n_dim)``, from the records.

        For each internal face this is the face centroid minus the midpoint of
        the two cell centres::

            skew_k = centroid_k - 0.5 * (center[cell_i_k] + center[cell_j_k])

        A two-point flux assumes the face centroid lies on the line joining the
        two cell centres; ``skew`` is exactly how far it does not. It is the
        **zero vector on a conforming face** (equal-size neighbours, the centroid
        is the centre midpoint) and **non-zero at a fine-coarse (hanging) face**,
        where the coarse cell's centre is laterally offset from the small shared
        face. A downstream assembler can apply a deferred skewness correction with
        this offset *without any* :class:`FaceRecords` *schema change*; it is
        derived purely from the already-published ``centroid`` and
        :meth:`cell_centers` (see the :meth:`face_neighbors` warning).

        Returns a fresh CPU float64 tensor (not cached; it is a cheap gather over
        the cached :meth:`face_neighbors`).
        """
        fr = self.face_neighbors()
        c = self._centers
        midpoint = 0.5 * (c[fr.cell_i] + c[fr.cell_j])
        return fr.centroid - midpoint

    def append_connections(self, extra: FaceRecords) -> "Mesh":
        """Return a connectivity-equivalent mesh with ``extra`` faces appended (NNC).

        Part of the :class:`ConnectivityMesh` contract. The result is an
        :class:`~geobrain.mesh.unstructured.UnstructuredMesh` built
        directly from this mesh's own SoA (``cell_centers``/``cell_volumes``/
        ``face_neighbors() + extra``/``boundary_faces()``): once non-neighbour
        connections (NNC) are appended the octree leaf invariants
        (``half_widths``/``levels``-derived adjacency) no longer describe the
        connectivity, so the returned mesh carries connectivity capabilities
        only (no :class:`PrismGeometryMesh`, no octree refinement API). The
        return semantics match :meth:`UnstructuredMesh.append_connections`, a
        new mesh whose ``face_neighbors()`` is the original face SoA with
        ``extra`` concatenated and whose boundary faces are unchanged.
        """
        from .unstructured import UnstructuredMesh

        return UnstructuredMesh.from_mesh(self, extra_faces=extra)

    # ------------------------------------------------------------------

    @classmethod
    def from_tensor_mesh(
        cls,
        tm: "TensorMesh",
        refine_mask: torch.Tensor | None = None,
        *,
        refine_fn: "_RefinePredicate | None" = None,
        refine_batch_fn: "_BatchRefinePredicate | None" = None,
        max_level: int = 4,
    ) -> "OctreeMesh":
        """Build an octree from a uniform TM base.

        Four modes:

        - ``refine_mask = None``, ``refine_fn = None``: no refinement;
          returns a flat copy of ``tm``'s cells (every cell at level 0).
        - ``refine_mask`` given: one-pass splitting; cells flagged
          ``True`` are split into ``2 ** n_dim`` children. Equivalent to
          ``cls.from_tensor_mesh(tm).refine(refine_mask)``.
        - ``refine_fn`` given: **recursive splitting**, scalar form:
          the callable receives ``(centre: Tensor, level: int)`` per cell
          and returns a bool (the historical API; Python-rate).
        - ``refine_batch_fn`` given: **recursive splitting**, vectorized
          form (preferred): the callable receives
          ``(centers (n, n_dim), levels (n,))`` once per pass and returns
          a ``(n,)`` bool tensor. The two predicate keywords are mutually
          exclusive and deliberately explicit, no probe/auto-detection,
          because a scalar predicate shaped ``bool(...) and level < N``
          evaluated on batch tensors can return a plausible mask and be
          silently misread.

          Either way the octree is refined repeatedly until no leaf is
          flagged or every flagged leaf has reached ``max_level``; cells
          already at ``max_level`` are never consulted.

        ``refine_mask``, ``refine_fn`` and ``refine_batch_fn`` are mutually
        exclusive; at most one may be given.
        """
        if not isinstance(tm, TensorMesh):
            raise GeoBrainError(
                "from_tensor_mesh expects a TensorMesh",
                object_name="OctreeMesh.from_tensor_mesh", field="tm",
                expected="TensorMesh", actual=type(tm).__name__,
            )
        # The octree leaf geometry is built from a single per-axis half-width
        # derived from ``tm.spacing``: valid only for a uniform base grid. A
        # non-uniform TensorMesh would silently get wrong leaf widths, so reject
        # it up front via the authoritative ``tm.is_uniform`` flag.
        if not tm.is_uniform:
            raise GeoBrainError(
                "OctreeMesh.from_tensor_mesh requires a uniform TensorMesh; "
                "got non-uniform cell widths",
                object_name="OctreeMesh.from_tensor_mesh", field="tm",
                expected="uniform spacing", actual="non-uniform cell_widths",
            )
        n_predicates = sum(
            x is not None for x in (refine_mask, refine_fn, refine_batch_fn)
        )
        if n_predicates > 1:
            raise GeoBrainError(
                "pass at most one of refine_mask, refine_fn or refine_batch_fn",
                object_name="OctreeMesh.from_tensor_mesh",
                field="refine_mask/refine_fn/refine_batch_fn",
                expected="at most one refinement driver", actual="several",
            )
        if max_level < 0:
            raise GeoBrainError(
                "max_level must be >= 0",
                object_name="OctreeMesh.from_tensor_mesh", field="max_level",
                expected=">= 0", actual=max_level,
            )

        centres = tm.cell_centers()
        n_dim = tm.n_dim
        n_base = centres.shape[0]
        half_widths = torch.tensor(tm.spacing, dtype=torch.float64).unsqueeze(0) / 2.0
        half_widths = half_widths.expand(n_base, n_dim).contiguous()
        levels = torch.zeros(n_base, dtype=torch.long)
        om = cls(centres, half_widths, n_dim=n_dim, levels=levels)

        if refine_mask is None and refine_fn is None and refine_batch_fn is None:
            return om

        if refine_mask is not None:
            mask_flat = refine_mask.reshape(-1).to(dtype=torch.bool)
            if mask_flat.shape[0] != n_base:
                raise GeoBrainError(
                    "refine_mask must match tm.shape",
                    object_name="OctreeMesh.from_tensor_mesh",
                    field="refine_mask",
                    expected=tm.shape, actual=tuple(refine_mask.shape),
                )
            return om.refine(mask_flat)

        # Recursive refinement until either fixed-point or max_level cap.
        # The predicate form is EXPLICIT (scalar refine_fn vs vectorized
        # refine_batch_fn): see the keyword docs above for why there is
        # deliberately no auto-detection. Cells already at max_level are
        # never consulted in either form.
        for _ in range(max_level + 1):
            below = om._levels < max_level
            if not bool(below.any()):
                break
            if refine_batch_fn is not None:
                out = refine_batch_fn(om._centers, om._levels)
                if not (isinstance(out, torch.Tensor)
                        and out.dtype == torch.bool
                        and tuple(out.shape) == (om.n_cells,)):
                    raise GeoBrainError(
                        "refine_batch_fn must return a (n_cells,) bool tensor",
                        object_name="OctreeMesh.from_tensor_mesh",
                        field="refine_batch_fn",
                        expected=f"BoolTensor of shape ({om.n_cells},)",
                        actual=(tuple(out.shape) if isinstance(out, torch.Tensor)
                                else type(out).__name__),
                    )
                mask = out & below
            else:
                mask = torch.tensor(
                    [
                        bool(below[i]) and bool(
                            refine_fn(om._centers[i], int(om._levels[i]))   # type: ignore[misc]
                        )
                        for i in range(om.n_cells)
                    ],
                    dtype=torch.bool,
                )
            if not bool(mask.any()):
                break
            om = om.refine(mask)
        return om

    # ------------------------------------------------------------------

    def refine(self, mask: torch.Tensor) -> "OctreeMesh":
        """
        Return a new :class:`OctreeMesh` with flagged leaves split into
        ``2 ** n_dim`` children.

        Args:
            mask: boolean tensor of shape ``(n_cells,)``. ``True`` cells
                are split; ``False`` cells are kept verbatim. The output
                cell order is "kept cells (original order) then split
                children (parent order × child sub-order)".
        """
        mask = mask.reshape(-1).to(dtype=torch.bool)
        if mask.shape[0] != self.n_cells:
            raise GeoBrainError(
                "mask must match n_cells",
                object_name="OctreeMesh.refine", field="mask",
                expected=(self.n_cells,), actual=tuple(mask.shape),
            )
        if not bool(mask.any()):
            return OctreeMesh(
                self._centers, self._half_widths,
                n_dim=self._n_dim, levels=self._levels,
            )

        keep_mask = ~mask
        kept_centres = self._centers[keep_mask]
        kept_half = self._half_widths[keep_mask]
        kept_levels = self._levels[keep_mask]

        split_centres = self._centers[mask]
        split_half = self._half_widths[mask]
        split_levels = self._levels[mask]
        child_half = split_half / 2.0

        sign_grid = _child_sign_grid(self._n_dim)
        new_centres = (
            split_centres.unsqueeze(1)
            + sign_grid.unsqueeze(0) * child_half.unsqueeze(1)
        ).reshape(-1, self._n_dim)
        new_half = child_half.repeat_interleave(sign_grid.shape[0], dim=0)
        new_levels = (split_levels + 1).repeat_interleave(sign_grid.shape[0])

        all_centres = torch.cat([kept_centres, new_centres], dim=0)
        all_half = torch.cat([kept_half, new_half], dim=0)
        all_levels = torch.cat([kept_levels, new_levels])
        return OctreeMesh(
            all_centres, all_half, n_dim=self._n_dim, levels=all_levels,
        )

    # ------------------------------------------------------------------
    # Geometric refinement vocabulary (the conventional refine_points/box/ball
    # equivalents): thin loops over the vectorized ``refine(mask)``.
    # Coordinates are in MESH-AXIS order (z, x[, y]), like cell_centers().
    # ------------------------------------------------------------------

    def _refine_where(self, hit, level: int) -> "OctreeMesh":
        """Split every leaf for which ``hit(om)`` is True until it reaches
        ``level`` (``hit``: OctreeMesh -> (n_cells,) bool)."""
        if level < 0:
            raise GeoBrainError(
                "level must be >= 0",
                object_name="OctreeMesh.refine_points/refine_box/refine_ball",
                field="level",
                expected=">= 0", actual=level,
            )
        om = self
        for _ in range(level + 1):
            mask = hit(om) & (om._levels < level)
            if not bool(mask.any()):
                break
            om = om.refine(mask)
        return om

    def refine_points(self, points: torch.Tensor, level: int) -> "OctreeMesh":
        """Refine every leaf CONTAINING one of ``points`` to ``level``.

        Args:
            points: ``(m, n_dim)`` coordinates in mesh-axis order
                ``(z, x[, y])``.
            level: target refinement level for the containing leaves.
        """
        pts = torch.as_tensor(points, dtype=torch.float64).reshape(-1, self._n_dim)

        def hit(om: "OctreeMesh") -> torch.Tensor:
            d = (om._centers.unsqueeze(1) - pts.unsqueeze(0)).abs()
            inside = (d <= om._half_widths.unsqueeze(1) + 1e-12).all(dim=2)
            return inside.any(dim=1)

        return self._refine_where(hit, level)

    def refine_box(self, lo, hi, level: int) -> "OctreeMesh":
        """Refine every leaf OVERLAPPING the axis-aligned box to ``level``.

        Args:
            lo, hi: per-axis box corners in mesh-axis order ``(z, x[, y])``.
            level: target refinement level for the overlapping leaves.
        """
        lo_t = torch.as_tensor(lo, dtype=torch.float64).reshape(1, self._n_dim)
        hi_t = torch.as_tensor(hi, dtype=torch.float64).reshape(1, self._n_dim)
        box_c = 0.5 * (lo_t + hi_t)
        box_h = 0.5 * (hi_t - lo_t)

        def hit(om: "OctreeMesh") -> torch.Tensor:
            # positive-measure overlap only: leaves merely TOUCHING the box
            # boundary (zero shared volume) are not refined
            return ((om._centers - box_c).abs()
                    < om._half_widths + box_h - 1e-12).all(dim=1)

        return self._refine_where(hit, level)

    def refine_ball(self, center, radius: float, level: int) -> "OctreeMesh":
        """Refine every leaf INTERSECTING the ball to ``level``.

        Args:
            center: ball centre in mesh-axis order ``(z, x[, y])``.
            radius: ball radius (same units as the mesh coordinates).
            level: target refinement level for the intersecting leaves.
        """
        c_t = torch.as_tensor(center, dtype=torch.float64).reshape(1, self._n_dim)

        def hit(om: "OctreeMesh") -> torch.Tensor:
            # distance from ball centre to the nearest point of each leaf box
            gap = ((om._centers - c_t).abs() - om._half_widths).clamp(min=0.0)
            return gap.norm(dim=1) <= float(radius) + 1e-12

        return self._refine_where(hit, level)

    # ------------------------------------------------------------------

    def balance(self) -> "OctreeMesh":
        """Return a new **2:1-balanced** octree (opt-in; default build unchanged).

        A balanced octree is one in which no two *face-adjacent* leaves differ by
        more than one refinement level. This is the precondition the octree MPFA-O
        assembler
        (:func:`~geobrain.physics.em.numerics.finite_volume.poisson_octree_mpfa.assemble_poisson_octree_3d_mpfa`)
        requires, an unbalanced mesh (a >1-level jump across a face) makes its
        hanging-face groups ill-formed and it raises.

        Algorithm (iterative 2:1 grading): repeatedly find every face-adjacent
        leaf pair whose levels differ by more than one and refine the **coarser**
        leaf of each such pair by one level; re-evaluate and repeat until no
        violating pair remains. Each pass strictly refines at least one coarse
        leaf and no leaf is ever refined past the mesh's finest existing level, so
        the process terminates (the ripple settles in at most a few level-span
        passes). Adjacency is read from the same O(n log n)
        :meth:`_internal_face_pairs` bucketing the connectivity uses.

        Guarantees:

        - **Tiling preserved**: :meth:`refine` splits a leaf into children whose
          volumes sum to the parent, so the leaf volumes still sum to the domain.
        - **Origin preserved**: refinement subdivides existing (absolute-
          coordinate) leaves, so the mesh's min-corner is unchanged.
        - **Idempotent**: an already-balanced mesh has no violating pair, so it is
          returned as an equivalent (geometry-identical) new mesh.

        .. note::
           Only refinement is used; **coarsening is deferred** (no consumer needs
           a coarsened octree yet). Balancing is opt-in: :meth:`from_tensor_mesh` /
           :meth:`refine` never balance implicitly, so existing meshes and their
           byte-identical connectivity are untouched.
        """
        import numpy as np

        om = self
        # Safety cap: the finest level never rises under balancing, and each pass
        # advances the coarsest violating leaf toward it, so a bound proportional
        # to the level span (with slack for spatial ripple) can never be hit in
        # practice; treat exceeding it as a bug rather than looping forever.
        max_passes = int(self._levels.max()) * 4 + 16
        for _ in range(max_passes):
            axis_pairs = om._internal_face_pairs()
            lv = om._levels
            refine_mask = torch.zeros(om.n_cells, dtype=torch.bool)
            for ii_np, jj_np in axis_pairs:
                if ii_np.shape[0] == 0:
                    continue
                ii_t = torch.from_numpy(np.ascontiguousarray(ii_np)).to(torch.long)
                jj_t = torch.from_numpy(np.ascontiguousarray(jj_np)).to(torch.long)
                li = lv[ii_t]
                lj = lv[jj_t]
                # A leaf is refined when its neighbour is finer by >1 level:
                # higher level == finer == smaller cell, so the COARSER leaf is
                # the one with the LOWER level.
                refine_mask[ii_t[(lj - li) > 1]] = True   # j finer -> refine i
                refine_mask[jj_t[(li - lj) > 1]] = True   # i finer -> refine j
            if not bool(refine_mask.any()):
                break
            om = om.refine(refine_mask)
        else:  # pragma: no cover - defensive; 2:1 grading converges well within
            raise GeoBrainError(
                "balance() did not converge",
                object_name="OctreeMesh.balance", field="max_passes",
                expected=f"<= {max_passes} passes", actual="exceeded",
            )
        # Always hand back a fresh object (never ``self``) so the contract "returns
        # a NEW octree" holds even for an already-balanced input.
        if om is self:
            return OctreeMesh(
                self._centers, self._half_widths,
                n_dim=self._n_dim, levels=self._levels,
            )
        return om

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Declarative, JSON-native serialization (round-trips via ``mesh_from_dict``).

        Schema (``version`` is :data:`~geobrain.mesh.serialization.MESH_SCHEMA_VERSION`)::

            {"type": "OctreeMesh", "version": 1,
             "n_dim": 2 | 3,
             "centers": <tensor-envelope (n_cells, n_dim)>,
             "half_widths": <tensor-envelope (n_cells, n_dim)>,
             "levels": <tensor-envelope (n_cells,) int64>}

        Stores the **raw leaf SoA**, the exact fields the ``OctreeMesh``
        constructor takes, because the flat-leaf representation retains no
        provenance (no base :class:`TensorMesh` or refine history is kept on the
        instance), so a base-mesh + refine-mask "provenance" payload is simply
        not recoverable from a built octree. The raw SoA is always correct
        regardless of how the octree was constructed, and rebuilding through the
        constructor reproduces ``cell_centers`` / ``half_widths`` / ``levels``
        (hence all derived connectivity) to float64.
        """
        from .serialization import MESH_SCHEMA_VERSION, _ser_tensor

        return {
            "type": "OctreeMesh",
            "version": MESH_SCHEMA_VERSION,
            "n_dim": int(self._n_dim),
            "centers": _ser_tensor(self._centers),
            "half_widths": _ser_tensor(self._half_widths),
            "levels": _ser_tensor(self._levels),
        }

    def __repr__(self) -> str:
        return (
            f"OctreeMesh(n_cells={self.n_cells}, n_dim={self.n_dim}, "
            f"levels={int(self._levels.min())}..{int(self._levels.max())})"
        )
