"""
Geometry / weight builders behind :class:`MeshProjection`'s kernels.

Module-level pure functions: given source/target mesh geometry they build the
precomputed grids, index maps and sparse weight matrices the per-mode kernels
consume; the operator module holds no geometry code of its own.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import itertools
import math

import torch

from ...core.errors import GeoBrainError
from ..octree import OctreeMesh
from ..tensor import TensorMesh


def _build_general_weights(
    src_centers: torch.Tensor,
    tgt_centers: torch.Tensor,
    *,
    method: str,
    k_neighbors: int,
    block_size: int | None = None,
    idw_power: float = 1.0,
) -> torch.Tensor:
    """Sparse (n_target, n_source) COO weight matrix from cell-centre geometry.

    ``nearest`` -> k=1 selection; ``idw`` -> k-NN inverse-distance (rows sum
    to 1). Built once and cached; values are constant geometry (no autograd).

    The pairwise distances are CHUNKED over target cells (default block
    :data:`_OM_TO_TM_TARGET_BLOCK`: the same memory fix as
    :func:`_build_om_to_tm_index_map`) so the dense ``(n_t, n_s)`` cdist is
    never materialised: peak memory is ``block × n_s`` regardless of ``n_t``.
    ``torch.topk`` is row-independent, so chunking cannot change the selection.

    Byte-identity contract: when ONE block covers every target
    (``n_t <= block``, every existing pinned mesh), the single chunk is the
    verbatim historical ``torch.cdist(tgt, src)`` call, bit-for-bit. When
    several blocks are needed, each block pins
    ``compute_mode='donot_use_mm_for_euclid_dist'``: cdist's direct kernel is
    exactly row-independent, so the result is invariant to the block size
    (the default mm kernel is NOT, BLAS reassociates per matrix shape at
    ~1e-13, so no chunked evaluation could reproduce a dense mm call
    bit-for-bit). The two regimes are the same interpolant up to that float
    reassociation noise. ``block_size`` overrides the default for tests only.
    """
    k_neighbors = _validate_k_neighbors(k_neighbors)
    src = src_centers.to(torch.float64)
    tgt = tgt_centers.to(torch.float64)
    # Build every index/weight tensor on the geometry device so the assembled
    # sparse W is single-device (was hard-coded to CPU, which crashed on a
    # CUDA-resident mesh pair). Mesh geometry is canonicalized to CPU at
    # construction, so this is CPU in practice; the registered-buffer ``.to()``
    # then moves W with the module exactly like the other precomputed kernels.
    device = src.device
    n_s = int(src.shape[0])
    n_t = int(tgt.shape[0])
    k = 1 if method == "nearest" else min(k_neighbors, n_s)
    block = int(block_size) if block_size else _OM_TO_TM_TARGET_BLOCK
    idx_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    for b0 in range(0, n_t, block):
        b1 = min(b0 + block, n_t)
        if n_t <= block:
            # Single chunk == the exact historical dense call (default mode).
            dist = torch.cdist(tgt, src)                       # (n_t, n_s)
        else:
            dist = torch.cdist(
                tgt[b0:b1], src,
                compute_mode="donot_use_mm_for_euclid_dist",
            )                                                  # (blk, n_s)
        knn = torch.topk(dist, k, dim=1, largest=False)        # idx (blk, k)
        idx_parts.append(knn.indices)
        val_parts.append(knn.values)
    idx = torch.cat(idx_parts)
    if method == "nearest":
        w = torch.ones(n_t, 1, dtype=torch.float64, device=device)
    else:
        knn_values = torch.cat(val_parts)
        # k-NN Shepard weights 1/d^p, rows normalized to sum 1. The historical
        # (and default) exponent is p=1 (NOT the classical Shepard p=2) kept
        # for backward compatibility; idw_power exposes the choice (review
        # item). Coincident points hard-select below regardless of p.
        inv = 1.0 / (knn_values + 1e-30) ** float(idw_power)   # IDW
        # exact-coincidence guard: if a neighbour is ~0 distance, hard-select it
        coincide = knn_values < 1e-12
        inv = torch.where(coincide, torch.full_like(inv, 1e30), inv)
        w = inv / inv.sum(dim=1, keepdim=True)
    rows = torch.arange(n_t, device=device).repeat_interleave(k)
    cols = idx.reshape(-1)
    vals = w.reshape(-1)
    W = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n_t, n_s), dtype=torch.float64,
    ).coalesce()
    return W


# Relative tolerances for the coverage checks: a target cell is "covered" if it
# sits inside the source domain up to this slack. ``_GRID_EPS`` is on the already
# normalised ([-1, 1]) grid_sample coordinates; ``_RANGE_EPS`` is scaled by the
# per-axis source span for the physical (tm_linear / general) checks.
_GRID_EPS = 1.0e-6
_RANGE_EPS = 1.0e-9


def _grid_uncovered(grid: torch.Tensor) -> torch.Tensor:
    """Flat ``(n_target,)`` bool mask from a normalised grid_sample grid.

    ``grid`` holds target cell centres normalised to ``[-1, 1]`` over the source
    extent (last dim = per-axis coord). A cell is uncovered iff any axis coord
    leaves ``[-1, 1]`` by more than ``_GRID_EPS``, exactly the region
    ``grid_sample(padding_mode='border')`` would silently clamp.
    """
    return (grid.detach().abs() > 1.0 + _GRID_EPS).any(dim=-1).reshape(-1)


def _general_uncovered(
    src_centers: torch.Tensor, tgt_centers: torch.Tensor,
    bounds: "tuple[torch.Tensor, torch.Tensor] | None" = None,
) -> torch.Tensor:
    """Flat ``(n_target,)`` bool mask: target centres outside the source domain.

    ``bounds`` (per-axis ``(lo, hi)`` physical domain corners from
    :func:`_source_domain_bounds`) gives the TRUE cell-hull coverage for
    box-carrying sources, without it the check falls back to the source
    CENTRE bounding box, which shrinks the declared coverage by half a
    boundary cell on every side (targets in that rim are physically inside
    the source cells but were reported uncovered).
    """
    src = src_centers.to(torch.float64)
    tgt = tgt_centers.to(torch.float64)
    if bounds is not None:
        lo, hi = (b.to(torch.float64) for b in bounds)
    else:
        lo = src.min(dim=0).values
        hi = src.max(dim=0).values
    tol = _RANGE_EPS * (hi - lo).clamp(min=1.0)
    outside = (tgt < lo - tol) | (tgt > hi + tol)
    return outside.any(dim=-1)


def _source_domain_bounds(source) -> "tuple[torch.Tensor, torch.Tensor] | None":
    """Per-axis ``(lo, hi)`` physical domain corners for box-carrying meshes.

    TensorMesh: ``origin .. origin + Σwidths``; OctreeMesh: leaf-box hull.
    ``None`` for centre-cloud sources (UnstructuredMesh), coverage then
    keeps the centre-bbox semantics (a centroid cloud has no box hull).
    """
    if isinstance(source, TensorMesh):
        lo = torch.tensor(
            [float(o) for o in source.origin], dtype=torch.float64,
        )
        hi = lo + torch.stack(
            [w.to(torch.float64).sum() for w in source.cell_widths]
        )
        return lo, hi
    if isinstance(source, OctreeMesh):
        c = source.cell_centers().to(torch.float64)
        hw = source.half_widths.to(torch.float64)
        return (c - hw).min(dim=0).values, (c + hw).max(dim=0).values
    return None


def _build_tm_linear_weights(
    src: "TensorMesh", tgt: "TensorMesh", *, method: str, padding: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact rectilinear linear (or nearest) weights for a structured TM→TM pair.

    Both meshes are axis-aligned tensor-product grids, so the interpolation
    separates per axis. Along each axis we ``searchsorted`` the target cell-centre
    coordinates into the source cell-centre coordinates
    (``cumsum(cell_widths) - 0.5*cell_widths``) to find the two bracketing source
    indices and their linear weights (``1-t`` / ``t``). The per-axis brackets are
    tensor-producted (``2**n_dim`` corners) into a sparse ``(n_target, n_source)``
    COO matrix with the SAME shape/return contract as
    :func:`_build_general_weights`, so ``_general``'s sparse-mm forward consumes
    it and autograd flows through unchanged.

    Because true linear inter/extrapolation reproduces any linear field EXACTLY
    (unlike IDW), a field ``a·z + b·x(+c·y) + d`` projects with ~1e-10 error even
    between two GRADED meshes.

    ``padding='border'`` clamps the target coordinate into the source cell-centre
    range before weighting (constant border extrapolation); ``'raise'`` / ``'zeros'``
    keep the raw (extrapolating) coordinate; for covered cells that stays exact,
    and uncovered cells are handled by the caller via the returned mask.

    Returns ``(W, uncovered)`` where ``uncovered`` is a ``(n_target,)`` bool mask
    of target cells whose centre falls outside the source DOMAIN on any axis.
    """
    n_dim = src.n_dim
    src_shape = tuple(src.shape)
    tgt_shape = tuple(tgt.shape)
    src_widths = [w.to(torch.float64) for w in src.cell_widths]
    tgt_widths = [w.to(torch.float64) for w in tgt.cell_widths]

    src_origin = src.origin
    tgt_origin = tgt.origin
    per_axis: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    uncov_axis: list[torch.Tensor] = []
    for a in range(n_dim):
        # Physical cell-centre coordinates: include each mesh's origin so the
        # searchsorted brackets the target against the source in PHYSICAL space
        # (a source/target with different origins occupy shifted regions).
        sc = torch.cumsum(src_widths[a], 0) - 0.5 * src_widths[a] + src_origin[a]
        tc = torch.cumsum(tgt_widths[a], 0) - 0.5 * tgt_widths[a] + tgt_origin[a]
        n_s = int(sc.shape[0])
        extent = float(src_widths[a].sum())                        # source width
        lo_bound = src_origin[a]                                    # source domain
        hi_bound = src_origin[a] + extent                          # [origin, origin+extent]
        tol = _RANGE_EPS * max(extent, 1.0)
        uncov_axis.append((tc < lo_bound - tol) | (tc > hi_bound + tol))

        tc_eff = torch.clamp(tc, float(sc[0]), float(sc[-1])) if padding == "border" else tc
        if n_s == 1:
            idx0 = torch.zeros_like(tc, dtype=torch.long)
            idx1 = idx0
            w0 = torch.ones_like(tc)
            w1 = torch.zeros_like(tc)
        else:
            i1 = torch.clamp(torch.searchsorted(sc, tc_eff), 1, n_s - 1)
            i0 = i1 - 1
            t = (tc_eff - sc[i0]) / (sc[i1] - sc[i0])              # unclamped => extrapolates
            if method == "nearest":
                pick1 = (t >= 0.5).to(torch.float64)
                w0, w1 = 1.0 - pick1, pick1
            else:
                w0, w1 = 1.0 - t, t
            idx0, idx1 = i0, i1
        per_axis.append((idx0, idx1, w0, w1))

    # Source flat strides (last axis fastest: matches cell_centers ordering).
    src_strides = [1] * n_dim
    for a in range(n_dim - 2, -1, -1):
        src_strides[a] = src_strides[a + 1] * src_shape[a + 1]
    n_s_total = 1
    for s in src_shape:
        n_s_total *= s

    # Per-axis target index grids (last axis fastest flatten => C-order rows).
    tgt_axis_idx = [torch.arange(tgt_shape[a]) for a in range(n_dim)]
    tgt_mesh = [g.reshape(-1) for g in torch.meshgrid(*tgt_axis_idx, indexing="ij")]
    n_t = int(tgt_mesh[0].shape[0])
    tgt_flat = torch.arange(n_t, dtype=torch.long)

    rows_list: list[torch.Tensor] = []
    cols_list: list[torch.Tensor] = []
    vals_list: list[torch.Tensor] = []
    for combo in itertools.product((0, 1), repeat=n_dim):
        col = torch.zeros(n_t, dtype=torch.long)
        val = torch.ones(n_t, dtype=torch.float64)
        for a, c in enumerate(combo):
            idx0, idx1, w0, w1 = per_axis[a]
            ia = (idx1 if c else idx0)[tgt_mesh[a]]
            wa = (w1 if c else w0)[tgt_mesh[a]]
            col = col + ia * src_strides[a]
            val = val * wa
        rows_list.append(tgt_flat)
        cols_list.append(col)
        vals_list.append(val)

    W = torch.sparse_coo_tensor(
        torch.stack([torch.cat(rows_list), torch.cat(cols_list)]),
        torch.cat(vals_list), size=(n_t, n_s_total), dtype=torch.float64,
    ).coalesce()

    uncovered = torch.zeros(n_t, dtype=torch.bool)
    for a in range(n_dim):
        uncovered = uncovered | uncov_axis[a][tgt_mesh[a]]
    return W, uncovered


# ----------------------------------------------------------------------
# Overlap-volume conservative restriction + on-demand weight materialization
# ----------------------------------------------------------------------

# Relative sliver tolerance for the overlap-volume builders: a per-axis overlap
# smaller than this fraction of the TARGET cell's own width is float noise from
# nominally-coincident cell faces (e.g. an aligned pair whose edges were built
# by two different cumsums) and is treated as zero: otherwise a shared face
# would spuriously couple a target cell to its face-neighbour source cells.
_OVERLAP_REL_TOL = 1.0e-9

# Relative tolerance for the per-axis "target coarser than source" gates.
_COARSER_REL_TOL = 1.0e-6


def _tm_axis_edges(tm: "TensorMesh") -> list[torch.Tensor]:
    """Per-axis float64 cell-EDGE coordinates ``origin + [0, cumsum(widths)]``
    (length ``shape[a] + 1`` each), the same construction as
    ``TensorMesh.cell_bounds``, so edges agree bit-for-bit with the boxes."""
    return [
        torch.cat([w.new_zeros(1), torch.cumsum(w.to(torch.float64), dim=0)]) + o
        for w, o in zip(tm.cell_widths, tm.origin)
    ]


def _tm_cell_boxes(tm: "TensorMesh") -> tuple[torch.Tensor, torch.Tensor]:
    """``(lo, hi)``: two ``(n_cells, n_dim)`` float64 tensors of axis-aligned
    cell bounds in ``cell_centers`` order (from ``cell_bounds``'s
    ``[lo0, hi0, lo1, hi1, ...]`` layout)."""
    b = tm.cell_bounds().to(torch.float64)
    return b[:, 0::2].contiguous(), b[:, 1::2].contiguous()


def _uniform_target_coarser(source: "TensorMesh", target: "TensorMesh") -> bool:
    """True iff the uniform target is genuinely coarser than the uniform source:
    no finer on any axis, strictly coarser on at least one (the same convention
    as the equal-extent ``F.interpolate('area')`` downsampling gate, but in
    physical spacing so it applies to unequal-extent / shifted pairs)."""
    return all(
        ts >= ss * (1.0 - _COARSER_REL_TOL)
        for ss, ts in zip(source.spacing, target.spacing)
    ) and any(
        ts > ss * (1.0 + _COARSER_REL_TOL)
        for ss, ts in zip(source.spacing, target.spacing)
    )


def _rect_target_coarser(source: "TensorMesh", target: "TensorMesh") -> bool:
    """Per-axis coarsening gate for rectilinear (possibly graded) pairs.

    True iff, restricted to the overlap region, EVERY target cell is at least
    as wide as EVERY source cell it overlaps on EVERY axis, and strictly wider
    somewhere, i.e. the target genuinely coarsens and overlap-volume averaging
    is the mass-preserving restriction. Any finer-than-source target cell keeps
    the exact linear interpolation instead (returns False)."""
    src_edges = _tm_axis_edges(source)
    tgt_edges = _tm_axis_edges(target)
    any_gt = False
    for a in range(source.n_dim):
        e = src_edges[a]
        sw = source.cell_widths[a].to(torch.float64)
        n_s = int(sw.numel())
        t_lo, t_hi = tgt_edges[a][:-1], tgt_edges[a][1:]
        sel = (t_hi > e[0]) & (t_lo < e[-1])       # cells overlapping the domain
        if not bool(sel.any()):
            continue
        tl, th = t_lo[sel].contiguous(), t_hi[sel].contiguous()
        j0 = (torch.searchsorted(e, tl, right=True) - 1).clamp(0, n_s - 1)
        j1 = (torch.searchsorted(e, th, right=False) - 1).clamp(0, n_s - 1)
        j1 = torch.maximum(j1, j0)
        k_max = int((j1 - j0 + 1).max())
        idx = (j0.unsqueeze(1) + torch.arange(k_max)).clamp(max=n_s - 1)
        valid = idx <= j1.unsqueeze(1)
        w_max = torch.where(
            valid, sw[idx], sw.new_tensor(float("-inf"))
        ).max(dim=1).values
        tw = th - tl
        if not bool((tw >= w_max * (1.0 - _COARSER_REL_TOL)).all()):
            return False
        if bool((tw > w_max * (1.0 + _COARSER_REL_TOL)).any()):
            any_gt = True
    return any_gt


def _build_rect_source_overlap(
    src_edges: list[torch.Tensor],
    tgt_lo: torch.Tensor,
    tgt_hi: torch.Tensor,
    *,
    full_volume_norm: bool,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse overlap-volume restriction weights against a RECTILINEAR source.

    For every axis-aligned target box ``T`` (rows of ``tgt_lo``/``tgt_hi``,
    ``(n_tgt, n_dim)``) and every source cell ``S`` of the rectilinear grid
    described by the per-axis ``src_edges``, the weight is
    ``w(T, S) = vol(T ∩ S) / norm(T)`` where the nD overlap is the per-axis
    separable product ``Π_a clamp(min(hi_T, hi_S) - max(lo_T, lo_S), 0)``.

    ``norm(T)`` implements the padding-policy semantics: the FULL box volume
    ``vol(T)`` when ``full_volume_norm`` (``padding='zeros'``, the uncovered
    fraction contributes 0), otherwise the COVERED volume ``Σ_S vol(T ∩ S)``
    (``'border'`` / ``'raise'``; the mean over the covered part; identical for
    fully-covered boxes). Rows are a partition of unity in the covered case, so
    constants are reproduced exactly.

    Assembly is CHUNKED over target rows (default block
    :data:`_OM_TO_TM_TARGET_BLOCK`) so peak memory is
    ``block × Π_a K_a`` (``K_a`` = max source cells overlapped per axis),
    bounded regardless of ``n_tgt``. Per-axis overlaps below
    :data:`_OVERLAP_REL_TOL` of the target width are dropped as float-noise
    slivers from nominally-coincident faces.

    Returns ``(W, covered)``: ``W`` sparse float64 ``(n_tgt, n_src)`` COO and
    ``covered`` a ``(n_tgt,)`` bool mask of rows with ANY positive overlap
    (rows outside the source domain are empty and flagged False).
    """
    n_dim = len(src_edges)
    tgt_lo = tgt_lo.to(torch.float64)
    tgt_hi = tgt_hi.to(torch.float64)
    n_tgt = int(tgt_lo.shape[0])
    edges = [e.to(torch.float64) for e in src_edges]
    src_shape = [int(e.numel()) - 1 for e in edges]
    strides = _flat_strides(src_shape)
    n_src = 1
    for s in src_shape:
        n_src *= s
    device = tgt_lo.device
    # Per-axis bracketing source-cell windows [j0, j1] for every target box.
    # side conventions keep exactly-shared faces OUT of the window: a target
    # edge sitting exactly on a source edge brackets to the cell it opens into.
    j0s: list[torch.Tensor] = []
    j1s: list[torch.Tensor] = []
    k_maxs: list[int] = []
    for a in range(n_dim):
        n_s = src_shape[a]
        j0 = (
            torch.searchsorted(edges[a], tgt_lo[:, a].contiguous(), right=True) - 1
        ).clamp(0, n_s - 1)
        j1 = (
            torch.searchsorted(edges[a], tgt_hi[:, a].contiguous(), right=False) - 1
        ).clamp(0, n_s - 1)
        j1 = torch.maximum(j1, j0)
        j0s.append(j0)
        j1s.append(j1)
        k_maxs.append(int((j1 - j0 + 1).max()))
    block = int(block_size) if block_size else _OM_TO_TM_TARGET_BLOCK
    rows_l: list[torch.Tensor] = []
    cols_l: list[torch.Tensor] = []
    vals_l: list[torch.Tensor] = []
    for b0 in range(0, n_tgt, block):
        b1 = min(b0 + block, n_tgt)
        w_axes: list[torch.Tensor] = []
        ix_axes: list[torch.Tensor] = []
        for a in range(n_dim):
            n_s = src_shape[a]
            j0 = j0s[a][b0:b1]
            idx = j0.unsqueeze(1) + torch.arange(k_maxs[a], device=device)
            valid = idx <= j1s[a][b0:b1].unsqueeze(1)
            idx = idx.clamp(max=n_s - 1)
            lo = torch.maximum(tgt_lo[b0:b1, a].unsqueeze(1), edges[a][idx])
            hi = torch.minimum(tgt_hi[b0:b1, a].unsqueeze(1), edges[a][idx + 1])
            ov = (hi - lo).clamp(min=0.0) * valid
            tol = _OVERLAP_REL_TOL * (
                tgt_hi[b0:b1, a] - tgt_lo[b0:b1, a]
            ).unsqueeze(1)
            ov = torch.where(ov > tol, ov, torch.zeros_like(ov))
            w_axes.append(ov)
            ix_axes.append(idx)
        w, col = _kron_expand(w_axes, ix_axes, strides)
        rows = (
            torch.arange(b0, b1, device=device).unsqueeze(1).expand_as(col)
        )
        keep = w > 0
        rows_l.append(rows[keep])
        cols_l.append(col[keep])
        vals_l.append(w[keep])
    rows = torch.cat(rows_l)
    cols = torch.cat(cols_l)
    vals = torch.cat(vals_l)
    cov_vol = torch.zeros(n_tgt, dtype=torch.float64, device=device)
    cov_vol.scatter_add_(0, rows, vals)
    covered = cov_vol > 0
    if full_volume_norm:
        norm = (tgt_hi - tgt_lo).prod(dim=1)
    else:
        norm = cov_vol
    vals = vals / norm[rows]
    W = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n_tgt, n_src),
        dtype=torch.float64,
    ).coalesce()
    return W, covered


def _build_om_to_tm_conservative(
    om: "OctreeMesh", tm: "TensorMesh", *, padding: str,
    block_size: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Overlap-volume averaging geometry for a COARSENING OctreeMesh→TensorMesh.

    The plain om_to_tm kernel gathers each TM cell's value from the single leaf
    containing its CENTRE, a TM cell covering several octree leaves reads ONE
    of them, aliasing integral quantities exactly like the tm_to_tm coarsening
    bug. Every TM cell whose box overlaps ``≥ 2`` leaves (i.e. it is coarser
    than, or straddles, the local octree) instead takes the overlap-volume
    weighted mean of the overlapped leaves; a cell inside a single leaf keeps
    the byte-identical gather.

    Chunked over TM (target) cells like :func:`_build_om_to_tm_index_map`
    (peak memory ``block × n_leaves × n_dim``). Normalization follows the
    padding policy: ``'zeros'`` divides by the FULL cell volume, ``'border'`` /
    ``'raise'`` by the COVERED volume (see :func:`_build_rect_source_overlap`).

    Returns ``(A, coarse_mask)``, ``A`` sparse float64 ``(n_tm, n_leaves)``
    with rows only for the ≥2-leaf cells, ``coarse_mask`` the ``(n_tm,)`` bool
    selector, or ``(None, None)`` when every TM cell sits inside a single leaf
    (the pure gather stays byte-identical to ``conservative=False``).
    """
    tm_lo, tm_hi = _tm_cell_boxes(tm)                     # (n_tm, n_dim) f64
    leaf_c = om.cell_centers().to(torch.float64)
    leaf_h = om.half_widths.to(torch.float64)
    s_lo = (leaf_c - leaf_h).unsqueeze(0)                 # (1, n_om, n_dim)
    s_hi = (leaf_c + leaf_h).unsqueeze(0)
    n_tm = int(tm_lo.shape[0])
    n_om = int(leaf_c.shape[0])
    device = tm_lo.device
    block = int(block_size) if block_size else _OM_TO_TM_TARGET_BLOCK
    coarse_mask = torch.zeros(n_tm, dtype=torch.bool, device=device)
    rows_l: list[torch.Tensor] = []
    cols_l: list[torch.Tensor] = []
    vals_l: list[torch.Tensor] = []
    for b0 in range(0, n_tm, block):
        b1 = min(b0 + block, n_tm)
        lo = torch.maximum(tm_lo[b0:b1].unsqueeze(1), s_lo)   # (blk, n_om, n_dim)
        hi = torch.minimum(tm_hi[b0:b1].unsqueeze(1), s_hi)
        ov_ax = (hi - lo).clamp(min=0.0)
        tol = _OVERLAP_REL_TOL * (tm_hi[b0:b1] - tm_lo[b0:b1]).unsqueeze(1)
        ov_ax = torch.where(ov_ax > tol, ov_ax, torch.zeros_like(ov_ax))
        ov = ov_ax.prod(dim=-1)                               # (blk, n_om)
        counts = (ov > 0).sum(dim=-1)
        coarse_blk = counts >= 2
        coarse_mask[b0:b1] = coarse_blk
        if bool(coarse_blk.any()):
            sub = ov[coarse_blk]                              # (m, n_om)
            local_rows = coarse_blk.nonzero(as_tuple=True)[0]
            nz_r, nz_c = (sub > 0).nonzero(as_tuple=True)
            rows_l.append(b0 + local_rows[nz_r])
            cols_l.append(nz_c)
            vals_l.append(sub[nz_r, nz_c])
    if not bool(coarse_mask.any()):
        return None, None
    rows = torch.cat(rows_l)
    cols = torch.cat(cols_l)
    vals = torch.cat(vals_l)
    if padding == "zeros":
        norm = (tm_hi - tm_lo).prod(dim=1)
    else:
        norm = torch.zeros(n_tm, dtype=torch.float64, device=device)
        norm.scatter_add_(0, rows, vals)
    vals = vals / norm[rows]
    A = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n_tm, n_om),
        dtype=torch.float64,
    ).coalesce()
    return A, coarse_mask


def _build_om_to_om_overlap(
    src: "OctreeMesh", tgt: "OctreeMesh", *, full_volume_norm: bool,
    block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact box-overlap volume weights for an OctreeMesh→OctreeMesh pair.

    Both meshes are sets of axis-aligned ``centre ± half`` boxes, so the
    overlap-volume restriction applies directly: for target leaf ``T`` and
    source leaf ``S``, ``w(T, S) = vol(T ∩ S) / norm(T)`` with the nD overlap
    the per-axis separable product (same construction as
    :func:`_build_rect_source_overlap`, against a leaf LIST instead of a
    rectilinear edge grid). ``norm(T)`` follows the padding policy:
    ``full_volume_norm`` (``padding='zeros'``) divides by the FULL leaf
    volume so the uncovered fraction contributes 0; otherwise
    (``'border'`` / ``'raise'``) by the COVERED volume; the mean over the
    covered part, a partition of unity, so constants are exact. The kernel is
    mass-conservative under coarsening and reduces to the exact containment
    gather (single weight 1) for a target leaf inside one source leaf.

    Chunked over target leaves like :func:`_build_om_to_tm_conservative`
    (peak memory ``block × n_src × n_dim``). Per-axis overlaps below
    :data:`_OVERLAP_REL_TOL` of the target-leaf width are dropped as float-
    noise slivers, so leaves sharing only a face do not couple.

    Returns ``(W, covered)``: ``W`` sparse float64 ``(n_tgt, n_src)`` COO and
    ``covered`` a ``(n_tgt,)`` bool mask of target leaves with ANY positive
    overlap (a zero-overlap leaf lies entirely outside the source leaf union;
    its row is empty and the caller applies the padding policy).
    """
    s_c = src.cell_centers().to(torch.float64)
    s_h = src.half_widths.to(torch.float64)
    t_c = tgt.cell_centers().to(torch.float64)
    t_h = tgt.half_widths.to(torch.float64)
    s_lo = (s_c - s_h).unsqueeze(0)                       # (1, n_src, n_dim)
    s_hi = (s_c + s_h).unsqueeze(0)
    t_lo = t_c - t_h                                      # (n_tgt, n_dim)
    t_hi = t_c + t_h
    n_t = int(t_c.shape[0])
    n_s = int(s_c.shape[0])
    device = t_c.device
    block = int(block_size) if block_size else _OM_TO_TM_TARGET_BLOCK
    rows_l: list[torch.Tensor] = []
    cols_l: list[torch.Tensor] = []
    vals_l: list[torch.Tensor] = []
    for b0 in range(0, n_t, block):
        b1 = min(b0 + block, n_t)
        lo = torch.maximum(t_lo[b0:b1].unsqueeze(1), s_lo)  # (blk, n_src, n_dim)
        hi = torch.minimum(t_hi[b0:b1].unsqueeze(1), s_hi)
        ov_ax = (hi - lo).clamp(min=0.0)
        tol = _OVERLAP_REL_TOL * (t_hi[b0:b1] - t_lo[b0:b1]).unsqueeze(1)
        ov_ax = torch.where(ov_ax > tol, ov_ax, torch.zeros_like(ov_ax))
        ov = ov_ax.prod(dim=-1)                             # (blk, n_src)
        nz_r, nz_c = (ov > 0).nonzero(as_tuple=True)
        rows_l.append(b0 + nz_r)
        cols_l.append(nz_c)
        vals_l.append(ov[nz_r, nz_c])
    rows = torch.cat(rows_l)
    cols = torch.cat(cols_l)
    vals = torch.cat(vals_l)
    cov_vol = torch.zeros(n_t, dtype=torch.float64, device=device)
    cov_vol.scatter_add_(0, rows, vals)
    covered = cov_vol > 0
    if full_volume_norm:
        norm = (t_hi - t_lo).prod(dim=1)
    else:
        norm = cov_vol
    vals = vals / norm[rows]
    W = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n_t, n_s), dtype=torch.float64,
    ).coalesce()
    return W, covered


def _nearest_center_rows(
    src_centers: torch.Tensor, tgt_centers: torch.Tensor,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    """Sparse ``(n_tgt, n_src)`` matrix with a single weight-1 entry per MASKED
    row, pointing at the nearest source centre, the 'border' clamp rows for
    targets outside the source coverage (splice via :func:`_replace_rows`).
    Built by the same k=1 kernel as ``method='nearest'`` for consistency."""
    rows = row_mask.nonzero(as_tuple=True)[0]
    sub = _build_general_weights(
        src_centers, tgt_centers[rows], method="nearest", k_neighbors=1,
    ).coalesce()
    idx = sub.indices()
    return torch.sparse_coo_tensor(
        torch.stack([rows[idx[0]], idx[1]]), sub.values(),
        size=(int(tgt_centers.shape[0]), int(src_centers.shape[0])),
        dtype=torch.float64,
    ).coalesce()


def _build_barycentric_weights(
    src_centers: torch.Tensor, tgt_centers: torch.Tensor, *, k_neighbors: int,
) -> torch.Tensor:
    """Sparse ``(n_tgt, n_src)`` linear-exact barycentric weights over a
    Delaunay triangulation of the SOURCE CELL CENTRES.

    Projection fields live on CELLS, so vertex-based barycentric interpolation
    over the mesh's own simplices does not apply; the cell-based linear-exact
    scheme instead triangulates the cell CENTRES (``scipy.spatial.Delaunay``;
    scipy is a hard dependency) and, for each target point, takes the
    barycentric weights of its containing centre-simplex over that simplex's
    vertex CELLS (``find_simplex`` + the affine ``transform``). Rows sum to 1
    and reproduce any linear field exactly (to float64 round-off, ~1e-10)
    INSIDE the centre-triangulation convex hull.

    Targets OUTSIDE the hull keep the existing k-NN IDW row VERBATIM (the
    ``_build_general_weights`` fallback): hull-based coverage is STRICTER than
    the bbox coverage the ``padding='raise'`` check uses
    (:func:`_general_uncovered`), so an in-bbox-but-outside-hull target, legal
    today under 'raise'; keeps working, just at IDW accuracy. The caller keeps
    the bbox mask for the raise/zeros policy exactly as before.

    Raises:
        GeoBrainError: when the centre cloud cannot be triangulated (fewer
            than ``n_dim + 1`` centres, or degenerate, collinear/coplanar).
    """
    from scipy.spatial import Delaunay, QhullError

    src = src_centers.to(torch.float64)
    tgt = tgt_centers.to(torch.float64)
    device = src.device
    n_s = int(src.shape[0])
    n_t = int(tgt.shape[0])
    n_dim = int(src.shape[1])
    try:
        tri = Delaunay(src.cpu().numpy())
    except (QhullError, ValueError) as exc:
        raise GeoBrainError(
            "method='barycentric' could not Delaunay-triangulate the source "
            "cell centres (degenerate cloud: too few, duplicate, or "
            "collinear/coplanar centres); use method='idw' or 'nearest' for "
            "this source",
            object_name="MeshProjection", field="method",
            expected="a Delaunay-triangulable source cell-centre cloud",
            actual=type(exc).__name__,
        ) from exc
    tgt_np = tgt.cpu().numpy()
    simplex = torch.from_numpy(tri.find_simplex(tgt_np)).to(torch.long)
    inside = simplex >= 0
    # Every row starts as the existing k-NN IDW interpolant; in-hull rows are
    # then replaced by the exact barycentric stencil, so outside-hull rows are
    # bit-identical to the plain method='idw' build.
    W_idw = _build_general_weights(
        src, tgt, method="idw", k_neighbors=k_neighbors,
    )
    if not bool(inside.any()):
        return W_idw
    si = simplex[inside].numpy()
    # Barycentric coordinates from Delaunay's stored affine maps: for point x
    # in simplex s, b = T_s @ (x - r_s) gives the first n_dim coordinates and
    # the last is 1 - Σ b (transform rows [:n_dim] are T_s, row n_dim is r_s).
    T = torch.from_numpy(tri.transform[si]).to(torch.float64)   # (m, d+1, d)
    x = tgt[inside]                                             # (m, d)
    b = torch.einsum("mij,mj->mi", T[:, :n_dim, :], x - T[:, n_dim, :])
    w = torch.cat([b, 1.0 - b.sum(dim=1, keepdim=True)], dim=1)   # (m, d+1)
    # scipy stores NaN transform rows for (near-)degenerate simplices; a target
    # landing in one must keep its finite IDW fallback row, not go NaN.
    finite = torch.isfinite(w).all(dim=1)
    if not bool(finite.all()):
        keep = inside.clone()
        keep[inside.nonzero(as_tuple=True)[0][~finite]] = False
        inside = keep
        si = si[finite.numpy()]
        w = w[finite]
        if not bool(inside.any()):
            return W_idw
    cols = torch.from_numpy(tri.simplices[si]).to(torch.long)     # (m, d+1)
    rows_in = inside.nonzero(as_tuple=True)[0]
    rows = rows_in.repeat_interleave(n_dim + 1)
    W_bary = torch.sparse_coo_tensor(
        torch.stack([rows.to(device), cols.reshape(-1).to(device)]),
        w.reshape(-1).to(device),
        size=(n_t, n_s), dtype=torch.float64,
    ).coalesce()
    return _replace_rows(W_idw, W_bary, inside.to(device))


def _flat_strides(shape: "list[int] | tuple[int, ...]") -> list[int]:
    """C-order flat strides (last axis fastest: matches cell_centers order)."""
    strides = [1] * len(shape)
    for a in range(len(shape) - 2, -1, -1):
        strides[a] = strides[a + 1] * shape[a + 1]
    return strides


def _kron_expand(
    w_axes: list[torch.Tensor],
    ix_axes: list[torch.Tensor],
    strides: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-aligned tensor-product combine of per-axis stencil tables.

    ``w_axes[a]`` / ``ix_axes[a]`` are ``(n_rows, K_a)`` per-axis weights and
    source AXIS indices for the same ``n_rows`` target cells. Returns
    ``(w, col)`` of shape ``(n_rows, Π K_a)``: the product weights and flat
    source-cell columns (``Σ_a idx_a · stride_a``) of the full nD stencil.
    """
    n_rows = int(w_axes[0].shape[0])
    w = w_axes[0]
    col = ix_axes[0] * strides[0]
    for a in range(1, len(w_axes)):
        w = (w.unsqueeze(2) * w_axes[a].unsqueeze(1)).reshape(n_rows, -1)
        col = (
            col.unsqueeze(2) + (ix_axes[a] * strides[a]).unsqueeze(1)
        ).reshape(n_rows, -1)
    return w, col


def _replace_rows(
    base: torch.Tensor, repl: torch.Tensor, row_mask: torch.Tensor,
) -> torch.Tensor:
    """Sparse-row surgery: drop ``base``'s entries on the masked rows and splice
    in ``repl``'s entries there (``repl`` must only hold rows inside the mask).
    Both sparse COO with identical shape; returns a coalesced sparse COO."""
    base = base.coalesce()
    repl = repl.coalesce()
    idx = base.indices()
    keep = ~row_mask[idx[0]]
    new_idx = torch.cat([idx[:, keep], repl.indices()], dim=1)
    new_vals = torch.cat([base.values()[keep], repl.values()])
    return torch.sparse_coo_tensor(
        new_idx, new_vals, size=base.shape, dtype=base.dtype,
    ).coalesce()


def _drop_rows(W: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    """Zero the masked rows of a sparse COO matrix (drop their entries),
    mirrors the forward's ``padding='zeros'`` output masking."""
    W = W.coalesce()
    idx = W.indices()
    keep = ~row_mask[idx[0]]
    return torch.sparse_coo_tensor(
        idx[:, keep], W.values()[keep], size=W.shape, dtype=W.dtype,
    ).coalesce()


def _grid_sample_weight_matrix(
    grid: torch.Tensor, src_shape: tuple[int, ...], *, method: str,
) -> torch.Tensor:
    """Sparse ``(n_points, n_src)`` matrix reproducing this module's
    ``grid_sample(..., padding_mode='border', align_corners=False)`` calls.

    ``grid`` is any of the precomputed normalised grids (``_grid`` /
    ``_tm_grid``); its flattened point order is the target cell order. Per
    grid_sample convention the LAST grid dim is ``[W, H]`` (2-D) or
    ``[W, H, D]`` (3-D) while the mesh axes are ``(H, W)`` / ``(D, H, W)``, so
    mesh axis ``a`` reads grid column ``n_dim - 1 - a``. Coordinates are
    unnormalised with ``x = ((g + 1)·n - 1) / 2`` and border-clipped into
    ``[0, n - 1]`` (exactly grid_sampler's ``align_corners=False`` +
    ``padding_mode='border'`` index math); ``method='linear'`` emits the
    ``2^n_dim``-corner multilinear stencil, ``'nearest'`` the round-half-even
    single corner.
    """
    n_dim = len(src_shape)
    g = grid.reshape(-1, n_dim).to(torch.float64)
    n_pts = int(g.shape[0])
    device = g.device
    strides = _flat_strides(list(src_shape))
    n_src = 1
    for s in src_shape:
        n_src *= s
    w_axes: list[torch.Tensor] = []
    ix_axes: list[torch.Tensor] = []
    for a in range(n_dim):
        size = src_shape[a]
        coord = g[:, n_dim - 1 - a]
        x = ((coord + 1.0) * size - 1.0) / 2.0
        x = x.clamp(0.0, float(size - 1))            # border padding
        if method == "linear":
            i0f = x.floor()
            w1 = x - i0f                              # 0 exactly at x == n-1
            i0 = i0f.long()
            i1 = (i0 + 1).clamp(max=size - 1)         # clamped corner has w1==0
            ix_axes.append(torch.stack([i0, i1], dim=1))
            w_axes.append(torch.stack([1.0 - w1, w1], dim=1))
        else:                                         # nearest: round half even
            i = torch.round(x).long().clamp(0, size - 1)
            ix_axes.append(i.unsqueeze(1))
            w_axes.append(torch.ones(n_pts, 1, dtype=torch.float64, device=device))
    w, col = _kron_expand(w_axes, ix_axes, strides)
    rows = torch.arange(n_pts, device=device).unsqueeze(1).expand_as(col)
    keep = w != 0
    return torch.sparse_coo_tensor(
        torch.stack([rows[keep], col[keep]]), w[keep],
        size=(n_pts, n_src), dtype=torch.float64,
    ).coalesce()


def _interpolate_weight_matrix(
    src_shape: tuple[int, ...], tgt_shape: tuple[int, ...], *,
    method: str, conservative: bool,
) -> torch.Tensor:
    """Sparse ``(n_tgt, n_src)`` matrix reproducing the equal-extent tm_to_tm
    ``F.interpolate`` fast paths (the same branch logic as ``_tm_to_tm``):

    - conservative + linear + pure downsampling → ``mode='area'``
      (adaptive average pooling: per-axis integer windows, uniform weights);
    - linear otherwise → bi/trilinear ``align_corners=False`` (per-axis
      source coordinate ``(i + 0.5)·(n_s/n_t) - 0.5`` clamped at 0, two-corner
      stencil with the top corner index-clamped like the ATen kernel);
    - nearest → ``floor(i·n_s/n_t)`` index-clamped.

    All three are per-axis separable, tensor-producted by
    :func:`_kron_expand`.
    """
    n_dim = len(src_shape)
    is_downsampling = (
        all(t <= s for t, s in zip(tgt_shape, src_shape))
        and tgt_shape != src_shape
    )
    use_area = conservative and method == "linear" and is_downsampling
    axis_idx: list[torch.Tensor] = []
    axis_w: list[torch.Tensor] = []
    for a in range(n_dim):
        n_s, n_t = int(src_shape[a]), int(tgt_shape[a])
        if use_area:
            i = torch.arange(n_t)
            start = (i * n_s) // n_t                       # floor
            end = ((i + 1) * n_s + n_t - 1) // n_t         # ceil
            k_max = int((end - start).max())
            idx = start.unsqueeze(1) + torch.arange(k_max)
            valid = idx < end.unsqueeze(1)
            idx = idx.clamp(max=n_s - 1)
            w = valid.to(torch.float64) / (end - start).to(torch.float64).unsqueeze(1)
        elif method == "linear":
            scale = n_s / n_t
            x = ((torch.arange(n_t, dtype=torch.float64) + 0.5) * scale - 0.5)
            x = x.clamp(min=0.0)
            i0f = x.floor()
            w1 = x - i0f
            i0 = i0f.long().clamp(max=n_s - 1)
            i1 = (i0 + 1).clamp(max=n_s - 1)
            idx = torch.stack([i0, i1], dim=1)
            w = torch.stack([1.0 - w1, w1], dim=1)
        else:                                              # nearest (legacy)
            scale = n_s / n_t
            i = (torch.arange(n_t, dtype=torch.float64) * scale).floor().long()
            idx = i.clamp(max=n_s - 1).unsqueeze(1)
            w = torch.ones(n_t, 1, dtype=torch.float64)
        axis_idx.append(idx)
        axis_w.append(w)
    # Expand the per-axis tables onto the flat target grid, then tensor-product.
    tgt_axes = [torch.arange(tgt_shape[a]) for a in range(n_dim)]
    tgt_mesh = [
        m.reshape(-1) for m in torch.meshgrid(*tgt_axes, indexing="ij")
    ]
    n_t_total = int(tgt_mesh[0].shape[0])
    w_axes = [axis_w[a][tgt_mesh[a]] for a in range(n_dim)]
    ix_axes = [axis_idx[a][tgt_mesh[a]] for a in range(n_dim)]
    strides = _flat_strides(list(src_shape))
    n_src = 1
    for s in src_shape:
        n_src *= s
    w, col = _kron_expand(w_axes, ix_axes, strides)
    rows = torch.arange(n_t_total).unsqueeze(1).expand_as(col)
    keep = w != 0
    return torch.sparse_coo_tensor(
        torch.stack([rows[keep], col[keep]]), w[keep],
        size=(n_t_total, n_src), dtype=torch.float64,
    ).coalesce()


def _validate_field_names(field_name: object) -> tuple[str, ...]:
    """Normalize ``field_name`` to a non-empty tuple of unique field names.

    A plain string (the common single-field case) becomes a 1-tuple; any other
    sequence of strings is validated (non-empty, all non-empty strings, no
    duplicates) and returned as a tuple. Anything else fails loudly.
    """
    if isinstance(field_name, str):
        if not field_name:
            raise GeoBrainError(
                "field_name is required",
                object_name="MeshProjection",
                field="field_name",
                expected="non-empty string",
                actual=field_name,
            )
        return (field_name,)
    try:
        names = tuple(field_name)                       # type: ignore[arg-type]
    except TypeError:
        raise GeoBrainError(
            "field_name must be a string or a sequence of strings",
            object_name="MeshProjection",
            field="field_name",
            expected="str or Sequence[str]",
            actual=field_name,
        ) from None
    if not names:
        raise GeoBrainError(
            "field_name sequence is empty",
            object_name="MeshProjection",
            field="field_name",
            expected="at least one field name",
            actual=field_name,
        )
    for n in names:
        if not isinstance(n, str) or not n:
            raise GeoBrainError(
                "every field name must be a non-empty string",
                object_name="MeshProjection",
                field="field_name",
                expected="non-empty strings",
                actual=n,
            )
    if len(set(names)) != len(names):
        raise GeoBrainError(
            "field names must be unique",
            object_name="MeshProjection",
            field="field_name",
            expected="no duplicate names",
            actual=names,
        )
    return names


def _validate_idw_power(idw_power: object) -> float:
    from numbers import Real

    if isinstance(idw_power, bool) or not isinstance(idw_power, Real) \
            or not float(idw_power) > 0:
        raise GeoBrainError(
            "MeshProjection idw_power must be a positive number",
            object_name="MeshProjection",
            field="idw_power",
            expected="> 0",
            actual=idw_power,
        )
    return float(idw_power)


def _validate_k_neighbors(k_neighbors: object) -> int:
    if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, int):
        raise GeoBrainError(
            "MeshProjection k_neighbors must be a positive integer",
            object_name="MeshProjection",
            field="k_neighbors",
            expected="positive integer",
            actual=k_neighbors,
        )
    if k_neighbors <= 0:
        raise GeoBrainError(
            "MeshProjection k_neighbors must be a positive integer",
            object_name="MeshProjection",
            field="k_neighbors",
            expected="positive integer",
            actual=k_neighbors,
        )
    return k_neighbors


# Target-cell block size for the chunked ``_build_om_to_tm_index_map`` search.
# The dense containment test materialises a ``(n_tm, n_om, n_dim)`` float64
# tensor (6.7 GB at a 32³ TM × 4096-leaf octree: OOM for realistic 3-D). We
# chunk over TARGET (TM) cells so peak memory is ``block × n_om × n_dim``,
# CONSTANT in n_tm, while the per-row containment / argmax / nearest math (and
# hence the returned indices/mask) stays byte-identical to the dense path.
_OM_TO_TM_TARGET_BLOCK = 4096


def _build_om_to_tm_index_map(
    om: "OctreeMesh", tm: "TensorMesh", *, block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each TM cell, return the ID of the OM leaf it reads, plus a coverage mask.

    O(n_tm × n_om) containment search, but CHUNKED over target (TM) cells so the
    dense ``(n_tm, n_om, n_dim)`` broadcast is never materialised: peak memory is
    ``block_size × n_om × n_dim`` regardless of n_tm. Every per-row operation
    (containment ``<= om_half + 1e-9``, first-True ``argmax`` for the containing
    leaf, nearest-leaf ``argmin`` tiebreak for uncovered cells) is row-independent,
    so the block-wise result is **byte-identical** to the un-chunked dense path,
    only the memory profile changes.

    ``block_size`` overrides the module default (:data:`_OM_TO_TM_TARGET_BLOCK`);
    it exists for tests (a tiny block exercises the chunk-boundary logic on small
    meshes) and does not change the result.

    Returns ``(indices, uncovered)`` where ``indices`` is ``(n_tm,)`` ``long``
    (the containing leaf for covered cells; the NEAREST leaf by centre distance
    for uncovered cells, so a ``padding='border'`` clamp is well defined) and
    ``uncovered`` is a ``(n_tm,)`` bool mask of TM cells whose centre lies in no
    OM leaf. The caller (``__init__``) enforces the coverage policy: raise under
    ``padding='raise'``, clamp under ``'border'``, or zero under ``'zeros'``.
    """
    # Keep TM centres in native float64: the subtraction below re-promotes to
    # float64 anyway (om_centres/om_half are float64), so a float32 cast here only
    # loses precision *before* the ``<= om_half + 1e-9`` containment test; its
    # float32 half-ulp at ~1e4 m (≈5e-4) is five orders above the 1e-9 tolerance
    # and can mis-bind a TM centre sitting just inside a leaf face.
    tm_centres = tm.cell_centers()
    om_centres = om.cell_centers()
    om_half = om.half_widths
    n_tm = int(tm_centres.shape[0])
    block = int(block_size) if block_size else _OM_TO_TM_TARGET_BLOCK
    # Broadcast constants for the leaf axis, computed ONCE (identical floats to
    # the dense expression ``om_centres.unsqueeze(0)`` / ``om_half.unsqueeze(0)
    # + 1e-9``): the per-block subtraction/compare reuses them.
    om_c = om_centres.unsqueeze(0)                       # (1, n_om, n_dim)
    om_h = om_half.unsqueeze(0) + 1.0e-9                 # (1, n_om, n_dim)
    idx_parts: list[torch.Tensor] = []
    unc_parts: list[torch.Tensor] = []
    for b0 in range(0, n_tm, block):
        b1 = min(b0 + block, n_tm)
        # diff_blk: (blk, n_om, n_dim): the exact slice ``diff[b0:b1]`` of the
        # dense broadcast, so every downstream per-row reduction matches it.
        diff = tm_centres[b0:b1].unsqueeze(1) - om_c
        inside = (diff.abs() <= om_h).all(dim=-1)        # (blk, n_om)
        covered = inside.any(dim=-1)
        # Argmax returns the first True per row (the containing leaf).
        indices = inside.long().argmax(dim=-1)
        if not bool(covered.all()):
            # Uncovered TM cells bind to their NEAREST OM leaf centre so 'border'
            # (clamp) and 'zeros' have a valid index to gather from.
            nearest = diff.norm(dim=-1).argmin(dim=-1)
            indices = torch.where(covered, indices, nearest)
        idx_parts.append(indices)
        unc_parts.append(~covered)
    return torch.cat(idx_parts), torch.cat(unc_parts)


def _tm_extents_match(source: "TensorMesh", target: "TensorMesh",
                      rel_tol: float = 1.0e-6) -> bool:
    """True iff the two uniform meshes span the same physical extent per axis.

    Extent is ``shape[i] * spacing[i]``; compared with a RELATIVE epsilon so
    float representation of spacings (e.g. spacing computed as extent/n) never
    spuriously flags an equal-extent pair.
    """
    return all(
        math.isclose(ns * hs, nt * ht, rel_tol=rel_tol)
        for ns, hs, nt, ht in zip(
            source.shape, source.spacing, target.shape, target.spacing
        )
    )


def _tm_origins_match(source: "TensorMesh", target: "TensorMesh",
                      rel_tol: float = 1.0e-6) -> bool:
    """True iff the two meshes share the same per-axis coordinate origin.

    Compared with an epsilon scaled by the per-axis source EXTENT so a float
    representation of an origin never spuriously flags an aligned pair. When
    origins differ the meshes occupy shifted physical regions, so an index-space
    projection would misplace the field, the caller routes such a pair to the
    physically-anchored grid instead.
    """
    return all(
        math.isclose(
            os, ot,
            rel_tol=0.0,
            abs_tol=rel_tol * max(abs(n * h), 1.0),
        )
        for os, ot, n, h in zip(
            source.origin, target.origin, source.shape, source.spacing
        )
    )


def _build_tm_to_tm_grid(src: "TensorMesh", tgt: "TensorMesh") -> torch.Tensor:
    """Normalised grid_sample grid for unequal-extent/shifted uniform TM→TM.

    Each target cell centre (a PHYSICAL coordinate that already includes the
    target's origin) is normalised to ``[-1, 1]`` over the SOURCE mesh's physical
    range ``[origin, origin + extent]``, subtract the source origin before the
    ``2*c/extent - 1`` map so a source with a non-zero origin lands correctly
    (same align_corners=False convention as ``_build_tm_to_om_grid``). Returns
    the grid in the shape ``grid_sample`` expects for a structured target:

    - 2-D: ``(1, nz_t, nx_t, 2)`` with last dim in grid_sample ``[W, H]`` order
      = ``[x, z]``.
    - 3-D: ``(1, nz_t, nx_t, ny_t, 3)`` with last dim in grid_sample
      ``[W, H, D]`` order = ``[y, x, z]``.
    """
    centres = tgt.cell_centers().reshape(*tgt.shape, tgt.n_dim)
    so = src.origin
    if src.n_dim == 2:
        # TM shape (nz, nx): grid_sample H=axis 0 (z), W=axis 1 (x).
        z_extent = src.shape[0] * src.spacing[0]   # H (z)
        x_extent = src.shape[1] * src.spacing[1]   # W (x)
        z_norm = 2.0 * (centres[..., 0] - so[0]) / z_extent - 1.0
        x_norm = 2.0 * (centres[..., 1] - so[1]) / x_extent - 1.0
        # grid last-dim [W, H] = [x, z]
        return torch.stack([x_norm, z_norm], dim=-1).unsqueeze(0)
    # 3-D: TM shape (nz, nx, ny): grid_sample D=axis 0 (z), H=axis 1 (x),
    # W=axis 2 (y). Each local is named for the axis it actually holds.
    z_extent = src.shape[0] * src.spacing[0]   # D (z)
    x_extent = src.shape[1] * src.spacing[1]   # H (x)
    y_extent = src.shape[2] * src.spacing[2]   # W (y)
    z_norm = 2.0 * (centres[..., 0] - so[0]) / z_extent - 1.0
    x_norm = 2.0 * (centres[..., 1] - so[1]) / x_extent - 1.0
    y_norm = 2.0 * (centres[..., 2] - so[2]) / y_extent - 1.0
    # grid last-dim [W, H, D] = [y, x, z]
    return torch.stack([y_norm, x_norm, z_norm], dim=-1).unsqueeze(0)


def _build_tm_to_om_grid(tm: "TensorMesh", om: "OctreeMesh") -> torch.Tensor:
    """Pre-build the normalised-coordinate grid used by ``grid_sample``.

    Returns a tensor in the shape ``grid_sample`` expects:

    - 2-D: ``(1, 1, n_cells, 2)`` with last dim in grid_sample ``[W, H]`` order
      = ``[x, z]``.
    - 3-D: ``(1, 1, 1, n_cells, 3)`` with last dim in grid_sample ``[W, H, D]``
      order = ``[y, x, z]``.

    Coordinates are normalised to ``[-1, 1]`` over the TM physical range
    ``[origin, origin + extent]`` (subtract the TM/source origin before the map
    so a shifted TM source samples the octree centres correctly).
    """
    om_centres = om.cell_centers()
    n_dim = tm.n_dim
    to = tm.origin
    if n_dim == 2:
        # TM shape (nz, nx): grid_sample H=axis 0 (z), W=axis 1 (x)
        z_extent = tm.shape[0] * tm.spacing[0]
        x_extent = tm.shape[1] * tm.spacing[1]
        z_norm = 2.0 * (om_centres[:, 0] - to[0]) / z_extent - 1.0   # H axis (z)
        x_norm = 2.0 * (om_centres[:, 1] - to[1]) / x_extent - 1.0   # W axis (x)
        # grid last-dim [W, H] = [x, z]
        grid = torch.stack([x_norm, z_norm], dim=-1)
        return grid.reshape(1, 1, -1, 2)
    # 3-D: TM shape (nz, nx, ny): grid_sample D=axis 0 (z), H=axis 1 (x),
    # W=axis 2 (y). Each local is named for the axis it actually holds.
    z_extent = tm.shape[0] * tm.spacing[0]   # D (z)
    x_extent = tm.shape[1] * tm.spacing[1]   # H (x)
    y_extent = tm.shape[2] * tm.spacing[2]   # W (y)
    z_norm = 2.0 * (om_centres[:, 0] - to[0]) / z_extent - 1.0    # D (z)
    x_norm = 2.0 * (om_centres[:, 1] - to[1]) / x_extent - 1.0    # H (x)
    y_norm = 2.0 * (om_centres[:, 2] - to[2]) / y_extent - 1.0    # W (y)
    # grid last-dim [W, H, D] = [y, x, z]
    grid = torch.stack([y_norm, x_norm, z_norm], dim=-1)
    return grid.reshape(1, 1, 1, -1, 3)


def _build_tm_to_om_conservative(
    tm: "TensorMesh", om: "OctreeMesh", *, padding: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Overlap-volume averaging geometry for a COARSENING TensorMesh→OctreeMesh.

    A fine TM field point-sampled at a coarse octree leaf centre (the plain
    ``grid_sample`` kernel) aliases and biases integral quantities, a compact
    source blob's mass swings by hundreds of percent depending on where the leaf
    centre lands. For every octree leaf GENUINELY COARSER than the (uniform)
    source, per axis no narrower than the source spacing, strictly wider on at
    least one axis; the mass-preserving answer is the overlap-volume weighted
    mean of the source cells it overlaps
    (:func:`_build_rect_source_overlap`; for a leaf aligned with the source
    grid this is exactly the plain mean of the covered cells, identical to the
    former centre-containment builder, and it stays exact for UNALIGNED or
    partially-covering leaves where centre counting is not). Finer-or-equal
    leaves keep the ``grid_sample`` point-sample. Normalization follows the
    padding policy: ``'zeros'`` divides by the full leaf volume, ``'border'`` /
    ``'raise'`` by the covered volume. A coarse leaf entirely OUTSIDE the
    source domain has no overlap; it is dropped from the mask so it keeps the
    grid_sample border/point value.

    Returns ``(A, coarse_mask)`` where ``A`` is a sparse ``(n_leaves, n_src)``
    COO matrix with rows only for the coarse leaves and ``coarse_mask`` is the
    ``(n_leaves,)`` bool selector; ``(None, None)`` when no leaf is coarser
    (the pure point-sample path stays byte-identical to
    ``conservative=False``). The weights are constant geometry, so the
    ``A @ field`` gather+mean is fully differentiable.
    """
    leaf_c = om.cell_centers().to(torch.float64)          # (n_leaf, n_dim)
    leaf_h = om.half_widths.to(torch.float64)             # (n_leaf, n_dim)
    n_leaf = int(leaf_c.shape[0])
    n_src = int(tm.n_cells)
    spacing = torch.tensor(tm.spacing, dtype=torch.float64)
    widths = 2.0 * leaf_h
    coarse_mask = (
        (widths >= spacing * (1.0 - _COARSER_REL_TOL)).all(dim=1)
        & (widths > spacing * (1.0 + _COARSER_REL_TOL)).any(dim=1)
    )
    if not bool(coarse_mask.any()):
        return None, None
    leaf_ids = coarse_mask.nonzero(as_tuple=True)[0]
    lo = (leaf_c - leaf_h)[leaf_ids]
    hi = (leaf_c + leaf_h)[leaf_ids]
    W_c, covered = _build_rect_source_overlap(
        _tm_axis_edges(tm), lo, hi,
        full_volume_norm=(padding == "zeros"),
    )
    if not bool(covered.all()):
        # Coarse leaves with NO source overlap keep the point-sample (their
        # rows are empty anyway); drop them from the selector.
        coarse_mask = coarse_mask.clone()
        coarse_mask[leaf_ids[~covered]] = False
        if not bool(coarse_mask.any()):
            return None, None
    idx = W_c.indices()
    A = torch.sparse_coo_tensor(
        torch.stack([leaf_ids[idx[0]], idx[1]]), W_c.values(),
        size=(n_leaf, n_src), dtype=torch.float64,
    ).coalesce()
    return A, coarse_mask


def _build_volume_weights(
    src_centers: torch.Tensor,
    tgt_centers: torch.Tensor,
    src_volumes: torch.Tensor,
    *,
    block_size: int | None = None,
) -> torch.Tensor:
    """Reverse-nearest volume-binned restriction: sparse ``(n_target, n_source)``.

    The conservative general-path coarsening (``method='volume'``): each
    SOURCE cell is assigned to its nearest target centre, a PARTITION of the
    source, and each target row volume-averages its assigned cells,
    ``w[t, s] = V_s / Σ_{s'→t} V_{s'}``. Rows are convex (sum 1, so constants
    are reproduced) and the partition property makes
    ``Σ_t out_t · (assigned volume)_t = Σ_s V_s f_s`` EXACT, the integral
    conservation the k-NN IDW interpolant cannot provide. A target that wins
    no source cell falls back to a nearest-source row (weight 1) so the field
    stays defined everywhere; such rows are interpolating, not conservative.

    Chunked over SOURCE cells with the same fixed-arithmetic policy as
    :func:`_build_general_weights` (argmin is row-independent).
    """
    src = src_centers.to(torch.float64)
    tgt = tgt_centers.to(torch.float64)
    vol = src_volumes.reshape(-1).to(torch.float64)
    device = src.device
    n_s = int(src.shape[0])
    n_t = int(tgt.shape[0])
    block = int(block_size) if block_size else _OM_TO_TM_TARGET_BLOCK
    assign_parts: list[torch.Tensor] = []
    for b0 in range(0, n_s, block):
        b1 = min(b0 + block, n_s)
        if n_s <= block:
            dist = torch.cdist(src, tgt)
        else:
            dist = torch.cdist(
                src[b0:b1], tgt,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
        assign_parts.append(dist.argmin(dim=1))
    assign = torch.cat(assign_parts)                        # (n_s,) target ids
    row_vol = torch.zeros(n_t, dtype=torch.float64, device=device)
    row_vol.index_add_(0, assign, vol)
    vals = vol / row_vol[assign]
    rows = assign
    cols = torch.arange(n_s, device=device)
    empty = row_vol == 0
    if bool(empty.any()):
        er = torch.nonzero(empty).reshape(-1)
        near = torch.cdist(tgt[er], src).argmin(dim=1)
        rows = torch.cat([rows, er])
        cols = torch.cat([cols, near])
        vals = torch.cat([
            vals, torch.ones(er.numel(), dtype=torch.float64, device=device),
        ])
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, size=(n_t, n_s),
        dtype=torch.float64,
    ).coalesce()


def _cyl_axis_edges_sq(mesh) -> list[torch.Tensor]:
    """Cyl axis edges in the ``(z, r²)`` measure frame.

    A ring cell's volume is ``π · Δz · Δ(r²)``, a tensor-product box volume
    in ``(z, r²)`` coordinates, so the EXACT conservative coarsening of a
    cyl↔cyl pair is plain rectangular overlap averaging in this frame
    (the constant π cancels between overlap and normalisation).
    """
    zn, rn = mesh.node_lines()
    return [zn.to(torch.float64), rn.to(torch.float64) ** 2]


def _cyl_cell_boxes_sq(mesh) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lo, hi)`` cyl cell boxes in the ``(z, r²)`` frame, C-order (z, r)."""
    zn, rn = (t.to(torch.float64) for t in mesh.node_lines())
    r2 = rn ** 2
    nz, nr = mesh.shape
    z_lo = zn[:-1, None].expand(nz, nr).reshape(-1)
    z_hi = zn[1:, None].expand(nz, nr).reshape(-1)
    r_lo = r2[None, :-1].expand(nz, nr).reshape(-1)
    r_hi = r2[None, 1:].expand(nz, nr).reshape(-1)
    return torch.stack([z_lo, r_lo], 1), torch.stack([z_hi, r_hi], 1)
