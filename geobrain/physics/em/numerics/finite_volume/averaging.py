"""
Cell-to-edge averaging operator for the 3D Yee mesh.

Returns ``A_e`` as a real ``float64`` sparse CSR tensor of shape
``(n_edges, n_cells)`` such that

    sigma_edge = A_e @ sigma_flat

equals the **volume-weighted** arithmetic average of the (up to 4) cells
adjacent to each Yee edge:

    σ_edge = Σ_c V_c · σ_c  /  Σ_c V_c

where ``V_c = wz·wy·wx`` is the cell dual volume from ``mesh.cell_widths``
(MESH1). On a uniform mesh all cells have the same volume, so the
formula collapses to a plain ``1 / n_existing`` arithmetic mean, and
the assembled operator is bit-identical to the pre-MESH3 build.
Boundary edges with fewer than 4 adjacent cells drop the missing
neighbours and renormalise, preserving ``A_e @ ones = ones``.

This convention matches:

- the sparse closed-form Jacobian weights
  ``geobrain.physics.em.numerics.finite_volume._sigma_jacobian.
  build_cell_to_edge_weights`` (volume-weighted, drop-missing-then-
  renormalise, the shared MT3D / FDEM3D / TEM3D cell→edge matrix);
- the dense functional averager ``_cell_to_edge_avg`` in ``curl_curl.py``
  (MESH3 made that one volume-aware too).

Edge index and family ordering exactly match :func:`yee_curl_e_to_f` in
``curl_curl.py``:

- Ex: ``i + j*nx + k*nx*(ny+1)``        for ``i in [0, nx)``, ``j in [0, ny]``, ``k in [0, nz]``.
- Ey: ``n_Ex + i + j*(nx+1) + k*(nx+1)*ny``  for ``i in [0, nx]``, ``j in [0, ny)``, ``k in [0, nz]``.
- Ez: ``n_Ex + n_Ey + i + j*(nx+1) + k*(nx+1)*(ny+1)``  for ``i in [0, nx]``, ``j in [0, ny]``, ``k in [0, nz)``.

Cell flat index: ``i + j*nx + k*nx*ny`` (x-fastest), matching the existing
3D Poisson and curl-curl assemblers.

Autograd: ``A_e`` itself is a constant geometry-only operator (no
parameters), so its ``.values()`` carries no gradient, but matvecs
``A_e @ sigma`` propagate gradients through ``sigma`` as usual via
``torch.sparse.mm`` / ``Tensor.__matmul__``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.mesh import TensorMesh
from geobrain.core.errors import GeoBrainError

__all__ = ["assemble_cell_to_edge_averaging"]


def _require_3d(mesh: TensorMesh, where: str) -> tuple[int, int, int]:
    if mesh.n_dim != 3:
        raise GeoBrainError(
            f"{where} requires a 3D TensorMesh, got n_dim={mesh.n_dim}",
            object_name=where, field="mesh.n_dim",
            expected="n_dim == 3", actual=mesh.n_dim,
        )
    nz, ny, nx = mesh.shape
    return int(nz), int(ny), int(nx)


def assemble_cell_to_edge_averaging(
    mesh: TensorMesh,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Sparse averaging operator ``A_e : cells -> edges``.

    Args:
        mesh: 3D :class:`~geobrain.mesh.TensorMesh`. Convention
            ``mesh.shape == (nz, ny, nx)``.
        device, dtype: Standard torch placement / precision controls. ``dtype``
            must be a real floating type; averaging weights are real.

    Returns:
        Sparse CSR tensor of shape ``(n_edges, n_cells)``. Row ``e`` contains 1, 2
        or 4 nonzero entries summing to 1.
    """
    nz, ny, nx = _require_3d(mesh, "assemble_cell_to_edge_averaging")

    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)
    n_ez = (nx + 1) * (ny + 1) * nz
    n_edges = n_ex + n_ey + n_ez
    n_cells = nx * ny * nz

    # MESH3: pull per-cell widths from the mesh and compute volume weights.
    # For an Ex edge the contributing cells all share the same x-cell, so
    # the relative volume reduces to wy[cy] * wz[cz] (wx[i] cancels in
    # numerator and denominator). Similarly for Ey/Ez. We therefore work
    # with the per-axis pair-volume only.
    wz_t, wy_t, wx_t = mesh.cell_widths
    wx = wx_t.to(device=device, dtype=dtype)                                  # (nx,)
    wy = wy_t.to(device=device, dtype=dtype)                                  # (ny,)
    wz = wz_t.to(device=device, dtype=dtype)                                  # (nz,)

    rows_list: list[torch.Tensor] = []
    cols_list: list[torch.Tensor] = []
    vals_list: list[torch.Tensor] = []

    long_dtype = torch.long

    def _cell_flat(i: torch.Tensor, cy: torch.Tensor, cz: torch.Tensor) -> torch.Tensor:
        return i + cy * nx + cz * (nx * ny)

    # ------------------------------------------------------------------
    # Ex edges: indexed (i, j, k) with i in [0, nx), j in [0, ny], k in [0, nz].
    # Adjacent cells: cy in {j-1, j} ∩ [0, ny), cz in {k-1, k} ∩ [0, nz).
    # Each surviving contribution gets weight (wy[cy] * wz[cz]) / Σ.
    # ------------------------------------------------------------------
    ii = torch.arange(nx, device=device, dtype=long_dtype)
    jj = torch.arange(ny + 1, device=device, dtype=long_dtype)
    kk = torch.arange(nz + 1, device=device, dtype=long_dtype)
    II, JJ, KK = torch.meshgrid(ii, jj, kk, indexing="ij")
    KK_p = KK.permute(2, 1, 0).reshape(-1)         # (k, j, i) -> flat
    JJ_p = JJ.permute(2, 1, 0).reshape(-1)
    II_p = II.permute(2, 1, 0).reshape(-1)
    edge_rows_ex = II_p + JJ_p * nx + KK_p * (nx * (ny + 1))

    # Σ wy[cy] over valid cy ∈ {j-1, j}; Σ wz[cz] similarly. Per-edge
    # total volume factor = (Σ wy) * (Σ wz).
    sum_wy_ex = (
        torch.where(JJ_p - 1 >= 0, wy[(JJ_p - 1).clamp(min=0)], torch.zeros((), dtype=dtype, device=device))
        + torch.where(JJ_p < ny, wy[JJ_p.clamp(max=ny - 1)], torch.zeros((), dtype=dtype, device=device))
    )                                                                          # (n_ex,)
    sum_wz_ex = (
        torch.where(KK_p - 1 >= 0, wz[(KK_p - 1).clamp(min=0)], torch.zeros((), dtype=dtype, device=device))
        + torch.where(KK_p < nz, wz[KK_p.clamp(max=nz - 1)], torch.zeros((), dtype=dtype, device=device))
    )
    inv_norm_ex = 1.0 / (sum_wy_ex * sum_wz_ex)                                # (n_ex,)

    for cy_off in (-1, 0):
        for cz_off in (-1, 0):
            cy = JJ_p + cy_off
            cz = KK_p + cz_off
            mask = (cy >= 0) & (cy < ny) & (cz >= 0) & (cz < nz)
            if not bool(mask.any()):
                continue
            sel_rows = edge_rows_ex[mask]
            sel_i = II_p[mask]
            sel_cy = cy[mask]
            sel_cz = cz[mask]
            sel_cols = _cell_flat(sel_i, sel_cy, sel_cz)
            # weight = (wy[cy] * wz[cz]) / (Σ wy)(Σ wz)
            sel_vals = (wy[sel_cy] * wz[sel_cz]) * inv_norm_ex[mask]
            rows_list.append(sel_rows)
            cols_list.append(sel_cols)
            vals_list.append(sel_vals)

    # ------------------------------------------------------------------
    # Ey edges: i in [0, nx], j in [0, ny), k in [0, nz]. Adjacent cells:
    # cx in {i-1, i} ∩ [0, nx), cz in {k-1, k} ∩ [0, nz).
    # ------------------------------------------------------------------
    ii = torch.arange(nx + 1, device=device, dtype=long_dtype)
    jj = torch.arange(ny, device=device, dtype=long_dtype)
    kk = torch.arange(nz + 1, device=device, dtype=long_dtype)
    II, JJ, KK = torch.meshgrid(ii, jj, kk, indexing="ij")
    KK_p = KK.permute(2, 1, 0).reshape(-1)
    JJ_p = JJ.permute(2, 1, 0).reshape(-1)
    II_p = II.permute(2, 1, 0).reshape(-1)
    edge_rows_ey = (
        n_ex + II_p + JJ_p * (nx + 1) + KK_p * ((nx + 1) * ny)
    )
    sum_wx_ey = (
        torch.where(II_p - 1 >= 0, wx[(II_p - 1).clamp(min=0)], torch.zeros((), dtype=dtype, device=device))
        + torch.where(II_p < nx, wx[II_p.clamp(max=nx - 1)], torch.zeros((), dtype=dtype, device=device))
    )
    sum_wz_ey = (
        torch.where(KK_p - 1 >= 0, wz[(KK_p - 1).clamp(min=0)], torch.zeros((), dtype=dtype, device=device))
        + torch.where(KK_p < nz, wz[KK_p.clamp(max=nz - 1)], torch.zeros((), dtype=dtype, device=device))
    )
    inv_norm_ey = 1.0 / (sum_wx_ey * sum_wz_ey)

    for cx_off in (-1, 0):
        for cz_off in (-1, 0):
            cx = II_p + cx_off
            cz = KK_p + cz_off
            mask = (cx >= 0) & (cx < nx) & (cz >= 0) & (cz < nz)
            if not bool(mask.any()):
                continue
            sel_rows = edge_rows_ey[mask]
            sel_cx = cx[mask]
            sel_j = JJ_p[mask]
            sel_cz = cz[mask]
            sel_cols = _cell_flat(sel_cx, sel_j, sel_cz)
            sel_vals = (wx[sel_cx] * wz[sel_cz]) * inv_norm_ey[mask]
            rows_list.append(sel_rows)
            cols_list.append(sel_cols)
            vals_list.append(sel_vals)

    # ------------------------------------------------------------------
    # Ez edges: i in [0, nx], j in [0, ny], k in [0, nz). Adjacent cells:
    # cx in {i-1, i} ∩ [0, nx), cy in {j-1, j} ∩ [0, ny).
    # ------------------------------------------------------------------
    ii = torch.arange(nx + 1, device=device, dtype=long_dtype)
    jj = torch.arange(ny + 1, device=device, dtype=long_dtype)
    kk = torch.arange(nz, device=device, dtype=long_dtype)
    II, JJ, KK = torch.meshgrid(ii, jj, kk, indexing="ij")
    KK_p = KK.permute(2, 1, 0).reshape(-1)
    JJ_p = JJ.permute(2, 1, 0).reshape(-1)
    II_p = II.permute(2, 1, 0).reshape(-1)
    edge_rows_ez = (
        n_ex + n_ey + II_p + JJ_p * (nx + 1) + KK_p * ((nx + 1) * (ny + 1))
    )
    sum_wx_ez = (
        torch.where(II_p - 1 >= 0, wx[(II_p - 1).clamp(min=0)], torch.zeros((), dtype=dtype, device=device))
        + torch.where(II_p < nx, wx[II_p.clamp(max=nx - 1)], torch.zeros((), dtype=dtype, device=device))
    )
    sum_wy_ez = (
        torch.where(JJ_p - 1 >= 0, wy[(JJ_p - 1).clamp(min=0)], torch.zeros((), dtype=dtype, device=device))
        + torch.where(JJ_p < ny, wy[JJ_p.clamp(max=ny - 1)], torch.zeros((), dtype=dtype, device=device))
    )
    inv_norm_ez = 1.0 / (sum_wx_ez * sum_wy_ez)

    for cx_off in (-1, 0):
        for cy_off in (-1, 0):
            cx = II_p + cx_off
            cy = JJ_p + cy_off
            mask = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
            if not bool(mask.any()):
                continue
            sel_rows = edge_rows_ez[mask]
            sel_cx = cx[mask]
            sel_cy = cy[mask]
            sel_k = KK_p[mask]
            sel_cols = _cell_flat(sel_cx, sel_cy, sel_k)
            sel_vals = (wx[sel_cx] * wy[sel_cy]) * inv_norm_ez[mask]
            rows_list.append(sel_rows)
            cols_list.append(sel_cols)
            vals_list.append(sel_vals)

    # ------------------------------------------------------------------
    # Assemble COO -> CSR.
    # ------------------------------------------------------------------
    rows = torch.cat(rows_list)
    cols = torch.cat(cols_list)
    vals = torch.cat(vals_list)
    indices = torch.stack([rows, cols], dim=0)

    A_coo = torch.sparse_coo_tensor(
        indices, vals, size=(n_edges, n_cells), device=device, dtype=dtype
    ).coalesce()
    return A_coo.to_sparse_csr()
