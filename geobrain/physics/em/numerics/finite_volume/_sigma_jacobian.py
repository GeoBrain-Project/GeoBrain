# pyright: reportPrivateImportUsage=false
"""
Cell-to-edge volume-weighted weights + closed-form σ-Jacobian helpers.

Shared between MT3D / FDEM3D / TEM3D; they all share the
**same** ``W`` cell-to-edge interpolation matrix (volume-weighted
arithmetic averaging on a Yee grid) and the **same** closed-form
σ-Jacobian shape: only the curl-curl diagonal depends on σ via
``A[e, e] += factor · σ_edge[e]`` where ``σ_edge = W @ σ``. The
per-operator wrappers just choose the scalar ``factor`` and call the
shared builder:

- MT3D / FDEM3D: ``factor = +i · ω · μ₀`` (complex). The bridge takes
  ``jac_vals`` as the literal entries of the assembled ``A``, so the
  closed-form factor must match the sign the assembler stamps. Under the
  ``exp(+iωt)`` convention used here ``yee_curl_curl_assemble`` stamps
  ``+i·ω·μ₀·σ_edge`` on the diagonal, so the derivative is ``+i·ω·μ₀·W``
  (see ``build_sigma_jacobian_curl_curl``; line 272 sets
  ``factor = 1j·ω·μ₀``).
- TEM3D per-step: ``factor = α₀_n`` (real BDF coefficient = ``1/dt_n``),
  multiplied by a per-edge dual-volume weighting ``V_edge``. TEM3D runs
  the volume-weighted mass ``M_eσ = diag(σ_edge·V_edge)``,
  so the operator passes a real ``edge_volume`` array (``self._V_edge``),
  *not* ``None``. ``edge_volume=None`` recovers the bare-mass form but
  no in-tree caller uses it.

This file does **not** depend on any operator-specific class; it only
needs a 3-D :class:`TensorMesh`. The DC3D σ-Jacobian (Poisson stencil)
is structurally different (face transmissibilities, harmonic-mean
quotient-rule derivatives) but is a pure numerics quantity too, so it
lives here as well: ``build_sigma_jacobian_dc3d`` (with the per-axis
helper ``_axis_block_sigma_jacobian`` and the closed-form solve
``_dc3d_solve_closed_form``). ``dc3d`` re-exports the public builder so
``from ...static.dc3d import build_sigma_jacobian_dc3d`` keeps working,
and ``ip`` imports it from this canonical location.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from geobrain.mesh import TensorMesh
from geobrain.core.errors import GeoBrainError


__all__ = [
    "build_cell_to_edge_weights",
    "build_sigma_jacobian_curl_curl",
    "build_sigma_jacobian_tem3d_step",
    "build_sigma_jacobian_dc3d",
]


def _cell_idx(i: int, j: int, k: int, nx: int, ny: int) -> int:
    """
    Linear cell index, x-fastest then y then z.

    Matches the (nz, ny, nx)-shaped sigma layout used by the
    curl-curl + Yee edge index conventions:
    ``flat = i + j·nx + k·nx·ny``.
    """
    return i + j * nx + k * nx * ny


def build_cell_to_edge_weights(mesh: TensorMesh) -> sp.csr_matrix:
    """
    Volume-weighted arithmetic cell → edge interpolation matrix.

    Mirrors ``natural_source/mt3d/jacobian.build_cell_to_edge_weights``.
    Returns a ``scipy.sparse.csr_matrix`` of shape ``(n_edges, n_cells)``
    in ``float64``. Each row corresponds to one Yee edge (in the global
    ``[ex; ey; ez]`` ordering used by :func:`yee_curl_e_to_f`) and
    contains 1 - 4 entries summing to 1.

    Edge ordering (matches ``curl_curl.yee_curl_e_to_f``):

    - Ex edges: ``i ∈ [0, nx)``, ``j ∈ [0, ny]``, ``k ∈ [0, nz]``;
      flat = ``i + j·nx + k·nx·(ny+1)``.
    - Ey edges (offset by ``n_Ex``): ``i ∈ [0, nx]``, ``j ∈ [0, ny)``,
      ``k ∈ [0, nz]``; flat = ``i + j·(nx+1) + k·(nx+1)·ny``.
    - Ez edges (offset by ``n_Ex+n_Ey``): ``i ∈ [0, nx]``,
      ``j ∈ [0, ny]``, ``k ∈ [0, nz)``;
      flat = ``i + j·(nx+1) + k·(nx+1)·(ny+1)``.

    Each interior edge averages 4 cells; boundary edges average 2; corner
    edges average 1. Weights are proportional to cell volumes so
    non-uniform meshes get a physically sensible average.

    Args:
        mesh: 3-D :class:`TensorMesh`.

    Returns:
        ``W`` as a ``scipy.sparse.csr_matrix``, shape
        ``(n_edges, n_cells)``, ``float64``.
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "build_cell_to_edge_weights requires a 3-D TensorMesh",
            object_name="build_cell_to_edge_weights",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )

    # ``mesh.shape == (nz, ny, nx)`` to match curl_curl.yee_curl_e_to_f.
    nz, ny, nx = mesh.shape
    wz_t, wy_t, wx_t = mesh.cell_widths
    dx = wx_t.detach().cpu().numpy().astype(np.float64, copy=False)
    dy = wy_t.detach().cpu().numpy().astype(np.float64, copy=False)
    dz = wz_t.detach().cpu().numpy().astype(np.float64, copy=False)

    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)
    n_ez = (nx + 1) * (ny + 1) * nz
    n_edges = n_ex + n_ey + n_ez
    n_cells = nx * ny * nz

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    # Ex edges: index = i + j·nx + k·nx·(ny+1).
    for k in range(nz + 1):
        z_lo = k - 1
        z_hi = k
        z_range = [z for z in (z_lo, z_hi) if 0 <= z < nz]
        if not z_range:
            continue
        for j in range(ny + 1):
            y_lo = j - 1
            y_hi = j
            y_range = [y for y in (y_lo, y_hi) if 0 <= y < ny]
            if not y_range:
                continue
            for i in range(nx):
                edge_row = i + j * nx + k * nx * (ny + 1)
                contribs: list[tuple[int, float]] = []
                total = 0.0
                for cz in z_range:
                    for cy in y_range:
                        c = _cell_idx(i, cy, cz, nx, ny)
                        v = dx[i] * dy[cy] * dz[cz]
                        contribs.append((c, v))
                        total += v
                inv_total = 1.0 / total
                for c, v in contribs:
                    rows.append(edge_row)
                    cols.append(c)
                    vals.append(v * inv_total)

    # Ey edges: offset = n_ex; index inside family = i + j·(nx+1) + k·(nx+1)·ny.
    for k in range(nz + 1):
        z_lo = k - 1
        z_hi = k
        z_range = [z for z in (z_lo, z_hi) if 0 <= z < nz]
        if not z_range:
            continue
        for j in range(ny):
            for i in range(nx + 1):
                edge_row = n_ex + i + j * (nx + 1) + k * (nx + 1) * ny
                x_lo = i - 1
                x_hi = i
                x_range = [x for x in (x_lo, x_hi) if 0 <= x < nx]
                if not x_range:
                    continue
                contribs = []
                total = 0.0
                for cz in z_range:
                    for cx in x_range:
                        c = _cell_idx(cx, j, cz, nx, ny)
                        v = dx[cx] * dy[j] * dz[cz]
                        contribs.append((c, v))
                        total += v
                inv_total = 1.0 / total
                for c, v in contribs:
                    rows.append(edge_row)
                    cols.append(c)
                    vals.append(v * inv_total)

    # Ez edges: offset = n_ex + n_ey;
    # index inside family = i + j·(nx+1) + k·(nx+1)·(ny+1).
    for k in range(nz):
        for j in range(ny + 1):
            y_lo = j - 1
            y_hi = j
            y_range = [y for y in (y_lo, y_hi) if 0 <= y < ny]
            if not y_range:
                continue
            for i in range(nx + 1):
                edge_row = (
                    n_ex + n_ey + i + j * (nx + 1) + k * (nx + 1) * (ny + 1)
                )
                x_lo = i - 1
                x_hi = i
                x_range = [x for x in (x_lo, x_hi) if 0 <= x < nx]
                if not x_range:
                    continue
                contribs = []
                total = 0.0
                for cy in y_range:
                    for cx in x_range:
                        c = _cell_idx(cx, cy, k, nx, ny)
                        v = dx[cx] * dy[cy] * dz[k]
                        contribs.append((c, v))
                        total += v
                inv_total = 1.0 / total
                for c, v in contribs:
                    rows.append(edge_row)
                    cols.append(c)
                    vals.append(v * inv_total)

    rows_arr = np.asarray(rows, dtype=np.int64)
    cols_arr = np.asarray(cols, dtype=np.int64)
    vals_arr = np.asarray(vals, dtype=np.float64)

    W = sp.coo_matrix(
        (vals_arr, (rows_arr, cols_arr)),
        shape=(n_edges, n_cells), dtype=np.float64,
    ).tocsr()
    return W


def build_sigma_jacobian_curl_curl(
    mesh: TensorMesh,
    omega: float,
    cell_to_edge_weights: sp.csr_matrix,
    edge_volume: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-form σ-Jacobian for the Yee curl-curl diagonal (MT3D / FDEM3D).

    The Yee curl-curl operator is
    ``A = K - i·ω·μ₀·diag(σ_edge·V_edge)`` or
    ``A = K + i·ω·μ₀·diag(σ_edge·V_edge)`` (the exp(+iωt) convention used here).

    Either way, the only σ-dependent block of ``A`` is the diagonal, and
    ``σ_edge = W @ σ``. So
    ``dA[e, e] / dσ[c] = ±i·ω·μ₀ · V_edge[e] · W[e, c]``.

    ``edge_volume`` carries the per-edge dual volume ``V_edge`` that the E1
    mimetic edge mass ``M_e(σ) = diag(σ_edge·V_edge)`` folds into A's
    diagonal (see :func:`~...curl_curl.yee_dual_volume_weights`). It is
    ``None`` on a uniform mesh, where the weights are all ``1``, recovering
    the bare ``±i·ω·μ₀·W`` derivative bit-identically. When A is assembled
    with the E1 weighting on a non-uniform mesh, pass the SAME ``edge_w`` so
    the closed-form derivative matches A's stamped diagonal.

    The :class:`SparseLinearSolveWithSigmaGrad` bridge takes
    ``jac_vals`` as the literal entries of the assembled ``A``, so its
    sign must match whatever the assembler stamps. the
    ``yee_curl_curl_assemble`` stamps ``+i·ω·μ₀·σ_edge`` on the diagonal,
    making the closed-form derivative ``+i·ω·μ₀ · W[e, c]``.

    An ``exp(-iωt)`` convention would stamp ``-i·ω·μ₀·σ_edge`` and use ``-iω``;
    to stay bit-identical to the assembled ``A`` we follow the ``+iω`` sign here.

    Args:
        mesh: 3-D :class:`TensorMesh` (used only for the dimension check).
        omega: angular frequency (rad/s).
        cell_to_edge_weights: output of :func:`build_cell_to_edge_weights`.

    Returns:
        4-tuple ``(jac_rows, jac_cols, jac_sigma_idx, jac_vals)`` consumed
        directly by ``SparseLinearSolveWithSigmaGrad``. ``jac_rows == jac_cols``
        (diagonal only), ``jac_sigma_idx`` is the cell index per
        entry, ``jac_vals`` is **complex128** because of the ``i·ω·μ₀``
        factor.
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "build_sigma_jacobian_curl_curl requires a 3-D TensorMesh",
            object_name="build_sigma_jacobian_curl_curl",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    from geobrain.physics.em.conventions import MU_0

    W_coo = cell_to_edge_weights.tocoo()
    edges = W_coo.row.astype(np.int64)
    cells = W_coo.col.astype(np.int64)
    weights = W_coo.data.astype(np.float64)
    # the assembler stamps +i·ω·μ₀·σ_edge·V_edge on the diagonal of A; the
    # closed-form derivative matches in sign (and dual-volume weighting).
    factor = 1j * float(omega) * float(MU_0)
    if edge_volume is not None:
        ev = np.asarray(edge_volume, dtype=np.float64)
        if ev.shape[0] != cell_to_edge_weights.shape[0]:
            raise GeoBrainError(
                "edge_volume length must match number of edges",
                object_name="build_sigma_jacobian_curl_curl",
                field="edge_volume.shape[0]",
                expected=cell_to_edge_weights.shape[0],
                actual=ev.shape[0],
            )
        weights = weights * ev[edges]
    vals = factor * weights.astype(np.complex128)
    return edges, edges.copy(), cells, vals


def build_sigma_jacobian_tem3d_step(
    mesh: TensorMesh,
    alpha0: float,
    cell_to_edge_weights: sp.csr_matrix,
    edge_volume: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Closed-form σ-Jacobian for one BDF time step of TEM3D.

    The per-step system matrix under the volume-weighted formulation is
    ``A_n = K + α₀_n · diag(σ_edge · V_edge)`` (``α₀_n = 1/dt_n``) so
    ``dA_n[e, e] / dσ[c] = α₀_n · V_edge[e] · W[e, c]`` (real).

    ``edge_volume`` carries the per-edge dual-cell volume ``V_edge``.
    TEM3D runs this volume-weighted mass, so the operator passes a real
    ``edge_volume`` array (``self._V_edge``). Passing ``edge_volume=None``
    recovers the bare-mass derivative ``α₀_n · W[e, c]``, but no in-tree
    caller uses it.

    Args:
        mesh: 3-D :class:`TensorMesh`.
        alpha0: BDF coefficient on the mass matrix for this step (1.0 for
            BDF1, ``(1 + 2r)/(1 + r)`` for variable-step BDF2).
        cell_to_edge_weights: output of :func:`build_cell_to_edge_weights`.
        edge_volume: optional ``(n_edges,)`` array of per-edge dual-cell
            volumes for the mimetic form (None ⇒ bare-mass).

    Returns:
        4-tuple ``(jac_rows, jac_cols, jac_sigma_idx, jac_vals)`` consumed
        by ``SparseLinearSolveWithSigmaGrad``. ``jac_vals`` is **float64**
        (real-time domain).
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "build_sigma_jacobian_tem3d_step requires a 3-D TensorMesh",
            object_name="build_sigma_jacobian_tem3d_step",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    W_coo = cell_to_edge_weights.tocoo()
    edges = W_coo.row.astype(np.int64)
    cells = W_coo.col.astype(np.int64)
    weights = W_coo.data.astype(np.float64)
    if edge_volume is None:
        vals = float(alpha0) * weights
    else:
        ev = np.asarray(edge_volume, dtype=np.float64)
        if ev.shape[0] != cell_to_edge_weights.shape[0]:
            raise GeoBrainError(
                "edge_volume length must match number of edges",
                object_name="build_sigma_jacobian_tem3d_step",
                field="edge_volume.shape[0]",
                expected=cell_to_edge_weights.shape[0],
                actual=ev.shape[0],
            )
        vals = float(alpha0) * weights * ev[edges]
    return edges, edges.copy(), cells, vals


# ---------------------------------------------------------------------------
# EM3: closed-form σ-Jacobian for the 7-point FV Poisson stencil
# ---------------------------------------------------------------------------
#
# Structurally distinct from the curl-curl Jacobian above (face
# transmissibilities + harmonic-mean quotient-rule derivatives rather than a
# σ-linear curl-curl diagonal), but it is the same kind of object: a pure
# closed-form sparse σ-Jacobian consumed by ``SparseLinearSolveWithSigmaGrad``.
# It is shared by the DC3D operator (via ``_dc3d_solve_closed_form``) and by
# IP3D's structured path, so it lives in the numerics layer
# beside its curl-curl sibling rather than inside an operator file.


def _axis_block_sigma_jacobian(
    axis: int,
    sigma_3d_np,  # numpy float64 / complex128
    dx,
    dy,
    dz,
    nz: int,
    ny: int,
    nx: int,
    rows_chunks: list,
    cols_chunks: list,
    sig_chunks: list,
    val_chunks: list,
) -> None:
    """
    Append one axis worth of sigma-Jacobian entries.

    The flat-index convention follows :func:`assemble_poisson_3d`: the
    last axis varies fastest. The internal names (nz, ny, nx) are labels;
    under the platform ``(nz, nx, ny)`` layout this is
    ``iy + ix*ny + iz*nx*ny``: the same row-major order used everywhere
    else in DC3D (assembler, forward, survey). The builder is
    layout-generic: it reads the shape/widths per axis.

    For the harmonic-mean transmissibility
    ``T = 2 s_l s_r A_face / (s_l d_r + s_r d_l)``,
    the quotient rule gives

        dT/ds_l = 2 s_r² A_face d_l / denom²
        dT/ds_r = 2 s_l² A_face d_r / denom²

    so each interior face stamps 8 (A-entry, σ-cell) Jacobian rows
    (4 A-entries × 2 σ-cells per face).
    """
    import numpy as np

    if axis == 0:  # x-faces: i, i+1
        if nx <= 1:
            return
        kk, jj, ii = np.meshgrid(
            np.arange(nz), np.arange(ny), np.arange(nx - 1), indexing="ij"
        )
        kk, jj, ii = kk.ravel(), jj.ravel(), ii.ravel()
        left = ii + jj * nx + kk * nx * ny
        right = left + 1
        s_l = sigma_3d_np[kk, jj, ii]
        s_r = sigma_3d_np[kk, jj, ii + 1]
        d_l = dx[ii]
        d_r = dx[ii + 1]
        area = dy[jj] * dz[kk]
    elif axis == 1:  # y-faces: j, j+1
        if ny <= 1:
            return
        kk, jj, ii = np.meshgrid(
            np.arange(nz), np.arange(ny - 1), np.arange(nx), indexing="ij"
        )
        kk, jj, ii = kk.ravel(), jj.ravel(), ii.ravel()
        left = ii + jj * nx + kk * nx * ny
        right = left + nx
        s_l = sigma_3d_np[kk, jj, ii]
        s_r = sigma_3d_np[kk, jj + 1, ii]
        d_l = dy[jj]
        d_r = dy[jj + 1]
        area = dx[ii] * dz[kk]
    elif axis == 2:  # z-faces: k, k+1
        if nz <= 1:
            return
        kk, jj, ii = np.meshgrid(
            np.arange(nz - 1), np.arange(ny), np.arange(nx), indexing="ij"
        )
        kk, jj, ii = kk.ravel(), jj.ravel(), ii.ravel()
        left = ii + jj * nx + kk * nx * ny
        right = left + nx * ny
        s_l = sigma_3d_np[kk, jj, ii]
        s_r = sigma_3d_np[kk + 1, jj, ii]
        d_l = dz[kk]
        d_r = dz[kk + 1]
        area = dx[ii] * dy[jj]
    else:
        raise GeoBrainError(
            f"axis must be 0, 1, or 2, got {axis}",
            object_name="_axis_block_sigma_jacobian", field="axis",
            expected="0, 1, or 2", actual=axis,
        )

    denom = s_l * d_r + s_r * d_l + 1e-30
    denom2 = denom * denom
    dT_dsl = 2.0 * s_r * s_r * area * d_l / denom2
    dT_dsr = 2.0 * s_l * s_l * area * d_r / denom2

    # 8 entries per face: 4 A-entries × 2 σ-cells.
    rows_chunks.append(np.concatenate([left, left, right, right, left, left, right, right]))
    cols_chunks.append(np.concatenate([left, left, right, right, right, right, left, left]))
    sig_chunks.append(np.concatenate([left, right, left, right, left, right, left, right]))
    val_chunks.append(
        np.concatenate(
            [
                +dT_dsl,  # d A[l,l] / d s_l
                +dT_dsr,  # d A[l,l] / d s_r
                +dT_dsl,  # d A[r,r] / d s_l
                +dT_dsr,  # d A[r,r] / d s_r
                -dT_dsl,  # d A[l,r] / d s_l
                -dT_dsr,  # d A[l,r] / d s_r
                -dT_dsl,  # d A[r,l] / d s_l
                -dT_dsr,  # d A[r,l] / d s_r
            ]
        )
    )


def build_sigma_jacobian_dc3d(
    mesh: TensorMesh,
    sigma_np,
    dirichlet_idx: int,
):
    """
    Closed-form sparse σ-Jacobian of the FV Poisson stencil.

    Matches :func:`assemble_poisson_3d`'s row-major (last-axis-fastest)
    flat index; under the platform ``(nz, nx, ny)`` layout this is
    ``iy + ix*ny + iz*nx*ny``. The builder is layout-generic (reads the
    mesh shape/widths per axis).

    Args:
        mesh: 3-D :class:`TensorMesh` matching the one passed to
            :func:`assemble_poisson_3d`. Uses ``mesh.cell_widths`` so
            non-uniform meshes are honoured.
        sigma_np: cell conductivities as numpy array, shape
            ``(n_cells,)`` (in flat row-major mesh order). Either
            ``float64`` (DC path) or ``complex128`` (SIP). ``vals`` is
            returned with the same dtype as ``sigma_np``.
        dirichlet_idx: cell index where the FV potential is pinned. Entries
            whose row or column equals this index are dropped because the
            symmetric pin removes them from ``A``.

    Returns:
        4-tuple ``(rows, cols, sig, vals)`` of equal-length numpy arrays
        consumed directly by :class:`SparseLinearSolveWithSigmaGrad`.
    """
    import numpy as np

    if mesh.n_dim != 3:
        raise GeoBrainError(
            "build_sigma_jacobian_dc3d requires a 3-D TensorMesh",
            object_name="build_sigma_jacobian_dc3d",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    nz, ny, nx = mesh.shape
    sigma_np = np.asarray(sigma_np)
    if sigma_np.shape != (nz * ny * nx,):
        raise GeoBrainError(
            "sigma_np must be flat with length nz·ny·nx",
            object_name="build_sigma_jacobian_dc3d",
            field="sigma_np.shape",
            expected=(nz * ny * nx,),
            actual=sigma_np.shape,
        )
    val_dtype = sigma_np.dtype
    sigma_3d = sigma_np.reshape(nz, ny, nx)

    wz_t, wy_t, wx_t = mesh.cell_widths
    dx_np = wx_t.detach().cpu().numpy().astype(np.float64, copy=False)
    dy_np = wy_t.detach().cpu().numpy().astype(np.float64, copy=False)
    dz_np = wz_t.detach().cpu().numpy().astype(np.float64, copy=False)

    rows_chunks: list = []
    cols_chunks: list = []
    sig_chunks: list = []
    val_chunks: list = []

    _axis_block_sigma_jacobian(0, sigma_3d, dx_np, dy_np, dz_np, nz, ny, nx,
                               rows_chunks, cols_chunks, sig_chunks, val_chunks)
    _axis_block_sigma_jacobian(1, sigma_3d, dx_np, dy_np, dz_np, nz, ny, nx,
                               rows_chunks, cols_chunks, sig_chunks, val_chunks)
    _axis_block_sigma_jacobian(2, sigma_3d, dx_np, dy_np, dz_np, nz, ny, nx,
                               rows_chunks, cols_chunks, sig_chunks, val_chunks)

    if not rows_chunks:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty, np.zeros(0, dtype=val_dtype)

    rows_arr = np.concatenate(rows_chunks).astype(np.int64)
    cols_arr = np.concatenate(cols_chunks).astype(np.int64)
    sig_arr = np.concatenate(sig_chunks).astype(np.int64)
    val_arr = np.concatenate(val_chunks).astype(val_dtype)

    # Symmetric Dirichlet pin: drop entries whose row OR column equals
    # the pinned cell. ``_assemble_dc3d_sparse`` zeros these out (row
    # becomes the unit row, column is dropped from every other row), so
    # the Jacobian must agree; otherwise the bridge emits spurious
    # gradient contributions for non-existent A entries.
    pin = int(dirichlet_idx)
    keep = (rows_arr != pin) & (cols_arr != pin)
    if not keep.all():
        rows_arr = rows_arr[keep]
        cols_arr = cols_arr[keep]
        sig_arr = sig_arr[keep]
        val_arr = val_arr[keep]

    return rows_arr, cols_arr, sig_arr, val_arr


def _dc3d_solve_closed_form(
    mesh: TensorMesh,
    A_csr_torch: torch.Tensor,
    sigma_f64: torch.Tensor,
    b: torch.Tensor,
    *,
    dirichlet_idx: int,
) -> torch.Tensor:
    """
    Closed-form-σ-Jacobian DC3D solve.

    Builds a scipy CSR mirror of the torch CSR ``A_csr_torch`` (values
    detached, the bridge re-routes the σ-gradient via the hand-derived
    sparse Jacobian) and dispatches to
    :class:`SparseLinearSolveWithSigmaGrad`.
    """
    import numpy as np
    import scipy.sparse as sp
    from geobrain.physics.em.numerics.sparse.lu_bridge import (
        SparseLinearSolveWithSigmaGrad,
    )

    crow = A_csr_torch.crow_indices().detach().cpu().numpy().astype(np.int32)
    col = A_csr_torch.col_indices().detach().cpu().numpy().astype(np.int32)
    vals = A_csr_torch.values().detach().cpu().numpy().astype(np.float64)
    n = int(A_csr_torch.shape[0])

    A_csr_np = sp.csr_matrix((vals, col, crow), shape=(n, n))

    sigma_flat_np = sigma_f64.detach().cpu().numpy().reshape(-1).astype(np.float64)
    jac_rows, jac_cols, jac_sig, jac_vals = build_sigma_jacobian_dc3d(
        mesh, sigma_flat_np, dirichlet_idx=int(dirichlet_idx),
    )

    sigma_flat_t = sigma_f64.reshape(-1).to(torch.float64)
    phi = SparseLinearSolveWithSigmaGrad.apply(
        A_csr_np.data,
        A_csr_np.indices,
        A_csr_np.indptr,
        A_csr_np.shape,
        b,
        sigma_flat_t,
        jac_rows,
        jac_cols,
        jac_sig,
        jac_vals,
    )
    return phi
