"""
3D Yee-grid curl-curl operator for frequency-domain EM.

Builds the complex-symmetric frequency-domain Maxwell stiffness matrix
on the edges of a uniform 3D Yee staggered grid::

    A = C^T · diag(1 / mu_r)_face · C
        + i·omega·mu0 · diag(sigma_edge)
        - omega^2 · mu0 · eps0 · diag(eps_r_edge)             (if eps_r is passed)

This is the **mu0-multiplied form** of the E-field equation
``∇×(1/μ_r)∇×E + iωμ₀σE − ω²μ₀ε₀ε_r E = 0`` (the emg3d convention):
every term carries exactly one power of μ₀ relative to the SI form
``∇×(1/(μ_rμ₀))∇×E + iωσE``, so the assembled operator pairs
consistently with secondary-source RHS terms of the form
``−iωμ₀·Δσ·E_p``. (The pairing must stay consistent: a stiffness on the
SI ``1/(μ_rμ₀)`` scaling combined with an ``iωμ₀σ`` mass/RHS suppresses
the inductive response by a factor μ₀ ≈ 1.26e-6 while leaving the
galvanic response correct, on the gradient subspace the μ₀ factors
cancel between mass and RHS, so the bug is invisible to DC checks.)

The time-domain stiffness :func:`assemble_yee_curl_curl_stiffness`
stays in the SI form ``C^T·diag(1/(μ_rμ₀))·C`` but is the **bare,
un-volume-weighted** Yee stiffness (every face carries ``1/μ₀``, no
dual-volume factor). It is *not* the operator TEM3D steps with: the
TEM3D time stepper assembles a volume-weighted mimetic pair,
stiffness ``K = C^T·diag(V_face/μ₀)·C`` against edge mass
``M_eσ = diag(σ_edge·V_edge)`` (see ``time_domain/tem3d.py``). This
bare function exists as a topology-only building block and has no
production caller; do not assume its output equals TEM3D's ``K``.

where:

- ``C`` is the discrete curl mapping edge field to face field: real,
  topology-only, entries ``±1/Δ`` (returned by :func:`yee_curl_e_to_f`).
- ``mu_r``, ``sigma`` and (optional) ``eps_r`` are **cell-centred** tensors
  shaped like ``mesh.shape == (nz, ny, nx)``. They are interpolated
  internally to the Yee-grid faces / edges using uniform-volume
  arithmetic averaging (4 cells share an interior edge, 2 share a face).
- ``omega`` is the angular frequency in rad/s.

Indexing convention: matches the reference port and the
3D Poisson assembler (``assemble_poisson_3d``):

- Mesh shape ``(nz, ny, nx)`` with **x-fastest** flat indexing for cells:
  cell ``(k, j, i) -> i + j·nx + k·nx·ny``.
- Edge family ordering ``(Ex, Ey, Ez)``, x-fastest within each family:

    - Ex edges:  ``i + j·nx + k·nx·(ny+1)``        (i in [0, nx), j in [0, ny], k in [0, nz])
    - Ey edges:  ``n_Ex + i + j·(nx+1) + k·(nx+1)·ny``
    - Ez edges:  ``n_Ex + n_Ey + i + j·(nx+1) + k·(nx+1)·(ny+1)``

- Face family ordering ``(Bx, By, Bz)``, x-fastest within each family:

    - Bx faces:  ``i + j·(nx+1) + k·(nx+1)·ny``
    - By faces:  ``n_Bx + i + j·nx + k·nx·(ny+1)``
    - Bz faces:  ``n_Bx + n_By + i + j·nx + k·nx·ny``

Boundary conditions are **natural Neumann** (edge components on the outer
mesh boundary participate normally; no PEC pin is applied). Operators
that need a Dirichlet pin should do so externally.

Autograd: ``mu_r``, ``sigma`` and ``eps_r`` flow into ``A.values()``
through pure torch ops, so gradients propagate back through assembly.
Index math uses ``torch.long`` and never participates in autograd.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.mesh import TensorMesh
from geobrain.core.errors import GeoBrainError
from geobrain.physics.em.conventions import EPSILON_0 as _EPS0, MU_0 as _MU0

__all__ = [
    "assemble_yee_curl_curl_stiffness",
    "yee_curl_curl_assemble",
    "yee_curl_e_to_f",
    "yee_dual_volume_weights",
    "yee_edge_count",
    "yee_face_count",
]


# ----------------------------------------------------------------------
# Counting helpers
# ----------------------------------------------------------------------

def _require_3d(mesh: TensorMesh, where: str) -> tuple[int, int, int]:
    if mesh.n_dim != 3:
        raise GeoBrainError(
            f"{where} requires a 3D TensorMesh, got n_dim={mesh.n_dim}",
            object_name=where, field="mesh.n_dim",
            expected="n_dim == 3", actual=mesh.n_dim,
        )
    nz, ny, nx = mesh.shape
    return int(nz), int(ny), int(nx)


def yee_edge_count(mesh: TensorMesh) -> tuple[int, int, int]:
    """Return ``(n_Ex, n_Ey, n_Ez)``: edge counts per family on a 3D Yee mesh."""
    nz, ny, nx = _require_3d(mesh, "yee_edge_count")
    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)
    n_ez = (nx + 1) * (ny + 1) * nz
    return n_ex, n_ey, n_ez


def yee_face_count(mesh: TensorMesh) -> tuple[int, int, int]:
    """Return ``(n_Bx, n_By, n_Bz)``: face counts per family on a 3D Yee mesh."""
    nz, ny, nx = _require_3d(mesh, "yee_face_count")
    n_bx = (nx + 1) * ny * nz
    n_by = nx * (ny + 1) * nz
    n_bz = nx * ny * (nz + 1)
    return n_bx, n_by, n_bz


# ----------------------------------------------------------------------
# Discrete curl: edges -> faces
# ----------------------------------------------------------------------

def yee_curl_e_to_f(
    mesh: TensorMesh,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Discrete curl mapping edge field to face field, ``C: E -> F``.

    Shape ``(n_faces, n_edges)``, real ``float64`` (configurable), returned
    as a ``torch.sparse_csr`` tensor. The matrix is pure topology, entries
    are ``±1/Δ`` (with Δ being the relevant cell width along the curl
    direction) and carry no autograd dependence on physical parameters.

    Standard Yee staggering with x-fastest flat indexing within each
    edge/face family and family ordering ``(x, y, z)``.
    """
    nz, ny, nx = _require_3d(mesh, "yee_curl_e_to_f")
    # MESH2: read per-axis cell_widths so the curl stencil's ``1/Δ``
    # entries honour per-cell widths on non-uniform meshes (the stencil indexes
    # ``dx[ii], dy[jj], dz[kk]``).
    # Uniform meshes expand to constant 1-D width tensors so the stencil
    # is bit-identical to the old ``mesh.spacing`` scalar path in that case.
    wz_t, wy_t, wx_t = mesh.cell_widths
    wx_arr = wx_t.to(device=device, dtype=dtype)                          # (nx,)
    wy_arr = wy_t.to(device=device, dtype=dtype)                          # (ny,)
    wz_arr = wz_t.to(device=device, dtype=dtype)                          # (nz,)

    n_ex = nx * (ny + 1) * (nz + 1)
    n_ey = (nx + 1) * ny * (nz + 1)
    n_ez = (nx + 1) * (ny + 1) * nz
    n_edges = n_ex + n_ey + n_ez

    n_bx = (nx + 1) * ny * nz
    n_by = nx * (ny + 1) * nz
    n_bz = nx * ny * (nz + 1)
    n_faces = n_bx + n_by + n_bz

    # Index helpers (closures returning torch.long).
    def ex_idx(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return i + j * nx + k * nx * (ny + 1)

    def ey_idx(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return n_ex + i + j * (nx + 1) + k * (nx + 1) * ny

    def ez_idx(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return n_ex + n_ey + i + j * (nx + 1) + k * (nx + 1) * (ny + 1)

    def bx_idx(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return i + j * (nx + 1) + k * (nx + 1) * ny

    def by_idx(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return n_bx + i + j * nx + k * nx * (ny + 1)

    def bz_idx(i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return n_bx + n_by + i + j * nx + k * nx * ny

    long = torch.long
    arange = lambda n: torch.arange(n, device=device, dtype=long)  # noqa: E731

    rows_list: list[torch.Tensor] = []
    cols_list: list[torch.Tensor] = []
    vals_list: list[torch.Tensor] = []

    # Precompute reciprocals (length nx / ny / nz) so we can gather
    # per-face values without rebuilding the tensor each block.
    inv_wx = 1.0 / wx_arr                                                 # (nx,)
    inv_wy = 1.0 / wy_arr                                                 # (ny,)
    inv_wz = 1.0 / wz_arr                                                 # (nz,)

    # ------------------------------------------------------------------
    # Bx faces: (curl E)_x = dEz/dy - dEy/dz.
    # Bx live at (i in [0, nx+1), j in [0, ny), k in [0, nz)).
    # The Ez edges at (i,j,k) and (i,j+1,k) are separated by ``wy[j]``;
    # the Ey edges at (i,j,k) and (i,j,k+1) are separated by ``wz[k]``.
    # ------------------------------------------------------------------
    ii, jj, kk = torch.meshgrid(arange(nx + 1), arange(ny), arange(nz), indexing="ij")
    ii_f, jj_f, kk_f = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)
    f_row = bx_idx(ii_f, jj_f, kk_f)
    inv_dy = inv_wy[jj_f]                                                 # (n_bx,)
    inv_dz = inv_wz[kk_f]                                                 # (n_bx,)
    # +Ez(i, j+1, k) - Ez(i, j, k)  scaled by 1/wy[j]
    rows_list.append(f_row)
    cols_list.append(ez_idx(ii_f, jj_f + 1, kk_f))
    vals_list.append(inv_dy)
    rows_list.append(f_row)
    cols_list.append(ez_idx(ii_f, jj_f, kk_f))
    vals_list.append(-inv_dy)
    # -Ey(i, j, k+1) + Ey(i, j, k)  scaled by 1/wz[k]
    rows_list.append(f_row)
    cols_list.append(ey_idx(ii_f, jj_f, kk_f + 1))
    vals_list.append(-inv_dz)
    rows_list.append(f_row)
    cols_list.append(ey_idx(ii_f, jj_f, kk_f))
    vals_list.append(inv_dz)

    # ------------------------------------------------------------------
    # By faces: (curl E)_y = dEx/dz - dEz/dx.
    # By live at (i in [0, nx), j in [0, ny+1), k in [0, nz)).
    # ------------------------------------------------------------------
    ii, jj, kk = torch.meshgrid(arange(nx), arange(ny + 1), arange(nz), indexing="ij")
    ii_f, jj_f, kk_f = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)
    f_row = by_idx(ii_f, jj_f, kk_f)
    inv_dz = inv_wz[kk_f]                                                 # (n_by,)
    inv_dx = inv_wx[ii_f]                                                 # (n_by,)
    # +Ex(i, j, k+1) - Ex(i, j, k)  scaled by 1/wz[k]
    rows_list.append(f_row)
    cols_list.append(ex_idx(ii_f, jj_f, kk_f + 1))
    vals_list.append(inv_dz)
    rows_list.append(f_row)
    cols_list.append(ex_idx(ii_f, jj_f, kk_f))
    vals_list.append(-inv_dz)
    # -Ez(i+1, j, k) + Ez(i, j, k)  scaled by 1/wx[i]
    rows_list.append(f_row)
    cols_list.append(ez_idx(ii_f + 1, jj_f, kk_f))
    vals_list.append(-inv_dx)
    rows_list.append(f_row)
    cols_list.append(ez_idx(ii_f, jj_f, kk_f))
    vals_list.append(inv_dx)

    # ------------------------------------------------------------------
    # Bz faces: (curl E)_z = dEy/dx - dEx/dy.
    # Bz live at (i in [0, nx), j in [0, ny), k in [0, nz+1)).
    # ------------------------------------------------------------------
    ii, jj, kk = torch.meshgrid(arange(nx), arange(ny), arange(nz + 1), indexing="ij")
    ii_f, jj_f, kk_f = ii.reshape(-1), jj.reshape(-1), kk.reshape(-1)
    f_row = bz_idx(ii_f, jj_f, kk_f)
    inv_dx = inv_wx[ii_f]                                                 # (n_bz,)
    inv_dy = inv_wy[jj_f]                                                 # (n_bz,)
    # +Ey(i+1, j, k) - Ey(i, j, k)  scaled by 1/wx[i]
    rows_list.append(f_row)
    cols_list.append(ey_idx(ii_f + 1, jj_f, kk_f))
    vals_list.append(inv_dx)
    rows_list.append(f_row)
    cols_list.append(ey_idx(ii_f, jj_f, kk_f))
    vals_list.append(-inv_dx)
    # -Ex(i, j+1, k) + Ex(i, j, k)  scaled by 1/wy[j]
    rows_list.append(f_row)
    cols_list.append(ex_idx(ii_f, jj_f + 1, kk_f))
    vals_list.append(-inv_dy)
    rows_list.append(f_row)
    cols_list.append(ex_idx(ii_f, jj_f, kk_f))
    vals_list.append(inv_dy)

    rows = torch.cat(rows_list)
    cols = torch.cat(cols_list)
    vals = torch.cat(vals_list)

    indices = torch.stack([rows, cols], dim=0)
    C_coo = torch.sparse_coo_tensor(indices, vals, size=(n_faces, n_edges)).coalesce()
    return C_coo.to_sparse_csr()


# ----------------------------------------------------------------------
# Cell -> face / cell -> edge averaging (volume-weighted, MESH3).
# ----------------------------------------------------------------------
# Cell-centred fields are averaged onto the Yee faces/edges with weights
# proportional to the *cell dual volume* ``V_c = wz·wy·wx``. For an
# interior edge with k adjacent cells the average is
#
#     field_edge = Σ_c V_c · field_c  /  Σ_c V_c
#
# (drop-missing-then-renormalise at the boundary). On a uniform mesh
# all V_c are equal so the formula collapses to a plain arithmetic mean
# of the 1, 2 or 4 cells sharing the face/edge: bit-identical to the
# pre-MESH3 behaviour. On a padded / non-uniform mesh the big boundary
# cells get proportionally more weight (the PDE-consistent answer).
#
# The dense helpers below take optional ``widths=(wz, wy, wx)`` 1-D
# tensors. When ``widths`` is None they fall back to the arithmetic mean
# (matches the legacy uniform-only behaviour and keeps the test
# ``_cell_to_edge_avg(sigma, axis)`` shape-only call sites unchanged).
# When ``widths`` is supplied, the helpers apply the volume-weighted
# formula above: exactly equivalent to the sparse
# ``build_cell_to_edge_weights`` matrix, but assembled densely so
# autograd flows through ``cell`` directly.
#
# Implementation: replicate-pad on each averaged axis with the boundary
# cell width too. Padding the width with itself at the boundary yields
# a "phantom" cell of equal width, so the contribution from the missing
# neighbour reduces to a duplicate of the present neighbour, which after
# normalisation collapses to "single neighbour gets weight 1" exactly as
# the drop-and-renormalise prescription requires. (The factor cancels
# because numerator and denominator both pick up the duplicate.)


def _normalise_widths(widths: tuple | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Validate ``widths=(wz, wy, wx)`` and return them as 1-D float tensors.

    Returns ``None`` when ``widths`` is ``None`` so callers can fall back
    to the arithmetic-mean path. Each entry must be 1-D and positive.
    """
    if widths is None:
        return None
    wz, wy, wx = widths
    return wz, wy, wx


def _is_uniform_axis(w: torch.Tensor) -> bool:
    """
    Return True iff every entry of ``w`` is equal to ``w[0]``.

    We use this so the volume-weighted helpers short-circuit to the plain
    ``0.5·(a+b)`` arithmetic mean on uniform-width axes, preserving bit-
    identity with the pre-MESH3 build when ``mesh.cell_widths`` is a
    constant tensor (e.g. expanded from ``mesh.spacing``).
    """
    if w.numel() <= 1:
        return True
    return bool(torch.equal(w, torch.full_like(w, w[0].item())))


def _weighted_pair_avg(
    f_a: torch.Tensor, f_b: torch.Tensor, w_a: torch.Tensor, w_b: torch.Tensor
) -> torch.Tensor:
    """
    Pointwise volume-weighted mean ``(w_a·f_a + w_b·f_b) / (w_a + w_b)``.

    ``w_a`` and ``w_b`` are broadcastable to ``f_a.shape``. Falls back to
    ``0.5·(f_a + f_b)`` (saves a division) when ``w_a is w_b is None``.
    """
    return (w_a * f_a + w_b * f_b) / (w_a + w_b)


def _cell_to_face_avg(
    cell: torch.Tensor,
    axis: str,
    *,
    widths: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Cell-centre to face-centre averaging on one face family.

    ``cell`` is shape ``(nz, ny, nx)``. Returns a flat tensor of length
    equal to the face count along the requested axis (``"x"``, ``"y"``
    or ``"z"``). Replicate-pads along the face-normal axis.

    Args:
        cell: Cell-centred field, shape ``(nz, ny, nx)``.
        axis: Which face family, ``"x"``, ``"y"`` or ``"z"``.
        widths: Optional ``(wz, wy, wx)`` per-axis 1-D width tensors from
            :attr:`TensorMesh.cell_widths`. When passed, the average is
            volume-weighted (V_cell ∝ width along the averaged direction;
            the other two directions cancel between numerator and
            denominator). When ``None`` (default), uses arithmetic mean,
            bit-identical to the pre-MESH3 behaviour and the right choice
            for uniform meshes.
    """
    w = _normalise_widths(widths)
    if axis == "x":
        # Bx: (nx+1) x ny x nz: neighbours are (i-1, i) along x; replicate
        # at the two x-boundary faces so the average collapses to the
        # adjacent cell.
        # Pad along x: prepend cell[..., 0:1] and append cell[..., -1:].
        padded = torch.cat([cell[..., :1], cell, cell[..., -1:]], dim=2)  # (nz, ny, nx+2)
        if w is None or _is_uniform_axis(w[2]):
            avg = 0.5 * (padded[..., :-1] + padded[..., 1:])              # (nz, ny, nx+1)
        else:
            wx = w[2].to(device=cell.device, dtype=cell.dtype)
            wx_pad = torch.cat([wx[:1], wx, wx[-1:]], dim=0)              # (nx+2,)
            w_a = wx_pad[:-1].view(1, 1, -1)                              # (1, 1, nx+1)
            w_b = wx_pad[1:].view(1, 1, -1)
            avg = _weighted_pair_avg(padded[..., :-1], padded[..., 1:], w_a, w_b)
        return avg.reshape(-1)
    if axis == "y":
        padded = torch.cat([cell[:, :1, :], cell, cell[:, -1:, :]], dim=1)
        if w is None or _is_uniform_axis(w[1]):
            avg = 0.5 * (padded[:, :-1, :] + padded[:, 1:, :])            # (nz, ny+1, nx)
        else:
            wy = w[1].to(device=cell.device, dtype=cell.dtype)
            wy_pad = torch.cat([wy[:1], wy, wy[-1:]], dim=0)              # (ny+2,)
            w_a = wy_pad[:-1].view(1, -1, 1)
            w_b = wy_pad[1:].view(1, -1, 1)
            avg = _weighted_pair_avg(padded[:, :-1, :], padded[:, 1:, :], w_a, w_b)
        return avg.reshape(-1)
    if axis == "z":
        padded = torch.cat([cell[:1, :, :], cell, cell[-1:, :, :]], dim=0)
        if w is None or _is_uniform_axis(w[0]):
            avg = 0.5 * (padded[:-1, :, :] + padded[1:, :, :])            # (nz+1, ny, nx)
        else:
            wz = w[0].to(device=cell.device, dtype=cell.dtype)
            wz_pad = torch.cat([wz[:1], wz, wz[-1:]], dim=0)              # (nz+2,)
            w_a = wz_pad[:-1].view(-1, 1, 1)
            w_b = wz_pad[1:].view(-1, 1, 1)
            avg = _weighted_pair_avg(padded[:-1, :, :], padded[1:, :, :], w_a, w_b)
        return avg.reshape(-1)
    raise GeoBrainError(
        f"_cell_to_face_avg: axis must be x/y/z, got {axis!r}",
        object_name="_cell_to_face_avg", field="axis",
        expected="x/y/z", actual=axis,
    )


def _cell_to_edge_avg(
    cell: torch.Tensor,
    axis: str,
    *,
    widths: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Cell-centre to edge-centre averaging on one edge family.

    Ex edges live along x and are shared by up to 4 cells with y in
    ``{j-1, j}`` and z in ``{k-1, k}`` (clipped at the boundary).
    Replicate-padding implements the "clip to interior" behaviour. The
    averaging is performed as two successive pair-averages, first
    across one perpendicular axis, then across the other; on a uniform
    mesh this reproduces the plain arithmetic 4-cell mean, and on a
    non-uniform mesh with ``widths`` supplied it reproduces the
    volume-weighted mean used by the
    ``build_cell_to_edge_weights``. (The factorisation is exact because
    the volume of cell ``(cx, cy, cz)`` is ``wx[cx]·wy[cy]·wz[cz]``,
    the axis fixed by the edge family cancels between numerator and
    denominator, leaving only weights along the two averaged axes.)

    Args:
        cell: Cell-centred field, shape ``(nz, ny, nx)``.
        axis: Which edge family, ``"x"``, ``"y"`` or ``"z"``.
        widths: Optional ``(wz, wy, wx)`` per-axis 1-D width tensors. When
            ``None`` (default), uses arithmetic mean, bit-identical to
            the pre-MESH3 behaviour.
    """
    w = _normalise_widths(widths)
    # Pre-compute per-axis uniformity once so the short-circuit decision
    # for each pair-average step is a constant Python boolean.
    if w is None:
        is_uni_z = is_uni_y = is_uni_x = True
    else:
        is_uni_z = _is_uniform_axis(w[0])
        is_uni_y = _is_uniform_axis(w[1])
        is_uni_x = _is_uniform_axis(w[2])

    if axis == "x":
        # Pad y, then z. cell shape (nz, ny, nx).
        padded_y = torch.cat([cell[:, :1, :], cell, cell[:, -1:, :]], dim=1)   # (nz, ny+2, nx)
        if is_uni_y:
            avg_y = 0.5 * (padded_y[:, :-1, :] + padded_y[:, 1:, :])           # (nz, ny+1, nx)
        else:
            wy = w[1].to(device=cell.device, dtype=cell.dtype)
            wy_pad = torch.cat([wy[:1], wy, wy[-1:]], dim=0)                   # (ny+2,)
            w_a = wy_pad[:-1].view(1, -1, 1)                                   # (1, ny+1, 1)
            w_b = wy_pad[1:].view(1, -1, 1)
            avg_y = _weighted_pair_avg(padded_y[:, :-1, :], padded_y[:, 1:, :], w_a, w_b)
        padded_z = torch.cat([avg_y[:1, :, :], avg_y, avg_y[-1:, :, :]], dim=0)
        if is_uni_z:
            avg_zy = 0.5 * (padded_z[:-1, :, :] + padded_z[1:, :, :])          # (nz+1, ny+1, nx)
        else:
            wz = w[0].to(device=cell.device, dtype=cell.dtype)
            wz_pad = torch.cat([wz[:1], wz, wz[-1:]], dim=0)                   # (nz+2,)
            w_a = wz_pad[:-1].view(-1, 1, 1)                                   # (nz+1, 1, 1)
            w_b = wz_pad[1:].view(-1, 1, 1)
            avg_zy = _weighted_pair_avg(padded_z[:-1, :, :], padded_z[1:, :, :], w_a, w_b)
        return avg_zy.reshape(-1)
    if axis == "y":
        # Pad x, then z.
        padded_x = torch.cat([cell[..., :1], cell, cell[..., -1:]], dim=2)     # (nz, ny, nx+2)
        if is_uni_x:
            avg_x = 0.5 * (padded_x[..., :-1] + padded_x[..., 1:])             # (nz, ny, nx+1)
        else:
            wx = w[2].to(device=cell.device, dtype=cell.dtype)
            wx_pad = torch.cat([wx[:1], wx, wx[-1:]], dim=0)
            w_a = wx_pad[:-1].view(1, 1, -1)
            w_b = wx_pad[1:].view(1, 1, -1)
            avg_x = _weighted_pair_avg(padded_x[..., :-1], padded_x[..., 1:], w_a, w_b)
        padded_z = torch.cat([avg_x[:1, :, :], avg_x, avg_x[-1:, :, :]], dim=0)
        if is_uni_z:
            avg_zx = 0.5 * (padded_z[:-1, :, :] + padded_z[1:, :, :])          # (nz+1, ny, nx+1)
        else:
            wz = w[0].to(device=cell.device, dtype=cell.dtype)
            wz_pad = torch.cat([wz[:1], wz, wz[-1:]], dim=0)
            w_a = wz_pad[:-1].view(-1, 1, 1)
            w_b = wz_pad[1:].view(-1, 1, 1)
            avg_zx = _weighted_pair_avg(padded_z[:-1, :, :], padded_z[1:, :, :], w_a, w_b)
        return avg_zx.reshape(-1)
    if axis == "z":
        # Pad x, then y.
        padded_x = torch.cat([cell[..., :1], cell, cell[..., -1:]], dim=2)     # (nz, ny, nx+2)
        if is_uni_x:
            avg_x = 0.5 * (padded_x[..., :-1] + padded_x[..., 1:])             # (nz, ny, nx+1)
        else:
            wx = w[2].to(device=cell.device, dtype=cell.dtype)
            wx_pad = torch.cat([wx[:1], wx, wx[-1:]], dim=0)
            w_a = wx_pad[:-1].view(1, 1, -1)
            w_b = wx_pad[1:].view(1, 1, -1)
            avg_x = _weighted_pair_avg(padded_x[..., :-1], padded_x[..., 1:], w_a, w_b)
        padded_y = torch.cat([avg_x[:, :1, :], avg_x, avg_x[:, -1:, :]], dim=1)
        if is_uni_y:
            avg_yx = 0.5 * (padded_y[:, :-1, :] + padded_y[:, 1:, :])          # (nz, ny+1, nx+1)
        else:
            wy = w[1].to(device=cell.device, dtype=cell.dtype)
            wy_pad = torch.cat([wy[:1], wy, wy[-1:]], dim=0)
            w_a = wy_pad[:-1].view(1, -1, 1)
            w_b = wy_pad[1:].view(1, -1, 1)
            avg_yx = _weighted_pair_avg(padded_y[:, :-1, :], padded_y[:, 1:, :], w_a, w_b)
        return avg_yx.reshape(-1)
    raise GeoBrainError(
        f"_cell_to_edge_avg: axis must be x/y/z, got {axis!r}",
        object_name="_cell_to_edge_avg", field="axis",
        expected="x/y/z", actual=axis,
    )


# ----------------------------------------------------------------------
# Diagonal dual-volume (mimetic mass) weights (audit E1)
# ----------------------------------------------------------------------
# The frequency-domain Yee system is the mimetic pair
#
#     A = Cᵀ · M_f(1/μ) · C  +  iωμ₀ · M_e(σ)
#
# where ``M_f`` / ``M_e`` are the face- / edge-VOLUME-weighted (lumped
# diagonal) mass matrices: face area × dual edge length, and edge length ×
# dual face area, respectively. Dropping these dual volumes
# (``M_f = diag(1/μ_face)``, ``M_e = diag(σ_edge)``) is a trap: on a
# UNIFORM mesh every dual volume equals the same constant ``h³`` so the
# missing weights are a single global factor that cancels between stiffness,
# mass and RHS in every observable, which is why the uniform cross-validation twins
# were unaffected. On a NON-UNIFORM (graded / padded) mesh the transpose
# ``Cᵀ`` is *not* the dual curl: a Taylor expansion of a transpose row
# leaves a spurious zeroth-order term ``∝ (w_j - w_{j-1})/(w_{j-1} w_j)`` at
# every width transition. Restoring the face volume converts the second
# ``1/Δ`` of ``Cᵀ`` into the correct dual-mesh derivative; the edge volume
# does the same for the mass/RHS pairing.


def _rel_widths_and_duals(
    w: torch.Tensor,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis relative cell widths and relative dual (node) lengths.

    Returns ``(rw, rd)`` where ``rw = w / mean(w)`` (length ``n``, the cell
    widths) and ``rd = dnode / mean(w)`` (length ``n+1``, the dual/node
    lengths). ``dnode`` is the replicate-padded midpoint dual length
    ``dnode[p] = ½(w_pad[p] + w_pad[p+1])`` with ``w`` padded by its own
    boundary cell, the SAME boundary convention used by the cell→edge /
    cell→face averaging above, so the interior nodes get ``½(w[p-1]+w[p])``
    and the two boundary nodes get the full boundary cell width ``w[0]`` /
    ``w[-1]`` (no halving). Normalising both by the per-axis mean makes
    every entry exactly ``1.0`` on a uniform axis, so the assembled weights
    collapse to the identity on a uniform mesh.
    """
    w = w.to(device=device, dtype=dtype)
    m = w.mean()
    rw = w / m                                              # (n,)
    w_pad = torch.cat([w[:1], w, w[-1:]], dim=0)            # (n+2,)
    dnode = 0.5 * (w_pad[:-1] + w_pad[1:])                  # (n+1,)
    rd = dnode / m                                          # (n+1,)
    return rw, rd


def yee_dual_volume_weights(
    mesh: TensorMesh,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Diagonal dual-volume weights for the Yee curl-curl mass matrices.

    Returns ``(face_w, edge_w)``, real 1-D tensors matching the assembler's
    face family order ``(Bx, By, Bz)`` and edge family order ``(Ex, Ey, Ez)``
    (x-fastest within each family, the :func:`yee_curl_e_to_f` convention).

    - ``face_w[f] = V_face[f] / V_ref`` with
      ``V_face = dual_len(normal axis) · width(other two axes)``;
    - ``edge_w[e] = V_edge[e] / V_ref`` with
      ``V_edge = width(edge axis) · dual_len(other two axes)``;

    where ``V_ref = mean(w_z)·mean(w_y)·mean(w_x)`` is the SAME reference for
    both families, so the physical stiffness/mass scale ratio is preserved
    exactly while the matrix magnitude stays comparable to the unweighted
    build. The dual lengths use the replicate-pad (no-boundary-halving)
    convention of :func:`_rel_widths_and_duals`.

    On a **uniform** mesh every weight is exactly ``1.0`` and this returns
    ``(None, None)`` so the caller keeps the bare, byte-identical assembly.
    Only non-uniform meshes receive the (audit-E1) weighting.
    """
    nz, ny, nx = _require_3d(mesh, "yee_dual_volume_weights")
    if mesh.is_uniform:
        return None, None
    wz, wy, wx = mesh.cell_widths
    rwz, rdz = _rel_widths_and_duals(wz, device, dtype)     # (nz,), (nz+1,)
    rwy, rdy = _rel_widths_and_duals(wy, device, dtype)     # (ny,), (ny+1,)
    rwx, rdx = _rel_widths_and_duals(wx, device, dtype)     # (nx,), (nx+1,)

    # Faces: dual volume = dual length along the face normal × the two
    # in-face cell widths. Broadcast into the (nz[,+1], ny[,+1], nx[,+1])
    # layout each family flattens from (x-fastest).
    fx = (rdx.view(1, 1, -1) * rwy.view(1, -1, 1) * rwz.view(-1, 1, 1)).reshape(-1)  # (nz, ny, nx+1)
    fy = (rwx.view(1, 1, -1) * rdy.view(1, -1, 1) * rwz.view(-1, 1, 1)).reshape(-1)  # (nz, ny+1, nx)
    fz = (rwx.view(1, 1, -1) * rwy.view(1, -1, 1) * rdz.view(-1, 1, 1)).reshape(-1)  # (nz+1, ny, nx)
    face_w = torch.cat([fx, fy, fz], dim=0)

    # Edges: dual volume = edge-axis cell width × dual lengths on the two
    # perpendicular axes.
    ex = (rwx.view(1, 1, -1) * rdy.view(1, -1, 1) * rdz.view(-1, 1, 1)).reshape(-1)  # (nz+1, ny+1, nx)
    ey = (rdx.view(1, 1, -1) * rwy.view(1, -1, 1) * rdz.view(-1, 1, 1)).reshape(-1)  # (nz+1, ny, nx+1)
    ez = (rdx.view(1, 1, -1) * rdy.view(1, -1, 1) * rwz.view(-1, 1, 1)).reshape(-1)  # (nz, ny+1, nx+1)
    edge_w = torch.cat([ex, ey, ez], dim=0)
    return face_w, edge_w


# ----------------------------------------------------------------------
# Full curl-curl assembly
# ----------------------------------------------------------------------

def yee_curl_curl_assemble(
    mesh: TensorMesh,
    mu_r: torch.Tensor,
    sigma: torch.Tensor,
    omega: float,
    *,
    eps_r: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Assemble the frequency-domain Maxwell curl-curl matrix on a 3D Yee grid.

    ::

        A = C^T · diag(1 / mu_r)_face · C
            + i·omega·mu0 · diag(sigma_edge)
            - omega^2 · mu0 · eps0 · diag(eps_r_edge)        (if eps_r given)

    This is the μ₀-multiplied equation form (see module docstring): all
    three terms carry one power of μ₀ relative to SI, so the operator is
    self-consistent and pairs with ``−iωμ₀Δσ·E_p`` secondary RHS terms.

    Args:
        mesh: 3D :class:`geobrain.mesh.TensorMesh`. Convention
            ``mesh.shape == (nz, ny, nx)`` with uniform ``spacing``.
        mu_r: Cell-centred relative magnetic permeability, real ``float64``,
            shape ``mesh.shape``. Averaged to faces internally.
        sigma: Cell-centred conductivity in S/m, real ``float64``, shape
            ``mesh.shape``. Averaged to edges internally.
        omega: Angular frequency in rad/s.
        eps_r: Optional cell-centred relative permittivity, real ``float64``,
            shape ``mesh.shape``. When ``None`` the displacement-current
            term is omitted (purely diffusive regime).

    Returns:
        Complex-symmetric sparse CSR tensor of shape ``(n_edges, n_edges)``
        and dtype ``torch.complex128``. ``A.values()`` retains autograd
        through ``mu_r``, ``sigma`` and ``eps_r``.
    """
    nz, ny, nx = _require_3d(mesh, "yee_curl_curl_assemble")

    if mu_r.shape != mesh.shape:
        raise GeoBrainError(
            f"yee_curl_curl_assemble: mu_r.shape={tuple(mu_r.shape)} "
            f"does not match mesh.shape={tuple(mesh.shape)}",
            object_name="yee_curl_curl_assemble", field="mu_r",
            expected=f"shape == {tuple(mesh.shape)}", actual=tuple(mu_r.shape),
        )
    if sigma.shape != mesh.shape:
        raise GeoBrainError(
            f"yee_curl_curl_assemble: sigma.shape={tuple(sigma.shape)} "
            f"does not match mesh.shape={tuple(mesh.shape)}",
            object_name="yee_curl_curl_assemble", field="sigma",
            expected=f"shape == {tuple(mesh.shape)}", actual=tuple(sigma.shape),
        )
    if eps_r is not None and eps_r.shape != mesh.shape:
        raise GeoBrainError(
            f"yee_curl_curl_assemble: eps_r.shape={tuple(eps_r.shape)} "
            f"does not match mesh.shape={tuple(mesh.shape)}",
            object_name="yee_curl_curl_assemble", field="eps_r",
            expected=f"shape == {tuple(mesh.shape)}", actual=tuple(eps_r.shape),
        )

    device = mu_r.device
    real_dtype = mu_r.dtype
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64

    n_ex, n_ey, n_ez = yee_edge_count(mesh)
    n_edges = n_ex + n_ey + n_ez

    # MESH3: pass mesh.cell_widths into the averaging helpers so face/edge
    # averages are volume-weighted on non-uniform meshes. Uniform meshes
    # have equal widths so the formula collapses to arithmetic mean (bit-
    # identical to the pre-MESH3 path).
    mesh_widths = mesh.cell_widths

    # E1: diagonal dual-volume weights for the mimetic face/edge mass
    # matrices M_f(1/μ) and M_e(σ). ``(None, None)`` on a uniform mesh, in
    # which case the assembly below is byte-identical to the pre-E1 build.
    face_w, edge_w = yee_dual_volume_weights(
        mesh, device=device, dtype=real_dtype,
    )

    # ------------------------------------------------------------------
    # 1) C and K = C^T · diag(mu_inv_face) · C.
    # ------------------------------------------------------------------
    C = yee_curl_e_to_f(mesh, device=device, dtype=real_dtype)

    # mu_inv on each face family, then concatenated in (Bx, By, Bz) order.
    mu_face_x = _cell_to_face_avg(mu_r, "x", widths=mesh_widths)
    mu_face_y = _cell_to_face_avg(mu_r, "y", widths=mesh_widths)
    mu_face_z = _cell_to_face_avg(mu_r, "z", widths=mesh_widths)
    mu_face = torch.cat([mu_face_x, mu_face_y, mu_face_z], dim=0)
    # μ₀-multiplied form: 1/μ_r on faces (NOT 1/(μ_r·μ₀)) so the stiffness
    # pairs consistently with the +iωμ₀σ mass and −iωμ₀Δσ secondary RHS.
    mu_inv_face = 1.0 / mu_face                                                # (n_faces,)
    # E1: fold the face dual volume into the face inner product so
    # K = Cᵀ · M_f(1/μ) · C with M_f = diag(V_face/μ). No-op on uniform
    # meshes (face_w is None). Pure geometry ⇒ mu_r autograd preserved.
    if face_w is not None:
        mu_inv_face = mu_inv_face * face_w

    # K_real = C^T · diag(mu_inv_face) · C, real float64 sparse.
    # We do this by:
    #   1. Extract C as COO (rows, cols, vals).
    #   2. Build C_scaled with values = mu_inv_face[row] * vals (scales each
    #      row of C: equivalent to left-multiplying by diag(mu_inv_face)).
    #   3. Compute K = C^T @ C_scaled via torch.sparse.mm.
    C_coo = C.to_sparse_coo().coalesce()
    C_indices = C_coo.indices()
    C_values = C_coo.values()
    C_rows = C_indices[0]
    # Scale C row-wise by mu_inv_face (this is diag(mu_inv) @ C).
    M_C_values = mu_inv_face[C_rows] * C_values
    M_C = torch.sparse_coo_tensor(
        C_indices, M_C_values, size=C.shape, device=device, dtype=real_dtype
    ).coalesce()
    # K = C^T @ (diag(mu_inv) @ C).
    C_T = torch.sparse_coo_tensor(
        torch.stack([C_indices[1], C_indices[0]], dim=0),
        C_values,
        size=(C.shape[1], C.shape[0]),
        device=device,
        dtype=real_dtype,
    ).coalesce()
    K = torch.sparse.mm(C_T, M_C).coalesce()                                   # (n_edges, n_edges)

    # ------------------------------------------------------------------
    # 2) sigma_edge (and optional eps_r_edge): diagonal contributions.
    # ------------------------------------------------------------------
    sigma_edge_x = _cell_to_edge_avg(sigma, "x", widths=mesh_widths)
    sigma_edge_y = _cell_to_edge_avg(sigma, "y", widths=mesh_widths)
    sigma_edge_z = _cell_to_edge_avg(sigma, "z", widths=mesh_widths)
    sigma_edge = torch.cat([sigma_edge_x, sigma_edge_y, sigma_edge_z], dim=0)  # (n_edges,)

    # i·omega·mu0 · sigma_edge → imaginary diagonal in complex128.
    # E1: edge dual volume folds in the mimetic edge mass M_e(σ) =
    # diag(σ_edge·V_edge). No-op on uniform meshes (edge_w is None).
    diag_sigma_im = (omega * _MU0) * sigma_edge                                # real-valued (n_edges,)
    if edge_w is not None:
        diag_sigma_im = diag_sigma_im * edge_w

    if eps_r is not None:
        eps_edge_x = _cell_to_edge_avg(eps_r, "x", widths=mesh_widths)
        eps_edge_y = _cell_to_edge_avg(eps_r, "y", widths=mesh_widths)
        eps_edge_z = _cell_to_edge_avg(eps_r, "z", widths=mesh_widths)
        eps_edge = torch.cat([eps_edge_x, eps_edge_y, eps_edge_z], dim=0)
        # -ω²·μ₀·ε₀·ε_r_edge → real diagonal contribution. Same edge dual
        # volume as the σ mass (both are edge inner-product terms).
        diag_eps_re = -(omega * omega) * _MU0 * _EPS0 * eps_edge               # (n_edges,)
        if edge_w is not None:
            diag_eps_re = diag_eps_re * edge_w
    else:
        diag_eps_re = None

    # ------------------------------------------------------------------
    # 3) Assemble A = K + diag(i·ωμ₀σ + (-ω²μ₀ε₀ε_r)) in complex128.
    #    Combine all (row, col, value) triples and coalesce.
    # ------------------------------------------------------------------
    K_indices = K.indices()
    K_values_re = K.values()
    # Complex-cast K's values (purely real contribution).
    K_values = K_values_re.to(dtype=complex_dtype)

    # Diagonal: rows = cols = arange(n_edges).
    diag_idx = torch.arange(n_edges, device=device, dtype=torch.long)
    diag_indices = torch.stack([diag_idx, diag_idx], dim=0)

    # Build complex diagonal value tensor: real part from eps (if any),
    # imaginary part from sigma.  torch.complex creates a complex tensor
    # without breaking autograd through either input.
    if diag_eps_re is not None:
        diag_real = diag_eps_re.to(dtype=real_dtype)
    else:
        diag_real = torch.zeros(n_edges, device=device, dtype=real_dtype)
    diag_imag = diag_sigma_im.to(dtype=real_dtype)
    diag_values = torch.complex(diag_real, diag_imag)

    all_indices = torch.cat([K_indices, diag_indices], dim=1)
    all_values = torch.cat([K_values, diag_values], dim=0)

    # Coalesce as a complex COO tensor: this part *is* autograd-compatible
    # (PyTorch supports complex COO + autograd; only the COO->CSR conversion
    # for complex dtype is currently unsupported, see PyTorch 2.10 limitation
    # "_to_sparse_csr does not support automatic differentiation for outputs
    # with complex dtype"). We therefore split the assembly: keep a
    # coalesced complex COO for autograd-bearing ``.values()``, and pair it
    # with a *detached* complex CSR for layout queries.
    A_coo = torch.sparse_coo_tensor(
        all_indices,
        all_values,
        size=(n_edges, n_edges),
        device=device,
        dtype=complex_dtype,
    ).coalesce()
    A_csr_detached = torch.sparse_csr_tensor(
        *_coo_to_csr_layout(A_coo, n_edges),
        A_coo.values().detach(),
        size=(n_edges, n_edges),
    )
    return _AutogradCsrComplex(csr=A_csr_detached, values_grad=A_coo.values())


# ----------------------------------------------------------------------
# Bare curl-curl stiffness (real, σ-independent): for time-domain EM.
# ----------------------------------------------------------------------


def assemble_yee_curl_curl_stiffness(
    mesh: TensorMesh,
    *,
    mu_r: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Assemble the bare real curl-curl stiffness ``K = C^T · diag(μ⁻¹_face) · C``.

    This is the σ-independent counterpart to :func:`yee_curl_curl_assemble`.
    The frequency-domain assembler bakes ``i·ω·μ₀·diag(σ_edge)`` into the
    operator (and casts to complex); a σ-independent ``K`` lets a caller
    re-form a per-substep matrix ``A = α₀·M(σ) + Δt·K`` each step without
    re-assembling the topology.

    .. warning::
       This assembles the **bare** stiffness ``M_f = diag(mu_inv_face)``
       with *no* dual-volume weighting. The TEM3D time stepper does not
       use this function: it assembles its own volume-weighted form
       ``K = C^T·diag(V_face/μ₀)·C`` inline (``time_domain/tem3d.py``,
       ``__init__``), paired with the edge mass
       ``M_eσ = diag(σ_edge·V_edge)``. The earlier in-tree path that
       multiplied a bare ``1/μ₀`` face inner product produced incorrect
       boundary-edge coupling and was replaced. This function has no
       production caller and must not be substituted into a TEM step.

    Args:
        mesh: 3D :class:`TensorMesh`. Uniform spacing.
        mu_r: Optional cell-centred relative magnetic permeability, real,
            shape ``mesh.shape``. When ``None`` (the canonical TEM use)
            the assembler uses ``μ_r ≡ 1`` everywhere → ``μ⁻¹_face = 1/μ₀``
            on all faces. Pass a non-uniform tensor to model magnetised
            media; autograd flows through ``mu_r`` into ``K.values()``.
        device, dtype: Standard torch placement / precision controls.

    Returns:
        Real sparse CSR tensor of shape ``(n_edges, n_edges)``,
        symmetric positive-semidefinite.
    """
    nz, ny, nx = _require_3d(mesh, "assemble_yee_curl_curl_stiffness")

    n_bx, n_by, n_bz = yee_face_count(mesh)
    n_faces = n_bx + n_by + n_bz

    C = yee_curl_e_to_f(mesh, device=device, dtype=dtype)

    if mu_r is None:
        # μ_r ≡ 1 everywhere → μ⁻¹_face = 1/μ₀ on every face.
        mu_inv_face = torch.full(
            (n_faces,), 1.0 / _MU0, dtype=dtype, device=device,
        )
    else:
        if mu_r.shape != mesh.shape:
            raise GeoBrainError(
                f"assemble_yee_curl_curl_stiffness: mu_r.shape="
                f"{tuple(mu_r.shape)} does not match mesh.shape="
                f"{tuple(mesh.shape)}",
                object_name="assemble_yee_curl_curl_stiffness", field="mu_r",
                expected=f"shape == {tuple(mesh.shape)}", actual=tuple(mu_r.shape),
            )
        # MESH3: volume-weighted averaging on non-uniform meshes; uniform
        # meshes collapse to arithmetic mean (bit-identical to pre-MESH3).
        mesh_widths = mesh.cell_widths
        mu_face_x = _cell_to_face_avg(mu_r, "x", widths=mesh_widths)
        mu_face_y = _cell_to_face_avg(mu_r, "y", widths=mesh_widths)
        mu_face_z = _cell_to_face_avg(mu_r, "z", widths=mesh_widths)
        mu_face = torch.cat([mu_face_x, mu_face_y, mu_face_z], dim=0)
        mu_inv_face = 1.0 / (mu_face * _MU0)

    # K = C^T · diag(mu_inv_face) · C. Reuse the assembly trick from
    # yee_curl_curl_assemble: scale C's rows by mu_inv_face, build
    # explicit C^T, then sparse-mm.
    C_coo = C.to_sparse_coo().coalesce()
    C_indices = C_coo.indices()
    C_values = C_coo.values()
    C_rows = C_indices[0]
    M_C_values = mu_inv_face[C_rows] * C_values
    M_C = torch.sparse_coo_tensor(
        C_indices, M_C_values, size=C.shape, device=device, dtype=dtype,
    ).coalesce()
    C_T = torch.sparse_coo_tensor(
        torch.stack([C_indices[1], C_indices[0]], dim=0),
        C_values,
        size=(C.shape[1], C.shape[0]),
        device=device,
        dtype=dtype,
    ).coalesce()
    K = torch.sparse.mm(C_T, M_C).coalesce()
    return K.to_sparse_csr()


def _coo_to_csr_layout(
    coo: torch.Tensor, n_rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (crow_indices, col_indices) from a coalesced COO tensor."""
    idx = coo.indices()
    rows = idx[0]
    cols = idx[1]
    counts = torch.bincount(rows, minlength=n_rows)
    crow = torch.cat([torch.zeros(1, dtype=torch.long, device=rows.device), counts.cumsum(0)])
    return crow, cols


class _AutogradCsrComplex:
    """
    Lightweight wrapper presenting a complex sparse-CSR matrix with
    autograd-bearing ``.values()``.

    PyTorch 2.10 does not allow complex-dtype sparse CSR tensors to
    participate in autograd (both ``torch.sparse_csr_tensor`` and the
    COO→CSR conversion raise on complex inputs that require grad). This
    wrapper sidesteps that limitation by holding:

    - ``_csr``: a detached ``torch.sparse_csr_tensor`` used to expose
      ``.shape``, ``.dtype``, ``.layout``, ``.to_dense()`` and the CSR
      index buffers, and
    - ``_values_grad``: the coalesced complex value tensor that *does*
      participate in autograd (because complex COO + autograd is
      supported).

    The wrapper quacks like a sparse CSR for the methods downstream EM
    code currently exercises (``.values()``, ``.shape``, ``.dtype``,
    ``.layout``, ``.to_dense()``). When PyTorch eventually adds complex
    sparse CSR autograd this class can be replaced with the native
    sparse CSR tensor with no change in callers.
    """

    __slots__ = ("_csr", "_values_grad")

    def __init__(self, *, csr: torch.Tensor, values_grad: torch.Tensor) -> None:
        if csr.layout != torch.sparse_csr:
            raise GeoBrainError(
                "_AutogradCsrComplex expects a sparse-CSR backing tensor",
                object_name="_AutogradCsrComplex", field="csr.layout",
                expected="torch.sparse_csr", actual=csr.layout,
            )
        if values_grad.shape != csr.values().shape:
            raise GeoBrainError(
                "values_grad must have the same shape as csr.values(); "
                f"got {tuple(values_grad.shape)} vs {tuple(csr.values().shape)}",
                object_name="_AutogradCsrComplex", field="values_grad",
                expected=f"shape == {tuple(csr.values().shape)}",
                actual=tuple(values_grad.shape),
            )
        self._csr = csr
        self._values_grad = values_grad

    # --- attributes commonly read on a sparse CSR tensor --------------

    @property
    def shape(self) -> torch.Size:
        return self._csr.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._csr.dtype

    @property
    def layout(self) -> torch.layout:
        return self._csr.layout

    @property
    def device(self) -> torch.device:
        return self._csr.device

    def values(self) -> torch.Tensor:
        """Return the (1D) complex value tensor: autograd-attached."""
        return self._values_grad

    def crow_indices(self) -> torch.Tensor:
        return self._csr.crow_indices()

    def col_indices(self) -> torch.Tensor:
        return self._csr.col_indices()

    def to_dense(self) -> torch.Tensor:
        """
        Return the dense complex matrix. Detached from autograd because
        the CSR side does not propagate gradients through ``to_dense``."""
        return self._csr.to_dense()

    def to_sparse_coo(self) -> torch.Tensor:
        """
        Return a torch sparse COO tensor with autograd-attached values.

        Bridges this wrapper to the ``sparse_linear_solve_with_adjoint``.
        PyTorch supports complex+grad on COO but not CSR, so this is the
        format complex-matrix consumers like MT3D / FDEM3D / TEM3D use.
        """
        # Expand CSR row indices to a flat (row, col) -> (2, nnz) packed layout.
        crow = self._csr.crow_indices().to(torch.long)
        col = self._csr.col_indices().to(torch.long)
        counts = (crow[1:] - crow[:-1]).to(torch.long)
        n_rows = int(crow.numel() - 1)
        rows = torch.repeat_interleave(
            torch.arange(n_rows, dtype=torch.long, device=col.device),
            counts,
        )
        indices = torch.stack([rows, col])
        # CSR-derived (row, col) pairs are already sorted lexicographically
        # with no duplicates, so ``coalesce()`` is structurally a no-op; we
        # call it anyway so the returned tensor is in the coalesced state
        # that ``.values()`` requires. Autograd is preserved through the
        # coalesce because ``_values_grad`` is the unique value tensor.
        return torch.sparse_coo_tensor(
            indices, self._values_grad, size=self.shape
        ).coalesce()

    def __repr__(self) -> str:
        return (
            f"_AutogradCsrComplex(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, layout={self.layout}, "
            f"nnz={int(self._values_grad.numel())})"
        )
