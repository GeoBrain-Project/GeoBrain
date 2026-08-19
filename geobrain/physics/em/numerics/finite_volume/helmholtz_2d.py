"""
Discretised complex Helmholtz operators on 2-D finite-volume meshes.

``assemble_helmholtz_2d`` (TE) and ``assemble_helmholtz_2d_tm`` (TM)
share the same dense 5-point cell-centred stencil. The two modes differ
in how σ enters the operator:

- **TE** (E-field parallel to strike): σ is the *reaction* coefficient::

      ∂²u/∂z² + ∂²u/∂y² + i·ω·μ₀·σ · u = 0

  σ enters the diagonal only.

- **TM** (H-field parallel to strike): σ enters the *diffusion* term as
  ``1/σ`` evaluated on faces (harmonic mean of adjacent cells)::

      ∂/∂z[(1/σ) ∂u/∂z] + ∂/∂y[(1/σ) ∂u/∂y] + i·ω·μ₀ · u = 0

  σ shows up in every face-coupling stencil entry.

Both modes share boundary conventions (matching the MT2D):

- top row (``i = 0``) and bottom row (``i = nz − 1``) hold identity
  rows, Dirichlet boundary; the caller supplies the prescribed value
  via the RHS;
- left/right (``j = 0`` and ``j = ny − 1``) use reflective Neumann via
  index-clamped neighbour couplings; ``scatter_add`` folds the clamped
  self-coupling onto the diagonal so the reflective BC emerges
  naturally without a separate boundary loop.

Memory is dense ``O(N²)`` so practical grids stay ≲ 40 × 40. A sparse
cell-centred FV variant for larger / unstructured 2-D meshes lives in
:mod:`~geobrain.physics.em.numerics.finite_volume.helmholtz_fv`
(``assemble_helmholtz_fv`` / ``assemble_helmholtz_fv_tm``), which MT2D
dispatches to for non-structured ``ConnectivityMesh`` inputs.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.mesh import TensorMesh
from geobrain.core.errors import GeoBrainError
from geobrain.physics.em.conventions import MU_0


def assemble_helmholtz_2d(
    mesh: TensorMesh,
    sigma: torch.Tensor,
    omega: float,
    *,
    mu0: float = MU_0,
) -> torch.Tensor:
    """
    Assemble the dense complex ``(N, N)`` 2D Helmholtz operator.

    Discretises ``∂²u/∂z² + ∂²u/∂y² + i·ω·μ₀·σ · u`` on the supplied
    ``TensorMesh`` of shape ``(nz, ny)`` with cell-centred finite
    volumes. Top + bottom rows of the returned matrix are identity
    rows (Dirichlet, caller specifies via the RHS); y-sides use
    reflective Neumann via index-clamped scatter.

    Args:
        mesh:  ``TensorMesh`` with shape ``(nz, ny)``.
        sigma: real cell-centre conductivity tensor, same shape as the
            mesh; cast internally to ``torch.complex64``.
        omega: angular frequency ``2·π·f`` in rad / s.
        mu0:   magnetic permeability of free space (default SI value);
            override only for unit-testing scaling laws.

    Returns:
        Complex ``(N, N)`` dense matrix where ``N = nz · ny``. Autograd
        flows through ``sigma``.
    """
    nz, ny = sigma.shape
    n = nz * ny
    device = sigma.device
    dtype = torch.complex64

    dz, dy = mesh.spacing

    sigma_c = sigma.to(dtype)
    i_omega_mu = torch.tensor(1j * omega * mu0, dtype=dtype, device=device)
    inv_dz2 = torch.tensor(1.0 / (dz * dz), dtype=dtype, device=device)
    inv_dy2 = torch.tensor(1.0 / (dy * dy), dtype=dtype, device=device)

    # --- Interior cells (i in [1, nz-2], j in [0, ny-1]) ---
    #
    # Per cell: diagonal ``-2/dz² - 2/dy² + i·ω·μ₀·σ`` plus four
    # off-diagonal ``+1/dz²`` or ``+1/dy²`` entries to the four
    # neighbours. Side cells (j = 0 or j = ny−1) clamp the y-neighbour
    # to themselves; ``scatter_add`` will fold the clamped contribution
    # onto the diagonal so the reflective Neumann BC emerges naturally.
    ii, jj = torch.meshgrid(
        torch.arange(1, nz - 1, device=device),
        torch.arange(ny, device=device),
        indexing="ij",
    )
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    k = ii * ny + jj                                       # (n_int,)
    n_int = k.numel()

    diag_vals = -2.0 * inv_dz2 - 2.0 * inv_dy2 + i_omega_mu * sigma_c[ii, jj]

    k_up = (ii - 1) * ny + jj
    k_dn = (ii + 1) * ny + jj
    j_l = (jj - 1).clamp(min=0)
    j_r = (jj + 1).clamp(max=ny - 1)
    k_l = ii * ny + j_l
    k_r = ii * ny + j_r

    full_inv_dz2 = inv_dz2.expand(n_int)
    full_inv_dy2 = inv_dy2.expand(n_int)

    rows = torch.cat([k, k, k, k, k])
    cols = torch.cat([k, k_up, k_dn, k_l, k_r])
    vals = torch.cat([diag_vals, full_inv_dz2, full_inv_dz2, full_inv_dy2, full_inv_dy2])

    # --- Top + bottom: identity rows for Dirichlet plane-wave BCs ---
    top_idx = torch.arange(ny, device=device, dtype=torch.long)
    bot_idx = (nz - 1) * ny + top_idx
    bdy_idx = torch.cat([top_idx, bot_idx])
    rows = torch.cat([rows, bdy_idx])
    cols = torch.cat([cols, bdy_idx])
    vals = torch.cat([vals, torch.ones(2 * ny, dtype=dtype, device=device)])

    flat_idx = rows * n + cols
    A_flat = torch.zeros(n * n, dtype=dtype, device=device).scatter_add(0, flat_idx, vals)
    return A_flat.view(n, n)


def _harmonic_face_2d(
    sigma: torch.Tensor, axis: int, eps: float = 1e-30,
) -> torch.Tensor:
    """
    Harmonic mean of adjacent cell σ along ``axis``.

    For TM-mode discretisation the face diffusivity is ``1/σ_face`` where
    ``σ_face = 2 σ_l σ_r / (σ_l + σ_r)``. We return the σ_face values; the
    caller takes the reciprocal at the stencil-stamp step.

    Args:
        sigma: ``(nz, ny)`` cell-centred σ tensor.
        axis:  ``0`` for z-direction (face between (i, j) and (i+1, j))
            or ``1`` for y-direction (face between (i, j) and (i, j+1)).
        eps:   small constant to avoid division-by-zero at σ ≈ 0.

    Returns:
        ``(nz - 1, ny)`` (axis=0) or ``(nz, ny - 1)`` (axis=1).
    """
    if axis == 0:
        s_l = sigma[:-1, :]
        s_r = sigma[1:, :]
    elif axis == 1:
        s_l = sigma[:, :-1]
        s_r = sigma[:, 1:]
    else:
        raise GeoBrainError(
            f"axis must be 0 or 1, got {axis}",
            object_name="_harmonic_face_2d", field="axis",
            expected="0 or 1", actual=axis,
        )
    return 2.0 * s_l * s_r / (s_l + s_r + eps)


def assemble_helmholtz_2d_tm(
    mesh: "TensorMesh",
    sigma: torch.Tensor,
    omega: float,
    *,
    mu0: float = MU_0,
) -> torch.Tensor:
    """
    Assemble the dense complex ``(N, N)`` 2-D TM-mode Helmholtz operator.

    Discretises ``∂/∂z[(1/σ) ∂u/∂z] + ∂/∂y[(1/σ) ∂u/∂y] + i·ω·μ₀ u`` on a
    cell-centred FV mesh with **harmonic-mean** face conductivities. The
    discrete stencil at cell ``(i, j)`` carries

        diag += -(d_zN + d_zS + d_yE + d_yW) + i·ω·μ₀
        A[k_(i,j), k_(i±1,j)] += d_z_face_(i∓½,j)
        A[k_(i,j), k_(i,j±1)] += d_y_face_(i,j∓½)

    where ``d_z_face = (1/σ_face_z) / dz²`` (and analogously for y), and
    ``σ_face`` is the harmonic mean of adjacent cell σ values. Boundary
    conventions match :func:`assemble_helmholtz_2d`: top + bottom rows
    are identity (Dirichlet via RHS); y-sides use index-clamped
    reflective Neumann, which here collapses the would-be off-mesh face
    flux onto the boundary cell's diagonal.

    Note that TM's σ-dependence is **off-diagonal** (the face term),
    not diagonal: a closed-form σ-Jacobian would need the harmonic-mean
    derivative ``dT/dσ_l = -1/(2 σ_l²) · 1/dist²``. There is no in-tree
    hand-derived TM Jacobian; MT2D materialises the full Jacobian via the
    autograd wrapper ``compute_jacobian`` in
    :mod:`~geobrain.physics.em.natural_source.mt2d` instead.

    Args:
        mesh:  ``TensorMesh`` of shape ``(nz, ny)``.
        sigma: real cell-centre σ tensor, same shape as the mesh.
        omega: angular frequency ``2·π·f`` in rad/s.
        mu0:   magnetic permeability of free space.

    Returns:
        Complex ``(N, N)`` dense matrix where ``N = nz · ny``. Autograd
        flows through ``sigma``.
    """
    nz, ny = sigma.shape
    n = nz * ny
    device = sigma.device
    dtype = torch.complex64

    dz, dy = mesh.spacing

    sigma_c = sigma.to(dtype)
    i_omega_mu = torch.tensor(1j * omega * mu0, dtype=dtype, device=device)
    inv_dz2 = 1.0 / (dz * dz)
    inv_dy2 = 1.0 / (dy * dy)

    # Per-face σ via harmonic mean (autograd through ``sigma``).
    sigma_face_z = _harmonic_face_2d(sigma_c, axis=0)  # (nz-1, ny)
    sigma_face_y = _harmonic_face_2d(sigma_c, axis=1)  # (nz, ny-1)
    inv_sigma_face_z = 1.0 / (sigma_face_z + 0)        # (nz-1, ny)
    inv_sigma_face_y = 1.0 / (sigma_face_y + 0)        # (nz, ny-1)

    # Per-cell transmissibility contributions on each face.
    #   d_z_face_k_(i+0.5, j) sits between cell (i, j) and (i+1, j) for
    #   i = 0..nz-2; d_y_face analogously.
    d_z_face = inv_sigma_face_z * inv_dz2           # (nz-1, ny)
    d_y_face = inv_sigma_face_y * inv_dy2           # (nz, ny-1)

    # Interior cells (i in [1, nz-2], j in [0, ny-1]).
    ii, jj = torch.meshgrid(
        torch.arange(1, nz - 1, device=device),
        torch.arange(ny, device=device),
        indexing="ij",
    )
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    k = ii * ny + jj

    # Z-faces around cell (i, j):  face above sits at index (i-1, j) of
    # ``d_z_face``; face below at (i, j). Both always exist for i in
    # [1, nz-2].
    d_z_above = d_z_face[ii - 1, jj]
    d_z_below = d_z_face[ii, jj]
    # Y-faces around cell (i, j): left face at (i, j-1), right face at
    # (i, j) of ``d_y_face``. Reflective Neumann at the y-sides: at j=0
    # the left face is missing, at j=ny-1 the right face is missing,
    # we substitute zero (no flux out of the boundary).
    zero_col = torch.zeros(nz, 1, dtype=dtype, device=device)
    if ny > 1:
        d_y_left_full = torch.cat([zero_col, d_y_face], dim=1)   # (nz, ny)
        d_y_right_full = torch.cat([d_y_face, zero_col], dim=1)  # (nz, ny)
    else:
        d_y_left_full = zero_col
        d_y_right_full = zero_col
    d_y_left = d_y_left_full[ii, jj]
    d_y_right = d_y_right_full[ii, jj]

    diag_vals = (
        -(d_z_above + d_z_below + d_y_left + d_y_right) + i_omega_mu
    )

    k_up = (ii - 1) * ny + jj
    k_dn = (ii + 1) * ny + jj
    j_l = (jj - 1).clamp(min=0)
    j_r = (jj + 1).clamp(max=ny - 1)
    k_l = ii * ny + j_l
    k_r = ii * ny + j_r

    rows = torch.cat([k, k, k, k, k])
    cols = torch.cat([k, k_up, k_dn, k_l, k_r])
    vals = torch.cat([diag_vals, d_z_above, d_z_below, d_y_left, d_y_right])

    # Top + bottom: identity rows for Dirichlet plane-wave BCs.
    top_idx = torch.arange(ny, device=device, dtype=torch.long)
    bot_idx = (nz - 1) * ny + top_idx
    bdy_idx = torch.cat([top_idx, bot_idx])
    rows = torch.cat([rows, bdy_idx])
    cols = torch.cat([cols, bdy_idx])
    vals = torch.cat([vals, torch.ones(2 * ny, dtype=dtype, device=device)])

    flat_idx = rows * n + cols
    A_flat = torch.zeros(n * n, dtype=dtype, device=device).scatter_add(
        0, flat_idx, vals,
    )
    return A_flat.view(n, n)


# ---------------------------------------------------------------------------
# Graded (non-uniform) sparse assembly
# ---------------------------------------------------------------------------
#
# A broadband MT mesh must be fine near the surface (to resolve the
# high-frequency skin depth) AND deep (to host the low-frequency penetration),
# a graded TensorMesh (``cell_widths``). The dense uniform stencils above can do
# neither cheaply (O(N²) memory) nor at all (uniform-only). These two functions
# generalise the SAME cell-centred 5-point scheme to per-cell widths and return
# a **sparse** complex128 COO operator (solved through the implicit-VJP sparse
# adjoint seam). The only change is the geometric face transmissibility:
# ``1/(w_i · h_face)`` with ``h_face`` the cell-CENTRE distance across the face,
# which reduces to ``1/dz²`` (``1/dy²``) on a uniform mesh, so a constant-width
# graded mesh reproduces the dense path's matrix exactly (up to dtype). The
# reflective Neumann y-sides emerge by simply omitting the missing boundary-face
# term (equivalent to the dense path's clamp-and-fold), and the top/bottom rows
# stay identity (Dirichlet plane-wave closure).


def _graded_interior(mesh, sigma):
    """Shared graded-stencil scaffolding: interior (ii, jj, k) index arrays and
    the per-cell geometric face transmissibilities (real, float64)."""
    nz, ny = sigma.shape
    device = sigma.device
    wz, wy = mesh.cell_widths
    wz = wz.to(device=device, dtype=torch.float64)
    wy = wy.to(device=device, dtype=torch.float64)
    hz = 0.5 * (wz[:-1] + wz[1:])                      # (nz-1,) z face centre-dist
    hy = 0.5 * (wy[:-1] + wy[1:])                      # (ny-1,) y face centre-dist

    ii, jj = torch.meshgrid(
        torch.arange(1, nz - 1, device=device),
        torch.arange(ny, device=device),
        indexing="ij",
    )
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)
    k = ii * ny + jj
    has_l = jj > 0
    has_r = jj < ny - 1
    # NEIGHBOUR CELL indices (clamp to self on a Neumann side: the paired
    # transmissibility is 0 there, so the self-entry adds nothing).
    jl_cell = (jj - 1).clamp(min=0)                   # left neighbour cell
    jr_cell = (jj + 1).clamp(max=ny - 1)              # right neighbour cell
    # FACE indices into hz/hy (and the harmonic σ-faces): the face to the LEFT
    # of cell j is index j-1, to the RIGHT is index j (≠ the neighbour-cell id).
    fl = (jj - 1).clamp(min=0)                        # left  y-face (j|j-1)
    fr = jj.clamp(max=ny - 2)                         # right y-face (j|j+1)
    zero = torch.zeros_like(wz[ii])
    # Geometric transmissibility 1/(w_cell · h_face); Neumann ⇒ 0 on a side
    # with no outward face (the dropped term reproduces the dense clamp-fold).
    gz_up = 1.0 / (wz[ii] * hz[ii - 1])               # cell (i,j) → (i-1,j)
    gz_dn = 1.0 / (wz[ii] * hz[ii])                   # cell (i,j) → (i+1,j)
    gy_l = torch.where(has_l, 1.0 / (wy[jj] * hy[fl]), zero)   # → (i,j-1)
    gy_r = torch.where(has_r, 1.0 / (wy[jj] * hy[fr]), zero)   # → (i,j+1)
    idx = dict(
        ii=ii, jj=jj, k=k, fl=fl, fr=fr, ny=ny, nz=nz,
        k_up=(ii - 1) * ny + jj, k_dn=(ii + 1) * ny + jj,
        k_l=ii * ny + jl_cell, k_r=ii * ny + jr_cell,
    )
    return idx, gz_up, gz_dn, gy_l, gy_r


def _graded_coo(idx, diag, v_up, v_dn, v_l, v_r, *, dtype, device):
    """Stamp interior + identity-boundary entries into a coalesced sparse COO."""
    nz, ny = idx["nz"], idx["ny"]
    n = nz * ny
    k = idx["k"]
    rows = torch.cat([k, k, k, k, k])
    cols = torch.cat([k, idx["k_up"], idx["k_dn"], idx["k_l"], idx["k_r"]])
    vals = torch.cat([diag, v_up, v_dn, v_l, v_r])
    top = torch.arange(ny, device=device, dtype=torch.long)
    bot = (nz - 1) * ny + top
    bdy = torch.cat([top, bot])
    rows = torch.cat([rows, bdy])
    cols = torch.cat([cols, bdy])
    vals = torch.cat([vals, torch.ones(2 * ny, dtype=dtype, device=device)])
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (n, n)
    ).coalesce()


def assemble_helmholtz_2d_graded(
    mesh: "TensorMesh", sigma: torch.Tensor, omega: float, *, mu0: float = MU_0,
) -> torch.Tensor:
    """Sparse complex128 ``(N, N)`` non-uniform **TE** Helmholtz operator.

    Graded-mesh analogue of :func:`assemble_helmholtz_2d` (σ on the diagonal,
    unit-diffusivity Laplacian). A uniform-width graded mesh reproduces the
    dense path's matrix. Autograd flows through ``sigma``."""
    dtype = torch.complex128
    device = sigma.device
    idx, gz_up, gz_dn, gy_l, gy_r = _graded_interior(mesh, sigma)
    i_omega_mu = torch.tensor(1j * omega * mu0, dtype=dtype, device=device)
    sigma_c = sigma.to(dtype)
    diag = (-(gz_up + gz_dn + gy_l + gy_r)).to(dtype) + i_omega_mu * sigma_c[idx["ii"], idx["jj"]]
    return _graded_coo(
        idx, diag, gz_up.to(dtype), gz_dn.to(dtype), gy_l.to(dtype), gy_r.to(dtype),
        dtype=dtype, device=device,
    )


def assemble_helmholtz_2d_tm_graded(
    mesh: "TensorMesh", sigma: torch.Tensor, omega: float, *, mu0: float = MU_0,
) -> torch.Tensor:
    """Sparse complex128 ``(N, N)`` non-uniform **TM** Helmholtz operator.

    Graded-mesh analogue of :func:`assemble_helmholtz_2d_tm`: harmonic-mean face
    conductivity (σ off-diagonal) times the non-uniform geometric
    transmissibility. A uniform-width graded mesh reproduces the dense path.
    Autograd flows through ``sigma`` (via the harmonic face means)."""
    dtype = torch.complex128
    device = sigma.device
    idx, gz_up, gz_dn, gy_l, gy_r = _graded_interior(mesh, sigma)
    ii, jj, fl, fr = idx["ii"], idx["jj"], idx["fl"], idx["fr"]
    sigma_c = sigma.to(dtype)
    inv_sf_z = 1.0 / _harmonic_face_2d(sigma_c, axis=0)   # (nz-1, ny), face i|i+1
    inv_sf_y = 1.0 / _harmonic_face_2d(sigma_c, axis=1)   # (nz, ny-1), face j|j+1
    # Diffusivity-weighted couplings: 1/σ_face × geometric transmissibility.
    # ``fl``/``fr`` index the harmonic σ-FACES (j-1|j and j|j+1); gy_l/gy_r are
    # 0 on a Neumann side so the clamped face value contributes nothing there.
    d_up = inv_sf_z[ii - 1, jj] * gz_up.to(dtype)         # face above cell (i,j)
    d_dn = inv_sf_z[ii, jj] * gz_dn.to(dtype)             # face below
    d_l = inv_sf_y[ii, fl] * gy_l.to(dtype)               # left face (0 on side)
    d_r = inv_sf_y[ii, fr] * gy_r.to(dtype)               # right face (0 on side)
    i_omega_mu = torch.tensor(1j * omega * mu0, dtype=dtype, device=device)
    diag = -(d_up + d_dn + d_l + d_r) + i_omega_mu
    return _graded_coo(idx, diag, d_up, d_dn, d_l, d_r, dtype=dtype, device=device)
