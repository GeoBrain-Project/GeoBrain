"""
Discretised Poisson/DC operators on finite-volume meshes.

``assemble_poisson_2d`` is the cell-centred 5-point stencil with
harmonic-mean face conductivities, lifted from the original DC2D
operator. ``assemble_poisson_3d`` is still a stub.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.mesh import TensorMesh
from geobrain.core.errors import GeoBrainError


def _harmonic(
    a: torch.Tensor, b: torch.Tensor, eps: float = 1e-30, floor: float = 1e-30,
) -> torch.Tensor:
    """Harmonic mean of face conductivities, preserving autograd through both.

    Conductivity must be positive. An unconstrained inversion can drive σ
    negative, which would make the harmonic mean, and hence the assembled
    operator ``A``, indefinite and silently corrupt the DC solve with no
    error. Clamp both arguments to a tiny positive floor so ``A`` stays
    positive semi-definite: a non-physical negative σ becomes a (near-)
    non-conductive face rather than a sign flip. The floor sits far below any
    physical conductivity, so valid models, and their gradients; are
    unaffected (clamp passes the gradient through wherever σ > floor).
    """
    a = a.clamp(min=floor)
    b = b.clamp(min=floor)
    return 2.0 * a * b / (a + b + eps)


def assemble_poisson_2d(
    mesh: TensorMesh,
    sigma: torch.Tensor,
    *,
    bc: str = "neumann",
) -> torch.Tensor:
    """
    Discretised −∇·(σ∇) on a 2D mesh, dense torch tensor.

    Cell-centred finite-volume 5-point stencil with harmonic-mean face
    conductivities::

        σ_face = 2 σ_l σ_r / (σ_l + σ_r)

    Each interior face contributes a symmetric ``±T = ±σ_face / spacing²``
    block to ``A``. With pure Neumann boundaries (the default and only
    supported BC) the resulting operator has a one-dimensional
    nullspace spanned by the constant vector; downstream callers (e.g.
    DC2D) are responsible for removing it (e.g. via a Dirichlet pin).

    Args:
        mesh:  a ``Mesh`` (``TensorMesh``) with shape ``(nz, nx)``.
        sigma: cell-centre conductivity tensor, same shape as the mesh.
        bc:    boundary condition; currently only ``"neumann"`` supported.

    Returns:
        ``A`` such that ``A @ V = rhs`` is the discretised Poisson system.
    """
    if bc != "neumann":
        raise NotImplementedError(
            f"assemble_poisson_2d: bc={bc!r}, only 'neumann' supported"
        )

    nz, nx = mesh.shape
    dz, dx = mesh.spacing
    n = nz * nx
    device, dtype = sigma.device, sigma.dtype

    # Harmonic-mean face conductivities.
    sig_xf = _harmonic(sigma[:, :-1], sigma[:, 1:])      # (nz, nx-1)
    sig_zf = _harmonic(sigma[:-1, :], sigma[1:, :])      # (nz-1, nx)

    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    vals: list[torch.Tensor] = []

    # --- x-direction (interior vertical faces between (i, j) and (i, j+1)) ---
    z_idx = torch.arange(nz, device=device)
    for j in range(nx - 1):
        k_l = z_idx * nx + j
        k_r = z_idx * nx + (j + 1)
        t = sig_xf[:, j] / (dx ** 2)
        rows.append(k_l)
        cols.append(k_r)
        vals.append(-t)
        rows.append(k_r)
        cols.append(k_l)
        vals.append(-t)
        rows.append(k_l)
        cols.append(k_l)
        vals.append(t)
        rows.append(k_r)
        cols.append(k_r)
        vals.append(t)

    # --- z-direction (interior horizontal faces between (i, j) and (i+1, j)) ---
    x_idx = torch.arange(nx, device=device)
    for i in range(nz - 1):
        k_t = i * nx + x_idx
        k_b = (i + 1) * nx + x_idx
        t = sig_zf[i, :] / (dz ** 2)
        rows.append(k_t)
        cols.append(k_b)
        vals.append(-t)
        rows.append(k_b)
        cols.append(k_t)
        vals.append(-t)
        rows.append(k_t)
        cols.append(k_t)
        vals.append(t)
        rows.append(k_b)
        cols.append(k_b)
        vals.append(t)

    # Free surface at i = 0: pure Neumann, no contribution.
    # Far boundaries: pure Neumann (no flux out). Removing the resulting
    # constant-vector nullspace (e.g. with a Dirichlet pin) is the caller's
    # responsibility.

    rows_t = torch.cat(rows) if rows else torch.zeros(0, device=device, dtype=torch.long)
    cols_t = torch.cat(cols) if cols else torch.zeros(0, device=device, dtype=torch.long)
    vals_t = torch.cat(vals) if vals else torch.zeros(0, device=device, dtype=dtype)

    flat_idx = rows_t * n + cols_t
    A_flat = torch.zeros(n * n, dtype=dtype, device=device).scatter_add(0, flat_idx, vals_t)
    return A_flat.view(n, n)


def assemble_poisson_3d(
    mesh: TensorMesh,
    sigma: torch.Tensor,
    *,
    bc: str = "neumann",
) -> torch.Tensor:
    """
    Discretised −∇·(σ∇) on a 3D mesh, sparse output.

    Cell-centred finite-volume 7-point stencil with harmonic-mean face
    conductivities (ported from the ``poisson_3d_assemble``)::

        T_f = 2 · σ_l · σ_r · A_face / (σ_l · d_r + σ_r · d_l)

    Each interior face contributes the symmetric block::

        A[l, l] += T;  A[r, r] += T;  A[l, r] -= T;  A[r, l] -= T

    With pure Neumann boundaries (the default and only supported BC) the operator has a one-dimensional nullspace spanned by
    the constant vector; downstream callers (e.g. DC3D) are responsible
    for removing it via a Dirichlet pin.

    Args:
        mesh:  a :class:`TensorMesh` with ``shape = (nz, nx, ny)``
            and ``spacing = (dz, dx, dy)`` (platform axis convention;
            the assembler is layout-generic and reads the shape/widths
            per-axis, so it adapts to any axis labelling).
        sigma: cell-centre conductivity tensor, same shape as the mesh.
            Must be 3-D. May be ``float64`` (DC/IP path) or ``complex128``
            (SIP path with σ_eff(ω) = σ·(1 − η(ω))).
        bc:    boundary condition; currently only ``"neumann"`` supported.

    Returns:
        ``A`` as a sparse torch tensor of shape ``(n, n)`` where
        ``n = nz · ny · nx``.

        - Real σ → ``torch.sparse_csr`` (DC3D path, backward-compatible).
        - Complex σ → ``torch.sparse_coo`` (SIP path). PyTorch 2.10 does
          not support autograd through ``to_sparse_csr`` on complex
          tensors, so the SIP branch stops at COO; downstream solvers
          must accept (or coalesce) the COO format.

        ``A.values()`` is a torch op chain on ``sigma`` so autograd flows
        back through assembly in both branches.

    Raises:
        ValueError: if ``sigma.ndim != 3`` or ``sigma.shape`` does not
            match ``mesh.shape``.
        NotImplementedError: for any ``bc`` other than ``"neumann"``.
    """
    if bc != "neumann":
        raise NotImplementedError(
            f"assemble_poisson_3d: bc={bc!r}, only 'neumann' supported"
        )

    if sigma.ndim != 3:
        raise GeoBrainError(
            f"assemble_poisson_3d: sigma must be 3-D, got ndim={sigma.ndim}",
            object_name="assemble_poisson_3d", field="sigma",
            expected="ndim == 3", actual=sigma.ndim,
        )
    if tuple(sigma.shape) != tuple(mesh.shape):
        raise GeoBrainError(
            f"assemble_poisson_3d: sigma.shape={tuple(sigma.shape)} does not "
            f"match mesh.shape={tuple(mesh.shape)}",
            object_name="assemble_poisson_3d", field="sigma",
            expected=f"shape == {tuple(mesh.shape)}", actual=tuple(sigma.shape),
        )
    if mesh.n_dim != 3:
        raise GeoBrainError(
            f"assemble_poisson_3d: mesh must be 3-D, got n_dim={mesh.n_dim}",
            object_name="assemble_poisson_3d", field="mesh.n_dim",
            expected="n_dim == 3", actual=mesh.n_dim,
        )

    nz, ny, nx = mesh.shape
    # MESH2: read per-axis cell_widths so the harmonic-mean transmissibility
    # ``T = 2·σ_l·σ_r·A_face / (σ_l·d_r + σ_r·d_l)`` honours per-cell widths
    # on non-uniform meshes (matches ``poisson_3d_assemble``). Uniform
    # meshes expand to constant 1-D width tensors so this code path is
    # bit-identical to the old ``mesh.spacing`` formula in that case.
    wz_t, wy_t, wx_t = mesh.cell_widths
    # Widths are real even when σ is complex (SIP). Multiplication with
    # a complex tensor will then promote the result to complex automatically.
    if sigma.is_complex():
        real_dtype = torch.float64 if sigma.dtype == torch.complex128 else torch.float32
    else:
        real_dtype = sigma.dtype
    wz = wz_t.to(device=sigma.device, dtype=real_dtype)
    wy = wy_t.to(device=sigma.device, dtype=real_dtype)
    wx = wx_t.to(device=sigma.device, dtype=real_dtype)
    n = nz * ny * nx
    device, dtype = sigma.device, sigma.dtype

    # ------------------------------------------------------------------
    # Harmonic-mean transmissibilities, axis by axis. The unpack names
    # (nz, ny, nx) are internal labels; under the platform ``(nz, nx, ny)``
    # convention the last axis is y, so the flat index of (k, j, i) is
    # ``i + j * nx + k * nx * ny`` = ``iy + ix*ny + iz*nx*ny`` (last-axis
    # fastest). The math is layout-generic either way.
    # ------------------------------------------------------------------
    eps = 1e-30

    rows_list: list[torch.Tensor] = []
    cols_list: list[torch.Tensor] = []
    vals_list: list[torch.Tensor] = []

    # --- x-faces (interior i between (k, j, i) and (k, j, i+1)) ---
    if nx > 1:
        s_l = sigma[:, :, :-1]                                # (nz, ny, nx-1)
        s_r = sigma[:, :, 1:]
        # Face area in x-normal: dy[j] · dz[k]; broadcast to (nz, ny, nx-1).
        area = (wz.view(nz, 1, 1) * wy.view(1, ny, 1)).expand(nz, ny, nx - 1)
        # Half-cell distances on each side of face between i and i+1.
        d_l = wx[:-1].view(1, 1, -1).expand(nz, ny, nx - 1)
        d_r = wx[1:].view(1, 1, -1).expand(nz, ny, nx - 1)
        t = 2.0 * s_l * s_r * area / (s_l * d_r + s_r * d_l + eps)
        # Flat row/col indices (no autograd needed for index math).
        kk, jj, ii = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.long),
            torch.arange(ny, device=device, dtype=torch.long),
            torch.arange(nx - 1, device=device, dtype=torch.long),
            indexing="ij",
        )
        left = (ii + jj * nx + kk * nx * ny).reshape(-1)
        right = left + 1
        t_flat = t.reshape(-1)
        rows_list.append(torch.cat([left, right, left, right]))
        cols_list.append(torch.cat([right, left, left, right]))
        vals_list.append(torch.cat([-t_flat, -t_flat, t_flat, t_flat]))

    # --- y-faces (interior j between (k, j, i) and (k, j+1, i)) ---
    if ny > 1:
        s_l = sigma[:, :-1, :]                                # (nz, ny-1, nx)
        s_r = sigma[:, 1:, :]
        area = (wz.view(nz, 1, 1) * wx.view(1, 1, nx)).expand(nz, ny - 1, nx)
        d_l = wy[:-1].view(1, -1, 1).expand(nz, ny - 1, nx)
        d_r = wy[1:].view(1, -1, 1).expand(nz, ny - 1, nx)
        t = 2.0 * s_l * s_r * area / (s_l * d_r + s_r * d_l + eps)
        kk, jj, ii = torch.meshgrid(
            torch.arange(nz, device=device, dtype=torch.long),
            torch.arange(ny - 1, device=device, dtype=torch.long),
            torch.arange(nx, device=device, dtype=torch.long),
            indexing="ij",
        )
        left = (ii + jj * nx + kk * nx * ny).reshape(-1)
        right = left + nx
        t_flat = t.reshape(-1)
        rows_list.append(torch.cat([left, right, left, right]))
        cols_list.append(torch.cat([right, left, left, right]))
        vals_list.append(torch.cat([-t_flat, -t_flat, t_flat, t_flat]))

    # --- z-faces (interior k between (k, j, i) and (k+1, j, i)) ---
    if nz > 1:
        s_l = sigma[:-1, :, :]                                # (nz-1, ny, nx)
        s_r = sigma[1:, :, :]
        area = (wy.view(1, ny, 1) * wx.view(1, 1, nx)).expand(nz - 1, ny, nx)
        d_l = wz[:-1].view(-1, 1, 1).expand(nz - 1, ny, nx)
        d_r = wz[1:].view(-1, 1, 1).expand(nz - 1, ny, nx)
        t = 2.0 * s_l * s_r * area / (s_l * d_r + s_r * d_l + eps)
        kk, jj, ii = torch.meshgrid(
            torch.arange(nz - 1, device=device, dtype=torch.long),
            torch.arange(ny, device=device, dtype=torch.long),
            torch.arange(nx, device=device, dtype=torch.long),
            indexing="ij",
        )
        left = (ii + jj * nx + kk * nx * ny).reshape(-1)
        right = left + nx * ny
        t_flat = t.reshape(-1)
        rows_list.append(torch.cat([left, right, left, right]))
        cols_list.append(torch.cat([right, left, left, right]))
        vals_list.append(torch.cat([-t_flat, -t_flat, t_flat, t_flat]))

    if rows_list:
        rows = torch.cat(rows_list)
        cols = torch.cat(cols_list)
        vals = torch.cat(vals_list)
    else:
        rows = torch.zeros(0, device=device, dtype=torch.long)
        cols = torch.zeros(0, device=device, dtype=torch.long)
        vals = torch.zeros(0, device=device, dtype=dtype)

    # Build the COO tensor, coalesce duplicate (row, col) entries (the
    # diagonal accumulates contributions from up to 6 faces). autograd
    # flows through ``vals`` end-to-end.
    #
    # Real σ (DC/IP) → convert to CSR for the downstream sparse_linear_solve
    # backward-compat path. Complex σ (SIP) → stop at COO: PyTorch 2.10's
    # ``to_sparse_csr`` raises in autograd for complex outputs, so we keep
    # the COO format and let the SIP solver consume it directly.
    indices = torch.stack([rows, cols], dim=0)
    A_coo = torch.sparse_coo_tensor(indices, vals, size=(n, n)).coalesce()
    if sigma.is_complex():
        return A_coo
    return A_coo.to_sparse_csr()
