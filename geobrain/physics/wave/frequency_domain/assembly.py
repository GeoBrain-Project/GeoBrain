"""
Sparse assembly for the 2-D acoustic Helmholtz equation.

This module owns matrix and right-hand-side construction only. Solving and the
implicit adjoint live in sibling modules so assembly can be verified against
independent continuous and discrete oracles.

- :func:`build_helmholtz_2d_coo`: assembles the 2-D Helmholtz system matrix as a
  *torch* complex128 sparse COO tensor. Three boundary treatments are supported,
  selectable through the ``boundary`` / ``n_pml`` arguments:

  1. **Sommerfeld first-order ABC** (``boundary='sommerfeld'``, ``n_pml=0``), the
     default. The diagonal of the four boundary rows gets ``-iω/(v·h)``.
  2. **Dirichlet damping ring** (``boundary='sommerfeld'`` or ``'dirichlet'`` with
     ``n_pml > 0``). Adds an exponentially tapered imaginary slowness inside the
     outer ``n_pml`` cells, a thin, cheap PML imitator.
  3. **Split-field PML, Berenger / complex coordinate stretching**
     (``boundary='pml'``, ``pml_thickness > 0``). Implements the frequency-domain
     version of Berenger's split-field PML using complex coordinate stretching::

         s_x(x) = 1 - i·σ_x(x)/ω      σ_x = σ_max · (d/L_pml)^p

     The 5-point Laplacian stencil is modified so each face derivative gets divided
     by the appropriate ``s_x`` (or ``s_z``). At reasonable thickness (~10-20 cells)
     the residual boundary reflection drops by 20-40 dB relative to Sommerfeld
     alone.

System-matrix sign convention::

    A u = s,  with A = -L_h - ω² · diag(1/v²)

so the off-diagonal stencil entries are ``-1/(s_x · h²)`` (PML) or ``-1/h²`` (no
PML), and the diagonal is the sum of the off-diagonals with sign flipped plus
``-ω²/v²``.

Time convention: ``exp(+iωt)`` (engineering). Its outgoing branch is
``H_0^(2)``; the matching complex stretch is ``s_x = 1 - iσ_x/ω`` with
``σ_x >= 0``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from geobrain.core import GeoBrainError

import math

import torch


def _flat_index(iz: torch.Tensor, ix: torch.Tensor, nx: int) -> torch.Tensor:
    return iz * nx + ix


def _pml_sigma_1d(
    n: int,
    n_pml: int,
    h: float,
    v_ref: torch.Tensor,
    p_order: float,
    target_reflection: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return ``(sigma_cell, sigma_face)`` 1-D PML damping profiles.

    ``sigma_cell[i]`` lives at cell centre ``i``; ``sigma_face[i]``
    lives at face ``i+1/2`` (between cell ``i`` and ``i+1``).

    Both arrays are zero in the interior and grow polynomially toward
    the boundary inside the outer ``n_pml`` cells.  Standard formula::

        σ_max = -(p+1) · v_ref · log(R) / (2 · L_pml)
        σ(ξ)  = σ_max · ξ^p    where ξ = (n_pml - d) / n_pml ∈ [0, 1]

    with ``d`` the cell-index distance to the boundary (0 at the
    outermost row, ``n_pml`` at the inner edge of the PML).  The
    polynomial order ``p`` is typically 2-4.
    """
    if n_pml <= 0:
        zeros = torch.zeros(n, dtype=torch.float64, device=device)
        return zeros, zeros.clone()

    L_pml = n_pml * h
    # Standard CPML d_max (Roden & Gedney 2000; Komatitsch & Martin 2007).
    # The (p+1) prefactor gives a slightly stronger damper than the
    # legacy (3/2) value when p=2 and is the form used in the
    # geophysics frequency-domain literature.
    sigma_max = -(p_order + 1.0) * v_ref * math.log(target_reflection) / (2.0 * L_pml)

    # ------------------------------------------------------------------
    # Cell-centred sigma at integer indices i = 0, 1, ..., n-1.
    # Distance-to-boundary in cells, clipped to n_pml so the interior
    # gets zero (n_pml - n_pml = 0).
    # ------------------------------------------------------------------
    idx = torch.arange(n, dtype=torch.float64, device=device)
    d_cell = torch.minimum(idx, (n - 1) - idx).clamp_max(n_pml)
    xi_cell = ((n_pml - d_cell) / float(n_pml)).clamp_min(0.0)
    sigma_cell = sigma_max * xi_cell.pow(p_order)

    # ------------------------------------------------------------------
    # Face-centred sigma at half-integer indices i + 1/2 for i = 0, ..., n-2.
    # The face between cells i and i+1 sits at i + 0.5 in cell-index
    # coordinates; its distance to the nearer boundary is
    # min(i+0.5, n-1.5-i).
    # ------------------------------------------------------------------
    face_idx = torch.arange(n - 1, dtype=torch.float64, device=device) + 0.5
    d_face = torch.minimum(face_idx, (n - 1) - face_idx).clamp_max(n_pml)
    xi_face = ((n_pml - d_face) / float(n_pml)).clamp_min(0.0)
    sigma_face = sigma_max * xi_face.pow(p_order)

    return sigma_cell, sigma_face


def build_helmholtz_2d_coo(
    vp: torch.Tensor,
    omega: float,
    dz: float,
    dx: float,
    *,
    abc: str = "sommerfeld",
    n_pml: int = 0,
    pml_alpha: float = 1.0,
    boundary: str = "auto",
    pml_thickness: int = 0,
    pml_decay_factor: float = 2.0,
    pml_target_reflection: float = 1e-6,
) -> torch.Tensor:
    """
    Assemble the 2-D Helmholtz system matrix as a torch complex128 COO.

    Args:
        vp: Real ``float64`` cell-centred velocity, shape ``(nz, nx)``. May require
            grad, the returned sparse tensor's values are a torch op of ``vp`` so
            gradients flow.
        omega: Angular frequency ``2π·f`` in rad/s.
        dz, dx: Cell spacings in metres (positive).
        abc: ``"sommerfeld"`` (default), first-order absorbing BC on the four outer
            rows: diagonal gets ``-iω/(v·h)``. ``"dirichlet"`` leaves the boundary
            diagonal untouched (homogeneous Dirichlet via the natural truncation of
            the 5-point stencil); use this for a hard wall. Ignored when
            ``boundary='pml'``: the PML supersedes the outer-row absorbing BC.
        n_pml: Number of Dirichlet-damping cells (the cheap PML imitator) along each
            boundary. If > 0, an exponentially-decaying imaginary perturbation is
            added to the diagonal inside the outermost ``n_pml`` rings. Independent
            of ``abc``. Ignored when ``boundary='pml'``.
        pml_alpha: Strength of the Dirichlet-damping profile (dimensionless).
            Damping added at depth ``d`` cells inside the ring is
            ``i · pml_alpha · ((n_pml - d) / n_pml)² · (ω/v)²``. Ignored when
            ``boundary='pml'``.
        boundary: Boundary treatment selector. ``"auto"`` (default) routes to the
            absorbing-BC + Dirichlet-damping-ring implementation. ``"pml"`` enables
            the split-field complex-coordinate-stretched PML (``pml_thickness`` must
            be > 0). ``"sommerfeld"`` / ``"dirichlet"`` are explicit aliases for the
            non-PML routes (same as ``"auto"`` with the corresponding ``abc``).
        pml_thickness: PML width in cells. Active only when ``boundary='pml'``.
            Typical values 10-20.
        pml_decay_factor: Polynomial order ``p`` of the σ(x) damping profile inside
            the PML. Standard choices are 2-4; higher values concentrate absorption
            near the outer boundary.
        pml_target_reflection: Target reflection coefficient ``R`` used to set the
            absorption strength via ``σ_max = -(p+1)·v·log(R)/(2·L_pml)``. Default
            ``1e-6`` mirrors Komatitsch & Martin 2007.

    Returns:
        Complex128 sparse COO tensor of shape ``(N, N)`` with ``N = nz · nx``.
        Coalesced.
    """
    if vp.ndim != 2:
        raise GeoBrainError(
            f"build_helmholtz_2d_coo: vp must be 2D, got shape {tuple(vp.shape)}",
        )
    if vp.dtype is not torch.float64:
        raise GeoBrainError(
            "build_helmholtz_2d_coo: vp must be float64 "
            f"(got dtype={vp.dtype}); cast to float64 first.",
        )
    if vp.device.type != "cpu":
        raise GeoBrainError(
            "build_helmholtz_2d_coo: vp must be on CPU "
            f"(got device={vp.device}); move it to CPU first.",
        )
    if abc not in ("sommerfeld", "dirichlet"):
        raise GeoBrainError(
            f"build_helmholtz_2d_coo: abc must be 'sommerfeld' or "
            f"'dirichlet'; got {abc!r}",
        )
    if boundary not in ("auto", "pml", "sommerfeld", "dirichlet"):
        raise GeoBrainError(
            f"build_helmholtz_2d_coo: boundary must be 'auto', 'pml', "
            f"'sommerfeld', or 'dirichlet'; got {boundary!r}",
        )

    # Split-field PML path. Routes to the dedicated assembly
    # function below; everything else falls through to the default
    # Sommerfeld + Dirichlet-damping path.
    if boundary == "pml":
        if pml_thickness <= 0:
            raise GeoBrainError(
                "build_helmholtz_2d_coo: boundary='pml' requires "
                f"pml_thickness > 0; got {pml_thickness}",
            )
        if pml_decay_factor <= 0:
            raise GeoBrainError(
                "build_helmholtz_2d_coo: pml_decay_factor must be > 0; "
                f"got {pml_decay_factor}",
            )
        if not (0.0 < pml_target_reflection < 1.0):
            raise GeoBrainError(
                "build_helmholtz_2d_coo: pml_target_reflection must be "
                f"in (0, 1); got {pml_target_reflection}",
            )
        return _build_helmholtz_2d_coo_pml(
            vp=vp,
            omega=omega,
            dz=dz,
            dx=dx,
            pml_thickness=pml_thickness,
            pml_decay_factor=pml_decay_factor,
            pml_target_reflection=pml_target_reflection,
        )

    nz, nx = vp.shape
    n = nz * nx
    device = vp.device
    inv_dx2 = 1.0 / (dx * dx)
    inv_dz2 = 1.0 / (dz * dz)

    vp64 = vp
    m_real = 1.0 / (vp64 * vp64)  # (nz, nx) real

    # ------------------------------------------------------------------
    # Diagonal entries
    # ------------------------------------------------------------------
    diag_real = (2.0 * inv_dx2 + 2.0 * inv_dz2) - (omega * omega) * m_real
    diag = torch.complex(
        diag_real, torch.zeros_like(diag_real),
    )  # (nz, nx) complex128

    # Sommerfeld first-order absorbing BC: subtract iω/(v·h) on each
    # boundary row. Build the four boundary diagonals as additive
    # corrections, then sum them with the bulk diagonal via
    # index_put_-style scatter-add, but we want to stay out-of-place
    # for autograd cleanliness, so we accumulate via masked sums.
    if abc == "sommerfeld":
        zero = torch.zeros_like(vp64)
        ikz = torch.complex(zero, -omega / (vp64 * dz))  # (nz, nx)
        ikx = torch.complex(zero, -omega / (vp64 * dx))  # (nz, nx)

        bnd_z_mask = torch.zeros_like(vp64)
        bnd_z_mask[0, :] = 1.0
        bnd_z_mask[-1, :] = 1.0
        bnd_x_mask = torch.zeros_like(vp64)
        bnd_x_mask[:, 0] = 1.0
        bnd_x_mask[:, -1] = 1.0
        bnd_z_c = torch.complex(bnd_z_mask, torch.zeros_like(bnd_z_mask))
        bnd_x_c = torch.complex(bnd_x_mask, torch.zeros_like(bnd_x_mask))

        diag = diag + bnd_z_c * ikz + bnd_x_c * ikx

    # Dirichlet damping layer (cheap PML imitator). Adds an imaginary
    # part to the squared wavenumber inside the outer n_pml rings.
    if n_pml > 0:
        if n_pml * 2 >= min(nz, nx):
            raise GeoBrainError(
                f"build_helmholtz_2d_coo: n_pml={n_pml} too large for "
                f"grid ({nz}x{nx}); need 2·n_pml < min(nz, nx)",
            )
        # Distance-from-boundary in cells, per axis (saturated to n_pml).
        iz = torch.arange(nz, dtype=torch.float64, device=device)
        ix = torch.arange(nx, dtype=torch.float64, device=device)
        dz_from_bnd = torch.minimum(iz, (nz - 1) - iz).clamp_max(n_pml)
        dx_from_bnd = torch.minimum(ix, (nx - 1) - ix).clamp_max(n_pml)
        # Profile: 1 at the outer boundary, 0 at depth n_pml inwards.
        prof_z = ((n_pml - dz_from_bnd) / float(n_pml)).clamp_min(0.0) ** 2
        prof_x = ((n_pml - dx_from_bnd) / float(n_pml)).clamp_min(0.0) ** 2
        profile = prof_z.unsqueeze(1) + prof_x.unsqueeze(0)  # (nz, nx)
        damp_real = pml_alpha * profile * (omega * omega) * m_real
        zero = torch.zeros_like(damp_real)
        diag = diag + torch.complex(zero, damp_real)

    # ------------------------------------------------------------------
    # Off-diagonal stencil (-1/h² for the four neighbours)
    # ------------------------------------------------------------------
    iz_grid = torch.arange(nz, device=device, dtype=torch.long)
    ix_grid = torch.arange(nx, device=device, dtype=torch.long)
    iiz, iix = torch.meshgrid(iz_grid, ix_grid, indexing="ij")
    iiz_flat = iiz.reshape(-1)
    iix_flat = iix.reshape(-1)
    flat = _flat_index(iiz_flat, iix_flat, nx)

    rows_list: list[torch.Tensor] = [flat]
    cols_list: list[torch.Tensor] = [flat]
    vals_list: list[torch.Tensor] = [diag.reshape(-1)]

    val_dxc = torch.complex(
        torch.tensor(-inv_dx2, dtype=torch.float64, device=device),
        torch.tensor(0.0, dtype=torch.float64, device=device),
    )
    val_dzc = torch.complex(
        torch.tensor(-inv_dz2, dtype=torch.float64, device=device),
        torch.tensor(0.0, dtype=torch.float64, device=device),
    )

    # x+1
    mask = iix_flat < (nx - 1)
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] + 1)
    vals_list.append(val_dxc.expand(int(mask.sum().item())).clone())

    # x-1
    mask = iix_flat > 0
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] - 1)
    vals_list.append(val_dxc.expand(int(mask.sum().item())).clone())

    # z+1
    mask = iiz_flat < (nz - 1)
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] + nx)
    vals_list.append(val_dzc.expand(int(mask.sum().item())).clone())

    # z-1
    mask = iiz_flat > 0
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] - nx)
    vals_list.append(val_dzc.expand(int(mask.sum().item())).clone())

    rows = torch.cat(rows_list)
    cols = torch.cat(cols_list)
    vals = torch.cat(vals_list)

    indices = torch.stack([rows, cols], dim=0)
    A = torch.sparse_coo_tensor(indices, vals, size=(n, n), dtype=torch.complex128)
    return A.coalesce()


def _build_helmholtz_2d_coo_pml(
    *,
    vp: torch.Tensor,
    omega: float,
    dz: float,
    dx: float,
    pml_thickness: int,
    pml_decay_factor: float,
    pml_target_reflection: float,
) -> torch.Tensor:
    """
    Split-field PML Helmholtz assembly.

    Frequency-domain complex-coordinate-stretched Helmholtz::

        (1/s_x) ∂_x ( (1/s_x) ∂_x p ) + (1/s_z) ∂_z ( (1/s_z) ∂_z p )
            + (ω/v)² p = -s

    Discretization on a uniform staggered grid:

    - The *outer* divergence at cell ``(i, j)`` uses cell-centred
      ``s_x^c[j]`` and ``s_z^c[i]``;
    - The *inner* gradient on face ``(i, j+1/2)`` uses face-centred
      ``s_x^f[j+1/2]`` (and similarly for z).

    The resulting matrix has the same 5-point sparsity pattern but the
    coefficients are now complex-valued and spatially varying.

    System sign convention::

        A = -L_h^stretched - ω² diag(1/v²)
        A u = s

    so each off-diagonal entry to neighbour ``(i, j+1)`` is
    ``-1 / (s_x^c[j] · s_x^f[j+1/2] · dx²)`` and the diagonal is the
    sum of the negatives of the four off-diagonal entries plus the
    ``-ω²/v²`` term.

    No outer absorbing BC is added on top: the PML already absorbs
    outgoing waves before they hit the matrix boundary, and the
    natural truncation of the 5-point stencil at the outer ring
    behaves as Dirichlet for the (already strongly damped) field.
    """
    if vp.ndim != 2:
        raise GeoBrainError(
            "_build_helmholtz_2d_coo_pml: vp must be 2D, "
            f"got shape {tuple(vp.shape)}",
        )

    nz, nx = vp.shape
    if 2 * pml_thickness >= min(nz, nx):
        raise GeoBrainError(
            f"_build_helmholtz_2d_coo_pml: pml_thickness={pml_thickness} "
            f"too large for grid ({nz}x{nx}); need 2·pml_thickness < min(nz, nx)",
        )

    n = nz * nx
    device = vp.device

    vp64 = vp
    v_ref = vp64.max()

    # ------------------------------------------------------------------
    # 1-D damping profiles σ_z, σ_x on cell centres and faces.
    # ------------------------------------------------------------------
    sigma_z_cell, sigma_z_face = _pml_sigma_1d(
        nz, pml_thickness, dz, v_ref, pml_decay_factor,
        pml_target_reflection, device,
    )
    sigma_x_cell, sigma_x_face = _pml_sigma_1d(
        nx, pml_thickness, dx, v_ref, pml_decay_factor,
        pml_target_reflection, device,
    )

    # ------------------------------------------------------------------
    # Complex stretching factors: s = 1 - iσ/ω. With exp(+iωt), this
    # damps the outgoing H_0^(2) branch. In the interior σ = 0 so s = 1.
    # ------------------------------------------------------------------
    inv_omega = 1.0 / float(omega)

    # Cell-centred: shape (nz,), (nx,)
    s_z_cell = torch.complex(torch.ones_like(sigma_z_cell), -sigma_z_cell * inv_omega)
    s_x_cell = torch.complex(torch.ones_like(sigma_x_cell), -sigma_x_cell * inv_omega)
    inv_sz_cell = 1.0 / s_z_cell  # (nz,)
    inv_sx_cell = 1.0 / s_x_cell  # (nx,)

    # Face-centred: shape (nz-1,), (nx-1,).  Face i corresponds to the
    # face between cells i and i+1, i.e. located at i+1/2.
    s_z_face = torch.complex(torch.ones_like(sigma_z_face), -sigma_z_face * inv_omega)
    s_x_face = torch.complex(torch.ones_like(sigma_x_face), -sigma_x_face * inv_omega)
    inv_sz_face = 1.0 / s_z_face  # (nz-1,)
    inv_sx_face = 1.0 / s_x_face  # (nx-1,)

    inv_dx2 = 1.0 / (dx * dx)
    inv_dz2 = 1.0 / (dz * dz)

    # ------------------------------------------------------------------
    # Build COO entries.  The flat index for cell (i, j) is i*nx + j.
    # ------------------------------------------------------------------
    iz_grid = torch.arange(nz, device=device, dtype=torch.long)
    ix_grid = torch.arange(nx, device=device, dtype=torch.long)
    iiz, iix = torch.meshgrid(iz_grid, ix_grid, indexing="ij")
    iiz_flat = iiz.reshape(-1)
    iix_flat = iix.reshape(-1)
    flat = _flat_index(iiz_flat, iix_flat, nx)

    # The off-diagonal coefficient to neighbour (i, j+1) is
    #   c_x_plus[i, j] = -inv_dx2 · inv_sx_cell[j] · inv_sx_face[j]      (for j < nx-1)
    # To neighbour (i, j-1):
    #   c_x_minus[i, j] = -inv_dx2 · inv_sx_cell[j] · inv_sx_face[j-1]   (for j > 0)
    # And similarly for z.
    #
    # The diagonal contribution from x is the negative of the sum of
    # the two x-neighbour coefficients (the discretized Laplacian of
    # constant 1 must vanish):
    #   d_x[i, j] = -(c_x_plus[i, j] + c_x_minus[i, j])
    # Add the -ω²/v² term to get the full diagonal.

    # x-direction face coefficients per cell (i, j).  At boundary j the
    # neighbour does not exist so the corresponding coefficient is zero;
    # this matches the natural Dirichlet truncation of the 5-point
    # stencil on the outermost row, but inside the PML the field is
    # already exponentially damped, so the residual reflection is
    # negligible.
    # Build face-coefficient arrays broadcastable to (nx,).
    # x+1 face index = j; defined for j in [0, nx-2]
    face_xp = torch.zeros(nx, dtype=torch.complex128, device=device)
    face_xp[:-1] = inv_sx_face            # face between j and j+1
    # x-1 face index = j-1; defined for j in [1, nx-1]
    face_xm = torch.zeros(nx, dtype=torch.complex128, device=device)
    face_xm[1:] = inv_sx_face             # face between j-1 and j

    # Per-cell x coefficients (column-wise broadcast to (nz, nx))
    cx_plus_1d = -inv_dx2 * inv_sx_cell * face_xp   # (nx,)
    cx_minus_1d = -inv_dx2 * inv_sx_cell * face_xm  # (nx,)
    # Broadcast to (nz, nx) by repeating along z.
    cx_plus = cx_plus_1d.unsqueeze(0).expand(nz, -1).contiguous()
    cx_minus = cx_minus_1d.unsqueeze(0).expand(nz, -1).contiguous()

    face_zp = torch.zeros(nz, dtype=torch.complex128, device=device)
    face_zp[:-1] = inv_sz_face
    face_zm = torch.zeros(nz, dtype=torch.complex128, device=device)
    face_zm[1:] = inv_sz_face

    cz_plus_1d = -inv_dz2 * inv_sz_cell * face_zp   # (nz,)
    cz_minus_1d = -inv_dz2 * inv_sz_cell * face_zm  # (nz,)
    cz_plus = cz_plus_1d.unsqueeze(1).expand(-1, nx).contiguous()
    cz_minus = cz_minus_1d.unsqueeze(1).expand(-1, nx).contiguous()

    # Diagonal:
    #   diag[i, j] = -(cx_plus + cx_minus + cz_plus + cz_minus)[i, j]
    #              + (-ω²/v²)
    # Note the off-diagonals are already negative; the diagonal must be
    # positive of the sum of |off-diagonals| (Laplacian of constant
    # field is zero), then we add the model term.
    m_real = 1.0 / (vp64 * vp64)
    omega2 = omega * omega
    diag_model = torch.complex(-omega2 * m_real, torch.zeros_like(m_real))  # (nz, nx)
    diag = -(cx_plus + cx_minus + cz_plus + cz_minus) + diag_model         # (nz, nx)

    rows_list: list[torch.Tensor] = [flat]
    cols_list: list[torch.Tensor] = [flat]
    vals_list: list[torch.Tensor] = [diag.reshape(-1)]

    # x+1
    mask = iix_flat < (nx - 1)
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] + 1)
    # Coefficient at cell (i, j) for neighbour (i, j+1) is cx_plus[i, j].
    cx_plus_flat = cx_plus.reshape(-1)
    vals_list.append(cx_plus_flat[mask])

    # x-1
    mask = iix_flat > 0
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] - 1)
    cx_minus_flat = cx_minus.reshape(-1)
    vals_list.append(cx_minus_flat[mask])

    # z+1
    mask = iiz_flat < (nz - 1)
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] + nx)
    cz_plus_flat = cz_plus.reshape(-1)
    vals_list.append(cz_plus_flat[mask])

    # z-1
    mask = iiz_flat > 0
    rows_list.append(flat[mask])
    cols_list.append(flat[mask] - nx)
    cz_minus_flat = cz_minus.reshape(-1)
    vals_list.append(cz_minus_flat[mask])

    rows = torch.cat(rows_list)
    cols = torch.cat(cols_list)
    vals = torch.cat(vals_list)

    indices = torch.stack([rows, cols], dim=0)
    A = torch.sparse_coo_tensor(indices, vals, size=(n, n), dtype=torch.complex128)
    return A.coalesce()


def build_packed_helmholtz_2d_rhs(
    *,
    sources_iz_ix: tuple[tuple[int, int], ...],
    source_shot_index: tuple[int, ...],
    amplitudes: torch.Tensor,
    n_shot: int,
    nz: int,
    nx: int,
    cell_area: float,
) -> torch.Tensor:
    """Assemble integrated point sources into packed shot columns.

    ``amplitude`` is the integrated 2-D source strength. A grid delta therefore
    has value ``amplitude / (dx*dz)``. Multiple source rows may share a shot and
    are accumulated into that shot's single right-hand side.
    """
    if amplitudes.dtype is not torch.complex128:
        raise GeoBrainError(
            "build_packed_helmholtz_2d_rhs: amplitudes must be complex128"
        )
    if amplitudes.device.type != "cpu":
        raise GeoBrainError(
            "build_packed_helmholtz_2d_rhs: amplitudes must be on CPU"
        )
    if len(sources_iz_ix) != len(source_shot_index) or amplitudes.numel() != len(
        sources_iz_ix
    ):
        raise GeoBrainError(
            "build_packed_helmholtz_2d_rhs: source metadata lengths must match"
        )
    if n_shot <= 0 or not math.isfinite(cell_area) or cell_area <= 0.0:
        raise GeoBrainError(
            "build_packed_helmholtz_2d_rhs: n_shot and cell_area must be positive"
        )
    rhs = torch.zeros(
        (nz * nx, n_shot), dtype=torch.complex128, device=amplitudes.device
    )
    rows = torch.tensor(
        [iz * nx + ix for iz, ix in sources_iz_ix],
        dtype=torch.int64,
        device=amplitudes.device,
    )
    columns = torch.tensor(
        source_shot_index, dtype=torch.int64, device=amplitudes.device
    )
    if bool((columns < 0).any()) or bool((columns >= n_shot).any()):
        raise GeoBrainError(
            "build_packed_helmholtz_2d_rhs: shot index is outside packed domain"
        )
    rhs.index_put_(
        (rows, columns), amplitudes / float(cell_area), accumulate=True
    )
    return rhs


__all__ = [
    "build_helmholtz_2d_coo",
    "build_packed_helmholtz_2d_rhs",
]
