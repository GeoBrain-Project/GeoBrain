# pyright: reportPrivateImportUsage=false
"""Mixed / Robin absorbing-boundary coefficients for the FV Poisson seam.

The truncated-domain DC/IP system on a SMALL box suffers boundary reflection of
the potential under pure-Neumann walls. A *mixed* (Robin) boundary condition
``∂u/∂n ≈ −α·u`` on the outer faces absorbs the far field: the FV seam adds
``σ_cell · α_face · area_face`` to the diagonal ``A[cell, cell]`` for each
boundary face (the far-field flux ``flux_out = σ α u A``).

This module computes the geometric coefficient ``α`` for a point source over a
half-space with a mirror image across the free surface, in the
:class:`~geobrain.mesh.capabilities.BoundaryRecords` form
(``cell``/``area``/``normal``/``centroid``) that **every**
:class:`~geobrain.mesh.capabilities.ConnectivityMesh` exposes via
``mesh.boundary_faces()`` (TensorMesh, OctreeMesh, UnstructuredMesh). The form is
the asymptotic decay rate of the point-source potential (the Dey–Morrison
``dcfemmodelling`` mixed-BC form, adapted to FV):

    α_f = [ |r'|²·(r·n̂)/|r| + |r|²·(r'·n̂)/|r'| ]
          / [ |r|·|r'|·(|r| + |r'|) ]

with ``r = x_face − x_ref``, ``r' = x_face − x_ref_mirror`` (``x_ref`` mirrored
across the free-surface plane), ``n̂`` the OUTWARD unit face normal. ``α`` is
clamped at ``≥ 0`` (a negative cosine means the face "sees" the source from
behind; clamping to Neumann there is standard). **Free-surface faces** (those on
the ``z = 0`` plane) are EXCLUDED from the Robin set entirely (they stay
Neumann: no current into the air).

Using a single fixed reference point (e.g. the electrode-array centroid) rather
than the true per-source location keeps ``A`` source-independent, so the LU
factorisation stays shared across all right-hand sides, the standard compromise
versus the per-source exact form.

``α`` is a pure function of mesh GEOMETRY and the reference point; it carries NO
gradient requirement (the σ-derivative of the Robin diagonal term ``σ·α·A`` lives
entirely in the ``σ`` factor, which the seam multiplies in). The returned tensors
are detached float64.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core.errors import GeoBrainError
from geobrain.mesh.capabilities import BoundaryRecords


__all__ = ["mixed_boundary_alpha"]


def mixed_boundary_alpha(
    boundary_records: BoundaryRecords,
    reference_point: torch.Tensor,
    *,
    free_surface_axis: int = 0,
    free_surface_coord: float = 0.0,
    atol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-boundary-face Robin coefficients ``(robin_cells, robin_alpha_area)``.

    Computes the mirror-image point-source decay rate ``α`` (see the module
    docstring) on each outer boundary face, drops the free-surface faces, and
    returns the owning-cell indices plus the PRE-multiplied ``α · area`` ready to
    hand straight to :func:`assemble_poisson_fv`'s ``robin_alpha_area`` key.

    Geometry only: no autograd flows through ``α`` (the returned tensors are
    detached float64). The σ-dependence of the Robin diagonal term ``σ·α·A`` is
    supplied by the seam, which multiplies ``α·area`` by ``sigma[cell]``.

    Args:
        boundary_records: :class:`BoundaryRecords` from ``mesh.boundary_faces()``
            (``cell``, ``area``, OUTWARD unit ``normal``, ``centroid``) over the
            ``nb`` outer faces of any :class:`ConnectivityMesh`.
        reference_point: ``(n_dim,)`` tensor in mesh coordinates (GeoBrain 3-D
            order ``(z, y, x)``; ``z`` is depth, the free surface at ``z ≈ 0``).
            Typically the centroid of the electrode array.
        free_surface_axis: the axis index whose ``≈ free_surface_coord`` plane is
            the free surface (default ``0`` = depth ``z``). The reference point is
            mirrored across this plane; faces ON this plane are dropped (Neumann).
        free_surface_coord: the coordinate of the free-surface plane along
            ``free_surface_axis`` (default ``0.0``).
        atol: relative tolerance for the free-surface test; a face is on the
            free surface when ``|centroid[axis] − free_surface_coord| ≤
            atol · domain_scale`` where ``domain_scale`` is the bounding-box
            diagonal of the boundary-face centroids (so the test is robust to the
            absolute size of the mesh). Defaults to ``1e-6``.

    Returns:
        ``(robin_cells, robin_alpha_area)``: a ``(m,)`` long tensor of owning
        cell indices and a ``(m,)`` float64 tensor of ``α · area`` for the ``m``
        non-free-surface boundary faces with ``α > 0``. Both are detached.

    Raises:
        GeoBrainError: on a non-1-D / wrong-length ``reference_point``, an
            out-of-range ``free_surface_axis``, or a non-finite ``atol < 0``.
    """
    if not isinstance(boundary_records, BoundaryRecords):
        raise GeoBrainError(
            "mixed_boundary_alpha: boundary_records must be a BoundaryRecords",
            object_name="mixed_boundary_alpha",
            field="boundary_records",
            expected="BoundaryRecords",
            actual=type(boundary_records).__name__,
        )
    centroid = boundary_records.centroid.to(torch.float64)
    normal = boundary_records.normal.to(torch.float64)
    area = boundary_records.area.to(torch.float64)
    cell = boundary_records.cell.to(torch.long)

    nb = int(cell.numel())
    if nb == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float64),
        )
    n_dim = int(centroid.shape[1])

    if not (0 <= int(free_surface_axis) < n_dim):
        raise GeoBrainError(
            "mixed_boundary_alpha: free_surface_axis out of range",
            object_name="mixed_boundary_alpha",
            field="free_surface_axis",
            expected=f"[0, {n_dim})",
            actual=free_surface_axis,
        )
    if atol < 0 or not torch.isfinite(torch.tensor(float(atol))):
        raise GeoBrainError(
            "mixed_boundary_alpha: atol must be a finite, non-negative float",
            object_name="mixed_boundary_alpha",
            field="atol",
            expected=">= 0 and finite",
            actual=atol,
        )

    ref = reference_point.detach().to(torch.float64).reshape(-1)
    if int(ref.numel()) != n_dim:
        raise GeoBrainError(
            "mixed_boundary_alpha: reference_point length must equal n_dim",
            object_name="mixed_boundary_alpha",
            field="reference_point",
            expected=(n_dim,),
            actual=tuple(reference_point.shape),
        )

    axis = int(free_surface_axis)
    fs = float(free_surface_coord)

    # Domain scale = bounding-box diagonal of the boundary-face centroids, so the
    # free-surface plane test scales with the mesh extent (a 10 m and a 10 km box
    # both resolve the surface at the same RELATIVE tolerance).
    bb = centroid.max(dim=0).values - centroid.min(dim=0).values
    domain_scale = float(torch.linalg.norm(bb))
    if domain_scale <= 0.0:
        domain_scale = 1.0
    fs_tol = float(atol) * domain_scale

    # Free-surface faces (centroid on the z = free_surface_coord plane) are
    # dropped from the Robin set: they remain pure Neumann.
    on_free_surface = (centroid[:, axis] - fs).abs() <= fs_tol
    keep_fs = ~on_free_surface

    # Mirror the reference point across the free-surface plane: reflect its
    # free_surface_axis coordinate about ``free_surface_coord`` (= negate when
    # the surface is at 0).
    ref_mirror = ref.clone()
    ref_mirror[axis] = 2.0 * fs - ref[axis]

    # r = x_face - x_ref ; r' = x_face - x_ref_mirror.
    r = centroid - ref.view(1, n_dim)            # (nb, n_dim)
    rp = centroid - ref_mirror.view(1, n_dim)    # (nb, n_dim)
    r_norm = torch.linalg.norm(r, dim=1)          # |r|
    rp_norm = torch.linalg.norm(rp, dim=1)        # |r'|
    r_dot_n = (r * normal).sum(dim=1)             # r · n̂  (outward)
    rp_dot_n = (rp * normal).sum(dim=1)           # r' · n̂

    eps = 1e-300  # guard the rare face-centroid-on-reference-point degeneracy
    numer = (
        rp_norm * rp_norm * r_dot_n / (r_norm + eps)
        + r_norm * r_norm * rp_dot_n / (rp_norm + eps)
    )
    denom = r_norm * rp_norm * (r_norm + rp_norm) + eps
    alpha = numer / denom

    # Clamp α ≥ 0: a negative value means the outward normal points away from
    # the source (the face "sees" the source from behind): Neumann (α = 0)
    # there is the standard choice. Non-finite (degenerate r/r') → 0 as well.
    alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = alpha.clamp_min(0.0)

    # Keep non-free-surface faces with a strictly positive α (α = 0 faces add
    # nothing to the diagonal, so dropping them keeps the COO minimal).
    keep = keep_fs & (alpha > 0.0)
    robin_cells = cell[keep]
    robin_alpha_area = (alpha[keep] * area[keep]).to(torch.float64)
    return robin_cells, robin_alpha_area
