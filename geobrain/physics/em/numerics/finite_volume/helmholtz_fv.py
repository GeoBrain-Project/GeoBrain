# pyright: reportPrivateImportUsage=false
"""Mesh-agnostic sparse FV assembly of the MT2D complex Helmholtz operators.

The sparse, :class:`FaceRecords`-driven counterpart of the dense structured
assemblers in :mod:`.helmholtz_2d` (same file pairing as ``poisson.py`` →
``poisson_fv.py``): :func:`assemble_helmholtz_fv` (TE) and
:func:`assemble_helmholtz_fv_tm` (TM) consume the connectivity SoA of **any**
2-D :class:`ConnectivityMesh` (triangle :class:`UnstructuredMesh`, TensorMesh
SoA, …) and return a complex128 sparse COO matrix plus its RHS.

Coefficient placement is derived line-by-line from ``helmholtz_2d.py`` (NOT
from textbook conventions; the structured stencils are the contract):

- **TE**: the structured interior row is ``∇²u + i·ω·μ₀·σ·u = 0`` (diagonal
  ``−2/dz² − 2/dy² + iωμ₀σ``, off-diagonals ``+1/h²``). The cell-integrated FV
  form is ``∮∇u·n̂ dS + iωμ₀·σ_c·V_c·u_c = 0``; with
  :func:`~.poisson_fv.assemble_poisson_fv` building ``−∮`` (unit coefficient,
  ``T_f = area/(dist_l + dist_r)``), the assembled system is the structured
  equation multiplied by ``−V_c``::

      [A_TPFA(1) + diag(−i·ω·μ₀·σ·V)] u = b          (σ in the MASS term)

  Since the RHS below is built in the same ``−V``-scaled convention, the
  solved field, and therefore Z, ρ_a, phase; carries the structured path's
  sign/time convention (exp-convention notes in ``mt2d.py``) unchanged.

- **TM**: the structured row is ``∇·[(1/σ)∇u] + i·ω·μ₀·u = 0``::

      [A_TPFA(1/σ) + diag(−i·ω·μ₀·V)] u = b          (1/σ in the FLUX term)

  One deliberate, documented deviation: the structured assembler evaluates the
  face diffusivity as ``1/harmonic-mean(σ)`` (= the *arithmetic* mean of ρ),
  while the TPFA series-harmonic transmissibility of ``assemble_poisson_fv``
  applied to ``ρ = 1/σ`` yields the *harmonic* mean of ρ, the locally
  flux-continuous two-point closure. The two coincide wherever σ is uniform
  and differ only on the few faces crossing a conductivity jump; the
  structured-vs-unstructured operator test bounds the end-to-end effect.

Boundary conditions replicate the structured conventions (top incident
plane wave, bottom absorbing, reflective sides) in their native FV form:

- **top** (outward normal pointing up, −z): Dirichlet ``u = 1`` imposed *at
  the boundary face* through the half-cell transmissibility
  ``T_b = coef_c · area / dist`` (``dist`` = perpendicular cell-point→face
  distance): ``diag[c] += T_b`` and ``b[c] += T_b · 1``. This is the exact
  two-point closure between the cell point and the face, chosen over pinning
  the top *cells* (the dense assembler's identity-row trick) because it lands
  the prescribed value on the physical surface ``z = 0`` itself rather than on
  the first row of cell centres. The diagonal half goes through
  ``assemble_poisson_fv``'s Robin seam (``robin_alpha_area = area/dist``,
  the seam stamps ``coef[cell]·alpha_area``, exactly ``T_b``).
- **bottom** (outward normal pointing down, +z): Dirichlet ``u = 0``: same
  ``T_b`` diagonal closure, zero RHS contribution.
- **sides**: natural no-flux (the FV analogue of the structured
  index-clamped reflective Neumann): boundary faces not classified as
  top/bottom simply contribute nothing.

Unlike the dense assemblers (boundary values supplied by the caller through
``b``), the Dirichlet *values* (top 1, bottom 0) are baked here, because the
inhomogeneous part ``T_b·u_D`` requires the per-face transmissibility that
only the assembler knows. Axis convention (read from the existing 2-D
operators, e.g. ``DC25D``): coordinate column 0 is depth ``z`` increasing
DOWNWARD (surface at min z), column 1 is the horizontal profile coordinate.

All numerics are float64 / complex128. Autograd flows through ``sigma`` into
the mass diagonal (TE) or the face transmissibilities + boundary closures +
RHS (TM).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geobrain.physics.em.conventions import MU_0

from geobrain.core.errors import GeoBrainError
from geobrain.mesh.capabilities import BoundaryRecords, FaceRecords

from .poisson_fv import assemble_poisson_fv


__all__ = [
    "HelmholtzBoundaryPartition",
    "partition_helmholtz_boundary",
    "assemble_helmholtz_fv",
    "assemble_helmholtz_fv_tm",
]


# A boundary face is classified as top/bottom when its outward unit normal's
# z-component dominates at least this much (|n_z| ≥ 1/√2, i.e. within 45° of
# vertical). On an axis-aligned box mesh the normals are exact unit vectors,
# so the threshold only matters for gently non-flat surfaces.
_AXIS_ALIGN_MIN = 0.7071067811865475


@dataclass(frozen=True)
class HelmholtzBoundaryPartition:
    """Top/bottom Dirichlet faces of a 2-D ``(z, y)`` mesh, struct-of-arrays.

    Produced by :func:`partition_helmholtz_boundary`; consumed by the FV
    Helmholtz assemblers (boundary closure) and by the MT2D operator (surface
    impedance extraction binds each station to its nearest top face).

    Attributes:
        top_cell / bottom_cell: ``(n,)`` long, owning cell per face.
        top_area / bottom_area: ``(n,)`` float64, face area (edge length).
        top_dist / bottom_dist: ``(n,)`` float64, perpendicular cell-point→
            face distance (the half-cell Dirichlet closure length).
        top_y: ``(n_top,)`` float64, face centroid horizontal coordinate
            (column 1), used for station binding.
    """

    top_cell: torch.Tensor
    top_area: torch.Tensor
    top_dist: torch.Tensor
    top_y: torch.Tensor
    bottom_cell: torch.Tensor
    bottom_area: torch.Tensor
    bottom_dist: torch.Tensor


def partition_helmholtz_boundary(
    boundary: BoundaryRecords,
    cell_centers: torch.Tensor,
) -> HelmholtzBoundaryPartition:
    """Classify a 2-D mesh's boundary faces into MT2D top / bottom / sides.

    Classification is by the outward unit normal (axis convention: column 0 is
    depth z, positive DOWN): ``n_z ≤ −1/√2`` → top (incident plane-wave
    Dirichlet), ``n_z ≥ +1/√2`` → bottom (absorbing Dirichlet), anything else
    → side (natural no-flux, not recorded). The perpendicular cell-point→face
    distance is computed per face as ``|(centroid − center[cell]) · normal|``,
    the same FV half-distance convention :class:`FaceRecords` carries for
    internal faces.

    Args:
        boundary: ``BoundaryRecords`` from ``mesh.boundary_faces()``.
        cell_centers: ``(n_cells, 2)`` float64 ``mesh.cell_centers()``.

    Returns:
        A :class:`HelmholtzBoundaryPartition`.

    Raises:
        GeoBrainError: when the mesh exposes no top or no bottom faces (the
            plane-wave problem is ill-posed without both), or when a boundary
            cell point lies on its face (degenerate half-cell distance).
    """
    if boundary.normal.ndim != 2 or int(boundary.normal.shape[1]) != 2:
        raise GeoBrainError(
            "partition_helmholtz_boundary needs 2-D boundary normals",
            object_name="partition_helmholtz_boundary",
            field="boundary.normal",
            expected="(n_bf, 2)",
            actual=tuple(boundary.normal.shape),
        )
    cell = boundary.cell.to(torch.long)
    area = boundary.area.to(torch.float64)
    normal = boundary.normal.to(torch.float64)
    centroid = boundary.centroid.to(torch.float64)
    centers = cell_centers.to(torch.float64)

    n_z = normal[:, 0]
    is_top = n_z <= -_AXIS_ALIGN_MIN
    is_bottom = n_z >= _AXIS_ALIGN_MIN
    if not bool(is_top.any()) or not bool(is_bottom.any()):
        raise GeoBrainError(
            "MT2D FV boundary partition found no top or no bottom faces, the "
            "plane-wave Dirichlet pair (top u=1, bottom u=0) needs both. Check "
            "the mesh's axis convention (column 0 must be depth z, +down).",
            object_name="partition_helmholtz_boundary",
            field="boundary_faces",
            expected="≥1 face with n_z ≤ −1/√2 and ≥1 with n_z ≥ +1/√2",
            actual=(int(is_top.sum()), int(is_bottom.sum())),
        )

    # Perpendicular distance cell point → face line, per boundary face.
    dist = ((centroid - centers[cell]) * normal).sum(dim=1).abs()
    if bool((dist[is_top | is_bottom] <= 1e-12).any()):
        raise GeoBrainError(
            "degenerate FV boundary geometry, a cell point lies on its "
            "top/bottom boundary face (half-cell distance ~ 0)",
            object_name="partition_helmholtz_boundary",
            field="dist",
            expected="> 1e-12",
            actual=float(dist[is_top | is_bottom].min()),
        )

    return HelmholtzBoundaryPartition(
        top_cell=cell[is_top],
        top_area=area[is_top],
        top_dist=dist[is_top],
        top_y=centroid[is_top, 1],
        bottom_cell=cell[is_bottom],
        bottom_area=area[is_bottom],
        bottom_dist=dist[is_bottom],
    )


def _check_sigma_flat(sigma: torch.Tensor, where: str) -> None:
    if sigma.ndim != 1:
        raise GeoBrainError(
            f"{where}: sigma must be flat (n_cells,) on the FV path",
            object_name=where,
            field="sigma.ndim",
            expected=1,
            actual=sigma.ndim,
        )


def _dirichlet_closure(
    partition: HelmholtzBoundaryPartition,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Robin-seam keys + the top half-cell geometric factor.

    Returns ``(boundary_dict, top_alpha, top_cell)`` where ``boundary_dict``
    carries the top+bottom diagonal closures for
    :func:`~.poisson_fv.assemble_poisson_fv` (the seam stamps
    ``coef[cell]·alpha_area``, i.e. exactly ``T_b = coef_c·area/dist``) and
    ``top_alpha = area/dist`` over the top faces (the RHS needs
    ``T_b·u_D`` with ``u_D = 1`` there; the bottom's ``u_D = 0`` contributes
    nothing).
    """
    top_alpha = partition.top_area / partition.top_dist
    bottom_alpha = partition.bottom_area / partition.bottom_dist
    boundary = {
        "robin_cells": torch.cat([partition.top_cell, partition.bottom_cell]),
        "robin_alpha_area": torch.cat([top_alpha, bottom_alpha]),
    }
    return boundary, top_alpha, partition.top_cell


def assemble_helmholtz_fv(
    face_records: FaceRecords,
    partition: HelmholtzBoundaryPartition,
    cell_volumes: torch.Tensor,
    sigma: torch.Tensor,
    omega: float,
    *,
    mu0: float = MU_0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the TE-mode sparse complex Helmholtz system ``A u = b``.

    Discretises the structured contract ``∇²u + i·ω·μ₀·σ·u = 0`` in
    cell-integrated form: unit-coefficient TPFA flux + the cell-diagonal mass
    term ``−i·ω·μ₀·σ_c·V_c`` (sign carries the overall ``−V`` scaling of the
    ``−∇·`` assembly convention; see the module docstring). Boundary: top
    Dirichlet ``u = 1`` / bottom ``u = 0`` via the half-cell transmissibility,
    sides natural no-flux.

    Args:
        face_records: internal-face SoA from ``mesh.face_neighbors()``.
        partition: top/bottom faces from :func:`partition_helmholtz_boundary`.
        cell_volumes: ``(n_cells,)`` from ``mesh.cell_volumes()``.
        sigma: real cell-centred conductivity, flat ``(n_cells,)``; autograd
            flows through the mass diagonal.
        omega: angular frequency ``2·π·f`` (rad/s).
        mu0: vacuum permeability (the MT2D operator passes its own ``4π·10⁻⁷``
            so both mesh paths share one constant).

    Returns:
        ``(A, b)``: coalesced complex128 sparse COO ``(n, n)`` and dense
        complex128 RHS ``(n,)``.
    """
    _check_sigma_flat(sigma, "assemble_helmholtz_fv")
    n = int(sigma.shape[0])
    device = sigma.device
    cdtype = torch.complex128

    # Unit diffusivity: TPFA degenerates to the pure geometric Laplacian
    # transmissibility T_f = area / (dist_l + dist_r).
    coef = torch.ones(n, dtype=cdtype, device=device)
    mass = (-1j * omega * mu0) * sigma.to(cdtype) * cell_volumes.to(cdtype)

    boundary, top_alpha, top_cell = _dirichlet_closure(partition)
    A = assemble_poisson_fv(face_records, coef, boundary=boundary, extra_diagonal=mass)

    # RHS: top faces carry T_b·u_D with u_D = 1 and coef ≡ 1 → area/dist.
    b = torch.zeros(n, dtype=cdtype, device=device).index_add(
        0, top_cell, top_alpha.to(cdtype)
    )
    return A, b


def assemble_helmholtz_fv_tm(
    face_records: FaceRecords,
    partition: HelmholtzBoundaryPartition,
    cell_volumes: torch.Tensor,
    sigma: torch.Tensor,
    omega: float,
    *,
    mu0: float = MU_0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble the TM-mode sparse complex Helmholtz system ``A u = b``.

    Discretises the structured contract ``∇·[(1/σ)∇u] + i·ω·μ₀·u = 0``: the
    flux coefficient is the resistivity ``ρ = 1/σ`` (TPFA's series-harmonic
    transmissibility takes over the face averaging; see the module docstring
    for the one documented deviation from the structured face rule) and the
    mass diagonal is the σ-independent ``−i·ω·μ₀·V_c``. Boundary closures as
    in :func:`assemble_helmholtz_fv`, with ``T_b = ρ_c·area/dist`` so the
    top RHS (like the matrix) stays autograd-attached to σ.

    Args / Returns: as :func:`assemble_helmholtz_fv`.
    """
    _check_sigma_flat(sigma, "assemble_helmholtz_fv_tm")
    n = int(sigma.shape[0])
    device = sigma.device
    cdtype = torch.complex128

    rho = (1.0 / sigma).to(cdtype)
    mass = (-1j * omega * mu0) * cell_volumes.to(cdtype)

    boundary, top_alpha, top_cell = _dirichlet_closure(partition)
    A = assemble_poisson_fv(face_records, rho, boundary=boundary, extra_diagonal=mass)

    # RHS: T_b·u_D = ρ_c·(area/dist)·1 on the top cells (σ graph attached).
    b = torch.zeros(n, dtype=cdtype, device=device).index_add(
        0, top_cell, rho[top_cell] * top_alpha.to(cdtype)
    )
    return A, b