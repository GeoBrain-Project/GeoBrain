# pyright: reportPrivateImportUsage=false
"""
Cell-centered finite-volume Poisson assembly on a 3-D :class:`OctreeMesh`.

Octree counterpart to :func:`assemble_poisson_3d` (TensorMesh path). The
discrete operator is

    A = D · diag(σ_face) · G

with the standard TPFA harmonic-mean face conductivity. Iteration walks
**internal-face records** ``(i, j, axis, area, dist)`` produced by
:func:`_find_face_neighbors`, which collapses hanging faces to one record
per (small-cell, big-cell) pair using the small face's area and the
cell-center distance along the face normal.

The face-neighbor enumerator replaces
``mesh/octree/operators._find_face_neighbors``, which stores per-cell
``(ix, iy, iz, level)`` integer indices on a structured grid, while
the :class:`OctreeMesh` keeps a flat leaf list with continuous
``(centers, half_widths, levels)`` data. We detect neighbours by walking
each axis and matching cell extents through a small per-axis spatial hash
on the cell-face coordinates, exact under the OctreeMesh refinement
rules (each split produces ``2**n_dim`` equal-sized children whose faces
align with their parent's).

Accuracy:
Standard TPFA is exact for piecewise-constant σ and conforming faces,
first-order at hanging faces. For homogeneous halfspaces on moderately
refined octrees the apparent resistivity sits within ~5-10 % of analytic.
For non-aligned octree faces, see :mod:`.poisson_octree_mpfa`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from geobrain.mesh import OctreeMesh
from geobrain.core.errors import GeoBrainError
from geobrain.core.adjoint import sparse_linear_solve_with_adjoint


__all__ = [
    "_find_octree_face_neighbors",
    "assemble_poisson_octree_3d",
    "solve_poisson_octree_3d",
]


def _find_octree_face_neighbors(
    mesh: OctreeMesh,
) -> List[Tuple[int, int, int, float, float]]:
    """
    Internal-face neighbour records for a 3-D :class:`OctreeMesh`.

    Returns one record per internal (small-cell, neighbour-cell) face pair::

        (i, j, axis, area, dist)

    - ``i, j``: flat leaf indices into the mesh's cell arrays.
    - ``axis``: face-normal mesh-axis column (0/1/2 in the platform
      ``(z, x, y)`` column order).
    - ``area``: face area (the smaller side's face; hanging-face pairs
      contribute one record per small-cell side).
    - ``dist``: distance between cell centres along the axis.

    This is a THIN ADAPTER over the mesh's own
    :meth:`OctreeMesh.face_neighbors` records (``axis`` recovered from the
    axis-aligned unit normal; ``dist = dist_l + dist_r``), the
    ``O(n_cells²)`` numpy scan is retired; its arithmetic survives as the
    reference oracle in the octree test suite.
    The record ORDER is unchanged (the mesh-side derivation reproduces the
    legacy emission order deterministically).
    """
    if not isinstance(mesh, OctreeMesh):
        raise GeoBrainError(
            "_find_octree_face_neighbors requires an OctreeMesh",
            object_name="_find_octree_face_neighbors",
            field="mesh",
            expected="OctreeMesh",
            actual=type(mesh).__name__,
        )
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "_find_octree_face_neighbors requires a 3-D OctreeMesh",
            object_name="_find_octree_face_neighbors",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    fr = mesh.face_neighbors()
    axis = fr.normal.abs().argmax(dim=1)
    dist = fr.dist_l + fr.dist_r
    return [
        (int(i), int(j), int(a), float(ar), float(d))
        for i, j, a, ar, d in zip(
            fr.cell_i.tolist(), fr.cell_j.tolist(), axis.tolist(),
            fr.area.tolist(), dist.tolist(),
        )
    ]


def assemble_poisson_octree_3d(
    mesh: OctreeMesh,
    sigma: torch.Tensor,
    *,
    dirichlet_idx: int = 0,
) -> torch.Tensor:
    """
    Assemble ``-∇·(σ∇)`` on a 3-D :class:`OctreeMesh` as a torch sparse tensor.

    TPFA cell-centered FV with harmonic-mean face conductivities::

        σ_face = 2 σ_l σ_r / (σ_l + σ_r)
        T_f    = σ_face · A_face / dist

    Each interior face stamps the symmetric ``±T`` quad. Hanging faces
    contribute one record per (small, big) pair via
    :func:`_find_octree_face_neighbors`: the smallface area + asymmetric
    cell-centre distance recover the standard first-order TPFA hanging-face
    stencil.

    The Dirichlet pin at ``dirichlet_idx`` is **symmetric**: row and column
    of the pin index are zeroed and the diagonal is set to 1, mirroring
    :func:`assemble_poisson_3d` (TensorMesh path) so the resulting matrix
    is drop-in for :func:`geobrain.core.sparse_linear_solve_with_adjoint`.

    Args:
        mesh:          finalized :class:`OctreeMesh` (after the last
            :meth:`OctreeMesh.refine` call).
        sigma:         cell-centered σ tensor of shape ``(mesh.n_cells,)``,
            ``float64`` (DC / IP path) or ``complex128`` (SIP path with
            ``σ_eff(ω) = σ(1 − η(ω))``). Autograd flows through ``sigma``.
        dirichlet_idx: cell index where φ is pinned to 0 (breaks the
            Neumann nullspace).

    Returns:
        ``A`` as a torch sparse COO tensor of shape ``(n_cells, n_cells)``.
        ``A.values()`` is a torch op chain on ``sigma`` so autograd flows
        back through assembly.
    """
    if not isinstance(mesh, OctreeMesh):
        raise GeoBrainError(
            "assemble_poisson_octree_3d requires an OctreeMesh",
            object_name="assemble_poisson_octree_3d",
            field="mesh",
            expected="OctreeMesh",
            actual=type(mesh).__name__,
        )
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "assemble_poisson_octree_3d requires a 3-D OctreeMesh",
            object_name="assemble_poisson_octree_3d",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )

    n_cells = int(mesh.n_cells)
    if sigma.ndim != 1 or sigma.shape[0] != n_cells:
        raise GeoBrainError(
            "sigma must be 1-D with length mesh.n_cells",
            object_name="assemble_poisson_octree_3d",
            field="sigma.shape",
            expected=(n_cells,),
            actual=tuple(sigma.shape),
        )
    if not (0 <= int(dirichlet_idx) < n_cells):
        raise GeoBrainError(
            "dirichlet_idx out of range",
            object_name="assemble_poisson_octree_3d",
            field="dirichlet_idx",
            expected=f"[0, {n_cells})",
            actual=dirichlet_idx,
        )

    device = sigma.device
    dtype = sigma.dtype
    eps = 1e-30

    fr = mesh.face_neighbors()
    n_faces = int(fr.cell_i.shape[0])

    if n_faces == 0:
        # Degenerate single-cell mesh: only the pin row remains.
        idx = torch.tensor(
            [[int(dirichlet_idx)], [int(dirichlet_idx)]],
            dtype=torch.long, device=device,
        )
        vals = torch.ones(1, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            idx, vals, size=(n_cells, n_cells)
        ).coalesce()

    # Consume the mesh's FaceRecords directly: same values and order
    # as the retired per-pair scan (dist = dist_l + dist_r), no O(n²) hotspot.
    rec_i = fr.cell_i.to(device=device)
    rec_j = fr.cell_j.to(device=device)
    rec_area = fr.area.to(device=device, dtype=torch.float64)
    rec_dist = (fr.dist_l + fr.dist_r).to(device=device, dtype=torch.float64)
    if sigma.is_complex():
        rec_area = rec_area.to(torch.complex128)
        rec_dist = rec_dist.to(torch.complex128)
    else:
        rec_area = rec_area.to(dtype)
        rec_dist = rec_dist.to(dtype)

    s_l = sigma[rec_i]
    s_r = sigma[rec_j]
    sigma_face = 2.0 * s_l * s_r / (s_l + s_r + eps)
    T = sigma_face * rec_area / rec_dist

    pin = int(dirichlet_idx)

    # Stamp the symmetric ``±T`` quad per record, dropping any entry whose
    # row OR column equals the pin (we restore the pin row/col explicitly
    # below to honour the symmetric Dirichlet condition).
    rows_quad = torch.cat([rec_i, rec_j, rec_i, rec_j])
    cols_quad = torch.cat([rec_i, rec_j, rec_j, rec_i])
    vals_quad = torch.cat([+T, +T, -T, -T])

    keep = (rows_quad != pin) & (cols_quad != pin)
    rows_kept = rows_quad[keep]
    cols_kept = cols_quad[keep]
    vals_kept = vals_quad[keep]

    pin_idx = torch.tensor([pin], dtype=torch.long, device=device)
    pin_val = torch.ones(1, dtype=dtype, device=device)

    rows_all = torch.cat([rows_kept, pin_idx])
    cols_all = torch.cat([cols_kept, pin_idx])
    vals_all = torch.cat([vals_kept, pin_val])

    indices = torch.stack([rows_all, cols_all], dim=0)
    return torch.sparse_coo_tensor(
        indices, vals_all, size=(n_cells, n_cells),
    ).coalesce()


def solve_poisson_octree_3d(
    mesh: OctreeMesh,
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    *,
    dirichlet_idx: int = 0,
) -> torch.Tensor:
    """
    Convenience wrapper: assemble + solve ``A(σ) φ = rhs`` on octree.

    Autograd flows through both ``sigma`` (via the assembled ``A.values()``)
    and ``rhs`` (via :func:`sparse_linear_solve_with_adjoint`).
    """
    A = assemble_poisson_octree_3d(mesh, sigma, dirichlet_idx=dirichlet_idx)
    return sparse_linear_solve_with_adjoint(A, rhs)
