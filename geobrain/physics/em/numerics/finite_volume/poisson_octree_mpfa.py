# pyright: reportPrivateImportUsage=false
"""
MPFA-O Poisson assembly on a 3-D :class:`OctreeMesh` (EM3).

Mirrors the :func:`build_octree_poisson_stiffness(method='mpfa')` but
ported to the flat-leaf :class:`OctreeMesh` (no structured
``(ix, iy, iz, level)`` indices).

Key design choices:

- **Conforming faces** (adjacent cells share a full face) use the same
  TPFA harmonic-mean stencil that
  :func:`assemble_poisson_octree_3d` ships, no change in accuracy or
  numerical behaviour for unrefined regions.
- **Hanging faces** (a coarse cell touches 4 fine cells on one side of
  one of its 6 faces, because each split produces 8 equal children whose
  faces align with the parent's) are handled by the symmetric 5x5
  MPFA-O block (:func:`_hanging_mpfa_block`). The block
  eliminates the face-centre potential via flux continuity so the result
  is second-order accurate at refinement boundaries.
- Hanging-face groups are detected geometrically by walking the same
  per-axis face-coincidence sweep used in
  :func:`_find_octree_face_neighbors` and clustering cells whose face
  plane and orthogonal extent overlap with a strictly-larger neighbour.

This port currently supports:

- Balanced (one-level-jump) 3-D octrees with all hanging-face groups
  having exactly 4 fine cells per coarse face. Unbalanced meshes raise.
- Real or complex σ. The 5×5 MPFA-O block is the harmonic-style
  combination ``σ_M σ_N / (σ_M + σ_N)``, algebraically valid for
  complex σ via complex division, so the assembler promotes
  ``A_coo`` to ``complex128`` when σ is complex. This unblocks
  SIP-style (Cole-Cole) runs on octree meshes.

For higher-order or genuinely multi-level-jump octrees a future
extension is required; those cases are not supported.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from geobrain.mesh import OctreeMesh
from geobrain.core.errors import GeoBrainError


__all__ = ["assemble_poisson_octree_3d_mpfa"]


# ---------------------------------------------------------------------------
# Geometric face-group discovery
# ---------------------------------------------------------------------------


def _find_octree_face_groups_mpfa(
    mesh: OctreeMesh,
) -> Tuple[List[Tuple[int, int, int, float, float]], List[dict]]:
    """
    Partition octree internal faces into ``(conforming, hanging)`` groups.

    Conforming faces follow the same record format as
    :func:`_find_octree_face_neighbors`. Hanging-face groups are dicts
    with keys ``coarse_cell``, ``fine_cells`` (list of 4 leaf indices),
    ``axis``, ``d_coarse``, ``d_fine``, ``A_fine``.

    Constraints:

    - 3-D mesh.
    - Balanced octree: every hanging-face group must contain exactly
      four fine cells. Multi-level jumps or boundary anomalies raise.
    """
    if not isinstance(mesh, OctreeMesh):
        raise GeoBrainError(
            "_find_octree_face_groups_mpfa requires an OctreeMesh",
            object_name="_find_octree_face_groups_mpfa",
            field="mesh",
            expected="OctreeMesh",
            actual=type(mesh).__name__,
        )
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "_find_octree_face_groups_mpfa requires a 3-D OctreeMesh",
            object_name="_find_octree_face_groups_mpfa",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )

    centers = mesh.cell_centers().to(torch.float64).cpu().numpy()
    half = mesh.half_widths.to(torch.float64).cpu().numpy()

    eps = 1e-9 * float(half.min() + 1e-300)

    conforming: list = []
    # hanging_map: (coarse_idx, axis, face_coord_rounded, sign) -> group dict.
    hanging_map: dict = {}

    # Single source of face pairs: the shared record stream from
    # ``_find_octree_face_neighbors`` (itself a thin adapter over the mesh's
    # ``face_neighbors()`` records): this module's own copy of the
    # ``O(n_cells²)`` scan is retired. Classification below is unchanged.
    from .poisson_octree import _find_octree_face_neighbors

    for i, j, axis, area, dist in _find_octree_face_neighbors(mesh):
        # Conforming (equal-size cells on this face).
        half_i = float(half[i, axis])
        half_j = float(half[j, axis])
        if abs(half_i - half_j) < eps:
            conforming.append((int(i), int(j), int(axis), area, dist))
            continue

        # Hanging: the larger cell is coarse, smaller is fine.
        if half_i > half_j:
            coarse_idx = int(i)
            fine_idx = int(j)
        else:
            coarse_idx = int(j)
            fine_idx = int(i)

        # Group key: (coarse, axis, face-plane-coord, sign).
        # ``sign`` says whether the hanging face is on the +axis
        # or -axis side of the coarse cell.
        c_centre = centers[coarse_idx]
        f_centre = centers[fine_idx]
        sign = 1.0 if f_centre[axis] > c_centre[axis] else -1.0
        half_coarse = float(half[coarse_idx, axis])
        face_coord = c_centre[axis] + sign * half_coarse
        # Round face_coord to a stable key (face planes geometrically
        # coincide exactly across the 4 fine children; the rounding guards
        # against minor numerical drift in constructions that don't use
        # bit-identical half-widths).
        fc_key = round(face_coord / max(eps, 1e-12)) * max(eps, 1e-12)
        gkey = (coarse_idx, int(axis), fc_key, float(sign))
        entry = hanging_map.setdefault(
            gkey,
            {
                "coarse_cell": coarse_idx,
                "fine_cells": [],
                "axis": int(axis),
                "d_coarse": half_coarse,
                "d_fine": float(half[fine_idx, axis]),
                "A_fine": area,  # area on the fine side = 1 fine cell's face
            },
        )
        if fine_idx not in entry["fine_cells"]:
            entry["fine_cells"].append(fine_idx)

    hanging: list = []
    for g in hanging_map.values():
        if len(g["fine_cells"]) != 4:
            raise GeoBrainError(
                f"hanging-face group has {len(g['fine_cells'])} fine cells "
                f"(expected 4); mesh may be unbalanced",
                object_name="_find_octree_face_groups_mpfa",
                field="fine_cells",
                expected=4,
                actual=len(g["fine_cells"]),
            )
        hanging.append(g)
    return conforming, hanging


def _hanging_mpfa_block(
    sigma_C, sigma_F, group: dict,
) -> np.ndarray:
    """
    Symmetric 5×5 MPFA stiffness block for one hanging face.

    Rows/cols are ordered ``[C, F1, F2, F3, F4]``.

    ``sigma_C`` and ``sigma_F`` may be ``float64`` (DC/IP path) or
    ``complex128`` (SIP path); the block is emitted in the promoted
    dtype. The formula uses only ``+``, ``*``, ``/`` so it generalises
    to complex σ without modification.
    """
    sigma_F = np.asarray(sigma_F)
    is_complex = (
        np.iscomplexobj(sigma_F) or isinstance(sigma_C, complex)
        or (hasattr(sigma_C, "dtype") and np.iscomplexobj(sigma_C))
    )
    block_dtype = np.complex128 if is_complex else np.float64
    if is_complex:
        sigma_F = sigma_F.astype(np.complex128, copy=False)
        sigma_C = complex(sigma_C)
    else:
        sigma_F = sigma_F.astype(np.float64, copy=False)
        sigma_C = float(sigma_C)
    A_C = 4.0 * group["A_fine"]
    T_C = sigma_C * A_C / group["d_coarse"]
    T = sigma_F * group["A_fine"] / group["d_fine"]
    Sigma = T_C + T.sum()

    block = np.zeros((5, 5), dtype=block_dtype)
    block[0, 0] = T_C * T.sum() / Sigma
    for i in range(4):
        Ti = T[i]
        block[0, 1 + i] = -T_C * Ti / Sigma
        block[1 + i, 0] = block[0, 1 + i]
        sum_others = T.sum() - Ti
        block[1 + i, 1 + i] = Ti * (T_C + sum_others) / Sigma
        for j in range(4):
            if i == j:
                continue
            block[1 + i, 1 + j] = -Ti * T[j] / Sigma
    return block


# ---------------------------------------------------------------------------
# Top-level assembler
# ---------------------------------------------------------------------------


def assemble_poisson_octree_3d_mpfa(
    mesh: OctreeMesh,
    sigma: torch.Tensor,
    *,
    dirichlet_idx: int = 0,
) -> torch.Tensor:
    """
    Assemble ``-∇·(σ∇)`` on a 3-D :class:`OctreeMesh` with MPFA-O hanging faces.

    Cell-centred FV with the standard harmonic-mean TPFA stencil on
    conforming faces, and the symmetric 5×5 MPFA-O block
    (:func:`_hanging_mpfa_block`) on hanging faces. Second-order accurate
    at refinement boundaries (vs. the first-order TPFA fallback that
    :func:`assemble_poisson_octree_3d` uses).

    The Dirichlet pin at ``dirichlet_idx`` is symmetric (row and column
    zeroed, diagonal entry set to 1), matching the TPFA octree path.

    Args:
        mesh: finalised 3-D :class:`OctreeMesh`. Must be balanced (one-
            level-jump octree); unbalanced meshes raise.
        sigma: cell-centred σ tensor of shape ``(mesh.n_cells,)``.
            Either ``float64`` (DC/IP path) or ``complex128`` (SIP /
            Cole-Cole path). When σ is complex the output ``A_coo`` is
            also complex; the harmonic-style block formula
            ``σ_M σ_N / (σ_M + σ_N)`` flows through complex arithmetic
            unchanged.
        dirichlet_idx: cell index where φ is pinned to 0.

    Returns:
        ``A`` as a torch sparse COO tensor of shape ``(n_cells, n_cells)``.
        Output dtype mirrors ``sigma.dtype`` (real σ → real ``A``;
        complex σ → complex ``A``). ``A.values()`` carries the autograd
        graph through σ (via the TPFA harmonic-mean fragment); MPFA-O
        contributions are assembled directly from σ in pure torch so
        autograd flows back through every block entry.
    """
    if not isinstance(mesh, OctreeMesh):
        raise GeoBrainError(
            "assemble_poisson_octree_3d_mpfa requires an OctreeMesh",
            object_name="assemble_poisson_octree_3d_mpfa",
            field="mesh",
            expected="OctreeMesh",
            actual=type(mesh).__name__,
        )
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "assemble_poisson_octree_3d_mpfa requires a 3-D OctreeMesh",
            object_name="assemble_poisson_octree_3d_mpfa",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    n_cells = int(mesh.n_cells)
    if sigma.ndim != 1 or sigma.shape[0] != n_cells:
        raise GeoBrainError(
            "sigma must be 1-D with length mesh.n_cells",
            object_name="assemble_poisson_octree_3d_mpfa",
            field="sigma.shape",
            expected=(n_cells,),
            actual=tuple(sigma.shape),
        )
    if not (0 <= int(dirichlet_idx) < n_cells):
        raise GeoBrainError(
            "dirichlet_idx out of range",
            object_name="assemble_poisson_octree_3d_mpfa",
            field="dirichlet_idx",
            expected=f"[0, {n_cells})",
            actual=dirichlet_idx,
        )

    device = sigma.device
    dtype = sigma.dtype
    eps = 1e-30

    conforming, hanging = _find_octree_face_groups_mpfa(mesh)

    pin = int(dirichlet_idx)

    rows_chunks: list[torch.Tensor] = []
    cols_chunks: list[torch.Tensor] = []
    vals_chunks: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Conforming faces: TPFA harmonic-mean.
    # ------------------------------------------------------------------
    if conforming:
        rec_i = torch.tensor(
            [r[0] for r in conforming], dtype=torch.long, device=device,
        )
        rec_j = torch.tensor(
            [r[1] for r in conforming], dtype=torch.long, device=device,
        )
        rec_area = torch.tensor(
            [r[3] for r in conforming], dtype=torch.float64, device=device,
        ).to(dtype)
        rec_dist = torch.tensor(
            [r[4] for r in conforming], dtype=torch.float64, device=device,
        ).to(dtype)
        s_l = sigma[rec_i]
        s_r = sigma[rec_j]
        sigma_face = 2.0 * s_l * s_r / (s_l + s_r + eps)
        T = sigma_face * rec_area / rec_dist
        # Stamp the symmetric ±T quad.
        rows_chunks.append(torch.cat([rec_i, rec_j, rec_i, rec_j]))
        cols_chunks.append(torch.cat([rec_i, rec_j, rec_j, rec_i]))
        vals_chunks.append(torch.cat([+T, +T, -T, -T]))

    # ------------------------------------------------------------------
    # Hanging-face groups: 5×5 MPFA-O block per group.
    # ------------------------------------------------------------------
    for group in hanging:
        c = int(group["coarse_cell"])
        fs = [int(f) for f in group["fine_cells"]]
        A_fine = float(group["A_fine"])
        d_coarse = float(group["d_coarse"])
        d_fine = float(group["d_fine"])

        sigma_C = sigma[c]
        sigma_F = sigma[torch.tensor(fs, dtype=torch.long, device=device)]

        # Torch-native build of the 5×5 block so autograd flows through σ.
        A_C = 4.0 * A_fine
        T_C = sigma_C * A_C / d_coarse  # scalar tensor
        T = sigma_F * A_fine / d_fine    # (4,)
        T_sum = T.sum()
        Sigma = T_C + T_sum

        # block[0, 0] = T_C * T.sum() / Sigma
        # block[0, 1+i] = -T_C * T[i] / Sigma
        # block[1+i, 1+i] = T[i] * (T_C + sum_others) / Sigma
        # block[1+i, 1+j] = -T[i] * T[j] / Sigma  (i != j)
        idxs = torch.tensor([c, *fs], dtype=torch.long, device=device)

        # Coarse-coarse entry.
        rows_chunks.append(idxs[0:1])
        cols_chunks.append(idxs[0:1])
        vals_chunks.append((T_C * T_sum / Sigma).reshape(1))

        # Coarse-fine and fine-coarse (symmetric).
        for i in range(4):
            block_0i = -T_C * T[i] / Sigma
            rows_chunks.append(idxs[0:1])
            cols_chunks.append(idxs[i + 1:i + 2])
            vals_chunks.append(block_0i.reshape(1))
            rows_chunks.append(idxs[i + 1:i + 2])
            cols_chunks.append(idxs[0:1])
            vals_chunks.append(block_0i.reshape(1))

        # Fine-fine diagonal + off-diagonal.
        for i in range(4):
            sum_others = T_sum - T[i]
            diag_ii = T[i] * (T_C + sum_others) / Sigma
            rows_chunks.append(idxs[i + 1:i + 2])
            cols_chunks.append(idxs[i + 1:i + 2])
            vals_chunks.append(diag_ii.reshape(1))
            for j in range(4):
                if i == j:
                    continue
                off_ij = -T[i] * T[j] / Sigma
                rows_chunks.append(idxs[i + 1:i + 2])
                cols_chunks.append(idxs[j + 1:j + 2])
                vals_chunks.append(off_ij.reshape(1))

    if not rows_chunks:
        # Degenerate single-cell mesh: only the pin row remains.
        idx = torch.tensor(
            [[pin], [pin]], dtype=torch.long, device=device,
        )
        vals = torch.ones(1, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(
            idx, vals, size=(n_cells, n_cells)
        ).coalesce()

    rows_all = torch.cat(rows_chunks)
    cols_all = torch.cat(cols_chunks)
    vals_all = torch.cat(vals_chunks)

    # Drop entries on the pinned row OR column (we restore the pin
    # row/col explicitly below). Mirrors the symmetric Dirichlet pin
    # convention used by :func:`assemble_poisson_octree_3d`.
    keep = (rows_all != pin) & (cols_all != pin)
    rows_kept = rows_all[keep]
    cols_kept = cols_all[keep]
    vals_kept = vals_all[keep]

    pin_idx = torch.tensor([pin], dtype=torch.long, device=device)
    pin_val = torch.ones(1, dtype=dtype, device=device)

    rows_out = torch.cat([rows_kept, pin_idx])
    cols_out = torch.cat([cols_kept, pin_idx])
    vals_out = torch.cat([vals_kept, pin_val])

    indices = torch.stack([rows_out, cols_out], dim=0)
    return torch.sparse_coo_tensor(
        indices, vals_out, size=(n_cells, n_cells),
    ).coalesce()
