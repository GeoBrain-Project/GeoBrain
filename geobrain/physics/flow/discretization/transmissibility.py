"""
Full-tensor / unstructured-grid transmissibility (two-point).

The Cartesian ``CartGrid.build_transmissibility`` assumes axis-aligned cells and
isotropic (scalar) permeability. For **full-tensor** permeability (anisotropic,
with off-diagonal terms) or a **general / unstructured** grid (arbitrary cell and
face geometry), the half-face transmissibility is::

    T_hf = A · (N · K · C) / (C · C)

where ``A`` is the face area, ``N`` the (oriented unit) face normal, ``K`` the
cell permeability tensor and ``C = face_centroid − cell_centroid`` the vector
from the cell centre to the face centre. The face transmissibility is the
harmonic average of its two half-faces::

    T_face = 1 / (1/T_hf_left + 1/T_hf_right)

This is consistent two-point flux on K-orthogonal grids (and reduces to
``A·k/d`` for axis-aligned cells + scalar perm); for strongly non-K-orthogonal
grids a multi-point (MPFA-O) scheme is the consistent extension. Permeability is
supplied per cell as 1 (isotropic), 3 (2-D ``Kxx,Kxy,Kyy``) or 6
(3-D ``Kxx,Kxy,Kxz,Kyy,Kyz,Kzz``) entries via :func:`expand_perm`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ....core import GeoBrainError
from .tpfa import _series_transmissibility


def expand_perm(perm: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-cell permeability entries → symmetric ``(..., dim, dim)`` tensor.

    Accepts ``(..., m)`` with ``m`` = 1 (isotropic), or the symmetric-tensor
    packing: 2-D ``m∈{2,3}`` (``Kxx,Kyy`` diagonal, or ``Kxx,Kxy,Kyy``), 3-D
    ``m∈{3,6}`` (diagonal ``Kxx,Kyy,Kzz``, or ``Kxx,Kxy,Kxz,Kyy,Kyz,Kzz``).
    """
    m = perm.shape[-1]
    batch = perm.shape[:-1]
    z = torch.zeros(batch, dtype=perm.dtype, device=perm.device)

    def _sym(rows: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
        return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)

    if dim == 1:
        return perm[..., :1].unsqueeze(-1)
    if dim == 2:
        if m == 1:
            kxx = kyy = perm[..., 0]
            kxy = z
        elif m == 2:
            kxx, kyy = perm[..., 0], perm[..., 1]
            kxy = z
        elif m == 3:
            kxx, kxy, kyy = perm[..., 0], perm[..., 1], perm[..., 2]
        else:
            raise GeoBrainError(
                "2-D permeability needs 1/2/3 entries per cell",
                object_name="expand_perm",
                field="perm",
                expected="1/2/3",
                actual=m,
            )
        return _sym([[kxx, kxy], [kxy, kyy]])
    if dim == 3:
        if m == 1:
            kxx = kyy = kzz = perm[..., 0]
            kxy = kxz = kyz = z
        elif m == 3:
            kxx, kyy, kzz = perm[..., 0], perm[..., 1], perm[..., 2]
            kxy = kxz = kyz = z
        elif m == 6:
            kxx, kxy, kxz, kyy, kyz, kzz = (perm[..., i] for i in range(6))
        else:
            raise GeoBrainError(
                "3-D permeability needs 1/3/6 entries per cell",
                object_name="expand_perm",
                field="perm",
                expected="1/3/6",
                actual=m,
            )
        return _sym([[kxx, kxy, kxz], [kxy, kyy, kyz], [kxz, kyz, kzz]])
    raise GeoBrainError(
        "dim must be 1, 2 or 3",
        object_name="expand_perm",
        field="dim",
        expected="1/2/3",
        actual=dim,
    )


def half_face_transmissibility(
    cell_centroids: torch.Tensor,
    face_centroids: torch.Tensor,
    face_normals: torch.Tensor,
    face_areas: torch.Tensor,
    perm_tensor: torch.Tensor,
    neighbors: torch.Tensor,
) -> torch.Tensor:
    """Per-face ``(T_hf_left, T_hf_right)`` half-face transmissibilities.

    Args (SI / consistent units):
        cell_centroids: ``(n_cells, dim)``.
        face_centroids / face_normals: ``(n_faces, dim)`` (normals oriented
            from the left to the right neighbour; magnitude is normalized here).
        face_areas: ``(n_faces,)``.
        perm_tensor: ``(n_cells, dim, dim)`` (from :func:`expand_perm`).
        neighbors: ``(n_faces, 2)`` left/right cell indices.

    Returns: ``(n_faces, 2)`` with the half-face transmissibility on each side.
    """
    nrm = face_normals / face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    out = []
    for side in (0, 1):
        cells = neighbors[:, side]
        C = face_centroids - cell_centroids[cells]  # (n_faces, dim)
        K = perm_tensor[cells]  # (n_faces, dim, dim)
        KC = torch.einsum("fij,fj->fi", K, C)
        n_dot_kc = (nrm * KC).sum(dim=-1)
        T_hf = face_areas * n_dot_kc.abs() / (C * C).sum(dim=-1).clamp_min(1e-300)
        out.append(T_hf)
    return torch.stack(out, dim=-1)


def full_tensor_face_transmissibility(
    cell_centroids: torch.Tensor,
    face_centroids: torch.Tensor,
    face_normals: torch.Tensor,
    face_areas: torch.Tensor,
    perm_tensor: torch.Tensor,
    neighbors: torch.Tensor,
) -> torch.Tensor:
    """Face transmissibility ``(n_faces,)``: harmonic average of the half-faces.

    Differentiable w.r.t. ``perm_tensor`` and geometry.
    """
    T_hf = half_face_transmissibility(
        cell_centroids, face_centroids, face_normals, face_areas, perm_tensor, neighbors
    )
    # Product-free series combination avoids overflow at extreme permeability
    # while retaining exact zero for sealed faces and finite gradients.
    return _series_transmissibility(T_hf[:, 0], T_hf[:, 1])


__all__ = ["expand_perm", "half_face_transmissibility", "full_tensor_face_transmissibility"]
