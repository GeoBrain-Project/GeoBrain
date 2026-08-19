"""
Shared σ-independent Whitney edge-element asset bundle.

The three 3-D inductive operators (TEM3D / FDEM3D / MT3D) each cache a frozen
bundle of the σ-independent Nédélec/Whitney assets of one mesh, reused across
every BDF substep (TEM) or frequency (FDEM/MT). The eight fields below are
identical across all three; each operator subclasses this base and adds only its
own extras (TEM: the BDF schedule + gate map; MT: the background-σ profile).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from geobrain.mesh.capabilities import EdgeRecords

from .assembly import EdgeAssemblyPlan


@dataclass(frozen=True)
class EdgeFemAssetsBase:
    """σ-independent Whitney assets of one mesh (see each operator's
    ``_edge_fem_assets``).

    Attributes:
        plan: the mesh's :class:`EdgeAssemblyPlan` (geometric factors + COO
            index pattern), reused across every BDF substep / frequency.
        boundary_mask: ``(n_edges,)`` bool, outer-boundary edges (PEC pins).
        edge_records: the mesh's :class:`EdgeRecords` (canonical node pairs,
            tangents, lengths, midpoints) for the primary-field projection.
        cell_edge_ids: ``(n_cells, 6)`` long, global edge ids per cell.
        cell_edge_signs: ``(n_cells, 6)`` float64, local→canonical signs.
        w_centroid: ``(n_cells, 6, 3)``, Whitney basis vectors at the cell
            barycentre, ``W_e(centroid) = (g_j - g_i)/4`` (float64 in the time
            domain, complex128 in the frequency domain).
        curl_w: ``(n_cells, 6, 3)``, constant per-cell curls,
            ``curl W_e = 2 g_i × g_j`` (dtype as ``w_centroid``).
        cell_centers: ``(n_cells, 3)`` float64, receiver/station-binding points.
    """

    plan: EdgeAssemblyPlan
    boundary_mask: torch.Tensor
    edge_records: EdgeRecords
    cell_edge_ids: torch.Tensor
    cell_edge_signs: torch.Tensor
    w_centroid: torch.Tensor
    curl_w: torch.Tensor
    cell_centers: torch.Tensor
