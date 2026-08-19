"""
Boundary-edge topology of a tetrahedral EdgeConnectivityMesh.

The one derived-topology fact the edge-element curl-curl family needs beyond
the mesh's own records: WHICH global edges lie on the outer boundary, so the
operators can pin them (PEC: tangential E = 0). Derivation is purely
combinatorial over ``cell_nodes()``, every tetrahedron contributes its four
faces as sorted node triples; a triple owned by exactly one cell is a boundary
face, and the three node-pair edges of those faces are the boundary edges.
Sorted triples make the three pairs already canonical (small node id → large),
so they reverse-look-up directly into ``edge_records().nodes``. Everything is
vectorised (``torch.unique`` + ``searchsorted``); no Python loop over cells.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from geobrain.core.errors import GeoBrainError
from geobrain.mesh import ensure_capable_mesh
from geobrain.mesh.capabilities import EdgeConnectivityMesh

__all__ = [
    "PecPinPlan",
    "apply_pec_pin",
    "boundary_edge_mask",
    "build_pec_pin_plan",
    "pec_zero_rhs",
]

# The four triangular faces of a tetrahedron over its STORED node order, each
# is the three nodes left after dropping one (same enumeration as the mesh's
# FV face derivation, ``geobrain.mesh._unstructured_builders.derive_faces_3d``).
_TET_LOCAL_FACES: torch.Tensor = torch.tensor(
    [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=torch.long,
)

# The three edges of a triangle whose nodes are SORTED ascending: each pair is
# then automatically in the canonical small-id → large-id direction.
_TRI_LOCAL_EDGES: torch.Tensor = torch.tensor(
    [[0, 1], [0, 2], [1, 2]], dtype=torch.long,
)


def boundary_edge_mask(mesh) -> torch.Tensor:
    """Boolean mask over ``edge_records()`` marking the outer-boundary edges.

    A global edge is a *boundary edge* when it lies on at least one boundary
    face, a triangular face owned by exactly one tetrahedron. These are the
    degrees of freedom a PEC (perfect electric conductor) truncation pins to
    zero in the edge-element curl-curl family (tangential E = 0 on the outer
    surface).

    Pipeline (all vectorised):

    1. ``cell_nodes()`` ``(nc, 4)`` → four faces per cell as node triples,
       sorted ascending per triple → ``(nc·4, 3)``;
    2. ``torch.unique(dim=0, return_counts=True)``; triples with count 1 are
       boundary faces (count 2 are interior; anything else is non-manifold);
    3. each boundary triple's three node pairs are canonical edges (the triple
       is sorted), deduplicated and reverse-looked-up into
       ``edge_records().nodes`` via a scalar-key ``searchsorted``.

    Args:
        mesh: a mesh declaring
            :class:`~geobrain.mesh.capabilities.EdgeConnectivityMesh`
            (instance-attribute declaration read first, class fallback, the
            ``require_mesh`` rule).

    Returns:
        ``(n_edges,)`` bool tensor, ``True`` on boundary edges, aligned with
        ``mesh.edge_records()``.

    Raises:
        GeoBrainError: when the mesh does not declare ``EdgeConnectivityMesh``,
            or when a boundary-face edge cannot be resolved against
            ``edge_records()`` (inconsistent mesh topology).
    """
    ensure_capable_mesh(mesh, EdgeConnectivityMesh, owner="boundary_edge_mask")
    cn = mesh.cell_nodes()                                   # (nc, 4) long
    nodes = mesh.edge_records().nodes                        # (ne, 2) canonical
    n_edges = int(nodes.shape[0])
    device = cn.device

    faces = cn[:, _TET_LOCAL_FACES.to(device)]               # (nc, 4, 3)
    faces = faces.sort(dim=-1).values.reshape(-1, 3)         # sorted triples
    uniq, counts = torch.unique(faces, dim=0, return_counts=True)
    boundary_tris = uniq[counts == 1]                        # (nb, 3)
    if int(boundary_tris.shape[0]) == 0:
        if int(cn.shape[0]) > 0:
            # A non-empty tetrahedral mesh embedded in R^3 always has a
            # boundary (it cannot tile a closed 3-manifold). An empty
            # boundary-face set therefore means the cell list is broken,
            # typically duplicated (each face counted twice) or mutually
            # contained cells, and a silently all-False mask would make the
            # PEC condition (pin tangential E = 0 on the outer surface)
            # vanish without a trace.
            raise GeoBrainError(
                "no boundary faces found on a non-empty tetrahedral mesh, "
                "the mesh likely contains duplicated or mutually contained "
                "cells, so the PEC boundary (tangential E = 0) cannot be "
                "defined",
                object_name=type(mesh).__name__, field="cell_nodes",
                expected="at least one face owned by exactly one cell",
                actual="every face owned by >= 2 cells",
            )
        return torch.zeros(n_edges, dtype=torch.bool, device=device)

    pairs = boundary_tris[:, _TRI_LOCAL_EDGES.to(device)]    # (nb, 3, 2)
    pairs = torch.unique(pairs.reshape(-1, 2), dim=0)        # canonical, deduped

    # Reverse lookup by scalar key lo·n_key + hi (unique since hi < n_key).
    # ``edge_records().nodes`` is lexicographically ordered by contract, but
    # sort the keys anyway so the lookup never depends on that ordering.
    n_key = int(nodes.max()) + 1
    key_all = nodes[:, 0] * n_key + nodes[:, 1]              # (ne,)
    key_sorted, order = key_all.sort()
    key_b = pairs[:, 0] * n_key + pairs[:, 1]                # (nbe,)
    pos = torch.searchsorted(key_sorted, key_b)
    pos = pos.clamp(max=n_edges - 1)
    if not bool((key_sorted[pos] == key_b).all()):
        raise GeoBrainError(
            "a boundary-face edge is missing from edge_records(), "
            "inconsistent cell_nodes / edge topology",
            object_name=type(mesh).__name__, field="cell_nodes",
            expected="every boundary-face node pair present in edge_records()",
            actual="unresolved node pair",
        )
    mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
    mask[order[pos]] = True
    return mask


@dataclass(frozen=True, eq=False)
class PecPinPlan:
    """σ/dt-independent scaffolding for the symmetric Dirichlet (PEC) pin.

    The boundary-edge truncation (tangential E = 0 on the outer surface) is a
    sparsity-pattern operation that does not depend on the assembled
    coefficients, so its index-level pieces precompute once per mesh and reuse
    across every assembly (per frequency / per timestep). ``eq=False``:
    equality/hash by object identity; a field-wise ``__eq__`` would call
    ``bool()`` on tensor comparisons and raise.

    Attributes:
        boundary_mask: ``(n_edges,)`` bool from :func:`boundary_edge_mask`.
        pinned_entry: ``(nnz,)`` bool, COO entries whose row OR column is a
            boundary edge (the entries the pin zeros).
        pin_indices: ``(2, n_pin)`` long, appended unit-diagonal slots
            ``(p, p)`` for every boundary edge ``p``.
    """

    boundary_mask: torch.Tensor
    pinned_entry: torch.Tensor
    pin_indices: torch.Tensor


def build_pec_pin_plan(
    boundary_mask: torch.Tensor,
    indices: torch.Tensor,
) -> PecPinPlan:
    """Precompute the coefficient-independent PEC-pin scaffolding.

    Args:
        boundary_mask: ``(n_edges,)`` bool from :func:`boundary_edge_mask`.
        indices: ``(2, nnz)`` long COO index pattern of the assembled operator
            (``plan.rows``/``plan.cols`` stacked; fixed for the mesh).

    Returns:
        A frozen :class:`PecPinPlan`.
    """
    pinned_entry = boundary_mask[indices[0]] | boundary_mask[indices[1]]
    pin_ids = boundary_mask.nonzero(as_tuple=True)[0]
    pin_indices = torch.stack([pin_ids, pin_ids])
    return PecPinPlan(
        boundary_mask=boundary_mask,
        pinned_entry=pinned_entry,
        pin_indices=pin_indices,
    )


def apply_pec_pin(
    pin_plan: PecPinPlan,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the symmetric Dirichlet (PEC) pin to one assembled COO operator.

    The DC3D pattern: zero every COO entry whose row OR column is a boundary
    edge; the sparsity pattern is preserved and the surviving values stay
    differentiable (``torch.where``), then append unit diagonal entries for
    the pinned dofs (duplicates are summed by the solve seam's coalesce; the
    pinned slots hold no other surviving entry). The appended ones take the
    operator's value dtype, so the operator's complex (FDEM3D/MT3D) vs real
    (TEM3D) system is handled with no per-caller divergence.

    Args:
        pin_plan: the mesh's :class:`PecPinPlan` from :func:`build_pec_pin_plan`.
        indices: ``(2, nnz)`` long COO indices (the plan's fixed pattern).
        values: ``(nnz,)`` assembled COO values (float64 or complex128).

    Returns:
        ``(pinned_indices, pinned_values)``: the input pattern with pinned
        entries zeroed and unit-diagonal pin entries appended.
    """
    values = torch.where(
        pin_plan.pinned_entry, torch.zeros((), dtype=values.dtype), values,
    )
    n_pin = int(pin_plan.pin_indices.shape[1])
    pinned_indices = torch.cat([indices, pin_plan.pin_indices], dim=1)
    pinned_values = torch.cat(
        [values, torch.ones(n_pin, dtype=values.dtype)],
    )
    return pinned_indices, pinned_values


def pec_zero_rhs(
    boundary_mask: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Zero the boundary-edge rows of a PEC right-hand side.

    The RHS counterpart of :func:`apply_pec_pin`: tangential E = 0 on the
    outer boundary means the pinned rows carry no forcing. ``torch.where``
    keeps the surviving rows differentiable.

    Args:
        boundary_mask: ``(n_edges,)`` bool from :func:`boundary_edge_mask`.
        rhs: ``(n_edges,)`` or ``(n_edges, k)`` right-hand side.

    Returns:
        ``rhs`` with boundary-edge rows set to zero, same shape and dtype.
    """
    mask = boundary_mask if rhs.dim() == 1 else boundary_mask.unsqueeze(-1)
    return torch.where(mask, torch.zeros((), dtype=rhs.dtype), rhs)
