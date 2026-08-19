"""
Lowest-order Nédélec (Whitney) edge-element local matrices on tetrahedra.

Batched, loop-free geometry + local 6×6 matrices for the curl-curl EM family
(FDEM3D/TEM3D/MT3D unstructured branches). The mesh side of the contract is
:class:`~geobrain.mesh.capabilities.EdgeConnectivityMesh`: the local edge
ordering here is EXACTLY the one ``cell_edges()`` documents, over the cell's
STORED node order, the six edges are the node pairs
``(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)`` with local direction first → second
node (:data:`TET_LOCAL_EDGES`).

Math (per tetrahedron ``T`` with nodes ``p_0..p_3``):

- Barycentric coordinates ``λ_a(x)`` are affine with CONSTANT gradients
  ``g_a = ∇λ_a``. With the 4×4 node matrix ``A[k, :] = [1, p_k]`` one has
  ``λ_a(x) = B[0, a] + B[1:, a]·x`` for ``B = A⁻¹`` (since ``λ_a(p_k) = δ_ak``),
  so ``g_a = B[1:, a]``, one batched ``inv`` over ``(nc, 4, 4)``. The volume is
  ``V = |det A| / 6``; the ABSOLUTE value matters because GeoBrain does not
  guarantee positively-oriented cell node order, and every formula below is
  even in the orientation sign (``g`` flips with ``det`` but only ``g·g`` and
  ``(g×g)·(g×g)`` products appear).
- Whitney edge basis for local edge ``e = (i, j)``:
  ``W_e = λ_i g_j − λ_j g_i`` and ``curl W_e = 2 g_i × g_j`` (constant).
- Local mass (unit σ): expanding ``W_e·W_f`` and integrating with
  ``∫_T λ_a λ_b dV = V (1 + δ_ab) / 20`` gives the closed form

  ``M[e,f] = V/20 · [ (1+δ_ik)(g_j·g_l) − (1+δ_il)(g_j·g_k)
  − (1+δ_jk)(g_i·g_l) + (1+δ_jl)(g_i·g_k) ]``   for ``e=(i,j), f=(k,l)``.

- Local stiffness (unit coefficient): the integrand is constant, so
  ``K[e,f] = 4 V (g_i×g_j)·(g_k×g_l)``.

Both closed forms are verified against exact symbolic integration on a
non-axis-aligned rational tetrahedron (machine precision) and, in the test
suite, against a degree-2 quadrature rule per cell.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core.errors import GeoBrainError

__all__ = [
    "TET_LOCAL_EDGES",
    "barycentric_gradients",
    "local_curl_curl",
    "local_edge_mass",
    "whitney_centroid_factors",
]

# The six local edges of a tetrahedron over its STORED node order, the same
# fixed table the EdgeConnectivityMesh contract pins for ``cell_edges()``
# (geobrain.mesh.unstructured._TET_LOCAL_EDGES): pairs
# (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), local direction first → second.
TET_LOCAL_EDGES: torch.Tensor = torch.tensor(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=torch.long,
)

# Constant (6, 6) Kronecker-delta coefficient masks of the closed-form mass
# formula, precomputed once from the fixed local-edge table: with edge
# e = (i_e, j_e) and f = (i_f, j_f),
#   _D_II[e, f] = 1 + δ(i_e, i_f),  _D_IJ[e, f] = 1 + δ(i_e, j_f),
#   _D_JI[e, f] = 1 + δ(j_e, i_f),  _D_JJ[e, f] = 1 + δ(j_e, j_f).
_I6 = TET_LOCAL_EDGES[:, 0]
_J6 = TET_LOCAL_EDGES[:, 1]
_D_II = 1.0 + (_I6[:, None] == _I6[None, :]).to(torch.float64)
_D_IJ = 1.0 + (_I6[:, None] == _J6[None, :]).to(torch.float64)
_D_JI = 1.0 + (_J6[:, None] == _I6[None, :]).to(torch.float64)
_D_JJ = 1.0 + (_J6[:, None] == _J6[None, :]).to(torch.float64)


def barycentric_gradients(
    node_coords: torch.Tensor,
    cell_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched barycentric-coordinate gradients + volumes of tetrahedral cells.

    Args:
        node_coords: ``(n_nodes, 3)`` node coordinates (any float dtype;
            promoted to float64).
        cell_nodes: ``(n_cells, 4)`` long cell→node table in STORED order (the
            order the local-edge convention is defined over).

    Returns:
        ``(g, volume)``: ``g`` of shape ``(n_cells, 4, 3)`` float64 with
        ``g[c, a] = ∇λ_a`` on cell ``c``, and ``volume`` of shape
        ``(n_cells,)`` float64. The volume is ``|det|/6``, orientation-free,
        since cell node order is NOT guaranteed positively oriented.

    Raises:
        GeoBrainError: on malformed shapes or a degenerate (zero-volume) cell.
    """
    if node_coords.dim() != 2 or int(node_coords.shape[1]) != 3:
        raise GeoBrainError(
            "barycentric_gradients: node_coords must be (n_nodes, 3)",
            object_name="barycentric_gradients", field="node_coords",
            expected="(n_nodes, 3)", actual=tuple(node_coords.shape),
        )
    if cell_nodes.dim() != 2 or int(cell_nodes.shape[1]) != 4:
        raise GeoBrainError(
            "barycentric_gradients: cell_nodes must be (n_cells, 4)",
            object_name="barycentric_gradients", field="cell_nodes",
            expected="(n_cells, 4)", actual=tuple(cell_nodes.shape),
        )
    coords = node_coords.to(torch.float64)
    cn = cell_nodes.to(torch.long)
    p = coords[cn]                                          # (nc, 4, 3)
    ones = torch.ones(p.shape[0], 4, 1, dtype=torch.float64, device=p.device)
    a4 = torch.cat([ones, p], dim=2)                        # (nc, 4, 4)
    det = torch.linalg.det(a4)
    volume = det.abs() / 6.0
    if bool((volume <= 0).any()) or not bool(torch.isfinite(volume).all()):
        raise GeoBrainError(
            "degenerate (zero-volume) tetrahedron in cell_nodes",
            object_name="barycentric_gradients", field="cell_nodes",
            expected="all cell volumes > 0", actual="non-positive volume",
        )
    b = torch.linalg.inv(a4)                                # (nc, 4, 4)
    # λ_a(x) = b[0, a] + b[1:, a]·x  ⇒  g[c, a, :] = b[c, 1:, a].
    g = b[:, 1:, :].transpose(1, 2).contiguous()            # (nc, 4, 3)
    return g, volume


def local_edge_mass(g: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """Local Whitney edge mass matrices at unit σ (the geometric factor).

    ``M[c, e, f] = ∫_{T_c} W_e·W_f dV`` in the closed form of the module
    docstring, one batched Gram ``einsum`` plus four constant-mask gathers,
    no per-cell loop. The cell-wise material coefficient multiplies this
    factor at assembly time (``mass_coeff`` in
    :func:`~geobrain.physics.em.numerics.edge_element.assemble_edge_operator`).

    Args:
        g: ``(n_cells, 4, 3)`` barycentric gradients from
            :func:`barycentric_gradients`.
        volume: ``(n_cells,)`` cell volumes.

    Returns:
        ``(n_cells, 6, 6)`` float64, symmetric positive-definite per cell.
    """
    gram = torch.einsum("nad,nbd->nab", g, g)               # (nc, 4, 4)
    dev = g.device
    i6 = _I6.to(dev)
    j6 = _J6.to(dev)
    ie, jf = i6[:, None], j6[None, :]
    je, if_ = j6[:, None], i6[None, :]
    m = (
        _D_II.to(dev) * gram[:, je, jf]
        - _D_IJ.to(dev) * gram[:, je, if_]
        - _D_JI.to(dev) * gram[:, ie, jf]
        + _D_JJ.to(dev) * gram[:, ie, if_]
    )
    return m * (volume.view(-1, 1, 1) / 20.0)


def local_curl_curl(g: torch.Tensor, volume: torch.Tensor) -> torch.Tensor:
    """Local curl-curl (stiffness) matrices at unit coefficient.

    ``K[c, e, f] = V_c · (curl W_e)·(curl W_f) = 4 V_c (g_i×g_j)·(g_k×g_l)``;
    the integrand is constant per cell, so this is exact. The cell-wise
    coefficient (e.g. ``1/μ``) multiplies this factor at assembly time
    (``stiffness_coeff`` in
    :func:`~geobrain.physics.em.numerics.edge_element.assemble_edge_operator`).

    Args:
        g: ``(n_cells, 4, 3)`` barycentric gradients from
            :func:`barycentric_gradients`.
        volume: ``(n_cells,)`` cell volumes.

    Returns:
        ``(n_cells, 6, 6)`` float64, symmetric positive-semidefinite per cell
        (rank 3; the discrete-gradient subspace is its kernel).
    """
    dev = g.device
    cross = torch.linalg.cross(g[:, _I6.to(dev), :], g[:, _J6.to(dev), :], dim=-1)
    return 4.0 * volume.view(-1, 1, 1) * torch.einsum("nea,nfa->nef", cross, cross)


def whitney_centroid_factors(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cell Whitney basis values at the barycentre and their constant curls.

    The two field-evaluation factors the curl-curl family reconstructs E and
    B/dB from the edge solution at a cell's centre. For local edge
    ``e = (i, j)`` over the cell's STORED node order (:data:`TET_LOCAL_EDGES`):

    - Whitney basis at the cell barycentre (``λ_a = 1/4 ∀a``):
      ``W_e(centroid) = λ_i g_j − λ_j g_i = (g_j − g_i)/4``;
    - ``curl W_e = 2 g_i × g_j``: constant per tetrahedron.

    Returned in float64: the FDEM3D/MT3D complex systems cast the result to
    complex128 at the call site, while TEM3D's real system keeps float64; the
    only intended divergence among the three operators is that post-cast, so it
    stays the caller's.

    Args:
        g: ``(n_cells, 4, 3)`` barycentric gradients from
            :func:`barycentric_gradients`.

    Returns:
        ``(w_centroid, curl_w)``: both ``(n_cells, 6, 3)`` float64, indexed by
        the fixed local-edge table.
    """
    dev = g.device
    i6 = _I6.to(dev)
    j6 = _J6.to(dev)
    w_centroid = 0.25 * (g[:, j6, :] - g[:, i6, :])             # (nc, 6, 3)
    curl_w = 2.0 * torch.linalg.cross(g[:, i6, :], g[:, j6, :], dim=-1)
    return w_centroid, curl_w
