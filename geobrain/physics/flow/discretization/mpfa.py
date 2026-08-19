"""
MPFA-O: consistent multi-point flux approximation (2-D interaction-region method).

The two-point flux is *inconsistent* on non-K-orthogonal grids, for a linear
pressure field it does not reproduce the exact Darcy flux. The MPFA-O method
(Aavatsmark) restores consistency by solving, around each grid **vertex**, a
local *interaction region*:

  * a continuity-point pressure ``u_k`` sits on each edge through the vertex;
  * inside each cell-corner the pressure gradient is linearly reconstructed from
    the cell-centre pressure and its two continuity points,
    ``∇p_i = D_i⁻¹·[u_a − p_i ; u_b − p_i]`` (``D_i`` = the centre→continuity-point
    matrix);
  * **normal-flux continuity** across each edge, ``n·K_i·∇p_i = n·K_j·∇p_j``,
    closes an ``m×m`` local system for the ``u_k`` in terms of the cell
    pressures: ``u = −R_u⁻¹·R_p·p``;
  * each half-edge flux ``−A_½·n·K_i·∇p_i`` then becomes a multi-cell stencil,
    and a face flux is the sum over its two endpoint vertices.

Because each cell's reconstructed gradient is the *exact* gradient of a linear
field, the resulting flux reproduces ``−A·nᵀK·∇p`` exactly on any grid and for a
full permeability tensor, the linearity-preservation property the tests verify
to machine precision (where the full-tensor two-point flux fails). The local
solves are differentiable in permeability and geometry.

This is the 2-D method; the grid is given by nodes + CCW cell-node lists.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import TypeAlias
import warnings

import torch

from ....core import GeoBrainError
from ..errors import FlowContractError

FaceStencils: TypeAlias = dict[int, dict[int, torch.Tensor]]
ScalarValue: TypeAlias = float | torch.Tensor


def _stable_mean(points: torch.Tensor) -> torch.Tensor:
    """Return a mean without overflowing the coordinate sum."""

    scale = points.abs().amax()
    direct_limit = torch.finfo(points.dtype).max / points.shape[0]
    if bool(scale <= direct_limit):
        return points.mean(dim=0)
    safe_scale = torch.where(
        torch.isfinite(scale) & (scale > 0),
        scale,
        torch.ones_like(scale),
    )
    return (points / safe_scale).mean(dim=0) * safe_scale


def _stable_vector_norm(vector: torch.Tensor) -> torch.Tensor:
    """Return a Euclidean norm without overflow or underflow."""

    scale = vector.abs().amax()
    finfo = torch.finfo(vector.dtype)
    lower_limit = finfo.tiny**0.5
    upper_limit = (finfo.max / vector.numel()) ** 0.5
    if bool((scale == 0) | ((scale >= lower_limit) & (scale <= upper_limit))):
        return torch.linalg.vector_norm(vector)
    safe_scale = torch.where(
        torch.isfinite(scale) & (scale > 0),
        scale,
        torch.ones_like(scale),
    )
    return torch.linalg.vector_norm(vector / safe_scale) * safe_scale


def _require_mpfa_derived(
    object_name: str,
    field_name: str,
    value: torch.Tensor,
    *,
    positive: bool = False,
) -> None:
    """Reject derived MPFA geometry that the selected dtype cannot represent."""

    valid = bool(torch.isfinite(value).all())
    if positive:
        valid = valid and bool((value > 0).all())
    if not valid:
        raise FlowContractError(
            "MPFA derived geometry must remain finite and physically valid",
            object_name=object_name,
            field=field_name,
            expected="all finite" + (" and > 0" if positive else ""),
            actual="invalid derived entries present",
        )


def _compensated_sum(values: torch.Tensor) -> torch.Tensor:
    """Accumulate a short geometry reduction without avoidable cancellation."""

    total = values.new_zeros(())
    compensation = values.new_zeros(())
    for value in values:
        corrected = value - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
    return total


def _stable_positive_product(factors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Multiply positive scalar factors in a log-balanced, autograd-safe order.

    Detached logarithms select an order whose intermediate exponents stay near
    zero.  The multiplication itself always uses the original tensors, unlike
    an ``exp(sum(log(...)))`` reconstruction whose backward pass loses the
    derivative when the representable result is subnormal.
    """

    pending: list[tuple[torch.Tensor, float]] = []
    for factor in factors:
        numeric_factor = float(factor.detach())
        if not math.isfinite(numeric_factor) or numeric_factor <= 0.0:
            return torch.stack(factors).prod()
        pending.append((factor, math.log(numeric_factor)))
    while len(pending) > 1:
        pending.sort(key=lambda item: item[1])
        smallest_factor, smallest_log = pending.pop(0)
        largest_factor, largest_log = pending.pop()
        pending.append(
            (
                smallest_factor * largest_factor,
                smallest_log + largest_log,
            )
        )
    return pending[0][0]


def _stable_polygon_area(points: torch.Tensor) -> torch.Tensor:
    """Return a translation-invariant polygon area with stable local scaling."""

    # Fan triangles use local edge vectors unconditionally.  Absolute-coordinate
    # shoelace products can lose several decimal digits well before a heuristic
    # cancellation threshold declares them unsafe.
    first_edges = points[1:-1] - points[0]
    second_edges = points[2:] - points[0]
    triangles = torch.stack((first_edges, second_edges), dim=1)
    column_scales = triangles.abs().amax(dim=(0, 1))
    safe_column_scales = torch.where(
        torch.isfinite(column_scales) & (column_scales > 0),
        column_scales,
        torch.ones_like(column_scales),
    )
    normalized_determinants = torch.linalg.det(triangles / safe_column_scales)
    normalized_area = 0.5 * _compensated_sum(normalized_determinants).abs()
    return _stable_positive_product((normalized_area, safe_column_scales[0], safe_column_scales[1]))


def _warn_singular_region(matrix: str, v: int) -> None:
    """A singular local interaction-region system (``matrix``) at node ``v`` means
    the MPFA-O stencil for that node cannot be formed, so its half-edge flux
    contributions are dropped, silently breaking local conservation there. Make
    the degradation observable (usually a degenerate/near-collinear cell geometry
    or extreme permeability anisotropy) instead of skipping without a trace."""
    warnings.warn(
        f"MPFA-O: singular local interaction region ({matrix}) at node {v}; its "
        "flux contribution is dropped and local conservation is broken there, "
        "check for degenerate cell geometry or extreme anisotropy near this node.",
        RuntimeWarning,
        stacklevel=3,
    )


class MPFAGrid2D:
    """Immutable SI 2-D grid with ``(x, y)`` coordinate columns."""

    def __init__(self, nodes: torch.Tensor, cell_nodes: list[list[int]]) -> None:
        _validate_grid_nodes(nodes, dimension=2, object_name="MPFAGrid2D")
        _validate_cell_nodes(
            cell_nodes,
            n_nodes=int(nodes.shape[0]),
            minimum_nodes=3,
            object_name="MPFAGrid2D",
        )
        self._nodes = nodes.clone()
        self._cell_nodes = tuple(tuple(node_ids) for node_ids in cell_nodes)
        minimum = self._nodes.amin(dim=0)
        self._origin_m = (float(minimum[0].detach()), float(minimum[1].detach()))
        self._cell_centroids = torch.stack(
            [_stable_mean(self._nodes[list(node_ids)]) for node_ids in self._cell_nodes]
        )
        _require_mpfa_derived("MPFAGrid2D", "cell_centroids", self._cell_centroids)
        emap: dict[tuple[int, int], int] = {}
        e_nodes: list[tuple[int, int]] = []
        e_cells: list[list[int]] = []
        for ci, cn in enumerate(self._cell_nodes):
            k = len(cn)
            for e in range(k):
                a, b = cn[e], cn[(e + 1) % k]
                key = (a, b) if a < b else (b, a)
                if key not in emap:
                    emap[key] = len(e_nodes)
                    e_nodes.append(key)
                    e_cells.append([])
                e_cells[emap[key]].append(ci)
        node_edges: list[list[int]] = [[] for _ in range(self._nodes.shape[0])]
        node_cells: list[list[int]] = [[] for _ in range(self._nodes.shape[0])]
        for ei, (a, b) in enumerate(e_nodes):
            node_edges[a].append(ei)
            node_edges[b].append(ei)
        for ci, cn in enumerate(self._cell_nodes):
            for nd in cn:
                node_cells[nd].append(ci)
        self._edge_nodes = tuple(e_nodes)
        self._edge_cells = tuple(tuple(cells) for cells in e_cells)
        self._node_edges = tuple(tuple(edges) for edges in node_edges)
        self._node_cells = tuple(tuple(cells) for cells in node_cells)
        for edge in range(self.n_edges):
            normal, length = self.edge_normal_area(edge)
            _require_mpfa_derived("MPFAGrid2D", f"edge[{edge}].normal", normal)
            _require_mpfa_derived("MPFAGrid2D", f"edge[{edge}].length", length, positive=True)

    @property
    def nodes(self) -> torch.Tensor:
        return self._nodes.clone()

    def _nodes_view(self) -> torch.Tensor:
        return self._nodes

    @property
    def cell_centroids(self) -> torch.Tensor:
        return self._cell_centroids.clone()

    def _cell_centroids_view(self) -> torch.Tensor:
        return self._cell_centroids

    @property
    def cell_nodes(self) -> tuple[tuple[int, ...], ...]:
        return self._cell_nodes

    @property
    def edge_nodes(self) -> tuple[tuple[int, int], ...]:
        return self._edge_nodes

    @property
    def edge_cells(self) -> tuple[tuple[int, ...], ...]:
        return self._edge_cells

    @property
    def node_edges(self) -> tuple[tuple[int, ...], ...]:
        return self._node_edges

    @property
    def node_cells(self) -> tuple[tuple[int, ...], ...]:
        return self._node_cells

    @property
    def coordinate_columns(self) -> tuple[str, str]:
        return ("x", "y")

    @property
    def z_positive_down(self) -> bool:
        return True

    @property
    def origin_m(self) -> tuple[float, float]:
        return self._origin_m

    @property
    def dtype(self) -> torch.dtype:
        return self._nodes.dtype

    @property
    def device(self) -> torch.device:
        return self._nodes.device

    @property
    def n_edges(self) -> int:
        return len(self._edge_nodes)

    def edge_midpoint(self, e: int) -> torch.Tensor:
        a, b = self._edge_nodes[e]
        return _stable_mean(self._nodes[[a, b]])

    def edge_normal_area(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit normal and length of edge ``e`` (2-D; area = length·1)."""
        a, b = self._edge_nodes[e]
        ev = self._nodes[b] - self._nodes[a]
        length = _stable_vector_norm(ev)
        safe_length = torch.where(length > 0, length, torch.ones_like(length))
        n = torch.stack([ev[1], -ev[0]]) / safe_length
        return n, length


def _stable_polygon_areas(grid: MPFAGrid2D) -> torch.Tensor:
    """Return validated cell areas for family-owned MPFA model consumers."""

    areas = torch.stack(
        [_stable_polygon_area(grid._nodes_view()[list(nodes)]) for nodes in grid.cell_nodes]
    )
    _require_mpfa_derived("MPFAGrid2D", "cell_areas", areas, positive=True)
    return areas


def _validate_grid_nodes(
    nodes: object,
    *,
    dimension: int,
    object_name: str,
) -> None:
    if not isinstance(nodes, torch.Tensor):
        raise FlowContractError(
            "MPFA nodes must be a tensor",
            object_name=object_name,
            field="nodes",
            expected=f"floating [node, {dimension}] tensor",
            actual=type(nodes).__name__,
        )
    if nodes.ndim != 2 or nodes.shape[1:] != (dimension,) or nodes.shape[0] == 0:
        raise FlowContractError(
            "MPFA nodes have an invalid coordinate shape",
            object_name=object_name,
            field="nodes",
            expected=f"non-empty [node, {dimension}]",
            actual=tuple(nodes.shape),
        )
    if nodes.dtype not in {torch.float32, torch.float64}:
        raise FlowContractError(
            "MPFA nodes require a supported floating dtype",
            object_name=object_name,
            field="nodes.dtype",
            expected=(str(torch.float32), str(torch.float64)),
            actual=str(nodes.dtype),
        )
    if not bool(torch.isfinite(nodes).all()):
        raise FlowContractError(
            "MPFA node coordinates must be finite",
            object_name=object_name,
            field="nodes",
            expected="all finite",
            actual="non-finite entries present",
        )


def _validate_cell_nodes(
    cell_nodes: object,
    *,
    n_nodes: int,
    minimum_nodes: int,
    object_name: str,
) -> None:
    if not isinstance(cell_nodes, list) or not cell_nodes:
        raise FlowContractError(
            "MPFA cell_nodes must be a non-empty list",
            object_name=object_name,
            field="cell_nodes",
            expected="non-empty list[list[int]]",
            actual=type(cell_nodes).__name__,
        )
    for cell, node_ids in enumerate(cell_nodes):
        if (
            not isinstance(node_ids, list)
            or len(node_ids) < minimum_nodes
            or any(
                isinstance(node, bool) or not isinstance(node, int) or not 0 <= node < n_nodes
                for node in node_ids
            )
        ):
            raise FlowContractError(
                "MPFA cell connectivity contains an invalid node id",
                object_name=object_name,
                field=f"cell_nodes[{cell}]",
                expected=f">={minimum_nodes} node ids in [0, {n_nodes})",
                actual=node_ids,
            )


def _validate_mpfa_field(
    grid: MPFAGrid2D,
    field_tensor: torch.Tensor,
    *,
    field: str,
) -> None:
    if field_tensor.device != grid.device or field_tensor.dtype != grid.dtype:
        raise FlowContractError(
            f"{field} dtype/device must match the MPFA grid",
            object_name="MPFAGrid2D",
            field=field,
            expected={"dtype": str(grid.dtype), "device": str(grid.device)},
            actual={
                "dtype": str(field_tensor.dtype),
                "device": str(field_tensor.device),
            },
        )
    if field_tensor.ndim == 0 or field_tensor.shape[0] != len(grid.cell_nodes):
        raise FlowContractError(
            f"{field} cell axis must match the MPFA grid",
            object_name="MPFAGrid2D",
            field=field,
            expected=len(grid.cell_nodes),
            actual=None if field_tensor.ndim == 0 else int(field_tensor.shape[0]),
        )


def build_mpfa_grid(nodes: torch.Tensor, cell_nodes: list[list[int]]) -> MPFAGrid2D:
    return MPFAGrid2D(nodes=nodes, cell_nodes=list(cell_nodes))


def mpfa_o_face_flux_stencils(
    grid: MPFAGrid2D, perm_tensor: torch.Tensor
) -> dict[int, dict[int, torch.Tensor]]:
    """Per **interior** edge, the MPFA-O flux stencil ``{cell: T}`` with the flux
    (left→right, by the edge's canonical normal) equal to ``Σ_cell T·p_cell``.

    ``perm_tensor`` is ``(n_cells, 2, 2)``; the result is differentiable in it.
    """
    _validate_mpfa_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (2, 2):
        raise GeoBrainError(
            "MPFA-O (2-D) needs (n_cells, 2, 2) permeability",
            object_name="mpfa_o_face_flux_stencils",
            field="perm_tensor",
            expected="(n_cells, 2, 2)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    stencils: dict[int, dict[int, torch.Tensor]] = {}

    def _add(face: int, cell: int, coef: torch.Tensor) -> None:
        d = stencils.setdefault(face, {})
        d[cell] = d.get(cell, coef.new_zeros(())) + coef

    for v in range(grid._nodes_view().shape[0]):
        edges_v = grid.node_edges[v]
        cells_v = grid.node_cells[v]
        # interior interaction region: every incident edge has two cells
        if any(len(grid.edge_cells[e]) != 2 for e in edges_v):
            continue
        m = len(edges_v)
        if m != len(cells_v):
            continue  # irregular region; skip (boundary-ish)
        e_local = {e: k for k, e in enumerate(edges_v)}
        c_local = {c: k for k, c in enumerate(cells_v)}
        ncv = len(cells_v)
        mid = {e: grid.edge_midpoint(e) for e in edges_v}

        # gradient operator per cell: ∇p_i = G_i @ [u (m); p (ncv)]
        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in cells_v:
            my_edges = [e for e in edges_v if c in grid.edge_cells[e]]
            if len(my_edges) != 2:
                region_valid = False
                break
            ea, eb = my_edges
            D = torch.stack([mid[ea] - cc[c], mid[eb] - cc[c]], dim=0)  # (2,2)
            Dinv = torch.linalg.inv(D)
            sel = perm_tensor.new_zeros(2, m + ncv)  # [u_a−p_c ; u_b−p_c]
            sel[0, e_local[ea]] = 1.0
            sel[1, e_local[eb]] = 1.0
            sel[0, m + c_local[c]] -= 1.0
            sel[1, m + c_local[c]] -= 1.0
            G[c] = Dinv @ sel  # (2, m+ncv)
        if not region_valid:
            continue

        # normal-flux continuity per edge: n·K_i·∇p_i − n·K_j·∇p_j = 0
        R = perm_tensor.new_zeros(m, m + ncv)
        for e in edges_v:
            ci, cj = grid.edge_cells[e]
            n, _ = grid.edge_normal_area(e)
            row = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            R[e_local[e]] = row
        R_u = R[:, :m]
        R_p = R[:, m:]
        if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
            _warn_singular_region("R_u", v)
            continue
        # u = −R_u⁻¹ R_p p ; full map [u;p] = W @ p
        U = -torch.linalg.solve(R_u, R_p)  # (m, ncv)
        W = torch.cat(
            [U, torch.eye(ncv, dtype=perm_tensor.dtype, device=perm_tensor.device)],
            dim=0,
        )  # (m+ncv, ncv)

        # half-edge flux contribution to each incident face (this vertex's half)
        for e in edges_v:
            ci, cj = grid.edge_cells[e]  # ci = "left"
            n, length = grid.edge_normal_area(e)
            # orient the normal from left (ci) to right (cj) so the assembled
            # flux is genuinely "left→right" (the canonical divergence scatters
            # +F to ci, −F to cj). edge_normal_area's sign follows node order,
            # not l→r.
            if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                n = -n
            A_half = 0.5 * length
            grad_left = G[ci] @ W  # (2, ncv) ∇p_left(p)
            flux_coef = -A_half * (n @ (perm_tensor[ci] @ grad_left))  # (ncv,) flux left→right
            for k, c in enumerate(cells_v):
                _add(e, c, flux_coef[k])
    return stencils


def mpfa_o_face_flux_stencils_full(
    grid: MPFAGrid2D, perm_tensor: torch.Tensor
) -> FaceStencils:
    """Per-interior-face MPFA-O flux stencils for a **no-flow-bounded** domain.

    Like :func:`mpfa_o_face_flux_stencils`, but processes **every** vertex
    (interior *and* boundary), closing boundary edges as no-flow (Neumann
    ``q = 0``). Each interior face therefore receives **both** half-edge
    contributions even when one endpoint is on the boundary, so on a
    K-orthogonal grid the stencil collapses to the full two-point ``T = K·A/d``
    (the plain interior-vertex version only assembles the interior half).

    Returns ``{face: {cell: T_c}}`` with ``flux_{l→r} = Σ_c T_c·p_c`` (``l, r =
    grid.edge_cells[face]``), differentiable in ``perm_tensor``. This is the
    stencil a no-flow multiphase transient (e.g. :class:`MPFATwoPhaseModel`)
    upwinds the phase mobility onto.
    """
    _validate_mpfa_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (2, 2):
        raise GeoBrainError(
            "MPFA-O (2-D) needs (n_cells, 2, 2) permeability",
            object_name="mpfa_o_face_flux_stencils_full",
            field="perm_tensor",
            expected="(n_cells, 2, 2)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    boundary = {e for e in range(grid.n_edges) if len(grid.edge_cells[e]) == 1}
    stencils: dict[int, dict[int, torch.Tensor]] = {}

    for v in range(grid._nodes_view().shape[0]):
        E_v = grid.node_edges[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [e for e in E_v if len(grid.edge_cells[e]) == 2]  # interior (unknown u)
        N_v = [e for e in E_v if e in boundary]  # boundary no-flow (unknown u)
        if any(len([e for e in E_v if c in grid.edge_cells[e]]) != 2 for c in C_v):
            continue
        nI, nN, nCv = len(I_v), len(N_v), len(C_v)
        nU = nI + nN
        width = nU + nCv  # [u_I, u_N, p]
        iuI = {e: k for k, e in enumerate(I_v)}
        iuN = {e: nI + k for k, e in enumerate(N_v)}
        ip = {c: nU + k for k, c in enumerate(C_v)}
        mid = {e: grid.edge_midpoint(e) for e in E_v}

        def col(e: int) -> int:
            return iuI[e] if e in iuI else iuN[e]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            ea, eb = [e for e in E_v if c in grid.edge_cells[e]]
            Dm = torch.stack([mid[ea] - cc[c], mid[eb] - cc[c]], dim=0)
            if float(torch.linalg.det(Dm).detach().abs()) < 1e-300:
                _warn_singular_region("Dm", v)
                region_valid = False
                break
            sel = perm_tensor.new_zeros(2, width)
            sel[0, col(ea)] += 1.0
            sel[1, col(eb)] += 1.0
            sel[0, ip[c]] -= 1.0
            sel[1, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(Dm) @ sel
        if not region_valid:
            continue

        if nU > 0:  # interior continuity + no-flow matching
            R = perm_tensor.new_zeros(nU, width)
            for e in I_v:
                ci, cj = grid.edge_cells[e]
                n, _ = grid.edge_normal_area(e)
                R[iuI[e]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            for e in N_v:  # half-flux out = 0 (no-flow), homogeneous
                c = grid.edge_cells[e][0]
                n, length = grid.edge_normal_area(e)
                if float((n @ (mid[e] - cc[c])).detach()) < 0.0:
                    n = -n
                R[iuN[e]] = -(0.5 * length) * (n @ (perm_tensor[c] @ G[c]))
            R_u = R[:, :nU]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                _warn_singular_region("R_u", v)
                continue
            W = torch.cat(
                [
                    -torch.linalg.solve(R_u, R[:, nU:]),
                    torch.eye(nCv, dtype=perm_tensor.dtype, device=perm_tensor.device),
                ],
                dim=0,
            )  # (width, nCv): [u;p]=W@p
        else:
            W = torch.eye(nCv, dtype=perm_tensor.dtype, device=perm_tensor.device)

        for e in I_v:  # accumulate half-edge flux (l→r) per face
            ci, cj = grid.edge_cells[e]
            n, length = grid.edge_normal_area(e)
            if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                n = -n
            coef = -(0.5 * length) * (n @ (perm_tensor[ci] @ (G[ci] @ W)))  # (nCv,) in terms of p
            d = stencils.setdefault(e, {})
            for k, c in enumerate(C_v):
                d[c] = d.get(c, perm_tensor.new_zeros(())) + coef[k]
    return stencils


def mpfa_o_face_flux_stencils_bc(
    grid: MPFAGrid2D,
    perm_tensor: torch.Tensor,
    dirichlet_edges: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Per-face MPFA-O flux stencils with **Dirichlet pressure boundaries**, as a
    ghost-cell augmented operator (for a multiphase pressure-BC residual).

    Boundary edges in ``dirichlet_edges`` carry a prescribed pressure; every other
    boundary edge is no-flow. Each Dirichlet boundary face is treated as a
    connection to a fixed-pressure **ghost cell**, so interior *and* boundary
    faces are uniform two-entity connections over the augmented unknown vector
    ``[p_cells ; p_ghosts]`` (``n_cells + n_dirichlet`` long).

    Returns ``(L, face_lr, dirichlet_edges)`` where ``L`` is ``(n_faces,
    n_cells + n_dir)``, ``face_lr`` ``(n_faces, 2)`` gives each face's two augmented
    endpoints (ghost index = ``n_cells + k``), and the geometric flux **from
    endpoint 0 to endpoint 1** of face ``f`` is ``(L @ φ_aug)[f]`` for any
    augmented potential field ``φ_aug``. Summed with unit mobility this reproduces
    :func:`assemble_mpfa_divergence_mixed` exactly, so the single-phase limit
    matches :func:`solve_mpfa_bvp` (on skewed / full-tensor grids too).
    """
    _validate_mpfa_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (2, 2):
        raise GeoBrainError(
            "MPFA-O (2-D) needs (n_cells, 2, 2) permeability",
            object_name="mpfa_o_face_flux_stencils_bc",
            field="perm_tensor",
            expected="(n_cells, 2, 2)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    nC = len(grid.cell_nodes)
    boundary = {e for e in range(grid.n_edges) if len(grid.edge_cells[e]) == 1}
    dir_set = {int(e) for e in dirichlet_edges}
    bad = dir_set - boundary
    if bad:
        raise GeoBrainError(
            "dirichlet_edges must all be boundary edges",
            object_name="mpfa_o_face_flux_stencils_bc",
            field="dirichlet_edges",
            expected="boundary edge ids",
            actual=sorted(bad),
        )
    dir_list = [e for e in sorted(boundary) if e in dir_set]
    g_index = {e: nC + k for k, e in enumerate(dir_list)}  # ghost-cell column per Dirichlet edge
    n_aug = nC + len(dir_list)
    dt = perm_tensor.dtype

    fstencil: dict[int, dict[int, torch.Tensor]] = {}  # face -> {aug_col: coef}, flux endpoint0→1
    flr: dict[int, tuple[int, int]] = {}

    for v in range(grid._nodes_view().shape[0]):
        E_v = grid.node_edges[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [e for e in E_v if len(grid.edge_cells[e]) == 2]  # interior (unknown)
        N_v = [e for e in E_v if e in boundary and e not in dir_set]  # no-flow boundary (unknown)
        Dd_v = [e for e in E_v if e in dir_set]  # Dirichlet boundary (known ghost)
        if any(len([e for e in E_v if c in grid.edge_cells[e]]) != 2 for c in C_v):
            continue
        nI, nN, nCv, nDv = len(I_v), len(N_v), len(C_v), len(Dd_v)
        nU = nI + nN
        width = nU + nCv + nDv  # [u_I, u_N, p, p_D]
        iuI = {e: k for k, e in enumerate(I_v)}
        iuN = {e: nI + k for k, e in enumerate(N_v)}
        ip = {c: nU + k for k, c in enumerate(C_v)}
        idd = {e: nU + nCv + k for k, e in enumerate(Dd_v)}
        mid = {e: grid.edge_midpoint(e) for e in E_v}

        def col(e: int) -> int:
            if e in iuI:
                return iuI[e]
            if e in iuN:
                return iuN[e]
            return idd[e]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            ea, eb = [e for e in E_v if c in grid.edge_cells[e]]
            Dm = torch.stack([mid[ea] - cc[c], mid[eb] - cc[c]], dim=0)
            if float(torch.linalg.det(Dm).detach().abs()) < 1e-300:
                _warn_singular_region("Dm", v)
                region_valid = False
                break
            sel = perm_tensor.new_zeros(2, width)
            sel[0, col(ea)] += 1.0
            sel[1, col(eb)] += 1.0
            sel[0, ip[c]] -= 1.0
            sel[1, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(Dm) @ sel
        if not region_valid:
            continue

        if nU > 0:  # interior continuity + no-flow matching
            R = perm_tensor.new_zeros(nU, width)
            for e in I_v:
                ci, cj = grid.edge_cells[e]
                n, _ = grid.edge_normal_area(e)
                R[iuI[e]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            for e in N_v:  # half-flux out = 0 (no-flow), homogeneous
                c = grid.edge_cells[e][0]
                n, length = grid.edge_normal_area(e)
                if float((n @ (mid[e] - cc[c])).detach()) < 0.0:
                    n = -n
                R[iuN[e]] = -(0.5 * length) * (n @ (perm_tensor[c] @ G[c]))
            R_u = R[:, :nU]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                _warn_singular_region("R_u", v)
                continue
            W = torch.cat(
                [
                    -torch.linalg.solve(R_u, R[:, nU:]),
                    torch.eye(nCv + nDv, dtype=dt, device=perm_tensor.device),
                ],
                dim=0,
            )  # (width, nCv+nDv): [u;p;p_D] = W@[p;p_D]
        else:
            W = torch.eye(nCv + nDv, dtype=dt, device=perm_tensor.device)

        # map the local [p; p_D] columns to augmented columns: cells, then ghosts
        aug_cols = [c for c in C_v] + [g_index[e] for e in Dd_v]

        def accumulate(
            face: int, a0: int, a1: int, coef: torch.Tensor
        ) -> None:  # coef over [p; p_D] (nCv+nDv,)
            flr.setdefault(face, (a0, a1))
            d = fstencil.setdefault(face, {})
            for k, ac in enumerate(aug_cols):
                d[ac] = d.get(ac, perm_tensor.new_zeros(())) + coef[k]

        for e in I_v:  # interior face: flux l→r
            ci, cj = grid.edge_cells[e]
            n, length = grid.edge_normal_area(e)
            if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                n = -n
            coef = -(0.5 * length) * (n @ (perm_tensor[ci] @ (G[ci] @ W)))
            accumulate(e, ci, cj, coef)
        for e in Dd_v:  # Dirichlet boundary face: flux cell→ghost
            c = grid.edge_cells[e][0]
            n, length = grid.edge_normal_area(e)
            if float((n @ (mid[e] - cc[c])).detach()) < 0.0:
                n = -n  # outward from the cell
            coef = -(0.5 * length) * (
                n @ (perm_tensor[c] @ (G[c] @ W))
            )  # flux OUT of cell (cell→ghost)
            accumulate(e, c, g_index[e], coef)

    faces = sorted(fstencil)
    L = perm_tensor.new_zeros(len(faces), n_aug)
    lr: list[list[int]] = []
    for fi, f in enumerate(faces):
        for ac, t in fstencil[f].items():
            L[fi, ac] = t
        lr.append(list(flr[f]))
    face_cells = torch.tensor(
        lr, dtype=torch.long, device=perm_tensor.device
    ).reshape(-1, 2)
    return L, face_cells, dir_list


def assemble_mpfa_divergence(
    grid: MPFAGrid2D, perm_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Assemble the MPFA-O flux operator over **all** vertices (interior +
    boundary), with Dirichlet continuity points on boundary edges.

    Returns sparse COO blocks ``(L_pp, L_pb, boundary_edges)`` where the net **outward
    divergence** is ``L_pp @ p + L_pb @ p_bc``; ``L_pp`` is ``(n_cells,
    n_cells)``, ``L_pb`` is ``(n_cells, n_boundary_edges)``, and ``p_bc`` is the
    Dirichlet pressure at each boundary-edge midpoint (order = ``boundary_edges``).
    Both blocks are differentiable in ``perm_tensor`` and their stored entries
    scale with mesh connectivity rather than ``n_cells²``. For a steady incompressible
    balance with sources ``q`` (injection +): solve
    ``L_pp @ p + L_pb @ p_bc - q = 0``.
    """
    _validate_mpfa_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (2, 2):
        raise GeoBrainError(
            "MPFA-O (2-D) needs (n_cells, 2, 2) permeability",
            object_name="assemble_mpfa_divergence",
            field="perm_tensor",
            expected="(n_cells, 2, 2)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    nC = len(grid.cell_nodes)
    boundary_edges = [e for e in range(grid.n_edges) if len(grid.edge_cells[e]) == 1]
    b_index = {e: k for k, e in enumerate(boundary_edges)}
    nB_tot = len(boundary_edges)
    dt = perm_tensor.dtype
    pp_rows: list[int] = []
    pp_columns: list[int] = []
    pp_values: list[torch.Tensor] = []
    pb_rows: list[int] = []
    pb_columns: list[int] = []
    pb_values: list[torch.Tensor] = []

    def add_pp(row: int, column: int, value: torch.Tensor) -> None:
        pp_rows.append(row)
        pp_columns.append(column)
        pp_values.append(value)

    def add_pb(row: int, column: int, value: torch.Tensor) -> None:
        pb_rows.append(row)
        pb_columns.append(column)
        pb_values.append(value)

    for v in range(grid._nodes_view().shape[0]):
        E_v = grid.node_edges[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [e for e in E_v if len(grid.edge_cells[e]) == 2]  # interior edges (unknown u)
        B_v = [e for e in E_v if len(grid.edge_cells[e]) == 1]  # boundary edges (Dirichlet u)
        # each cell at v must have exactly its two corner edges present in E_v
        if any(len([e for e in E_v if c in grid.edge_cells[e]]) != 2 for c in C_v):
            continue
        nI, nCv, nBv = len(I_v), len(C_v), len(B_v)
        width = nI + nCv + nBv
        iu = {e: k for k, e in enumerate(I_v)}  # u_I block [0:nI)
        ip = {c: nI + k for k, c in enumerate(C_v)}  # p block
        ib = {e: nI + nCv + k for k, e in enumerate(B_v)}  # bc block
        mid = {e: grid.edge_midpoint(e) for e in E_v}

        def col(e: int) -> int:  # column of edge e's continuity pressure
            return iu[e] if e in iu else ib[e]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            ea, eb = [e for e in E_v if c in grid.edge_cells[e]]
            D = torch.stack([mid[ea] - cc[c], mid[eb] - cc[c]], dim=0)
            if float(torch.linalg.det(D).detach().abs()) < 1e-300:
                _warn_singular_region("D", v)
                region_valid = False
                break
            sel = perm_tensor.new_zeros(2, width)
            sel[0, col(ea)] += 1.0
            sel[1, col(eb)] += 1.0
            sel[0, ip[c]] -= 1.0
            sel[1, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(D) @ sel
        if not region_valid:
            continue

        # flux continuity on interior edges → solve interior continuity pressures
        if nI > 0:
            R = perm_tensor.new_zeros(nI, width)
            for e in I_v:
                ci, cj = grid.edge_cells[e]
                n, _ = grid.edge_normal_area(e)
                R[iu[e]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            R_u = R[:, :nI]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                _warn_singular_region("R_u", v)
                continue
            R_rest = R[:, nI:]
            U = -torch.linalg.solve(R_u, R_rest)  # (nI, nCv+nBv) : u_I = U @ [p; bc]
            W = torch.cat(
                [
                    U,  # u_I rows
                    torch.cat(
                        [
                            torch.eye(nCv, dtype=dt, device=perm_tensor.device),
                            perm_tensor.new_zeros(nCv, nBv),
                        ],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            perm_tensor.new_zeros(nBv, nCv),
                            torch.eye(nBv, dtype=dt, device=perm_tensor.device),
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )  # (width, nCv+nBv): [u;p;bc] = W @ [p;bc]
        else:
            W = torch.cat(
                [
                    torch.cat(
                        [
                            torch.eye(nCv, dtype=dt, device=perm_tensor.device),
                            perm_tensor.new_zeros(nCv, nBv),
                        ],
                        dim=1,
                    ),
                    torch.cat(
                        [
                            perm_tensor.new_zeros(nBv, nCv),
                            torch.eye(nBv, dtype=dt, device=perm_tensor.device),
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )

        for e in E_v:
            n, length = grid.edge_normal_area(e)
            A_half = 0.5 * length
            if len(grid.edge_cells[e]) == 2:  # interior edge: flux left→right
                ci, cj = grid.edge_cells[e]
                if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                    n = -n
                coef = -A_half * (n @ (perm_tensor[ci] @ (G[ci] @ W)))  # (nCv+nBv,) flux l→r
                for k, c in enumerate(C_v):  # into ci: −coef ; into cj: +coef
                    add_pp(ci, c, -coef[k])
                    add_pp(cj, c, coef[k])
                for k, e_b in enumerate(B_v):
                    add_pb(ci, b_index[e_b], -coef[nCv + k])
                    add_pb(cj, b_index[e_b], coef[nCv + k])
            else:  # boundary edge: flux into its cell
                c = grid.edge_cells[e][0]
                if float((n @ (mid[e] - cc[c])).detach()) < 0.0:
                    n = -n  # outward from the cell
                coef = A_half * (n @ (perm_tensor[c] @ (G[c] @ W)))  # flux INTO c
                for k, cc_ in enumerate(C_v):
                    add_pp(c, cc_, coef[k])
                for k, e_b in enumerate(B_v):
                    add_pb(c, b_index[e_b], coef[nCv + k])
    # Local assembly above follows the historical flux-into convention. Expose
    # only the platform-wide canonical outward divergence at the API boundary.
    L_pp = _sparse_from_triplets(
        pp_rows,
        pp_columns,
        [-value for value in pp_values],
        shape=(nC, nC),
        like=perm_tensor,
    )
    L_pb = _sparse_from_triplets(
        pb_rows,
        pb_columns,
        [-value for value in pb_values],
        shape=(nC, nB_tot),
        like=perm_tensor,
    )
    return L_pp, L_pb, boundary_edges


def assemble_mpfa_divergence_mixed(
    grid: MPFAGrid2D,
    perm_tensor: torch.Tensor,
    neumann_edges: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """MPFA-O flux operator with **mixed Dirichlet / Neumann** boundaries.

    Boundary edges listed in ``neumann_edges`` carry a *prescribed total outward
    flux* ``q_N`` (``q_N = 0`` ⇒ a no-flow / impermeable boundary); their
    continuity points are *unknowns* closed by flux matching (the reconstructed
    half-edge flux out of the cell equals ``½·q_N`` per endpoint). Every other
    boundary edge is Dirichlet (prescribed pressure at its midpoint).

    Returns ``(L_pp, L_pb, L_pn, dirichlet_edges, neumann_edges)`` where the net
    outward divergence is ``L_pp @ p + L_pb @ p_D + L_pn @ q_N``
    (``p_D`` ordered by ``dirichlet_edges``, ``q_N`` by ``neumann_edges``). All
    three blocks are differentiable in ``perm_tensor``. Steady incompressible
    balance with sources ``s`` (injection +): ``L_pp @ p + L_pb @ p_D +
    L_pn @ q_N - s = 0``, pin a datum cell if there is no Dirichlet edge.
    """
    _validate_mpfa_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (2, 2):
        raise GeoBrainError(
            "MPFA-O (2-D) needs (n_cells, 2, 2) permeability",
            object_name="assemble_mpfa_divergence_mixed",
            field="perm_tensor",
            expected="(n_cells, 2, 2)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    nC = len(grid.cell_nodes)
    boundary_edges = [e for e in range(grid.n_edges) if len(grid.edge_cells[e]) == 1]
    neu = {int(e) for e in neumann_edges}
    bad = neu - set(boundary_edges)
    if bad:
        raise GeoBrainError(
            "neumann_edges must all be boundary edges",
            object_name="assemble_mpfa_divergence_mixed",
            field="neumann_edges",
            expected="boundary edge ids",
            actual=sorted(bad),
        )
    dirichlet_edges = [e for e in boundary_edges if e not in neu]
    neumann_list = [e for e in boundary_edges if e in neu]
    d_index = {e: k for k, e in enumerate(dirichlet_edges)}
    n_index = {e: k for k, e in enumerate(neumann_list)}
    dt = perm_tensor.dtype
    L_pp = perm_tensor.new_zeros(nC, nC)
    L_pb = perm_tensor.new_zeros(nC, len(dirichlet_edges))
    L_pn = perm_tensor.new_zeros(nC, len(neumann_list))

    for v in range(grid._nodes_view().shape[0]):
        E_v = grid.node_edges[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [e for e in E_v if len(grid.edge_cells[e]) == 2]  # interior (unknown u)
        N_v = [e for e in E_v if e in neu]  # Neumann (unknown u)
        Dd_v = [
            e for e in E_v if len(grid.edge_cells[e]) == 1 and e not in neu
        ]  # Dirichlet (known)
        if any(len([e for e in E_v if c in grid.edge_cells[e]]) != 2 for c in C_v):
            continue
        nI, nN, nCv, nDv = len(I_v), len(N_v), len(C_v), len(Dd_v)
        nU = nI + nN
        width = nU + nCv + nDv + nN  # [u_I, u_N, p, p_D, q_N]
        iuI = {e: k for k, e in enumerate(I_v)}  # u_I block [0:nI)
        iuN = {e: nI + k for k, e in enumerate(N_v)}  # u_N block [nI:nU)
        ip = {c: nU + k for k, c in enumerate(C_v)}  # p block
        idd = {e: nU + nCv + k for k, e in enumerate(Dd_v)}  # p_D block
        iqn = {e: nU + nCv + nDv + k for k, e in enumerate(N_v)}  # q_N block
        mid = {e: grid.edge_midpoint(e) for e in E_v}

        def col(e: int) -> int:  # column of edge e's continuity pressure
            if e in iuI:
                return iuI[e]
            if e in iuN:
                return iuN[e]
            return idd[e]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            ea, eb = [e for e in E_v if c in grid.edge_cells[e]]
            Dm = torch.stack([mid[ea] - cc[c], mid[eb] - cc[c]], dim=0)
            if float(torch.linalg.det(Dm).detach().abs()) < 1e-300:
                _warn_singular_region("Dm", v)
                region_valid = False
                break
            sel = perm_tensor.new_zeros(2, width)
            sel[0, col(ea)] += 1.0
            sel[1, col(eb)] += 1.0
            sel[0, ip[c]] -= 1.0
            sel[1, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(Dm) @ sel
        if not region_valid:
            continue

        # nU closure equations: interior flux continuity + Neumann flux matching
        if nU > 0:
            R = perm_tensor.new_zeros(nU, width)
            for e in I_v:
                ci, cj = grid.edge_cells[e]
                n, _ = grid.edge_normal_area(e)
                R[iuI[e]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            for e in N_v:  # half-flux OUT of cell = ½·q_N
                c = grid.edge_cells[e][0]
                n, length = grid.edge_normal_area(e)
                if float((n @ (mid[e] - cc[c])).detach()) < 0.0:
                    n = -n  # outward from the cell
                row = -(0.5 * length) * (n @ (perm_tensor[c] @ G[c]))  # = ½·q_N
                R[iuN[e]] = row
                R[iuN[e], iqn[e]] += -0.5  # move ½·q_N to LHS
            R_u = R[:, :nU]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                _warn_singular_region("R_u", v)
                continue
            U = -torch.linalg.solve(R_u, R[:, nU:])  # (nU, nCv+nDv+nN)
            W = torch.cat(
                [
                    U,
                    torch.eye(
                        nCv + nDv + nN,
                        dtype=dt,
                        device=perm_tensor.device,
                    ),
                ],
                dim=0,
            )  # (width, ·)
        else:
            W = torch.eye(
                nCv + nDv + nN,
                dtype=dt,
                device=perm_tensor.device,
            )

        for e in E_v:
            n, length = grid.edge_normal_area(e)
            A_half = 0.5 * length
            if len(grid.edge_cells[e]) == 2:  # interior edge: flux left→right
                ci, cj = grid.edge_cells[e]
                if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                    n = -n
                coef = -A_half * (n @ (perm_tensor[ci] @ (G[ci] @ W)))  # (nCv+nDv+nN,)
                for k, c in enumerate(C_v):
                    L_pp[ci, c] += -coef[k]
                    L_pp[cj, c] += coef[k]
                for k, e_d in enumerate(Dd_v):
                    L_pb[ci, d_index[e_d]] += -coef[nCv + k]
                    L_pb[cj, d_index[e_d]] += coef[nCv + k]
                for k, e_n in enumerate(N_v):
                    L_pn[ci, n_index[e_n]] += -coef[nCv + nDv + k]
                    L_pn[cj, n_index[e_n]] += coef[nCv + nDv + k]
            elif e in neu:  # Neumann edge: prescribed half-flux
                c = grid.edge_cells[e][0]
                L_pn[c, n_index[e]] += -0.5  # flux INTO c = −½·q_N per endpoint
            else:  # Dirichlet edge: flux into its cell
                c = grid.edge_cells[e][0]
                if float((n @ (mid[e] - cc[c])).detach()) < 0.0:
                    n = -n  # outward from the cell
                coef = A_half * (n @ (perm_tensor[c] @ (G[c] @ W)))  # flux INTO c
                for k, cc_ in enumerate(C_v):
                    L_pp[c, cc_] += coef[k]
                for k, e_d in enumerate(Dd_v):
                    L_pb[c, d_index[e_d]] += coef[nCv + k]
                for k, e_n in enumerate(N_v):
                    L_pn[c, n_index[e_n]] += coef[nCv + nDv + k]
    return -L_pp, -L_pb, -L_pn, dirichlet_edges, neumann_list


def solve_mpfa_bvp(
    grid: MPFAGrid2D,
    perm_tensor: torch.Tensor,
    *,
    dirichlet: Mapping[int, ScalarValue] | None = None,
    neumann: Mapping[int, ScalarValue] | None = None,
    source: torch.Tensor | None = None,
    datum: tuple[int, ScalarValue] | None = None,
) -> torch.Tensor:
    """Solve the steady MPFA-O boundary-value problem with mixed BCs.

    ``dirichlet``: ``{boundary_edge: pressure}`` (prescribed pressure). Every
    boundary edge **not** in ``dirichlet`` is treated as Neumann with outward
    flux ``neumann.get(edge, 0.0)``, so the default boundary is no-flow
    (impermeable), the reservoir convention. ``source``: per-cell injection
    (``+`` in). A pure-Neumann problem (no Dirichlet edge) is singular up to a
    constant; supply ``datum=(cell, value)`` to pin it. Returns the cell
    pressures, differentiable in ``perm_tensor`` / BC values.
    """
    dirichlet = {int(e): v for e, v in (dirichlet or {}).items()}
    neumann = {int(e): v for e, v in (neumann or {}).items()}
    boundary_edges = [e for e in range(grid.n_edges) if len(grid.edge_cells[e]) == 1]
    neumann_edges = [e for e in boundary_edges if e not in dirichlet]
    L_pp, L_pb, L_pn, d_edges, n_edges = assemble_mpfa_divergence_mixed(
        grid, perm_tensor, neumann_edges
    )
    nC = L_pp.shape[0]
    dt = perm_tensor.dtype

    def as_col(
        values: Mapping[int, ScalarValue],
        edges: Sequence[int],
        default: ScalarValue | None = None,
    ) -> torch.Tensor:
        if not edges:
            return perm_tensor.new_zeros(0)
        if default is None:
            return torch.stack(
                [
                    torch.as_tensor(values[e], dtype=dt, device=perm_tensor.device)
                    for e in edges
                ]
            )
        return torch.stack(
            [
                torch.as_tensor(
                    values.get(e, default), dtype=dt, device=perm_tensor.device
                )
                for e in edges
            ]
        )

    p_D = as_col(dirichlet, d_edges)
    q_N = as_col(neumann, n_edges, default=0.0)  # absent Neumann edges ⇒ no-flow
    s = source if source is not None else perm_tensor.new_zeros(nC)
    rhs = s - (L_pb @ p_D if d_edges else 0.0) - (L_pn @ q_N if n_edges else 0.0)

    A = L_pp
    if datum is not None:  # pin the otherwise-free constant
        dcell, dval = int(datum[0]), torch.as_tensor(
            datum[1], dtype=dt, device=perm_tensor.device
        )
        row = perm_tensor.new_zeros(nC)
        row[dcell] = 1.0
        A = torch.cat([A[:dcell], row.unsqueeze(0), A[dcell + 1 :]], dim=0)
        rhs = torch.cat([rhs[:dcell], dval.reshape(1), rhs[dcell + 1 :]], dim=0)
    return torch.linalg.solve(A, rhs)


def solve_mpfa_steady(
    grid: MPFAGrid2D,
    perm_tensor: torch.Tensor,
    bc_values: torch.Tensor,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit dense-reference solve for a small steady MPFA-O problem.

    Steady incompressible single-phase MPFA-O solve with Dirichlet boundary
    pressures. ``bc_values``: pressure at each boundary-edge midpoint (order from
    :func:`assemble_mpfa_divergence`). Returns cell pressures ``p`` (differentiable).
    Production reservoir solves consume the sparse assembly and a declared sparse
    solver; this helper deliberately selects dense direct execution by name and
    never acts as an implicit layout fallback.
    """
    L_pp, L_pb, _ = assemble_mpfa_divergence(grid, perm_tensor)
    rhs = -(L_pb @ bc_values)
    if source is not None:
        rhs = rhs + source
    # This convenience routine is the explicitly dense, differentiable
    # manufactured-solution reference. Production callers consume the sparse
    # blocks returned by ``assemble_mpfa_divergence`` and choose a declared
    # sparse solver instead of receiving an implicit fallback here.
    return torch.linalg.solve(_explicit_dense_reference(L_pp), rhs)


def _sparse_from_triplets(
    rows: list[int],
    columns: list[int],
    values: list[torch.Tensor],
    *,
    shape: tuple[int, int],
    like: torch.Tensor,
) -> torch.Tensor:
    indices = torch.tensor([rows, columns], dtype=torch.long, device=like.device)
    coefficients = torch.stack(values) if values else like.new_empty((0,))
    return torch.sparse_coo_tensor(
        indices,
        coefficients,
        size=shape,
        dtype=like.dtype,
        device=like.device,
    ).coalesce()


def _explicit_dense_reference(matrix: torch.Tensor) -> torch.Tensor:
    """Build a dense differentiable matrix for an explicitly dense reference API."""

    coo = matrix if matrix.layout == torch.sparse_coo else matrix.to_sparse_coo()
    coo = coo.coalesce()
    indices = coo.indices()
    return coo.values().new_zeros(coo.shape).index_put(
        (indices[0], indices[1]), coo.values(), accumulate=True
    )


@dataclass(frozen=True, slots=True)
class SparseFaceStencil:
    """Connectivity-scaled face/cell flux operator.

    ``matrix`` stores one row per physical interior face and one column per
    reservoir cell. ``face_cells`` stores the oriented two-cell endpoints in
    exactly the same row order. No dense shadow copy is retained.
    """

    matrix: torch.Tensor
    face_cells: torch.Tensor
    nnz: int

    def __post_init__(self) -> None:
        if self.matrix.layout not in {torch.sparse_coo, torch.sparse_csr}:
            raise FlowContractError(
                "SparseFaceStencil matrix must use COO or CSR storage",
                object_name="SparseFaceStencil",
                field="matrix.layout",
                expected="torch.sparse_coo or torch.sparse_csr",
                actual=str(self.matrix.layout),
            )
        if self.face_cells.shape != (self.matrix.shape[0], 2):
            raise FlowContractError(
                "SparseFaceStencil endpoints must align with matrix rows",
                object_name="SparseFaceStencil",
                field="face_cells.shape",
                expected=(self.matrix.shape[0], 2),
                actual=tuple(self.face_cells.shape),
            )
        if self.nnz != self.matrix._nnz():
            raise FlowContractError(
                "SparseFaceStencil nnz must equal its stored entries",
                object_name="SparseFaceStencil",
                field="nnz",
                expected=self.matrix._nnz(),
                actual=self.nnz,
            )


def _pack_sparse_face_stencil(
    stencils: dict[int, dict[int, torch.Tensor]],
    face_cells: Sequence[Sequence[int]],
    *,
    n_cells: int,
    like: torch.Tensor,
) -> SparseFaceStencil:
    faces = sorted(stencils)
    rows: list[int] = []
    columns: list[int] = []
    values: list[torch.Tensor] = []
    endpoints: list[list[int]] = []
    for face in faces:
        cells = face_cells[face]
        if len(cells) != 2:
            continue
        row = len(endpoints)
        endpoints.append([int(cells[0]), int(cells[1])])
        for cell, coefficient in sorted(stencils[face].items()):
            rows.append(row)
            columns.append(int(cell))
            values.append(coefficient)
    indices = torch.tensor([rows, columns], dtype=torch.long, device=like.device)
    coefficients = torch.stack(values) if values else like.new_empty((0,))
    matrix = torch.sparse_coo_tensor(
        indices,
        coefficients,
        size=(len(endpoints), n_cells),
        dtype=like.dtype,
        device=like.device,
    ).coalesce()
    endpoint_tensor = torch.tensor(
        endpoints, dtype=torch.long, device=like.device
    ).reshape(-1, 2)
    return SparseFaceStencil(matrix, endpoint_tensor, matrix._nnz())


def assemble_sparse_face_stencil(
    grid: MPFAGrid2D,
    perm_tensor: torch.Tensor,
) -> SparseFaceStencil:
    """Assemble the no-flow MPFA face operator directly into sparse COO."""

    stencils = mpfa_o_face_flux_stencils_full(grid, perm_tensor)
    return _pack_sparse_face_stencil(
        stencils,
        grid.edge_cells,
        n_cells=len(grid.cell_nodes),
        like=perm_tensor,
    )


__all__ = [
    "MPFAGrid2D",
    "SparseFaceStencil",
    "assemble_mpfa_divergence",
    "assemble_sparse_face_stencil",
    "build_mpfa_grid",
    "mpfa_o_face_flux_stencils",
    "solve_mpfa_steady",
]
