"""
MPFA-O in 3-D: the triplet interaction-region method (hexahedral grids).

The direct 3-D extension of the 2-D interaction-region MPFA-O
(:mod:`geobrain.physics.flow.discretization.mpfa`). Around each grid **vertex** the local
interaction region couples the surrounding cells: a continuity-point pressure
sits on each incident face, and in each cell-corner the pressure gradient is
linearly reconstructed from the cell centre and the **three** faces meeting there
(the "triplet", vs the 2-D duo)::

    ∇p_i = D_i⁻¹·[u_a − p_i ; u_b − p_i ; u_c − p_i]   (D_i = centre→face-centroid, 3×3)

Normal-flux continuity across every interior face, ``n·K_i·∇p_i = n·K_j·∇p_j``,
closes the local system for the continuity pressures; boundary-face continuity
points are set by Dirichlet data. The assembled operator gives the net outward
divergence, ``div_out = L_pp·p + L_pb·p_bc``, and a steady incompressible Darcy
BVP is ``L_pp·p + L_pb·p_bc - q = 0``. Because each cell's reconstructed gradient
is the exact gradient of a linear field, the scheme reproduces ``−∮ K∇p·n``
exactly on non-K-orthogonal hex grids with a full permeability tensor (the 3-D
patch test, verified to machine precision). Differentiable in permeability and
boundary data.

Grid: nodes (n,3) + VTK-ordered hexahedral cell-node lists (8 each); quad faces
are assumed planar (e.g. an affine-sheared Cartesian grid).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import torch

from ....core import GeoBrainError
from ..errors import FlowContractError
from .mpfa import (
    SparseFaceStencil,
    _compensated_sum,
    _explicit_dense_reference,
    _pack_sparse_face_stencil,
    _require_mpfa_derived,
    _stable_mean,
    _stable_positive_product,
    _stable_vector_norm,
    _sparse_from_triplets,
    _validate_cell_nodes,
    _validate_grid_nodes,
)

FaceStencils: TypeAlias = dict[int, dict[int, torch.Tensor]]
ScalarValue: TypeAlias = float | torch.Tensor

# the 6 quad faces of a VTK-ordered hexahedron (node-local indices, ordered round each face)
_HEX_FACES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),  # bottom (z−), top (z+)
    (0, 1, 5, 4),
    (3, 2, 6, 7),  # y−, y+
    (0, 3, 7, 4),
    (1, 2, 6, 5),  # x−, x+
)


class MPFAGrid3D:
    """Immutable SI hexahedral grid with canonical xyz coordinate columns."""

    def __init__(self, nodes: torch.Tensor, cell_nodes: list[list[int]]) -> None:
        _validate_grid_nodes(nodes, dimension=3, object_name="MPFAGrid3D")
        _validate_cell_nodes(
            cell_nodes,
            n_nodes=int(nodes.shape[0]),
            minimum_nodes=8,
            object_name="MPFAGrid3D",
        )
        for cell, node_ids in enumerate(cell_nodes):
            if len(node_ids) != 8:
                raise FlowContractError(
                    "MPFAGrid3D requires exactly eight VTK nodes per cell",
                    object_name="MPFAGrid3D",
                    field=f"cell_nodes[{cell}]",
                    expected=8,
                    actual=len(node_ids),
                )
        self._nodes = nodes.clone()
        self._cell_nodes = tuple(tuple(node_ids) for node_ids in cell_nodes)
        minimum = self._nodes.amin(dim=0)
        self._origin_m = tuple(float(value.detach()) for value in minimum)
        self._cell_centroids = torch.stack(
            [_stable_mean(self._nodes[list(node_ids)]) for node_ids in self._cell_nodes]
        )
        _require_mpfa_derived("MPFAGrid3D", "cell_centroids", self._cell_centroids)
        fmap: dict[tuple[int, ...], int] = {}
        f_nodes: list[tuple[int, ...]] = []
        f_cells: list[list[int]] = []
        for ci, cn in enumerate(self._cell_nodes):
            for loc in _HEX_FACES:
                quad = tuple(cn[i] for i in loc)
                key = tuple(sorted(quad))
                if key not in fmap:
                    fmap[key] = len(f_nodes)
                    f_nodes.append(quad)
                    f_cells.append([])
                f_cells[fmap[key]].append(ci)
        node_faces: list[list[int]] = [[] for _ in range(self._nodes.shape[0])]
        node_cells: list[list[int]] = [[] for _ in range(self._nodes.shape[0])]
        for fi, quad in enumerate(f_nodes):
            for nd in set(quad):
                node_faces[nd].append(fi)
        for ci, cn in enumerate(self._cell_nodes):
            for nd in set(cn):
                node_cells[nd].append(ci)
        self._face_nodes = tuple(f_nodes)
        self._face_cells = tuple(tuple(cells) for cells in f_cells)
        self._node_faces = tuple(tuple(faces) for faces in node_faces)
        self._node_cells = tuple(tuple(cells) for cells in node_cells)
        for face in range(self.n_faces):
            centroid = self.face_centroid(face)
            normal, area = self.face_normal_area(face)
            _require_mpfa_derived("MPFAGrid3D", f"face[{face}].centroid", centroid)
            _require_mpfa_derived("MPFAGrid3D", f"face[{face}].normal", normal)
            _require_mpfa_derived("MPFAGrid3D", f"face[{face}].area", area, positive=True)
        hex_cell_volumes(self)

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
    def face_nodes(self) -> tuple[tuple[int, ...], ...]:
        return self._face_nodes

    @property
    def face_cells(self) -> tuple[tuple[int, ...], ...]:
        return self._face_cells

    @property
    def node_faces(self) -> tuple[tuple[int, ...], ...]:
        return self._node_faces

    @property
    def node_cells(self) -> tuple[tuple[int, ...], ...]:
        return self._node_cells

    @property
    def coordinate_columns(self) -> tuple[str, str, str]:
        return ("x", "y", "z")

    @property
    def z_positive_down(self) -> bool:
        return True

    @property
    def origin_m(self) -> tuple[float, ...]:
        return self._origin_m

    @property
    def dtype(self) -> torch.dtype:
        return self._nodes.dtype

    @property
    def device(self) -> torch.device:
        return self._nodes.device

    @property
    def n_faces(self) -> int:
        return len(self._face_nodes)

    def face_centroid(self, f: int) -> torch.Tensor:
        return _stable_mean(self._nodes[list(self._face_nodes[f])])

    def face_normal_area(self, f: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Unit normal and area of (planar) quad face ``f`` via its diagonals."""
        a, b, c, d = (self._nodes[i] for i in self._face_nodes[f])
        first_diagonal = c - a
        second_diagonal = d - b
        direct_cross = torch.linalg.cross(first_diagonal, second_diagonal)
        if bool(torch.isfinite(direct_cross).all()) and not bool(
            (direct_cross == 0).all()
            & (first_diagonal.abs().amax() > 0)
            & (second_diagonal.abs().amax() > 0)
        ):
            cross_norm = _stable_vector_norm(direct_cross)
            safe_norm = torch.where(cross_norm > 0, cross_norm, torch.ones_like(cross_norm))
            return direct_cross / safe_norm, 0.5 * cross_norm

        # Scale only after the ordinary path proves unrepresentable.  This
        # preserves its established rounding on reservoir-scale coordinates.
        first_scale = first_diagonal.abs().amax()
        second_scale = second_diagonal.abs().amax()
        safe_first_scale = torch.where(
            torch.isfinite(first_scale) & (first_scale > 0),
            first_scale,
            torch.ones_like(first_scale),
        )
        safe_second_scale = torch.where(
            torch.isfinite(second_scale) & (second_scale > 0),
            second_scale,
            torch.ones_like(second_scale),
        )
        scaled_cross = torch.linalg.cross(
            first_diagonal / safe_first_scale,
            second_diagonal / safe_second_scale,
        )
        scaled_norm = _stable_vector_norm(scaled_cross)
        safe_norm = torch.where(scaled_norm > 0, scaled_norm, torch.ones_like(scaled_norm))
        normal = scaled_cross / safe_norm
        area = ((0.5 * scaled_norm) * first_scale) * second_scale
        return normal, area


def _validate_mpfa_3d_field(
    grid: MPFAGrid3D,
    field_tensor: torch.Tensor,
    *,
    field: str,
) -> None:
    if field_tensor.device != grid.device or field_tensor.dtype != grid.dtype:
        raise FlowContractError(
            f"{field} dtype/device must match the MPFA grid",
            object_name="MPFAGrid3D",
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
            object_name="MPFAGrid3D",
            field=field,
            expected=len(grid.cell_nodes),
            actual=None if field_tensor.ndim == 0 else int(field_tensor.shape[0]),
        )


def build_mpfa_grid_3d(nodes: torch.Tensor, cell_nodes: list[list[int]]) -> MPFAGrid3D:
    return MPFAGrid3D(nodes=nodes, cell_nodes=list(cell_nodes))


def hex_cell_volumes(grid: MPFAGrid3D) -> torch.Tensor:
    """Per-cell hexahedron volume by tetra-decomposition of its six faces.

    Each of the six quad faces is fanned into four triangles, each forming a
    tetrahedron with the cell centre; ``Σ |det| / 6`` is the volume (the
    ``_HEX_FACES`` ordering gives opposite faces opposite orientation, so the
    *signed* sum cancels, the absolute value per tet is required).
    """
    vols = []
    for cn in grid.cell_nodes:
        pts = grid._nodes_view()[list(cn)]
        centre = _stable_mean(pts)
        tetrahedra: list[torch.Tensor] = []
        for loc in _HEX_FACES:
            quad = pts[list(loc)]
            fc = _stable_mean(quad)
            for t in range(4):
                a = quad[t] - fc
                b = quad[(t + 1) % 4] - fc
                vectors = torch.stack([a, b, centre - fc])
                tetrahedra.append(vectors)
        stacked_tetrahedra = torch.stack(tetrahedra)
        column_scales = stacked_tetrahedra.abs().amax(dim=(0, 1))
        safe_column_scales = torch.where(
            torch.isfinite(column_scales) & (column_scales > 0),
            column_scales,
            torch.ones_like(column_scales),
        )
        normalized_determinants = torch.linalg.det(stacked_tetrahedra / safe_column_scales).abs()
        normalized_volume = _compensated_sum(normalized_determinants) / 6.0
        v = _stable_positive_product(
            (
                normalized_volume,
                safe_column_scales[0],
                safe_column_scales[1],
                safe_column_scales[2],
            )
        )
        vols.append(v)
    volumes = torch.stack(vols)
    _require_mpfa_derived("MPFAGrid3D", "cell_volumes", volumes, positive=True)
    return volumes


def assemble_mpfa_divergence_3d(
    grid: MPFAGrid3D, perm_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """3-D MPFA-O flux operator over all vertices (Dirichlet on boundary faces).

    Returns sparse COO blocks ``(L_pp, L_pb, boundary_faces)``: net outward divergence is
    ``L_pp @ p + L_pb @ p_bc`` (``p_bc`` = Dirichlet pressure at each boundary-face
    centroid, order = ``boundary_faces``). Differentiable in ``perm_tensor``;
    stored entries scale with local connectivity rather than ``n_cells²``.
    """
    _validate_mpfa_3d_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (3, 3):
        raise GeoBrainError(
            "3-D MPFA-O needs (n_cells, 3, 3) permeability",
            object_name="assemble_mpfa_divergence_3d",
            field="perm_tensor",
            expected="(n_cells, 3, 3)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    nC = len(grid.cell_nodes)
    dt = perm_tensor.dtype
    boundary_faces = [f for f in range(grid.n_faces) if len(grid.face_cells[f]) == 1]
    b_index = {f: k for k, f in enumerate(boundary_faces)}
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
        F_v = grid.node_faces[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [f for f in F_v if len(grid.face_cells[f]) == 2]
        B_v = [f for f in F_v if len(grid.face_cells[f]) == 1]
        if any(len([f for f in F_v if c in grid.face_cells[f]]) != 3 for c in C_v):
            continue  # each hex corner must touch exactly 3 faces here
        nI, nCv, nBv = len(I_v), len(C_v), len(B_v)
        width = nI + nCv + nBv
        iu = {f: k for k, f in enumerate(I_v)}
        ip = {c: nI + k for k, c in enumerate(C_v)}
        ib = {f: nI + nCv + k for k, f in enumerate(B_v)}
        mid = {f: grid.face_centroid(f) for f in F_v}

        def col(f: int) -> int:
            return iu[f] if f in iu else ib[f]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            fa, fb, fc = [f for f in F_v if c in grid.face_cells[f]]
            D = torch.stack([mid[fa] - cc[c], mid[fb] - cc[c], mid[fc] - cc[c]], dim=0)  # (3,3)
            if float(torch.linalg.det(D).detach().abs()) < 1e-300:
                region_valid = False
                break
            sel = perm_tensor.new_zeros(3, width)
            for r, f in enumerate((fa, fb, fc)):
                sel[r, col(f)] += 1.0
                sel[r, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(D) @ sel
        if not region_valid:
            continue

        if nI > 0:
            R = perm_tensor.new_zeros(nI, width)
            for f in I_v:
                ci, cj = grid.face_cells[f]
                n, _ = grid.face_normal_area(f)
                R[iu[f]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            R_u = R[:, :nI]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                continue
            U = -torch.linalg.solve(R_u, R[:, nI:])
            W = torch.cat(
                [
                    U,
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

        for f in F_v:
            n, area = grid.face_normal_area(f)
            A_sub = 0.25 * area  # corner sub-face ≈ ¼ of the quad face
            if len(grid.face_cells[f]) == 2:  # interior face: flux left→right
                ci, cj = grid.face_cells[f]
                if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                    n = -n
                coef = -A_sub * (n @ (perm_tensor[ci] @ (G[ci] @ W)))
                for k, c in enumerate(C_v):
                    add_pp(ci, c, -coef[k])
                    add_pp(cj, c, coef[k])
                for k, fb in enumerate(B_v):
                    add_pb(ci, b_index[fb], -coef[nCv + k])
                    add_pb(cj, b_index[fb], coef[nCv + k])
            else:  # boundary face: flux into its cell
                c = grid.face_cells[f][0]
                if float((n @ (mid[f] - cc[c])).detach()) < 0.0:
                    n = -n
                coef = A_sub * (n @ (perm_tensor[c] @ (G[c] @ W)))
                for k, cc_ in enumerate(C_v):
                    add_pp(c, cc_, coef[k])
                for k, fb in enumerate(B_v):
                    add_pb(c, b_index[fb], coef[nCv + k])
    # Normalise the historical local flux-into assembly to canonical outward
    # divergence at the public API boundary.
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
        shape=(nC, len(boundary_faces)),
        like=perm_tensor,
    )
    return L_pp, L_pb, boundary_faces


def solve_mpfa_steady_3d(
    grid: MPFAGrid3D,
    perm_tensor: torch.Tensor,
    bc_values: torch.Tensor,
    source: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit dense-reference solve for a small 3-D MPFA-O Dirichlet BVP.

    Production reservoir paths consume :func:`assemble_mpfa_divergence_3d`
    with a declared sparse solver; no layout fallback occurs here.
    """
    L_pp, L_pb, _ = assemble_mpfa_divergence_3d(grid, perm_tensor)
    rhs = -(L_pb @ bc_values)
    if source is not None:
        rhs = rhs + source
    return torch.linalg.solve(_explicit_dense_reference(L_pp), rhs)


def mpfa_o_face_flux_stencils_3d_bc(
    grid: MPFAGrid3D,
    perm_tensor: torch.Tensor,
    dirichlet_faces: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Per-face 3-D MPFA-O flux stencils with **Dirichlet pressure boundaries**,
    as a ghost-cell augmented operator (3-D counterpart of
    :func:`~geobrain.physics.flow.discretization.mpfa.mpfa_o_face_flux_stencils_bc`).

    Boundary faces in ``dirichlet_faces`` connect their cell to a fixed-pressure
    ghost cell; every other boundary face is no-flow. Returns ``(L, face_lr,
    dirichlet_faces)`` with ``L`` ``(n_faces, n_cells + n_dir)``, ``face_lr``
    ``(n_faces, 2)`` augmented endpoints (ghost index = ``n_cells + k``), and the
    flux endpoint0→endpoint1 of face ``f`` = ``(L @ φ_aug)[f]``. Summed with unit
    mobility this reproduces :func:`assemble_mpfa_divergence_3d_mixed`, so the
    single-phase limit matches :func:`solve_mpfa_bvp_3d` on sheared full-tensor hex.
    """
    _validate_mpfa_3d_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (3, 3):
        raise GeoBrainError(
            "3-D MPFA-O needs (n_cells, 3, 3) permeability",
            object_name="mpfa_o_face_flux_stencils_3d_bc",
            field="perm_tensor",
            expected="(n_cells, 3, 3)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    nC = len(grid.cell_nodes)
    dt = perm_tensor.dtype
    boundary = {f for f in range(grid.n_faces) if len(grid.face_cells[f]) == 1}
    dir_set = {int(f) for f in dirichlet_faces}
    bad = dir_set - boundary
    if bad:
        raise GeoBrainError(
            "dirichlet_faces must all be boundary faces",
            object_name="mpfa_o_face_flux_stencils_3d_bc",
            field="dirichlet_faces",
            expected="boundary face ids",
            actual=sorted(bad),
        )
    dir_list = [f for f in sorted(boundary) if f in dir_set]
    g_index = {f: nC + k for k, f in enumerate(dir_list)}
    n_aug = nC + len(dir_list)

    fstencil: dict[int, dict[int, torch.Tensor]] = {}
    flr: dict[int, tuple[int, int]] = {}

    for v in range(grid._nodes_view().shape[0]):
        F_v = grid.node_faces[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [f for f in F_v if len(grid.face_cells[f]) == 2]  # interior (unknown)
        N_v = [f for f in F_v if f in boundary and f not in dir_set]  # no-flow boundary (unknown)
        Dd_v = [f for f in F_v if f in dir_set]  # Dirichlet (ghost)
        if any(len([f for f in F_v if c in grid.face_cells[f]]) != 3 for c in C_v):
            continue
        nI, nN, nCv, nDv = len(I_v), len(N_v), len(C_v), len(Dd_v)
        nU = nI + nN
        width = nU + nCv + nDv  # [u_I, u_N, p, p_D]
        iuI = {f: k for k, f in enumerate(I_v)}
        iuN = {f: nI + k for k, f in enumerate(N_v)}
        ip = {c: nU + k for k, c in enumerate(C_v)}
        idd = {f: nU + nCv + k for k, f in enumerate(Dd_v)}
        mid = {f: grid.face_centroid(f) for f in F_v}

        def col(f: int) -> int:
            if f in iuI:
                return iuI[f]
            if f in iuN:
                return iuN[f]
            return idd[f]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            fa, fb, fc = [f for f in F_v if c in grid.face_cells[f]]
            Dm = torch.stack([mid[fa] - cc[c], mid[fb] - cc[c], mid[fc] - cc[c]], dim=0)
            if float(torch.linalg.det(Dm).detach().abs()) < 1e-300:
                region_valid = False
                break
            sel = perm_tensor.new_zeros(3, width)
            for rr, f in enumerate((fa, fb, fc)):
                sel[rr, col(f)] += 1.0
                sel[rr, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(Dm) @ sel
        if not region_valid:
            continue

        if nU > 0:
            R = perm_tensor.new_zeros(nU, width)
            for f in I_v:
                ci, cj = grid.face_cells[f]
                n, _ = grid.face_normal_area(f)
                R[iuI[f]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            for f in N_v:
                c = grid.face_cells[f][0]
                n, area = grid.face_normal_area(f)
                if float((n @ (mid[f] - cc[c])).detach()) < 0.0:
                    n = -n
                R[iuN[f]] = -(0.25 * area) * (n @ (perm_tensor[c] @ G[c]))
            R_u = R[:, :nU]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                continue
            W = torch.cat(
                [
                    -torch.linalg.solve(R_u, R[:, nU:]),
                    torch.eye(nCv + nDv, dtype=dt, device=perm_tensor.device),
                ],
                dim=0,
            )
        else:
            W = torch.eye(nCv + nDv, dtype=dt, device=perm_tensor.device)

        aug_cols = [c for c in C_v] + [g_index[f] for f in Dd_v]

        def accumulate(face: int, a0: int, a1: int, coef: torch.Tensor) -> None:
            flr.setdefault(face, (a0, a1))
            d = fstencil.setdefault(face, {})
            for k, ac in enumerate(aug_cols):
                d[ac] = d.get(ac, perm_tensor.new_zeros(())) + coef[k]

        for f in I_v:  # interior face: flux l→r
            ci, cj = grid.face_cells[f]
            n, area = grid.face_normal_area(f)
            if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                n = -n
            coef = -(0.25 * area) * (n @ (perm_tensor[ci] @ (G[ci] @ W)))
            accumulate(f, ci, cj, coef)
        for f in Dd_v:  # Dirichlet face: flux cell→ghost
            c = grid.face_cells[f][0]
            n, area = grid.face_normal_area(f)
            if float((n @ (mid[f] - cc[c])).detach()) < 0.0:
                n = -n  # outward from the cell
            coef = -(0.25 * area) * (n @ (perm_tensor[c] @ (G[c] @ W)))
            accumulate(f, c, g_index[f], coef)

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


def mpfa_o_face_flux_stencils_3d_full(
    grid: MPFAGrid3D, perm_tensor: torch.Tensor
) -> FaceStencils:
    """Per-interior-face 3-D MPFA-O flux stencils for a **no-flow-bounded** domain.

    The 3-D analogue of
    :func:`~geobrain.physics.flow.discretization.mpfa.mpfa_o_face_flux_stencils_full`: process
    every vertex (interior *and* boundary), close boundary faces as no-flow
    (Neumann ``q = 0``), and accumulate each interior face's four corner
    quarter-face contributions, so a boundary-adjacent face still receives its
    full flux and, on a K-orthogonal hex grid, the stencil collapses to the
    seven-point ``T = K·A/d``.

    Returns ``{face: {cell: T_c}}`` with ``flux_{l→r} = Σ_c T_c·p_c`` (``l, r =
    grid.face_cells[face]``), differentiable in ``perm_tensor``. This is the
    stencil a no-flow 3-D multiphase transient upwinds the phase mobility onto.
    """
    _validate_mpfa_3d_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (3, 3):
        raise GeoBrainError(
            "3-D MPFA-O needs (n_cells, 3, 3) permeability",
            object_name="mpfa_o_face_flux_stencils_3d_full",
            field="perm_tensor",
            expected="(n_cells, 3, 3)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    dt = perm_tensor.dtype
    boundary = {f for f in range(grid.n_faces) if len(grid.face_cells[f]) == 1}
    stencils: dict[int, dict[int, torch.Tensor]] = {}

    for v in range(grid._nodes_view().shape[0]):
        F_v = grid.node_faces[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [f for f in F_v if len(grid.face_cells[f]) == 2]  # interior (unknown u)
        N_v = [f for f in F_v if f in boundary]  # boundary no-flow (unknown u)
        if any(len([f for f in F_v if c in grid.face_cells[f]]) != 3 for c in C_v):
            continue
        nI, nN, nCv = len(I_v), len(N_v), len(C_v)
        nU = nI + nN
        width = nU + nCv  # [u_I, u_N, p]
        iuI = {f: k for k, f in enumerate(I_v)}
        iuN = {f: nI + k for k, f in enumerate(N_v)}
        ip = {c: nU + k for k, c in enumerate(C_v)}
        mid = {f: grid.face_centroid(f) for f in F_v}

        def col(f: int) -> int:
            return iuI[f] if f in iuI else iuN[f]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            fa, fb, fc = [f for f in F_v if c in grid.face_cells[f]]
            D = torch.stack([mid[fa] - cc[c], mid[fb] - cc[c], mid[fc] - cc[c]], dim=0)
            if float(torch.linalg.det(D).detach().abs()) < 1e-300:
                region_valid = False
                break
            sel = perm_tensor.new_zeros(3, width)
            for r, f in enumerate((fa, fb, fc)):
                sel[r, col(f)] += 1.0
                sel[r, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(D) @ sel
        if not region_valid:
            continue

        if nU > 0:  # interior continuity + no-flow matching
            R = perm_tensor.new_zeros(nU, width)
            for f in I_v:
                ci, cj = grid.face_cells[f]
                n, _ = grid.face_normal_area(f)
                R[iuI[f]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            for f in N_v:  # ¼-face flux out = 0, homogeneous
                c = grid.face_cells[f][0]
                n, area = grid.face_normal_area(f)
                if float((n @ (mid[f] - cc[c])).detach()) < 0.0:
                    n = -n
                R[iuN[f]] = -(0.25 * area) * (n @ (perm_tensor[c] @ G[c]))
            R_u = R[:, :nU]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                continue
            W = torch.cat(
                [
                    -torch.linalg.solve(R_u, R[:, nU:]),
                    torch.eye(nCv, dtype=dt, device=perm_tensor.device),
                ],
                dim=0,
            )  # (width, nCv): [u;p]=W@p
        else:
            W = torch.eye(nCv, dtype=dt, device=perm_tensor.device)

        for f in I_v:  # accumulate ¼-face flux (l→r) per face
            ci, cj = grid.face_cells[f]
            n, area = grid.face_normal_area(f)
            if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                n = -n
            coef = -(0.25 * area) * (n @ (perm_tensor[ci] @ (G[ci] @ W)))  # (nCv,) in terms of p
            d = stencils.setdefault(f, {})
            for k, c in enumerate(C_v):
                d[c] = d.get(c, perm_tensor.new_zeros(())) + coef[k]
    return stencils


def assemble_mpfa_divergence_3d_mixed(
    grid: MPFAGrid3D,
    perm_tensor: torch.Tensor,
    neumann_faces: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int]]:
    """3-D MPFA-O flux operator with **mixed Dirichlet / Neumann** boundaries.

    The 3-D counterpart of :func:`~geobrain.physics.flow.discretization.mpfa.assemble_mpfa_divergence_mixed`.
    Boundary faces in ``neumann_faces`` carry a *prescribed total outward flux*
    ``q_N`` (``q_N = 0`` ⇒ no-flow / impermeable); their continuity points become
    unknowns closed by flux matching, each of the face's four corner sub-faces
    carries ``¼·q_N`` of the reconstructed flux. Every other boundary face is
    Dirichlet (prescribed pressure at its centroid).

    Returns ``(L_pp, L_pb, L_pn, dirichlet_faces, neumann_faces)`` with net outward
    divergence ``L_pp @ p + L_pb @ p_D + L_pn @ q_N`` (all blocks
    differentiable in ``perm_tensor``). Steady incompressible balance with
    sources ``s`` (injection +): ``L_pp @ p + L_pb @ p_D + L_pn @ q_N - s = 0``,
    pin a datum cell if there is no Dirichlet face.
    """
    _validate_mpfa_3d_field(grid, perm_tensor, field="perm_tensor")
    if perm_tensor.shape[-2:] != (3, 3):
        raise GeoBrainError(
            "3-D MPFA-O needs (n_cells, 3, 3) permeability",
            object_name="assemble_mpfa_divergence_3d_mixed",
            field="perm_tensor",
            expected="(n_cells, 3, 3)",
            actual=tuple(perm_tensor.shape),
        )
    cc = grid._cell_centroids_view()
    nC = len(grid.cell_nodes)
    dt = perm_tensor.dtype
    boundary_faces = [f for f in range(grid.n_faces) if len(grid.face_cells[f]) == 1]
    neu = {int(f) for f in neumann_faces}
    bad = neu - set(boundary_faces)
    if bad:
        raise GeoBrainError(
            "neumann_faces must all be boundary faces",
            object_name="assemble_mpfa_divergence_3d_mixed",
            field="neumann_faces",
            expected="boundary face ids",
            actual=sorted(bad),
        )
    dirichlet_faces = [f for f in boundary_faces if f not in neu]
    neumann_list = [f for f in boundary_faces if f in neu]
    d_index = {f: k for k, f in enumerate(dirichlet_faces)}
    n_index = {f: k for k, f in enumerate(neumann_list)}
    L_pp = perm_tensor.new_zeros(nC, nC)
    L_pb = perm_tensor.new_zeros(nC, len(dirichlet_faces))
    L_pn = perm_tensor.new_zeros(nC, len(neumann_list))

    for v in range(grid._nodes_view().shape[0]):
        F_v = grid.node_faces[v]
        C_v = grid.node_cells[v]
        if not C_v:
            continue
        I_v = [f for f in F_v if len(grid.face_cells[f]) == 2]  # interior (unknown)
        N_v = [f for f in F_v if f in neu]  # Neumann (unknown)
        Dd_v = [
            f for f in F_v if len(grid.face_cells[f]) == 1 and f not in neu
        ]  # Dirichlet (known)
        if any(len([f for f in F_v if c in grid.face_cells[f]]) != 3 for c in C_v):
            continue
        nI, nN, nCv, nDv = len(I_v), len(N_v), len(C_v), len(Dd_v)
        nU = nI + nN
        width = nU + nCv + nDv + nN  # [u_I, u_N, p, p_D, q_N]
        iuI = {f: k for k, f in enumerate(I_v)}
        iuN = {f: nI + k for k, f in enumerate(N_v)}
        ip = {c: nU + k for k, c in enumerate(C_v)}
        idd = {f: nU + nCv + k for k, f in enumerate(Dd_v)}
        iqn = {f: nU + nCv + nDv + k for k, f in enumerate(N_v)}
        mid = {f: grid.face_centroid(f) for f in F_v}

        def col(f: int) -> int:
            if f in iuI:
                return iuI[f]
            if f in iuN:
                return iuN[f]
            return idd[f]

        G: dict[int, torch.Tensor] = {}
        region_valid = True
        for c in C_v:
            fa, fb, fc = [f for f in F_v if c in grid.face_cells[f]]
            D = torch.stack([mid[fa] - cc[c], mid[fb] - cc[c], mid[fc] - cc[c]], dim=0)
            if float(torch.linalg.det(D).detach().abs()) < 1e-300:
                region_valid = False
                break
            sel = perm_tensor.new_zeros(3, width)
            for r, f in enumerate((fa, fb, fc)):
                sel[r, col(f)] += 1.0
                sel[r, ip[c]] -= 1.0
            G[c] = torch.linalg.inv(D) @ sel
        if not region_valid:
            continue

        if nU > 0:
            R = perm_tensor.new_zeros(nU, width)
            for f in I_v:
                ci, cj = grid.face_cells[f]
                n, _ = grid.face_normal_area(f)
                R[iuI[f]] = n @ (perm_tensor[ci] @ G[ci]) - n @ (perm_tensor[cj] @ G[cj])
            for f in N_v:  # ¼-face flux OUT of cell = ¼·q_N
                c = grid.face_cells[f][0]
                n, area = grid.face_normal_area(f)
                if float((n @ (mid[f] - cc[c])).detach()) < 0.0:
                    n = -n
                R[iuN[f]] = -(0.25 * area) * (n @ (perm_tensor[c] @ G[c]))
                R[iuN[f], iqn[f]] += -0.25
            R_u = R[:, :nU]
            if float(torch.linalg.det(R_u).detach().abs()) < 1e-300:
                continue
            U = -torch.linalg.solve(R_u, R[:, nU:])
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
            )
        else:
            W = torch.eye(
                nCv + nDv + nN,
                dtype=dt,
                device=perm_tensor.device,
            )

        for f in F_v:
            n, area = grid.face_normal_area(f)
            A_sub = 0.25 * area
            if len(grid.face_cells[f]) == 2:  # interior face: flux left→right
                ci, cj = grid.face_cells[f]
                if float((n @ (cc[cj] - cc[ci])).detach()) < 0.0:
                    n = -n
                coef = -A_sub * (n @ (perm_tensor[ci] @ (G[ci] @ W)))
                for k, c in enumerate(C_v):
                    L_pp[ci, c] += -coef[k]
                    L_pp[cj, c] += coef[k]
                for k, f_d in enumerate(Dd_v):
                    L_pb[ci, d_index[f_d]] += -coef[nCv + k]
                    L_pb[cj, d_index[f_d]] += coef[nCv + k]
                for k, f_n in enumerate(N_v):
                    L_pn[ci, n_index[f_n]] += -coef[nCv + nDv + k]
                    L_pn[cj, n_index[f_n]] += coef[nCv + nDv + k]
            elif f in neu:  # Neumann face: prescribed ¼-flux
                c = grid.face_cells[f][0]
                L_pn[c, n_index[f]] += -0.25  # flux INTO c = −¼·q_N per corner
            else:  # Dirichlet face: flux into its cell
                c = grid.face_cells[f][0]
                if float((n @ (mid[f] - cc[c])).detach()) < 0.0:
                    n = -n
                coef = A_sub * (n @ (perm_tensor[c] @ (G[c] @ W)))
                for k, cc_ in enumerate(C_v):
                    L_pp[c, cc_] += coef[k]
                for k, f_d in enumerate(Dd_v):
                    L_pb[c, d_index[f_d]] += coef[nCv + k]
                for k, f_n in enumerate(N_v):
                    L_pn[c, n_index[f_n]] += coef[nCv + nDv + k]
    return -L_pp, -L_pb, -L_pn, dirichlet_faces, neumann_list


def solve_mpfa_bvp_3d(
    grid: MPFAGrid3D,
    perm_tensor: torch.Tensor,
    *,
    dirichlet: Mapping[int, ScalarValue] | None = None,
    neumann: Mapping[int, ScalarValue] | None = None,
    source: torch.Tensor | None = None,
    datum: tuple[int, ScalarValue] | None = None,
) -> torch.Tensor:
    """Solve the steady 3-D MPFA-O BVP with mixed Dirichlet / Neumann boundaries.

    ``dirichlet``: ``{boundary_face: pressure}``. Every boundary face **not** in
    ``dirichlet`` is Neumann with outward flux ``neumann.get(face, 0.0)``, the
    default boundary is no-flow (impermeable). ``source``: per-cell injection
    (+). Supply ``datum=(cell, value)`` to pin the constant of a pure-Neumann
    problem. Differentiable in ``perm_tensor`` / BC values.
    """
    dirichlet = {int(f): v for f, v in (dirichlet or {}).items()}
    neumann = {int(f): v for f, v in (neumann or {}).items()}
    boundary_faces = [f for f in range(grid.n_faces) if len(grid.face_cells[f]) == 1]
    neumann_faces = [f for f in boundary_faces if f not in dirichlet]
    L_pp, L_pb, L_pn, d_faces, n_faces = assemble_mpfa_divergence_3d_mixed(
        grid, perm_tensor, neumann_faces
    )
    nC = L_pp.shape[0]
    dt = perm_tensor.dtype

    def as_col(
        values: Mapping[int, ScalarValue],
        faces: Sequence[int],
        default: ScalarValue | None = None,
    ) -> torch.Tensor:
        if not faces:
            return perm_tensor.new_zeros(0)
        if default is None:
            return torch.stack(
                [
                    torch.as_tensor(values[f], dtype=dt, device=perm_tensor.device)
                    for f in faces
                ]
            )
        return torch.stack(
            [
                torch.as_tensor(
                    values.get(f, default), dtype=dt, device=perm_tensor.device
                )
                for f in faces
            ]
        )

    p_D = as_col(dirichlet, d_faces)
    q_N = as_col(neumann, n_faces, default=0.0)
    s = source if source is not None else perm_tensor.new_zeros(nC)
    rhs = s - (L_pb @ p_D if d_faces else 0.0) - (L_pn @ q_N if n_faces else 0.0)

    A = L_pp
    if datum is not None:
        dcell, dval = int(datum[0]), torch.as_tensor(
            datum[1], dtype=dt, device=perm_tensor.device
        )
        row = perm_tensor.new_zeros(nC)
        row[dcell] = 1.0
        A = torch.cat([A[:dcell], row.unsqueeze(0), A[dcell + 1 :]], dim=0)
        rhs = torch.cat([rhs[:dcell], dval.reshape(1), rhs[dcell + 1 :]], dim=0)
    return torch.linalg.solve(A, rhs)


def assemble_sparse_face_stencil_3d(
    grid: MPFAGrid3D,
    perm_tensor: torch.Tensor,
) -> SparseFaceStencil:
    """Assemble the no-flow 3-D MPFA face operator directly into sparse COO."""

    stencils = mpfa_o_face_flux_stencils_3d_full(grid, perm_tensor)
    return _pack_sparse_face_stencil(
        stencils,
        grid.face_cells,
        n_cells=len(grid.cell_nodes),
        like=perm_tensor,
    )


__all__ = [
    "MPFAGrid3D",
    "build_mpfa_grid_3d",
    "hex_cell_volumes",
    "assemble_mpfa_divergence_3d",
    "assemble_mpfa_divergence_3d_mixed",
    "assemble_sparse_face_stencil_3d",
    "mpfa_o_face_flux_stencils_3d_full",
    "mpfa_o_face_flux_stencils_3d_bc",
    "solve_mpfa_steady_3d",
    "solve_mpfa_bvp_3d",
]
