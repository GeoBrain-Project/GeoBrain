"""Shared finite-volume operator factory on mesh records.

The records-based FV operator layer of the platform: free functions that turn a
:class:`~geobrain.mesh.capabilities.ConnectivityMesh`'s records into
sparse differential/averaging operators, plus differentiable coefficient
maps. The mesh classes themselves stay records-only; operator algebra is
NEVER a mesh method (the seam philosophy documented in
:mod:`~geobrain.mesh.capabilities`); this module is the single shared
consumer that every physics family may build on instead of hand-rolling its
own stencils.

Contracts (matching the records contract):

- Sparse operators are coalesced CPU float64 ``torch.sparse_coo_tensor``;
  CONSUMERS move them to their own device/dtype.
- Face (column/row) ordering is exactly the ``face_neighbors()`` /
  ``boundary_faces()`` record order of the mesh.
- Coefficient-dependent maps (:func:`harmonic_face_values`,
  :func:`face_transmissibility`) are differentiable torch *functions* of the
  per-cell values; autograd through them is the platform's native
  replacement for hand-written inner-product derivatives. They
  run at the input's dtype/device.

Sign/units conventions:

- :func:`face_divergence` / :func:`boundary_divergence` act on the flux
  *density* along each record's unit normal (i→j for internal faces,
  outward for boundary faces) and return the per-cell volume-normalised FV
  divergence: ``div u |_c = (1/V_c) * Σ_f ±area_f · u_f``.
- :func:`cell_gradient` returns the normal-directional two-point gradient
  ``(u_j − u_i) / (dist_l + dist_r)`` per internal face, exact for linear
  fields wherever the two cell centres are collinear with the face normal
  (tensor meshes, equal-size octree neighbours).
- :func:`face_transmissibility` uses the series-harmonic half-distance form
  ``σ_l·σ_r·area / (σ_l·dist_r + σ_r·dist_l + eps)``: the SINGLE form both
  legacy Poisson assemblers reduce to (see
  ``physics/em/numerics/finite_volume/poisson_fv.py``); keep it bit-mirrored
  so consolidation stays byte-identical.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.errors import GeoBrainError
from ..core._require import ensure_capable_mesh
from .capabilities import ConnectivityMesh, FaceRecords, StructuredMesh

__all__ = [
    "face_divergence",
    "boundary_divergence",
    "cell_gradient",
    "average_cell_to_face",
    "average_face_to_cell",
    "harmonic_face_values",
    "face_transmissibility",
    "edge_curl",
    "EdgeCurl",
]

_EPS = 1e-30  # matches the legacy Poisson assemblers' harmonic-mean reg.


def _connectivity(mesh, owner: str):
    ensure_capable_mesh(mesh, ConnectivityMesh, owner=owner)
    return mesh


def _face_records(mesh_or_records, owner: str) -> FaceRecords:
    if isinstance(mesh_or_records, FaceRecords):
        return mesh_or_records
    return _connectivity(mesh_or_records, owner).face_neighbors()


def _flat_volumes(mesh) -> torch.Tensor:
    return mesh.cell_volumes().reshape(-1).to(torch.float64)


def _coo(rows, cols, vals, shape) -> torch.Tensor:
    idx = torch.stack([rows, cols])
    return torch.sparse_coo_tensor(
        idx, vals.to(torch.float64), size=shape,
    ).coalesce()


def face_divergence(mesh) -> torch.Tensor:
    """Sparse ``(n_cells, n_internal_faces)`` FV divergence (internal part).

    Acts on flux densities along each face's record normal (i→j). The full
    FV divergence of a vector field is
    ``face_divergence @ u_int + boundary_divergence @ u_bnd``.
    """
    mesh = _connectivity(mesh, "face_divergence")
    fr = mesh.face_neighbors()
    vol = _flat_volumes(mesh)
    nf = fr.cell_i.numel()
    f_idx = torch.arange(nf, dtype=torch.long)
    rows = torch.cat([fr.cell_i, fr.cell_j])
    cols = torch.cat([f_idx, f_idx])
    vals = torch.cat([
        fr.area / vol[fr.cell_i],
        -fr.area / vol[fr.cell_j],
    ])
    return _coo(rows, cols, vals, (mesh.n_cells, nf))


def boundary_divergence(mesh) -> torch.Tensor:
    """Sparse ``(n_cells, n_boundary_faces)`` FV divergence (boundary part).

    Acts on flux densities along the OUTWARD boundary normals; complements
    :func:`face_divergence` to close the divergence theorem per cell.
    """
    mesh = _connectivity(mesh, "boundary_divergence")
    br = mesh.boundary_faces()
    vol = _flat_volumes(mesh)
    nb = br.cell.numel()
    return _coo(
        br.cell,
        torch.arange(nb, dtype=torch.long),
        br.area / vol[br.cell],
        (mesh.n_cells, nb),
    )


def cell_gradient(mesh) -> torch.Tensor:
    """Sparse ``(n_internal_faces, n_cells)`` two-point normal gradient.

    Per face: ``(u_j − u_i) / (dist_l + dist_r)``, the directional gradient
    along the record normal. Boundary faces carry no row (zero-Neumann by
    omission); stamp boundary conditions consumer-side.
    """
    mesh = _connectivity(mesh, "cell_gradient")
    fr = mesh.face_neighbors()
    nf = fr.cell_i.numel()
    inv_d = 1.0 / (fr.dist_l + fr.dist_r)
    f_idx = torch.arange(nf, dtype=torch.long)
    rows = torch.cat([f_idx, f_idx])
    cols = torch.cat([fr.cell_i, fr.cell_j])
    vals = torch.cat([-inv_d, inv_d])
    return _coo(rows, cols, vals, (nf, mesh.n_cells))


def average_cell_to_face(mesh) -> torch.Tensor:
    """Sparse ``(n_internal_faces, n_cells)`` distance-weighted interpolation.

    Linear interpolation of a cell-centred scalar to the face point:
    ``u_f = (dist_r·u_i + dist_l·u_j) / (dist_l + dist_r)``: exact for
    linear fields on tensor meshes.
    """
    mesh = _connectivity(mesh, "average_cell_to_face")
    fr = mesh.face_neighbors()
    nf = fr.cell_i.numel()
    inv_d = 1.0 / (fr.dist_l + fr.dist_r)
    f_idx = torch.arange(nf, dtype=torch.long)
    rows = torch.cat([f_idx, f_idx])
    cols = torch.cat([fr.cell_i, fr.cell_j])
    vals = torch.cat([fr.dist_r * inv_d, fr.dist_l * inv_d])
    return _coo(rows, cols, vals, (nf, mesh.n_cells))


def average_face_to_cell(mesh) -> torch.Tensor:
    """Sparse ``(n_cells, n_internal_faces)`` uniform face→cell averaging.

    Each cell averages the internal faces it touches with equal weight
    ``1/deg(cell)`` (rows of cells with no internal face stay empty).
    """
    mesh = _connectivity(mesh, "average_face_to_cell")
    fr = mesh.face_neighbors()
    nf = fr.cell_i.numel()
    f_idx = torch.arange(nf, dtype=torch.long)
    rows = torch.cat([fr.cell_i, fr.cell_j])
    cols = torch.cat([f_idx, f_idx])
    deg = torch.bincount(rows, minlength=mesh.n_cells).clamp(min=1)
    vals = 1.0 / deg[rows].to(torch.float64)
    return _coo(rows, cols, vals, (mesh.n_cells, nf))


@dataclass(frozen=True, eq=False)
class EdgeCurl:
    """Edge→face curl operator + the face/edge SoA that indexes it.

    ``matrix`` is sparse ``(n_all_faces, n_all_edges)`` over ALL faces
    (internal + boundary) and all unique grid edges; the curl needs the
    complete staggered sets, so it carries its own enumeration rather than
    reusing the internal-only :class:`FaceRecords`. Ordering: families in
    mesh-axis order (axis-0 faces/edges first), C-order (last axis fastest)
    within each family. ``matrix @ (E·t)`` returns the per-face circulation
    density ``∮E·dl / area`` with the right-hand rule taken about each
    face's ``+axis`` normal **in mesh-axis component order**, the mesh-axis
    triple is treated as right-handed; frame interpretation (the platform's
    z-down axis 0) is the consumer's concern.

    Attributes:
        matrix: sparse ``(nf, ne)`` signed edge-to-face incidence / length
            matrix implementing the discrete curl.
        face_centroids / face_normals / face_areas: face SoA the operator
            produces circulations on.
        edge_midpoints / edge_tangents / edge_lengths: edge SoA the
            operator consumes.
    """

    matrix: torch.Tensor
    face_centroids: torch.Tensor
    face_normals: torch.Tensor
    face_areas: torch.Tensor
    edge_midpoints: torch.Tensor
    edge_tangents: torch.Tensor
    edge_lengths: torch.Tensor


def edge_curl(mesh) -> EdgeCurl:
    """Discretize-parity edge curl for a 3-D structured (tensor) mesh.

    Built geometrically: each face row holds ``±edge_length / face_area``
    for its four boundary edges, the sign being the orientation of the
    edge's ``+axis`` tangent in the right-hand circulation about the face's
    ``+axis`` normal (``sign(t · (n × r))`` with ``r`` the midpoint offset).
    Validated entry-exact against a reference implementation by the
    operator parity suite.
    """
    ensure_capable_mesh(mesh, StructuredMesh, owner="edge_curl")
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "edge_curl requires a 3-D structured mesh",
            object_name="edge_curl",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    n = [int(s) for s in mesh.shape]
    centers = [t.to(torch.float64) for t in mesh.center_lines()]
    nodes = [t.to(torch.float64) for t in mesh.node_lines()]
    widths = [w.to(torch.float64) for w in mesh.cell_widths]

    # --- edge enumeration: family a runs along axis a (cell index along a,
    # node indices along the other two axes), C-order flatten per family ---
    edims, e_off = [], []
    e_mid, e_tan, e_len = [], [], []
    total_e = 0
    for a in range(3):
        dims = [n[0] + 1, n[1] + 1, n[2] + 1]
        dims[a] = n[a]
        edims.append(dims)
        coords = [nodes[0], nodes[1], nodes[2]]
        coords[a] = centers[a]
        grid = torch.meshgrid(*coords, indexing="ij")
        e_mid.append(torch.stack([g.reshape(-1) for g in grid], dim=1))
        n_fam = e_mid[a].shape[0]
        tan = torch.zeros(n_fam, 3, dtype=torch.float64)
        tan[:, a] = 1.0
        e_tan.append(tan)
        ia = torch.meshgrid(
            *[torch.arange(d) for d in dims], indexing="ij"
        )[a].reshape(-1)
        e_len.append(widths[a][ia])
        e_off.append(total_e)
        total_e += n_fam

    def _eflat(fam, i0, i1, i2):
        d = edims[fam]
        return e_off[fam] + (i0 * d[1] + i1) * d[2] + i2

    # --- face enumeration + the 4 boundary edges per face ---
    rows, cols, vals = [], [], []
    f_cent, f_norm, f_area = [], [], []
    total_f = 0
    for a in range(3):
        b, c = [ax for ax in range(3) if ax != a]
        fdims = [n[0], n[1], n[2]]
        fdims[a] = n[a] + 1
        grid = torch.meshgrid(
            *[torch.arange(d) for d in fdims], indexing="ij"
        )
        gidx = [g.reshape(-1) for g in grid]
        ia, ib, ic = gidx[a], gidx[b], gidx[c]
        nf_fam = ia.numel()

        coords = [None, None, None]
        coords[a] = nodes[a][ia]
        coords[b] = centers[b][ib]
        coords[c] = centers[c][ic]
        cent = torch.stack(coords, dim=1)
        nrm = torch.zeros(nf_fam, 3, dtype=torch.float64)
        nrm[:, a] = 1.0
        area = widths[b][ib] * widths[c][ic]
        f_cent.append(cent)
        f_norm.append(nrm)
        f_area.append(area)

        f_idx = total_f + torch.arange(nf_fam, dtype=torch.long)
        total_f += nf_fam

        for fam, other, off in ((b, c, 0), (b, c, 1), (c, b, 0), (c, b, 1)):
            ii = [None, None, None]
            ii[a] = ia
            ii[fam] = ib if fam == b else ic
            ii[other] = (ic if fam == b else ib) + off
            e = _eflat(fam, ii[0], ii[1], ii[2])
            local = e - e_off[fam]
            r = e_mid[fam][local] - cent
            orient = (e_tan[fam][local]
                      * torch.cross(nrm, r, dim=1)).sum(dim=1)
            sign = torch.sign(orient)
            rows.append(f_idx)
            cols.append(e)
            vals.append(sign * e_len[fam][local] / area)

    matrix = torch.sparse_coo_tensor(
        torch.stack([torch.cat(rows), torch.cat(cols)]),
        torch.cat(vals),
        size=(total_f, total_e),
    ).coalesce()
    return EdgeCurl(
        matrix=matrix,
        face_centroids=torch.cat(f_cent),
        face_normals=torch.cat(f_norm),
        face_areas=torch.cat(f_area),
        edge_midpoints=torch.cat(e_mid),
        edge_tangents=torch.cat(e_tan),
        edge_lengths=torch.cat(e_len),
    )


def _coefficients(mesh_or_records, cell_values, owner: str):
    fr = _face_records(mesh_or_records, owner)
    k = cell_values.reshape(-1)
    ci = fr.cell_i.to(k.device)
    cj = fr.cell_j.to(k.device)
    return fr, k, k[ci], k[cj]


def harmonic_face_values(mesh_or_records, cell_values,
                         eps: float = _EPS) -> torch.Tensor:
    """Per-face harmonic mean ``2·k_i·k_j / (k_i + k_j + eps)`` (differentiable).

    Accepts a :class:`ConnectivityMesh` or a :class:`FaceRecords` directly;
    ``cell_values`` may be any cell-shaped tensor (flattened in record cell
    order). Runs at the input's dtype/device; gradients flow to
    ``cell_values``.

    Args:
        mesh_or_records: a mesh exposing ``faces`` or the
            :class:`~geobrain.mesh.FaceRecords` directly.
        cell_values: ``(n_cells,)`` per-cell values to average.
        eps: floor keeping the denominator finite.
    """
    _, _, ki, kj = _coefficients(mesh_or_records, cell_values,
                                 "harmonic_face_values")
    return 2.0 * ki * kj / (ki + kj + eps)


def face_transmissibility(mesh_or_records, cell_values,
                          eps: float = _EPS) -> torch.Tensor:
    """Series-harmonic TPFA transmissibility per internal face (differentiable).

    ``T_f = k_i·k_j·area / (k_i·dist_r + k_j·dist_l + eps)``: the canonical
    half-distance form both legacy Poisson assemblers reduce to. Same input
    flexibility as :func:`harmonic_face_values`.

    Args:
        mesh_or_records: a mesh exposing ``faces`` or the
            :class:`~geobrain.mesh.FaceRecords` directly.
        cell_values: ``(n_cells,)`` per-cell coefficient (e.g. permeability).
        eps: floor keeping the harmonic denominator finite.
    """
    fr, k, ki, kj = _coefficients(mesh_or_records, cell_values,
                                  "face_transmissibility")
    area = fr.area.to(device=k.device, dtype=ki.dtype)
    dist_l = fr.dist_l.to(device=k.device, dtype=ki.dtype)
    dist_r = fr.dist_r.to(device=k.device, dtype=ki.dtype)
    return ki * kj * area / (ki * dist_r + kj * dist_l + eps)
