"""Axisymmetric (E_phi) frequency-domain EM on a CylindricalMesh.

The mimetic E-B staggered system on a symmetric cylindrical mesh,
rebuilt from the GeoBrain
:class:`~geobrain.mesh.cylindrical.CylindricalMesh` geometry and
validated ENTRY-EXACT by the cross-validation suite:

- E_phi DOFs on the theta-edges: full circles at ``(r_node > 0, z_node)``;
  the symmetry axis carries no DOF (E_phi(0) = 0 exactly). Edge "length" is
  the circumference ``2 pi r``.
- B flux DOFs on the two staggered face families: radial faces
  ``(r_node > 0, z_center)`` (area ``2 pi r dz``) and vertical annuli
  ``(r_center, z_node)`` (area ``pi (r2^2 - r1^2)``).
- ``C``: the circulation/area curl (entries ``±2 pi r / A_f``).
- ``M_e(sigma) = diag( sum_adj sigma_c V_c / 4 )`` (quarter-ring volumes,
  boundary edges keep fewer terms, no renormalisation),
  ``M_f(1/mu) = diag( sum_adj V_c / (2 mu) )``: the mimetic inner
  products, pinned entry-exact by the parity test.
- System (E-formulation, ``e^{+i omega t}``):
  ``(C^T M_f C + i omega M_e) e = C^T M_f s_m − i omega s_e``,
  ``b = −(C e) / (i omega)``: s_m does NOT enter the b reconstruction
  (pinned to 1e−16 against the reference identity).

Assembly is numpy/scipy + splu. :func:`solve_fdem_cyl` is the plain
forward; :func:`solve_fdem_cyl_autograd` carries the
sigma-gradient through :class:`SparseLinearSolveWithSigmaGrad` with the
closed-form diagonal Jacobian ``dA[e,e]/dsigma_c = i omega V_c/4`` on the
stored quarter-volume support; the registered operator face is
:class:`geobrain.physics.em.frequency_domain.fdem_cyl.FDEMCyl`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from geobrain.core.errors import GeoBrainError
from geobrain.mesh.cylindrical import CylindricalMesh
from geobrain.physics.em.conventions import MU_0

__all__ = ["CylEBSystem", "build_cyl_eb_system", "solve_fdem_cyl",
           "solve_fdem_cyl_autograd", "loop_edge_source",
           "surface_bz_face_indices"]


@dataclass(frozen=True, eq=False)
class CylEBSystem:
    """Staggered axisymmetric system + the SoA that indexes it.

    Edge ordering: ``iz_node * nr + ir`` (``ir`` counts r-NODES 1..nr).
    Face ordering: all radial faces first (``iz_cell * nr + ir``), then the
    vertical annuli (``iz_node * nr + j``). Coordinates are ``(z, r)`` in
    the platform frame (z down).
    """

    edge_coords: np.ndarray      # (nE, 2) (z, r)
    edge_lengths: np.ndarray     # (nE,)   2 pi r
    face_coords: np.ndarray      # (nF, 2)
    face_areas: np.ndarray       # (nF,)
    n_radial_faces: int
    curl: sp.csr_matrix          # (nF, nE)
    me_sigma: np.ndarray         # (nE,)
    mf_mui: np.ndarray           # (nF,)
    # quarter-ring volume weights behind M_e: COO triplets with
    # me_sigma = index_add(me_rows, me_vals * sigma[me_cols]): the
    # closed-form dA[e,e]/dsigma_c = i*omega*me_vals support the
    # lu_bridge sigma-adjoint consumes.
    me_rows: np.ndarray          # (nnz,) edge ids
    me_cols: np.ndarray          # (nnz,) cell ids (flat (z, r) C-order)
    me_vals: np.ndarray          # (nnz,) V_c / 4


def build_cyl_eb_system(
    mesh: CylindricalMesh,
    sigma: torch.Tensor | np.ndarray,
    *,
    mu: float = MU_0,
) -> CylEBSystem:
    """Assemble the axisymmetric E_phi staggered system from ring geometry."""
    if not isinstance(mesh, CylindricalMesh):
        raise GeoBrainError(
            "build_cyl_eb_system requires a CylindricalMesh",
            object_name="build_cyl_eb_system", field="mesh",
            expected="CylindricalMesh", actual=type(mesh).__name__,
        )
    nz, nr = mesh.shape
    sig = (sigma.detach().cpu().numpy() if isinstance(sigma, torch.Tensor)
           else np.asarray(sigma, dtype=np.float64)).reshape(nz, nr)
    zn, rn = (t.numpy() for t in mesh.node_lines())
    zc, rc = (t.numpy() for t in mesh.center_lines())
    vol = mesh.cell_volumes().numpy().reshape(nz, nr)

    r_nodes = rn[1:]                       # (nr,), the DOF-carrying nodes
    two_pi_r = 2.0 * np.pi * r_nodes

    # ---- edges: (iz_node 0..nz) x (ir 0..nr-1) --------------------------
    izn, ire = np.meshgrid(np.arange(nz + 1), np.arange(nr), indexing="ij")
    izn, ire = izn.ravel(), ire.ravel()
    edge_coords = np.column_stack([zn[izn], r_nodes[ire]])
    edge_lengths = two_pi_r[ire]
    n_e = edge_coords.shape[0]

    def eidx(iz_node, ir):
        return iz_node * nr + ir

    # quarter-ring volume sums: adjacent z-cells {iz-1, iz}, rings {ir, ir+1}.
    # The (edge, cell, V/4) triplets are ALSO kept: they are the exact
    # sigma-Jacobian support of the M_e diagonal.
    me = np.zeros(n_e)
    me_rows_l, me_cols_l, me_vals_l = [], [], []
    for dz_cell in (-1, 0):
        zcell = izn + dz_cell
        for dring in (0, 1):
            ring = ire + dring
            ok = (zcell >= 0) & (zcell < nz) & (ring < nr)
            w = vol[zcell[ok], ring[ok]] / 4.0
            me[ok] += sig[zcell[ok], ring[ok]] * w
            me_rows_l.append(np.nonzero(ok)[0])
            me_cols_l.append(zcell[ok] * nr + ring[ok])
            me_vals_l.append(w)
    me_rows = np.concatenate(me_rows_l).astype(np.int64)
    me_cols = np.concatenate(me_cols_l).astype(np.int64)
    me_vals = np.concatenate(me_vals_l)

    # ---- radial faces: (iz_cell 0..nz-1) x (r-node ir) ------------------
    izc, irf = np.meshgrid(np.arange(nz), np.arange(nr), indexing="ij")
    izc, irf = izc.ravel(), irf.ravel()
    rad_coords = np.column_stack([zc[izc], r_nodes[irf]])
    wz = mesh.cell_widths[0].numpy()
    rad_areas = two_pi_r[irf] * wz[izc]
    n_rad = rad_coords.shape[0]

    mf_rad = np.zeros(n_rad)
    for dring in (0, 1):
        ring = irf + dring
        ok = ring < nr
        mf_rad[ok] += vol[izc[ok], ring[ok]] / 2.0

    # ---- vertical faces: (iz_node) x (ring j) ---------------------------
    izv, jv = np.meshgrid(np.arange(nz + 1), np.arange(nr), indexing="ij")
    izv, jv = izv.ravel(), jv.ravel()
    ver_coords = np.column_stack([zn[izv], rc[jv]])
    ring_area = np.pi * (rn[1:] ** 2 - rn[:-1] ** 2)
    ver_areas = ring_area[jv]
    n_ver = ver_coords.shape[0]

    mf_ver = np.zeros(n_ver)
    for dz_cell in (-1, 0):
        zcell = izv + dz_cell
        ok = (zcell >= 0) & (zcell < nz)
        mf_ver[ok] += vol[zcell[ok], jv[ok]] / 2.0

    # ---- curl: circulation / area ---------------------------------------
    rows, cols, vals = [], [], []
    # radial face (iz_cell, ir): d(E)/dz-down between its two z-node edges
    f_rad = np.arange(n_rad)
    rows += [f_rad, f_rad]
    cols += [eidx(izc + 1, irf), eidx(izc, irf)]
    vals += [two_pi_r[irf] / rad_areas, -two_pi_r[irf] / rad_areas]
    # vertical face (iz_node, j): outer-edge minus inner-edge circulation
    f_ver = n_rad + np.arange(n_ver)
    rows += [f_ver]
    cols += [eidx(izv, jv)]
    vals += [two_pi_r[jv] / ver_areas]
    inner = jv > 0
    rows += [f_ver[inner]]
    cols += [eidx(izv[inner], jv[inner] - 1)]
    vals += [-two_pi_r[jv[inner] - 1] / ver_areas[inner]]

    curl = sp.coo_matrix(
        (np.concatenate(vals),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_rad + n_ver, n_e),
    ).tocsr()

    return CylEBSystem(
        edge_coords=edge_coords,
        edge_lengths=edge_lengths,
        face_coords=np.concatenate([rad_coords, ver_coords]),
        face_areas=np.concatenate([rad_areas, ver_areas]),
        n_radial_faces=n_rad,
        curl=curl,
        me_sigma=me,
        mf_mui=np.concatenate([mf_rad, mf_ver]) / mu,
        me_rows=me_rows,
        me_cols=me_cols,
        me_vals=me_vals,
    )


def solve_fdem_cyl_autograd(
    system: CylEBSystem,
    omega: float,
    sigma: torch.Tensor,
    s_e: np.ndarray,
) -> torch.Tensor:
    """Differentiable ``e = A(σ)^{-1}(−iω s_e)`` via the splu σ-adjoint bridge.

    ``system`` must have been built from the SAME ``sigma`` values (the
    assembled ``A`` uses ``system.me_sigma``); gradients w.r.t. ``sigma``
    flow through the closed-form diagonal Jacobian
    ``dA[e,e]/dσ_c = iω · (V_c/4)`` on the stored quarter-volume support,
    the same :class:`SparseLinearSolveWithSigmaGrad` pattern as the Yee
    curl-curl family. The RHS is σ-independent (a stamped current loop),
    so the σ-path lives entirely in ``A``.
    """
    from geobrain.physics.em.numerics.sparse.lu_bridge import (
        SparseLinearSolveWithSigmaGrad,
    )

    c = system.curl
    mf = sp.diags(system.mf_mui)
    a_csr = (c.T @ mf @ c
             + 1j * omega * sp.diags(system.me_sigma)).tocsr()
    b = torch.as_tensor(
        -1j * omega * np.asarray(s_e, dtype=np.complex128))
    jac_vals = (1j * omega * system.me_vals).astype(np.complex128)
    return SparseLinearSolveWithSigmaGrad.apply(
        a_csr.data, a_csr.indices, a_csr.indptr, a_csr.shape,
        b, sigma.reshape(-1).to(torch.float64),
        system.me_rows, system.me_rows.copy(), system.me_cols, jac_vals,
    )


def solve_fdem_cyl(
    system: CylEBSystem,
    omega: float,
    *,
    s_m: np.ndarray | None = None,
    s_e: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``(C^T M_f C + iω M_e) e = C^T M_f s_m − iω s_e``; return (e, b).

    ``s_m`` is a face vector (magnetic source, e.g. a primary loop field),
    ``s_e`` an edge vector (galvanic/current stamping); either may be None.
    ``b = −(C e) / (iω)``: the E-formulation identity (s_m does not
    enter b; verified to 1e−16 against the reference fields).
    """
    c = system.curl
    mf = sp.diags(system.mf_mui)
    a_mat = (c.T @ mf @ c
             + 1j * omega * sp.diags(system.me_sigma)).tocsc()
    rhs = np.zeros(c.shape[1], dtype=np.complex128)
    if s_m is not None:
        rhs += c.T @ (system.mf_mui * s_m)
    if s_e is not None:
        rhs -= 1j * omega * np.asarray(s_e, dtype=np.complex128)
    e = spla.splu(a_mat).solve(rhs)
    b = -(c @ e) / (1j * omega)
    return e, b


def loop_edge_source(
    system: CylEBSystem, *, radius: float, z: float, current: float = 1.0,
) -> np.ndarray:
    """``s_e`` for a physical current loop lying on the nearest theta-edge.

    The loop is a delta current sheet on one edge circle: the weak-form
    source integral over that edge's basis is ``I · 2πr`` (current times
    loop length). Snaps to the nearest edge; keep loop radius/height on
    grid nodes for an exact representation.
    """
    d2 = ((system.edge_coords[:, 0] - z) ** 2
          + (system.edge_coords[:, 1] - radius) ** 2)
    k = int(np.argmin(d2))
    s_e = np.zeros(system.edge_coords.shape[0], dtype=np.complex128)
    s_e[k] = current * system.edge_lengths[k]
    return s_e


def surface_bz_face_indices(
    system: CylEBSystem, r_targets: np.ndarray, *, z: float = 0.0,
) -> np.ndarray:
    """Vertical-annulus face indices nearest ``(z, r_k)``: bz sample points."""
    ver = np.arange(system.n_radial_faces, system.face_coords.shape[0])
    coords = system.face_coords[ver]
    out = []
    for r_t in np.atleast_1d(r_targets):
        d2 = (coords[:, 0] - z) ** 2 + (coords[:, 1] - r_t) ** 2
        out.append(ver[int(np.argmin(d2))])
    return np.asarray(out, dtype=np.int64)
