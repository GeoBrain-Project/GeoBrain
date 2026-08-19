"""TEM3D numerics: Yee edge/face geometry diagonals, VMD source vector
potentials, and the Nedelec edge-FEM assets (split from tem3d.py).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations


from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from geobrain.physics.em.conventions import MU_0
from geobrain.core import GeoBrainError
from geobrain.mesh import TensorMesh
from geobrain.mesh.capabilities import (
    EdgeRecords,
)
from geobrain.physics.em.surveys import (
    MagneticDipoleSource,
)
from geobrain.physics.em.numerics.edge_element import (
    EdgeFemAssetsBase,
)

# Platform canonical vacuum permeability (core.constants, CODATA-2018),
# imported, never re-declared: the classical 4*pi*1e-7 sits ~5.4e-10
# relative away and would silently split the family's canon.
_MU0 = MU_0


def _safe_half_sum_uniform(d: float, idx: torch.Tensor, n: int) -> torch.Tensor:
    """
    One-sided half-sum for uniform spacing ``d`` of an ``n``-cell axis.

    For node index ``idx`` (range ``[0, n]``), returns

    - ``d`` (interior nodes ``1 <= idx <= n - 1``, both neighbours valid),
    - ``d / 2`` (boundary nodes ``idx == 0`` or ``idx == n``).

    Mirrors the :func:`_safe_half_sum` semantics on a uniform grid.
    """
    out = torch.zeros_like(idx, dtype=torch.float64)
    # Left half-cell exists when ``idx - 1`` is in [0, n-1] -> idx >= 1.
    mask_left = idx >= 1
    # Right half-cell exists when ``idx`` is in [0, n-1] -> idx <= n - 1.
    mask_right = idx <= n - 1
    out = out + mask_left.to(torch.float64) * (d / 2.0)
    out = out + mask_right.to(torch.float64) * (d / 2.0)
    return out


def _build_edge_volume_diagonal(mesh: TensorMesh) -> torch.Tensor:
    """
    Per-edge dual-cell volume for the mimetic edge inner product.

    For a uniform 3D Yee mesh with cell width ``(dx, dy, dz)``:

    - x-edge at ``(i, j, k)`` (``i ∈ [0, nx)``, ``j ∈ [0, ny]``, ``k ∈
      [0, nz]``): ``V = dx · half_dy(j) · half_dz(k)``.
    - y-edge at ``(i, j, k)`` (``i ∈ [0, nx]``, ``j ∈ [0, ny)``, ``k ∈
      [0, nz]``): ``V = half_dx(i) · dy · half_dz(k)``.
    - z-edge at ``(i, j, k)`` (``i ∈ [0, nx]``, ``j ∈ [0, ny]``, ``k ∈
      [0, nz)``): ``V = half_dx(i) · half_dy(j) · dz``.

    where ``half_dα(idx) = dα`` at interior nodes and ``dα / 2`` at
    boundary nodes (a one-sided fall-back consistent with
    :func:`_safe_half_sum`).

    Edge ordering matches :func:`yee_curl_e_to_f` (flat index
    ``i + j*nx + k*nx*(ny+1)`` for x-edges, etc.), on the uniform-spacing
    TensorMesh.
    """
    nz, ny, nx = mesh.shape
    dz, dy, dx = mesh.spacing
    dx_f = float(dx)
    dy_f = float(dy)
    dz_f = float(dz)

    def _flat_indices(ni: int, nj: int, nk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return (II, JJ, KK) flattened in the curl-curl edge order.

        Flat ordering is ``flat = i + j*ni + k*ni*nj`` (i-fastest, then j,
        then k), matching :func:`yee_curl_e_to_f`'s
        ``ex_idx = i + j*nx + k*nx*(ny+1)`` etc.
        """
        II = torch.arange(ni, dtype=torch.long).view(ni, 1, 1).expand(ni, nj, nk)
        JJ = torch.arange(nj, dtype=torch.long).view(1, nj, 1).expand(ni, nj, nk)
        KK = torch.arange(nk, dtype=torch.long).view(1, 1, nk).expand(ni, nj, nk)
        # Permute to (k, j, i) so reshape(-1) gives i-fastest, then j,
        # then k, matching the index expressions above.
        return (
            II.permute(2, 1, 0).reshape(-1),
            JJ.permute(2, 1, 0).reshape(-1),
            KK.permute(2, 1, 0).reshape(-1),
        )

    II, JJ, KK = _flat_indices(nx, ny + 1, nz + 1)
    V_ex = dx_f * _safe_half_sum_uniform(dy_f, JJ, ny) * _safe_half_sum_uniform(dz_f, KK, nz)

    II, JJ, KK = _flat_indices(nx + 1, ny, nz + 1)
    V_ey = _safe_half_sum_uniform(dx_f, II, nx) * dy_f * _safe_half_sum_uniform(dz_f, KK, nz)

    II, JJ, KK = _flat_indices(nx + 1, ny + 1, nz)
    V_ez = _safe_half_sum_uniform(dx_f, II, nx) * _safe_half_sum_uniform(dy_f, JJ, ny) * dz_f

    return torch.cat([V_ex, V_ey, V_ez]).to(torch.float64)


def _build_face_dual_volume(mesh: TensorMesh) -> torch.Tensor:
    """
    Per-face dual volume for the mimetic face inner product.

    For a uniform Yee mesh, the face dual volume is

    - Fx face at ``(i, j, k)`` (``i ∈ [0, nx]``, ``j ∈ [0, ny)``, ``k ∈
      [0, nz)``): ``V = half_dx(i) · dy · dz``.
    - Fy face at ``(i, j, k)``: ``V = dx · half_dy(j) · dz``.
    - Fz face at ``(i, j, k)``: ``V = dx · dy · half_dz(k)``.

    where ``half_dα`` is the standard one-sided half-cell at the
    boundary (``dα / 2`` at the wall nodes, ``dα`` inside). This makes
    the mimetic face inner product ``M_f(1/μ) = diag(V_face / μ)``
    consistent with the Yee curl operator (whose entries are already
    ``±1/Δ``: the metric is baked into ``C``, not ``M_f``).

    Used by the (revised) TEM3D forward to build a volume-weighted
    curl-curl operator ``K = C^T diag(V_face/μ) C`` and a proper edge
    mass matrix ``M_eσ = diag(σ_edge · V_edge)``.
    """
    nz, ny, nx = mesh.shape
    dz, dy, dx = mesh.spacing
    dx_f = float(dx)
    dy_f = float(dy)
    dz_f = float(dz)

    # Fx faces: i ∈ [0, nx], j ∈ [0, ny), k ∈ [0, nz)
    ii = torch.arange(nx + 1, dtype=torch.long)
    iiKK = ii.view(nx + 1, 1, 1).expand(nx + 1, ny, nz)
    # Flatten to the face order: i + j*(nx+1) + k*(nx+1)*ny
    II_x = iiKK.permute(2, 1, 0).reshape(-1)
    V_fx = _safe_half_sum_uniform(dx_f, II_x, nx) * dy_f * dz_f

    # Fy faces: i ∈ [0, nx), j ∈ [0, ny], k ∈ [0, nz)
    jj = torch.arange(ny + 1, dtype=torch.long)
    JJ_y = jj.view(1, ny + 1, 1).expand(nx, ny + 1, nz)
    JJ_y_flat = JJ_y.permute(2, 1, 0).reshape(-1)
    V_fy = dx_f * _safe_half_sum_uniform(dy_f, JJ_y_flat, ny) * dz_f

    # Fz faces: i ∈ [0, nx), j ∈ [0, ny), k ∈ [0, nz]
    kk = torch.arange(nz + 1, dtype=torch.long)
    KK_z = kk.view(1, 1, nz + 1).expand(nx, ny, nz + 1)
    KK_z_flat = KK_z.permute(2, 1, 0).reshape(-1)
    V_fz = dx_f * dy_f * _safe_half_sum_uniform(dz_f, KK_z_flat, nz)

    return torch.cat([V_fx, V_fy, V_fz]).to(torch.float64)


def _edge_positions(mesh: TensorMesh) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Yee edge centre positions, one ``(N, 3)`` tensor per family.

    Used by the revised TEM3D forward to evaluate the analytic free-space
    VMD vector potential ``A_p(r)`` at every Yee edge centre.
    """
    nodes_z, nodes_y, nodes_x = mesh.node_lines()
    centers_z, centers_y, centers_x = mesh.center_lines()
    nz = int(centers_z.numel())
    ny = int(centers_y.numel())
    nx = int(centers_x.numel())

    # Ex edges: x = (i + 0.5)*dx (cell-centre x), y = j*dy (node), z = k*dz (node).
    # Flat ordering: i fastest, then j, then k.
    X = centers_x.view(nx, 1, 1).expand(nx, ny + 1, nz + 1)
    Y = nodes_y.view(1, ny + 1, 1).expand(nx, ny + 1, nz + 1)
    Z = nodes_z.view(1, 1, nz + 1).expand(nx, ny + 1, nz + 1)
    ex_pos = torch.stack(
        [
            X.permute(2, 1, 0).reshape(-1),
            Y.permute(2, 1, 0).reshape(-1),
            Z.permute(2, 1, 0).reshape(-1),
        ],
        dim=-1,
    )

    # Ey edges: x = i*dx, y = (j + 0.5)*dy, z = k*dz
    X = nodes_x.view(nx + 1, 1, 1).expand(nx + 1, ny, nz + 1)
    Y = centers_y.view(1, ny, 1).expand(nx + 1, ny, nz + 1)
    Z = nodes_z.view(1, 1, nz + 1).expand(nx + 1, ny, nz + 1)
    ey_pos = torch.stack(
        [
            X.permute(2, 1, 0).reshape(-1),
            Y.permute(2, 1, 0).reshape(-1),
            Z.permute(2, 1, 0).reshape(-1),
        ],
        dim=-1,
    )

    # Ez edges: x = i*dx, y = j*dy, z = (k + 0.5)*dz
    X = nodes_x.view(nx + 1, 1, 1).expand(nx + 1, ny + 1, nz)
    Y = nodes_y.view(1, ny + 1, 1).expand(nx + 1, ny + 1, nz)
    Z = centers_z.view(1, 1, nz).expand(nx + 1, ny + 1, nz)
    ez_pos = torch.stack(
        [
            X.permute(2, 1, 0).reshape(-1),
            Y.permute(2, 1, 0).reshape(-1),
            Z.permute(2, 1, 0).reshape(-1),
        ],
        dim=-1,
    )

    return ex_pos, ey_pos, ez_pos


def _build_vmd_vector_potential_on_edges(
    mesh: TensorMesh,
    source_pos: tuple[float, float, float],
    source_orientation: tuple[float, float, float],
    source_moment: float,
    *,
    eps_singular: float = 1.0e-3,
) -> torch.Tensor:
    """
    Analytic free-space VMD vector potential projected onto Yee edge tangents.

    For a magnetic dipole at ``source_pos`` with moment vector
    ``m = moment * orientation`` (with ``orientation`` a unit 3-vector),
    the vector potential in free space is

    .. math::

        \\vec{A}(\\vec{r}) = \\frac{\\mu_0}{4\\pi}\\,
            \\frac{\\vec{m} \\times (\\vec{r} - \\vec{r}_s)}{|\\vec{r}-\\vec{r}_s|^3}.

    This helper evaluates ``A · t_edge`` (with ``t_edge`` the unit
    tangent) at every Yee edge centre, with a small-radius safeguard
    that clamps ``|r - r_s|`` to ``eps_singular`` metres (default 1 m)
    so the source-cell edge values stay finite. Returns a flat tensor
    of length ``n_edges`` ordered ``[Ex; Ey; Ez]``.

    Used to build the discrete primary B-field
    ``b_p = C @ a_e`` (Tesla, on faces) for the step-off source impulse
    in ``TEM3D._forward``.
    """
    ex_pos, ey_pos, ez_pos = _edge_positions(mesh)
    src = torch.tensor(source_pos, dtype=torch.float64)
    m_hat = torch.tensor(source_orientation, dtype=torch.float64)
    m_norm = float(torch.linalg.norm(m_hat))
    if m_norm <= 0.0:
        raise GeoBrainError(
            "build_vmd_vector_potential_on_edges: orientation must be nonzero",
            object_name="_build_vmd_vector_potential_on_edges",
            field="source_orientation",
            expected="nonzero norm",
            actual=m_norm,
        )
    m_vec = (m_hat / m_norm) * float(source_moment)
    prefac = _MU0 / (4.0 * math.pi)

    def _project_a(pos: torch.Tensor, tangent_idx: int) -> torch.Tensor:
        r = pos - src.unsqueeze(0)
        r_norm = torch.linalg.norm(r, dim=1)
        r_norm_safe = torch.clamp(r_norm, min=eps_singular)
        # a = (μ_0 / 4π) (m × r) / |r|^3
        # cross product m × r:
        m_x, m_y, m_z = m_vec[0], m_vec[1], m_vec[2]
        rx, ry, rz = r[:, 0], r[:, 1], r[:, 2]
        if tangent_idx == 0:
            a_t = m_y * rz - m_z * ry
        elif tangent_idx == 1:
            a_t = m_z * rx - m_x * rz
        else:
            a_t = m_x * ry - m_y * rx
        return prefac * a_t / r_norm_safe**3

    a_ex = _project_a(ex_pos, 0)
    a_ey = _project_a(ey_pos, 1)
    a_ez = _project_a(ez_pos, 2)
    return torch.cat([a_ex, a_ey, a_ez]).to(torch.float64)


@dataclass(frozen=True)
class _TemEdgeFemAssets(EdgeFemAssetsBase):
    """σ-independent Whitney assets of one mesh (see ``_edge_fem_assets``).

    Attributes:
        plan: the mesh's :class:`EdgeAssemblyPlan` (geometric factors + COO
            index pattern), reused across every BDF substep.
        boundary_mask: ``(n_edges,)`` bool, outer-boundary edges (PEC pins).
        edge_records: the mesh's :class:`EdgeRecords` (canonical node pairs,
            tangents, lengths, midpoints) for the vector-potential projection.
        cell_edge_ids: ``(n_cells, 6)`` long, global edge ids per cell.
        cell_edge_signs: ``(n_cells, 6)`` float64, local→canonical signs.
        w_centroid: ``(n_cells, 6, 3)`` float64, Whitney basis vectors at
            the cell barycentre, ``W_e(centroid) = (g_j - g_i)/4``.
        curl_w: ``(n_cells, 6, 3)`` float64, constant per-cell curls,
            ``curl W_e = 2 g_i × g_j``.
        cell_centers: ``(n_cells, 3)`` float64, receiver-binding points.
        dts: ``(n_steps,)`` float64, the shared log-substep BDF schedule
            (:func:`build_log_substep_schedule` with the Yee ``dt_max`` cap).
        gate_step_idx: ``(n_gates,)`` int64, gate→step indices; the E
            history index of gate ``g`` is ``gate_step_idx[g] + 1``.
    """

    dts: torch.Tensor
    gate_step_idx: torch.Tensor


def _vmd_vector_potential_at_points(
    points: torch.Tensor,
    source_pos: tuple[float, float, float],
    source_orientation: tuple[float, float, float],
    source_moment: float,
    *,
    eps_singular: float = 1.0e-3,
) -> torch.Tensor:
    """Analytic free-space VMD vector potential at arbitrary points.

    The SAME closed form as :func:`_build_vmd_vector_potential_on_edges`;
    which is this expression sampled at Yee edge midpoints and projected onto
    the per-family tangents, evaluated at an arbitrary ``(n, 3)`` point set
    and returned as full ``(n, 3)`` vectors:

    .. math::

        \\vec{A}(\\vec{r}) = \\frac{\\mu_0}{4\\pi}\\,
            \\frac{\\vec{m} \\times (\\vec{r} - \\vec{r}_s)}{|\\vec{r}-\\vec{r}_s|^3},

    with the same ``|r - r_s| ≥ eps_singular`` clamp keeping source-adjacent
    values finite. The edge-element branch projects these onto Whitney edge
    dofs (:func:`_edge_vmd_vector_potential_dofs`); the cross-implementation
    consistency test pins this function against the Yee sampler on a shared
    edge-midpoint set.

    Args:
        points: ``(n, 3)`` evaluation coordinates in metres.
        source_pos: dipole position ``(x, y, z)`` in metres.
        source_orientation: dipole moment direction (normalised internally).
        source_moment: dipole moment magnitude (A·m²).
        eps_singular: lower clamp on ``|r - r_s|`` in metres (Yee default).

    Returns:
        ``(n, 3)`` float64 vector-potential vectors (V·s/m).
    """
    pts = points.to(torch.float64)
    if pts.dim() != 2 or int(pts.shape[1]) != 3:
        raise GeoBrainError(
            "vmd_vector_potential_at_points expects points of shape (n, 3), "
            f"got {tuple(pts.shape)}",
            object_name="_vmd_vector_potential_at_points",
            field="points",
            expected="shape (n, 3)",
            actual=tuple(pts.shape),
        )
    src = torch.tensor(source_pos, dtype=torch.float64)
    m_hat = torch.tensor(source_orientation, dtype=torch.float64)
    m_norm = float(torch.linalg.norm(m_hat))
    if m_norm <= 0.0:
        raise GeoBrainError(
            "vmd_vector_potential_at_points: orientation must be nonzero",
            object_name="_vmd_vector_potential_at_points",
            field="source_orientation",
            expected="nonzero norm",
            actual=m_norm,
        )
    m_vec = (m_hat / m_norm) * float(source_moment)
    r = pts - src.unsqueeze(0)
    r_norm_safe = torch.clamp(torch.linalg.norm(r, dim=1), min=eps_singular)
    cross = torch.linalg.cross(m_vec.expand_as(r), r, dim=-1)
    prefac = _MU0 / (4.0 * math.pi)
    return prefac * cross / r_norm_safe.unsqueeze(-1) ** 3


def _edge_vmd_vector_potential_dofs(
    records: EdgeRecords,
    source: MagneticDipoleSource,
) -> torch.Tensor:
    """Whitney edge dofs of the analytic free-space VMD vector potential.

    The lowest-order Nédélec dof is the tangential circulation
    ``dof_e = ∫_e A·dl`` along the edge's canonical direction; the midpoint
    rule gives ``dof_e ≈ A(midpoint_e)·tangent_e·length_e``, the same
    quadrature the FDEM3D edge branch uses for its primary-field projection.
    With ``a_e`` in the Whitney edge space the discrete primary flux is
    ``b_p = curl_w a_e`` and the step-off impulse of the first BDF step is
    ``q_1 = (1/Δt_0)·K_w(1/μ₀)·a_e`` (the Yee ``q_1 = (1/dt_0)·K·a_e``).

    Args:
        records: the mesh's :class:`EdgeRecords` (midpoints, unit tangents,
            lengths along the canonical small-node → large-node direction).
        source: the :class:`MagneticDipoleSource`.

    Returns:
        ``(n_edges,)`` float64 vector-potential edge circulations (V·s).
    """
    A_mid = _vmd_vector_potential_at_points(
        records.midpoint[:, (1, 2, 0)],
        source.position,
        source.orientation,
        source.magnetic_moment_am2,
    )  # (ne, 3)
    tangent_xyz = records.tangent[:, (1, 2, 0)]
    return (A_mid * tangent_xyz).sum(dim=1) * records.length

