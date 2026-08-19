"""MT3D numerics: plane-wave primary, secondary curl-curl solves
(Yee + Nedelec edge paths), station extraction and impedance assembly
(split from mt3d.py).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import torch
from geobrain.core import GeoBrainError
from geobrain.mesh import TensorMesh
from geobrain.mesh.capabilities import (
    EdgeRecords,
)
from geobrain.physics.em.conventions import MU_0
from geobrain.physics.em.numerics.edge_element import (
    EdgeFemAssetsBase,
    EdgeAssemblyPlan,
    build_pec_pin_plan,
    edge_operator_matvec,
    assemble_pec_pinned_operator,
    pec_zero_rhs,
    solve_pec_pinned_edge_system,
)
from geobrain.physics.em.numerics.finite_volume.curl_curl import (
    _cell_to_edge_avg,
    yee_curl_curl_assemble,
    yee_dual_volume_weights,
    yee_edge_count,
)
from geobrain.physics.em.numerics.finite_volume.receiver_sampling import (
    apply_yee_receiver_projection,
)
from geobrain.physics.em.receivers import (
    YeeReceiverProjection,
    build_yee_receiver_projection,
)
from geobrain.core.adjoint import sparse_linear_solve_with_adjoint


def _plane_wave_primary_field(
    mesh: TensorMesh,
    *,
    sigma_bg_1d: torch.Tensor,
    layer_thicknesses: torch.Tensor,
    omega: float,
    polarization: str,
) -> torch.Tensor:
    """
    Analytic 1D-layered plane-wave primary E on Yee edges.

    Mirrors the ``_primary_e_field_halfspace`` + ``_build_primary_edge_field``
    pair, collapsed into a single helper.

    For a layered halfspace the downgoing plane wave has a piecewise
    representation: in layer ``n`` (between depths ``z_top_n`` and
    ``z_top_n + h_n``) we have

        E(z) = E_top_n · exp(-k_n · (z - z_top_n)),
        k_n  = sqrt(i · omega · mu0 · sigma_n).

    The surface field is normalised to ``E(0) = 1 + 0j``. At each
    layer interface we use field continuity: ``E_top_{n+1} = E(z_top_{n+1})``
    from the previous-layer formula. The bottom (n_layer-1) layer is
    treated as a half-space; its thickness is ignored (matching
    :func:`mt1d_wait_impedance`).

    The resulting depth-profile ``E_1d(k_z)`` is sampled at every Yee
    z-node ``z_k = k · dz`` (k = 0, ..., nz), broadcast across (x, y), and
    placed on the ``Ex`` block (x-polarisation) or ``Ey`` block
    (y-polarisation) of the concatenated edge vector. The orthogonal
    edge families are zero.

    Args:
        mesh: 3D :class:`TensorMesh`. With ``mesh.shape == (nz, ny, nx)`` per
            the Yee curl-curl assembler convention.
        sigma_bg_1d: ``(n_layer,)`` real-valued layer conductivities in S/m, top->bottom.
        layer_thicknesses: ``(n_layer - 1,)`` real-valued layer thicknesses in m. Empty
            for a single half-space.
        omega: Angular frequency in rad/s.
        polarization: ``"x"`` or ``"y"``.

    Returns:
        Complex128 tensor of shape ``(n_edges,) = (n_Ex + n_Ey + n_Ez,)``
        in the same family ordering as :func:`yee_curl_curl_assemble`.
    """
    if polarization not in ("x", "y"):
        raise GeoBrainError(
            f"polarization must be 'x' or 'y'; got {polarization!r}",
            object_name="_plane_wave_primary_field",
            field="polarization",
            expected="'x' or 'y'",
            actual=polarization,
        )
    if sigma_bg_1d.ndim != 1:
        raise GeoBrainError(
            f"sigma_bg_1d must be 1D, got shape {tuple(sigma_bg_1d.shape)}",
            object_name="_plane_wave_primary_field",
            field="sigma_bg_1d",
            expected="1D tensor",
            actual=tuple(sigma_bg_1d.shape),
        )
    if layer_thicknesses.ndim != 1:
        raise GeoBrainError(
            f"layer_thicknesses must be 1D, got shape {tuple(layer_thicknesses.shape)}",
            object_name="_plane_wave_primary_field",
            field="layer_thicknesses",
            expected="1D tensor",
            actual=tuple(layer_thicknesses.shape),
        )
    n_layer = int(sigma_bg_1d.numel())
    if layer_thicknesses.numel() != max(0, n_layer - 1):
        raise GeoBrainError(
            "layer_thicknesses must have n_layer-1 entries "
            f"(got n_layer={n_layer}, "
            f"thicknesses={int(layer_thicknesses.numel())})",
            object_name="_plane_wave_primary_field",
            field="layer_thicknesses",
            expected=(max(0, n_layer - 1),),
            actual=int(layer_thicknesses.numel()),
        )

    if mesh.n_dim != 3:
        raise GeoBrainError(
            f"_plane_wave_primary_field requires a 3D mesh, got n_dim={mesh.n_dim}",
            object_name="_plane_wave_primary_field",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    # curl_curl.py reads ``nz, ny, nx = mesh.shape``; mirror that here so
    # the broadcast layout matches the edge-family flat indexing.
    nz, ny, nx = mesh.shape
    # MT3D-NU: use per-cell z-widths so z-nodes sit at the true cumulative
    # depths on a padded mesh. On a uniform mesh ``mesh.cell_widths[0]`` is
    # a constant-dz tensor and the per-cell path is bit-identical to the
    # legacy ``mesh.spacing[0]`` path.
    wz = mesh.cell_widths[0].to(dtype=torch.float64, device=sigma_bg_1d.device)

    device = sigma_bg_1d.device
    omega_c = torch.tensor(omega, dtype=torch.complex128, device=device)
    sigma_c = sigma_bg_1d.to(torch.complex128)

    # Per-layer wavenumber k_n = sqrt(i · omega · mu0 · sigma_n).
    k_layer = torch.sqrt(1j * omega_c * MU_0 * sigma_c)  # (n_layer,)

    # Build E_1d[k_z] at every Yee z-node (nz + 1 entries), with surface
    # normalised to 1+0j. Walk through layers in order; within each layer
    # advance node-by-node by exp(-k_n · dz). At interfaces, ``E_top`` of
    # the next layer is just the field at the last z-node we wrote.
    #
    # Layer boundaries are at cumulative depths z_intf_m (m = 1, ..., n_layer-1).
    # For each z-node z_k = k · dz we find the layer index and the depth
    # ``z - z_top_of_layer``.
    if n_layer == 1:
        layer_bottoms = torch.tensor([float("inf")], dtype=torch.float64, device=device)
    else:
        # Depth of the bottom of each layer (positive, increasing).
        cum = torch.cumsum(layer_thicknesses.to(torch.float64), dim=0)
        layer_bottoms = torch.cat(
            [cum, torch.tensor([float("inf")], dtype=torch.float64, device=device)]
        )

    # Cumulative z-node depths from per-cell widths. Node ``k`` sits at
    # depth ``cumsum(wz)[k-1]`` for k>=1; node 0 sits at the surface.
    z_nodes = torch.cat(
        [
            torch.zeros(1, dtype=torch.float64, device=device),
            torch.cumsum(wz, dim=0),
        ]
    )  # (nz + 1,)

    # Build the depth profile by concatenation (no in-place writes: keeps
    # autograd through sigma_bg_1d).
    E_list: list[torch.Tensor] = [torch.tensor(1.0 + 0.0j, dtype=torch.complex128, device=device)]
    current_layer = 0
    for k_z in range(nz):
        # Advance from z_node[k_z] to z_node[k_z + 1], possibly crossing
        # one layer interface. Field continuity at the interface means we
        # can split the step into two exponentials; for layer changes
        # between adjacent z-nodes this stays exact.
        z_lo = float(z_nodes[k_z].item())
        z_hi = float(z_nodes[k_z + 1].item())
        E_curr = E_list[-1]
        # Walk z_lo -> z_hi in (at most a few) pieces, one per layer crossed.
        while True:
            # Bottom of the active layer (depth, positive downward).
            bot = float(layer_bottoms[current_layer].item())
            if z_hi <= bot or current_layer == n_layer - 1:
                step = torch.tensor(z_hi - z_lo, dtype=torch.complex128, device=device)
                E_curr = E_curr * torch.exp(-k_layer[current_layer] * step)
                break
            # Cross the interface at depth ``bot``.
            step = torch.tensor(bot - z_lo, dtype=torch.complex128, device=device)
            E_curr = E_curr * torch.exp(-k_layer[current_layer] * step)
            z_lo = bot
            current_layer += 1
        E_list.append(E_curr)
    E_1d = torch.stack(E_list)  # (nz+1,) complex128

    # Zero blocks for the orthogonal edge families.
    n_ex, n_ey, n_ez = yee_edge_count(mesh)
    zeros_ex = torch.zeros(n_ex, dtype=torch.complex128, device=device)
    zeros_ey = torch.zeros(n_ey, dtype=torch.complex128, device=device)
    zeros_ez = torch.zeros(n_ez, dtype=torch.complex128, device=device)

    if polarization == "x":
        # Ex layout (k_z, j_y, i_x): index = i + j*nx + k*nx*(ny+1).
        # Field is constant in (x, y), depending only on k_z.
        block_x = E_1d.reshape(nz + 1, 1, 1).expand(nz + 1, ny + 1, nx).reshape(-1).contiguous()
        return torch.cat([block_x, zeros_ey, zeros_ez])

    # polarization == "y"
    # Ey layout (k_z, j_y, i_x): index = n_Ex + i + j*(nx+1) + k*(nx+1)*ny.
    block_y = E_1d.reshape(nz + 1, 1, 1).expand(nz + 1, ny, nx + 1).reshape(-1).contiguous()
    return torch.cat([zeros_ex, block_y, zeros_ez])


def _secondary_field_solve(
    mesh: TensorMesh,
    *,
    sigma_3d: torch.Tensor,
    sigma_bg_1d: torch.Tensor,
    omega: float,
    E_primary: torch.Tensor,
    layer_thicknesses: torch.Tensor | None = None,
    return_system: bool = False,
) -> torch.Tensor:
    """
    Solve A(σ) E_s = -iω·μ₀·(σ_edge - σ_bg_edge)·E_p for the secondary field.

    Uses the :func:`yee_curl_curl_assemble` (via its
    :meth:`_AutogradCsrComplex.to_sparse_coo` bridge) + E5's
    :func:`sparse_linear_solve_with_adjoint`. Returns the secondary E-field
    on the same Yee edge ordering as ``E_primary``.

    This is the **Option B** formulation: the RHS is built from ``sigma_3d``
    via pure torch ops (so autograd flows through the RHS) while ``A`` is
    factorised by scipy splu inside the bridge (so autograd flows through
    ``A.values()`` via the IMPLICIT_VJP adjoint solve). The two
    sigma-paths sum automatically in autograd.

    Args:
        mesh: 3D :class:`TensorMesh` with ``mesh.shape == (nz, ny, nx)`` per the
            Yee curl-curl assembler convention.
        sigma_3d: Cell-centred 3D conductivity in S/m, real ``float64``, shape
            ``mesh.shape``.
        sigma_bg_1d: ``(n_layer,)`` real-valued 1D background layer conductivities (top
            to bottom). Broadcast across the 3D mesh to form ``sigma_bg_3d``;
            the layer assignment uses ``layer_thicknesses`` (or treats the
            whole domain as a single half-space if ``layer_thicknesses`` is
            ``None`` / empty).
        omega: Angular frequency in rad/s.
        E_primary: ``(n_edges,)`` complex128 primary E-field on Yee edges, e.g. from
            :func:`_plane_wave_primary_field`.
        layer_thicknesses: ``(n_layer - 1,)`` real-valued layer thicknesses in m. ``None`` or
            empty for a single half-space.

    Returns:
        Complex128 tensor of shape ``(n_edges,)``, the secondary E-field.
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            f"_secondary_field_solve requires a 3D mesh, got n_dim={mesh.n_dim}",
            object_name="_secondary_field_solve",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    nz, ny, nx = mesh.shape
    if sigma_3d.shape != mesh.shape:
        raise GeoBrainError(
            "sigma_3d.shape must match mesh.shape; got "
            f"{tuple(sigma_3d.shape)} vs {tuple(mesh.shape)}",
            object_name="_secondary_field_solve",
            field="sigma_3d",
            expected=tuple(mesh.shape),
            actual=tuple(sigma_3d.shape),
        )
    if sigma_bg_1d.ndim != 1:
        raise GeoBrainError(
            f"sigma_bg_1d must be 1D, got shape {tuple(sigma_bg_1d.shape)}",
            object_name="_secondary_field_solve",
            field="sigma_bg_1d",
            expected="1D tensor",
            actual=tuple(sigma_bg_1d.shape),
        )

    device = sigma_3d.device
    # The analytic plane-wave primary is built (and cached) from the CPU
    # background-σ profile; move it onto the compute device so the secondary
    # RHS q = -iωμ₀·Δσ·E_primary and the E_total = E_primary + E_secondary sum
    # stay device-consistent when σ lives on the GPU.
    E_primary = E_primary.to(device)
    # MT3D-NU: use per-cell z-widths so z-cell centres sit at the true
    # cumulative depths on a padded mesh. On a uniform mesh this is
    # bit-identical to the legacy ``mesh.spacing[0]`` path.
    wz = mesh.cell_widths[0].to(dtype=torch.float64, device=device)

    # ------------------------------------------------------------------
    # 1) Build sigma_bg_3d by broadcasting sigma_bg_1d across layers.
    #    Half-space: fill entire mesh with sigma_bg_1d[0].
    #    Layered:   assign each z-cell to a layer based on cumulative
    #               thicknesses; cells above the last interface get the
    #               corresponding layer's σ, cells below all get the
    #               last (bottom) layer's σ.
    # ------------------------------------------------------------------
    n_layer = int(sigma_bg_1d.numel())
    if layer_thicknesses is None or layer_thicknesses.numel() == 0:
        # Half-space
        if n_layer != 1:
            raise GeoBrainError(
                "layer_thicknesses is empty/None but sigma_bg_1d has "
                f"{n_layer} layers; expected exactly 1 for half-space",
                object_name="_secondary_field_solve",
                field="sigma_bg_1d",
                expected="1 layer for half-space",
                actual=n_layer,
            )
        sigma_bg_3d = sigma_bg_1d[0] * torch.ones(
            mesh.shape,
            dtype=sigma_3d.dtype,
            device=device,
        )
    else:
        if layer_thicknesses.numel() != n_layer - 1:
            raise GeoBrainError(
                "layer_thicknesses must have n_layer-1 entries "
                f"(got n_layer={n_layer}, "
                f"thicknesses={int(layer_thicknesses.numel())})",
                object_name="_secondary_field_solve",
                field="layer_thicknesses",
                expected=(n_layer - 1,),
                actual=int(layer_thicknesses.numel()),
            )
        # Bottom depth of each layer; last is +inf (half-space).
        cum = torch.cumsum(layer_thicknesses.to(torch.float64), dim=0).to(device)
        layer_bottoms = torch.cat(
            [cum, torch.tensor([float("inf")], dtype=torch.float64, device=device)]
        )
        # z-cell centre depth from per-cell widths: cumsum(wz) - 0.5*wz.
        z_centres = torch.cumsum(wz, dim=0) - 0.5 * wz  # (nz,)
        # For each z-cell pick the smallest layer index whose bottom
        # depth >= z_centre.
        layer_per_z = torch.searchsorted(layer_bottoms, z_centres)
        layer_per_z = torch.clamp(layer_per_z, max=n_layer - 1)
        sigma_per_z = sigma_bg_1d[layer_per_z].to(sigma_3d.dtype)  # (nz,)
        sigma_bg_3d = sigma_per_z.reshape(nz, 1, 1).expand(nz, ny, nx).contiguous()

    # ------------------------------------------------------------------
    # 2) Cell -> edge averaging for sigma_3d and sigma_bg_3d. Reuses
    #    curl_curl._cell_to_edge_avg so the averaging matches the iωμ₀σ
    #    diagonal term assembled inside A. MESH3: pass mesh.cell_widths
    #    so the average is volume-weighted on non-uniform meshes (and
    #    bit-identical to arithmetic on uniform meshes).
    # ------------------------------------------------------------------
    mesh_widths = mesh.cell_widths
    sigma_edge = torch.cat(
        [
            _cell_to_edge_avg(sigma_3d, "x", widths=mesh_widths),
            _cell_to_edge_avg(sigma_3d, "y", widths=mesh_widths),
            _cell_to_edge_avg(sigma_3d, "z", widths=mesh_widths),
        ],
        dim=0,
    )
    sigma_bg_edge = torch.cat(
        [
            _cell_to_edge_avg(sigma_bg_3d, "x", widths=mesh_widths),
            _cell_to_edge_avg(sigma_bg_3d, "y", widths=mesh_widths),
            _cell_to_edge_avg(sigma_bg_3d, "z", widths=mesh_widths),
        ],
        dim=0,
    )

    # ------------------------------------------------------------------
    # 3) RHS: q = -i·ω·μ₀·(σ_edge - σ_bg_edge) · E_primary.
    # Real (σ - σ_bg) times complex E_p, scaled by -iω·μ₀. Sign matches
    # the curl-curl diagonal +i·ω·μ₀·σ_edge term: moving the background
    # piece to the RHS introduces the opposite sign.
    # ------------------------------------------------------------------
    delta_sigma_edge = (sigma_edge - sigma_bg_edge).to(torch.complex128)
    scale = (-1j * omega * MU_0) * delta_sigma_edge  # (n_edges,)
    # E1: the RHS is the edge-mass action −iωμ₀·M_e(Δσ)·E_p, so it carries
    # the SAME edge dual volume as the +iωμ₀·M_e(σ) diagonal that
    # ``yee_curl_curl_assemble`` stamps into A. Without this the secondary
    # source and the operator would be weighted inconsistently on a graded
    # mesh. ``edge_w`` is None on a uniform mesh ⇒ byte-identical RHS.
    _, edge_w = yee_dual_volume_weights(
        mesh,
        device=device,
        dtype=torch.float64,
    )
    if edge_w is not None:
        scale = scale * edge_w
    # ``E_primary`` may be ``(n_edges,)`` for one polarisation, or
    # ``(n_edges, k)`` to solve several primaries against the SAME A in a
    # single factorisation: broadcast ``scale`` over the RHS columns.
    q = scale.unsqueeze(-1) * E_primary if E_primary.ndim == 2 else scale * E_primary

    # ------------------------------------------------------------------
    # 4) Assemble A(σ) and solve A·E_s = q via the splu bridge.
    # ------------------------------------------------------------------
    mu_r = torch.ones(mesh.shape, dtype=sigma_3d.dtype, device=device)
    A_wrapper = yee_curl_curl_assemble(mesh, mu_r, sigma_3d, omega)
    A_coo = A_wrapper.to_sparse_coo()

    if return_system:
        # Hand the assembled operator + RHS back so the caller can stack several
        # frequencies into one block-diagonal solve (frequency batching).
        return A_coo, q
    E_s = sparse_linear_solve_with_adjoint(A_coo, q)
    return E_s


def _extract_fields_at_station(
    mesh: TensorMesh,
    *,
    E: torch.Tensor,
    B: torch.Tensor,
    station_xy: tuple[float, float],
    station_z: float = 0.0,
    projections: Mapping[str, YeeReceiverProjection] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample ``(Ex, Ey, Hx, Hy, Hz)`` at a surface station from Yee vectors.

    The station depth is the public ``station_z`` coordinate. Each field uses
    its component-specific immutable Yee plan, so off-grid stations are sampled
    multilinearly and any coordinate outside the closed staggered support is
    rejected instead of clamped.

    The returned ``Hx, Hy, Hz`` are the supplied ``B`` values sampled on
    Yee faces. the operator divides ``curl(E) / (-iω)`` by ``μ₀`` to
    convert ``B`` to ``H`` before calling the impedance assembly; the
    callers (the full operator) can fold that ``1/μ₀`` factor in either
    upstream of ``B`` or downstream of ``Z``. The resulting impedance stays
    in the native ``exp(+iωt)`` convention used by
    :func:`mt1d_wait_impedance`; no final conjugation is applied.

    Args:
        mesh: 3D :class:`TensorMesh`. Shape convention ``(nz, ny, nx)`` per the
            Yee curl-curl assembler; nonuniform cell widths are supported.
        E: Complex ``(n_edges,)`` total E-field on Yee edges (``[Ex; Ey; Ez]``).
        B: Complex ``(n_faces,)`` total B-field (or H-field) on Yee faces
            (``[Bx; By; Bz]``). The convention is opaque to this routine;
            it samples the supplied vector with the matching component plan.
        station_xy: Public ``(x, y)`` station coordinates in metres.
        station_z: Public downward-positive station depth in metres.

    Returns:
        Five complex128 0-d tensors: ``(Ex, Ey, Hx, Hy, Hz)``. Autograd
        flows through ``E`` and ``B`` (this is a constant weighted gather).
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            f"_extract_fields_at_station requires a 3D mesh, got n_dim={mesh.n_dim}",
            object_name="_extract_fields_at_station",
            field="mesh.n_dim",
            expected=3,
            actual=mesh.n_dim,
        )
    if projections is None:
        projections = _build_mt_station_projections(
            mesh,
            (station_xy[0], station_xy[1], station_z),
        )
    values = tuple(
        apply_yee_receiver_projection(
            projections[channel],
            E=E,
            B=B,
            receiver_index=0,
        ).to(torch.complex128)
        for channel in ("ex", "ey", "hx", "hy", "hz")
    )
    return values


def _build_mt_station_projections(
    mesh: TensorMesh,
    position_xyz: tuple[float, float, float],
) -> dict[str, YeeReceiverProjection]:
    """Build MT station plans with the explicit surface-H convention.

    A station at public depth zero is located on the air-earth interface.
    Tangential magnetic components live at Yee z-centres rather than z-nodes,
    so their one-sided surface value is represented by the first cell centre.
    This MT-specific physical convention leaves the shared receiver builder
    strict: arbitrary points outside a component's support still fail.
    """
    base_position = torch.tensor([position_xyz], dtype=torch.float64)
    plans: dict[str, YeeReceiverProjection] = {}
    for channel in ("ex", "ey", "hx", "hy", "hz"):
        channel_position = base_position
        if channel in ("hx", "hy") and position_xyz[2] == 0.0:
            channel_position = base_position.clone()
            channel_position[0, 2] = 0.5 * mesh.cell_widths[0][0]
        plans[channel] = build_yee_receiver_projection(
            mesh,
            channel_position,
            channel=channel,
            layout="cartesian",
            n_sources=2,
        )
    return plans


def _impedance_and_tipper_from_two_pol(
    Ex_xp: torch.Tensor,
    Ey_xp: torch.Tensor,
    Hx_xp: torch.Tensor,
    Hy_xp: torch.Tensor,
    Hz_xp: torch.Tensor,
    Ex_yp: torch.Tensor,
    Ey_yp: torch.Tensor,
    Hx_yp: torch.Tensor,
    Hy_yp: torch.Tensor,
    Hz_yp: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Solve the 2x2 impedance system and the 2x1 tipper system from two
    polarisations.

    Impedance assembly: build the (component, polarisation) field matrices
    and solve via ``E = Z @ H ⇒ Z = E @ H^{-1}``.

    ::

        [Ex_xp  Ex_yp]   [Zxx  Zxy] [Hx_xp  Hx_yp]
        [Ey_xp  Ey_yp] = [Zyx  Zyy] [Hy_xp  Hy_yp]

    Tipper assembly: the per-station
    2x2 system relates ``Hz`` to ``(Hx, Hy)`` via

    ::

        [Hx_xp  Hy_xp] [Tzx]   [Hz_xp]
        [Hx_yp  Hy_yp] [Tzy] = [Hz_yp]

    Solved with :func:`torch.linalg.solve`.

    Args:
        Ex_xp, Ey_xp, Hx_xp, Hy_xp, Hz_xp: Complex128 0-d tensors, fields at the
            station for the x-polarised primary.
        Ex_yp, Ey_yp, Hx_yp, Hy_yp, Hz_yp: Complex128 0-d tensors, fields at the
            station for the y-polarised primary.

    Returns:
        ``(Z_xx, Z_xy, Z_yx, Z_yy, T_zx, T_zy)``: six complex128 0-d tensors.
        **No** MT-convention complex conjugation is applied here or by the full
        operator. The result already matches ``mt1d_wait_impedance``'s native
        ``exp(+iωt)`` phase. Autograd flows through every input.
    """
    # E_mat, H_mat: 2x2 complex matrices with rows indexed by output
    # component and columns by polarisation.
    E_mat = torch.stack(
        [
            torch.stack([Ex_xp, Ex_yp]),
            torch.stack([Ey_xp, Ey_yp]),
        ]
    )  # (2, 2)
    H_mat = torch.stack(
        [
            torch.stack([Hx_xp, Hx_yp]),
            torch.stack([Hy_xp, Hy_yp]),
        ]
    )  # (2, 2)

    # Z = E_mat @ H_mat^{-1}. We solve H_mat^T @ Z^T = E_mat^T instead
    # of building H_mat^{-1} explicitly: numerically equivalent but
    # autograd-friendly without an explicit inverse.
    #     E_mat = Z @ H_mat
    # =>  E_mat^T = H_mat^T @ Z^T
    Z_T = torch.linalg.solve(H_mat.transpose(-1, -2), E_mat.transpose(-1, -2))
    Z = Z_T.transpose(-1, -2)
    Z_xx = Z[0, 0]
    Z_xy = Z[0, 1]
    Z_yx = Z[1, 0]
    Z_yy = Z[1, 1]

    # Tipper: per-receiver 2x2 system using polarisation rows.
    H_pol_rows = torch.stack(
        [
            torch.stack([Hx_xp, Hy_xp]),
            torch.stack([Hx_yp, Hy_yp]),
        ]
    )  # (2, 2): row = polarisation, col = component
    Hz_vec = torch.stack([Hz_xp, Hz_yp]).unsqueeze(-1)  # (2, 1)
    T = torch.linalg.solve(H_pol_rows, Hz_vec).squeeze(-1)
    T_zx = T[0]
    T_zy = T[1]

    return Z_xx, Z_xy, Z_yx, Z_yy, T_zx, T_zy


@dataclass(frozen=True)
class _EdgeMTAssets(EdgeFemAssetsBase):
    """σ-independent Whitney + background assets of one mesh
    (see ``MT3D._edge_fem_assets``).

    Attributes:
        plan: the mesh's :class:`EdgeAssemblyPlan` (geometric factors + COO
            index pattern), reused across frequencies.
        boundary_mask: ``(n_edges,)`` bool, outer-boundary edges (PEC pins).
        edge_records: the mesh's :class:`EdgeRecords` (canonical node pairs,
            tangents, lengths, midpoints) for the primary-field projection.
        cell_edge_ids: ``(n_cells, 6)`` long, global edge ids per cell.
        cell_edge_signs: ``(n_cells, 6)`` float64, local→canonical signs.
        w_centroid: ``(n_cells, 6, 3)`` complex128, Whitney basis vectors at
            the cell barycentre, ``W_e(centroid) = (g_j - g_i)/4``.
        curl_w: ``(n_cells, 6, 3)`` complex128, constant per-cell curls,
            ``curl W_e = 2 g_i × g_j``.
        cell_centers: ``(n_cells, 3)`` float64, station-binding points.
        sigma_bg_1d: ``(n_layer,)`` float64, validated background profile.
        layer_thicknesses: ``(n_layer-1,)`` float64, background interfaces.
        sigma_bg_cells: ``(n_cells,)`` float64, background σ expanded to the
            cells by the structured centre-depth layer rule.
    """

    sigma_bg_1d: torch.Tensor
    layer_thicknesses: torch.Tensor
    sigma_bg_cells: torch.Tensor


def _layered_plane_wave_E_at_depths(
    depths: torch.Tensor,
    *,
    sigma_bg_1d: torch.Tensor,
    layer_thicknesses: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Downgoing 1D-layered plane-wave E at arbitrary depths (closed form).

    The SAME piecewise field :func:`_plane_wave_primary_field` walks node by
    node on the Yee grid, written as its closed form so it can be sampled at
    arbitrary (edge-midpoint) depths::

        E(z) = E_top(m) · exp(-k_m · (z - z_top(m))),   z in layer m,
        k_m  = sqrt(i·ω·μ₀·σ_m),
        E_top(0) = 1,  E_top(m+1) = E_top(m) · exp(-k_m · h_m),

    i.e. surface-normalised (``E(0) = 1``), continuous across interfaces,
    decaying with each layer's own wavenumber, bottom layer a half-space
    (``exp(+iωt)`` convention, matches the structured recursion and
    :func:`mt1d_wait_impedance`'s ``k = sqrt(iωμ₀σ)`` branch). The
    node-stepping and closed-form evaluations agree exactly in real
    arithmetic (exponentials of summed steps factor into products); the
    parity is pinned by a consistency test against the structured field.

    Args:
        depths: ``(n,)`` float64 depths in m (positive downward, surface 0).
        sigma_bg_1d: ``(n_layer,)`` real layer conductivities, top→bottom.
        layer_thicknesses: ``(n_layer-1,)`` real thicknesses in m; empty for
            a half-space.
        omega: angular frequency in rad/s.

    Returns:
        ``(n,)`` complex128 horizontal E amplitude at each depth.
    """
    if sigma_bg_1d.ndim != 1:
        raise GeoBrainError(
            f"sigma_bg_1d must be 1D, got shape {tuple(sigma_bg_1d.shape)}",
            object_name="_layered_plane_wave_E_at_depths",
            field="sigma_bg_1d",
            expected="1D tensor",
            actual=tuple(sigma_bg_1d.shape),
        )
    n_layer = int(sigma_bg_1d.numel())
    if layer_thicknesses.ndim != 1 or (int(layer_thicknesses.numel()) != max(0, n_layer - 1)):
        raise GeoBrainError(
            "layer_thicknesses must have n_layer-1 entries "
            f"(got n_layer={n_layer}, "
            f"thicknesses={int(layer_thicknesses.numel())})",
            object_name="_layered_plane_wave_E_at_depths",
            field="layer_thicknesses",
            expected=(max(0, n_layer - 1),),
            actual=int(layer_thicknesses.numel()),
        )
    device = sigma_bg_1d.device
    # ``contiguous()``: callers pass column slices (edge-midpoint z), which
    # torch.searchsorted would otherwise warn about and copy internally.
    z = depths.to(dtype=torch.float64, device=device).contiguous()
    omega_c = torch.tensor(omega, dtype=torch.complex128, device=device)
    sigma_c = sigma_bg_1d.to(torch.complex128)
    k_layer = torch.sqrt(1j * omega_c * MU_0 * sigma_c)  # (n_layer,)

    if n_layer == 1:
        layer_tops = torch.zeros(1, dtype=torch.float64, device=device)
        layer_bottoms = torch.tensor(
            [float("inf")],
            dtype=torch.float64,
            device=device,
        )
        e_tops = torch.ones(1, dtype=torch.complex128, device=device)
    else:
        th = layer_thicknesses.to(dtype=torch.float64, device=device)
        cum = torch.cumsum(th, dim=0)
        layer_tops = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), cum])
        layer_bottoms = torch.cat(
            [cum, torch.tensor([float("inf")], dtype=torch.float64, device=device)]
        )
        decay = torch.exp(-k_layer[:-1] * th.to(torch.complex128))
        e_tops = torch.cat(
            [
                torch.ones(1, dtype=torch.complex128, device=device),
                torch.cumprod(decay, dim=0),
            ]
        )

    # Smallest layer index whose bottom depth >= z: the structured
    # z-centre layer rule (interface depths land in the upper layer; the
    # field is continuous there so the choice is value-neutral).
    idx = torch.clamp(torch.searchsorted(layer_bottoms, z), max=n_layer - 1)
    dz_in_layer = (z - layer_tops[idx]).to(torch.complex128)
    return e_tops[idx] * torch.exp(-k_layer[idx] * dz_in_layer)


def _edge_mt_primary_dofs(
    records: EdgeRecords,
    *,
    sigma_bg_1d: torch.Tensor,
    layer_thicknesses: torch.Tensor,
    omega: float,
    polarization: str,
) -> torch.Tensor:
    """Project the layered plane-wave primary E onto Whitney edge dofs.

    The lowest-order Nédélec dof is the tangential circulation
    ``dof_e = ∫_e E·dl`` along the edge's canonical direction; the midpoint
    rule gives ``dof_e ≈ E(midpoint_e)·tangent_e·length_e``, the edge
    counterpart of the Yee path placing ``E_1d(z_node)`` on its x/y edge
    blocks. ``E_p`` carries only the horizontal polarisation component and
    depends on depth alone, so the dof reduces to
    ``E_1d(z_mid) · t̂_{x|y} · L``.

    Args:
        records: the mesh's :class:`EdgeRecords` (midpoints, unit tangents,
            lengths along the canonical small-node → large-node direction).
        sigma_bg_1d: ``(n_layer,)`` background layer conductivities.
        layer_thicknesses: ``(n_layer-1,)`` thicknesses (empty = half-space).
        omega: angular frequency in rad/s.
        polarization: ``"x"`` or ``"y"``.

    Returns:
        ``(n_edges,)`` complex128 primary edge circulations.
    """
    if polarization not in ("x", "y"):
        raise GeoBrainError(
            f"polarization must be 'x' or 'y'; got {polarization!r}",
            object_name="_edge_mt_primary_dofs",
            field="polarization",
            expected="'x' or 'y'",
            actual=polarization,
        )
    comp = 1 if polarization == "x" else 2
    e_mid = _layered_plane_wave_E_at_depths(
        records.midpoint[:, 0],
        sigma_bg_1d=sigma_bg_1d,
        layer_thicknesses=layer_thicknesses,
        omega=omega,
    )  # (ne,)
    proj = (records.tangent[:, comp] * records.length).to(torch.complex128)
    return e_mid * proj


def _edge_mt_secondary_solve(
    plan: EdgeAssemblyPlan,
    boundary_mask: torch.Tensor,
    *,
    sigma_cells: torch.Tensor,
    sigma_bg_cells: torch.Tensor,
    omega: float,
    e_primary: torch.Tensor,
    return_system: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Solve the Whitney MT secondary system for one frequency.

    The edge-element analogue of :func:`_secondary_field_solve`, the same
    ``exp(+iωt)`` physics in the edge family's SI scaling (the structured
    μ₀-multiplied system divided by μ₀)::

        A(σ) e_s = -iω (M(σ) - M(σ_1d)) e_p,
        A(σ)     = (1/μ₀)·K + iω·M(σ),

    where ``K``/``M`` are the Whitney curl-curl and mass operators of
    ``plan`` and ``σ_1d`` is the 1D background expanded to the cells. The
    RHS is the OPERATOR-DIFFERENCE form, one mass matvec at coefficient
    ``σ - σ_1d`` applied to ``e_p`` (identically zero when ``σ ≡ σ_1d``).
    PEC truncation: tangential ``e_s = 0`` on the outer boundary, imposed
    symmetrically, boundary rows AND columns of the COO values are zeroed
    (``torch.where``, keeping the values differentiable) with appended unit
    diagonal entries, and the boundary RHS rows are zeroed. (The FDEM3D
    edge-branch pattern with a per-cell layered background instead of a
    scalar one.)

    Args:
        plan: the mesh's :class:`EdgeAssemblyPlan`.
        boundary_mask: ``(n_edges,)`` bool from :func:`boundary_edge_mask`.
        sigma_cells: ``(n_cells,)`` float64 cell conductivities (autograd-live).
        sigma_bg_cells: ``(n_cells,)`` float64 background σ per cell.
        omega: angular frequency in rad/s.
        e_primary: ``(n_edges,)`` or ``(n_edges, k)`` complex128 primary edge
            circulations, both polarisations solve against ONE factorisation.

    Returns:
        ``e_s`` with the same shape as ``e_primary``, complex128.
    """
    n_cells = plan.n_cells
    delta_sigma = sigma_cells - sigma_bg_cells  # (nc,) float64
    zero_stiff = torch.zeros(n_cells, dtype=torch.float64)

    e_p2 = e_primary if e_primary.dim() == 2 else e_primary.unsqueeze(-1)
    minus_iomega = torch.complex(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-omega, dtype=torch.float64),
    )
    q = torch.stack(
        [
            minus_iomega
            * edge_operator_matvec(
                plan,
                delta_sigma,
                zero_stiff,
                e_p2[:, j],
            )
            for j in range(int(e_p2.shape[1]))
        ],
        dim=1,
    )  # (ne, k)
    mass_coeff = (1j * omega) * sigma_cells.to(torch.complex128)
    stiff_coeff = torch.full((n_cells,), 1.0 / MU_0, dtype=torch.float64)
    pin_plan = build_pec_pin_plan(
        boundary_mask,
        torch.stack([plan.rows, plan.cols]),
    )
    matrix = assemble_pec_pinned_operator(
        plan,
        pin_plan,
        mass_coeff=mass_coeff,
        stiffness_coeff=stiff_coeff,
    )
    rhs = pec_zero_rhs(boundary_mask, q)
    if return_system:
        return matrix, rhs
    e_s = solve_pec_pinned_edge_system(
        plan,
        boundary_mask,
        mass_coeff=mass_coeff,
        stiffness_coeff=stiff_coeff,
        rhs=q,
        pin_plan=pin_plan,
    )
    return e_s if e_primary.dim() == 2 else e_s.squeeze(-1)

