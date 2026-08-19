"""
3D Magnetotelluric (MT) forward operator.

Plane-wave natural-source incidence on a 1D-layered background; the heterogeneity
contribution is solved via the secondary-field formulation:

    E_total = E_primary + E_secondary
    A(σ) · E_s = -iω · (σ_edge - σ_bg_edge) · E_p

Uses the ``yee_curl_curl_assemble`` (complex sparse via the ``to_sparse_coo``
bridge) + the ``sparse_linear_solve_with_adjoint``. Primary field comes from
the ``mt1d_wait_impedance`` + depth recursion. Differentiability:
``IMPLICIT_VJP``. Time convention: physics ``e^{+iωt}`` (matches MT1D).

The historical ``use_closed_form_sigma_jacobian=True`` mode is retained only
as a migration sentinel and is rejected before mesh dispatch: its private LU
bridge cannot honor the exact repeated-frequency factor-reuse contract.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from geobrain.physics.em.config import EMExecutionConfig
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from geobrain.core import (
    ForwardContext,
    FieldShapeError,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from geobrain.mesh import TensorMesh
from geobrain.mesh.capabilities import (
    ConnectivityMesh,
    EdgeConnectivityMesh,
    StructuredMesh,
)
from geobrain.core.linalg import ScipySpluSolver, SparseFactorSolver
from geobrain.core.differentiability import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
)
from geobrain.physics.em.conventions import MU_0
from geobrain.physics.em.surveys import (
    EMReceiver,
    FrequencyDomainSurvey,
)
from geobrain.physics.em._engine import (
    AssemblyCacheKey,
    EMExecutionCache,
    exact_float_token,
    material_fingerprint,
    mesh_fingerprint,
    resolve_engine_solver,
    resolve_inductive_mesh_path,
    solver_execution_metadata,
    solver_factor_key,
    solver_settings,
    solve_compatible_rhs,
)
from geobrain.physics.em.numerics.edge_element import (
    barycentric_gradients,
    boundary_edge_mask,
    build_edge_assembly_plan,
    whitney_centroid_factors,
)
from geobrain.physics.em.numerics.finite_volume.curl_curl import (
    yee_curl_e_to_f,
)
from geobrain.physics.em.numerics.finite_volume.axis_convention import (
    swap_xy_cell_field,
    to_engine_mesh,
)
from geobrain.physics.em.receivers import (
    EdgeReceiverProjection,
    build_edge_receiver_projection,
)
from geobrain.physics.em.errors import EMCapabilityError
from geobrain.mesh import require_field_matches_mesh
from geobrain.physics.em.natural_source._mt3d_numerics import (  # noqa: F401  re-export: split section
    _EdgeMTAssets,
    _build_mt_station_projections,
    _edge_mt_primary_dofs,
    _edge_mt_secondary_solve,
    _extract_fields_at_station,
    _impedance_and_tipper_from_two_pol,
    _layered_plane_wave_E_at_depths,
    _plane_wave_primary_field,
    _secondary_field_solve,
)


@dataclass(frozen=True)
class MT3DStation(EMReceiver):
    """
    An MT station. ``position`` is ``(x, y, z)``; ``z`` is the depth of the
    air-earth interface the impedance is referenced to (default ``0`` = the mesh
    top, the legacy flat-surface convention). A non-zero ``z`` drapes the station
    on a topographic surface (an air band above it) so the impedance is extracted
    at the local surface; see ``_extract_fields_at_station``."""

    pass


@dataclass(frozen=True)
class MT3DSurvey(FrequencyDomainSurvey):
    """
    3D MT survey: stations (receivers) + frequencies + optional 1D background σ.

    For natural-source MT there are no point sources; ``sources`` is typically
    an empty tuple. ``sigma_background_1d``, if supplied, is a 1D layer-σ
    tensor used to build the analytic plane-wave primary field. If None, the
    operator computes the background as the z-slice mean of the cell-centred
    σ at forward time.

    ``sigma_background_thicknesses`` carries the ``(n_layer - 1,)`` layer
    thicknesses (m, top→bottom; empty/None for a half-space) belonging to
    ``sigma_background_1d``. It is consumed ONLY by the unstructured
    edge-element branch, which has no z-cell grid to infer interfaces from.
    The structured Yee branch keeps its original semantics, interfaces from
    the mesh z-cell widths (``n_layer == nz``) or the mean-dz fallback, and
    ignores this field entirely, preserving bit-level behaviour.

    Attributes:
        sources / receivers: acquisition tables.
        frequencies: sounding frequencies [Hz].
        sigma_background_1d / sigma_background_thicknesses: layered
            background model [S/m] and thicknesses [m].
    """

    sigma_background_1d: torch.Tensor | None = None
    sigma_background_thicknesses: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # Chain to EMSurvey's abstract-instance guard.
        super().__post_init__()
        # Pairing contract: thicknesses describe the interfaces OF a
        # layered ``sigma_background_1d`` profile, so they cannot stand
        # alone and must carry exactly n_layer-1 entries. Enforced at
        # construction so a mispaired survey fails here, not deep inside
        # the edge-element forward.
        if self.sigma_background_thicknesses is not None:
            if self.sigma_background_1d is None:
                raise GeoBrainError(
                    "MT3DSurvey.sigma_background_thicknesses requires "
                    "sigma_background_1d, thicknesses describe the "
                    "interfaces of that layered profile",
                    object_name="MT3DSurvey",
                    field="sigma_background_thicknesses",
                    expected="sigma_background_1d is not None",
                    actual=None,
                )
            n_layer = int(self.sigma_background_1d.numel())
            if int(self.sigma_background_thicknesses.numel()) != n_layer - 1:
                raise GeoBrainError(
                    "MT3DSurvey.sigma_background_thicknesses must have "
                    "n_layer-1 entries (one per interface of "
                    "sigma_background_1d)",
                    object_name="MT3DSurvey",
                    field="sigma_background_thicknesses",
                    expected=(n_layer - 1,),
                    actual=tuple(self.sigma_background_thicknesses.shape),
                )


class MT3D(EMOperatorDiscovery, ForwardOperator):
    """3D MT forward operator: plane-wave natural-source incidence.

    Device: CPU / float64 only (scipy sparse LU); move inputs to CPU first.

    Axis convention (platform-wide): the public contract is
    ``(nz, nx, ny)``: ``sigma`` shaped ``mesh.shape`` with mesh axis-1 = x and
    axis-2 = y. Station positions are physical ``(x, y, z)`` and impedance /
    tipper component labels (``z_xy`` etc.) are in that physical frame. The
    internal Yee curl-curl engine runs in ``(nz, ny, nx)``; the operator bridges
    the two by an x↔y swap of the mesh + σ at its input boundary (curl-curl is
    x↔y symmetric, so this is a numerically-exact relabel).

    Mirrors ``MT3DOperator`` element-for-element on uniform meshes:
    builds two analytic 1D-layered plane-wave primary fields (x- and
    y-polarisation), solves the secondary-field equation
    ``A(σ) E_s = -iωμ₀(σ-σ_bg) E_p`` for each, samples ``(E, H)`` at
    every station, and assembles the 2x2 impedance tensor plus 2x1
    tipper via :func:`_impedance_and_tipper_from_two_pol`.

    Time convention: ``exp(+iωt)`` (matches both the curl-curl
    assembler, which carries a ``+iωμ₀σ`` diagonal, and
    :func:`mt1d_wait_impedance`). **No** complex conjugation is applied
    at the end: this operator works directly in the MT field-practice
    ``exp(+iωt)`` convention, so no convention swap is needed.

    The faces vector passed into :func:`_extract_fields_at_station` is
    ``H = -curl(E)/(iωμ₀)`` (Faraday + free-space ``μ_r = 1``), so the
    ``E/H`` ratio is the physical surface impedance in ohm.

    Output ``ForwardOutput.data`` channels (each a single native-complex
    tensor of shape ``(n_stations, n_freq)``, one key per complex channel,
    the platform-wide complex-data contract):

    ``z_xx``, ``z_xy``, ``z_yx``, ``z_yy`` (impedance tensor),
    ``t_zx``, ``t_zy`` (tipper).

    Differentiability: :attr:`IMPLICIT_VJP` through ``sigma``. The σ-path
    enters both the system matrix ``A`` (handled by the splu-bridge
    adjoint) and the RHS ``q`` (handled by torch autograd through
    :func:`_cell_to_edge_avg`); the two contributions sum automatically.

    Mesh dispatch (declaration-based, mirroring FDEM3D): a mesh declaring
    ``StructuredMesh`` takes the original Yee path above bit-identically
    (``sigma`` shaped ``mesh.shape``); a mesh declaring
    ``EdgeConnectivityMesh``: a 3-D simplex-built
    :class:`~geobrain.mesh.unstructured.UnstructuredMesh`: takes the
    lowest-order Nédélec/Whitney edge-element path
    (:meth:`_forward_edge_fem`, ``sigma`` flat ``(n_cells,)``). The
    capability is read off the INSTANCE (3-D simplex instances refine the
    class declaration), per the ``require_mesh`` rule. Any other
    ``ConnectivityMesh`` (e.g. OctreeMesh) raises a :class:`GeoBrainError`.
    Both paths share the SAME physics convention, ``exp(+iωt)``
    throughout, downgoing 1D-layered plane-wave primary, secondary-field
    formulation, ``H = -curl(E)/(iωμ₀)``, no final conjugation, and emit
    identical ``data`` keys and shapes. The edge branch follows the
    edge-element family's SI scaling ``A = (1/μ₀)·K + iω·M(σ)`` with RHS
    ``−iω(M(σ)−M(σ_1d))e_p``, which is the structured μ₀-multiplied system
    scaled by ``1/μ₀``, the identical physics.

    Args:
        survey: MT 3-D acquisition.
        config: :class:`~geobrain.physics.em.EMExecutionConfig` execution
            policy (solver selection, closed-form sigma Jacobian).
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        output_keys=(
            "z_xx",
            "z_xy",
            "z_yx",
            "z_yy",
            "t_zx",
            "t_zy",
        ),
    )
    # Relaxed from (StructuredMesh,) when the edge-element branch landed
    # (the FDEM3D precedent): in-operator dispatch picks the Yee or the
    # Nédélec path; non-simplex non-structured meshes raise explicitly.
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (ConnectivityMesh,)
    # Sufficiency is disjunctive (see geobrain.mesh.capabilities): the flat field
    # above stays the NECESSARY set; at least one group below must also be
    # fully declared. Structured -> Yee finite volume; EdgeConnectivity ->
    # Nedelec edge elements. Octree satisfies neither (resolve_mesh_path).
    requires_mesh_capabilities_any: ClassVar[tuple[tuple[type, ...], ...]] = (
        (StructuredMesh,),
        (EdgeConnectivityMesh,),
    )

    def __init__(
        self,
        survey: MT3DSurvey,
        *,
        config: EMExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, MT3DSurvey):
            raise GeoBrainError(
                "MT3D requires an MT3DSurvey",
                object_name="MT3D",
                field="survey",
                expected="MT3DSurvey",
                actual=type(survey).__name__,
            )
        self.survey = survey
        cfg = config if config is not None else EMExecutionConfig()
        self._use_closed_form_sigma_jacobian = cfg.use_closed_form_sigma_jacobian
        self._solver = resolve_engine_solver(cfg.resolve_solver())
        # M6: cache the σ-independent Yee curl (edge→face) per mesh so an
        # inversion loop does not rebuild the topology every forward.
        # NOTE: cached payloads are σ-INDEPENDENT pure geometry/topology (keyed
        # on mesh identity, invalidated on mesh change): no autograd tensor is
        # cached, so a stale cache cannot poison a VJP. One operator instance
        # assumes one mesh per thread; these caches are NOT thread-safe (safe in
        # practice: the inverter is a strictly sequential backward() loop).
        self._C_coo = None
        self._C_cache_mesh = None
        # Edge-element branch: σ-independent Whitney assets (assembly plan,
        # PEC boundary mask, centroid/curl evaluation factors, per-cell
        # background σ) cached per mesh, plus the per-(mesh, freq) primary
        # edge-dof cache: both σ-independent, reused across forwards.
        self._edge_assets: _EdgeMTAssets | None = None
        self._edge_assets_mesh = None
        self._edge_primary_cache: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
        # Unit-4 axis bridge: cache the x↔y-swapped (nz, ny, nx) engine mesh per
        # user-mesh identity so the σ-independent curl/topology caches (keyed on
        # the engine mesh) still hit across an inversion loop.
        self._engine_mesh_cache: TensorMesh | None = None
        self._engine_mesh_src = None

    def _engine_mesh(self, user_mesh: TensorMesh) -> TensorMesh:
        """x↔y-swapped ``(nz, ny, nx)`` engine mesh for a ``(nz, nx, ny)`` user
        mesh, cached on the user-mesh identity (see
        :func:`~geobrain.physics.em.numerics.finite_volume.axis_convention.to_engine_mesh`)."""
        if self._engine_mesh_src is not user_mesh:
            self._engine_mesh_cache = to_engine_mesh(user_mesh)
            self._engine_mesh_src = user_mesh
        return self._engine_mesh_cache

    def _curl_coo(self, mesh: TensorMesh) -> torch.Tensor:
        """Yee edge→face curl as a coalesced complex128 COO, cached per mesh.

        Pure mesh topology (entries ±1/Δ): σ-independent and carrying no
        autograd graph, so the cached operator is reused across forwards and
        frequencies, gradients still flow through the fields it multiplies (M6).
        Keyed on mesh identity (a stored reference, not ``id``), so a new mesh
        rebuilds it.
        """
        if self._C_coo is None or self._C_cache_mesh is not mesh:
            C_real = yee_curl_e_to_f(mesh)
            self._C_coo = C_real.to_sparse_coo().coalesce().to(torch.complex128)
            self._C_cache_mesh = mesh
        return self._C_coo

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (sigma_3d,) = state.fetch("sigma")
        if self._use_closed_form_sigma_jacobian:
            raise EMCapabilityError(
                "MT3D closed-form sigma Jacobian mode does not support exact "
                "frequency-factor reuse",
                details={
                    "mode": "closed_form_sigma_jacobian",
                    "remediation": "use the default implicit-VJP sparse factor path",
                },
                object_name="MT3D",
                field="use_closed_form_sigma_jacobian",
                expected=False,
                actual=True,
                hint="use MT3D(..., use_closed_form_sigma_jacobian=False)",
            )
        if sigma_3d.device.type != "cpu" and isinstance(self._solver, ScipySpluSolver):
            raise GeoBrainError(
                "MT3D's configured scipy splu solver is CPU-only; move sigma to cpu "
                "or inject a GPU SparseFactorSolver.",
                object_name="MT3D",
                field="sigma.device",
                expected="cpu (or a GPU sparse backend)",
                actual=sigma_3d.device,
            )
        # ``sparse_linear_solve_with_adjoint`` (scipy splu path) requires
        # float64. Promote sigma → float64 at the entry (the DC3D pattern)
        # instead of failing deep inside the solver with an
        # ``A_sparse.dtype`` error; ``.to()`` is differentiable, so the
        # autograd chain to the caller's σ is preserved.
        sigma_3d = sigma_3d if sigma_3d.dtype == torch.float64 else sigma_3d.to(torch.float64)
        mesh = ctx.require_mesh()
        if mesh.n_dim != 3:
            raise GeoBrainError(
                "MT3D requires a 3D mesh (pass the mesh via ForwardContext.of(mesh=...))",
                object_name="MT3D",
                field="mesh",
                expected="3D",
                actual=mesh.n_dim,
            )

        # Capability dispatch (declaration-based, the shared inductive-EM
        # ladder): StructuredMesh → Yee FV here, EdgeConnectivityMesh → the
        # Nédélec edge-element path, anything else (octree) raises.
        if resolve_inductive_mesh_path(mesh, operator_name="MT3D") == "edge":
            return self._forward_edge_fem(sigma_3d, mesh)

        require_field_matches_mesh(mesh, sigma_3d, name="sigma", owner="MT3D")

        station_projections = tuple(
            _build_mt_station_projections(mesh, station.position)
            for station in self.survey.receivers
        )

        # Unit-4 axis bridge: the operator's public contract is the platform
        # (nz, nx, ny) layout (axis-1 = x, axis-2 = y); the Yee engine below is
        # written in (nz, ny, nx). Swap BOTH mesh geometry and σ so the engine
        # receives the algebraically-identical (nz, ny, nx) encoding of the same
        # physics. Station coords and impedance component labels are
        # layout-independent and pass through unchanged. (curl-curl is x↔y
        # symmetric ⇒ exact relabel; validated by test_axis_convention_engine.)
        mesh = self._engine_mesh(mesh)
        sigma_3d = swap_xy_cell_field(sigma_3d)

        # Background σ profile (1D layered): from survey or computed.
        # ``layer_thicknesses`` is sized n_layer-1 (empty for half-space).
        # MT3D-NU: use the actual per-cell z-widths from ``mesh.cell_widths[0]``
        # so layer interfaces sit at the real Yee z-node depths on a padded
        # mesh. On a uniform mesh ``mesh.cell_widths[0]`` is a constant-dz
        # tensor and the result is bit-identical to the legacy
        # ``mesh.spacing[0]`` path.
        wz = mesh.cell_widths[0].to(
            dtype=torch.float64,
            device=sigma_3d.device,
        )
        nz_mesh = int(mesh.shape[0])
        if self.survey.sigma_background_1d is not None:
            sigma_bg_1d = self.survey.sigma_background_1d
            n_layer = int(sigma_bg_1d.numel())
            if n_layer == 1:
                layer_thicknesses = torch.tensor(
                    [],
                    dtype=torch.float64,
                    device=sigma_3d.device,
                )
            elif n_layer == nz_mesh:
                # One layer per z-cell: thicknesses follow the actual
                # cell heights from ``mesh.cell_widths[0]``. The bottom
                # (n_layer-1) layer is the half-space (thickness ignored).
                layer_thicknesses = wz[: n_layer - 1].contiguous()
            else:
                # Caller-supplied layering that does NOT match the mesh
                # z-cell count: fall back to uniform spacing using the
                # mean-dz fallback (legacy-compatible behaviour).
                dz_val = float(mesh.spacing[0])
                layer_thicknesses = torch.full(
                    (n_layer - 1,),
                    dz_val,
                    dtype=torch.float64,
                    device=sigma_3d.device,
                )
        else:
            # Default: z-slice mean per layer; thicknesses = cell heights.
            sigma_bg_1d = sigma_3d.mean(dim=(1, 2))  # (nz,)
            n_layer = int(sigma_bg_1d.numel())
            # n_layer == nz_mesh by construction; thicknesses = wz[:-1].
            layer_thicknesses = wz[: n_layer - 1].contiguous()

        # Yee curl operator (edge -> face), cached per mesh (M6): pure
        # topology reused across frequencies, polarisations, and forwards.
        C_coo = self._curl_coo(mesh).to(sigma_3d.device)

        n_stations = len(self.survey.receivers)
        n_freq = len(self.survey.frequencies)

        # Per-output buffers: each entry is a list of (n_freq,) tensors,
        # one per station, that we stack at the end into
        # ``(n_stations, n_freq)`` shape.
        out: dict[str, list[list[torch.Tensor]]] = {
            k: [[None] * n_freq for _ in range(n_stations)]
            for k in self.differentiability.output_keys
        }

        # One explicit cache owns this forward only. Both polarisations at an
        # exact frequency ride as two RHS columns through one factor; distinct
        # frequencies remain distinct matrix keys and factors.
        execution_cache = EMExecutionCache()
        matrix_mesh_version = mesh_fingerprint(mesh)
        matrix_material_version = material_fingerprint(sigma_3d)
        matrix_requires_gradient = bool(torch.is_grad_enabled() and sigma_3d.requires_grad)
        inductive_solver = self._solver
        inductive_solver_settings = solver_settings(inductive_solver)
        rhs_by_key: dict[AssemblyCacheKey, torch.Tensor] = {}
        primaries: list[tuple[float, torch.Tensor, torch.Tensor]] = []
        resolved: dict[int, torch.Tensor] = {}  # f_idx -> E_secondary (n_edges, 2)
        for f_idx, freq in enumerate(self.survey.frequencies):
            omega = 2.0 * math.pi * float(freq)
            E_xp_p = _plane_wave_primary_field(
                mesh,
                sigma_bg_1d=sigma_bg_1d,
                layer_thicknesses=layer_thicknesses,
                omega=omega,
                polarization="x",
            ).to(sigma_3d.device)
            E_yp_p = _plane_wave_primary_field(
                mesh,
                sigma_bg_1d=sigma_bg_1d,
                layer_thicknesses=layer_thicknesses,
                omega=omega,
                polarization="y",
            ).to(sigma_3d.device)
            primaries.append((omega, E_xp_p, E_yp_p))
            E_p_stack = torch.stack([E_xp_p, E_yp_p], dim=1)  # (n_edges, 2)
            assembly_key = AssemblyCacheKey(
                formulation_version="mt3d-yee-secondary-v1",
                mesh_fingerprint=matrix_mesh_version,
                material_version=matrix_material_version,
                boundary="pec",
                sample_value=exact_float_token(float(freq)),
                dtype=str(sigma_3d.dtype),
                device=str(sigma_3d.device),
                backend=inductive_solver_settings.backend,
                requires_gradient=matrix_requires_gradient,
            )
            if assembly_key in rhs_by_key:
                execution_cache.reuse_assembly(assembly_key)
                rhs = rhs_by_key[assembly_key]
            else:
                matrix, rhs = _secondary_field_solve(
                    mesh,
                    sigma_3d=sigma_3d,
                    sigma_bg_1d=sigma_bg_1d,
                    omega=omega,
                    E_primary=E_p_stack,
                    layer_thicknesses=layer_thicknesses,
                    return_system=True,
                )
                execution_cache.bind_assembly(assembly_key, matrix)
                rhs_by_key[assembly_key] = rhs
            resolved[f_idx], _ = solve_compatible_rhs(
                solver_factor_key(assembly_key, inductive_solver),
                rhs,
                cache=execution_cache,
                solver=inductive_solver,
            )

        for f_idx, (omega, E_xp_p, E_yp_p) in enumerate(primaries):
            E_s_stack = resolved[f_idx]
            E_xp = E_xp_p + E_s_stack[:, 0]
            E_yp = E_yp_p + E_s_stack[:, 1]

            # H = -curl(E) / (iω·μ₀). Free-space μ_r = 1 ⇒ H = B/μ₀.
            # Time convention exp(+iωt) ⇒ Faraday: curl(E) = -iω·B
            # ⇒ B = -curl(E)/(iω) ⇒ H = -curl(E)/(iω·μ₀).
            inv_iomega_mu0 = -1.0 / (1j * omega * MU_0)
            H_xp = inv_iomega_mu0 * torch.sparse.mm(
                C_coo,
                E_xp.unsqueeze(-1),
            ).squeeze(-1)
            H_yp = inv_iomega_mu0 * torch.sparse.mm(
                C_coo,
                E_yp.unsqueeze(-1),
            ).squeeze(-1)

            for s_idx, station in enumerate(self.survey.receivers):
                Ex_xp, Ey_xp, Hx_xp, Hy_xp, Hz_xp = _extract_fields_at_station(
                    mesh,
                    E=E_xp,
                    B=H_xp,
                    station_xy=(station.position[0], station.position[1]),
                    station_z=station.position[2],
                    projections=station_projections[s_idx],
                )
                Ex_yp, Ey_yp, Hx_yp, Hy_yp, Hz_yp = _extract_fields_at_station(
                    mesh,
                    E=E_yp,
                    B=H_yp,
                    station_xy=(station.position[0], station.position[1]),
                    station_z=station.position[2],
                    projections=station_projections[s_idx],
                )

                Z_xx, Z_xy, Z_yx, Z_yy, T_zx, T_zy = _impedance_and_tipper_from_two_pol(
                    Ex_xp,
                    Ey_xp,
                    Hx_xp,
                    Hy_xp,
                    Hz_xp,
                    Ex_yp,
                    Ey_yp,
                    Hx_yp,
                    Hy_yp,
                    Hz_yp,
                )

                # exp(+iωt) is used throughout (curl-curl, primary field,
                # and mt1d_wait_impedance all agree), so no conjugation
                # at the end.

                for name, val in (
                    ("z_xx", Z_xx),
                    ("z_xy", Z_xy),
                    ("z_yx", Z_yx),
                    ("z_yy", Z_yy),
                    ("t_zx", T_zx),
                    ("t_zy", T_zy),
                ):
                    out[name][s_idx][f_idx] = val

        # Stack into (n_stations, n_freq) native-complex tensors: one key
        # per complex channel.
        data: dict[str, torch.Tensor] = {}
        for key, rows in out.items():
            data[key] = torch.stack([torch.stack(row) for row in rows])
        diagnostics = execution_cache.diagnostics
        diagnostics.projection_count = 2 * n_stations * n_freq
        diagnostics.projected_element_count = 2 * n_stations * n_freq
        metadata = solver_execution_metadata(diagnostics, inductive_solver_settings)
        execution_cache.close()
        return ForwardOutput(data=data, metadata=metadata)

    # ------------------------------------------------------------------
    # Edge-element (Nédélec/Whitney) branch: 3-D simplex UnstructuredMesh
    # ------------------------------------------------------------------

    def _edge_background_profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Validated ``(sigma_bg_1d, layer_thicknesses)`` for the edge branch.

        The edge branch REQUIRES an explicit ``survey.sigma_background_1d``;
        the structured default (z-slice mean with thicknesses from the
        mesh z-cell widths) has no analogue on a mesh without z-slices, and
        silently substituting a flat mean would change the layered-primary
        semantics. Multi-layer profiles additionally need
        ``survey.sigma_background_thicknesses`` (``(n_layer-1,)``); a
        single-entry profile is a half-space and needs none.
        """
        sigma_bg_1d = self.survey.sigma_background_1d
        if sigma_bg_1d is None:
            raise GeoBrainError(
                "MT3D (edge-element path) requires survey.sigma_background_1d "
                "; the structured z-slice-mean default needs a z-cell grid "
                "that an unstructured mesh does not have",
                object_name="MT3D",
                field="survey.sigma_background_1d",
                expected="(n_layer,) float64 tensor",
                actual=None,
            )
        if sigma_bg_1d.ndim != 1 or int(sigma_bg_1d.numel()) < 1:
            raise GeoBrainError(
                "MT3D: sigma_background_1d must be a non-empty 1D tensor",
                object_name="MT3D",
                field="survey.sigma_background_1d",
                expected="(n_layer,)",
                actual=tuple(sigma_bg_1d.shape),
            )
        n_layer = int(sigma_bg_1d.numel())
        thicknesses = self.survey.sigma_background_thicknesses
        if thicknesses is None:
            thicknesses = torch.tensor([], dtype=torch.float64)
        if thicknesses.ndim != 1 or int(thicknesses.numel()) != n_layer - 1:
            raise GeoBrainError(
                "MT3D (edge-element path): sigma_background_thicknesses must "
                "have n_layer-1 entries (empty/None for a half-space)",
                object_name="MT3D",
                field="survey.sigma_background_thicknesses",
                expected=(n_layer - 1,),
                actual=tuple(thicknesses.shape),
            )
        return (
            sigma_bg_1d.to(torch.float64),
            thicknesses.to(torch.float64),
        )

    def _edge_fem_assets(self, mesh) -> "_EdgeMTAssets":
        """σ-independent Whitney assets for ``mesh``, cached per mesh identity.

        Pure geometry/topology plus the per-cell BACKGROUND σ (survey-fixed,
        layer value at the cell-centre depth, the structured z-centre rule):
        none of it carries an autograd graph, so the cache is reused across
        forwards and frequencies, exactly like the Yee ``_C_coo``. A mesh
        change also invalidates the per-frequency primary-dof cache.
        """
        if self._edge_assets is None or self._edge_assets_mesh is not mesh:
            node_coords = mesh.node_coords()
            # Depth convention guard: the edge branch mirrors the structured
            # Yee convention: z is DEPTH (positive downward) with the
            # air-earth interface at the z = 0 plane. Cells above the
            # surface (z < 0) have no meaning for the downgoing layered
            # primary, so they are rejected up front.
            z_min = float(node_coords[:, 0].min())
            z_extent = float(node_coords[:, 0].max()) - z_min
            if z_min < -1e-9 * max(1.0, z_extent):
                raise GeoBrainError(
                    "MT3D (edge-element path) interprets node z as DEPTH with "
                    "the surface at z = 0; the mesh must not extend above "
                    "z = 0",
                    object_name="MT3D",
                    field="mesh.node_coords",
                    expected="min(z) >= 0",
                    actual=z_min,
                )
            plan = build_edge_assembly_plan(mesh)
            cell_edge_ids, cell_edge_signs = mesh.cell_edges()
            g, _volume = barycentric_gradients(node_coords, mesh.cell_nodes())
            # Whitney barycentre value + constant curl (float64; cast to
            # complex128 below for this operator's complex E-field system).
            w_centroid, curl_w = whitney_centroid_factors(g)  # (nc, 6, 3)
            cell_centers = mesh.cell_centers()
            # Background σ per cell: structured semantics: each cell takes
            # the layer whose depth interval contains its CENTRE depth
            # (smallest layer index whose bottom depth >= z_centre).
            sigma_bg_1d, layer_thicknesses = self._edge_background_profile()
            n_layer = int(sigma_bg_1d.numel())
            if n_layer == 1:
                layer_per_cell = torch.zeros(
                    int(mesh.n_cells),
                    dtype=torch.long,
                )
            else:
                cum = torch.cumsum(layer_thicknesses, dim=0)
                layer_bottoms = torch.cat(
                    [
                        cum,
                        torch.tensor([float("inf")], dtype=torch.float64),
                    ]
                )
                layer_per_cell = torch.clamp(
                    torch.searchsorted(
                        layer_bottoms,
                        cell_centers[:, 0].contiguous(),
                    ),
                    max=n_layer - 1,
                )
            sigma_bg_cells = sigma_bg_1d[layer_per_cell]  # (nc,) float64
            self._edge_assets = _EdgeMTAssets(
                plan=plan,
                boundary_mask=boundary_edge_mask(mesh),
                edge_records=mesh.edge_records(),
                cell_edge_ids=cell_edge_ids,
                cell_edge_signs=cell_edge_signs,
                w_centroid=w_centroid.to(torch.complex128),
                curl_w=curl_w.to(torch.complex128),
                cell_centers=cell_centers,
                sigma_bg_1d=sigma_bg_1d,
                layer_thicknesses=layer_thicknesses,
                sigma_bg_cells=sigma_bg_cells,
            )
            self._edge_assets_mesh = mesh
            self._edge_primary_cache = {}
        return self._edge_assets

    def _edge_primary_pair(
        self,
        assets: "_EdgeMTAssets",
        freq: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(x-pol, y-pol) primary edge dofs at ``freq``, cached per (mesh, freq).

        σ-independent (the background profile is survey-fixed), so an
        inversion loop reuses the projection across forwards.
        """
        key = float(freq)
        cached = self._edge_primary_cache.get(key)
        if cached is None:
            omega = 2.0 * math.pi * key
            cached = tuple(
                _edge_mt_primary_dofs(
                    assets.edge_records,
                    sigma_bg_1d=assets.sigma_bg_1d,
                    layer_thicknesses=assets.layer_thicknesses,
                    omega=omega,
                    polarization=pol,
                )
                for pol in ("x", "y")
            )
            self._edge_primary_cache[key] = cached
        return cached

    def _forward_edge_fem(self, sigma: torch.Tensor, mesh) -> ForwardOutput:
        """Edge-element path for a 3-D simplex ``EdgeConnectivityMesh``.

        Symbol-for-symbol the structured pipeline on Whitney edge dofs
        (``dof_e = ∫_e E·dl`` along the canonical edge direction):

        1. ``e_p``, the SAME downgoing 1D-layered plane-wave closed form the
           Yee path samples at its z-nodes, projected to edge circulations
           by the midpoint rule (``E_p(z_mid)·t̂·L``; ``E_p`` is horizontal
           and depends on depth only; two polarisations, x and y);
        2. ``A(σ) e_s = -iω (M(σ) - M(σ_1d)) e_p`` with
           ``A = (1/μ₀)·K + iω·M(σ)`` (``exp(+iωt)``, the SI scaling of the
           edge-element family, the structured μ₀-multiplied system scaled
           by ``1/μ₀``, hence the identical physics) and PEC pinning of the
           outer-boundary edges, via the implicit-VJP sparse solve. The RHS
           is the OPERATOR-DIFFERENCE form: identically zero when
           ``σ ≡ σ_1d(z)``;
        3. station response from the DISCRETE total ``e = e_p + e_s`` (the
           structured semantics; its grid-projected primary is summed with
           the secondary before sampling): each surface station ``(x, y)``
           binds to the cell whose centre is nearest to ``(x, y, 0)``;
           ``E`` is the Whitney interpolant at that cell's barycentre and
           ``H = -curl(e)/(iωμ₀)`` the cell's constant Whitney curl;
        4. the 2x2 impedance + tipper algebra is REUSED verbatim
           (:func:`_impedance_and_tipper_from_two_pol`); no conjugation,
           identical ``data`` keys/shapes/convention as the Yee path.

        Applicability domain: the PEC truncation pins the tangential
        SECONDARY field on the whole outer boundary (top surface included),
        so 3-D anomalies must sit well inside the mesh, lateral/bottom
        margins of at least a skin depth, anomaly buried below the surface
       , and station fields are sampled at the (quarter-cell-deep)
        barycentre of the binding surface tetrahedron rather than at z = 0,
        where the pinned secondary would vanish identically. ``sigma`` is
        the flat ``(n_cells,)`` cell conductivity; the survey must supply
        ``sigma_background_1d`` (+ thicknesses when layered). Differentiable
        in ``sigma`` end-to-end: σ enters ``A`` (splu-bridge adjoint) and
        the RHS (autograd through the Whitney mass matvec); the
        contributions sum automatically.
        """
        n_cells = int(mesh.n_cells)
        if sigma.dim() != 1 or int(sigma.shape[0]) != n_cells:
            raise FieldShapeError(
                "MT3D (edge-element path): sigma must be flat (n_cells,)",
                object_name="MT3D",
                field="sigma",
                expected=(n_cells,),
                actual=tuple(sigma.shape),
            )

        assets = self._edge_fem_assets(mesh)

        n_stations = len(self.survey.receivers)
        n_freq = len(self.survey.frequencies)

        station_pts = torch.tensor(
            [station.position for station in self.survey.receivers],
            dtype=torch.float64,
        )  # (n_st, 3)
        edge_projections: tuple[EdgeReceiverProjection, ...] = tuple(
            build_edge_receiver_projection(
                mesh,
                station_pts[index : index + 1],
                channel="ex",
                layout="cartesian",
                n_sources=2,
            )
            for index in range(n_stations)
        )
        st_cells = torch.tensor(
            [projection.element_indices[0] for projection in edge_projections],
            dtype=torch.long,
        )
        st_ids = torch.tensor(
            [projection.local_edge_dof_indices[0] for projection in edge_projections],
            dtype=torch.long,
        )
        st_signs = torch.tensor(
            [projection.orientation_signs[0] for projection in edge_projections],
            dtype=torch.complex128,
        )
        st_basis = torch.tensor(
            [projection.basis_weights[0] for projection in edge_projections],
            dtype=torch.complex128,
        ).reshape(n_stations, 6, 3)
        st_curl_w = assets.curl_w[st_cells][:, :, (1, 2, 0)]  # public (x,y,z)

        # Same per-output buffering as the structured path.
        out: dict[str, list[list[torch.Tensor]]] = {
            k: [[None] * n_freq for _ in range(n_stations)]
            for k in self.differentiability.output_keys
        }

        execution_cache = EMExecutionCache()
        matrix_mesh_version = mesh_fingerprint(mesh)
        matrix_material_version = material_fingerprint(sigma)
        matrix_requires_gradient = bool(torch.is_grad_enabled() and sigma.requires_grad)
        inductive_solver = self._solver
        inductive_solver_settings = solver_settings(inductive_solver)
        rhs_by_key: dict[AssemblyCacheKey, torch.Tensor] = {}
        for f_idx, freq in enumerate(self.survey.frequencies):
            omega = 2.0 * math.pi * float(freq)
            inv_iomega_mu0 = -1.0 / (1j * omega * MU_0)

            # Both polarisations share the SAME A(σ, ω); only the RHS
            # differs: one factorisation per frequency (the structured
            # stacking pattern).
            e_p_x, e_p_y = self._edge_primary_pair(assets, float(freq))
            e_p_stack = torch.stack([e_p_x, e_p_y], dim=1)  # (ne, 2)
            assembly_key = AssemblyCacheKey(
                formulation_version="mt3d-whitney-secondary-v1",
                mesh_fingerprint=matrix_mesh_version,
                material_version=matrix_material_version,
                boundary="pec",
                sample_value=exact_float_token(float(freq)),
                dtype=str(sigma.dtype),
                device=str(sigma.device),
                backend=inductive_solver_settings.backend,
                requires_gradient=matrix_requires_gradient,
            )
            if assembly_key in rhs_by_key:
                execution_cache.reuse_assembly(assembly_key)
                rhs = rhs_by_key[assembly_key]
            else:
                matrix, rhs = _edge_mt_secondary_solve(
                    assets.plan,
                    assets.boundary_mask,
                    sigma_cells=sigma,
                    sigma_bg_cells=assets.sigma_bg_cells,
                    omega=omega,
                    e_primary=e_p_stack,
                    return_system=True,
                )
                execution_cache.bind_assembly(assembly_key, matrix)
                rhs_by_key[assembly_key] = rhs
            e_s_stack, _ = solve_compatible_rhs(
                solver_factor_key(assembly_key, inductive_solver),
                rhs,
                cache=execution_cache,
                solver=inductive_solver,
            )

            # Whitney evaluation of the total field at the station cells,
            # one polarisation per column: E (interpolant at the
            # barycentre) and H = -curl(e)/(iωμ₀) (per-tet constant).
            fields: list[tuple[torch.Tensor, ...]] = []
            for col in range(2):
                e_tot = e_p_stack[:, col] + e_s_stack[:, col]  # (ne,)
                local = st_signs * e_tot[st_ids]  # (n_st, 6)
                E_vec = torch.einsum("se,sek->sk", local, st_basis)
                H_vec = inv_iomega_mu0 * torch.einsum(
                    "se,sek->sk",
                    local,
                    st_curl_w,
                )
                fields.append((E_vec, H_vec))
            (E_xp, H_xp), (E_yp, H_yp) = fields

            for s_idx in range(n_stations):
                Z_xx, Z_xy, Z_yx, Z_yy, T_zx, T_zy = _impedance_and_tipper_from_two_pol(
                    E_xp[s_idx, 0],
                    E_xp[s_idx, 1],
                    H_xp[s_idx, 0],
                    H_xp[s_idx, 1],
                    H_xp[s_idx, 2],
                    E_yp[s_idx, 0],
                    E_yp[s_idx, 1],
                    H_yp[s_idx, 0],
                    H_yp[s_idx, 1],
                    H_yp[s_idx, 2],
                )
                # exp(+iωt) end-to-end: no conjugation (same as the Yee
                # path).
                for name, val in (
                    ("z_xx", Z_xx),
                    ("z_xy", Z_xy),
                    ("z_yx", Z_yx),
                    ("z_yy", Z_yy),
                    ("t_zx", T_zx),
                    ("t_zy", T_zy),
                ):
                    out[name][s_idx][f_idx] = val

        data: dict[str, torch.Tensor] = {}
        for key, rows in out.items():
            data[key] = torch.stack([torch.stack(row) for row in rows])
        diagnostics = execution_cache.diagnostics
        diagnostics.projection_count = 2 * n_stations * n_freq
        diagnostics.projected_element_count = 2 * n_stations * n_freq
        metadata = solver_execution_metadata(diagnostics, inductive_solver_settings)
        execution_cache.close()
        return ForwardOutput(data=data, metadata=metadata)


# ----------------------------------------------------------------------
# Private helper: analytic 1D-layered plane-wave primary E-field on Yee edges
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# Private helper: secondary-field sparse solve (Option B formulation)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# Private helpers: station-field extraction + 2x2 impedance/tipper assembly
# ----------------------------------------------------------------------








# ----------------------------------------------------------------------
# Edge-element (Nédélec/Whitney) branch helpers
# ----------------------------------------------------------------------








