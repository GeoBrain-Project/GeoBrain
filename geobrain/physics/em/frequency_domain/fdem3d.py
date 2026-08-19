"""
3D Frequency-domain EM (FDEM) controlled-source operator.

Magnetic-dipole source over a 3D earth. Output: complex EM-field
components at receivers across a frequency sweep. Differentiability:
``IMPLICIT_VJP`` via the ``sparse_linear_solve_with_adjoint`` + autograd
through the secondary RHS. Same secondary-field formulation as MT3D
(E6c), but with an analytic *wholespace magnetic-dipole* primary field
instead of a 1D plane-wave.

Time convention: public outputs use the canonical ``exp(+iωt)`` convention
of the curl-curl assembler without a return-path transformation.

Dual mesh paths: the capability contract is ``ConnectivityMesh``. A mesh
that also declares ``StructuredMesh`` (TensorMesh) takes the original Yee
staggered-grid path bit-identically; a mesh declaring
``EdgeConnectivityMesh`` (3-D simplex-built UnstructuredMesh) takes the
lowest-order Nédélec/Whitney edge-element path with the SAME physics
convention (``exp(+iωt)`` assembly, wholespace dipole primary, secondary
operator-difference RHS). Anything else (OctreeMesh)
raises; octree inductive EM is not implemented.

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

from geobrain.mesh import require_field_matches_mesh
from geobrain.core import (
    ForwardContext,
    FieldShapeError,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from geobrain.mesh import TensorMesh
from geobrain.core.linalg import ScipySpluSolver, SparseFactorSolver
from geobrain.core.differentiability import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
)
from geobrain.mesh.capabilities import (
    ConnectivityMesh,
    EdgeConnectivityMesh,
    EdgeRecords,
    StructuredMesh,
)
from geobrain.physics.em.results import FieldComponent
from geobrain.physics.em.conventions import MU_0
from geobrain.physics.em.surveys import (
    EMReceiver,
    FrequencyDomainSurvey,
    MagneticDipoleSource,
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
    EdgeFemAssetsBase,
    EdgeAssemblyPlan,
    barycentric_gradients,
    boundary_edge_mask,
    build_edge_assembly_plan,
    build_pec_pin_plan,
    edge_operator_matvec,
    pick_field_component,
    pec_zero_rhs,
    assemble_pec_pinned_operator,
    solve_pec_pinned_edge_system,
    whitney_centroid_factors,
)
from geobrain.physics.em.numerics.finite_volume.curl_curl import (
    yee_curl_e_to_f,
    yee_edge_count,
)
from geobrain.physics.em.numerics.finite_volume.receiver_sampling import (
    apply_yee_receiver_projection,
)
from geobrain.physics.em.numerics.finite_volume.axis_convention import (
    swap_xy_cell_field,
    to_engine_mesh,
)
from geobrain.physics.em.natural_source.mt3d import (
    _secondary_field_solve,
)
from geobrain.physics.em.receivers import (
    EdgeReceiverProjection,
    build_edge_receiver_projection,
    build_yee_receiver_projection,
)
from geobrain.physics.em.errors import EMCapabilityError


@dataclass(frozen=True)
class FDEM3DReceiver(EMReceiver):
    """FDEM3D receiver: adds field-component selector to ``EMReceiver``.

    Defaults to ``FieldComponent.BZ`` (the canonical airborne / loop-loop
    observable). Supported components: Ex, Ey, Ez, Bx, By, Bz (and the
    H-field variants HX/HY/HZ if needed; the operator handles conversion).

    Attributes:
        position: receiver location [m].
        component: recorded field component.
    """

    component: FieldComponent = FieldComponent.BZ


@dataclass(frozen=True)
class FDEM3DSurvey(FrequencyDomainSurvey):
    """
    3D FDEM survey: dipole source(s) + component receiver(s) + frequencies.

    Optional ``sigma_background_1d`` for the wholespace primary; if None,
    the operator computes it as the global mean of the cell-centred σ
    (detached; the background is a frozen primary-field parameter, not a
    σ-gradient path).

    Attributes:
        sources / receivers: acquisition tables.
        frequencies: modelling frequencies [Hz].
        sigma_background_1d: layered background conductivity [S/m].
    """

    sigma_background_1d: torch.Tensor | None = None


class FDEM3D(EMOperatorDiscovery, ForwardOperator):
    """
    3D frequency-domain EM forward operator (controlled-source magnetic dipole).

    Device: CPU / float64 only (scipy sparse LU); move inputs to CPU first.

    Axis convention (platform-wide): the public contract is
    ``(nz, nx, ny)``: ``sigma`` shaped ``mesh.shape`` with mesh axis-1 = x and
    axis-2 = y. Source / receiver positions are physical ``(x, y, z)`` and field
    component labels (``bz`` etc.) are in that physical frame. The internal Yee
    curl-curl engine runs in ``(nz, ny, nx)``; the operator bridges the two by an
    x↔y swap of the mesh + σ at its input boundary (curl-curl is x↔y symmetric,
    so this is a numerically-exact relabel).

    Maps a :class:`ModelState` containing ``"sigma"`` (cell-centred 3D
    conductivity, shape ``mesh.shape``) to a :class:`ForwardOutput` whose
    ``data`` dict holds complex-valued field-component channels indexed
    by ``(n_src, n_rcv, n_freq)``.

    Output schema:
    For each unique receiver component requested by the survey, one
    native-complex ``(n_src, n_rcv, n_freq)`` tensor is emitted under
    ``data["{comp}"]`` (the platform-wide complex-data contract, one key
    per complex channel, not split real/imag).

    ``{comp}`` is the lowercase :class:`FieldComponent` ``.value`` string
    (e.g. ``"bz"``, ``"ex"``, ``"hz"``). Receivers that don't request a
    given component contribute zeros in those slots (in case the user mixes
    different-component receivers in a single survey).

    Formulation:
    Secondary-field formulation:

    1. ``E_p`` analytic wholespace magnetic-dipole primary at ``sigma_bg``;
    2. ``A(σ) E_s = -iωμ₀ (σ_edge - σ_bg_edge) E_p`` via the E5 implicit-VJP
       sparse solve;
    3. ``E_total = E_p + E_s``;
    4. ``B = -curl(E_total) / (iω)`` from Faraday;
    5. Sample components at each receiver position.

    Time convention:
        The :func:`yee_curl_curl_assemble` and public output both use
        ``exp(+iωt)``. Consumers select comparison conventions with an
        explicit adapter.

    Differentiability: :attr:`IMPLICIT_VJP` through ``sigma``. The σ-path
    enters both ``A`` (handled by the splu-bridge adjoint inside
    :func:`_fdem3d_secondary_solve`) and the RHS (handled by torch
    autograd through ``_cell_to_edge_avg``); the two contributions sum
    automatically.

    Mesh dispatch (declaration-based, mirroring MT2D): a mesh declaring
    ``StructuredMesh`` takes the original Yee path above bit-identically
    (``sigma`` shaped ``mesh.shape``); a mesh declaring
    ``EdgeConnectivityMesh``: a 3-D simplex-built
    :class:`~geobrain.mesh.unstructured.UnstructuredMesh`: takes the
    lowest-order Nédélec/Whitney edge-element path
    (:meth:`_forward_edge_fem`, ``sigma`` flat ``(n_cells,)``). The
    capability is read off the INSTANCE (3-D simplex instances refine the
    class declaration), per the ``require_mesh`` rule. Any other
    ``ConnectivityMesh`` (e.g. OctreeMesh) raises a :class:`GeoBrainError`.
    Both paths share the SAME physics convention, ``exp(+iωt)`` assembly,
    wholespace magnetic-dipole primary at the same scalar ``sigma_bg``,
    operator-difference secondary RHS, ``B = -curl(E)/(iω)``, and emit
    identical ``data`` keys and shapes.

    Args:
        survey: frequency-domain 3-D acquisition.
        config: :class:`~geobrain.physics.em.EMExecutionConfig` execution
            policy (solver selection, closed-form sigma Jacobian).
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        # Output keys are component-based; the spec lists all 9 possible
        # for typing, but at runtime only the components requested by
        # survey receivers are populated. Each is a single native-complex
        # channel (one key per complex field component).
        output_keys=(
            "ex",
            "ey",
            "ez",
            "bx",
            "by",
            "bz",
            "hx",
            "hy",
            "hz",
        ),
    )
    # Relaxed from (StructuredMesh,) when the edge-element branch landed
    # (the MT2D precedent): in-operator dispatch picks the Yee or the
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
        survey: FDEM3DSurvey,
        *,
        config: EMExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, FDEM3DSurvey):
            raise GeoBrainError(
                "FDEM3D requires an FDEM3DSurvey",
                object_name="FDEM3D",
                field="survey",
                expected="FDEM3DSurvey",
                actual=type(survey).__name__,
            )
        self.survey = survey
        # Retained only as a migration sentinel; forward rejects True before
        # dispatch because that bridge cannot participate in exact factor reuse.
        cfg = config if config is not None else EMExecutionConfig()
        self._use_closed_form_sigma_jacobian = cfg.use_closed_form_sigma_jacobian
        self._solver = resolve_engine_solver(cfg.resolve_solver())
        # M6: cache the σ-independent Yee curl (edge→face) per mesh so an
        # inversion loop does not rebuild the topology every forward.
        # NOTE: every cached payload is σ-INDEPENDENT pure geometry/topology
        # (keyed on mesh identity, invalidated on mesh change), so it never
        # carries an autograd tensor and cannot poison a VJP. One operator
        # instance assumes one mesh per thread: these caches are NOT thread-safe
        # (the inverter is a strictly sequential backward() loop, so this is safe
        # in practice; do not call forward() on one instance concurrently).
        self._C_coo = None
        self._C_cache_mesh = None
        # Edge-element branch: σ-independent Whitney assets (assembly plan,
        # PEC boundary mask, centroid/curl evaluation factors) cached per
        # mesh, same identity-keyed pattern as ``_C_coo``.
        self._edge_assets: _EdgeFemAssets | None = None
        self._edge_assets_mesh = None
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
        autograd graph, so the cached operator is reused across forwards,
        sources, and frequencies, gradients still flow through the fields it
        multiplies (M6). Keyed on mesh identity (a stored reference, not
        ``id``), so a new mesh rebuilds it.
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
                "FDEM3D closed-form sigma Jacobian mode does not support exact "
                "frequency-factor reuse",
                details={
                    "mode": "closed_form_sigma_jacobian",
                    "remediation": "use the default implicit-VJP sparse factor path",
                },
                object_name="FDEM3D",
                field="use_closed_form_sigma_jacobian",
                expected=False,
                actual=True,
                hint="use FDEM3D(..., use_closed_form_sigma_jacobian=False)",
            )
        if sigma_3d.device.type != "cpu" and isinstance(self._solver, ScipySpluSolver):
            raise GeoBrainError(
                "FDEM3D's configured scipy splu solver is CPU-only; move sigma to cpu "
                "or inject a GPU SparseFactorSolver.",
                object_name="FDEM3D",
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
                "FDEM3D requires a 3D mesh (pass the mesh via ForwardContext.of(mesh=...))",
                object_name="FDEM3D",
                field="mesh",
                expected="3D",
                actual=mesh.n_dim,
            )

        # Capability dispatch (declaration-based, the shared inductive-EM
        # ladder): StructuredMesh → Yee FV here, EdgeConnectivityMesh → the
        # Nédélec edge-element path, anything else (octree) raises.
        if resolve_inductive_mesh_path(mesh, operator_name="FDEM3D") == "edge":
            return self._forward_edge_fem(sigma_3d, mesh)

        # The structured Yee FV path (assembly + receiver projection) reads scalar
        # ``mesh.spacing`` and assumes a UNIFORM TensorMesh. Reject non-uniform /
        # graded meshes here rather than silently mis-assembling and mis-binding
        # receivers to the wrong Yee cell. MT3D is the non-uniform-capable
        # inductive operator (it assembles from per-cell ``cell_widths``).
        if not mesh.is_uniform:
            raise GeoBrainError(
                "FDEM3D structured Yee FV path requires a uniform TensorMesh; "
                "non-uniform/graded meshes are not supported on this path "
                "(use MT3D for graded meshes, or an unstructured edge mesh)",
                object_name="FDEM3D",
                field="mesh",
                expected="uniform TensorMesh",
                actual="non-uniform TensorMesh",
            )

        require_field_matches_mesh(mesh, sigma_3d, name="sigma", owner="FDEM3D")

        receiver_projections = tuple(
            build_yee_receiver_projection(
                mesh,
                torch.tensor([receiver.position], dtype=torch.float64),
                channel=receiver.component.value,
                layout="cartesian",
                n_sources=len(self.survey.sources),
            )
            for receiver in self.survey.receivers
        )

        # Unit-4 axis bridge: the operator's public contract is the platform
        # (nz, nx, ny) layout (axis-1 = x, axis-2 = y); the Yee engine below is
        # written in (nz, ny, nx). Swap BOTH mesh geometry and σ so the engine
        # receives the algebraically-identical (nz, ny, nx) encoding of the same
        # physics. Source/receiver coords and field-component labels are
        # layout-independent and pass through unchanged. (curl-curl is x↔y
        # symmetric ⇒ exact relabel; validated by test_axis_convention_engine.)
        mesh = self._engine_mesh(mesh)
        sigma_3d = swap_xy_cell_field(sigma_3d)

        # Scalar background σ: uniform-halfspace wholespace primary.
        # If the survey supplies a 1D profile, use its first entry as the
        # scalar background (the wholespace primary needs a single σ
        # value); fall back to the global cell-mean otherwise. The mean is
        # DELIBERATELY detached: the background is a frozen primary-field
        # parameter, not a σ-gradient path (and float() on an
        # autograd-live tensor would raise a UserWarning).
        if self.survey.sigma_background_1d is not None:
            sigma_bg = float(self.survey.sigma_background_1d[0])
        else:
            sigma_bg = float(sigma_3d.detach().mean())

        # Edge -> face curl, cached per mesh (M6): pure topology, σ-independent,
        # reused across sources / frequencies / forwards.
        C_coo = self._curl_coo(mesh).to(sigma_3d.device)

        sources = self.survey.sources
        receivers = self.survey.receivers
        frequencies = self.survey.frequencies
        n_src = len(sources)
        n_rcv = len(receivers)
        n_freq = len(frequencies)

        # Per-component output buffers: comp_name -> list of (s, r, f)
        # nested lists holding scalar Tensors. Stack at the end to keep
        # autograd intact (no in-place writes on torch tensors).
        # Each receiver fills exactly one component slot; the others
        # are zero placeholders so we always emit consistent shapes.
        zero = torch.zeros((), dtype=torch.complex128)
        comp_buffers: dict[str, list[list[list[torch.Tensor]]]] = {}

        def _ensure_buffer(name: str) -> list[list[list[torch.Tensor]]]:
            if name not in comp_buffers:
                comp_buffers[name] = [
                    [[zero for _ in range(n_freq)] for _ in range(n_rcv)] for _ in range(n_src)
                ]
            return comp_buffers[name]

        # Pre-register the components actually requested so even
        # zero-only receivers produce stable output keys.
        for rcv in receivers:
            _ensure_buffer(rcv.component.value)

        # One explicit cache owns this forward only. All sources at one exact
        # frequency share a multi-column RHS and one factor. Different
        # frequencies retain distinct keys/factors; a block-diagonal matrix
        # would hide that physical accounting and prevent selective reuse.
        execution_cache = EMExecutionCache()
        matrix_mesh_version = mesh_fingerprint(mesh)
        matrix_material_version = material_fingerprint(sigma_3d)
        matrix_requires_gradient = bool(torch.is_grad_enabled() and sigma_3d.requires_grad)
        inductive_solver = self._solver
        inductive_solver_settings = solver_settings(inductive_solver)
        rhs_by_key: dict[AssemblyCacheKey, torch.Tensor] = {}
        primaries_per_freq: list[tuple[float, complex, list[torch.Tensor]]] = []
        resolved: dict[int, torch.Tensor] = {}  # f_idx -> E_secondary (n_edges, n_src)
        for f_idx, freq in enumerate(frequencies):
            omega = 2.0 * math.pi * float(freq)
            inv_iomega = -1.0 / (1j * omega)  # for B = -curl(E)/(iω)
            E_p_list = [
                _wholespace_dipole_primary_E(
                    mesh=mesh,
                    source=source,
                    sigma_bg=sigma_bg,
                    omega=omega,
                ).to(sigma_3d.device)  # analytic primary is built on CPU; move to model device
                for source in sources
            ]
            primaries_per_freq.append((omega, inv_iomega, E_p_list))
            E_p_stack = torch.stack(E_p_list, dim=1)  # (n_edges, n_src)
            assembly_key = AssemblyCacheKey(
                formulation_version="fdem3d-yee-secondary-v1",
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
                matrix, rhs = _fdem3d_secondary_solve(
                    mesh=mesh,
                    sigma_3d=sigma_3d,
                    sigma_bg=sigma_bg,
                    omega=omega,
                    E_primary=E_p_stack,
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

        for f_idx, (omega, inv_iomega, E_p_list) in enumerate(primaries_per_freq):
            E_s_stack = resolved[f_idx]
            for s_idx, source in enumerate(sources):
                E_total = E_p_list[s_idx] + E_s_stack[:, s_idx]

                # B = -curl(E_total) / (iω) on Yee faces.
                B_total = inv_iomega * torch.sparse.mm(
                    C_coo,
                    E_total.unsqueeze(-1),
                ).squeeze(-1)

                for r_idx, rcv in enumerate(receivers):
                    val = apply_yee_receiver_projection(
                        receiver_projections[r_idx],
                        E=E_total,
                        B=B_total,
                        receiver_index=0,
                    )
                    if rcv.component in (
                        FieldComponent.HX,
                        FieldComponent.HY,
                        FieldComponent.HZ,
                    ):
                        val = val / MU_0
                    buf = _ensure_buffer(rcv.component.value)
                    buf[s_idx][r_idx][f_idx] = val

        # Stack buffers into (n_src, n_rcv, n_freq) native-complex tensors,
        # one key per complex field component.
        data: dict[str, torch.Tensor] = {}
        for name, src_rows in comp_buffers.items():
            # Build (n_src, n_rcv, n_freq) by stacking.
            per_src = []
            for row in src_rows:  # row : list[ list[T] ] (n_rcv x n_freq)
                per_rcv = [torch.stack(freqs) for freqs in row]  # each (n_freq,)
                per_src.append(torch.stack(per_rcv))  # (n_rcv, n_freq)
            stacked = torch.stack(per_src)  # (n_src, n_rcv, n_freq)
            data[name] = stacked.contiguous()

        diagnostics = execution_cache.diagnostics
        diagnostics.projection_count = n_src * n_rcv * n_freq
        diagnostics.projected_element_count = n_src * n_rcv * n_freq
        metadata = solver_execution_metadata(diagnostics, inductive_solver_settings)
        execution_cache.close()
        return ForwardOutput(data=data, metadata=metadata)

    # ------------------------------------------------------------------
    # Edge-element (Nédélec/Whitney) branch: 3-D simplex UnstructuredMesh
    # ------------------------------------------------------------------

    def _edge_fem_assets(self, mesh) -> "_EdgeFemAssets":
        """σ-independent Whitney assets for ``mesh``, cached per mesh identity.

        Pure geometry/topology (assembly plan, PEC boundary mask, per-cell
        Whitney centroid-value and curl factors, receiver-binding centres);
        carries no autograd graph, so the same cache is reused across
        forwards, sources and frequencies, exactly like the Yee ``_C_coo``.
        """
        if self._edge_assets is None or self._edge_assets_mesh is not mesh:
            plan = build_edge_assembly_plan(mesh)
            cell_edge_ids, cell_edge_signs = mesh.cell_edges()
            g, _volume = barycentric_gradients(
                mesh.node_coords(),
                mesh.cell_nodes(),
            )
            # Whitney barycentre value + constant curl (float64; cast to
            # complex128 below for this operator's complex E-field system).
            w_centroid, curl_w = whitney_centroid_factors(g)  # (nc, 6, 3)
            self._edge_assets = _EdgeFemAssets(
                plan=plan,
                boundary_mask=boundary_edge_mask(mesh),
                edge_records=mesh.edge_records(),
                cell_edge_ids=cell_edge_ids,
                cell_edge_signs=cell_edge_signs,
                w_centroid=w_centroid.to(torch.complex128),
                curl_w=curl_w.to(torch.complex128),
                cell_centers=mesh.cell_centers(),
            )
            self._edge_assets_mesh = mesh
        return self._edge_assets

    def _forward_edge_fem(self, sigma: torch.Tensor, mesh) -> ForwardOutput:
        """Edge-element path for a 3-D simplex ``EdgeConnectivityMesh``.

        Symbol-for-symbol the same physics as the Yee path, on Whitney edge
        dofs (``dof_e = ∫_e E·dl`` along the canonical edge direction):

        1. ``E_p`` analytic wholespace magnetic-dipole primary at the scalar
           ``sigma_bg`` (the SAME closed form as the Yee path), projected to
           edge circulations by the midpoint rule;
        2. ``A(σ) e_s = -iω (M(σ) - M(σ_bg)) e_p`` with
           ``A = (1/μ₀)·K + iω·M(σ)`` (``exp(+iωt)``) and PEC pinning of the
           outer-boundary edges, via the implicit-VJP sparse solve, the Yee
           system scaled by ``1/μ₀``, so ``e_s`` solves the identical physics;
        3. secondary fields in each receiver's containing cell: the Whitney
           interpolant at the physical receiver for ``E_s`` and the constant
           per-cell ``B_s = -curl(E_s)/(iω)``;
        4. totals add the ANALYTIC primary (``E_p``, ``B_p = -curl(E_p)/(iω)``
           in closed form) at the exact receiver position;
        5. per-component sampling, identical ``data`` keys/shapes/convention
           as the Yee path.

        ``sigma`` is the flat ``(n_cells,)`` cell conductivity; ``sigma_bg``
        keeps the Yee semantics (``survey.sigma_background_1d[0]`` when given,
        else the cell mean). Differentiable in ``sigma`` end-to-end: the
        σ-path enters ``A`` (splu-bridge adjoint) and the RHS (autograd
        through the Whitney mass coefficients); the contributions sum
        automatically.
        """
        n_cells = int(mesh.n_cells)
        if sigma.dim() != 1 or int(sigma.shape[0]) != n_cells:
            raise FieldShapeError(
                "FDEM3D (edge-element path): sigma must be flat (n_cells,)",
                object_name="FDEM3D",
                field="sigma",
                expected=(n_cells,),
                actual=tuple(sigma.shape),
            )

        # Scalar background σ: same semantics as the Yee path (global
        # cell-mean fallback, deliberately detached: frozen primary-field
        # parameter, not a σ-gradient path).
        if self.survey.sigma_background_1d is not None:
            sigma_bg = float(self.survey.sigma_background_1d[0])
        else:
            sigma_bg = float(sigma.detach().mean())

        assets = self._edge_fem_assets(mesh)
        records = assets.edge_records

        sources = self.survey.sources
        receivers = self.survey.receivers
        frequencies = self.survey.frequencies
        n_src = len(sources)
        n_rcv = len(receivers)
        n_freq = len(frequencies)

        rcv_pos = torch.tensor(
            [rcv.position for rcv in receivers],
            dtype=torch.float64,
        )  # (n_rcv, 3)
        edge_projections: tuple[EdgeReceiverProjection, ...] = tuple(
            build_edge_receiver_projection(
                mesh,
                rcv_pos[index : index + 1],
                channel=receiver.component.value,
                layout="cartesian",
                n_sources=n_src,
            )
            for index, receiver in enumerate(receivers)
        )
        rcv_cells = torch.tensor(
            [projection.element_indices[0] for projection in edge_projections],
            dtype=torch.long,
        )
        rcv_ids = torch.tensor(
            [projection.local_edge_dof_indices[0] for projection in edge_projections],
            dtype=torch.long,
        )
        rcv_signs = torch.tensor(
            [projection.orientation_signs[0] for projection in edge_projections],
            dtype=torch.complex128,
        )
        rcv_basis = torch.tensor(
            [projection.basis_weights[0] for projection in edge_projections],
            dtype=torch.complex128,
        ).reshape(n_rcv, 6, 3)
        rcv_curl_w = assets.curl_w[rcv_cells][:, :, (1, 2, 0)]  # public (x,y,z)

        # Same per-component output buffering as the Yee path (zero
        # placeholders keep shapes/keys identical when components mix).
        zero = torch.zeros((), dtype=torch.complex128)
        comp_buffers: dict[str, list[list[list[torch.Tensor]]]] = {}

        def _ensure_buffer(name: str) -> list[list[list[torch.Tensor]]]:
            if name not in comp_buffers:
                comp_buffers[name] = [
                    [[zero for _ in range(n_freq)] for _ in range(n_rcv)] for _ in range(n_src)
                ]
            return comp_buffers[name]

        for rcv in receivers:
            _ensure_buffer(rcv.component.value)

        execution_cache = EMExecutionCache()
        matrix_mesh_version = mesh_fingerprint(mesh)
        matrix_material_version = material_fingerprint(sigma)
        matrix_requires_gradient = bool(torch.is_grad_enabled() and sigma.requires_grad)
        inductive_solver = self._solver
        inductive_solver_settings = solver_settings(inductive_solver)
        rhs_by_key: dict[AssemblyCacheKey, torch.Tensor] = {}
        for f_idx, freq in enumerate(frequencies):
            omega = 2.0 * math.pi * float(freq)
            inv_iomega = -1.0 / (1j * omega)  # for B = -curl(E)/(iω)

            # One A(σ, ω) per frequency: all sources share it; the plan and
            # the PEC mask are reused, only the COO values are re-assembled.
            e_p_stack = torch.stack(
                [
                    _edge_primary_dofs(
                        records,
                        source=source,
                        sigma_bg=sigma_bg,
                        omega=omega,
                    )
                    for source in sources
                ],
                dim=1,
            )  # (n_edges, n_src)
            assembly_key = AssemblyCacheKey(
                formulation_version="fdem3d-whitney-secondary-v1",
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
                matrix, rhs = _edge_fem_secondary_solve(
                    assets.plan,
                    assets.boundary_mask,
                    sigma_cells=sigma,
                    sigma_bg=sigma_bg,
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

            for s_idx, source in enumerate(sources):
                dof_s = e_s_stack[:, s_idx]
                local = rcv_signs * dof_s[rcv_ids]  # (n_rcv, 6)
                E_s = torch.einsum("re,rek->rk", local, rcv_basis)
                B_s = inv_iomega * torch.einsum("re,rek->rk", local, rcv_curl_w)

                # Analytic primary at the exact receiver positions (the Yee
                # path samples its grid primary instead; same wholespace
                # closed form, same sigma_bg).
                E_p = _wholespace_dipole_E_at_points(
                    rcv_pos,
                    source=source,
                    sigma_bg=sigma_bg,
                    omega=omega,
                )
                B_p = _wholespace_dipole_B_at_points(
                    rcv_pos,
                    source=source,
                    sigma_bg=sigma_bg,
                    omega=omega,
                )
                E_total = E_p + E_s
                B_total = B_p + B_s

                for r_idx, rcv in enumerate(receivers):
                    val = pick_field_component(
                        E_total[r_idx],
                        B_total[r_idx],
                        rcv.component,
                    )
                    buf = _ensure_buffer(rcv.component.value)
                    buf[s_idx][r_idx][f_idx] = val

        data: dict[str, torch.Tensor] = {}
        for name, src_rows in comp_buffers.items():
            per_src = []
            for row in src_rows:
                per_rcv = [torch.stack(freqs) for freqs in row]
                per_src.append(torch.stack(per_rcv))
            data[name] = torch.stack(per_src).contiguous()

        diagnostics = execution_cache.diagnostics
        diagnostics.projection_count = n_src * n_rcv * n_freq
        diagnostics.projected_element_count = n_src * n_rcv * n_freq
        metadata = solver_execution_metadata(diagnostics, inductive_solver_settings)
        execution_cache.close()
        return ForwardOutput(data=data, metadata=metadata)


# ----------------------------------------------------------------------
# Wholespace magnetic-dipole primary E-field on Yee edges
# ----------------------------------------------------------------------


def _wholespace_dipole_primary_E(
    mesh: TensorMesh,
    source: MagneticDipoleSource,
    sigma_bg: float,
    omega: float,
) -> torch.Tensor:
    """
    Analytic wholespace magnetic-dipole primary E-field on Yee E-edges.

    Implements the closed-form Ward & Hohmann (1988, §4) primary E-field
    for a magnetic dipole of moment vector ``m = moment * orientation``
    at ``source.position = (src_x, src_y, src_z)`` embedded in a
    homogeneous wholespace of conductivity ``sigma_bg``::

        E_vec(r) = -(i*omega*MU_0*m / 4pi) * (n_hat x r) / R^3
                                           * (1 + i*k*R) * exp(-i*k*R)

    where ``n_hat`` is the unit moment direction, ``r = field - source``
    is the offset from source to field point, ``R = |r|``, and the
    wavenumber ``k = sqrt(-i*omega*MU_0*sigma_bg)`` is taken on the
    principal branch so ``Re(k) > 0`` (decaying away from the source).

    The field is sampled at Yee edge midpoints and packed into a flat
    ``complex128`` array of length ``sum(yee_edge_count(mesh))``, in the
    canonical (Ex, Ey, Ez) family order with x-fastest flat indexing
    within each family, matching ``yee_curl_e_to_f`` / ``yee_curl_curl_assemble``.

    Edge-midpoint coordinates come from the mesh's own ``node_lines`` /
    ``center_lines`` (origin-aware; for a zero-origin mesh the nodes sit at
    integer multiples of the spacing, bitwise the historical arrays):

    - Ex midpoint at edge ``(i, j, k)``:  ``((i+0.5)*dx, j*dy, k*dz)``,
      with ``i in [0, nx)``, ``j in [0, ny+1)``, ``k in [0, nz+1)``.
    - Ey midpoint at edge ``(i, j, k)``:  ``(i*dx, (j+0.5)*dy, k*dz)``.
    - Ez midpoint at edge ``(i, j, k)``:  ``(i*dx, j*dy, (k+0.5)*dz)``.

    For a VMD (``orientation = (0, 0, 1)``), ``n_hat x r`` has zero
    z-component, so the Ez block is identically zero.

    Singularity handling: edge midpoints with ``R == 0`` (i.e. exactly
    coincident with the source) return ``0+0j``, same convention as
    the ``vmd_primary_e_on_edges``. Callers should place sources at
    nodes or cell centres (e.g. half-cell offsets) rather than on edge
    midpoints to avoid this.

    Args:
        mesh: 3D :class:`geobrain.mesh.TensorMesh` with ``shape=(nz, ny, nx)``
            and ``spacing=(dz, dy, dx)``.
        source: :class:`MagneticDipoleSource` carrying ``position`` (x, y, z),
            ``orientation`` (unit 3-vector) and ``magnetic_moment_am2`` (A·m²).
        sigma_bg: Background conductivity of the wholespace in S/m. Must be
            non-negative; ``sigma_bg = 0`` degenerates to ``k = 0`` (static
            limit) and is allowed.
        omega: Angular frequency ``2*pi*f`` in rad/s.

    Returns:
        Flat ``complex128`` tensor of length ``n_edges``.
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "_wholespace_dipole_primary_E requires a 3D TensorMesh",
            object_name="_wholespace_dipole_primary_E",
            field="mesh.n_dim",
            expected="3",
            actual=mesh.n_dim,
        )

    nz, ny, nx = mesh.shape
    dz, dy, dx = mesh.spacing

    src_x, src_y, src_z = source.position
    orient = torch.tensor(source.orientation, dtype=torch.float64)
    # Normalise orientation in case caller didn't (e.g. (0, 0, 1) is
    # already unit but (1, 1, 0) is not).
    orient_norm = orient.norm()
    if float(orient_norm) <= 0.0:
        raise GeoBrainError(
            "MagneticDipoleSource.orientation must be a non-zero vector",
            object_name="_wholespace_dipole_primary_E",
            field="orientation",
            expected="non-zero vector",
            actual=tuple(source.orientation),
        )
    n_hat = orient / orient_norm
    moment = float(source.magnetic_moment_am2)

    n_ex, n_ey, n_ez = yee_edge_count(mesh)

    # Wavenumber on the principal branch so Re(k) >= 0.
    # k^2 = -i * omega * MU_0 * sigma_bg  =>  k = sqrt(-i * omega * MU_0 * sigma_bg).
    k_sq = torch.complex(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-omega * MU_0 * sigma_bg, dtype=torch.float64),
    )
    k = torch.sqrt(k_sq)
    # E = -iωA under exp(+iωt): the prefactor is -i·ω·μ₀·m/(4π) (audit E1).
    prefactor = torch.complex(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-omega * MU_0 * moment / (4.0 * math.pi), dtype=torch.float64),
    )

    out = torch.zeros(n_ex + n_ey + n_ez, dtype=torch.complex128)

    def _eval_axis(
        xs: torch.Tensor,
        ys: torch.Tensor,
        zs: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """
        Evaluate E_vec[axis] at the meshgrid of (xs, ys, zs).

        ``xs``, ``ys``, ``zs`` are 1-D coordinate arrays along x, y, z
        respectively. The returned 1-D tensor is flattened in i-fastest
        order (i.e. ``i + j*len(xs) + k*len(xs)*len(ys)``), matching the
        Yee edge layout in ``curl_curl.py``.
        """
        # Use indexing="ij" on (xs, ys, zs) -> grids shaped (nx, ny, nz),
        # then permute so the resulting flat layout is i-fastest, j-next,
        # k-slowest. With contiguous (nx, ny, nz) ij-grids the natural
        # C-order ravel is k-fastest (last axis), so we permute to
        # (nz, ny, nx) and ravel: that yields i-fastest.
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
        dx_off = xx - src_x
        dy_off = yy - src_y
        dz_off = zz - src_z
        R = torch.sqrt(dx_off * dx_off + dy_off * dy_off + dz_off * dz_off)

        # (n_hat x r) component for `axis`:
        #   (n x r)_x = n_y * dz - n_z * dy
        #   (n x r)_y = n_z * dx - n_x * dz
        #   (n x r)_z = n_x * dy - n_y * dx
        if axis == 0:
            ncross = n_hat[1] * dz_off - n_hat[2] * dy_off
        elif axis == 1:
            ncross = n_hat[2] * dx_off - n_hat[0] * dz_off
        else:
            ncross = n_hat[0] * dy_off - n_hat[1] * dx_off

        E_axis = torch.zeros_like(R, dtype=torch.complex128)
        safe = R > 0.0
        if bool(safe.any()):
            Rs = R[safe]
            kR = k * Rs
            E_axis[safe] = (
                prefactor
                * ncross[safe].to(torch.complex128)
                / (Rs**3).to(torch.complex128)
                * (1.0 + 1j * kR)
                * torch.exp(-1j * kR)
            )

        # Flatten i-fastest: permute (nx, ny, nz) -> (nz, ny, nx), then
        # C-order ravel gives k-slowest / j-mid / i-fastest.
        return E_axis.permute(2, 1, 0).reshape(-1)

    # Coordinate arrays for each edge family, from the mesh's axis lines,
    # origin-aware. The engine mesh here is uniform (enforced at the
    # operator entry) with zero origin, and node_lines/center_lines take
    # the multiplicative branch on uniform meshes, so these ARE the
    # historical ``arange()*d`` / ``(arange()+0.5)*d`` arrays bitwise.
    # Engine frame is ``(nz, ny, nx)`` so the lines unpack as (z, y, x).
    nl_z, nl_y, nl_x = mesh.node_lines()
    cl_z, cl_y, cl_x = mesh.center_lines()

    # Ex edges live at x = (i+0.5)*dx, y = j*dy, z = k*dz with
    #   i in [0, nx), j in [0, ny+1), k in [0, nz+1).
    out[:n_ex] = _eval_axis(cl_x, nl_y, nl_z, axis=0)

    # Ey edges live at x = i*dx, y = (j+0.5)*dy, z = k*dz with
    #   i in [0, nx+1), j in [0, ny), k in [0, nz+1).
    out[n_ex : n_ex + n_ey] = _eval_axis(nl_x, cl_y, nl_z, axis=1)

    # Ez edges live at x = i*dx, y = j*dy, z = (k+0.5)*dz with
    #   i in [0, nx+1), j in [0, ny+1), k in [0, nz).
    out[n_ex + n_ey :] = _eval_axis(nl_x, nl_y, cl_z, axis=2)

    return out


# ----------------------------------------------------------------------
# Secondary-field sparse solve (delegates to MT3D's identical machinery)
# ----------------------------------------------------------------------


def _fdem3d_secondary_solve(
    mesh: TensorMesh,
    *,
    sigma_3d: torch.Tensor,
    sigma_bg: float,
    omega: float,
    E_primary: torch.Tensor,
    return_system: bool = False,
) -> torch.Tensor:
    """
    Solve ``A(σ) E_s = -iω·μ₀·(σ_edge - σ_bg_edge) E_p`` for the FDEM3D
    secondary field.

    Thin wrapper around MT3D's :func:`_secondary_field_solve`, the
    underlying equation is identical to the natural-source case; the
    only structural difference is that FDEM3D takes a *scalar* uniform
    background ``sigma_bg`` (S/m) rather than MT3D's 1D layered
    ``sigma_bg_1d``. We bridge the two by wrapping the scalar as a
    1-element 1D tensor and passing an empty ``layer_thicknesses``,
    which tells :func:`_secondary_field_solve` to treat the entire
    domain as a single half-space.

    Args:
        mesh: 3D :class:`TensorMesh` matching the curl-curl assembler convention.
        sigma_3d: Cell-centred 3D conductivity in S/m, real ``float64``, shape
            ``mesh.shape``.
        sigma_bg: Scalar uniform background conductivity in S/m. Used both for
            the wholespace primary (upstream) and the σ-difference RHS here.
        omega: Angular frequency in rad/s.
        E_primary: ``(n_edges,)`` complex128 primary E-field on Yee edges, e.g.
            from :func:`_wholespace_dipole_primary_E`.

    Returns:
        Complex128 tensor of shape ``(n_edges,)``, the secondary E-field.
    """
    sigma_bg_1d = torch.tensor([sigma_bg], dtype=torch.float64)
    layer_thicknesses = torch.tensor([], dtype=torch.float64)
    return _secondary_field_solve(
        mesh=mesh,
        sigma_3d=sigma_3d,
        sigma_bg_1d=sigma_bg_1d,
        omega=omega,
        E_primary=E_primary,
        layer_thicknesses=layer_thicknesses,
        return_system=return_system,
    )


# ----------------------------------------------------------------------
# Component-aware receiver extraction
# ----------------------------------------------------------------------


def _extract_component_at_receiver(
    mesh: TensorMesh,
    *,
    E: torch.Tensor,
    B: torch.Tensor,
    position: tuple[float, float, float],
    component: FieldComponent,
) -> torch.Tensor:
    """
    Strict multilinear extraction of one field component at ``(x, y, z)``.

    Uses the same immutable origin-aware Yee plan as the full FDEM operator:

    - Coordinates must lie in the closed support of the selected staggered
      component (with the explicit exact-mesh-boundary convention); no outside
      coordinate is clamped or extrapolated.
    - Returns a **single** complex scalar for the requested ``component``,
      not the full ``(Ex, Ey, Hx, Hy, Hz)`` quintuple. ``EX/EY/EZ`` index
      into ``E`` (Yee edges); ``BX/BY/BZ`` index into ``B`` (Yee faces);
      ``HX/HY/HZ`` are obtained by dividing the corresponding ``B`` value
      by ``μ₀``.

    The Yee anchor table: each component lives on either the cell-centre
    or the node along each axis:

    ====  ==============  ==============  ==============
    Comp  x-anchor        y-anchor        z-anchor
    ====  ==============  ==============  ==============
    Ex    cell-centre     node            node
    Ey    node            cell-centre     node
    Ez    node            node            cell-centre
    Bx    node            cell-centre     cell-centre
    By    cell-centre     node            cell-centre
    Bz    cell-centre     cell-centre     node
    ====  ==============  ==============  ==============

    Args:
        mesh: 3D :class:`TensorMesh`; ``mesh.shape == (nz, ny, nx)`` per the
            curl-curl assembler convention.
        E: Complex ``(n_edges,)`` E-field on Yee edges (``[Ex; Ey; Ez]``).
        B: Complex ``(n_faces,)`` B-field on Yee faces (``[Bx; By; Bz]``).
        position: Public ``(x, y, z)`` tuple in metres.
        component: :class:`FieldComponent` selector. One of ``EX/EY/EZ`` (read from
            ``E``), ``BX/BY/BZ`` (read from ``B``), or ``HX/HY/HZ`` (read
            from ``B`` and divided by ``μ₀``).

    Returns:
        Complex128 0-d tensor for the requested component. Autograd
        flows through ``E`` / ``B`` (weighted gathers + a constant
        scaling for ``HX/HY/HZ``).
    """
    if mesh.n_dim != 3:
        raise GeoBrainError(
            "_extract_component_at_receiver requires a 3D mesh",
            object_name="_extract_component_at_receiver",
            field="mesh.n_dim",
            expected="3",
            actual=mesh.n_dim,
        )
    projection = build_yee_receiver_projection(
        mesh,
        torch.tensor([position], dtype=torch.float64),
        channel=component.value,
        layout="cartesian",
        n_sources=1,
    )
    val = apply_yee_receiver_projection(
        projection,
        E=E,
        B=B,
        receiver_index=0,
    ).to(torch.complex128)
    if component in (FieldComponent.HX, FieldComponent.HY, FieldComponent.HZ):
        val = val / MU_0  # H = B / μ₀
    return val


# ----------------------------------------------------------------------
# Edge-element (Nédélec/Whitney) branch helpers
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _EdgeFemAssets(EdgeFemAssetsBase):
    """σ-independent Whitney assets of one mesh (see ``_edge_fem_assets``).

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
        cell_centers: ``(n_cells, 3)`` float64, receiver-binding points.
    """


def _dipole_geometry(
    points: torch.Tensor,
    source: MagneticDipoleSource,
    sigma_bg: float,
    omega: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Shared geometry/wavenumber prep of the wholespace dipole closed forms.

    Returns ``(r, R, n_hat, k, moment)`` with ``r = points - source`` of shape
    ``(n, 3)`` float64, ``R = |r|`` ``(n,)``, the unit moment direction, and
    the principal-branch wavenumber ``k = sqrt(-iωμ₀σ_bg)`` (``Re k ≥ 0``,
    identical branch to :func:`_wholespace_dipole_primary_E`).
    """
    pts = points.to(torch.float64)
    if pts.dim() != 2 or int(pts.shape[1]) != 3:
        raise GeoBrainError(
            "wholespace dipole evaluation expects points of shape (n, 3)",
            object_name="_dipole_geometry",
            field="points",
            expected="(n, 3)",
            actual=tuple(pts.shape),
        )
    orient = torch.tensor(source.orientation, dtype=torch.float64)
    orient_norm = orient.norm()
    if float(orient_norm) <= 0.0:
        raise GeoBrainError(
            "MagneticDipoleSource.orientation must be a non-zero vector",
            object_name="_dipole_geometry",
            field="orientation",
            expected="non-zero vector",
            actual=tuple(source.orientation),
        )
    n_hat = orient / orient_norm
    src = torch.tensor(source.position, dtype=torch.float64)
    r = pts - src
    R = torch.linalg.norm(r, dim=1)
    k_sq = torch.complex(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-omega * MU_0 * sigma_bg, dtype=torch.float64),
    )
    k = torch.sqrt(k_sq)
    return r, R, n_hat, k, float(source.magnetic_moment_am2)


def _wholespace_dipole_E_at_points(
    points: torch.Tensor,
    *,
    source: MagneticDipoleSource,
    sigma_bg: float,
    omega: float,
) -> torch.Tensor:
    """Analytic wholespace magnetic-dipole primary E at arbitrary points.

    The SAME Ward & Hohmann (1988, §4) closed form as
    :func:`_wholespace_dipole_primary_E`, which is this expression sampled at
    Yee edge midpoints: evaluated at an arbitrary ``(n, 3)`` point set::

        E(r) = -(i·omega·MU_0·m / 4π) · (n_hat × r) / R³
                                      · (1 + i·k·R) · exp(-i·k·R)

    with the principal-branch ``k = sqrt(-i·omega·MU_0·sigma_bg)``
    (``exp(+iωt)`` convention, ``Re(k) ≥ 0``). Points with ``R == 0`` return
    ``0+0j``: the same singularity convention as the Yee sampler.

    Args:
        points: ``(n, 3)`` field-point coordinates in metres.
        source: the :class:`MagneticDipoleSource` (position/orientation/moment).
        sigma_bg: wholespace background conductivity in S/m (``≥ 0``).
        omega: angular frequency in rad/s.

    Returns:
        ``(n, 3)`` complex128 E-field vectors.
    """
    r, R, n_hat, k, moment = _dipole_geometry(points, source, sigma_bg, omega)
    # E = -iωA under exp(+iωt): the prefactor is -i·ω·μ₀·m/(4π) (audit E1).
    prefactor = torch.complex(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-omega * MU_0 * moment / (4.0 * math.pi), dtype=torch.float64),
    )
    ncross = torch.linalg.cross(n_hat.expand_as(r), r, dim=-1)  # (n, 3)
    E = torch.zeros(r.shape, dtype=torch.complex128)
    safe = R > 0.0
    if bool(safe.any()):
        Rs = R[safe]
        kR = k * Rs
        radial = prefactor / (Rs**3).to(torch.complex128) * (1.0 + 1j * kR) * torch.exp(-1j * kR)
        E[safe] = ncross[safe].to(torch.complex128) * radial.unsqueeze(-1)
    return E


def _wholespace_dipole_B_at_points(
    points: torch.Tensor,
    *,
    source: MagneticDipoleSource,
    sigma_bg: float,
    omega: float,
) -> torch.Tensor:
    """Analytic wholespace magnetic-dipole primary B at arbitrary points.

    Exactly ``B = -curl(E)/(iω)`` of :func:`_wholespace_dipole_E_at_points`
    (Faraday under ``exp(+iωt)``, the same relation both mesh paths apply to
    their discrete fields), in closed form::

        B(r) = (MU_0·m / 4π) · exp(-i·k·R) / R³
               · [ (3 + 3·i·k·R - k²R²)·(n_hat·r_hat)·r_hat
                   - (1 + i·k·R - k²R²)·n_hat ]

    obtained from ``curl(f(R)·(n_hat × r)) = 2 f n_hat + f' R (n_hat -
    (n_hat·r_hat) r_hat)`` with ``f(R) = (1 + ikR)·exp(-ikR)/R³``. Same
    wavenumber branch and ``R == 0 → 0`` convention as the E-field form; the
    test battery pins the identity against a finite-difference curl of the
    E closed form.

    Args:
        points: ``(n, 3)`` field-point coordinates in metres.
        source: the :class:`MagneticDipoleSource` (position/orientation/moment).
        sigma_bg: wholespace background conductivity in S/m (``≥ 0``).
        omega: angular frequency in rad/s.

    Returns:
        ``(n, 3)`` complex128 B-field vectors (tesla).
    """
    r, R, n_hat, k, moment = _dipole_geometry(points, source, sigma_bg, omega)
    B = torch.zeros(r.shape, dtype=torch.complex128)
    safe = R > 0.0
    if bool(safe.any()):
        Rs = R[safe]
        r_hat = r[safe] / Rs.unsqueeze(-1)  # (m, 3)
        kR = k * Rs  # (m,) complex
        kR2 = kR * kR
        n_dot_rhat = (r_hat * n_hat).sum(dim=-1).to(torch.complex128)
        # Companion of the (fixed-sign) E closed form: B = -curl(E)/(iω)
        # now carries a POSITIVE leading factor, so the k → 0 limit is the
        # textbook magnetostatic dipole field (audit E1).
        radial = (
            (MU_0 * moment / (4.0 * math.pi)) * torch.exp(-1j * kR) / (Rs**3).to(torch.complex128)
        )
        term_r = (3.0 + 3j * kR - kR2) * n_dot_rhat  # (m,)
        term_n = 1.0 + 1j * kR - kR2  # (m,)
        B[safe] = radial.unsqueeze(-1) * (
            term_r.unsqueeze(-1) * r_hat.to(torch.complex128)
            - term_n.unsqueeze(-1) * n_hat.to(torch.complex128)
        )
    return B


def _edge_primary_dofs(
    records: EdgeRecords,
    *,
    source: MagneticDipoleSource,
    sigma_bg: float,
    omega: float,
) -> torch.Tensor:
    """Project the analytic primary E onto Whitney edge dofs.

    The lowest-order Nédélec dof is the tangential circulation
    ``dof_e = ∫_e E·dl`` along the edge's canonical direction; the midpoint
    rule gives ``dof_e ≈ E(midpoint_e)·tangent_e·length_e``. It is exact for
    fields affine along the edge and carries an ``O(L_e²)``-relative
    truncation from the field's curvature otherwise, adequate whenever edge
    lengths resolve the source distance and skin depth, the same resolution
    regime the Yee midpoint sampling assumes.

    Args:
        records: the mesh's :class:`EdgeRecords` (midpoints, unit tangents,
            lengths along the canonical small-node → large-node direction).
        source: the :class:`MagneticDipoleSource`.
        sigma_bg: wholespace background conductivity in S/m.
        omega: angular frequency in rad/s.

    Returns:
        ``(n_edges,)`` complex128 primary edge circulations.
    """
    E_mid = _wholespace_dipole_E_at_points(
        records.midpoint[:, (1, 2, 0)],
        source=source,
        sigma_bg=sigma_bg,
        omega=omega,
    )  # (ne, 3)
    tangent_xyz = records.tangent[:, (1, 2, 0)].to(torch.complex128)
    tangential = (E_mid * tangent_xyz).sum(dim=1)
    return tangential * records.length.to(torch.complex128)


def _edge_fem_secondary_solve(
    plan: EdgeAssemblyPlan,
    boundary_mask: torch.Tensor,
    *,
    sigma_cells: torch.Tensor,
    sigma_bg: float,
    omega: float,
    e_primary: torch.Tensor,
    return_system: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Solve the Whitney secondary system for one frequency.

    The edge-element analogue of :func:`_fdem3d_secondary_solve`, the same
    ``exp(+iωt)`` physics scaled by ``1/μ₀``::

        A(σ) e_s = -iω (M(σ) - M(σ_bg)) e_p,
        A(σ)     = (1/μ₀)·K + iω·M(σ),

    where ``K``/``M`` are the Whitney curl-curl and mass operators of
    ``plan``. The RHS is the OPERATOR-DIFFERENCE form, one mass assembly at
    coefficient ``σ - σ_bg`` applied to ``e_p`` (identically zero when
    ``σ ≡ σ_bg``). PEC truncation: tangential ``E_s = 0`` on the outer
    boundary, imposed symmetrically, boundary rows AND columns of the COO
    values are zeroed (``torch.where``, keeping the values differentiable)
    with appended unit diagonal entries, and the boundary RHS rows are
    zeroed.

    Args:
        plan: the mesh's :class:`EdgeAssemblyPlan`.
        boundary_mask: ``(n_edges,)`` bool from :func:`boundary_edge_mask`.
        sigma_cells: ``(n_cells,)`` float64 cell conductivities (autograd-live).
        sigma_bg: scalar background conductivity in S/m.
        omega: angular frequency in rad/s.
        e_primary: ``(n_edges,)`` or ``(n_edges, k)`` complex128 primary edge
            circulations, multiple sources solve against ONE factorisation.

    Returns:
        ``e_s`` with the same shape as ``e_primary``, complex128.
    """
    n_cells = plan.n_cells
    delta_sigma = sigma_cells - sigma_bg  # (nc,) float64
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
    # Compatibility leaf path: callers that do not orchestrate an explicit
    # cache keep the historical one-shot behavior.
    e_s = solve_pec_pinned_edge_system(
        plan,
        boundary_mask,
        mass_coeff=mass_coeff,
        stiffness_coeff=stiff_coeff,
        rhs=q,
        pin_plan=pin_plan,
    )
    return e_s if e_primary.dim() == 2 else e_s.squeeze(-1)
