# pyright: reportPrivateImportUsage=false
"""
3D Time-domain EM (TEM) operator.

Step-off vertical magnetic dipole (VMD) over a 3D conductivity model; BDF1 → BDF2
log-substepped time loop on the Yee curl-curl operator, solved per substep with
``sparse_linear_solve_with_adjoint``. Output is per-receiver, per-component,
per-time-gate ``dB/dt`` (V/m^2) or ``E`` (V/m), real-valued.

Differentiability:
    ``IMPLICIT_VJP`` through ``sigma``, the per-substep linear system
    ``A(σ) x^{n+1} = b(x^n, ...)`` reuses the splu-bridge adjoint, and the σ-paths
    through the RHS and edge-averaging flow through standard torch autograd.

Time convention:
    The curl-curl assembler uses ``exp(+iωt)`` in the frequency domain, but TEM3D
    operates entirely in the *time* domain (real-valued arithmetic; no complex
    sqrt). Step-off responses are sign-aligned by physics.

Pipeline:

- The dataclass surface: :class:`MagneticDipoleSource`,
  :class:`TEM3DReceiver`, :class:`TEM3DSurvey`.
- A constructible operator whose ``__init__`` pre-assembles all σ-independent
  quantities (``K``, ``A_e``, per-source ``s_e``, BDF time schedule, per-receiver
  Yee anchors).
- The full BDF1+BDF2 time loop driving ``_forward``: bare-mass
  ``M = diag(σ_edge)``, step-off IC ``E^0 = (s_e / V_edge) / σ_edge``, per-step
  solve via :func:`sparse_linear_solve_with_adjoint`, ``dB/dt`` receiver sampling
  via ``-C @ E`` at the requested gates.

Surface mirrors :class:`SIP` and :class:`FDEM3D`.

Dual mesh paths: the capability contract is ``ConnectivityMesh``. A mesh
that also declares ``StructuredMesh`` (TensorMesh) takes the original Yee
staggered-grid path bit-identically; a mesh declaring
``EdgeConnectivityMesh`` (3-D simplex-built UnstructuredMesh) takes the
lowest-order Nédélec/Whitney edge-element path with the SAME step-off
formulation (``e_0 = 0`` + one-shot impulse ``q_1 = (1/Δt_0)·K_w(1/μ₀)·a_e``,
backward-Euler on the shared log-substep schedule, ``dB/dt = -curl E``
gate sampling). Anything else (OctreeMesh) raises, octree inductive EM
is not implemented.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from geobrain.physics.em.config import EMExecutionConfig
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from geobrain.core import ForwardContext, GeoBrainError, ModelState, ForwardOperator, ForwardOutput
from geobrain.mesh import TensorMesh
from geobrain.mesh.capabilities import (
    ConnectivityMesh,
    EdgeConnectivityMesh,
    StructuredMesh,
)
from geobrain.core.differentiability import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
)
from geobrain.physics.em.conventions import MU_0
from geobrain.physics.em.results import FieldComponent
from geobrain.physics.em.surveys import (
    EMReceiver,
    MagneticDipoleSource,
    TimeDomainSurvey,
)
from geobrain.core.adjoint import (
    SparseFactor,
    factorize_sparse,
    sparse_linear_solve_with_adjoint,
)
from geobrain.core.linalg import SparseFactorSolver
from geobrain.mesh import require_field_matches_mesh
from geobrain.physics.em._engine import (
    AssemblyCacheKey,
    EMExecutionCache,
    exact_float_token,
    material_fingerprint,
    mesh_fingerprint,
    recording_execution_metadata,
    resolve_engine_solver,
    resolve_inductive_mesh_path,
    solver_execution_metadata,
    solver_factor_key,
    solver_settings,
)
from geobrain.physics.em._engine.recording import (
    RecordingPolicy,
    execute_recorded_recurrence,
    prepare_recording,
)
from geobrain.physics.em.errors import EMCapabilityError
from geobrain.physics.em.numerics.edge_element import (
    assemble_pec_pinned_operator,
    barycentric_gradients,
    boundary_edge_mask,
    build_edge_assembly_plan,
    build_pec_pin_plan,
    edge_operator_matvec,
    pec_zero_rhs,
    pick_field_component,
    whitney_centroid_factors,
)
from geobrain.physics.em.numerics.finite_volume import (
    assemble_cell_to_edge_averaging,
    yee_edge_count,
    yee_face_count,
)
from geobrain.physics.em.numerics.finite_volume.curl_curl import (
    yee_curl_e_to_f,
)
from geobrain.physics.em.numerics.finite_volume.axis_convention import (
    swap_xy_cell_field,
    to_engine_mesh,
)
from geobrain.physics.em.numerics.time_stepping.schedule import (
    build_log_substep_schedule,
)
from geobrain.physics.em.time_domain.tem3d_sources import (
    build_vmd_step_off_source,
)
from geobrain.physics.em.receivers import (
    EdgeReceiverProjection,
    ReceiverLayout,
    YeeReceiverProjection,
    build_edge_receiver_projection,
    build_yee_receiver_projection,
)
from geobrain.physics.em.time_domain._tem3d_numerics import (  # noqa: F401  re-export: split section
    _TemEdgeFemAssets,
    _build_edge_volume_diagonal,
    _build_face_dual_volume,
    _build_vmd_vector_potential_on_edges,
    _edge_positions,
    _edge_vmd_vector_potential_dofs,
    _safe_half_sum_uniform,
    _vmd_vector_potential_at_points,
)

_MU0 = MU_0  # platform canon, single-sourced in core.constants

__all__ = [
    "MagneticDipoleSource",
    "TEM3D",
    "TEM3DReceiver",
    "TEM3DSurvey",
]


# ---------------------------------------------------------------------------
# Edge-volume helper (geometry-only; needed to convert the raw mimetic
# ``s_e`` into the bare-mass row units the BDF stepper
# consumes). Mirrors ``_build_edge_volume_diagonal`` (operator.py
# L286-L342) restricted to the uniform-spacing TensorMesh.
# ---------------------------------------------------------------------------












# ---------------------------------------------------------------------------
# Source / receiver dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TEM3DReceiver(EMReceiver):
    """TEM3D point receiver: adds a field-component selector.

    Extends :class:`EMReceiver` (which carries ``position``) with a
    :class:`FieldComponent` selector. Defaults to ``BZ`` (the canonical
    single-channel ground-TEM observable).

    Semantic note:
    For TEM step-off, the operator outputs the **time derivative** of
    the magnetic field (``dB/dt``) for ``BX/BY/BZ`` channels, that is
    what TEM instruments physically measure at time gates. ``EX/EY/EZ``
    channels report the raw E-field. The :class:`FieldComponent` enum
    name (``BZ``) labels the physical field whose derivative is reported;
    output data uses the explicit ``"dbdt_z"`` channel.

    Attributes:
        position: receiver location [m].
        component: recorded component (``'dbdt_z'``...).
    """

    component: FieldComponent = FieldComponent.BZ


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TEM3DSurvey(TimeDomainSurvey):
    """
    3D TEM survey: VMD source(s) + component receiver(s) + time gates.

    Positions are public ``(x, y, z)`` metres (z depth, positive down, on
    the mesh datum). Transmitter positions are SNAPPED to the nearest cell
    in xy and the nearest node in z; receivers are strict multilinear,
    never snapped.

    Inherits ``sources`` and ``receivers`` from :class:`TimeDomainSurvey`.
    The base class's ``times`` field is left empty here, TEM3D uses a
    dedicated ``time_gates`` field whose semantics (strictly positive,
    strictly ascending, drives the BDF1+BDF2 log-substepped time loop)
    differ from the generic "waveform sample times" implied by
    ``TimeDomainSurvey.times``.

    Args:
        sources: Tuple of :class:`MagneticDipoleSource` transmitters.
        receivers: Tuple of :class:`TEM3DReceiver` observation points.
        time_gates: Tuple of strictly-positive, strictly-ascending observation
            times in seconds. The BDF1+BDF2 loop substeps log-uniformly
            between gates; ``n_substeps_per_decade`` controls the step
            density.
        n_substeps_per_decade: Substeps per logarithmic decade between consecutive gates.
            Default ``8`` matches the default. Must be ``>= 1``.

    Notes:
    the TEM3DSurvey carries the ``mesh`` directly and pre-resolves
    per-receiver Yee indices at construction. the pattern (matching
    FDEM3D/MT3D) keeps the mesh on the operator instance, and the
    operator resolves Yee indices on the fly during ``_forward``.
    """

    time_gates: tuple[float, ...] = ()
    n_substeps_per_decade: int = 8

    def __post_init__(self) -> None:
        # Chain to EMSurvey's abstract-instance guard.
        super().__post_init__()

        if not self.time_gates:
            raise GeoBrainError("TEM3DSurvey.time_gates must be non-empty")
        for k, t in enumerate(self.time_gates):
            if not (t > 0):
                raise GeoBrainError(
                    f"TEM3DSurvey.time_gates[{k}] = {t} must be strictly positive",
                )
            if k > 0 and not (t > self.time_gates[k - 1]):
                raise GeoBrainError(
                    f"TEM3DSurvey.time_gates must be strictly ascending; "
                    f"index {k - 1} = {self.time_gates[k - 1]} >= "
                    f"index {k} = {t}",
                )
        if self.n_substeps_per_decade < 1:
            raise GeoBrainError(
                f"TEM3DSurvey.n_substeps_per_decade must be >= 1, got {self.n_substeps_per_decade}",
            )

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_receivers(self) -> int:
        return len(self.receivers)

    @property
    def n_time_gates(self) -> int:
        return len(self.time_gates)


# ---------------------------------------------------------------------------
# TEM3D operator
# ---------------------------------------------------------------------------


class TEM3D(EMOperatorDiscovery, ForwardOperator):
    """
    3D time-domain EM forward operator (BDF1+BDF2 log-substepped).

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
    ``data`` dict holds real-valued field-component channels indexed by
    ``(n_src, n_rcv, n_time_gates)``. For B-family components the
    reported quantity is ``dB/dt`` (step-off convention).

    ``__init__`` pre-assembles all the σ-independent quantities
    (``K``, ``A_e``, per-source ``s_e``, BDF time schedule,
    per-receiver Yee anchors); ``_forward`` runs the BDF1+BDF2 time
    loop on top of that scaffolding.

    Formulation:

    1. Discretise the step-off VMD as a 4-edge loop on Yee edges
       (:func:`build_vmd_step_off_source`).
    2. Solve a DC magnetostatic system for the ``t < 0`` steady-state
       E-field; this is the initial condition ``E^0``.
    3. BDF1 first substep + BDF2 thereafter; each substep solves
       ``(M_σ + Δt · K) x^{n+1} = M_σ · combine(x^n, x^{n-1})`` with E5's
       sparse-linear-solve adjoint.
    4. Sample strict multilinear Yee receiver plans at the requested time
       gates and take ``-curl(E) / 1`` (B on faces) → finite-difference in
       time → ``dB/dt`` for B-family components.

    Differentiability:
    :attr:`IMPLICIT_VJP` through ``sigma``. The σ-path enters both the
    per-substep matrix ``A(σ)`` (handled by the splu-bridge adjoint)
    and the right-hand side via edge-averaging (handled by torch
    autograd through :func:`assemble_cell_to_edge_averaging`);
    the two contributions sum automatically.

    Mesh contract:
        TEM3D is **ctx-threaded**: it reads ``mesh = ctx.require_mesh()`` as the
        authoritative assembly mesh, and ``requires_mesh_capabilities`` enforces
        a ``StructuredMesh``. It takes no mesh at construction. For efficiency it
        memoizes the σ-independent, mesh-dependent assembly on the instance,
        caching the context mesh as ``self.mesh`` keyed on identity (see
        :meth:`_ensure_assembly`); ``self.mesh`` therefore always tracks the
        current context mesh and cannot diverge from it. (That cache attribute is
        the only reason the capability-contract governance test lists
        TEM3D alongside :class:`~geobrain.physics.em.static.ip.IP3D`, which,
        unlike TEM3D, sources its mesh genuinely outside the context.)

    Args:
        survey: 3-D TEM acquisition.
        config: :class:`~geobrain.physics.em.EMExecutionConfig` execution
            policy (solver selection, closed-form sigma Jacobian).
        receiver_layout: cartesian vs borehole receiver layout.
        recording_policy: which time channels are recorded.
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        # Output keys are component.{real,imag}-style but TEM3D is
        # real-valued, so only the bare component string is emitted.
        # Receivers whose component differs (Ex/Ey/Ez/Bx/By/Bz) fill
        # the matching key; the rest stay absent. Listed here are the
        # canonical six time-domain channels.
        output_keys=("ex", "ey", "ez", "dbdt_x", "dbdt_y", "dbdt_z"),
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
        survey: TEM3DSurvey,
        *,
        config: EMExecutionConfig | None = None,
        receiver_layout: ReceiverLayout | str = ReceiverLayout.CARTESIAN,
        recording_policy: RecordingPolicy | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, TEM3DSurvey):
            raise GeoBrainError(
                f"TEM3D survey must be TEM3DSurvey, got {type(survey).__name__}",
            )
        self.survey = survey
        try:
            self._receiver_layout = ReceiverLayout(receiver_layout)
        except (TypeError, ValueError) as error:
            raise EMCapabilityError(
                "TEM3D receiver layout must be cartesian or paired",
                object_name="TEM3D",
                field="receiver_layout",
                expected=["cartesian", "paired"],
                actual=str(receiver_layout),
                details={
                    "field": "receiver_layout",
                    "received": str(receiver_layout),
                    "remediation": "select ReceiverLayout.CARTESIAN or PAIRED",
                },
                hint="use paired only when source and receiver counts match",
            ) from error
        if (
            self._receiver_layout is ReceiverLayout.PAIRED
            and survey.n_sources != survey.n_receivers
        ):
            raise EMCapabilityError(
                "paired TEM3D execution requires one receiver per source",
                object_name="TEM3D",
                field="receiver_layout",
                expected=survey.n_sources,
                actual=survey.n_receivers,
                details={
                    "field": "receiver_layout",
                    "receiver_count": survey.n_receivers,
                    "source_count": survey.n_sources,
                    "remediation": "make source and receiver counts equal or use cartesian",
                },
                hint="pair source i with receiver i",
            )
        if recording_policy is None:
            recording_policy = RecordingPolicy(mode="output_only")
        if type(recording_policy) is not RecordingPolicy:
            raise EMCapabilityError(
                "TEM3D recording policy must be a RecordingPolicy",
                object_name="TEM3D",
                field="recording_policy",
                expected="RecordingPolicy",
                actual=type(recording_policy).__qualname__,
                details={
                    "field": "recording_policy",
                    "remediation": "construct an explicit RecordingPolicy",
                },
                hint="use RecordingPolicy(mode='output_only')",
            )
        self._recording_policy = recording_policy
        # Retained only as an explicit migration sentinel. The historical
        # bridge factors every step internally and cannot satisfy the exact
        # repeated-dt cache contract, so forward rejects True before assembly.
        cfg = config if config is not None else EMExecutionConfig()
        self._use_closed_form_sigma_jacobian = cfg.use_closed_form_sigma_jacobian
        self._solver = resolve_engine_solver(cfg.resolve_solver())
        # Mesh-dependent assembly is built lazily on the first forward (mesh
        # now arrives via ctx.require_mesh(), not the constructor) and cached,
        # keyed on mesh identity: see :meth:`_ensure_assembly`. The cached
        # assembly is σ-INDEPENDENT pure geometry/topology, so no autograd tensor
        # is cached and a stale cache cannot poison a VJP. One instance assumes
        # one mesh per thread; the cache is NOT thread-safe (safe in practice;
        # the inverter is a strictly sequential backward() loop).
        self._assembled_mesh = None
        # Edge-element branch: σ-independent Whitney assets (assembly plan,
        # PEC boundary mask, centroid/curl evaluation factors, BDF schedule)
        # cached per mesh, same identity-keyed pattern as ``_assembled_mesh``.
        self._edge_assets: _TemEdgeFemAssets | None = None
        self._edge_assets_mesh = None
        # Unit-4 axis bridge: cache the x↔y-swapped (nz, ny, nx) engine mesh per
        # user-mesh identity so the σ-independent assembly cache (keyed on the
        # engine mesh in _ensure_assembly) still hits across an inversion loop.
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

    def _ensure_assembly(
        self,
        mesh: TensorMesh,
        *,
        receiver_mesh: TensorMesh | None = None,
    ) -> None:
        """
        Build the σ-independent, mesh-dependent assembly once per mesh.

        Mesh arrives via the :class:`ForwardContext` (``ctx.require_mesh()``)
        rather than the constructor, so the (expensive) curl-curl stiffness,
        face/edge mass diagonals, per-source step-off RHS, and BDF time-step
        schedule are pre-assembled lazily on the first :meth:`_forward` and
        cached on the instance, rebuilt only if a *different* mesh is seen.
        """
        if self._assembled_mesh is mesh:
            return
        self.mesh = mesh

        # Cache edge-family counts (used by the receiver-anchor resolver
        # and by _forward when assembling per-substep RHSs).
        n_ex, n_ey, n_ez = yee_edge_count(mesh)
        self._n_ex = int(n_ex)
        self._n_ey = int(n_ey)
        self._n_ez = int(n_ez)
        self._n_edges = int(n_ex + n_ey + n_ez)

        n_fx, n_fy, n_fz = yee_face_count(mesh)
        self._n_fx = int(n_fx)
        self._n_fy = int(n_fy)
        self._n_fz = int(n_fz)
        self._n_faces = int(n_fx + n_fy + n_fz)

        # ------------------------------------------------------------------
        # σ-independent pre-assembly (cached on the operator instance):
        #   * K:      bare curl-curl stiffness ``C^T · diag(μ⁻¹_face) · C``;
        #             includes μ₀ on every face (μ_r ≡ 1 by default).
        #   * A_e:    cell-to-edge averaging operator (real, geometry only).
        #   * s_e:    per-source step-off VMD RHS, stacked
        #             ``(n_sources, n_edges)``.
        #   * dts / gate_step_idx: BDF time-step schedule landing on each
        #             gate exactly (log-substepped).
        # ------------------------------------------------------------------
        # Yee curl operator C: edge -> face. Entries ``±1/Δ`` (metric
        # baked in). Used both for dB/dt = -C @ E sampling and to build
        # the volume-weighted stiffness K = C^T M_f(1/μ₀) C below.
        self._C_coo = yee_curl_e_to_f(mesh).to_sparse_coo().coalesce()

        # Mimetic face inner product ``M_f(1/μ₀) = diag(V_face/μ₀)``.
        # ``V_face`` is the per-face dual volume (uniform mesh: ``area ×
        # Δ_normal`` interior, halved at boundary nodes).
        self._V_face = _build_face_dual_volume(mesh)

        # Volume-weighted stiffness K = C^T diag(V_face/μ₀) C. This replaces
        # the bare ``C^T diag(1/μ₀) C`` of the original path (which
        # was missing the V_face weighting and produced incorrect
        # boundary-edge coupling: see commit notes in ``_forward``).
        C_indices = self._C_coo.indices()
        C_values = self._C_coo.values()
        face_weights = (self._V_face / _MU0).to(C_values.dtype)
        # K = C^T W C, with W = diag(face_weights). Scale C rows by W and
        # multiply by C^T to obtain K as a sparse matrix.
        W_C_values = face_weights[C_indices[0]] * C_values
        n_edges_K = int(self._C_coo.shape[1])
        n_faces_K = int(self._C_coo.shape[0])
        W_C = torch.sparse_coo_tensor(
            C_indices,
            W_C_values,
            size=(n_faces_K, n_edges_K),
            dtype=C_values.dtype,
        ).coalesce()
        C_T = torch.sparse_coo_tensor(
            torch.stack([C_indices[1], C_indices[0]], dim=0),
            C_values,
            size=(n_edges_K, n_faces_K),
            dtype=C_values.dtype,
        ).coalesce()
        self._K_coo = torch.sparse.mm(C_T, W_C).coalesce()
        # Legacy alias retained for tests that introspect ``_K`` directly.
        self._K = self._K_coo.to_sparse_csr()

        self._A_e = assemble_cell_to_edge_averaging(mesh)

        # Per-edge dual volume V_e (n_edges,): used for the volume-weighted
        # edge mass matrix ``M_eσ = diag(σ_edge · V_edge)``.
        self._V_edge = _build_edge_volume_diagonal(mesh)

        # Legacy: 4-edge mimetic VMD source loop (retained so older callers
        # that introspect ``_s_e`` still see something sensible; the revised
        # forward no longer uses it).
        s_e_list = [
            build_vmd_step_off_source(mesh, src.position, src.magnetic_moment_am2) for src in self.survey.sources
        ]
        self._s_e = torch.stack(s_e_list, dim=0)

        # Per-source analytic VMD vector potential projected onto Yee edge
        # tangents (n_sources, n_edges). The discrete primary B-field is
        # ``b_p = C @ a_e``; the step-off RHS impulse at the first BDF
        # step is ``q_1 = (1/dt_0) · K · a_e`` (equivalently
        # ``q_1 = -(1/dt_0)(s_e_1 − s_e_0)`` with ``s_e_0 = C^T M_f
        # b_p = K @ a_e`` and ``s_e_1 = 0``).
        a_e_list = [
            _build_vmd_vector_potential_on_edges(
                mesh,
                src.position,
                src.orientation,
                src.magnetic_moment_am2,
            )
            for src in self.survey.sources
        ]
        self._a_e = torch.stack(a_e_list, dim=0)

        # Cap individual backward-Euler step size at ``min_gate / 10``.
        # The bare log-substep schedule lets dt grow geometrically across
        # decades; at late times that pushes dt up to a few % of the gate
        # interval, which over-damps the diffusion modes responsible for
        # the dB/dt decay tail and inflates the cross-validation
        # error by an order of magnitude on receivers outside the early-
        # time diffusion radius. Capping dt to ``min_gate / 10`` ensures
        # the first gate (and every later one, by monotonicity) is
        # resolved with ≥ 10 sub-steps of fixed-or-smaller dt.
        first_gate = float(self.survey.time_gates[0])
        dt_max = first_gate / 10.0
        self._dts, self._gate_step_idx = build_log_substep_schedule(
            self.survey.time_gates,
            self.survey.n_substeps_per_decade,
            dt_max=dt_max,
        )

        public_mesh = mesh if receiver_mesh is None else receiver_mesh
        receiver_positions = torch.tensor(
            [receiver.position for receiver in self.survey.receivers],
            dtype=torch.float64,
        )
        if self._receiver_layout is ReceiverLayout.PAIRED:
            # A paired plan already contains one row per source/receiver pair.
            # Share that immutable batched plan across receivers of the same
            # channel instead of rebuilding and retaining n identical n-row
            # plans. VTEM's single BZ channel therefore retains exactly n rows.
            paired_projections: dict[str, YeeReceiverProjection] = {}
            for receiver in self.survey.receivers:
                channel = receiver.component.value
                if channel not in paired_projections:
                    paired_projections[channel] = build_yee_receiver_projection(
                        public_mesh,
                        receiver_positions,
                        channel=channel,
                        layout=self._receiver_layout,
                        n_sources=self.survey.n_sources,
                    )
            self._receiver_projections = tuple(
                paired_projections[receiver.component.value] for receiver in self.survey.receivers
            )
        else:
            self._receiver_projections = tuple(
                build_yee_receiver_projection(
                    public_mesh,
                    receiver_positions[index : index + 1],
                    channel=receiver.component.value,
                    layout=self._receiver_layout,
                    n_sources=self.survey.n_sources,
                )
                for index, receiver in enumerate(self.survey.receivers)
            )
        self._rx_indices = tuple(
            (
                "E" if projection.channel.startswith("e") else "B",
                projection.dof_indices[row_index][
                    max(
                        range(len(projection.interpolation_weights[row_index])),
                        key=projection.interpolation_weights[row_index].__getitem__,
                    )
                ],
            )
            for receiver_index, projection in enumerate(self._receiver_projections)
            for row_index in (
                receiver_index if self._receiver_layout is ReceiverLayout.PAIRED else 0,
            )
        )
        self._rx_interp: tuple[tuple[str, torch.Tensor, torch.Tensor], ...] = tuple(
            (
                "E" if projection.channel.startswith("e") else "B",
                torch.tensor(projection.dof_indices[row_index], dtype=torch.long),
                torch.tensor(projection.interpolation_weights[row_index], dtype=torch.float64),
            )
            for receiver_index, projection in enumerate(self._receiver_projections)
            for row_index in (
                receiver_index if self._receiver_layout is ReceiverLayout.PAIRED else 0,
            )
        )

        self._assembled_mesh = mesh

    def _resolve_receiver_anchor(
        self,
        receiver: TEM3DReceiver,
    ) -> tuple[str, int]:
        """
        Resolve a receiver's Yee anchor to ``(buffer, flat_index)``.

        Mirrors :func:`~geobrain.physics.em.frequency_domain.
        fdem3d._extract_component_at_receiver`: each Yee component lives
        on either a cell-centre or a node along each axis. This legacy view
        returns the dominant index of the strict multilinear plan:

        - ``buffer == "E"``: gather from a ``(n_edges,)`` E-field tensor.
        - ``buffer == "B"``: gather from a ``(n_faces,)`` B-field tensor;
          the operator emits ``dB/dt`` for these channels.

        ``HX/HY/HZ`` are not exposed by ``TEM3DReceiver`` (its component
        type is :class:`FieldComponent` but the dataclass docstring scopes
        TEM observables to E/B); a future extension would gather from
        ``B`` and post-divide by ``μ₀``.
        """
        comp = receiver.component
        # TEM scopes observables to E/B (no H); reject others as before. The
        # Yee anchor arithmetic itself lives in the shared kernel.
        if comp not in (
            FieldComponent.EX,
            FieldComponent.EY,
            FieldComponent.EZ,
            FieldComponent.BX,
            FieldComponent.BY,
            FieldComponent.BZ,
        ):
            raise GeoBrainError(
                f"TEM3D._resolve_receiver_anchor: unsupported component "
                f"{comp!r}; expected one of EX/EY/EZ/BX/BY/BZ",
                object_name="TEM3D._resolve_receiver_anchor",
                field="receiver.component",
                expected="one of EX/EY/EZ/BX/BY/BZ",
                actual=repr(comp),
            )
        projection = build_yee_receiver_projection(
            self.mesh,
            torch.tensor([receiver.position], dtype=torch.float64),
            channel=comp.value,
            layout="cartesian",
            n_sources=self.survey.n_sources,
        )
        weights = projection.interpolation_weights[0]
        maximum = max(range(len(weights)), key=weights.__getitem__)
        family = "E" if projection.channel.startswith("e") else "B"
        return family, projection.dof_indices[0][maximum]

    def _resolve_receiver_interp(
        self,
        receiver: TEM3DReceiver,
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """
        Trilinear stencil for a receiver's Yee field component.

        Returns ``(buffer, idx, weights)`` where

        - ``buffer`` is ``"E"`` (gather from ``(n_edges,)`` E-field) or
          ``"B"`` (gather from ``(n_faces,)`` ``dB/dt`` field);
        - ``idx`` is an int64 tensor of length 8 containing the flat
          indices of the 8 Yee anchors surrounding the receiver;
        - ``weights`` is a float64 tensor of length 8 of trilinear
          interpolation weights summing to 1.

        Coordinates are resolved by the shared origin-aware projection builder.
        Outside points fail; exact upper support boundaries use the final legal
        interval.
        """
        comp = receiver.component
        if comp not in (
            FieldComponent.EX,
            FieldComponent.EY,
            FieldComponent.EZ,
            FieldComponent.BX,
            FieldComponent.BY,
            FieldComponent.BZ,
        ):
            raise GeoBrainError(
                f"TEM3D._resolve_receiver_interp: unsupported component "
                f"{comp!r}; expected one of EX/EY/EZ/BX/BY/BZ",
                object_name="TEM3D._resolve_receiver_interp",
                field="receiver.component",
                expected="one of EX/EY/EZ/BX/BY/BZ",
                actual=repr(comp),
            )
        projection = build_yee_receiver_projection(
            self.mesh,
            torch.tensor([receiver.position], dtype=torch.float64),
            channel=comp.value,
            layout="cartesian",
            n_sources=self.survey.n_sources,
        )
        return (
            "E" if projection.channel.startswith("e") else "B",
            torch.tensor(projection.dof_indices[0], dtype=torch.long),
            torch.tensor(projection.interpolation_weights[0], dtype=torch.float64),
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        """
        Volume-weighted backward-Euler step-off time loop.

        Algorithm:
        The forward originally used a bare-mass BDF1+BDF2 stepper
        with ``E^0 = (I·L_edge / V_edge) / σ_edge`` as the step-off IC
        and a face-volume-free ``K``. That formulation is numerically
        unstable in the presence of air cells (the IC diverges where
        ``σ_air → 0``) and over-damped on diffusive responses, giving
        median ~8 %, p90 ~58 %, and 4/18 sign flips against the reference
        implementation on a step-off VMD problem.

        The revised forward mirrors the reference mimetic discretization:

        - **Stiffness** ``K = C^T M_f(1/μ₀) C`` with the face inner
          product ``M_f(1/μ₀) = diag(V_face/μ₀)`` (pre-assembled in
          ``__init__``). ``V_face`` is the per-face dual volume; this
          weighting was missing from the original ``K``, which left
          boundary-edge couplings off by a constant factor.
        - **Edge mass** ``M_eσ = diag(σ_edge · V_edge)`` (proper edge
          inner product). Previously the bare ``diag(σ_edge)``.
        - **Initial condition** ``e_0 = 0``. The step-off jump is
          encoded as a one-shot RHS impulse at the first BDF step:
          ``q_1 = (1/dt_0) · K · a_e``, where ``a_e`` is the analytic
          free-space VMD vector potential projected onto Yee edge
          tangents. (Equivalently:
          ``q_1 = -(1/dt_0)(s_{e,1} − s_{e,0})`` with
          ``s_{e,0} = C^T M_f b_p = K @ a_e`` and ``s_{e,1} = 0`` for
          the step-off waveform.)
        - **Time stepping** backward Euler at every step:
          ``(K + (1/dt_n) M_eσ) e_{n+1} = q_{n+1} + (1/dt_n) M_eσ e_n``.
          BDF2 was dropped because (a) the reference implementation uses BE and (b) BE is
          more dissipative and stable for stiff diffusion problems
          with strongly heterogeneous σ (air vs earth).

        IMPLICIT_VJP autograd through ``sigma`` is preserved: the
        per-step matrix ``A_n = K + (1/dt_n) M_eσ`` carries the σ-graph
        through ``M_eσ``, and ``sparse_linear_solve_with_adjoint``
        propagates gradients via the splu-bridge adjoint.

        Receiver sampling: at each requested time gate, build
        ``dB/dt = -C @ E`` on Yee faces and gather one scalar per
        receiver via the cached ``(buffer, flat_index)`` anchors.

        Output is per-component:
        ``ForwardOutput.data[<component>]`` of shape
        ``(n_sources, n_receivers, n_time_gates)``, real ``float64``.
        For B-family components the value is ``dB/dt`` (T/s).
        """
        (sigma_3d,) = state.fetch("sigma")
        if self._use_closed_form_sigma_jacobian:
            raise EMCapabilityError(
                "TEM3D closed-form sigma Jacobian mode does not support exact "
                "repeated-step factor reuse",
                details={
                    "mode": "closed_form_sigma_jacobian",
                    "remediation": "use the default implicit-VJP sparse factor path",
                },
                object_name="TEM3D",
                field="use_closed_form_sigma_jacobian",
                expected=False,
                actual=True,
                hint="use TEM3D(..., config=EMExecutionConfig(use_closed_form_sigma_jacobian=False))",
            )
        if sigma_3d.device.type != "cpu":
            raise GeoBrainError(
                "TEM3D is CPU-only (scipy sparse LU); move sigma to cpu first",
                object_name="TEM3D",
                field="sigma.device",
                expected="cpu",
                actual=sigma_3d.device,
            )
        mesh = ctx.require_mesh()
        if mesh.n_dim != 3:
            raise GeoBrainError(
                "TEM3D requires a 3D mesh (pass the mesh via ForwardContext.of(mesh=...))",
                object_name="TEM3D",
                field="mesh",
                expected="3D",
                actual=mesh.n_dim,
            )

        # Capability dispatch (declaration-based, the shared inductive-EM
        # ladder): StructuredMesh → Yee FV here, EdgeConnectivityMesh → the
        # Nédélec edge-element path, anything else (octree) raises.
        if resolve_inductive_mesh_path(mesh, operator_name="TEM3D") == "edge":
            return self._forward_edge_fem(sigma_3d, mesh)

        # The structured Yee FV path (curl / edge-face mass assembly, edge-centre
        # coordinates, AND receiver projection) reads scalar ``mesh.spacing``; it
        # assumes a UNIFORM TensorMesh. Reject non-uniform / graded meshes here
        # rather than silently mis-assembling the operator and mis-binding
        # receivers to the wrong Yee cell. MT3D is the non-uniform-capable
        # inductive operator (it assembles from per-cell ``cell_widths``).
        if not mesh.is_uniform:
            raise GeoBrainError(
                "TEM3D structured Yee FV path requires a uniform TensorMesh; "
                "non-uniform/graded meshes are not supported on this path "
                "(use MT3D for graded meshes, or an unstructured edge mesh)",
                object_name="TEM3D",
                field="mesh",
                expected="uniform TensorMesh",
                actual="non-uniform TensorMesh",
            )

        # Unit-4 axis bridge: the operator's public contract is the platform
        # (nz, nx, ny) layout (axis-1 = x, axis-2 = y); the Yee engine below is
        # written in (nz, ny, nx). Validate σ against the user mesh, then swap
        # BOTH mesh geometry and σ so the engine receives the
        # algebraically-identical (nz, ny, nx) encoding of the same physics.
        # Source/receiver coords and field-component labels are
        # layout-independent and pass through unchanged. (curl-curl is x↔y
        # symmetric ⇒ exact relabel; validated by test_axis_convention_engine.)
        require_field_matches_mesh(mesh, sigma_3d, name="sigma", owner="TEM3D")
        receiver_mesh = mesh
        mesh = self._engine_mesh(mesh)
        sigma_3d = swap_xy_cell_field(sigma_3d)

        self._ensure_assembly(mesh, receiver_mesh=receiver_mesh)

        # ------------------------------------------------------------------
        # σ on edges (autograd through σ → σ_edge → M_eσ).
        # ``sparse_linear_solve_with_adjoint`` (scipy splu path) requires
        # float64. Promote sigma → float64 explicitly (the DC3D pattern)
        # rather than relying on an incidental cast; ``.to()`` is
        # differentiable, so the autograd chain to the caller's σ is
        # preserved.
        # ------------------------------------------------------------------
        sigma_f64 = sigma_3d if sigma_3d.dtype == torch.float64 else sigma_3d.to(torch.float64)
        sigma_flat = sigma_f64.reshape(-1)
        sigma_edge = torch.sparse.mm(
            self._A_e,
            sigma_flat.unsqueeze(1),
        ).squeeze(1)  # (n_edges,)
        M_eσ_diag = sigma_edge * self._V_edge  # (n_edges,)

        n_src = self.survey.n_sources
        n_rcv = self.survey.n_receivers
        n_gates = self.survey.n_time_gates
        n_steps = int(self._dts.numel())

        # ------------------------------------------------------------------
        # Initial condition: e_0 = 0 (step-off VMD convention).
        # ------------------------------------------------------------------
        n_edges = self._n_edges
        E0 = torch.zeros(
            (n_edges, n_src),
            dtype=torch.float64,
            device=sigma_edge.device,
        )

        # ------------------------------------------------------------------
        # Pre-compute the source impulse ``K · a_e`` for each source. The
        # step-off RHS at the first BDF step is ``(1/dt_0) K @ a_e``.
        # Shape: (n_edges, n_src).
        # ------------------------------------------------------------------
        a_e_t = self._a_e.t().contiguous()  # (n_edges, n_src)
        K_at_a = torch.sparse.mm(self._K_coo, a_e_t)  # (n_edges, n_src)

        # ------------------------------------------------------------------
        # COO topology cache for fast per-step rebuild of
        # ``A_n = K + (1/dt_n) diag(M_eσ_diag)``.
        # ------------------------------------------------------------------
        K_indices = self._K_coo.indices()
        K_values = self._K_coo.values()

        diag_idx = torch.arange(n_edges, dtype=torch.long)
        diag_indices = torch.stack([diag_idx, diag_idx], dim=0)

        A_indices = torch.cat([K_indices, diag_indices], dim=1)

        # ------------------------------------------------------------------
        # Backward-Euler time loop.
        #
        # Step 0:  (K + (1/dt_0) M_eσ) e_1 = (1/dt_0) K · a_e
        # Step n:  (K + (1/dt_n) M_eσ) e_{n+1} = (1/dt_n) M_eσ · e_n
        # ------------------------------------------------------------------
        component_names = tuple(dict.fromkeys(rcv.component.value for rcv in self.survey.receivers))
        channel_names = tuple(
            {"bx": "dbdt_x", "by": "dbdt_y", "bz": "dbdt_z"}.get(name, name)
            for name in component_names
        )
        observation_shape = (
            (len(component_names), n_src)
            if self._receiver_layout is ReceiverLayout.PAIRED
            else (len(component_names), n_src, n_rcv)
        )
        matrix_requires_gradient = bool(torch.is_grad_enabled() and sigma_flat.requires_grad)
        recording_plan = prepare_recording(
            self._recording_policy,
            n_steps=n_steps,
            gate_history_indices=tuple(
                int(self._gate_step_idx[index].item()) + 1 for index in range(n_gates)
            ),
            state=E0,
            observation_shape=observation_shape,
            observation_dtype=torch.float64,
            requires_gradient=matrix_requires_gradient,
        )

        # ``A_n = K + (1/dt_n) M_eσ`` is numerically identical for every
        # exact repeated ``dt_n``. Build every distinct matrix/factor once
        # before recurrence execution. The resulting matrices are explicit
        # differentiable inputs to checkpoint segments; no hidden cache state
        # crosses a checkpoint boundary during backward recomputation.
        execution_cache = EMExecutionCache()
        matrix_mesh_version = mesh_fingerprint(mesh)
        matrix_material_version = material_fingerprint(sigma_flat)
        inductive_solver = self._solver
        inductive_solver_settings = solver_settings(inductive_solver)
        step_matrix_slots: list[int] = []
        matrix_slots: dict[str, int] = {}
        matrices: list[torch.Tensor] = []
        factors: list[SparseFactor] = []
        for n in range(n_steps):
            dt_n = float(self._dts[n].item())
            inv_dt = 1.0 / dt_n
            sample_token = exact_float_token(dt_n)
            assembly_key = AssemblyCacheKey(
                formulation_version="tem3d-yee-backward-euler-v1",
                mesh_fingerprint=matrix_mesh_version,
                material_version=matrix_material_version,
                boundary="natural-mass",
                sample_value=sample_token,
                dtype=str(sigma_flat.dtype),
                device=str(sigma_flat.device),
                backend=inductive_solver_settings.backend,
                requires_gradient=matrix_requires_gradient,
            )

            def _assemble_step() -> torch.Tensor:
                A_values = torch.cat(
                    [
                        K_values,
                        inv_dt * M_eσ_diag,
                    ],
                    dim=0,
                )
                return torch.sparse_coo_tensor(
                    A_indices,
                    A_values,
                    size=(n_edges, n_edges),
                    dtype=torch.float64,
                ).coalesce()

            A_coo = execution_cache.get_or_assemble(
                assembly_key,
                _assemble_step,
            )
            factor = execution_cache.get_or_factor(
                solver_factor_key(assembly_key, inductive_solver),
                lambda: factorize_sparse(A_coo, solver=inductive_solver),
            )
            if sample_token not in matrix_slots:
                matrix_slots[sample_token] = len(matrices)
                matrices.append(A_coo)
                factors.append(factor)
            step_matrix_slots.append(matrix_slots[sample_token])

        def _step(
            index: int,
            previous: torch.Tensor,
            mass_diagonal: torch.Tensor,
            *step_matrices: torch.Tensor,
        ) -> torch.Tensor:
            dt_n = float(self._dts[index].item())
            inv_dt = 1.0 / dt_n
            rhs = (
                inv_dt * K_at_a if index == 0 else inv_dt * (mass_diagonal.unsqueeze(1) * previous)
            )
            slot = step_matrix_slots[index]
            next_state = sparse_linear_solve_with_adjoint(
                step_matrices[slot],
                rhs,
                factor=factors[slot],
            )
            execution_cache.diagnostics.solve_count += 1
            execution_cache.diagnostics.rhs_count += n_src
            return next_state

        def _project_gate(E_at_gate: torch.Tensor) -> torch.Tensor:
            dBdt_at_gate = -torch.sparse.mm(self._C_coo, E_at_gate)  # (n_faces, n_src)
            zero = E_at_gate.new_zeros(())
            channels: list[torch.Tensor] = []
            for component_name in component_names:
                if self._receiver_layout is ReceiverLayout.PAIRED:
                    paired_values: list[torch.Tensor] = []
                    for s_idx, rcv in enumerate(self.survey.receivers):
                        if rcv.component.value != component_name:
                            paired_values.append(zero)
                            continue
                        buf, idx_tensor, wt_tensor = self._rx_interp[s_idx]
                        field = E_at_gate[:, s_idx] if buf == "E" else dBdt_at_gate[:, s_idx]
                        paired_values.append((field[idx_tensor] * wt_tensor).sum())
                    channels.append(torch.stack(paired_values))
                    continue
                source_rows: list[torch.Tensor] = []
                for s_idx in range(n_src):
                    receiver_values: list[torch.Tensor] = []
                    for r_idx, rcv in enumerate(self.survey.receivers):
                        if rcv.component.value != component_name:
                            receiver_values.append(zero)
                            continue
                        buf, idx_tensor, wt_tensor = self._rx_interp[r_idx]
                        field = E_at_gate[:, s_idx] if buf == "E" else dBdt_at_gate[:, s_idx]
                        receiver_values.append((field[idx_tensor] * wt_tensor).sum())
                    source_rows.append(torch.stack(receiver_values))
                channels.append(torch.stack(source_rows))
            return torch.stack(channels)

        gate_outputs, recording_diagnostics = execute_recorded_recurrence(
            recording_plan,
            E0,
            step=_step,
            observe=_project_gate,
            differentiable_inputs=(M_eσ_diag, *matrices),
        )
        recorded = torch.stack(gate_outputs, dim=-1)
        data = {name: recorded[index].contiguous() for index, name in enumerate(channel_names)}

        diagnostics = execution_cache.diagnostics
        projected_elements = (
            n_src * n_gates
            if self._receiver_layout is ReceiverLayout.PAIRED
            else n_src * n_rcv * n_gates
        )
        diagnostics.projection_count = projected_elements
        diagnostics.projected_element_count = projected_elements
        metadata = {
            **solver_execution_metadata(diagnostics, inductive_solver_settings),
            **recording_execution_metadata(recording_diagnostics, self._recording_policy),
            "method": "tem3d",
            "n_sources": n_src,
            "n_receivers": n_rcv,
            "n_time_gates": n_gates,
            "n_steps": n_steps,
            "formulation": "step_off",
            "source_kind": "magnetic_dipole",
            "time_integrator": "backward_euler",
            "receiver_layout": self._receiver_layout.value,
        }
        execution_cache.close()
        return ForwardOutput(
            data=data,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Edge-element (Nédélec/Whitney) branch: 3-D simplex UnstructuredMesh
    # ------------------------------------------------------------------

    def _edge_fem_assets(self, mesh) -> "_TemEdgeFemAssets":
        """σ-independent Whitney assets for ``mesh``, cached per mesh identity.

        Pure geometry/topology plus the survey's BDF schedule (assembly plan,
        PEC boundary mask, per-cell Whitney centroid-value and curl factors,
        receiver-binding centres, log-substep ``dts``/``gate_step_idx``);
        carries no autograd graph, so the same cache is reused across forwards
        and sources, exactly like the Yee ``_ensure_assembly`` scaffolding.
        The schedule is the SAME :func:`build_log_substep_schedule` call the
        Yee path makes (incl. the ``dt_max = first_gate / 10`` cap).
        """
        if self._edge_assets is None or self._edge_assets_mesh is not mesh:
            plan = build_edge_assembly_plan(mesh)
            cell_edge_ids, cell_edge_signs = mesh.cell_edges()
            g, _volume = barycentric_gradients(
                mesh.node_coords(),
                mesh.cell_nodes(),
            )
            # Whitney barycentre value + constant curl (float64: kept real for
            # this operator's real time-domain system, no complex post-cast).
            w_centroid, curl_w = whitney_centroid_factors(g)  # (nc, 6, 3)
            first_gate = float(self.survey.time_gates[0])
            dts, gate_step_idx = build_log_substep_schedule(
                self.survey.time_gates,
                self.survey.n_substeps_per_decade,
                dt_max=first_gate / 10.0,
            )
            self._edge_assets = _TemEdgeFemAssets(
                plan=plan,
                boundary_mask=boundary_edge_mask(mesh),
                edge_records=mesh.edge_records(),
                cell_edge_ids=cell_edge_ids,
                cell_edge_signs=cell_edge_signs,
                w_centroid=w_centroid,
                curl_w=curl_w,
                cell_centers=mesh.cell_centers(),
                dts=dts,
                gate_step_idx=gate_step_idx,
            )
            self._edge_assets_mesh = mesh
        return self._edge_assets

    def _forward_edge_fem(self, sigma: torch.Tensor, mesh) -> ForwardOutput:
        """Edge-element path for a 3-D simplex ``EdgeConnectivityMesh``.

        Symbol-for-symbol the Yee step-off formulation on Whitney edge dofs
        (``dof_e = ∫_e E·dl`` along the canonical edge direction):

        1. **Initial condition** ``e_0 = 0``; the step-off jump enters as a
           one-shot RHS impulse at the first BDF step,
           ``q_1 = (1/Δt_0)·K_w(1/μ₀)·a_e``, where ``a_e`` is the analytic
           free-space VMD vector potential projected onto Whitney edge dofs
           by the midpoint rule (the Yee path samples the same closed form
           at Yee edge tangents, :func:`_vmd_vector_potential_at_points`
           vs :func:`_build_vmd_vector_potential_on_edges`).
        2. **Time stepping** backward Euler at every step on the SAME
           log-substep schedule: ``A_n e_{n+1} = q_{n+1} + (1/Δt_n)·M_w(σ)·e_n``
           with ``A_n = K_w(1/μ₀) + (1/Δt_n)·M_w(σ)``, the Whitney mass and
           curl-curl operators carry the volume metric the Yee path encodes
           via ``V_edge``/``V_face`` diagonals, so no extra volume factor
           appears. PEC truncation pins the outer-boundary edges (rows AND
           columns zeroed + unit diagonal, RHS rows zeroed, the FDEM3D
           edge-branch pattern). One splu factor per unique ``Δt_n`` is
           cached and reused across substeps (the Yee factor-cache policy).
        3. **Receiver sampling** at each requested gate via the E history
           index ``gate_step_idx[g] + 1`` (the Yee off-by-one contract):
           strict containing-cell binding, the Whitney interpolant at the
           physical receiver for E channels and the per-cell constant
           ``dB/dt = -curl E`` for B channels (T/s, step-off convention).

        ``sigma`` is the flat ``(n_cells,)`` cell conductivity.
        Differentiable in ``sigma`` end-to-end: the σ-path enters every
        ``A_n`` (splu-bridge adjoint) and every RHS mass matvec (autograd
        through the Whitney mass coefficients); the contributions sum
        automatically. Output ``data`` keys/shapes/units are identical to
        the Yee path.
        """
        n_cells = int(mesh.n_cells)
        if sigma.dim() != 1 or int(sigma.shape[0]) != n_cells:
            raise GeoBrainError(
                "TEM3D (edge-element path): sigma must be flat (n_cells,)",
                object_name="TEM3D",
                field="sigma",
                expected=(n_cells,),
                actual=tuple(sigma.shape),
            )

        assets = self._edge_fem_assets(mesh)
        plan = assets.plan
        bmask = assets.boundary_mask
        records = assets.edge_records
        sigma_cells = sigma.to(torch.float64)

        sources = self.survey.sources
        receivers = self.survey.receivers
        n_src = self.survey.n_sources
        n_rcv = self.survey.n_receivers
        n_gates = self.survey.n_time_gates
        n_steps = int(assets.dts.numel())
        n_edges = plan.n_edges

        # Cell coefficients of the two Whitney terms: stiffness 1/μ₀ and the
        # per-step mass σ/Δt_n (assembled inside the loop). ``zero_coeff``
        # isolates single-term matvecs (K-only impulse, M-only RHS).
        inv_mu0 = torch.full((n_cells,), 1.0 / _MU0, dtype=torch.float64)
        zero_coeff = torch.zeros(n_cells, dtype=torch.float64)

        # ------------------------------------------------------------------
        # Source impulse ``K_w(1/μ₀) · a_e`` per source (n_edges, n_src).
        # ------------------------------------------------------------------
        K_at_a = torch.stack(
            [
                edge_operator_matvec(
                    plan,
                    zero_coeff,
                    inv_mu0,
                    _edge_vmd_vector_potential_dofs(records, src),
                )
                for src in sources
            ],
            dim=1,
        )

        # Symmetric Dirichlet (PEC) pin scaffolding (σ/dt-independent):
        # precomputed once and reused per step, fed to the hot loop's
        # :func:`assemble_pec_pinned_operator` (which zeroes the boundary
        # rows/columns and appends the unit diagonal entries per Δt).
        pin_plan = build_pec_pin_plan(
            bmask,
            torch.stack([plan.rows, plan.cols]),
        )

        # ------------------------------------------------------------------
        # Backward-Euler time loop (the Yee loop on Whitney operators).
        #
        # Step 0:  (K_w + (1/Δt_0) M_w(σ)) e_1 = (1/Δt_0) K_w · a_e
        # Step n:  (K_w + (1/Δt_n) M_w(σ)) e_{n+1} = (1/Δt_n) M_w(σ) · e_n
        # ------------------------------------------------------------------
        E0 = torch.zeros((n_edges, n_src), dtype=torch.float64)
        rcv_pos = torch.tensor(
            [rcv.position for rcv in receivers],
            dtype=torch.float64,
        )
        if self._receiver_layout is ReceiverLayout.PAIRED:
            # As on Yee meshes, one immutable n-row plan serves every paired
            # receiver of a channel. This also avoids repeating tetrahedron
            # point-location work n times for an n-station flight line.
            paired_edge_projections: dict[str, EdgeReceiverProjection] = {}
            for receiver in receivers:
                channel = receiver.component.value
                if channel not in paired_edge_projections:
                    paired_edge_projections[channel] = build_edge_receiver_projection(
                        mesh,
                        rcv_pos,
                        channel=channel,
                        layout=self._receiver_layout,
                        n_sources=n_src,
                    )
            edge_projections = tuple(
                paired_edge_projections[receiver.component.value] for receiver in receivers
            )
        else:
            edge_projections = tuple(
                build_edge_receiver_projection(
                    mesh,
                    rcv_pos[index : index + 1],
                    channel=receiver.component.value,
                    layout=self._receiver_layout,
                    n_sources=n_src,
                )
                for index, receiver in enumerate(receivers)
            )
        projection_rows = tuple(
            index if self._receiver_layout is ReceiverLayout.PAIRED else 0 for index in range(n_rcv)
        )
        rcv_cells = torch.tensor(
            [
                projection.element_indices[row]
                for projection, row in zip(edge_projections, projection_rows, strict=True)
            ],
            dtype=torch.long,
        )
        rcv_ids = torch.tensor(
            [
                projection.local_edge_dof_indices[row]
                for projection, row in zip(edge_projections, projection_rows, strict=True)
            ],
            dtype=torch.long,
        )
        rcv_signs = torch.tensor(
            [
                projection.orientation_signs[row]
                for projection, row in zip(edge_projections, projection_rows, strict=True)
            ],
            dtype=torch.float64,
        )
        rcv_basis = torch.tensor(
            [
                projection.basis_weights[row]
                for projection, row in zip(edge_projections, projection_rows, strict=True)
            ],
            dtype=torch.float64,
        ).reshape(n_rcv, 6, 3)
        rcv_curl_w = assets.curl_w[rcv_cells][:, :, (1, 2, 0)]

        component_names = tuple(dict.fromkeys(receiver.component.value for receiver in receivers))
        channel_names = tuple(
            {"bx": "dbdt_x", "by": "dbdt_y", "bz": "dbdt_z"}.get(name, name)
            for name in component_names
        )
        observation_shape = (
            (len(component_names), n_src)
            if self._receiver_layout is ReceiverLayout.PAIRED
            else (len(component_names), n_src, n_rcv)
        )
        matrix_requires_gradient = bool(torch.is_grad_enabled() and sigma_cells.requires_grad)
        recording_plan = prepare_recording(
            self._recording_policy,
            n_steps=n_steps,
            gate_history_indices=tuple(
                int(assets.gate_step_idx[index].item()) + 1 for index in range(n_gates)
            ),
            state=E0,
            observation_shape=observation_shape,
            observation_dtype=torch.float64,
            requires_gradient=matrix_requires_gradient,
        )

        # Prebuild one matrix/factor per exact time-step key. Matrices and the
        # material vector are explicit checkpoint inputs; only recurrence
        # states are recomputed during backward.
        execution_cache = EMExecutionCache()
        matrix_mesh_version = mesh_fingerprint(mesh)
        matrix_material_version = material_fingerprint(sigma_cells)
        inductive_solver = self._solver
        inductive_solver_settings = solver_settings(inductive_solver)
        step_matrix_slots: list[int] = []
        matrix_slots: dict[str, int] = {}
        matrices: list[torch.Tensor] = []
        factors: list[SparseFactor] = []
        for n in range(n_steps):
            dt_n = float(assets.dts[n].item())
            inv_dt = 1.0 / dt_n
            sample_token = exact_float_token(dt_n)
            assembly_key = AssemblyCacheKey(
                formulation_version="tem3d-whitney-backward-euler-v1",
                mesh_fingerprint=matrix_mesh_version,
                material_version=matrix_material_version,
                boundary="pec",
                sample_value=sample_token,
                dtype=str(sigma_cells.dtype),
                device=str(sigma_cells.device),
                backend=inductive_solver_settings.backend,
                requires_gradient=matrix_requires_gradient,
            )
            A_sp = execution_cache.get_or_assemble(
                assembly_key,
                lambda: assemble_pec_pinned_operator(
                    plan,
                    pin_plan,
                    mass_coeff=inv_dt * sigma_cells,
                    stiffness_coeff=inv_mu0,
                ),
            )
            factor = execution_cache.get_or_factor(
                solver_factor_key(assembly_key, inductive_solver),
                lambda: factorize_sparse(A_sp, solver=inductive_solver),
            )
            if sample_token not in matrix_slots:
                matrix_slots[sample_token] = len(matrices)
                matrices.append(A_sp)
                factors.append(factor)
            step_matrix_slots.append(matrix_slots[sample_token])

        def _step(
            index: int,
            previous: torch.Tensor,
            material: torch.Tensor,
            *step_matrices: torch.Tensor,
        ) -> torch.Tensor:
            dt_n = float(assets.dts[index].item())
            inv_dt = 1.0 / dt_n
            rhs = (
                inv_dt * K_at_a
                if index == 0
                else inv_dt
                * torch.stack(
                    [
                        edge_operator_matvec(
                            plan,
                            material,
                            zero_coeff,
                            previous[:, source_index],
                        )
                        for source_index in range(n_src)
                    ],
                    dim=1,
                )
            )
            rhs = pec_zero_rhs(bmask, rhs)
            slot = step_matrix_slots[index]
            next_state = sparse_linear_solve_with_adjoint(
                step_matrices[slot],
                rhs,
                factor=factors[slot],
            )
            execution_cache.diagnostics.solve_count += 1
            execution_cache.diagnostics.rhs_count += n_src
            return next_state

        def _project_gate(E_at_gate: torch.Tensor) -> torch.Tensor:
            zero = E_at_gate.new_zeros(())
            channels: list[torch.Tensor] = []
            for component_name in component_names:
                if self._receiver_layout is ReceiverLayout.PAIRED:
                    paired_values: list[torch.Tensor] = []
                    for source_index, receiver in enumerate(receivers):
                        if receiver.component.value != component_name:
                            paired_values.append(zero)
                            continue
                        dof = E_at_gate[:, source_index]
                        local = rcv_signs[source_index] * dof[rcv_ids[source_index]]
                        E_vec = torch.einsum("e,ek->k", local, rcv_basis[source_index])
                        dBdt_vec = -torch.einsum("e,ek->k", local, rcv_curl_w[source_index])
                        paired_values.append(
                            pick_field_component(E_vec, dBdt_vec, receiver.component)
                        )
                    channels.append(torch.stack(paired_values))
                    continue
                source_rows: list[torch.Tensor] = []
                for source_index in range(n_src):
                    dof = E_at_gate[:, source_index]
                    local = rcv_signs * dof[rcv_ids]
                    E_vec = torch.einsum("re,rek->rk", local, rcv_basis)
                    dBdt_vec = -torch.einsum("re,rek->rk", local, rcv_curl_w)
                    receiver_values = [
                        (
                            pick_field_component(
                                E_vec[receiver_index],
                                dBdt_vec[receiver_index],
                                receiver.component,
                            )
                            if receiver.component.value == component_name
                            else zero
                        )
                        for receiver_index, receiver in enumerate(receivers)
                    ]
                    source_rows.append(torch.stack(receiver_values))
                channels.append(torch.stack(source_rows))
            return torch.stack(channels)

        gate_outputs, recording_diagnostics = execute_recorded_recurrence(
            recording_plan,
            E0,
            step=_step,
            observe=_project_gate,
            differentiable_inputs=(sigma_cells, *matrices),
        )
        recorded = torch.stack(gate_outputs, dim=-1)
        data = {name: recorded[index].contiguous() for index, name in enumerate(channel_names)}

        diagnostics = execution_cache.diagnostics
        projected_elements = (
            n_src * n_gates
            if self._receiver_layout is ReceiverLayout.PAIRED
            else n_src * n_rcv * n_gates
        )
        diagnostics.projection_count = projected_elements
        diagnostics.projected_element_count = projected_elements
        metadata = {
            **solver_execution_metadata(diagnostics, inductive_solver_settings),
            **recording_execution_metadata(recording_diagnostics, self._recording_policy),
            "method": "tem3d",
            "n_sources": n_src,
            "n_receivers": n_rcv,
            "n_time_gates": n_gates,
            "n_steps": n_steps,
            "formulation": "step_off",
            "source_kind": "magnetic_dipole",
            "time_integrator": "backward_euler",
            "receiver_layout": self._receiver_layout.value,
        }
        execution_cache.close()
        return ForwardOutput(
            data=data,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Edge-element (Nédélec/Whitney) branch helpers
# ---------------------------------------------------------------------------






