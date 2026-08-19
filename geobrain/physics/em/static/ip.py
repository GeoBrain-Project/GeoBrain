# pyright: reportPrivateImportUsage=false
"""Induced Polarization (IP): time-domain chargeability via dual DC solve.

Time-domain IP measures the apparent chargeability ``M_a`` at each
receiver via two DC forward solves with the same electrode geometry:

    V_∞:    primary potential (high-frequency limit; σ_∞ everywhere)
    V_η:    steady-state potential (DC limit; σ_∞ · (1 − η) everywhere)

For a homogeneous medium with uniform η,

    V_∞ ∝ 1/σ_∞
    V_η ∝ 1/(σ_∞ · (1 − η))   =   V_∞ / (1 − η)

so the apparent chargeability ``M_a = (V_η − V_∞) / V_η`` collapses to
the intrinsic η. The same identity (Seigel 1959; Newmont convention)
generalises to non-uniform media as the linearised approximation that
IP inversions rely on.

This operator runs DC2D twice, once with σ_∞ and once with the
"loaded" σ_eff = σ_∞ · (1 − η). The shared 5-point FV stencil and
symmetric Dirichlet pin come from :func:`assemble_dc2d_system` (EM-T2);
the autograd-friendly solve from :func:`linear_solve_with_adjoint`.

Architectural notes:

- The diagnostic fields ``V_primary`` and ``V_steady`` are exposed
  alongside the apparent chargeability, useful for sanity-checking the
  two DC passes during inversion debugging.
- An ``eta_max`` constructor knob bounds the allowed chargeability
  range (default ``1 − 1e-12`` to keep σ_eff > 0), with a
  rejection-vs-clamp policy: raise rather than silently truncate.
- A ``dirichlet_idx`` constructor knob threads through to both DC
  solves so the pin choice is consistent across the V_∞ / V_η pair.

Input field names follow the conventions (``sigma_infty``,
``chargeability``) rather than the (``sigma``, ``eta``). The
core framework's ``_check_trainable_inputs`` enforces those names at
the boundary; the underlying physics is identical to the IPOperator.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import ClassVar

import torch
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from ....core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    GeoBrainError,
    ModelState,
    ForwardOperator,
    ForwardOutput,
    linear_solve_with_adjoint,
)
from ....mesh import TensorMesh
from ....mesh import canonicalize_cell_field
from ....mesh.capabilities import ConnectivityMesh, StructuredMesh, UniformMesh
from ....core.survey import coords_to_cell_indices
from .dc2d import DC2DSurvey, assemble_dc2d_system
from .dc3d import BoundDipoleDipoleSurvey, DipoleDipoleSurvey


def _dc2d_voltage(
    sigma: torch.Tensor,
    mesh: TensorMesh,
    survey: DC2DSurvey,
    *,
    dirichlet_idx: int | None = None,
) -> torch.Tensor:
    """
    Single DC2D potential at the survey's receivers.

    Internal helper shared between the σ_∞ and σ_eff passes, uses the
    new symmetric-pin assembly from :mod:`dc2d`. Returns the per-receiver
    voltage; gradients flow through ``sigma`` via the implicit-function
    adjoint inside :func:`linear_solve_with_adjoint`.
    """
    nz, nx = mesh.shape
    n = nz * nx
    dz, dx = mesh.spacing
    oz, ox = mesh.origin
    device, dtype = sigma.device, sigma.dtype

    pin = (
        dirichlet_idx
        if dirichlet_idx is not None
        else (nz - 1) * nx + (nx // 2)
    )
    A = assemble_dc2d_system(sigma, dx=dx, dz=dz, dirichlet_idx=pin)

    # Electrode positions are physical metres; snap each to its cell.
    sz = coords_to_cell_indices(survey.source_z, dz, nz, origin=oz)
    sx = coords_to_cell_indices(survey.source_x, dx, nx, origin=ox)
    kz = coords_to_cell_indices(survey.sink_z, dz, nz, origin=oz)
    kx = coords_to_cell_indices(survey.sink_x, dx, nx, origin=ox)
    src_flat = sz * nx + sx
    snk_flat = kz * nx + kx
    b = torch.zeros(n, dtype=dtype, device=device).scatter_add(
        0,
        torch.tensor([src_flat, snk_flat], device=device, dtype=torch.long),
        torch.tensor(
            [+survey.current, -survey.current],
            dtype=dtype, device=device,
        ),
    )
    if src_flat == pin or snk_flat == pin:
        b = b.clone()
        b[pin] = 0.0

    phi = linear_solve_with_adjoint(A, b)
    phi_2d = phi.view(nz, nx)
    rcv_z = coords_to_cell_indices(survey.rcv_z, dz, nz, origin=oz).to(device)
    rcv_x = coords_to_cell_indices(survey.rcv_x, dx, nx, origin=ox).to(device)
    return phi_2d[rcv_z, rcv_x]


class IP2D(EMOperatorDiscovery, ForwardOperator):
    """
    Time-domain Induced Polarization (TDIP) 2D operator.

    Inputs (ModelState):
        - ``sigma_infty``: full-frequency conductivity (S/m), shape
          ``mesh.shape``.
        - ``chargeability``: intrinsic chargeability η ∈ [0, eta_max]
          per cell.

    Output (ForwardOutput):
        - ``data["chargeability_obs"]``: apparent chargeability at each
          receiver (dimensionless); the sole observation channel.
        - ``fields["V_primary"]``: V_∞ at each receiver (diagnostic).
        - ``fields["V_steady"]``: V_η at each receiver (diagnostic).

    Constructor:
        - ``survey``: :class:`DC2DSurvey` defining electrodes.
        - ``dirichlet_idx``: optional pin override (default bottom-center
          of the 2D mesh; threaded through to both DC passes).
        - ``eta_max``: upper bound for η (default ``1 − 1e-12``); kept
          below 1.0 so σ_eff > 0 stays strictly positive.

    Both ``sigma_infty`` and ``chargeability`` carry gradients through
    the dual DC2D solve (`IMPLICIT_VJP`).

    Args:
        survey: 2-D galvanic acquisition.
        dirichlet_idx: node pinned to zero potential (``None`` = auto).
        eta_max: chargeability ceiling keeping (1 - eta) positive.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma_infty", "chargeability"),
        output_keys=("chargeability_obs",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (UniformMesh,)

    def __init__(
        self,
        survey: DC2DSurvey,
        *,
        dirichlet_idx: int | None = None,
        eta_max: float = 1.0 - 1e-12,
    ) -> None:
        super().__init__()
        if not isinstance(survey, DC2DSurvey):
            raise GeoBrainError(
                "IP2D requires a DC2DSurvey",
                object_name="IP2D", field="survey",
                expected=DC2DSurvey, actual=type(survey),
            )
        if not (0.0 < eta_max < 1.0):
            raise GeoBrainError(
                "eta_max must lie in (0, 1)",
                object_name="IP2D",
                field="eta_max",
                expected="(0, 1)",
                actual=eta_max,
            )
        self.survey = survey
        self._dirichlet_idx = dirichlet_idx
        self.eta_max = float(eta_max)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        sigma_inf, m = state.fetch("sigma_infty", "chargeability")

        mesh = ctx.require_mesh()
        if mesh.n_dim != 2:
            raise GeoBrainError(
                "IP2D requires a 2D mesh",
                object_name="IP2D", field="mesh",
                expected="2D", actual=mesh.n_dim,
            )
        if tuple(sigma_inf.shape) != mesh.shape:
            raise GeoBrainError(
                "sigma_infty shape mismatches mesh.shape",
                object_name="IP2D", field="sigma_infty",
                expected=mesh.shape, actual=tuple(sigma_inf.shape),
            )
        if m.shape != sigma_inf.shape:
            raise GeoBrainError(
                "chargeability and sigma_infty must share shape",
                object_name="IP2D", field="chargeability",
                expected=tuple(sigma_inf.shape), actual=tuple(m.shape),
            )
        # Range check. The detached min/max keeps autograd uncoupled from
        # the validation branch.
        m_min = float(m.detach().min())
        m_max = float(m.detach().max())
        if m_min < 0.0 or m_max > self.eta_max:
            raise GeoBrainError(
                f"chargeability must lie in [0, {self.eta_max}]",
                object_name="IP2D",
                field="chargeability",
                expected=f"[0, {self.eta_max}]",
                actual=(m_min, m_max),
            )

        pin = self._dirichlet_idx  # None → DC2D applies its default

        # Pass A: primary potential at full conductivity σ_∞.
        V_p = _dc2d_voltage(sigma_inf, mesh, self.survey, dirichlet_idx=pin)

        # Pass B: steady-state at loaded conductivity σ_eff = σ_∞ · (1 − η).
        # The torch op routes autograd through both σ_∞ and η via the
        # chain rule.
        sigma_s = sigma_inf * (1.0 - m)
        V_s = _dc2d_voltage(sigma_s, mesh, self.survey, dirichlet_idx=pin)

        # Apparent chargeability. Guard against V_s collapsing through
        # zero (geometrically possible only if an electrode lands on the
        # pin); a tiny floor keeps the ratio finite.
        eps = 1e-12
        denom = torch.where(V_s.abs() > eps, V_s, torch.full_like(V_s, eps))
        M_obs = (V_s - V_p) / denom

        return ForwardOutput(
            data={"chargeability_obs": M_obs},
            fields={"V_primary": V_p, "V_steady": V_s},  # diagnostics, not observations
            metadata={
                "n_rcv": self.survey.n_rcv,
                "current": self.survey.current,
                "eta_max": self.eta_max,
            },
        )


# ---------------------------------------------------------------------------
# EM3: 3D Induced Polarization
# ---------------------------------------------------------------------------


from dataclasses import dataclass  # noqa: E402  (after-imports OK)
import numpy as np  # noqa: E402

from ....core import DifferentiabilityLevel, DifferentiabilitySpec  # noqa: E402
from ..numerics.finite_volume._sigma_jacobian import (  # noqa: E402
    build_sigma_jacobian_dc3d,
)


@dataclass
class IPChargeabilityModel:
    """
    Cell-centred ``(σ, η)`` model on a 3-D :class:`TensorMesh`.

    Keeps σ
    on the :class:`ModelState` (the standard EM operator backbone) and
    holds only the IP-specific extras (mesh + sigma + chargeability)
    on the model dataclass for convenience.

    Args:
        mesh: 3-D :class:`TensorMesh`.
        sigma: cell-centred conductivity (S/m), shape
            ``(n_cells,)``, ``torch.float64``. May carry
            ``requires_grad=True``.
        chargeability: cell-centred chargeability ``η ∈ [0, 1)``, same
            shape and dtype as ``sigma``.
    """

    mesh: TensorMesh
    sigma: torch.Tensor
    chargeability: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, TensorMesh):
            raise GeoBrainError(
                "IPChargeabilityModel requires a TensorMesh",
                object_name="IPChargeabilityModel",
                field="mesh",
                expected="TensorMesh",
                actual=type(self.mesh).__name__,
            )
        if self.mesh.n_dim != 3:
            raise GeoBrainError(
                "IPChargeabilityModel requires a 3-D TensorMesh",
                object_name="IPChargeabilityModel",
                field="mesh.n_dim",
                expected=3,
                actual=self.mesh.n_dim,
            )
        nz, nx, ny = self.mesh.shape
        n_cells = nz * nx * ny
        for name, t in (("sigma", self.sigma), ("chargeability", self.chargeability)):
            if not isinstance(t, torch.Tensor):
                raise GeoBrainError(
                    f"IPChargeabilityModel.{name} must be a torch.Tensor",
                    object_name="IPChargeabilityModel",
                    field=name,
                    expected="torch.Tensor",
                    actual=type(t).__name__,
                )
            if t.dtype != torch.float64:
                raise GeoBrainError(
                    f"IPChargeabilityModel.{name} must be float64",
                    object_name="IPChargeabilityModel",
                    field=f"{name}.dtype",
                    expected="float64",
                    actual=str(t.dtype),
                )
            if t.dim() != 1 or int(t.numel()) != n_cells:
                raise GeoBrainError(
                    f"IPChargeabilityModel.{name} must be 1-D with length n_cells",
                    object_name="IPChargeabilityModel",
                    field=f"{name}.shape",
                    expected=(n_cells,),
                    actual=tuple(t.shape),
                )


class IP3D(EMOperatorDiscovery, ForwardOperator):
    """
    3-D time-domain Induced Polarization operator.

    Implements the standard
    two-pass DC formulation (Seigel 1959; Oldenburg & Li 1994):

    1. Solve DC with conductivity ``σ`` → ``V_∞`` (high-frequency /
       instantaneous voltage; IP cells behave as conductors).
    2. Solve DC with ``σ_eff = σ · (1 − η)`` → ``V_η`` (DC steady-state
       voltage; IP cells are "loaded").
    3. Apparent chargeability per quadripole:
       ``M_a = (V_η − V_∞) / V_η``.

    Both solves share the FV stencil, the electrode→cell binding and
    the closed-form σ-Jacobian sparse-LU autograd bridge with
    :class:`DC3D`. They cannot share the LU factorisation itself;
    ``A(σ)`` and ``A(σ_eff)`` are different matrices.

    Inputs (ModelState):
        - ``sigma_infty`` (or ``sigma``): full-frequency conductivity
          (S/m), shape ``(n_cells,)`` ``float64``.
        - ``chargeability`` (or ``eta``): intrinsic ``η ∈ [0, eta_max]``,
          same shape and dtype.

    Outputs (ForwardOutput):
        - ``data["chargeability_obs"]``: apparent chargeability ``M_a``
          per quadripole, shape ``(n_obs,)``.
        - ``fields["V_primary"]``: primary voltages, shape ``(n_obs,)``
          (diagnostic).
        - ``fields["V_steady"]``: steady-state voltages, shape
          ``(n_obs,)`` (diagnostic).

    Constructor:
        - ``survey``: :class:`DipoleDipoleSurvey` (IP and DC share the
          same ABMN-quadripole electrode geometry).
        - ``dirichlet_idx``: optional pin override (default
          bottom-back-right corner; threaded through both DC passes).
        - ``current``: injection current magnitude (A); default ``1.0``.
          Cancels in the ratio so the value is irrelevant to ``M_a``.
        - ``eta_max``: upper bound for ``η`` (default ``1 − 1e-12``).

    Mesh contract:
        IP3D is **ctx-threaded**: it discretises on the context mesh
        ``ctx.require_mesh()``: any 3-D mesh with the ``ConnectivityMesh``
        capability (both the structured ``TensorMesh`` and a fully unstructured
        ``UnstructuredMesh`` / octree are supported), enforced by the base
        :meth:`ForwardOperator.forward` via ``requires_mesh_capabilities`` plus
        an explicit dimensionality check; ``ConnectivityMesh`` is required
        because the FV assembly routes through ``mesh.face_neighbors()``,
        matching SIP / DC3D. The survey is pure geometry (positions + ABMN
        quadripoles + binding mode) and carries no mesh; the electrode→cell
        binding and the Dirichlet pin are computed against the **ctx** mesh at
        forward time via :meth:`DipoleDipoleSurvey.bind` (on a structured mesh
        trilinear or nearest; on an unstructured mesh ``binding="nearest"`` is
        required; trilinear has no 8-corner cell box there). σ_∞ / η are flat
        ``(n_cells,)`` in ``cell_centers()`` order on any mesh, or ``(nz, nx,
        ny)`` on a structured mesh. On the structured **CPU** path the σ-gradient
        flows through the closed-form sparse Jacobian + splu σ-bridge; on the
        unstructured path (and on CUDA for any mesh) it flows through autograd on
        the FV ``A.values()``. The mesh contract is uniform with every other
        ctx-threaded operator.

    Device:
        GPU-capable via :class:`~geobrain.core.adjoint.PcgGpuSolver`, mirroring
        DC3D: IP3D is **no longer unconditionally CPU-only**. Both DC passes
        solve the same real symmetric-positive-definite FV Poisson as DC, so
        installing the GPU sparse backend
        (``set_default_sparse_solver(PcgGpuSolver())``) runs the whole forward +
        adjoint on CUDA: FV assembly, both SPD solves, receiver M/N sampling and
        the σ / η gradients all stay on-device. The structured closed-form
        σ-Jacobian is a **CPU-only fast path** (assembled in numpy/scipy,
        byte-identical to the validated structured reference gradient); on CUDA the
        σ-gradient routes through the device-preserving autograd bridge on
        ``A.values()`` instead, which gives the mathematically-identical exact
        adjoint. The scipy CPU default is unchanged, and a CUDA input on the
        default scipy backend is rejected by the shared bridge with an actionable
        error naming ``PcgGpuSolver`` (no silent CPU fallback). Caveat:
        ``PcgGpuSolver`` is Jacobi-preconditioned CG, so CUDA parity holds to the
        PCG tolerance (~1e-6..1e-9), not to bit-for-bit LU.

    Args:
        survey: 3-D dipole-dipole acquisition.
        dirichlet_idx: node pinned to zero potential (``None`` = auto).
        current: source current [A].
        eta_max: chargeability ceiling keeping (1 - eta) positive.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma_infty", "chargeability"),
        output_keys=("chargeability_obs",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (ConnectivityMesh,)

    def __init__(
        self,
        survey: DipoleDipoleSurvey,
        *,
        dirichlet_idx: int | None = None,
        current: float = 1.0,
        eta_max: float = 1.0 - 1e-12,
    ) -> None:
        super().__init__()
        if not isinstance(survey, DipoleDipoleSurvey):
            raise GeoBrainError(
                "IP3D requires a DipoleDipoleSurvey",
                object_name="IP3D",
                field="survey",
                expected="DipoleDipoleSurvey",
                actual=type(survey).__name__,
            )
        if not (0.0 < eta_max < 1.0):
            raise GeoBrainError(
                "eta_max must lie in (0, 1)",
                object_name="IP3D",
                field="eta_max",
                expected="(0, 1)",
                actual=eta_max,
            )
        self.survey = survey
        self.current = float(current)
        self.eta_max = float(eta_max)
        # Pin choice. ``None`` → default bottom-back-right corner, resolved
        # against the CTX mesh at forward time (the survey's constructor mesh is
        # not the authority). An explicit override is stored as-is.
        self._dirichlet_idx = None if dirichlet_idx is None else int(dirichlet_idx)
        # Per-call binding cache: the electrode→cell binding is mesh-specific, so
        # it is (re)built against the ctx mesh on the first forward that sees a
        # given mesh and reused while the same mesh recurs (keyed on identity,
        # mirroring TEM3D's ``_ensure_assembly``). Holds the ctx-bound survey and
        # the resolved pin.
        self._bound_mesh: object = None
        self._bound_survey: BoundDipoleDipoleSurvey | None = None
        self._pin: int | None = None

    # ------------------------------------------------------------------
    # Internal: one DC solve with closed-form-σ-jacobian autograd
    # ------------------------------------------------------------------

    def _bind_to_mesh(self, mesh: TensorMesh) -> BoundDipoleDipoleSurvey:
        """
        Return the survey bound to ``mesh`` (the CTX mesh), caching per mesh.

        The construction survey is pure geometry (positions, ABMN quadripoles,
        binding mode); the electrode→cell binding and the Dirichlet pin are
        mesh-specific, so they are computed against the **ctx** mesh here. A call
        to :meth:`DipoleDipoleSurvey.bind` reuses the original
        electrodes/quadripoles/binding and builds the trilinear / nearest cell
        indices, returning a :class:`BoundDipoleDipoleSurvey`. The result (plus
        the resolved pin) is cached on the instance keyed by mesh identity,
        mirroring TEM3D's ``_ensure_assembly``.
        """
        if self._bound_mesh is mesh and self._bound_survey is not None:
            return self._bound_survey
        bound = self.survey.bind(mesh)
        if self._dirichlet_idx is not None:
            pin = int(self._dirichlet_idx)
        elif mesh.declares(StructuredMesh):
            # Structured far-corner (bottom-back-right) cell: the historical
            # default, read from ``mesh.shape`` only behind the capability guard.
            nz, nx, ny = mesh.shape
            pin = int(ny - 1 + (nx - 1) * ny + (nz - 1) * nx * ny)
        else:
            # ConnectivityMesh-only: deepest cell (argmax of the depth
            # coordinate, column 0 of ``cell_centers()`` in depth-first
            # ``(z, x, y)`` order): DC3D's unstructured pin convention, far
            # from the surface electrodes.
            pin = int(torch.argmax(mesh.cell_centers()[:, 0]))
        self._bound_mesh = mesh
        self._bound_survey = bound
        self._pin = pin
        return bound

    def _solve_dc(
        self,
        sigma_flat_t: torch.Tensor,
        b_torch: torch.Tensor,
        mesh,
    ) -> torch.Tensor:
        """
        Solve ``A(σ_flat) φ = b_torch`` with σ-gradient autograd.

        ``A`` is assembled via the mesh-agnostic finite-volume Poisson seam
        :func:`assemble_poisson_fv` (``mesh.face_neighbors()`` + symmetric
        Dirichlet pin). The σ-gradient routes one of two ways depending on the
        mesh:

        - **StructuredMesh**: the closed-form sparse Jacobian
          (:func:`build_sigma_jacobian_dc3d`, TensorMesh-only; it reads
          ``mesh.cell_widths`` and the ``(nz, nx, ny)`` flat layout) feeds
          :class:`SparseLinearSolveWithSigmaGrad`. This keeps IP3D's structured
          output and gradient byte-identical to the validated reference path.
        - **ConnectivityMesh-only**: the σ-gradient flows through autograd on the
          FV ``A.values()`` via :func:`sparse_linear_solve_with_adjoint`, the
          same route DC3D's unstructured path takes. The closed-form Jacobian is
          structured-only, so it is not used here.

        ``mesh`` is the CTX mesh threaded in from :meth:`_forward`. Returns ``φ``
        of shape ``(n_cells, n_obs)``.
        """
        from ..numerics.finite_volume.poisson_fv import assemble_poisson_fv

        pin = self._pin
        A_coo = assemble_poisson_fv(
            mesh.face_neighbors(),
            sigma_flat_t,
            boundary={"dirichlet_idx": pin},
        )
        A_csr_pinned = A_coo.to_sparse_csr()

        is_structured = mesh.declares(StructuredMesh)
        # The closed-form numpy σ-Jacobian path is a CPU-only fast path: it
        # assembles the Jacobian and factors A in numpy/scipy on the host. It is
        # used ONLY when the mesh is structured AND σ is on CPU. Otherwise (an
        # unstructured mesh, OR a CUDA σ on any mesh) route through the autograd
        # bridge on A.values(): device-preserving, and mathematically the same
        # exact adjoint σ-gradient, just via autograd-through-A.values instead of
        # the closed form. On CUDA the installed GPU solver (PcgGpuSolver) runs
        # the SPD solve; on CPU the bridge is byte-identical to the closed form
        # only in the structured+CPU case, which is why that case keeps the
        # closed form below.
        if not is_structured or sigma_flat_t.device.type != "cpu":
            # Connectivity-generic / non-CPU path: autograd through A.values()
            # (real σ). Device-preserving; routes CUDA matrices to the installed
            # GPU sparse backend via the shared bridge.
            from ....core import sparse_linear_solve_with_adjoint

            return sparse_linear_solve_with_adjoint(A_csr_pinned, b_torch)

        # Structured CPU path: closed-form σ-Jacobian + splu σ-bridge (unchanged;
        # byte-identical to the validated structured reference gradient).
        import scipy.sparse as sp
        from geobrain.physics.em.numerics.sparse.lu_bridge import (
            SparseLinearSolveWithSigmaGrad,
        )

        n_cells = int(A_csr_pinned.shape[0])
        crow_np = A_csr_pinned.crow_indices().detach().cpu().numpy().astype(np.int32)
        col_np = A_csr_pinned.col_indices().detach().cpu().numpy().astype(np.int32)
        vals_np = A_csr_pinned.values().detach().cpu().numpy().astype(np.float64)
        A_csr_np = sp.csr_matrix((vals_np, col_np, crow_np), shape=(n_cells, n_cells))

        sigma_flat_np = sigma_flat_t.detach().cpu().numpy().astype(np.float64)
        jac_rows, jac_cols, jac_sig, jac_vals = build_sigma_jacobian_dc3d(
            mesh, sigma_flat_np, dirichlet_idx=pin,
        )

        return SparseLinearSolveWithSigmaGrad.apply(
            A_csr_np.data,
            A_csr_np.indices,
            A_csr_np.indptr,
            A_csr_np.shape,
            b_torch,
            sigma_flat_t,
            jac_rows,
            jac_cols,
            jac_sig,
            jac_vals,
        )

    def _sample_mn_diff(
        self, phi: torch.Tensor, survey: BoundDipoleDipoleSurvey
    ) -> torch.Tensor:
        """Sample ``φ_M − φ_N`` per quadripole from a full ``(n_cells, n_obs)`` field.

        ``survey`` is the ctx-bound survey (see :meth:`_bind_to_mesh`) carrying
        the electrode→cell binding resolved against the ctx mesh.
        """
        # Build the index / weight tensors on ``phi.device`` so the gather
        # ``phi[idx, obs_idx]`` stays on-device (CPU or CUDA). The numpy-built
        # indices/weights are host arrays; ``.to(phi.device)`` moves them once so
        # a CUDA ``phi`` never hits a cross-device index. CPU results are
        # unchanged (``.to("cpu")`` on a CPU tensor is a no-op).
        n_obs = survey.n_obs
        obs_idx = torch.arange(n_obs, dtype=torch.long, device=phi.device)
        if getattr(survey, "binding", "trilinear") == "trilinear":
            tri_c = survey.trilinear_cells
            tri_w = survey.trilinear_weights
            m_e = survey.quadripoles[:, 2]
            n_e = survey.quadripoles[:, 3]
            phi_m = torch.zeros(n_obs, dtype=phi.dtype, device=phi.device)
            phi_n = torch.zeros(n_obs, dtype=phi.dtype, device=phi.device)
            for c in range(8):
                m_idx = torch.from_numpy(
                    np.ascontiguousarray(tri_c[m_e, c], dtype=np.int64)
                ).to(phi.device)
                n_idx = torch.from_numpy(
                    np.ascontiguousarray(tri_c[n_e, c], dtype=np.int64)
                ).to(phi.device)
                m_w = torch.from_numpy(
                    np.ascontiguousarray(tri_w[m_e, c]),
                ).to(dtype=phi.dtype, device=phi.device)
                n_w = torch.from_numpy(
                    np.ascontiguousarray(tri_w[n_e, c]),
                ).to(dtype=phi.dtype, device=phi.device)
                phi_m = phi_m + m_w * phi[m_idx, obs_idx]
                phi_n = phi_n + n_w * phi[n_idx, obs_idx]
            return phi_m - phi_n
        # nearest binding
        cells = survey.electrode_cells
        m_cells = cells[survey.quadripoles[:, 2]]
        n_cells_ = cells[survey.quadripoles[:, 3]]
        m_idx = torch.from_numpy(
            np.ascontiguousarray(m_cells, dtype=np.int64)
        ).to(phi.device)
        n_idx = torch.from_numpy(
            np.ascontiguousarray(n_cells_, dtype=np.int64)
        ).to(phi.device)
        return phi[m_idx, obs_idx] - phi[n_idx, obs_idx]

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        # Resolve σ_∞: prefer the ``sigma_infty`` over the ``sigma``.
        # The :class:`ModelState` is contract-checked via the differentiability
        # spec; here we accept either alias so existing fixtures load unchanged.
        if "sigma_infty" in state.tensors:
            sigma_inf = state.tensors["sigma_infty"]
        elif "sigma" in state.tensors:
            sigma_inf = state.tensors["sigma"]
        else:
            raise GeoBrainError(
                "IP3D requires 'sigma_infty' (or 'sigma') in ModelState",
                object_name="IP3D",
                field="state",
                expected="'sigma_infty' or 'sigma'",
                actual=list(state.tensors.keys()),
            )
        if "chargeability" in state.tensors:
            eta = state.tensors["chargeability"]
        elif "eta" in state.tensors:
            eta = state.tensors["eta"]
        else:
            raise GeoBrainError(
                "IP3D requires 'chargeability' (or 'eta') in ModelState",
                object_name="IP3D",
                field="state",
                expected="'chargeability' or 'eta'",
                actual=list(state.tensors.keys()),
            )

        # Device policy is deferred to the shared sparse bridge (single source of
        # truth, like DC3D): on CPU the closed-form structured σ-Jacobian fast
        # path runs in numpy/scipy; on CUDA the σ-gradient routes through the
        # autograd bridge on ``A.values()`` (device-preserving), and the SPD solve
        # runs on the installed GPU backend (:class:`PcgGpuSolver`). A CUDA σ with
        # the default scipy backend is rejected by the bridge with an actionable
        # error naming PcgGpuSolver: no silent CPU fallback. Receiver sampling
        # (:meth:`_sample_mn_diff`) is device-preserving.

        # Mesh comes from the CTX (ctx-threaded contract). The base
        # ``forward`` already enforced the ``ConnectivityMesh`` capability; the
        # capability marker does not constrain dimensionality, so check 3-D here.
        mesh = ctx.require_mesh()
        if mesh.n_dim != 3:
            raise GeoBrainError(
                "IP3D requires a 3-D mesh",
                object_name="IP3D",
                field="mesh.n_dim",
                expected=3,
                actual=mesh.n_dim,
            )
        # Bind the survey (electrode→cell binding) and resolve the pin against
        # the CTX mesh.
        survey = self._bind_to_mesh(mesh)

        # Either flat (n_cells,) or the grid shape on a structured mesh; flat
        # only on a ConnectivityMesh-only mesh: the shared platform contract.
        sigma_inf = canonicalize_cell_field(
            mesh, sigma_inf.to(torch.float64), name="sigma_infty", owner="IP3D",
        )
        eta = canonicalize_cell_field(
            mesh, eta.to(torch.float64), name="chargeability", owner="IP3D",
        )

        # Range check on η.
        eta_min = float(eta.detach().min())
        eta_max = float(eta.detach().max())
        if eta_min < 0.0 or eta_max > self.eta_max:
            raise GeoBrainError(
                f"chargeability must lie in [0, {self.eta_max}]",
                object_name="IP3D",
                field="chargeability",
                expected=f"[0, {self.eta_max}]",
                actual=(eta_min, eta_max),
            )

        # Build RHS once: identical for both passes. Move to σ's device so a
        # CUDA σ assembles a CUDA A and pairs with a CUDA b (the FV assembly
        # follows σ.device); on CPU this is a no-op.
        b_np = survey.ab_currents(current=self.current)
        b_np[self._pin, :] = 0.0
        b_torch = torch.from_numpy(b_np).to(sigma_inf.device)

        # Pass A: steady-state DC at σ_∞.
        phi_inf = self._solve_dc(sigma_inf, b_torch, mesh)
        dV_inf = self._sample_mn_diff(phi_inf, survey)

        # Pass B: IP-loaded DC at σ_eff = σ_∞ · (1 − η).
        sigma_eff = sigma_inf * (1.0 - eta)
        phi_eta = self._solve_dc(sigma_eff, b_torch, mesh)
        dV_eta = self._sample_mn_diff(phi_eta, survey)

        # Apparent chargeability: Seigel sign convention
        # ``IPOperator`` and the reference implementation. Guard against ``dV_eta`` collapsing
        # through zero with a tiny floor.
        eps = 1e-12
        denom = torch.where(dV_eta.abs() > eps, dV_eta, torch.full_like(dV_eta, eps))
        M_a = (dV_eta - dV_inf) / denom

        return ForwardOutput(
            data={"chargeability_obs": M_a},
            fields={"V_primary": dV_inf, "V_steady": dV_eta},  # diagnostics, not observations
            metadata={
                "method": "ip_tdip_3d",
                "current": self.current,
                "n_obs": survey.n_obs,
                "eta_max": self.eta_max,
                "pin": self._pin,
            },
        )


class IPSimulator:
    """
    Convenience wrapper over :class:`IP3D`.

    Mirrors ``IPSimulator``, bakes the operator at construction
    time and exposes a one-liner ``dpred`` for inversion callers that
    don't want to thread ``ModelState`` / ``ForwardContext`` through
    their own code.

    Args:
        model: :class:`IPChargeabilityModel` carrying mesh + σ + η.
        survey: :class:`DipoleDipoleSurvey` (IP and DC share the same
            ABMN-quadripole electrode geometry).
        dirichlet_idx: optional pin override (default bottom-back-right
            corner of the mesh).
    """

    def __init__(
        self,
        model: IPChargeabilityModel,
        survey: DipoleDipoleSurvey,
        *,
        dirichlet_idx: int | None = None,
    ) -> None:
        if not isinstance(model, IPChargeabilityModel):
            raise GeoBrainError(
                "IPSimulator requires an IPChargeabilityModel",
                object_name="IPSimulator",
                field="model",
                expected="IPChargeabilityModel",
                actual=type(model).__name__,
            )
        self._op = IP3D(
            survey=survey, dirichlet_idx=dirichlet_idx,
        )
        self._model = model

    @property
    def operator(self) -> IP3D:
        return self._op

    def dpred(
        self,
        sigma: torch.Tensor | None = None,
        chargeability: torch.Tensor | None = None,
    ) -> ForwardOutput:
        """Forward pass; returns the full :class:`ForwardOutput`."""
        s = self._model.sigma if sigma is None else sigma
        e = self._model.chargeability if chargeability is None else chargeability
        state = ModelState({"sigma_infty": s, "chargeability": e})
        # IP3D is ctx-threaded: thread the model's mesh through the context.
        ctx = ForwardContext.of(mesh=self._model.mesh)
        return self._op(state, ctx)
