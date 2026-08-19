# pyright: reportPrivateImportUsage=false
"""
Spectral induced polarization (SIP) operator.

Cole-Cole frequency-dependent complex chargeability + two-pass DC
formulation. Both passes route through the mesh-agnostic
``assemble_poisson_fv`` seam; pass A on the real-σ branch, pass B on the
complex-σ branch (``σ_eff(ω) = σ(1 − η_eff(ω))``), so SIP runs on any
``ConnectivityMesh`` (structured ``TensorMesh`` or fully unstructured).

References: Cole & Cole 1941; Pelton et al. 1978 (IP applications).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
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
    sparse_linear_solve_with_adjoint,
)
from ....mesh import canonicalize_cell_field
from ....mesh.capabilities import ConnectivityMesh, StructuredMesh
from ..surveys import FrequencyDomainSurvey


def cole_cole_eta(
    omega: torch.Tensor,
    eta_0: torch.Tensor,
    tau: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    Cole-Cole effective complex chargeability ``η_eff(ω)``.

    ::

        η_eff(ω) = η₀ · (1 - 1/(1 + (iωτ)^c))

    For ω = 0: η_eff = 0 (no polarisation at DC).
    For ω → ∞: η_eff → η₀ (saturation).

    Args:
        omega: ``(n_freq,)`` real angular frequencies (rad/s).
        eta_0: scalar or shape-broadcasting tensor, DC chargeability,
            typically in [0, 1).
        tau:   scalar or shape-broadcasting, relaxation time (s), > 0.
        c:     scalar or shape-broadcasting, frequency-broadening exponent,
            typically in (0, 1] (1 = Debye, smaller = broader spectrum).

    Returns:
        ``(n_freq, ...)`` complex128 tensor (broadcasting against η₀/τ/c shape).
    """
    omega_c = omega.to(torch.complex128)
    eta_0_c = eta_0.to(torch.complex128)
    tau_c = tau.to(torch.complex128)
    c_c = c.to(torch.complex128)

    iωτ = 1j * omega_c * tau_c
    # (iωτ)^c using complex pow.
    iωτ_pow_c = iωτ ** c_c
    return eta_0_c * (1.0 - 1.0 / (1.0 + iωτ_pow_c))


# ---------------------------------------------------------------------------
# Cole-Cole model dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIPColeColeModel:
    """
    Cell-centred Cole-Cole parameter triple ``(η₀, τ, c)``.

    Frozen dataclass holding the three Cole-Cole parameters as torch
    tensors. ``σ`` is intentionally NOT carried here, conductivity
    flows through ``ModelState`` (matching the convention where the
    operator dataclass carries only the model-specific extras, not the
    backbone σ tensor that every EM operator already consumes). The
    IMPLICIT_VJP architecture moves σ to ``ModelState`` (handled by
    ``state.fetch("sigma")`` inside ``_forward``) and relies on torch's
    ``requires_grad`` on the underlying tensors. The fields ``eta_0``,
    ``tau``, ``c`` keep their conventional names.

    Attributes:
        eta_0: DC chargeability ``∈ [0, 1)``. Scalar or per-cell tensor.
        tau:   Relaxation time constant in seconds, strictly positive.
            Scalar or per-cell tensor.
        c:     Frequency-broadening exponent ``∈ (0, 1]`` (1 ≡ Debye,
            smaller ≡ broader spectrum). Scalar or per-cell tensor.
    """

    eta_0: torch.Tensor
    tau: torch.Tensor
    c: torch.Tensor


# ---------------------------------------------------------------------------
# Survey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIPSurvey(FrequencyDomainSurvey):
    """
    SIP survey: ABMN electrode quadripoles + frequency sweep.

    Inherits ``sources``, ``receivers`` and ``frequencies`` (in Hz) from
    :class:`FrequencyDomainSurvey`. Extends with four index tuples that
    select A/B/M/N electrodes per quadripole from the parent's
    ``sources``/``receivers`` lists (mirroring the
    :class:`~geobrain.physics.em.surveys.GalvanicSurvey`
    naming so an importer can map the ABMN quadripole matrix straight onto
    the fields). The four-tuple split here is the convention already used by
    :class:`GalvanicSurvey`.

    Attributes:
        a_electrode_idx, b_electrode_idx: current-injection electrode
            indices (A = +I source, B = -I sink) per quadripole.
        m_electrode_idx, n_electrode_idx: potential-measurement
            electrode indices (V = phi_M - phi_N) per quadripole.
        frequencies: inherited, survey frequencies in **Hz** (not
            angular). Operator converts to ω = 2π f.
    """

    a_electrode_idx: tuple[int, ...] = ()
    b_electrode_idx: tuple[int, ...] = ()
    m_electrode_idx: tuple[int, ...] = ()
    n_electrode_idx: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # ``FrequencyDomainSurvey`` itself does not declare
        # ``__post_init__``; the abstract guard lives on ``EMSurvey``
        # and trips only when ``type(self) is EMSurvey``. So we can
        # validate here freely without chaining a missing parent hook.
        n = len(self.a_electrode_idx)
        for name in ("b_electrode_idx", "m_electrode_idx", "n_electrode_idx"):
            other = getattr(self, name)
            if len(other) != n:
                raise GeoBrainError(
                    f"SIPSurvey.{name} length must match a_electrode_idx",
                    object_name="SIPSurvey",
                    field=name,
                    expected=n,
                    actual=len(other),
                )

    @property
    def n_quadripoles(self) -> int:
        return len(self.a_electrode_idx)

    @property
    def n_freq(self) -> int:
        return len(self.frequencies)


# ---------------------------------------------------------------------------
# SIP operator
# ---------------------------------------------------------------------------


class SIP(EMOperatorDiscovery, ForwardOperator):
    """Spectral IP forward operator: Cole-Cole + two-pass DC.

    Device: CPU / float64 only (scipy sparse LU); move inputs to CPU first.

    Maps a :class:`ModelState` with ``sigma`` (real σ_∞, shape
    ``mesh.shape``) plus Cole-Cole ``eta_0``, ``tau``, ``c`` (per-cell
    or scalar, shape-broadcasting) to a :class:`ForwardOutput` whose
    ``data`` carries the complex apparent-chargeability matrix as a single
    native-complex channel ``data["m_app"]`` of shape
    ``(n_quadripoles, n_freq)`` (the platform-wide complex-data contract).

    Differentiability: :attr:`IMPLICIT_VJP` through all four parameters
    (σ, η₀, τ, c). Pass A's σ-gradient flows through the sparse
    adjoint (real CSR path); pass B's complex σ_eff(ω) gradient flows
    through the COO complex assembly + E5 adjoint, and the Cole-Cole
    chain rule routes (η₀, τ, c) gradients via standard torch autograd
    through :func:`cole_cole_eta`.

    Mesh contract:
        SIP discretises on the **context mesh** ``ctx.require_mesh()``, any 3-D
        mesh with the ``ConnectivityMesh`` capability (both the structured
        ``TensorMesh`` and a fully unstructured ``UnstructuredMesh``/octree are
        supported), enforced by the base :meth:`ForwardOperator.forward` via
        ``requires_mesh_capabilities``; the FV assembly routes through
        ``mesh.face_neighbors()``. On a ``StructuredMesh`` σ may be supplied
        either ``(nz, nx, ny)`` or flat ``(n_cells,)``; on a
        ``ConnectivityMesh``-only mesh σ must be flat ``(n_cells,)`` in
        ``cell_centers()`` order, and the ABMN ``*_electrode_idx`` are flat cell
        indices into that order (compute them yourself, e.g. nearest-centre
        look-up). Only the survey is bound at construction; the mesh arrives
        per-call through the :class:`ForwardContext`, like every other
        ctx-threaded operator.

    Args:
        survey: spectral-IP acquisition.
        dirichlet_idx: node pinned to zero potential (``None`` = auto).
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma", "eta_0", "tau", "c"),
        output_keys=("m_app",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (ConnectivityMesh,)

    def __init__(
        self,
        survey: SIPSurvey,
        *,
        dirichlet_idx: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, SIPSurvey):
            raise GeoBrainError(
                "SIP requires a SIPSurvey",
                object_name="SIP",
                field="survey",
                expected=SIPSurvey,
                actual=type(survey),
            )
        self.survey = survey
        # Symmetric Dirichlet pin override. ``None`` resolves at forward time to
        # the structured far-corner cell (last flat index) on a StructuredMesh,
        # or the deepest cell (argmax depth) on a ConnectivityMesh-only mesh.
        self._dirichlet_idx = None if dirichlet_idx is None else int(dirichlet_idx)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        """
        Two-pass DC + Cole-Cole forward.

        For each ω the algorithm runs:

        1. **Pass A** (frequency-independent, computed once):
           solve ``A_real(σ) · φ_∞ = b`` with the real Poisson stencil.
        2. **Pass B** (per ω):
           solve ``A_complex(σ_eff(ω)) · φ_η = b`` with
           ``σ_eff(ω) = σ · (1 − η_eff(ω))`` (complex COO assembly).

        Apparent chargeability per quadripole follows the Seigel
        convention::

            m_app = (V_η − V_∞) / V_η

        where ``V_∞ = φ_∞[M] − φ_∞[N]`` and likewise for ``V_η``. Both
        passes share the same symmetric Dirichlet pin (the structured
        far-corner cell on a ``TensorMesh``, the deepest cell on a
        ``ConnectivityMesh``-only mesh, or an explicit ``dirichlet_idx``
        override) to remove the pure-Neumann nullspace.
        """
        sigma, eta_0, tau, c = state.fetch("sigma", "eta_0", "tau", "c")
        # CPU-only, by a real constraint (NOT just the default backend): SIP's
        # chargeable Pass B assembles a COMPLEX128 Poisson operator σ_eff(ω) and
        # solves it. The GPU PcgGpuSolver is a real-SPD CG backend and rejects
        # complex systems, so there is no GPU path for SIP today (an indefinite
        # complex operator needs a direct or Krylov GMRES/BiCGStab backend). Keep
        # the narrow per-operator guard here and defer to scipy on CPU.
        if sigma.device.type != "cpu":
            raise GeoBrainError(
                "SIP is CPU-only (its chargeable pass solves a complex system, "
                "which the real-SPD GPU CG backend cannot handle); "
                "move sigma to cpu first",
                object_name="SIP", field="sigma.device",
                expected="cpu", actual=sigma.device,
            )
        mesh = ctx.require_mesh()
        if mesh.n_dim != 3:
            raise GeoBrainError(
                "SIP requires a 3D mesh",
                object_name="SIP", field="mesh", expected="3D", actual=mesh.n_dim,
            )

        # ------------------------------------------------------------------
        # Validate inputs. σ is accepted either ``(nz, nx, ny)`` (StructuredMesh,
        # back-compat) or flat ``(n_cells,)`` (any ConnectivityMesh: the only
        # form a fully unstructured mesh can carry). Mirrors DC3D's dual-shape
        # handling: ``mesh.shape`` is read ONLY behind the structured-capability
        # guard so a ConnectivityMesh-only mesh (no ``shape``) is never touched.
        # ------------------------------------------------------------------
        is_structured = mesh.declares(StructuredMesh)
        n_cells = int(mesh.n_cells)
        sigma_flat = canonicalize_cell_field(mesh, sigma, name="sigma",
                                             owner="SIP")
        if sigma_flat.dtype != torch.float64:
            sigma_flat = sigma_flat.to(torch.float64)

        # ------------------------------------------------------------------
        # Symmetric Dirichlet pin to remove the Neumann nullspace. An explicit
        # override wins. Otherwise: on a StructuredMesh the default is the last
        # flat index: the bottom-back-right corner (y-fastest ``(nz, nx, ny)``
        # flat index ``iy + ix*ny + iz*nx*ny``); on a ConnectivityMesh-only mesh the
        # default is the deepest cell (argmax of the depth coordinate, column 0
        # of ``cell_centers()`` in the platform ``(z, x, y)`` column order),
        # the DC3D unstructured convention, a cell far from the surface
        # electrodes.
        # ------------------------------------------------------------------
        if self._dirichlet_idx is not None:
            pin = int(self._dirichlet_idx)
        elif is_structured:
            pin = n_cells - 1
        else:
            pin = int(torch.argmax(mesh.cell_centers()[:, 0]))
        if not (0 <= pin < n_cells):
            raise GeoBrainError(
                "SIP dirichlet_idx out of range",
                object_name="SIP",
                field="dirichlet_idx",
                expected=f"[0, {n_cells})",
                actual=pin,
            )

        # ------------------------------------------------------------------
        # Pass A: real DC potential φ_∞ for every quadripole, multi-RHS.
        # ------------------------------------------------------------------
        n_quad = self.survey.n_quadripoles
        rhs_real = torch.zeros(n_cells, n_quad, dtype=torch.float64)
        for q in range(n_quad):
            a_idx = int(self.survey.a_electrode_idx[q])
            b_idx = int(self.survey.b_electrode_idx[q])
            if not (0 <= a_idx < n_cells) or not (0 <= b_idx < n_cells):
                raise GeoBrainError(
                    "SIP electrode index out of range",
                    object_name="SIP",
                    field="a_electrode_idx/b_electrode_idx",
                    expected=f"[0, {n_cells})",
                    actual=(a_idx, b_idx),
                )
            rhs_real[a_idx, q] = +1.0
            rhs_real[b_idx, q] = -1.0
        # If an electrode collides with the pin we must zero that row of
        # the RHS so the pinned identity equation is consistent (φ_pin = 0).
        rhs_real[pin, :] = 0.0

        # Pass A's real DC reference solve routes through the mesh-agnostic
        # finite-volume Poisson seam. On a TensorMesh
        # ``assemble_poisson_fv(mesh.face_neighbors(), σ_flat,
        # boundary={"dirichlet_idx": pin})`` reproduces the legacy
        # ``assemble_poisson_3d`` + symmetric Dirichlet pin to atol 1e-12
        # (≈1e-15 in practice; the forms differ only by eps placement and float
        # reassociation), so SIP / DC3D / IP3D now share one assembler with
        # SIP's TensorMesh output unchanged. The real branch's symmetric
        # Dirichlet pin is now applied inside ``assemble_poisson_fv`` via its
        # ``boundary={"dirichlet_idx": pin}`` arg.
        # ``.to_sparse_csr()`` keeps the CSR layout the adjoint solve consumes.
        # σ flat index is y-fastest C-order on a TensorMesh (``reshape(-1)`` of
        # ``(nz, nx, ny)``) and ``cell_centers()`` order on a ConnectivityMesh,
        # matching ``face_neighbors()`` in both cases. The complex pass-B branch
        # below ALSO routes through the same seam (the complex-σ branch of
        # ``assemble_poisson_fv``), so SIP is mesh-agnostic end to end.
        from ..numerics.finite_volume.poisson_fv import assemble_poisson_fv

        face_records = mesh.face_neighbors()
        A_real_csr_pinned = assemble_poisson_fv(
            face_records,
            sigma_flat,
            boundary={"dirichlet_idx": pin},
        ).to_sparse_csr()
        phi_inf = sparse_linear_solve_with_adjoint(A_real_csr_pinned, rhs_real)
        # phi_inf shape: (n_cells, n_quad).

        # M / N selection indices (long, contiguous).
        m_idx = torch.as_tensor(self.survey.m_electrode_idx, dtype=torch.long)
        n_idx = torch.as_tensor(self.survey.n_electrode_idx, dtype=torch.long)
        if m_idx.numel() != n_quad or n_idx.numel() != n_quad:
            raise GeoBrainError(
                "SIP M/N index lengths must match quadripole count",
                object_name="SIP",
                field="m_electrode_idx/n_electrode_idx",
                expected=n_quad,
                actual=(int(m_idx.numel()), int(n_idx.numel())),
            )
        q_arange = torch.arange(n_quad, dtype=torch.long)
        V_inf = phi_inf[m_idx, q_arange] - phi_inf[n_idx, q_arange]
        V_inf_c = V_inf.to(torch.complex128)

        # ------------------------------------------------------------------
        # Pass B: complex DC at σ_eff(ω) per frequency.
        # ------------------------------------------------------------------
        omega_t = (
            2.0 * math.pi
            * torch.as_tensor(self.survey.frequencies, dtype=torch.float64)
        )
        rhs_complex = rhs_real.to(torch.complex128)

        sigma_flat_c = sigma_flat.to(torch.complex128)
        m_app_cols: list[torch.Tensor] = []
        for k in range(int(omega_t.numel())):
            omega_k = omega_t[k:k + 1]
            # cole_cole_eta signature: (omega, eta_0, tau, c). Output is
            # complex128, broadcasting against (eta_0/tau/c) shape. With
            # scalar params + 1-element omega, eta_eff_k is shape (1,).
            eta_eff_k = cole_cole_eta(omega_k, eta_0, tau, c)
            # σ_eff(ω) = σ · (1 − η_eff(ω)), held FLAT ``(n_cells,)`` so it feeds
            # the mesh-agnostic FV seam directly. With a scalar η_eff the
            # broadcast is trivial; per-cell η_eff (future work) would already be
            # length-n_cells here.
            sigma_eff = sigma_flat_c * (1.0 - eta_eff_k.reshape(()))

            # Pass B routes through the SAME finite-volume Poisson seam as pass A
            #, ``assemble_poisson_fv`` branches on ``sigma.is_complex()`` and
            # applies the symmetric Dirichlet pin internally, so the legacy
            # ``assemble_poisson_3d`` (structured-only) + ``_pin_sparse_coo``
            # pair is retired. On a TensorMesh the result reproduces the legacy
            # complex assembly to machine precision (verified by the SIP FV
            # regression golden); on a ConnectivityMesh-only mesh it is the
            # natural complex FV operator. The COO is fed to the adjoint solve
            # AS-IS: ``to_sparse_csr()`` has no autograd support for complex
            # dtype, and ``sparse_linear_solve_with_adjoint`` already accepts a
            # coalesced complex COO (this is exactly what the retired
            # ``_pin_sparse_coo`` returned).
            A_complex_pinned = assemble_poisson_fv(
                face_records,
                sigma_eff,
                boundary={"dirichlet_idx": pin},
            )
            phi_eta = sparse_linear_solve_with_adjoint(
                A_complex_pinned, rhs_complex,
            )
            V_eta = phi_eta[m_idx, q_arange] - phi_eta[n_idx, q_arange]

            # Seigel apparent chargeability ``m_app = (V_η − V_∞) / V_η``.
            m_app_cols.append((V_eta - V_inf_c) / V_eta)

        # Stack to (n_quad, n_freq) complex.
        m_app = torch.stack(m_app_cols, dim=1)

        return ForwardOutput(
            data={
                "m_app": m_app,
            },
            metadata={
                "method": "sip_cole_cole",
                "n_quadripoles": n_quad,
                "n_freq": int(omega_t.numel()),
                "frequencies_hz": tuple(float(f) for f in self.survey.frequencies),
                "pin": pin,
            },
        )
