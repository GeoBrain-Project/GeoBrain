# pyright: reportPrivateImportUsage=false
"""2D magnetotelluric: TE / TM / dual-mode complex Helmholtz solve.

For a plane wave normally incident on a 2D conductivity model σ(z, y),
two independent scalar PDEs govern the response:

- **TE** (E-field parallel to strike): ``E_x`` Helmholtz with σ on the
  diagonal::

      ∂²E_x/∂z² + ∂²E_x/∂y² + i·ω·μ₀·σ · E_x = 0

- **TM** (H-field parallel to strike): ``H_x`` diffusion with ``1/σ``
  face coefficients::

      ∂/∂z[(1/σ) ∂H_x/∂z] + ∂/∂y[(1/σ) ∂H_x/∂y] + i·ω·μ₀ · H_x = 0

Both share the boundary conditions baked into the Helmholtz
assemblers::

    z = 0          : u = 1                      (normalised incident plane wave)
    z = z_max      : u = 0                      (absorbing, model bottom)
    y = 0, y_max   : ∂u/∂y = 0                  (reflective sides)

The complex linear system is built dense via vectorised ``scatter_add``
and solved with :func:`linear_solve_with_adjoint`. Surface impedance is
extracted per mode at each receiver:

- TE: ``Z_TE = E_x / H_y`` with ``H_y = -∂E_x/∂z · (i·ω·μ₀)⁻¹``.
- TM: ``Z_TM = E_x / H_y`` with ``E_x = (1/σ_top) · ∂H_x/∂z`` (Ampère,
  quasi-static).

For a homogeneous half-space both modes reproduce the Cagniard apparent
resistivity ``ρ_a = |Z|² / (ω·μ₀) = 1/σ`` and the canonical ``+π/4`` phase
under the sole production ``exp(+iωt)`` convention.

Architecture additions:

- :class:`MT2D` now takes a ``mode`` argument (``"TE"``, ``"TM"``, or
  ``"both"``; default ``"both"``). Backwards-incompatible by design;
  callers that were silently TE-only should pass ``mode="TE"``
  explicitly. Output ``data`` dict gains per-mode suffixed keys
  (``apparent_resistivity_TE``, ``phase_TM``, ...) only in the dual-mode
  case; single-mode runs keep the existing unprefixed keys.
- **Dual mesh paths**: the capability contract is ``ConnectivityMesh``.
  A UNIFORM mesh that also declares ``StructuredMesh`` (TensorMesh) takes the
  original dense complex64 path above, bit-identically (a graded TensorMesh
  takes an in-branch sparse complex128 graded stencil; see below). Any other 2-D
  ``ConnectivityMesh`` (triangle :class:`UnstructuredMesh`, …) takes the
  sparse complex128 cell-centred FV path assembled by
  :mod:`~geobrain.physics.em.numerics.finite_volume.helmholtz_fv`
  (TPFA flux + cell-diagonal mass, half-cell-transmissibility plane-wave
  Dirichlet at top/bottom, natural no-flux sides) and solved with
  :func:`sparse_linear_solve_with_adjoint` (CPU / scipy splu). On that
  path ``sigma`` is flat ``(n_cells,)`` and stations are PHYSICAL
  horizontal coordinates ``MT2DSurvey.rcv_y_m`` (metres), each bound to
  its nearest top boundary face. The output ``data`` schema (keys and
  ``(n_freq, n_rcv)`` shapes) is identical to the structured path.
- :func:`compute_jacobian` returns a dense ``(n_freq · n_rcv, n_cells)``
  σ-Jacobian of ``log|Z|`` per mode. It is an **autograd convenience
  wrapper** (``torch.autograd.functional.jacobian`` driving the
  operator's IMPLICIT_VJP adjoint row by row), *not* a hand-derived
  closed form, and the :class:`~geobrain.optim.Inverter` does not call
  it; it is a standalone diagnostic for materialising the full
  Jacobian on small (≲ 40 × 40) meshes.
- Mesh paths (three, dispatched by capability then uniformity): a UNIFORM
  TensorMesh takes the dense complex64 stencil (divides by a single
  ``dz``/``dy`` per axis; GPU-capable). A GRADED TensorMesh (non-uniform
  ``cell_widths``, still declaring ``StructuredMesh``) is NOT rejected; it
  takes an in-branch sparse complex128 graded stencil with per-axis face
  transmissibilities (``assemble_mt2d_system_graded``), CPU-only (scipy splu).
  A genuinely unstructured 2-D ``ConnectivityMesh`` (triangle
  ``UnstructuredMesh``) takes the cell-centred FV path
  (``_forward_connectivity``), which reads per-face records.

Operator level: ``IMPLICIT_VJP``. Dense ``(N, N)`` complex memory
scales as ``N²``, so practical grids stay ≲ 40 × 40.

Time convention: MT2D solves and returns the sole production
``exp(+iωt)`` convention.  Half-space phase is ``+π/4`` and all phase output
uses the principal radian interval.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

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
    sparse_linear_solve_with_adjoint,
)
from ....mesh import TensorMesh
from ....core.adjoint import default_sparse_solver_is_cpu_only
from ....mesh import require_field_matches_mesh
from ....mesh.capabilities import ConnectivityMesh, StructuredMesh
from ....core.linalg import block_diag
from ..numerics.finite_volume.helmholtz_2d import (
    assemble_helmholtz_2d,
    assemble_helmholtz_2d_graded,
    assemble_helmholtz_2d_tm,
    assemble_helmholtz_2d_tm_graded,
)
from ..conventions import MU_0
from ..numerics.finite_volume.helmholtz_fv import (
    assemble_helmholtz_fv,
    assemble_helmholtz_fv_tm,
    partition_helmholtz_boundary,
)


MU0 = MU_0  # platform canon, single-sourced in core.constants

MT2DMode = Literal["TE", "TM", "both"]


@dataclass(frozen=True)
class MT2DSurvey:
    """
    MT 2D survey geometry.

    Naming note: the "y" in ``rcv_y``/``rcv_y_m`` is the MT-literature
    profile (strike-normal) coordinate, which lies on platform mesh
    **axis-1** of the ``(nz, nx)`` grid, ``rcv_y`` indexes that axis,
    ``rcv_y_m`` measures metres along it.

    Attributes:
        frequencies: 1-D real tensor of frequencies (Hz).
        rcv_y: optional 1-D long tensor of receiver column indices on the
            surface row (i = 0). Required by, and only meaningful on,
            the structured (TensorMesh) path, where stations address grid
            columns.
        rcv_y_m: optional 1-D float tensor of PHYSICAL station horizontal
            coordinates (metres, mesh column 1). Required by, and only
            meaningful on, the connectivity (unstructured FV) path,
            where column indices do not exist; each station binds to the
            top boundary face whose centroid is nearest in y.

    At least one of ``rcv_y`` / ``rcv_y_m`` must be supplied (which one
    the forward actually consumes depends on the mesh path).
    """

    frequencies: torch.Tensor
    rcv_y: torch.Tensor | None = None
    rcv_y_m: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.frequencies, torch.Tensor)
            or self.frequencies.ndim != 1
        ):
            raise GeoBrainError(
                "MT2DSurvey.frequencies must be a 1D Tensor",
                object_name="MT2DSurvey", field="frequencies",
                expected="1D Tensor",
                actual=tuple(getattr(self.frequencies, "shape", ())),
            )
        if self.rcv_y is None and self.rcv_y_m is None:
            raise GeoBrainError(
                "MT2DSurvey needs station locations: set rcv_y (surface "
                "column indices, consumed by the structured TensorMesh "
                "path) and/or rcv_y_m (physical metres, consumed by the "
                "unstructured FV path)",
                object_name="MT2DSurvey", field="rcv_y",
                expected="rcv_y and/or rcv_y_m", actual=None,
            )
        if self.rcv_y is not None and (
            not isinstance(self.rcv_y, torch.Tensor) or self.rcv_y.ndim != 1
        ):
            raise GeoBrainError(
                "MT2DSurvey.rcv_y must be a 1D Tensor",
                object_name="MT2DSurvey", field="rcv_y",
                expected="1D Tensor",
                actual=tuple(getattr(self.rcv_y, "shape", ())),
            )
        if self.rcv_y_m is not None and (
            not isinstance(self.rcv_y_m, torch.Tensor)
            or self.rcv_y_m.ndim != 1
            or self.rcv_y_m.numel() == 0
        ):
            raise GeoBrainError(
                "MT2DSurvey.rcv_y_m must be a non-empty 1D Tensor when given",
                object_name="MT2DSurvey", field="rcv_y_m",
                expected="non-empty 1D Tensor",
                actual=tuple(getattr(self.rcv_y_m, "shape", ())),
            )
        if (self.frequencies <= 0).any():
            raise GeoBrainError(
                "frequencies must be positive",
                object_name="MT2DSurvey", field="frequencies",
                expected="> 0", actual=float(self.frequencies.min()),
            )

    @property
    def n_freq(self) -> int:
        return int(self.frequencies.numel())

    @property
    def n_rcv(self) -> int:
        """Receiver count from whichever station field the survey carries.

        ``rcv_y`` (structured column indices) when present, else
        ``rcv_y_m`` (FV physical coordinates), ``__post_init__``
        guarantees at least one of the two exists.
        """
        if self.rcv_y is not None:
            return int(self.rcv_y.numel())
        return int(self.rcv_y_m.numel())


def assemble_mt2d_system(
    sigma: torch.Tensor,
    omega: float,
    dz: float,
    dy: float,
    *,
    mode: Literal["TE", "TM"] = "TE",
) -> torch.Tensor:
    """
    Assemble the complex dense ``(N, N)`` MT2D Helmholtz operator.

    Thin MT2D-flavoured adapter over the
    :mod:`~...numerics.finite_volume.helmholtz_2d` stencils. ``mode="TE"``
    uses the σ-on-diagonal Helmholtz; ``mode="TM"`` uses the
    σ-on-face-coefficient diffusion form. Top (``i = 0``) and bottom
    (``i = nz − 1``) rows hold identity rows so the linear solve carries
    the Dirichlet plane-wave boundary values supplied in ``b``. Sides use
    reflective Neumann via index-clamped couplings.
    """
    nz, ny = sigma.shape
    mesh = TensorMesh(shape=(nz, ny), spacing=(dz, dy))
    if mode == "TE":
        return assemble_helmholtz_2d(mesh, sigma, omega, mu0=MU0)
    if mode == "TM":
        return assemble_helmholtz_2d_tm(mesh, sigma, omega, mu0=MU0)
    raise GeoBrainError(
        "assemble_mt2d_system: mode must be 'TE' or 'TM'",
        object_name="assemble_mt2d_system",
        field="mode",
        expected="'TE' or 'TM'",
        actual=mode,
    )


def assemble_mt2d_system_graded(
    sigma: torch.Tensor,
    omega: float,
    mesh: TensorMesh,
    *,
    mode: Literal["TE", "TM"] = "TE",
) -> torch.Tensor:
    """Assemble the SPARSE complex128 MT2D Helmholtz operator on a GRADED
    (non-uniform ``cell_widths``) TensorMesh.

    Graded-mesh counterpart of :func:`assemble_mt2d_system`: the same TE
    (σ-on-diagonal) / TM (σ-on-face) physics and Dirichlet-top/bottom +
    Neumann-side boundary closure, on per-cell widths, returned as a sparse COO
    tensor for the implicit-VJP sparse adjoint solve. A uniform-width mesh
    reproduces the dense path. Lets a single mesh be fine at the surface AND
    deep, the prerequisite for a broadband MT inversion."""
    if mode == "TE":
        return assemble_helmholtz_2d_graded(mesh, sigma, omega, mu0=MU0)
    if mode == "TM":
        return assemble_helmholtz_2d_tm_graded(mesh, sigma, omega, mu0=MU0)
    raise GeoBrainError(
        "assemble_mt2d_system_graded: mode must be 'TE' or 'TM'",
        object_name="assemble_mt2d_system_graded",
        field="mode",
        expected="'TE' or 'TM'",
        actual=mode,
    )


class MT2D(EMOperatorDiscovery, ForwardOperator):
    """
    2D magnetotelluric operator (IMPLICIT_VJP, complex-valued).

    Inputs (ModelState): ``sigma``, shape ``(nz, ny)``.

    Outputs (ForwardOutput): the channels depend on ``mode``.

    - ``mode="TE"`` or ``mode="TM"``:
        ``data["impedance"]``, ``data["apparent_resistivity"]``,
        ``data["phase"]``; plus the diagnostic ``fields["E_field"]`` (the
        solved-for scalar field on the mesh, ``E_x`` for TE, ``H_x`` for TM).
    - ``mode="both"`` (default):
        per-mode suffixed observables ``data["impedance_TE"]``,
        ``data["impedance_TM"]``, ``data["apparent_resistivity_TE"]``,
        ``data["apparent_resistivity_TM"]``, ``data["phase_TE"]``,
        ``data["phase_TM"]``; plus diagnostics ``fields["E_field_TE"]``,
        ``fields["H_field_TM"]``.

    Mesh dispatch: a mesh declaring ``StructuredMesh`` takes the original
    dense complex64 path (solved with :func:`linear_solve_with_adjoint`,
    O(N²) memory → grids ≲ 40 × 40); any other 2-D ``ConnectivityMesh``
    takes the sparse complex128 FV path (``sigma`` flat ``(n_cells,)``,
    stations via ``survey.rcv_y_m``, CPU-only scipy splu, diagnostic
    fields flat ``(n_freq, n_cells)``). ``data`` keys and shapes are
    identical across both paths.

    Args:
        survey: MT station/frequency acquisition.
        mode: ``'te'`` / ``'tm'`` / ``'both'``.
    """

    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (ConnectivityMesh,)

    def __init__(
        self,
        survey: MT2DSurvey,
        *,
        mode: MT2DMode = "both",
    ) -> None:
        super().__init__()
        if not isinstance(survey, MT2DSurvey):
            raise GeoBrainError(
                "MT2D requires an MT2DSurvey",
                object_name="MT2D", field="survey",
                expected=MT2DSurvey, actual=type(survey),
            )
        if mode not in ("TE", "TM", "both"):
            raise GeoBrainError(
                "MT2D mode must be 'TE', 'TM', or 'both'",
                object_name="MT2D", field="mode",
                expected="'TE' | 'TM' | 'both'", actual=mode,
            )
        self.survey = survey
        self.mode = mode
        # output_keys reflects the actual emitted MT observables (impedance,
        # apparent_resistivity, phase: all are MT inversion data), suffixed per
        # mode for "both". The raw solved E/H field goes to ForwardOutput.fields.
        _obs = ("impedance", "apparent_resistivity", "phase")
        _keys = (tuple(f"{o}_{m}" for o in _obs for m in ("TE", "TM"))
                 if mode == "both" else _obs)
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.IMPLICIT_VJP,
            trainable_inputs=("sigma",),
            output_keys=_keys,
        )

    def _modes(self) -> tuple[str, ...]:
        return ("TE", "TM") if self.mode == "both" else (self.mode,)

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (sigma,) = state.fetch("sigma")
        mesh = ctx.require_mesh()
        if mesh.n_dim != 2:
            raise GeoBrainError(
                "MT2D requires a 2D mesh",
                object_name="MT2D", field="mesh",
                expected="2D", actual=mesh.n_dim,
            )
        # Capability dispatch (declaration-based, mirroring DC25D): meshes that
        # declare StructuredMesh keep the original dense path bit-identically;
        # every other ConnectivityMesh takes the sparse FV path.
        # ``Mesh.declares`` reads the INSTANCE attribute first: capabilities
        # may be refined per instance, so a class-level read would miss them.
        is_structured = mesh.declares(StructuredMesh)
        if not is_structured:
            return self._forward_connectivity(sigma, mesh)
        # Within the structured branch, split on uniformity (see below): the
        # UNIFORM dense stencil reads scalar ``mesh.spacing`` (1/(dz·dz),
        # 1/(dy·dy)). A graded TensorMesh does NOT silently degrade to the
        # per-axis mean and is NOT rejected: it takes the in-branch sparse
        # graded stencil (per-axis face transmissibilities), CPU-only.
        # Uniform → dense complex64 path (bit-identical, reference-validated).
        # Graded (per-axis cell_widths) → sparse complex128 path: the same
        # cell-centred stencil with non-uniform face transmissibilities, solved
        # through the implicit-VJP sparse adjoint. A graded mesh lets one grid be
        # fine at the surface AND deep: the prerequisite for a BROADBAND MT
        # inversion that no single uniform mesh can serve. The sparse path is
        # CPU-only (scipy splu), matching the unstructured FV path.
        uniform = mesh.is_uniform
        if not uniform and sigma.device.type != "cpu" and default_sparse_solver_is_cpu_only():
            raise GeoBrainError(
                "MT2D's graded (non-uniform TensorMesh) path uses the sparse adjoint "
                "solve, whose default backend is CPU-only (scipy sparse LU). Move sigma "
                "to cpu, use a uniform mesh (dense GPU path), or activate a GPU backend "
                "via sparse_solver(KrylovSolver(...))",
                object_name="MT2D", field="sigma.device",
                expected="cpu (or a GPU sparse backend)", actual=sigma.device,
            )
        if self.survey.rcv_y is None:
            raise GeoBrainError(
                "MT2D on a structured (TensorMesh) grid needs surface "
                "column indices: set MT2DSurvey.rcv_y; rcv_y_m physical "
                "coordinates only address the unstructured FV path",
                object_name="MT2D", field="survey.rcv_y",
                expected="1D long Tensor (column indices)", actual=None,
            )
        require_field_matches_mesh(mesh, sigma, name="sigma", owner="MT2D")

        nz, ny = mesh.shape
        n = nz * ny
        device = sigma.device
        rcv_y = self.survey.rcv_y.to(device).to(torch.long)

        if uniform:
            dz, dy = mesh.spacing
            dz_surf = dz                          # surface centre-distance
            cdtype = torch.complex64
        else:
            wz, _wy = mesh.cell_widths
            dz_surf = float((0.5 * (wz[0] + wz[1])).item())  # row-0→row-1 centres
            cdtype = torch.complex128

        omegas = (2.0 * math.pi * self.survey.frequencies.to(device))

        # Per-mode buffers.
        modes = self._modes()
        Z_per_mode: dict[str, list[torch.Tensor]] = {m: [] for m in modes}
        field_per_mode: dict[str, list[torch.Tensor]] = {m: [] for m in modes}

        # RHS shared between modes: top row = 1 (incident plane wave).
        def _rhs() -> torch.Tensor:
            b = torch.zeros(n, dtype=cdtype, device=device)
            top_idx = torch.arange(ny, device=device, dtype=torch.long)
            return b.scatter_add(0, top_idx, torch.ones(ny, dtype=cdtype, device=device))

        # Assemble every (frequency, mode) system, then solve. On the graded
        # (sparse) path the independent per-frequency systems are stacked into ONE
        # block-diagonal solve: a single large matvec per Krylov iteration instead
        # of one small solve per frequency (a large GPU win via dispatch
        # amortization). On the CPU scipy-splu path a block-diagonal factorization
        # is byte-identical to the per-block solves, so outputs are unchanged. The
        # uniform (dense) path keeps the per-frequency loop.
        solved: list[tuple[float, str, torch.Tensor]] = []
        if uniform:
            for omega in omegas.tolist():
                b = _rhs()
                for m in modes:
                    A = assemble_mt2d_system(sigma, -float(omega), dz, dy, mode=m)
                    solved.append((omega, m, linear_solve_with_adjoint(A, b)))
        else:
            A_list: list[torch.Tensor] = []
            b_list: list[torch.Tensor] = []
            meta: list[tuple[float, str]] = []
            for omega in omegas.tolist():
                b = _rhs()
                for m in modes:
                    A_list.append(
                        assemble_mt2d_system_graded(sigma, -float(omega), mesh, mode=m))
                    b_list.append(b)
                    meta.append((omega, m))
            U_stacked = sparse_linear_solve_with_adjoint(
                block_diag(A_list), torch.cat(b_list))
            for idx, (omega, m) in enumerate(meta):
                solved.append((omega, m, U_stacked[idx * n:(idx + 1) * n]))

        for omega, m, U_flat in solved:
            U_2d = U_flat.view(nz, ny)

            # One-sided forward z-derivative at the surface (centre-distance
            # ``dz_surf`` = dz on a uniform grid, (wz0+wz1)/2 when graded).
            U_top = U_2d[0, :]
            U_next = U_2d[1, :]

            if m == "TE":
                # E_x = U; H_y = -∂E_x/∂z · (i·ω·μ₀)^{-1}.
                dU_dz = (U_next - U_top) / dz_surf
                H_y_top = -dU_dz / (1j * omega * MU0)
                Z_y = U_top / H_y_top
            else:  # m == "TM"
                # H_x = U; E_y = (1/σ_top) · ∂H_x/∂z. Surface σ at the top row.
                sigma_top = sigma[0, :].to(cdtype)
                dU_dz = (U_next - U_top) / dz_surf
                E_y_top = dU_dz / sigma_top
                # Conventional Z_TM = -E_y / H_x; sign carries the impedance into
                # the same +135°/-45° Cagniard family as TE so phase /
                # apparent-resistivity outputs match.
                Z_y = -E_y_top / U_top

            Z_per_mode[m].append(Z_y[rcv_y])
            field_per_mode[m].append(U_2d)

        omega_col = omegas.unsqueeze(-1)

        if self.mode == "both":
            data: dict[str, torch.Tensor] = {}
            fields: dict[str, torch.Tensor] = {}
            for m in modes:
                Z = torch.stack(Z_per_mode[m], dim=0)
                F = torch.stack(field_per_mode[m], dim=0)
                rho_a = Z.abs().pow(2) / (omega_col.to(Z.real.dtype) * MU0)
                phase = torch.atan2(Z.imag, Z.real)
                data[f"impedance_{m}"] = Z
                data[f"apparent_resistivity_{m}"] = rho_a
                data[f"phase_{m}"] = phase
                # TE solves for E_x; TM solves for H_x: the raw solved field is a
                # diagnostic, not an observation → ForwardOutput.fields.
                field_key = "E_field_TE" if m == "TE" else "H_field_TM"
                fields[field_key] = F
            return ForwardOutput(
                data=data,
                fields=fields,
                metadata={
                    "n_freq": self.survey.n_freq,
                    "n_rcv": self.survey.n_rcv,
                    "frequencies": self.survey.frequencies.tolist(),
                    "modes": list(modes),
                },
            )

        # Single-mode path: keep the legacy unprefixed key layout.
        m = modes[0]
        Z = torch.stack(Z_per_mode[m], dim=0)
        F = torch.stack(field_per_mode[m], dim=0)
        rho_a = Z.abs().pow(2) / (omega_col.to(Z.real.dtype) * MU0)
        phase = torch.atan2(Z.imag, Z.real)
        return ForwardOutput(
            data={
                "impedance": Z,
                "apparent_resistivity": rho_a,
                "phase": phase,
            },
            fields={"E_field": F},  # raw solved field, diagnostic, not an observation
            metadata={
                "n_freq": self.survey.n_freq,
                "n_rcv": self.survey.n_rcv,
                "frequencies": self.survey.frequencies.tolist(),
                "modes": [m],
            },
        )

    def _forward_connectivity(self, sigma: torch.Tensor, mesh) -> ForwardOutput:
        """Sparse FV path for non-structured 2-D ``ConnectivityMesh``es.

        Solves the SAME boundary-value problems as the dense path (top
        plane-wave Dirichlet ``u = 1``, bottom absorbing ``u = 0``, no-flux
        sides) through the TPFA seam of
        :mod:`~geobrain.physics.em.numerics.finite_volume.helmholtz_fv`, in
        float64 / complex128 on CPU (scipy splu adjoint bridge).

        Surface impedance per station: the same physical definition as the
        structured extraction, written on the FV boundary closure: the
        structured path reads the prescribed surface value (the pinned top
        row, ``u = 1``) and a one-sided forward ``∂u/∂z`` between the first
        two cell rows; here the prescribed value sits ON the top boundary
        face (``z = 0``), so the one-sided difference runs from the face to
        its owning cell point over the half-cell distance:
        ``∂u/∂z ≈ (u_cell − 1) / dist``. Both are first-order one-sided
        estimates of the surface derivative whose leading error scales with
        the near-surface cell size; they agree as the meshes refine (bounded
        end-to-end by the structured-vs-unstructured operator test). Each
        station (physical ``survey.rcv_y_m``, metres) binds to the nearest
        top boundary face, the nearest-cell convention of DC25D. For TM the
        Ampère factor ``1/σ_top`` uses that face's owning-cell σ (the
        structured path's top-row σ at the receiver column).
        """
        if sigma.device.type != "cpu" and default_sparse_solver_is_cpu_only():
            raise GeoBrainError(
                "MT2D's FV path uses the sparse adjoint solve, whose default backend "
                "is CPU-only (scipy sparse LU); move sigma to cpu, or activate a GPU "
                "backend via sparse_solver(KrylovSolver(...))",
                object_name="MT2D", field="sigma.device",
                expected="cpu (or a GPU sparse backend)", actual=sigma.device,
            )
        if tuple(sigma.shape) != (mesh.n_cells,):
            raise GeoBrainError(
                "sigma must be flat (n_cells,) on a non-tensor (triangle) mesh",
                object_name="MT2D", field="sigma",
                expected=(mesh.n_cells,), actual=tuple(sigma.shape),
            )
        if self.survey.rcv_y_m is None:
            raise GeoBrainError(
                "MT2D on an unstructured mesh needs physical station "
                "coordinates: set MT2DSurvey.rcv_y_m (metres); rcv_y column "
                "indices only address a structured grid",
                object_name="MT2D", field="survey.rcv_y_m",
                expected="1D float Tensor (metres)", actual=None,
            )

        device = sigma.device
        sigma_f64 = sigma if sigma.dtype == torch.float64 else sigma.to(torch.float64)

        face_records = mesh.face_neighbors()
        cell_volumes = mesh.cell_volumes().to(torch.float64)
        centers = mesh.cell_centers().to(torch.float64)
        partition = partition_helmholtz_boundary(mesh.boundary_faces(), centers)

        # Station binding: nearest top boundary face per physical y (metres).
        rcv_y_m = self.survey.rcv_y_m.to(device=device, dtype=torch.float64)
        face_idx = (partition.top_y.unsqueeze(0) - rcv_y_m.unsqueeze(1)).abs().argmin(dim=1)
        rcv_cell = partition.top_cell[face_idx]               # (n_rcv,)
        rcv_dist = partition.top_dist[face_idx].to(torch.complex128)
        n_rcv = int(rcv_y_m.numel())

        omegas = (2.0 * math.pi * self.survey.frequencies.to(device))

        modes = self._modes()
        Z_per_mode: dict[str, list[torch.Tensor]] = {m: [] for m in modes}
        field_per_mode: dict[str, list[torch.Tensor]] = {m: [] for m in modes}

        for omega in omegas.tolist():
            for m in modes:
                assemble = (
                    assemble_helmholtz_fv if m == "TE" else assemble_helmholtz_fv_tm
                )
                A, b = assemble(
                    face_records, partition, cell_volumes, sigma_f64,
                    -float(omega), mu0=MU0,
                )
                # Complex COO fed AS-IS: ``to_sparse_csr()`` has no autograd
                # support for complex dtype (same note as the SIP pass-B solve).
                U = sparse_linear_solve_with_adjoint(A, b)

                # One-sided forward z-derivative at the surface: prescribed
                # face value u_D = 1 → owning cell point (half-cell distance).
                U_cell = U[rcv_cell]
                dU_dz = (U_cell - 1.0) / rcv_dist

                if m == "TE":
                    # E_x(surface) = u_D = 1; H_y = -∂E_x/∂z · (i·ω·μ₀)^{-1}.
                    H_y_top = -dU_dz / (1j * omega * MU0)
                    Z_y = 1.0 / H_y_top
                else:  # m == "TM"
                    # H_x(surface) = u_D = 1; E_y = (1/σ_top) · ∂H_x/∂z.
                    sigma_top = sigma_f64[rcv_cell].to(torch.complex128)
                    E_y_top = dU_dz / sigma_top
                    # Same sign convention as the structured path: carry the
                    # impedance into the TE-matching Cagniard family.
                    Z_y = -E_y_top

                Z_per_mode[m].append(Z_y)
                field_per_mode[m].append(U)

        omega_col = omegas.unsqueeze(-1)

        data: dict[str, torch.Tensor] = {}
        fields: dict[str, torch.Tensor] = {}
        for m in modes:
            Z = torch.stack(Z_per_mode[m], dim=0)            # (n_freq, n_rcv)
            F = torch.stack(field_per_mode[m], dim=0)        # (n_freq, n_cells)
            rho_a = Z.abs().pow(2) / (omega_col.to(Z.real.dtype) * MU0)
            phase = torch.atan2(Z.imag, Z.real)
            if self.mode == "both":
                data[f"impedance_{m}"] = Z
                data[f"apparent_resistivity_{m}"] = rho_a
                data[f"phase_{m}"] = phase
                fields["E_field_TE" if m == "TE" else "H_field_TM"] = F
            else:
                data["impedance"] = Z
                data["apparent_resistivity"] = rho_a
                data["phase"] = phase
                fields["E_field"] = F
        return ForwardOutput(
            data=data,
            fields=fields,  # raw solved field, flat per cell, diagnostic
            metadata={
                "n_freq": self.survey.n_freq,
                "n_rcv": n_rcv,
                "frequencies": self.survey.frequencies.tolist(),
                "modes": list(modes),
            },
        )


# ----------------------------------------------------------------------
# Closed-form σ-Jacobian
# ----------------------------------------------------------------------


def compute_jacobian(
    operator: "MT2D",
    sigma: torch.Tensor,
    mesh: TensorMesh,
) -> dict[str, torch.Tensor]:
    """
    Return the per-mode ``∂(log|Z|)/∂σ`` Jacobian via autograd.

    Convenience wrapper around ``torch.autograd.functional.jacobian``
    that exercises the operator's IMPLICIT_VJP adjoint path; there is
    no hand-derived closed form here. For dense meshes (≲ 40 × 40) this
    is exact to machine precision; for larger meshes the per-row
    autograd loop dominates runtime and you should switch to a
    matrix-free Lanczos / CG approach. The :class:`~geobrain.optim.Inverter`
    does not use this helper; it is a standalone diagnostic.

    Args:
        operator: an :class:`MT2D` instance. ``operator.mode`` selects
            which modes are evaluated.
        sigma:    cell-centred σ, shape ``(nz, ny)``.
        mesh:     the 2-D :class:`TensorMesh` the operator runs on.

    Returns:
        Dict ``{mode: J}`` where ``J`` is a real ``(n_freq · n_rcv,
        nz · ny)`` tensor. Each row is ``∂(log|Z_{freq, rcv}|) / ∂σ_c``.
    """
    if not isinstance(operator, MT2D):
        raise GeoBrainError(
            "compute_jacobian requires an MT2D operator",
            object_name="compute_jacobian",
            field="operator",
            expected="MT2D",
            actual=type(operator).__name__,
        )
    ctx = ForwardContext.of(mesh=mesh)
    out: dict[str, torch.Tensor] = {}
    for m in operator._modes():
        key = "impedance" if operator.mode != "both" else f"impedance_{m}"

        def _log_abs_Z(sig: torch.Tensor) -> torch.Tensor:
            pred = operator(ModelState(tensors={"sigma": sig}), ctx)
            Z = pred.data[key]
            return torch.log(Z.abs()).reshape(-1)

        sigma_g = sigma.detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(_log_abs_Z, sigma_g)
        # ``J`` has shape (n_freq * n_rcv, nz, ny): flatten the σ axis.
        out[m] = J.reshape(J.shape[0], -1).detach()
    return out
