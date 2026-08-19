# pyright: reportPrivateImportUsage=false
"""2.5-D DC resistivity (point electrodes over a 2-D conductivity structure).

The 2.5-D ("two-and-a-half-D") problem is the field-standard ERT forward model:
the conductivity is invariant along the strike direction y, ``σ = σ(z, x)``, but
the current electrodes are true **point** sources (not the infinite line sources
of the plain-2-D :class:`~geobrain.physics.em.static.dc2d.DC2D`, which solves a
genuinely different observable, the potential of a line of current per unit
length). Because the medium is y-invariant but the source is a point, the 3-D
Poisson problem separates under a Fourier-cosine transform in y: for each
spatial wavenumber ``k`` the transformed potential ``ũ_k(z, x)`` satisfies the
2-D modified-Helmholtz equation

    ∇·(σ ∇ũ_k) − k² σ ũ_k = −I δ(x − x_s)                    in the (z, x) plane

(free surface ``∂ũ_k/∂n = 0`` at z = 0), and the physical surface potential is
recovered by the inverse cosine transform evaluated at y = 0:

    U(z, x; y=0) = (2/π) ∫₀^∞ ũ_k dk  ≈  Σ_i w_i · ũ_{k_i}.

This is the standard 2.5-D ERT formulation, recast here in the GeoBrain
cell-centred finite-volume seam. The discretisation is exquisitely clean: the
mass term ``−k² σ`` is a *cell-diagonal* addition to the TPFA stiffness matrix,

    A_k = A_TPFA(σ)  +  diag(k² · σ_cell · V_cell),

so each wavenumber's system is assembled by the SAME mesh-agnostic
:func:`~geobrain.physics.em.numerics.finite_volume.poisson_fv.assemble_poisson_fv`
seam used by DC3D/IP/SP, with the mass diagonal passed through the new
``extra_diagonal`` keyword. Because ``k > 0`` the mass term makes ``A_k``
positive-definite (NON-singular) on its own, so NO Dirichlet pin is needed and
NO ``k = 0`` sample is taken (``k = 0`` would recover the singular pure-Neumann
2-D operator and the divergent line-source observable). The mass diagonal
carries σ's autograd graph, so the conductivity gradient flows through BOTH the
stiffness and the mass term; each k-solve is a standard implicit-VJP adjoint
(``IMPLICIT_VJP``), and the receiver voltage is the weighted sum
``Σ_i w_i ũ_{k_i}`` over the wavenumber quadrature.

Because the per-k matrices differ (the mass diagonal scales with ``k²``), the
loop is k-OUTER: factor ``A_{k_i}`` once per wavenumber, solve the ±I source/sink
RHS, accumulate ``w_i ũ_{k_i}`` at the receiver cells. Like DC3D this runs on the
scipy sparse-LU bridge (``sparse_linear_solve_with_adjoint``), so **DC25D is
CPU / float64 by design**; non-CPU inputs are rejected up front.

Wavenumber quadrature (calibrated against the homogeneous-half-space
analytic solution; see the module-level constant docstrings and the
DC2.5D test suite)
-------------------------------------------------------------------------------
Following the standard split-interval scheme, the integral ``∫₀^∞ ũ_k dk`` is
split at a knot ``k0 = 1/(2·r_min)`` (``r_min`` = the smallest electrode-to-
receiver / inter-electrode separation of the survey):

* ``(0, k0]``: **Gauss–Legendre** with ``n_leg = max(int(6·log10(r_max/r_min)),
  4)`` points, mapped by the substitution ``k = k0·t²`` (``t ∈ (0, 1]`` the
  Legendre nodes shifted to ``[0, 1]``). The substitution concentrates nodes
  near ``k → 0`` where ``ũ_k`` is largest. With the synthesis prefactor
  ``1/π`` folded in, the transformed weight is ``w = (1/π)·2·k0·t·w_leg``
  (the ``2·k0·t`` is ``dk/dt``).
* ``[k0, ∞)``: **Gauss–Laguerre** with ``n_lag`` points (default 4): the
  Laguerre rule integrates ``∫₀^∞ e^{-x} f(x) dx``; with ``k = k0·(1 + x)`` so
  ``dk = k0 dx`` and the rule's implicit ``e^{-x}`` undone, the transformed
  weight is ``w = (1/π)·k0·e^{x}·w_lag``.

The synthesis prefactor is ``1/π``, not the continuous transform's ``2/π``:
the cell-centred FV operator solved on the ``z ≥ 0`` domain with a NEUMANN free
surface at ``z = 0`` already embeds the half-space image, so its transformed
surface kernel is ``(I/πσ)·K₀(kr) = 2 ×`` the full-space ``(I/2πσ)·K₀``, the
discrete doubling replaces the transform's. The same ``2/π → 1/π``
halving is standard practice in cell-centred 2.5-D implementations. The prefactor and the ``dk`` Jacobians are VERIFIED by
the homogeneous test, which a half-space must satisfy to ``<1 %`` on a fine
structured mesh (the test is the calibrator of record; see
:data:`_QUADRATURE_NOTES`).

Governing / observable distinction vs DC2D
------------------------------------------
:class:`DC2D` solves ``−∇·(σ∇φ) = q`` with a LINE source in the 2-D plane, the
potential of an infinite line of current, a different physical observable.
:class:`DC25D` here solves the POINT-source potential over the same 2-D σ via the
wavenumber synthesis above and reproduces the true 3-D point-electrode field
``φ = ρI/(2π r)`` over a half-space. Use DC25D for ERT field data (point
electrodes); DC2D models the line-source idealisation only.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

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
from ....mesh.capabilities import ConnectivityMesh, StructuredMesh
from ....mesh import canonicalize_cell_field, require_capable_mesh


# Human-readable record of the calibrated quadrature, kept next to the code so
# the scheme that the analytic test pins down is documented in one place.
_QUADRATURE_NOTES = """\
Fourier-cosine wavenumber quadrature for the 2.5-D synthesis U = C·∫₀^∞ ũ_k dk.

prefactor C = 1/π  (NOT the continuous transform's 2/π, the FV Neumann free
          surface at z=0 already embeds the half-space image, doubling ũ_k; standard
          2/π→1/π halving).
knot      k0 = 1 / (2·r_min)
legendre  n_leg = max(int(6·log10(r_max/r_min)), 4) nodes on (0, k0] via k = k0·t²
          (t = Gauss-Legendre nodes mapped to [0,1]); weight (1/π)·(2·k0·t)·w_leg.
laguerre  n_lag (default 4) nodes on [k0, ∞) via k = k0·(1+x); weight
          (1/π)·k0·e^{x}·w_lag (e^{x} undoes the Laguerre kernel e^{-x}).

r_min / r_max are the min / max electrode↔receiver and inter-electrode
separations of the survey. Calibrated so a homogeneous half-space reproduces
φ = ρI/(2π r) at the receivers to <1% on a fine structured mesh (measured:
~0.4% max on 60×120 @ 1 m).
"""


@dataclass(frozen=True)
class DC25DSurvey:
    """
    Pole-dipole / dipole-dipole DC geometry for the 2.5-D operator, in
    PHYSICAL coordinates (metres) in the 2-D mesh frame.

    The mesh is the strike-perpendicular ``(z, x)`` plane: ``z`` is depth
    (free surface at ``z = 0``), ``x`` is the profile direction. Electrodes are
    true POINT sources at ``y = 0`` (the y-invariance of σ is what makes the
    problem 2.5-D). Coordinates are physical metres measured from the mesh's
    top-left corner (the implicit origin of ``cell_centers``); for a uniform
    mesh cell ``(iz, ix)`` has centre ``((iz + 0.5)·dz, (ix + 0.5)·dx)``.

    A single current dipole (source ``+I`` at A, sink ``−I`` at B) drives a list
    of receiver positions, mirroring :class:`DC3DSurvey`'s shape (an A–B pair
    plus a receiver list), dropped to the 2-D ``(z, x)`` frame.

    Attributes:
        source_z, source_x: physical coordinate (m) of the current-injection
            electrode A (positive current ``+I``).
        sink_z, sink_x:     physical coordinate (m) of the sink B (``−I``).
        rcv_z, rcv_x:       1-D float tensors of receiver coordinates (m).
        current:            injected current magnitude (A).
    """

    source_z: float
    source_x: float
    sink_z: float
    sink_x: float
    rcv_z: torch.Tensor
    rcv_x: torch.Tensor
    current: float = 1.0

    def __post_init__(self) -> None:
        for name in ("rcv_z", "rcv_x"):
            t = getattr(self, name)
            if not isinstance(t, torch.Tensor):
                raise GeoBrainError(
                    f"DC25DSurvey.{name} must be a torch.Tensor",
                    object_name="DC25DSurvey", field=name,
                    expected=torch.Tensor, actual=type(t),
                )
            if t.ndim != 1:
                raise GeoBrainError(
                    f"DC25DSurvey.{name} must be 1D",
                    object_name="DC25DSurvey", field=name,
                    expected="1D Tensor", actual=tuple(t.shape),
                )
        if self.rcv_z.numel() != self.rcv_x.numel():
            raise GeoBrainError(
                "rcv_z and rcv_x must share length",
                object_name="DC25DSurvey", field="rcv_x",
                expected=self.rcv_z.numel(), actual=self.rcv_x.numel(),
            )
        if self.current <= 0:
            raise GeoBrainError(
                "current must be positive",
                object_name="DC25DSurvey", field="current",
                expected="> 0", actual=self.current,
            )

    @property
    def n_rcv(self) -> int:
        return int(self.rcv_z.numel())

    def _electrode_xz(self) -> torch.Tensor:
        """All electrode ``(z, x)`` positions stacked, shape ``(2 + n_rcv, 2)``."""
        zs = torch.cat([
            torch.tensor([self.source_z, self.sink_z], dtype=torch.float64),
            self.rcv_z.to(torch.float64),
        ])
        xs = torch.cat([
            torch.tensor([self.source_x, self.sink_x], dtype=torch.float64),
            self.rcv_x.to(torch.float64),
        ])
        return torch.stack([zs, xs], dim=-1)

    def separation_bounds(self) -> tuple[float, float]:
        """``(r_min, r_max)`` over all electrode↔receiver and A↔B separations (m).

        These set the wavenumber-quadrature knot ``k0 = 1/(2·r_min)`` and the
        Legendre node count ``n_leg = max(int(6·log10(r_max/r_min)), 4)``. The
        pairwise set is the source/sink against every receiver plus the A–B
        dipole length, the physical scales the synthesis must resolve. Zero /
        coincident pairs (a receiver atop an electrode) are ignored when taking
        the min so they don't collapse ``r_min`` to 0.
        """
        pos = self._electrode_xz()
        d = torch.cdist(pos, pos)                     # (m, m) pairwise distances
        iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
        seps = d[iu[0], iu[1]]
        positive = seps[seps > 0]
        if positive.numel() == 0:
            raise GeoBrainError(
                "DC25DSurvey: all electrodes coincide; cannot set wavenumber scale",
                object_name="DC25DSurvey", field="separation_bounds",
                expected="at least one positive electrode separation",
                actual="all coincident",
            )
        return float(positive.min()), float(seps.max())


def wavenumber_quadrature(
    r_min: float,
    r_max: float,
    *,
    n_gauss_legendre: int | None = None,
    n_gauss_laguerre: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourier-cosine wavenumber nodes ``k_i`` and synthesis weights ``w_i``.

    Implements a split-interval quadrature scheme calibrated by
    the homogeneous half-space analytic test (see the module docstring and
    :data:`_QUADRATURE_NOTES`). Returns ``(k, w)`` such that

        U(receiver) ≈ Σ_i w_i · ũ_{k_i}

    reproduces ``φ = ρ I / (2π r)`` over a half-space (the ``1/π`` synthesis
    prefactor, *not* the continuous transform's ``2/π``; the FV Neumann
    free surface already doubles the kernel, see the comment on ``k0``
    below, and the per-interval ``dk`` Jacobians are folded into ``w``).

    Args:
        r_min: smallest electrode↔receiver / inter-electrode separation (m).
        r_max: largest such separation (m).
        n_gauss_legendre: Legendre node count on ``(0, k0]``. ``None`` (default)
            → ``max(int(6·log10(r_max/r_min)), 4)`` (the calibrated node-count formula).
        n_gauss_laguerre: Laguerre node count on ``[k0, ∞)`` (default 4).

    Returns:
        ``(k, w)``: two 1-D float64 tensors of length
        ``n_gauss_legendre + n_gauss_laguerre``. All ``k_i > 0``.
    """
    import numpy as np

    if not (r_min > 0.0 and r_max >= r_min):
        raise GeoBrainError(
            "wavenumber_quadrature needs 0 < r_min <= r_max",
            object_name="wavenumber_quadrature", field="r_min/r_max",
            expected="0 < r_min <= r_max", actual=(r_min, r_max),
        )
    if n_gauss_legendre is None:
        n_leg = max(int(6.0 * np.log10(max(r_max / r_min, 1.0 + 1e-12))), 4)
    else:
        n_leg = int(n_gauss_legendre)
    if n_leg < 1:
        raise GeoBrainError(
            "n_gauss_legendre must be >= 1",
            object_name="wavenumber_quadrature", field="n_gauss_legendre",
            expected=">= 1", actual=n_leg,
        )
    n_lag = int(n_gauss_laguerre)
    if n_lag < 1:
        raise GeoBrainError(
            "n_gauss_laguerre must be >= 1",
            object_name="wavenumber_quadrature", field="n_gauss_laguerre",
            expected=">= 1", actual=n_lag,
        )

    k0 = 1.0 / (2.0 * r_min)
    # Synthesis prefactor. The continuous inverse cosine transform carries
    # ``2/π``, BUT the cell-centred FV operator solved on the z ≥ 0 domain with a
    # NEUMANN free surface at z = 0 already embeds the half-space image: its
    # transformed kernel for a surface source is ``(I/πσ)·K₀(kr)`` =
    # 2 × the full-space ``(I/2πσ)·K₀``. So the discrete doubling replaces the
    # transform's, and the prefactor that lands U on ``ρI/(2π r)`` is ``1/π``
    # (verified to <1% by the homogeneous half-space test; ∫₀^∞ K₀(kr)dk =
    # π/(2r) ⇒ (1/π)·(I/πσ)·(π/2r) = I/(2πσr) = ρI/(2π r)). The same
    # ``2/π → 1/π`` halving is standard in cell-centred 2.5-D codes.
    one_over_pi = 1.0 / np.pi

    # --- Gauss-Legendre on (0, k0] via k = k0·t², t ∈ (0, 1] -------------
    # leggauss returns nodes/weights on [-1, 1]; shift to [0, 1] (t = (x+1)/2,
    # w_t = w/2). Then k = k0·t² maps [0,1] → [0, k0] with dk = 2·k0·t·dt, so
    #   ∫₀^{k0} f(k) dk = ∫₀¹ f(k0·t²)·2·k0·t dt = Σ f(k_j)·(2·k0·t_j)·w_{t,j}.
    # The synthesis prefactor 1/π multiplies every weight.
    x_leg, w_leg = np.polynomial.legendre.leggauss(n_leg)
    t = 0.5 * (x_leg + 1.0)
    w_t = 0.5 * w_leg
    k_leg = k0 * t * t
    wk_leg = one_over_pi * (2.0 * k0 * t) * w_t

    # --- Gauss-Laguerre on [k0, ∞) via k = k0·(1 + x), x ∈ [0, ∞) -------
    # laggauss integrates ∫₀^∞ e^{-x} g(x) dx = Σ g(x_j)·w_{lag,j}. We want
    #   ∫_{k0}^∞ f(k) dk = ∫₀^∞ f(k0·(1+x))·k0 dx
    #                    = Σ f(k0·(1+x_j))·k0·e^{x_j}·w_{lag,j}
    # (the e^{x} undoes the rule's built-in e^{-x}). Prefactor 1/π again.
    x_lag, w_lag = np.polynomial.laguerre.laggauss(n_lag)
    k_lag = k0 * (1.0 + x_lag)
    wk_lag = one_over_pi * k0 * np.exp(x_lag) * w_lag

    k = np.concatenate([k_leg, k_lag])
    w = np.concatenate([wk_leg, wk_lag])
    return (
        torch.from_numpy(k).to(torch.float64),
        torch.from_numpy(w).to(torch.float64),
    )


class DC25D(EMOperatorDiscovery, ForwardOperator):
    """
    2.5-D DC resistivity forward operator (point electrodes, y-invariant σ).

    Solves the point-electrode potential over a 2-D conductivity structure
    ``σ = σ(z, x)`` by Fourier-cosine wavenumber synthesis (see the module
    docstring). Runs on BOTH a structured 2-D ``TensorMesh`` and an unstructured
    2-D triangle :class:`UnstructuredMesh`, through the mesh-agnostic
    cell-centred FV seam; the per-wavenumber stiffness is the standard TPFA
    Poisson matrix and the ``−k² σ`` mass term is a cell-diagonal addition
    (``extra_diagonal = k²·σ·V``).

    Device: CPU / float64 only (scipy sparse LU); move inputs to CPU first.

    Inputs (ModelState):
        ``sigma``: ``(nz, nx)`` on a structured ``TensorMesh`` (dual-shape like
        DC3D), or flat ``(n_cells,)`` on a 2-D ``UnstructuredMesh``.

    Construction:
        ``n_gauss_legendre`` (default ``None`` → ``max(int(6·log10(r_max/r_min)),
        4)``) and ``n_gauss_laguerre`` (default 4) set the wavenumber quadrature;
        the nodes/weights are derived from the survey's separation bounds.

    Outputs (ForwardOutput):
        ``data["voltage"]``:      receiver voltages ``Σ_i w_i ũ_{k_i}``, shape
            ``(n_rcv,)`` (registered ``voltage`` channel).
        ``fields["potential"]``: synthesised surface ``φ`` field, shape
            ``(nz, nx)`` on a structured mesh or ``(n_cells,)`` unstructured
            (diagnostic, the y=0 inverse-transform of the wavenumber stack).
        ``metadata["wavenumbers"]`` / ``["weights"]``, the calibrated
            quadrature (1-D float64 tensors).
        ``metadata["n_wavenumbers"]`` / ``["r_min"]`` / ``["r_max"]``, scheme
            diagnostics.

    Differentiability: ``IMPLICIT_VJP``. Each k-solve is the standard sparse
    implicit adjoint; the σ-gradient flows through both the TPFA stiffness and
    the ``k²·σ·V`` mass diagonal, accumulated over the wavenumber sum.

    This is the 2.5-D point-source observable, DISTINCT from
    :class:`~geobrain.physics.em.static.dc2d.DC2D`, which solves the genuinely
    different infinite-line-source 2-D potential.

    Args:
        survey: 2.5-D galvanic acquisition.
        n_gauss_legendre / n_gauss_laguerre: wavenumber-quadrature sizes
            of the 2.5-D Fourier synthesis.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        output_keys=("voltage",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (ConnectivityMesh,)

    def __init__(
        self,
        survey: DC25DSurvey,
        *,
        n_gauss_legendre: int | None = None,
        n_gauss_laguerre: int = 4,
    ) -> None:
        super().__init__()
        if not isinstance(survey, DC25DSurvey):
            raise GeoBrainError(
                "DC25D requires a DC25DSurvey",
                object_name="DC25D", field="survey",
                expected=DC25DSurvey, actual=type(survey),
            )
        self.survey = survey
        self._n_gauss_legendre = n_gauss_legendre
        self._n_gauss_laguerre = int(n_gauss_laguerre)

    @staticmethod
    def _nearest_cell(centers: torch.Tensor, z: float, x: float) -> int:
        """Flat cell index nearest the PHYSICAL electrode ``(z, x)``.

        ``centers`` is ``mesh.cell_centers()`` (``(n_cells, 2)``, columns
        ``(z, x)`` metres). Electrode coordinates are physical metres, so the
        binding is an argmin of the squared distance over the cell centres,
        exact on both the structured grid and the triangle mesh (it compares
        against the actual cell geometry rather than reconstructing from any
        single spacing).
        """
        target = torch.tensor([float(z), float(x)], dtype=torch.float64)
        d2 = ((centers - target) ** 2).sum(dim=1)
        return int(torch.argmin(d2))

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (sigma,) = state.fetch("sigma")
        if sigma.device.type != "cpu":
            raise GeoBrainError(
                "DC25D is CPU-only (scipy sparse LU); move sigma to cpu first",
                object_name="DC25D", field="sigma.device",
                expected="cpu", actual=sigma.device,
            )
        # Physical conductivity must be finite and strictly positive. A sigma<=0
        # cell makes the per-wavenumber system indefinite: the solve still
        # returns finite, plausible-looking, but meaningless voltages. Reject the
        # non-physical field up front (vectorized: no per-element python).
        if not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0).any()):
            raise GeoBrainError(
                "DC25D requires a finite, strictly-positive conductivity field "
                "(sigma > 0 S/m); a non-physical sigma <= 0 (or non-finite) makes "
                "the wavenumber system indefinite and yields meaningless voltages",
                object_name="DC25D", field="sigma",
                expected="finite and all > 0",
                actual="contains sigma <= 0 or non-finite entries",
            )
        mesh = require_capable_mesh(ctx, ConnectivityMesh)
        if mesh.n_dim != 2:
            raise GeoBrainError(
                "DC25D requires a 2-D mesh (the strike-perpendicular z–x plane)",
                object_name="DC25D", field="mesh.n_dim",
                expected=2, actual=mesh.n_dim,
            )

        is_structured = mesh.declares(StructuredMesh)
        sigma = canonicalize_cell_field(mesh, sigma, name="sigma",
                                        owner="DC25D")

        from ..numerics.finite_volume.poisson_fv import assemble_poisson_fv

        device, dtype = sigma.device, sigma.dtype
        n = int(mesh.n_cells)
        # The scipy splu bridge requires float64; promote for assembly + solve,
        # cast the synthesised field back to the caller's dtype at the end.
        sigma_f64 = sigma if sigma.dtype == torch.float64 else sigma.to(torch.float64)
        sigma_flat = sigma_f64.reshape(-1)               # (n_cells,) flat view

        face_records = mesh.face_neighbors()
        cell_volumes = mesh.cell_volumes().to(torch.float64)     # (n_cells,)
        centers = mesh.cell_centers().to(torch.float64)          # (n_cells, 2)

        # Bind electrodes to nearest cell centres (physical metres → cell).
        src_flat = self._nearest_cell(
            centers, self.survey.source_z, self.survey.source_x
        )
        snk_flat = self._nearest_cell(
            centers, self.survey.sink_z, self.survey.sink_x
        )
        if src_flat == snk_flat:
            raise GeoBrainError(
                "DC25D source and sink electrodes map to the same cell, the "
                "±I current would cancel to a zero RHS (silent phi=0). Refine "
                "the mesh near the electrodes or separate the A–B dipole.",
                object_name="DC25D", field="survey",
                expected="source and sink in distinct cells",
                actual="same cell, mesh too coarse for this A–B separation",
            )

        # ±I point-source RHS (shared across all wavenumbers): the full current
        # I at the source/sink cells. The free surface at z = 0 is a Neumann
        # face, so the half-space image is automatic in the discrete operator;
        # which is why the synthesis prefactor is 1/π (not the continuous
        # transform's 2/π) and U → ρI/(2π r). See ``wavenumber_quadrature``.
        b = torch.zeros(n, dtype=torch.float64, device=device).scatter_add(
            0,
            torch.tensor([src_flat, snk_flat], device=device, dtype=torch.long),
            torch.tensor(
                [+self.survey.current, -self.survey.current],
                dtype=torch.float64, device=device,
            ),
        )

        # Wavenumber quadrature from the survey's physical separation bounds.
        r_min, r_max = self.survey.separation_bounds()
        k_nodes, k_weights = wavenumber_quadrature(
            r_min, r_max,
            n_gauss_legendre=self._n_gauss_legendre,
            n_gauss_laguerre=self._n_gauss_laguerre,
        )
        k_nodes = k_nodes.to(device)
        k_weights = k_weights.to(device)

        # k-OUTER synthesis loop. Each A_k = A_TPFA(σ) + diag(k²·σ·V) is a
        # DIFFERENT matrix (mass diagonal scales with k²), so we factor per k.
        # k > 0 makes A_k positive-definite → NO pin, NO k=0 sample. The mass
        # diagonal carries σ's autograd graph, so the σ-gradient flows through
        # both the stiffness and the mass term; each solve is the standard
        # sparse implicit adjoint and the weighted sum stays on the graph.
        phi = torch.zeros(n, dtype=torch.float64, device=device)
        for k_i, w_i in zip(k_nodes, k_weights):
            k2 = float(k_i) * float(k_i)
            mass_diag = (k2 * sigma_flat) * cell_volumes     # (n_cells,), σ live
            A_k = assemble_poisson_fv(
                face_records, sigma_flat, extra_diagonal=mass_diag,
            ).to_sparse_csr()
            u_k = sparse_linear_solve_with_adjoint(A_k, b)
            phi = phi + w_i * u_k

        if phi.dtype != dtype:
            phi = phi.to(dtype)

        # Receiver voltages from the synthesised field (nearest-cell sampling).
        rcv_flat = torch.tensor(
            [
                self._nearest_cell(centers, z, x)
                for z, x in zip(
                    self.survey.rcv_z.tolist(), self.survey.rcv_x.tolist()
                )
            ],
            dtype=torch.long, device=device,
        )
        voltage = phi[rcv_flat]

        # Diagnostic potential field: (nz, nx) on a structured mesh, flat on a
        # triangle mesh (mirrors DC3D's dual-shape ``fields["potential"]``).
        if is_structured:
            phi_field = phi.view(*mesh.shape)
        else:
            phi_field = phi

        return ForwardOutput(
            data={"voltage": voltage},
            fields={"potential": phi_field},  # y=0 inverse transform, diagnostic
            metadata={
                "units": {"voltage": "V"},
                "current": self.survey.current,
                "n_rcv": self.survey.n_rcv,
                "wavenumbers": k_nodes.detach().cpu(),
                "weights": k_weights.detach().cpu(),
                "n_wavenumbers": int(k_nodes.numel()),
                "r_min": r_min,
                "r_max": r_max,
            },
        )
