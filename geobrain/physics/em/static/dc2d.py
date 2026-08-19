"""
2D DC resistivity (Direct Current).

Cell-centered finite-volume Poisson assembly on a 2D ``TensorMesh``,
torch-native dense solve via :func:`linear_solve_with_adjoint`. The
implementation follows the physics conventions (symmetric Dirichlet
pin, harmonic-mean face conductivities, multi-RHS readiness) but stays
on the dense-matrix + adjoint-solve substrate, no scipy.sparse, no
hand-written autograd bridge, no OctreeMesh dependency.

Governing equation:

    −∇ · (σ ∇φ) = q                                    in Ω
    ∂φ/∂n      = 0                                     at z = 0 (free surface)
    φ          = 0                                     at one pinned cell

5-point stencil with harmonic-mean σ on each interior face:

    σ_face = 2 σ_l σ_r / (σ_l + σ_r)

Each face contributes a symmetric ``±T = ±σ_face / spacing²`` block to
``A``. One far cell (default **bottom-center**) is pinned to ``φ = 0``
via a symmetric row+column replacement, removes the singular Neumann
nullspace without over-constraining the problem the way the legacy
three-wall Dirichlet did. Bottom-center is chosen (instead of the
corner) because typical DC surveys are x-mirror-symmetric about the
mesh midpoint; a center pin preserves that symmetry in the linear
system and gives antisymmetric potentials for symmetric electrode
arrangements.

The discretised system ``A φ = b`` is solved via
:func:`~geobrain.core.adjoint.linear_solve_with_adjoint`.
Gradients flow through both ``A(σ)`` and ``b`` per the implicit function
theorem. Operator level: ``IMPLICIT_VJP``.

Multi-RHS readiness: ``A`` is assembled once and ``torch.linalg.solve``
accepts either ``(n_cells,)`` or ``(n_cells, n_rhs)`` RHS shapes, so
future multi-quadripole surveys plug in without re-assembling.

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
    linear_solve_with_adjoint,
)
from ....mesh import TensorMesh
from ....mesh import require_field_matches_mesh
from ....mesh.capabilities import UniformMesh
from ....core.survey import coords_to_cell_indices
from ..numerics.finite_volume.poisson import assemble_poisson_2d


@dataclass(frozen=True)
class DC2DSurvey:
    """
    4-electrode DC resistivity geometry on a 2D grid.

    Single source / sink current pair, an array of receiver locations.
    All positions are **physical coordinates in metres**, the repo-wide survey
    convention (see :mod:`geobrain.core.survey`), matching DC3DSurvey. ``_forward``
    snaps each to its containing cell via ``floor(coord / spacing)``. Build from
    grid indices with :meth:`from_grid_indices`.

    Attributes:
        source_z, source_x: metre coordinate of the current injection electrode
            (positive current ``+I``).
        sink_z, sink_x:     metre coordinate of the current sink (``−I``).
        rcv_z, rcv_x:       1-D float tensors of receiver coordinates (metres).
        current:            magnitude of injected current, in amperes.
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
                    f"DC2DSurvey.{name} must be a torch.Tensor",
                    object_name="DC2DSurvey",
                    field=name,
                    expected=torch.Tensor,
                    actual=type(t),
                )
            if t.ndim != 1:
                raise GeoBrainError(
                    f"DC2DSurvey.{name} must be 1D",
                    object_name="DC2DSurvey",
                    field=name,
                    expected="1D Tensor",
                    actual=tuple(t.shape),
                )
        if self.rcv_z.numel() != self.rcv_x.numel():
            raise GeoBrainError(
                "rcv_z and rcv_x must have matching length",
                object_name="DC2DSurvey",
                field="rcv_x",
                expected=self.rcv_z.numel(),
                actual=self.rcv_x.numel(),
            )
        if self.current <= 0:
            raise GeoBrainError(
                "current must be positive",
                object_name="DC2DSurvey",
                field="current",
                expected="> 0",
                actual=self.current,
            )

    @property
    def n_rcv(self) -> int:
        return int(self.rcv_z.numel())

    @classmethod
    def from_grid_indices(
        cls, source_z, source_x, sink_z, sink_x, rcv_z, rcv_x,
        *, spacing, current: float = 1.0,
    ) -> "DC2DSurvey":
        """Build a metre-coordinate survey from grid indices (back-compat).

        ``spacing`` is the target mesh's ``(dz, dx)``; index ``i`` maps to the
        cell centre ``(i + 0.5)·spacing`` so it round-trips exactly through the
        ``floor(coord / spacing)`` snap in :meth:`DC2D._forward`.
        """
        dz, dx = float(spacing[0]), float(spacing[1])
        rz = torch.as_tensor(rcv_z, dtype=torch.float64)
        rx = torch.as_tensor(rcv_x, dtype=torch.float64)
        return cls(
            source_z=(float(source_z) + 0.5) * dz,
            source_x=(float(source_x) + 0.5) * dx,
            sink_z=(float(sink_z) + 0.5) * dz,
            sink_x=(float(sink_x) + 0.5) * dx,
            rcv_z=(rz + 0.5) * dz,
            rcv_x=(rx + 0.5) * dx,
            current=current,
        )


def assemble_dc2d_system(
    sigma: torch.Tensor,
    dx: float,
    dz: float,
    *,
    dirichlet_idx: int | None = None,
) -> torch.Tensor:
    """
    Assemble the symmetric ``(n, n)`` DC operator ``A`` for a 2D mesh.

    Free surface (Neumann) at ``i = 0``; far-corner Dirichlet pin
    (default ``(nz - 1, nx - 1)``) breaks the otherwise singular
    Neumann nullspace. Symmetric pin replacement preserves ``A = Aᵀ``.

    Args:
        sigma: ``(nz, nx)`` conductivity tensor (S/m).
        dx, dz: cell spacings (m).
        dirichlet_idx: cell flat index where ``φ = 0`` is enforced.
            ``None`` → default ``(nz - 1) * nx + (nx - 1)`` (bottom-right corner).

    Returns:
        ``(n_cells, n_cells)`` dense ``A`` with autograd hooked through
        ``sigma`` and the harmonic-mean face conductivities.
    """
    nz, nx = sigma.shape
    n = nz * nx
    device, dtype = sigma.device, sigma.dtype

    if dirichlet_idx is None:
        # Bottom-center cell: preserves x-mirror symmetry for typical
        # surveys with electrodes laid out symmetrically along z = 0.
        dirichlet_idx = (nz - 1) * nx + (nx // 2)
    if not (0 <= int(dirichlet_idx) < n):
        raise GeoBrainError(
            "dirichlet_idx out of range",
            object_name="assemble_dc2d_system",
            field="dirichlet_idx",
            expected=f"[0, {n})",
            actual=dirichlet_idx,
        )
    pin = int(dirichlet_idx)

    # Pure −∇·(σ∇) discretisation with harmonic-mean face conductivities
    # and free-flux (Neumann) boundaries is lifted into the shared
    # ``numerics/finite_volume`` layer; the DC2D-specific symmetric
    # Dirichlet pin overlay is applied below to remove the otherwise
    # singular Neumann nullspace.
    mesh = TensorMesh(shape=(nz, nx), spacing=(dz, dx))
    A = assemble_poisson_2d(mesh, sigma)

    # --- Symmetric Dirichlet pin: replace row+column ``pin`` with identity ---
    #
    # We zero row ``pin`` and column ``pin`` (preserving symmetry) and set
    # A[pin, pin] = 1. The matching RHS row is set to 0 so the linear system
    # forces phi[pin] = 0 cleanly.
    #
    # Implemented as a masked overlay so autograd flows through the unpinned
    # part of A as before.
    n_arr = torch.arange(n, device=device)
    pin_mask = (n_arr == pin).to(dtype)                  # (n,), one-hot at pin
    A = A * (1.0 - pin_mask.unsqueeze(1)) * (1.0 - pin_mask.unsqueeze(0))
    A = A + torch.outer(pin_mask, pin_mask)              # set A[pin, pin] = 1
    return A


class DC2D(EMOperatorDiscovery, ForwardOperator):
    """
    2D DC resistivity forward operator.

    Inputs (ModelState):  ``sigma`` shape ``(nz, nx)`` matching the mesh.

    Outputs (ForwardOutput):
        ``data["voltage"]``:       receiver voltages, shape ``(n_rcv,)``.
        ``fields["potential"]``:   full 2D ``φ`` field, shape ``(nz, nx)`` (diagnostic).
        ``metadata["pin"]``:       flat cell index of the Dirichlet pin.

    Args:
        survey: 2-D galvanic acquisition.
        dirichlet_idx: node pinned to zero potential (``None`` = auto).
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        output_keys=("voltage",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (UniformMesh,)

    def __init__(
        self, survey: DC2DSurvey, *, dirichlet_idx: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, DC2DSurvey):
            raise GeoBrainError(
                "DC2D requires a DC2DSurvey",
                object_name="DC2D",
                field="survey",
                expected=DC2DSurvey,
                actual=type(survey),
            )
        self.survey = survey
        self._dirichlet_idx = dirichlet_idx

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (sigma,) = state.fetch("sigma")
        mesh = ctx.require_mesh()
        if mesh.n_dim != 2:
            raise GeoBrainError(
                "DC2D needs a 2D mesh",
                object_name="DC2D",
                field="mesh",
                expected="n_dim=2",
                actual=mesh.n_dim,
            )
        require_field_matches_mesh(mesh, sigma, name="sigma", owner="DC2D")
        # Physical conductivity must be finite and strictly positive. A sigma<=0
        # cell makes the DC system indefinite (still solvable, returning finite
        # but meaningless voltages); reject it (vectorized) once the field's
        # structure (mesh dim + shape) has been validated.
        if not bool(torch.isfinite(sigma).all()) or bool((sigma <= 0).any()):
            raise GeoBrainError(
                "DC2D requires a finite, strictly-positive conductivity field "
                "(sigma > 0 S/m); a non-physical sigma <= 0 (or non-finite) makes "
                "the DC system indefinite and yields meaningless voltages",
                object_name="DC2D", field="sigma",
                expected="finite and all > 0",
                actual="contains sigma <= 0 or non-finite entries",
            )

        nz, nx = mesh.shape
        n = nz * nx
        dz, dx = mesh.spacing
        oz, ox = mesh.origin
        device, dtype = sigma.device, sigma.dtype

        pin = (
            self._dirichlet_idx
            if self._dirichlet_idx is not None
            else (nz - 1) * nx + (nx // 2)
        )
        A = assemble_dc2d_system(sigma, dx=dx, dz=dz, dirichlet_idx=pin)

        # Source / sink current vector. Electrode positions are physical metres;
        # snap each to its containing cell (floor(coord/spacing), the repo-wide
        # rule). Pin row must be 0 so the symmetric pin enforces phi[pin] = 0
        # cleanly. ``scatter_add`` is autograd-safe.
        sz = coords_to_cell_indices(self.survey.source_z, dz, nz, origin=oz)
        sx = coords_to_cell_indices(self.survey.source_x, dx, nx, origin=ox)
        kz = coords_to_cell_indices(self.survey.sink_z, dz, nz, origin=oz)
        kx = coords_to_cell_indices(self.survey.sink_x, dx, nx, origin=ox)
        src_flat = sz * nx + sx
        snk_flat = kz * nx + kx
        b = torch.zeros(n, dtype=dtype, device=device).scatter_add(
            0,
            torch.tensor([src_flat, snk_flat], device=device, dtype=torch.long),
            torch.tensor(
                [+self.survey.current, -self.survey.current],
                dtype=dtype, device=device,
            ),
        )
        if src_flat == pin or snk_flat == pin:
            # Pin collides with an electrode: zero the RHS row defensively.
            b = b.clone()
            b[pin] = 0.0

        phi = linear_solve_with_adjoint(A, b)             # (n,)
        phi_2d = phi.view(nz, nx)

        rcv_z = coords_to_cell_indices(self.survey.rcv_z, dz, nz, origin=oz).to(device)
        rcv_x = coords_to_cell_indices(self.survey.rcv_x, dx, nx, origin=ox).to(device)
        voltage = phi_2d[rcv_z, rcv_x]                    # (n_rcv,)

        return ForwardOutput(
            data={"voltage": voltage},
            fields={"potential": phi_2d},  # full φ field, diagnostic, not an observation
            metadata={
                "units": {"voltage": "V"},
                "n_rcv": self.survey.n_rcv,
                "pin": pin,
            },
        )
