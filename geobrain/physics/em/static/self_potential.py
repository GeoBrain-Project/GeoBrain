"""2D self-potential (SP): electrokinetic / streaming-current forward.

Self-potential is the passive electric field set up by a *source current
density* ``j_s`` in the subsurface (most commonly the electrokinetic streaming
current driven by groundwater flow). With no injected current, charge
conservation gives the boundary-value problem::

    −∇·(σ ∇φ) = ∇·j_s ≡ q                                  in Ω

where ``q`` is the volumetric current source (A/m³), ``σ`` the conductivity, and
``φ`` the self-potential sampled at the measurement electrodes (relative to a
reference / pinned cell). This is the **same** discrete operator as DC
resistivity, only the right-hand side differs (a distributed source ``q``
instead of a point injection), so :func:`assemble_dc2d_system` is reused.

What makes this GeoBrain's, not just another SP solver: **both** ``sigma`` and
``source_current`` are differentiable inputs. The streaming source ``q`` is
typically a differentiable function of the Darcy velocity (``physics.flow``) and
the excess charge density / coupling coefficient (``physics.rock``), and ``σ``
is itself a differentiable function of saturation/porosity via Archie, so SP
becomes a **differentiable hydro-electric coupling axis** for joint inversion:
``∂(SP data)/∂(flow/rock parameters)`` flows by autograd through ``q`` and ``σ``.
Operator level: ``IMPLICIT_VJP``.

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
    ForwardOperator,
    GeoBrainError,
    ModelState,
    ForwardOutput,
    linear_solve_with_adjoint,
)
from ....mesh import require_field_matches_mesh
from ....mesh.capabilities import UniformMesh
from ....core.survey import coords_to_cell_indices
from .dc2d import assemble_dc2d_system


@dataclass(frozen=True)
class SelfPotentialSurvey:
    """Measurement electrodes for a 2D self-potential survey.

    SP has no current injection; the potential is driven by the distributed
    ``source_current`` in the :class:`ModelState`. Potentials are reported
    relative to the reference (the Dirichlet-pinned cell). Electrode positions
    are **physical coordinates in metres** (repo-wide convention, matching DC);
    ``_forward`` snaps each to its cell. Build from grid indices with
    :meth:`from_grid_indices`.

    Attributes:
        rcv_z, rcv_x: 1-D float tensors of measurement-electrode coordinates (m).
    """

    rcv_z: torch.Tensor
    rcv_x: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("rcv_z", "rcv_x"):
            t = getattr(self, name)
            if not isinstance(t, torch.Tensor):
                raise GeoBrainError(
                    f"SelfPotentialSurvey.{name} must be a torch.Tensor",
                    object_name="SelfPotentialSurvey", field=name,
                    expected=torch.Tensor, actual=type(t),
                )
            if t.ndim != 1:
                raise GeoBrainError(
                    f"SelfPotentialSurvey.{name} must be 1D",
                    object_name="SelfPotentialSurvey", field=name,
                    expected="1D Tensor", actual=tuple(t.shape),
                )
        if self.rcv_z.numel() != self.rcv_x.numel():
            raise GeoBrainError(
                "rcv_z and rcv_x must have matching length",
                object_name="SelfPotentialSurvey", field="rcv_x",
                expected=self.rcv_z.numel(), actual=self.rcv_x.numel(),
            )
        if self.rcv_z.numel() == 0:
            raise GeoBrainError(
                "SelfPotentialSurvey needs at least one receiver electrode",
                object_name="SelfPotentialSurvey", field="rcv_z",
                expected="> 0 receivers", actual=0,
            )

    @property
    def n_rcv(self) -> int:
        return int(self.rcv_z.numel())

    @classmethod
    def from_grid_indices(cls, rcv_z, rcv_x, *, spacing) -> "SelfPotentialSurvey":
        """Build a metre-coordinate SP survey from grid indices (back-compat).

        ``spacing`` is the target mesh's ``(dz, dx)``; index ``i`` maps to the
        cell centre ``(i + 0.5)·spacing`` (round-trips through ``floor``).
        """
        dz, dx = float(spacing[0]), float(spacing[1])
        rz = torch.as_tensor(rcv_z, dtype=torch.float64)
        rx = torch.as_tensor(rcv_x, dtype=torch.float64)
        return cls(rcv_z=(rz + 0.5) * dz, rcv_x=(rx + 0.5) * dx)


class SelfPotential2D(EMOperatorDiscovery, ForwardOperator):
    """2D self-potential forward operator (electrokinetic / streaming current).

    Inputs (``ModelState``):
        - ``sigma``: conductivity, shape ``(nz, nx)`` matching the mesh.
        - ``source_current``: volumetric current source ``q = ∇·j_s`` per
          cell, shape ``(nz, nx)`` (A per cell, same per-cell convention as
          the DC right-hand side). Differentiable: a Darcy-driven
          streaming current.

    Outputs (``ForwardOutput``):
        - ``data["sp"]``: self-potential at the electrodes (V),
          ``(n_rcv,)``.
        - ``fields["potential"]``: full 2D ``φ`` field, shape
          ``(nz, nx)`` (diagnostic).

    Differentiable (``IMPLICIT_VJP``) through both ``sigma`` and
    ``source_current``: the hydro-electric coupling axis.

    Args:
        survey: SP station acquisition.
        dirichlet_idx: node pinned to zero potential (``None`` = auto).
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma", "source_current"),
        output_keys=("sp",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (UniformMesh,)

    def __init__(
        self, survey: SelfPotentialSurvey, *, dirichlet_idx: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, SelfPotentialSurvey):
            raise GeoBrainError(
                "SelfPotential2D requires a SelfPotentialSurvey",
                object_name="SelfPotential2D", field="survey",
                expected=SelfPotentialSurvey, actual=type(survey),
            )
        self.survey = survey
        self._dirichlet_idx = dirichlet_idx

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        sigma, source_current = state.fetch("sigma", "source_current")
        mesh = ctx.require_mesh()
        if mesh.n_dim != 2:
            raise GeoBrainError(
                "SelfPotential2D needs a 2D mesh",
                object_name="SelfPotential2D", field="mesh",
                expected="n_dim=2", actual=mesh.n_dim,
            )
        require_field_matches_mesh(mesh, sigma, name="sigma",
                                   owner="SelfPotential2D")
        require_field_matches_mesh(mesh, source_current, name="source_current",
                                   owner="SelfPotential2D")

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

        # RHS = distributed source current. The pinned (reference) row must be
        # zero so the symmetric pin enforces φ[pin] = 0; mask it out (autograd-safe).
        keep = torch.ones(n, dtype=dtype, device=device)
        keep[pin] = 0.0
        b = source_current.reshape(n) * keep

        phi = linear_solve_with_adjoint(A, b)
        phi_2d = phi.view(nz, nx)

        rcv_z = coords_to_cell_indices(self.survey.rcv_z, dz, nz, origin=oz).to(device)
        rcv_x = coords_to_cell_indices(self.survey.rcv_x, dx, nx, origin=ox).to(device)
        sp = phi_2d[rcv_z, rcv_x]

        return ForwardOutput(
            data={"sp": sp},
            fields={"potential": phi_2d},  # full φ field, diagnostic, not an observation
            metadata={"units": {"sp": "V"}, "n_rcv": self.survey.n_rcv, "pin": pin},
        )
