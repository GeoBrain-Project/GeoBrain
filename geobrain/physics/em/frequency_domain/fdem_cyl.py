"""FDEMCyl: axisymmetric frequency-domain EM forward operator.

The registered-operator face of the entry-exact ``cyl_eb`` numerics:
a horizontal current loop over a
conductivity model on a :class:`~geobrain.mesh.cylindrical.CylindricalMesh`,
solved in the entry-exact mimetic E_φ/B staggered system and sampled
as ``b_z`` at surface (vertical-annulus face) receivers.

Device: CPU / float64-complex128 only (scipy splu). Differentiability:
:attr:`IMPLICIT_VJP` through ``sigma``; the σ-path is the closed-form
diagonal Jacobian ``dA[e,e]/dσ_c = iω·V_c/4`` on the quarter-ring-volume
support, through the same splu σ-adjoint bridge as the Yee curl-curl
family (:func:`~...cyl_eb.solve_fdem_cyl_autograd`); the stamped-loop RHS
is σ-independent.

Conventions: ``e^{+iωt}`` assembly; the emitted ``bz`` follows the
reference data convention this stack is validated against (native
complex, NO final conjugation, the cross-validation suite pins the
native loop source at 1–5 % of the reference primary field).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from geobrain.core.containers import ForwardOutput, ModelState
from geobrain.core.context import ForwardContext
from geobrain.core.differentiability import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
)
from geobrain.core.errors import GeoBrainError
from geobrain.mesh.capabilities import CylindricalGeometryMesh
from geobrain.core.operator import ForwardOperator
from geobrain.physics.em.numerics.finite_volume.cyl_eb import (
    build_cyl_eb_system,
    loop_edge_source,
    solve_fdem_cyl_autograd,
    surface_bz_face_indices,
)

__all__ = ["FDEMCyl", "FDEMCylSurvey"]


@dataclass(frozen=True)
class FDEMCylSurvey:
    """Axisymmetric loop-source FDEM acquisition.

    Args:
        frequencies: source frequencies in Hz.
        loop_radius: transmitter loop radius (m); keep it on an r-node for
            an exact edge representation (the source snaps to the nearest
            θ-edge).
        loop_z: loop elevation in the platform z-down frame (0 = surface
            when the mesh origin puts the surface at z = 0).
        current: loop current (A).
        rx_r: receiver offsets (m); b_z is sampled at the vertical-annulus
            face nearest ``(rx_z, r)`` per offset.
        rx_z: receiver elevation (z-down), default the loop plane.
    """

    frequencies: tuple[float, ...]
    loop_radius: float
    rx_r: tuple[float, ...]
    loop_z: float = 0.0
    current: float = 1.0
    rx_z: float = 0.0

    def __post_init__(self) -> None:
        if not self.frequencies or any(f <= 0 for f in self.frequencies):
            raise GeoBrainError(
                "frequencies must be a non-empty tuple of positive Hz",
                object_name="FDEMCylSurvey", field="frequencies",
                expected="positive frequencies", actual=self.frequencies,
            )
        if self.loop_radius <= 0:
            raise GeoBrainError(
                "loop_radius must be positive",
                object_name="FDEMCylSurvey", field="loop_radius",
                expected="> 0", actual=self.loop_radius,
            )
        if not self.rx_r:
            raise GeoBrainError(
                "rx_r must name at least one receiver offset",
                object_name="FDEMCylSurvey", field="rx_r",
                expected="non-empty tuple", actual=self.rx_r,
            )


class FDEMCyl(EMOperatorDiscovery, ForwardOperator):
    """Axisymmetric loop-source FDEM on a CylindricalMesh (E_φ mimetic).

    Maps a :class:`ModelState` holding ``"sigma"`` (shape ``(nz, nr)`` or
    flat ``(n_cells,)``) to ``data["bz"]``, a native-complex
    ``(n_src=1, n_rx, n_freq)`` tensor matching the platform's EM data
    layout. See the module docstring for conventions and validation.
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        output_keys=("bz",),
    )
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = (
        CylindricalGeometryMesh,
    )

    def __init__(self, survey: FDEMCylSurvey) -> None:
        super().__init__()
        if not isinstance(survey, FDEMCylSurvey):
            raise GeoBrainError(
                "FDEMCyl requires an FDEMCylSurvey",
                object_name="FDEMCyl", field="survey",
                expected="FDEMCylSurvey", actual=type(survey).__name__,
            )
        self.survey = survey

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        mesh = ctx.require_mesh()
        (sigma,) = state.fetch("sigma")
        nz, nr = mesh.shape
        if int(sigma.numel()) != nz * nr:
            raise GeoBrainError(
                "sigma size mismatches the cylindrical mesh",
                object_name="FDEMCyl", field="sigma",
                expected=(nz, nr), actual=tuple(sigma.shape),
            )
        sigma64 = sigma.reshape(-1).to(torch.float64)

        # Geometry + M_e support from the DETACHED values; the live σ-path
        # re-enters through the closed-form Jacobian in the autograd solve.
        system = build_cyl_eb_system(mesh, sigma64.detach())
        s_e = loop_edge_source(
            system, radius=self.survey.loop_radius,
            z=self.survey.loop_z, current=self.survey.current,
        )
        rx_faces = surface_bz_face_indices(
            system, np.asarray(self.survey.rx_r, dtype=np.float64),
            z=self.survey.rx_z,
        )
        curl_t = torch.sparse_coo_tensor(
            torch.stack([
                torch.as_tensor(system.curl.tocoo().row, dtype=torch.long),
                torch.as_tensor(system.curl.tocoo().col, dtype=torch.long),
            ]),
            torch.as_tensor(system.curl.tocoo().data, dtype=torch.complex128),
            size=system.curl.shape,
        ).coalesce()

        per_freq = []
        for f in self.survey.frequencies:
            omega = 2.0 * np.pi * float(f)
            e = solve_fdem_cyl_autograd(system, omega, sigma64, s_e)
            b = -torch.sparse.mm(curl_t, e.unsqueeze(1)).reshape(-1) \
                / (1j * omega)
            per_freq.append(b[torch.as_tensor(rx_faces, dtype=torch.long)])
        bz = torch.stack(per_freq, dim=1).unsqueeze(0)   # (1, n_rx, n_freq)
        return ForwardOutput(data={"bz": bz})
