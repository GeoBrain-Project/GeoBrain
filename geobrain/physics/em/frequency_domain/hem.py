"""
HEM (Helicopter EM) airborne facade over FDEM3D.

For a flight line, build one batched FDEM3DSurvey whose transmitter columns
share each frequency's matrix/factor. The paired BZ receiver at each station
flies ``flight_altitude`` metres ABOVE the air-earth interface and is separated
from its transmitter by ``coil_offset`` in x. Only paired Tx/Rx responses are
returned; the Cartesian off-diagonal responses remain internal.

z convention (platform-wide, shared with MT3D/FDEM3D): z is DEPTH,
positive DOWNWARD, with the air-earth interface at ``z = surface_z``
(default 0). A bird at altitude ``h`` therefore sits at::

    z_bird = surface_z - flight_altitude

i.e. at SMALLER z than the surface, never at ``z = +flight_altitude``,
which would bury the bird ``h`` metres underground.

"Zero new physics", per the docstring. Autograd flows through the one batched
FDEM3D call and the paired diagonal gather.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

from geobrain.physics.em.config import EMExecutionConfig
from geobrain.core import GeoBrainError

from dataclasses import dataclass, field
from typing import ClassVar

import torch
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from geobrain.core import (
    ForwardContext,
    ModelState,
    ForwardOperator,
    ForwardOutput,
)
from geobrain.core.differentiability import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
)
from geobrain.core.linalg import SparseFactorSolver
from geobrain.mesh.capabilities import (
    ConnectivityMesh,
    EdgeConnectivityMesh,
    StructuredMesh,
)
from geobrain.physics.em.results import FieldComponent
from geobrain.physics.em.surveys import (
    FrequencyDomainSurvey,
)

from .fdem3d import (
    FDEM3D,
    FDEM3DReceiver,
    FDEM3DSurvey,
    MagneticDipoleSource,
)


@dataclass(frozen=True)
class HEMSurvey(FrequencyDomainSurvey):
    """
    HEM airborne survey: flight stations + frequencies + coil geometry.

    Each station carries an implicit co-located VMD source + BZ receiver
    flying ``flight_altitude`` metres above the air-earth interface,
    separated by ``coil_offset`` in x (Tx and Rx share the same z).

    z convention: the platform z axis is DEPTH, positive DOWNWARD, with
    the air-earth interface at ``z = surface_z`` in mesh coordinates
    (default ``0.0``, surface at the mesh top). The bird is placed at
    ``z = surface_z - flight_altitude``, i.e. ABOVE the surface.

    Because :class:`~geobrain.mesh.TensorMesh` nodes start at ``z = 0``
    (no negative coordinates), a physically faithful airborne survey
    should build the mesh with an air pad (``sigma ~ 1e-8`` S/m)
    occupying mesh-local ``z in [0, surface_z)`` and set ``surface_z``
    to the pad thickness, so the bird at ``surface_z - flight_altitude``
    lies INSIDE the air cells and the secondary-field receiver sample is
    taken in the air rather than clamped to the mesh top.

    Attributes:
        sources / receivers: acquisition tables.
        frequencies: system frequencies [Hz].
        stations_xy: flight-line station coordinates [m].
        flight_altitude: sensor altitude [m].
        coil_offset: transmitter-receiver offset [m].
        surface_z: ground elevation [m].
        magnetic_moment_am2: transmitter dipole moment [A*m^2].
        sigma_background_1d: layered background conductivity [S/m].
    """

    stations_xy: tuple[tuple[float, float], ...] = ()
    flight_altitude: float = 30.0
    coil_offset: float = 8.0
    surface_z: float = 0.0
    magnetic_moment_am2: float = field(kw_only=True)
    # Optional background conductivity for the wholespace primary, forwarded
    # verbatim to each per-station FDEM3DSurvey (zero new physics). For a
    # physically faithful airborne configuration set this to the AIR value
    # (e.g. 1e-8 S/m): the primary is then the free-space dipole field and
    # the secondary RHS couples only where sigma differs from air: i.e. the
    # earth. None keeps FDEM3D's default (detached global cell-mean sigma).
    sigma_background_1d: torch.Tensor | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.magnetic_moment_am2) is not float
            or not math.isfinite(self.magnetic_moment_am2)
            or self.magnetic_moment_am2 <= 0.0
        ):
            raise GeoBrainError(
                "HEMSurvey.magnetic_moment_am2 must be a finite positive float",
                object_name="HEMSurvey",
                field="magnetic_moment_am2",
                expected="finite float > 0",
                actual=self.magnetic_moment_am2,
            )


class HEM(EMOperatorDiscovery, ForwardOperator):
    """HEM airborne facade: one batched FDEM3D survey over all flight stations.

    Output: a single native-complex ``ForwardOutput.data["bz"]`` of shape
    ``(n_stations, n_freq)`` (the platform-wide complex-data contract).

    Args:
        survey: helicopter-EM acquisition.
        config: :class:`~geobrain.physics.em.EMExecutionConfig` execution
            policy (solver selection).
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        output_keys=("bz",),
    )
    # HEM iterates the per-station FDEM3D via its private ``_forward`` (see
    # below), which bypasses FDEM3D's forward-wrapper capability enforcement.
    # HEM therefore honors the mesh-capability contract itself: the SAME
    # capability FDEM3D/VTEM declare, so HEM.forward's wrapper runs the
    # capability check once for the whole (shared-mesh) survey.
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
        survey: HEMSurvey,
        *,
        config: EMExecutionConfig | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, HEMSurvey):
            raise GeoBrainError(
                f"HEM survey must be HEMSurvey, got {type(survey).__name__}",
            )
        self.survey = survey
        cfg = config if config is not None else EMExecutionConfig()
        self._solver = cfg.resolve_solver()

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        (sigma_3d,) = state.fetch("sigma")
        offset = self.survey.coil_offset
        # z is DEPTH (positive down, surface at z = surface_z): a bird at
        # flight_altitude metres ABOVE the surface sits at
        # z = surface_z - flight_altitude (smaller z), NOT +flight_altitude
        # (which would be flight_altitude metres underground). Tx and Rx
        # share z_bird; the coil offset is purely horizontal (in-line x).
        z_bird = self.survey.surface_z - self.survey.flight_altitude

        if not self.survey.stations_xy:
            raise GeoBrainError(
                "HEMSurvey.stations_xy must be non-empty for forward execution",
                object_name="HEM",
                field="stations_xy",
                expected="at least one station",
                actual=0,
            )

        # All stations at one frequency share the same numerical matrix. Build
        # one Cartesian FDEM survey so station transmitters ride as RHS columns
        # through one factorization, then retain the co-located Tx/Rx diagonal.
        # Receiver coordinates never affect the matrix key.
        sources = tuple(
            MagneticDipoleSource(
                position=(sx, sy, z_bird),
                orientation=(0.0, 0.0, 1.0),
                magnetic_moment_am2=self.survey.magnetic_moment_am2,
            )
            for sx, sy in self.survey.stations_xy
        )
        receivers = tuple(
            FDEM3DReceiver(
                position=(sx + offset, sy, z_bird),
                component=FieldComponent.BZ,
            )
            for sx, sy in self.survey.stations_xy
        )
        fdem_survey = FDEM3DSurvey(
            sources=sources,
            receivers=receivers,
            frequencies=self.survey.frequencies,
            sigma_background_1d=self.survey.sigma_background_1d,
        )
        fdem_output = FDEM3D(
            fdem_survey, config=EMExecutionConfig(solver=self._solver) if self._solver is not None else None
        )._forward(state, ctx)
        cartesian = fdem_output.data["bz"]
        bz = torch.stack([cartesian[index, index] for index in range(len(sources))])
        metadata = dict(fdem_output.metadata)
        metadata.update(
            {
                "method": "hem",
                "station_layout": "paired",
                "n_stations": len(sources),
                "magnetic_moment_am2": self.survey.magnetic_moment_am2,
            }
        )
        return ForwardOutput(data={"bz": bz}, metadata=metadata)
