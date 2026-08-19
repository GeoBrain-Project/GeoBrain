# pyright: reportPrivateImportUsage=false
"""
VTEM (Versatile Time-domain ElectroMagnetic) airborne facade.

VTEM is the time-domain analogue of HEM: a transmitter+receiver pair on
a bird records ``dB_z/dt`` (or another component) at each station along
a flight line. The forward physics is identical to TEM3D, only the
multi-station survey geometry is new, so this module is a thin facade
that gathers all stations into one paired :class:`TEM3DSurvey` and runs a
single :class:`TEM3D` batched multi-source solve. Source column ``i`` is
projected only through receiver plan ``i``.

Mirrors the HEM-over-FDEM3D pattern. No new physics; the
differentiability of :class:`VTEM` follows directly from :class:`TEM3D`
(``IMPLICIT_VJP`` through σ via the splu-bridge adjoint).

This module ships:

- :class:`VTEMSurvey`: the multi-station flight-line dataclass.
- :class:`VTEM`: the operator that runs one batched, directly paired
  :class:`TEM3D` over all stations.

Output:
:meth:`VTEM._forward` returns a :class:`ForwardOutput` whose
``data["dbdt_z"]`` channel is a real ``float64``
tensor of shape ``(n_stations, n_time_gates)``, one ``dB/dt`` sample
per (station, gate). Sign and units follow TEM3D exactly; the VTEM facade
names this quantity ``dbdt_z`` in T/s.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.physics.em.config import EMExecutionConfig
from geobrain.core import GeoBrainError
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from dataclasses import dataclass, field
import math
from typing import ClassVar

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
from geobrain.mesh import canonicalize_cell_field, require_field_matches_mesh
from geobrain.mesh.capabilities import (
    ConnectivityMesh,
    EdgeConnectivityMesh,
    StructuredMesh,
)
from geobrain.physics.em._engine.recording import RecordingPolicy
from geobrain.physics.em.results import FieldComponent
from geobrain.physics.em.surveys import TimeDomainSurvey
from geobrain.physics.em.receivers import ReceiverLayout

from .tem3d import (
    TEM3D,
    MagneticDipoleSource,
    TEM3DReceiver,
    TEM3DSurvey,
)


@dataclass(frozen=True)
class VTEMSurvey(TimeDomainSurvey):
    """
    Airborne TEM survey: flight-line stations + per-station TEM3D geometry.

    Each station carries an implicit co-located VMD source + Bz receiver;
    the receiver sits at ``station + sensor_offset``. The default
    ``sensor_offset = (0, 0, 0)`` matches the typical VTEM bird where Tx
    and Rx are co-located.

    z convention (platform-wide, shared with :class:`HEMSurvey`): z is
    DEPTH, positive DOWNWARD, with the air-earth interface at
    ``z = surface_z`` in mesh coordinates (default ``0.0``, surface at
    the mesh top). The bird is placed at
    ``z = surface_z - flight_altitude``, i.e. ABOVE the surface. Because
    :class:`~geobrain.mesh.TensorMesh` nodes start at ``z = 0``, a
    physically faithful airborne survey should build the mesh with an air
    pad (``sigma ~ 1e-8`` S/m) occupying mesh-local ``z in [0,
    surface_z)`` and set ``surface_z`` to the pad thickness, so the bird
    lies INSIDE the air cells.

    Args:
        stations_xy: Tuple of ``(x, y)`` bird positions along the flight line.
            ``z`` is filled in as ``surface_z - flight_altitude``. Must be
            non-empty.
        flight_altitude: Sensor altitude above the air-earth interface
            (metres). Default ``30.0``.
        surface_z: Air-earth interface depth in mesh coordinates (metres,
            the air-pad thickness). Default ``0.0``.
        sensor_offset: Constant ``(dx, dy, dz)`` offset of the receiver relative
            to the transmitter for every station. Default ``(0, 0, 0)``.
        moment: Explicit finite positive transmitter VMD moment (A·m²). It has
            no implicit unit-moment default.
        time_gates: Tuple of strictly-positive, strictly-ascending observation
            times (seconds). Must be non-empty.
        n_substeps_per_decade: Per-station :class:`TEM3DSurvey` substep density.
            Default ``8`` (matches TEM3D defaults).

    The :class:`VTEMSurvey` carries the full ``stations`` (with ``z``
    baked in), ``source_orientation``, ``source_moment``, and
    ``components`` fields. the surface is narrower by design; it
    pins orientation to VMD (``+z``) and the receiver component to
    ``BZ`` to match the canonical VTEM channel. Multi-orientation /
    multi-component support can be added later without breaking this
    surface (extra fields with sensible defaults).
    """

    stations_xy: tuple[tuple[float, float], ...] = ()
    flight_altitude: float = 30.0
    sensor_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_z: float = 0.0
    magnetic_moment_am2: float = field(kw_only=True)
    time_gates: tuple[float, ...] = ()
    n_substeps_per_decade: int = 8

    def __post_init__(self) -> None:
        # Chain to EMSurvey's abstract-instance guard.
        super().__post_init__()

        if not self.stations_xy:
            raise GeoBrainError("VTEMSurvey.stations_xy must be non-empty")
        if not self.time_gates:
            raise GeoBrainError("VTEMSurvey.time_gates must be non-empty")
        for k, t in enumerate(self.time_gates):
            if not (t > 0):
                raise GeoBrainError(
                    f"VTEMSurvey.time_gates[{k}] = {t} must be strictly positive",
                )
            if k > 0 and not (t > self.time_gates[k - 1]):
                raise GeoBrainError(
                    "VTEMSurvey.time_gates must be strictly ascending; "
                    f"index {k - 1} = {self.time_gates[k - 1]} >= "
                    f"index {k} = {t}",
                )
        if (
            type(self.magnetic_moment_am2) not in (int, float)
            or not math.isfinite(float(self.magnetic_moment_am2))
            or float(self.magnetic_moment_am2) <= 0.0
        ):
            raise GeoBrainError(
                "VTEMSurvey.magnetic_moment_am2 must be explicitly provided as a finite, "
                f"strictly positive magnetic moment in A·m², got {self.magnetic_moment_am2}",
            )
        object.__setattr__(self, "magnetic_moment_am2", float(self.magnetic_moment_am2))
        if self.n_substeps_per_decade < 1:
            raise GeoBrainError(
                f"VTEMSurvey.n_substeps_per_decade must be >= 1, got {self.n_substeps_per_decade}",
            )

    @property
    def n_stations(self) -> int:
        return len(self.stations_xy)

    @property
    def n_time_gates(self) -> int:
        return len(self.time_gates)

class VTEM(EMOperatorDiscovery, ForwardOperator):
    """Airborne TEM facade: flight-line loop over TEM3D.

    Maps a :class:`ModelState` containing ``"sigma"`` (cell-centred 3D
    conductivity, shape ``mesh.shape``) to a :class:`ForwardOutput` whose
    ``data["dbdt_z"]`` channel holds ``dB_z/dt`` (T/s) of shape
    ``(n_stations, n_time_gates)``. Autograd flows through the single
    batched, paired TEM3D solve.

    The whole flight line runs in a SINGLE TEM3D (all transmitters as
    sources, all receivers, one batched ``(n_edges, n_src)`` solve), so the
    mesh-only assembly and per-``dt`` factorizations are shared across
    stations rather than rebuilt per station. The inner TEM3D is built ONCE
    at construction (its survey depends only on the frozen
    :class:`VTEMSurvey`), so TEM3D's mesh-identity-keyed assembly cache
    survives across forwards, an inversion loop pays the σ-independent
    assembly once, not once per forward.

    Args:
        survey: VTEM airborne acquisition.
        config: :class:`~geobrain.physics.em.EMExecutionConfig` execution
            policy (solver selection).
        recording_policy: which time channels are recorded.
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.IMPLICIT_VJP,
        trainable_inputs=("sigma",),
        # Canonical VTEM channel is BZ → dB_z/dt. Multi-component
        # extension can broaden this tuple later.
        output_keys=("dbdt_z",),
    )
    # Relaxed from (StructuredMesh,) alongside TEM3D's edge-element branch:
    # VTEM is a pure survey facade over TEM3D, whose in-operator dispatch
    # picks the Yee or the Nédélec path (and raises explicitly on octree).
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
        survey: VTEMSurvey,
        *,
        config: EMExecutionConfig | None = None,
        recording_policy: RecordingPolicy | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(survey, VTEMSurvey):
            raise GeoBrainError(
                f"VTEM survey must be VTEMSurvey, got {type(survey).__name__}",
            )
        self.survey = survey
        # Build the whole-line inner TEM3D ONCE: its survey is a pure
        # function of the (frozen) VTEMSurvey, so constructing it here,
        # rather than per forward: lets TEM3D's mesh-identity-keyed
        # assembly cache (`_ensure_assembly` / edge assets) actually hit
        # across repeated forwards in an inversion loop.
        # Bird above the surface, HEM parity: depth-down frame, interface
        # at z = surface_z, so the bird sits at surface_z - flight_altitude.
        z_bird = survey.surface_z - survey.flight_altitude
        off_x, off_y, off_z = survey.sensor_offset
        sources = tuple(
            MagneticDipoleSource(
                position=(float(sx), float(sy), float(z_bird)),
                magnetic_moment_am2=survey.magnetic_moment_am2,
            )
            for sx, sy in survey.stations_xy
        )
        receivers = tuple(
            TEM3DReceiver(
                position=(
                    float(sx) + float(off_x),
                    float(sy) + float(off_y),
                    float(z_bird) + float(off_z),
                ),
                component=FieldComponent.BZ,
            )
            for sx, sy in survey.stations_xy
        )
        self._line_op = TEM3D(
            survey=TEM3DSurvey(
                sources=sources,
                receivers=receivers,
                time_gates=survey.time_gates,
                n_substeps_per_decade=survey.n_substeps_per_decade,
            ),
            config=config,
            receiver_layout=ReceiverLayout.PAIRED,
            recording_policy=recording_policy,
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        """
        Whole-line forward in ONE TEM3D call (no per-station operator).

        Every station's transmitter (VMD at ``(sx, sy, altitude)``) and receiver
        (BZ at ``Tx + sensor_offset``) were gathered into a single
        :class:`TEM3DSurvey` at construction, and the ONE instance-held
        :class:`TEM3D` runs against the shared ``mesh``. TEM3D solves all
        stations as a single ``(n_edges, n_src)`` batched right-hand side, so
        the σ-independent ``K`` / ``A_e`` / time-schedule are built once for
        the whole line (and, being cached per mesh identity, once across
        repeated forwards) and the backward-Euler loop reuses one splu factor
        per ``dt`` across *all* stations, versus the old design, which span a
        full TEM3D (assembly + schedule + per-step factorizations) per station.

        VTEM is a coincident-loop system: each station records only its own
        transmitter. TEM3D therefore uses its paired layout and allocates the
        final ``(n_stations, n_gates)`` ``data["dbdt_z"]`` tensor directly.

        Note: batching holds the full ``(n_edges, n_stations)`` field across
        substeps, so peak memory scales with the number of stations; chunk a
        very long flight line into separate VTEM calls if memory-bound.
        """
        # Validate σ shape early; mirrors TEM3D's contract.
        (sigma_3d,) = state.fetch("sigma")
        mesh = ctx.require_mesh()
        if mesh.n_dim != 3:
            raise GeoBrainError(
                "VTEM requires a 3D mesh (pass the mesh via ForwardContext.of(mesh=...))",
                object_name="VTEM",
                field="mesh",
                expected="3D",
                actual=mesh.n_dim,
            )
        # Early σ-shape validation per mesh path (instance-first capability
        # read via ``Mesh.declares``: the TEM3D dispatch rule). The
        # structured Yee path takes ``mesh.shape``-shaped σ; the edge-element
        # path takes flat ``(n_cells,)`` σ; anything else (octree) is left to
        # the delegated TEM3D, which raises its explicit dispatch error.
        if mesh.declares(StructuredMesh):
            require_field_matches_mesh(mesh, sigma_3d, name="sigma", owner="VTEM")
        elif mesh.declares(EdgeConnectivityMesh):
            canonicalize_cell_field(mesh, sigma_3d, name="sigma", owner="VTEM")

        survey = self.survey
        alt = survey.flight_altitude

        # The inner TEM3D is itself paired: it projects source column ``i``
        # through receiver plan ``i`` at each gate and never constructs the
        # Cartesian source×receiver response.
        inner = self._line_op._forward(state, ctx)
        dbdt_z = inner.data["dbdt_z"]

        return ForwardOutput(
            data={"dbdt_z": dbdt_z},
            metadata={
                **inner.metadata,
                "method": "vtem",
                "n_stations": survey.n_stations,
                "n_time_gates": survey.n_time_gates,
                "flight_altitude": float(alt),
                "surface_z": float(survey.surface_z),
                "sensor_offset": tuple(float(c) for c in survey.sensor_offset),
                "magnetic_moment_am2": survey.magnetic_moment_am2,
                "source_normalization": "absolute",
                "receiver_layout": ReceiverLayout.PAIRED.value,
            },
        )
