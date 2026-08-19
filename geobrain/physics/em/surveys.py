"""Shared electromagnetic source, receiver, and survey records in SI units.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from geobrain.core import GeoBrainError


@dataclass(frozen=True, slots=True)
class EMSource:
    """Point source position ``(x, y, z)`` in metres."""

    position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MagneticDipoleSource(EMSource):
    """Magnetic-dipole source with an explicit finite moment in A·m².

    Attributes:
        position: dipole location [m].
        magnetic_moment_am2: dipole moment [A*m^2].
        orientation: dipole axis unit vector.
    """

    magnetic_moment_am2: float
    orientation: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if not math.isfinite(self.magnetic_moment_am2) or self.magnetic_moment_am2 <= 0.0:
            raise GeoBrainError(
                "MagneticDipoleSource.magnetic_moment_am2 must be finite and positive",
                object_name="MagneticDipoleSource",
                field="magnetic_moment_am2",
                expected="finite value > 0 A·m²",
                actual=self.magnetic_moment_am2,
            )


@dataclass(frozen=True, slots=True)
class CentralLoopSource(EMSource):
    """Circular loop source with explicit radius, current, and turns.

    Attributes:
        position: loop centre [m].
        radius_m: loop radius [m].
        current_a: transmitter current [A].
        turns: number of loop turns.
    """

    radius_m: float
    current_a: float
    turns: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise GeoBrainError("CentralLoopSource.radius_m must be finite and positive")
        if not math.isfinite(self.current_a) or self.current_a == 0.0:
            raise GeoBrainError("CentralLoopSource.current_a must be finite and non-zero")
        if type(self.turns) is not int or self.turns < 1:
            raise GeoBrainError("CentralLoopSource.turns must be a positive integer")

    @property
    def magnetic_moment_am2(self) -> float:
        """Return ``current_a × turns × π × radius_m²`` in A·m²."""
        return self.current_a * self.turns * math.pi * self.radius_m**2


@dataclass(frozen=True, slots=True)
class EMReceiver:
    """Point receiver position ``(x, y, z)`` in metres.

    Components are physical world order, x along mesh axis-1, y along mesh
    axis-2, z depth (positive DOWN), deliberately NOT the ``(z, x, y)`` mesh
    column order that :meth:`~geobrain.mesh.Mesh.cell_centers` uses. Coming
    from an elevation-up frame (GIS convention), negate z or run the stacked
    coordinates through :func:`geobrain.mesh.apply_axes_spec` first.
    """

    position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TimeWaveform:
    """Source-current waveform sampled at ``dt`` second intervals.

    Attributes:
        samples: waveform samples.
        dt: sample interval [s].
    """

    samples: tuple[float, ...]
    dt: float


@dataclass(frozen=True, slots=True)
class EMSurvey:
    """Abstract source/receiver survey base class.

    Coordinate frame (family-wide contract): every spatial entry on the
    ``sources``/``receivers`` tables carries physical ``(x, y, z)``
    components in metres, x along mesh axis-1, y along mesh axis-2, z
    depth-down; see :class:`EMReceiver`. Concrete surveys that bind to a
    grid instead document their index semantics on their own fields.

    Attributes:
        sources / receivers: the acquisition tables.
    """

    sources: tuple[EMSource, ...]
    receivers: tuple[EMReceiver, ...]

    def __post_init__(self) -> None:
        if type(self) is EMSurvey:
            raise TypeError("EMSurvey is abstract, instantiate a concrete survey type.")


@dataclass(frozen=True, slots=True)
class FrequencyDomainSurvey(EMSurvey):
    """Frequency-domain survey with frequencies in Hz.

    Attributes:
        sources / receivers: acquisition tables.
        frequencies: modelling frequencies [Hz].
    """

    frequencies: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class TimeDomainSurvey(EMSurvey):
    """Time-domain survey with observation times in seconds.

    Attributes:
        sources / receivers: acquisition tables.
        times: gate times [s].
        waveform: transmitter :class:`TimeWaveform`.
    """

    times: tuple[float, ...] = ()
    waveform: TimeWaveform | None = None


@dataclass(frozen=True, slots=True)
class GalvanicSurvey(EMSurvey):
    """DC/IP survey electrode-index records.

    Attributes:
        sources / receivers: electrode coordinate tables.
        a_electrode_idx / b_electrode_idx: current-electrode indices.
        m_electrode_idx / n_electrode_idx: potential-electrode indices.
    """

    a_electrode_idx: tuple[int, ...] = ()
    b_electrode_idx: tuple[int, ...] = ()
    m_electrode_idx: tuple[int, ...] = ()
    n_electrode_idx: tuple[int, ...] = ()


__all__ = [
    "CentralLoopSource",
    "EMReceiver",
    "EMSource",
    "EMSurvey",
    "FrequencyDomainSurvey",
    "GalvanicSurvey",
    "MagneticDipoleSource",
    "TimeDomainSurvey",
    "TimeWaveform",
]
