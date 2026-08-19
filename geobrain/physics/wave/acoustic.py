"""Public immutable acoustic Wave facades and CFL helper.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

from geobrain.core import DifferentiabilityLevel, DifferentiabilitySpec

from ._engine.equations.acoustic import AcousticVelocityStress
from ._engine.equations.acoustic3d import AcousticVelocityStress3D
from ._facade import _TimeDomainFacade
from .acquisition import Seismic2DSurvey, Seismic3DSurvey
from .config import WaveSimulationConfig
from .errors import WaveContractError


class Acoustic2D(_TimeDomainFacade):
    """Packed two-dimensional isotropic acoustic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "rho"),
        output_keys=("seismic",),
        input_units={"vp": "m/s", "rho": "kg/m^3"},
    )
    _dimension = 2
    _physics = "acoustic"
    _model_fields = ("vp", "rho")
    _survey_type = Seismic2DSurvey
    _maturity = "production"
    _supports_native = True
    _supports_free_surface = True
    _native_component_sets = (("pressure",), ("pressure", "vx", "vz"))

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> AcousticVelocityStress:
        return AcousticVelocityStress(fd_order=config.discretization.fd_order)


class Acoustic3D(_TimeDomainFacade):
    """Packed three-dimensional isotropic acoustic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = Acoustic2D.differentiability
    _dimension = 3
    _physics = "acoustic"
    _model_fields = ("vp", "rho")
    _survey_type = Seismic3DSurvey
    _maturity = "production"
    _supports_native = True
    _native_component_sets = (("pressure",),)

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> AcousticVelocityStress3D:
        return AcousticVelocityStress3D(fd_order=config.discretization.fd_order)


def cfl_max_vp(dx: float, dz: float, dt: float, *, cfl: float = 0.606) -> float:
    """Return the historical two-dimensional acoustic CFL velocity ceiling."""
    values = (dx, dz, dt, cfl)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0.0
        for value in values
    ):
        raise WaveContractError(
            "CFL inputs must be positive",
            object_name="cfl_max_vp",
            field="dx/dz/dt/cfl",
            expected="positive finite values",
            actual=(dx, dz, dt, cfl),
        )
    return float(cfl) * min(float(dx), float(dz)) / float(dt)


__all__ = ["Acoustic2D", "Acoustic3D", "cfl_max_vp"]
