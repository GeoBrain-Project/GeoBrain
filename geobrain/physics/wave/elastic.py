"""Public immutable isotropic elastic Wave facades.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core import DifferentiabilityLevel, DifferentiabilitySpec

from ._engine.equations.elastic import ElasticVelocityStress
from ._engine.equations.elastic3d import ElasticVelocityStress3D
from ._facade import _TimeDomainFacade
from .acquisition import Seismic2DSurvey, Seismic3DSurvey
from .config import WaveSimulationConfig


class Elastic2D(_TimeDomainFacade):
    """Packed two-dimensional isotropic elastic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho"),
        output_keys=("seismic",),
        input_units={"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"},
    )
    _dimension = 2
    _physics = "elastic"
    _maturity = "production"
    _model_fields = ("vp", "vs", "rho")
    _survey_type = Seismic2DSurvey
    _supports_native = True
    _supports_free_surface = True
    _native_component_sets = (("pressure",), ("pressure", "vx", "vz"))

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> ElasticVelocityStress:
        return ElasticVelocityStress(fd_order=config.discretization.fd_order)


class Elastic3D(_TimeDomainFacade):
    """Packed three-dimensional isotropic elastic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = Elastic2D.differentiability
    _dimension = 3
    _physics = "elastic"
    _maturity = "production"
    _model_fields = ("vp", "vs", "rho")
    _survey_type = Seismic3DSurvey
    _supports_native = False
    _native_unsupported_reason = (
        "native Elastic3D still uses the legacy z-y-x component ABI"
    )
    _native_unsupported_remediation = (
        "select the eager backend until a public-axis native adapter is validated"
    )

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> ElasticVelocityStress3D:
        return ElasticVelocityStress3D(fd_order=config.discretization.fd_order)


__all__ = ["Elastic2D", "Elastic3D"]
