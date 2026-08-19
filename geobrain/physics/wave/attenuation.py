"""Immutable attenuating Wave facades (production-accepted).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar

from geobrain.core import DifferentiabilityLevel, DifferentiabilitySpec

from ._engine.equations.viscoacoustic import ViscoAcousticVelocityStress
from ._engine.equations.viscoelastic import ViscoElasticVelocityStress
from ._facade import _TimeDomainFacade
from .acquisition import Seismic2DSurvey
from .capabilities import WaveCapabilityReport, WaveUnsupportedCombination
from .config import WaveSimulationConfig


class _FixedReferenceAttenuationFacade(_TimeDomainFacade):
    """Expose the immutable calibration convention to Agent/UI discovery."""

    _attenuation_rheology: ClassVar[str]
    _attenuation_quality_fields: ClassVar[tuple[str, ...]]

    @classmethod
    def capabilities(cls) -> WaveCapabilityReport:
        """Add the fixed-reference exclusion without widening the stable config."""
        report = super().capabilities()
        fixed_reference = WaveUnsupportedCombination(
            selection=(("attenuation.reference_frequency_hz", "other-than-15"),),
            reason=(
                "the GeoBrain 0.2.0 attenuation facade is calibrated at a fixed 15 Hz"
            ),
            remediation=(
                "resample model Q for 15 Hz or use a separately calibrated equation"
            ),
        )
        return replace(report, unsupported=(*report.unsupported, fixed_reference))

    @classmethod
    def input_schema(cls) -> Mapping[str, object]:
        """Describe fixed attenuation calibration as non-input schema metadata."""
        schema = dict(super().input_schema())
        schema["x-geobrain-attenuation"] = {
            "rheology": cls._attenuation_rheology,
            "reference_frequency_hz": 15.0,
            "reference_frequency_configurable": False,
            "validated_frequency_band_hz": [7.5, 30.0],
            "quality_factor_fields": list(cls._attenuation_quality_fields),
        }
        return schema


class ViscoAcoustic2D(_FixedReferenceAttenuationFacade):
    """Packed two-dimensional constant-Q acoustic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "rho", "Q"),
        output_keys=("seismic",),
        input_units={"vp": "m/s", "rho": "kg/m^3", "Q": "1"},
    )
    _dimension = 2
    _maturity = "production"
    _physics = "viscoacoustic"
    _model_fields = ("vp", "rho", "Q")
    _attenuation_rheology = "single-standard-linear-solid"
    _attenuation_quality_fields = ("Q",)
    _survey_type = Seismic2DSurvey
    _supports_boundary = False
    _boundary_unsupported_reason = "dissipative state is not reversibly reconstructible"

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> ViscoAcousticVelocityStress:
        return ViscoAcousticVelocityStress(fd_order=config.discretization.fd_order)


class ViscoElastic2D(_FixedReferenceAttenuationFacade):
    """Packed two-dimensional constant-Q elastic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho", "Qp", "Qs"),
        output_keys=("seismic",),
        input_units={"vp": "m/s", "vs": "m/s", "rho": "kg/m^3", "Qp": "1", "Qs": "1"},
    )
    _dimension = 2
    _maturity = "production"
    _physics = "viscoelastic"
    _model_fields = ("vp", "vs", "rho", "Qp", "Qs")
    _attenuation_rheology = "single-generalized-standard-linear-solid"
    _attenuation_quality_fields = ("Qp", "Qs")
    _survey_type = Seismic2DSurvey
    _supports_boundary = False
    _boundary_unsupported_reason = "dissipative state is not reversibly reconstructible"

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> ViscoElasticVelocityStress:
        return ViscoElasticVelocityStress(fd_order=config.discretization.fd_order)


__all__ = ["ViscoAcoustic2D", "ViscoElastic2D"]
