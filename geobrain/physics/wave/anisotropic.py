"""Immutable anisotropic elastic Wave facades.

The facades use the complete tilted-TI model. Setting ``theta=0`` selects VTI;
zero Thomsen parameters recover isotropic elasticity. Both dimensions carry
production acceptance evidence.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from geobrain.core import DifferentiabilityLevel, DifferentiabilitySpec

from ._engine.equations.elastic import ElasticTTI
from ._engine.equations.elastic3d import ElasticTTI3D
from ._facade import _TimeDomainFacade
from .acquisition import Seismic2DSurvey, Seismic3DSurvey
from .config import WaveSimulationConfig
from .errors import WaveNumericsError


def _validate_ti_hyperbolicity(object_name: str, model: Mapping[str, torch.Tensor]) -> None:
    """Reject real-valued Thomsen stiffnesses with negative phase-speed square."""
    vp, vs, rho = model["vp"], model["vs"], model["rho"]
    c33 = rho * vp.square()
    c44 = rho * vs.square()
    c11 = c33 * (1.0 + 2.0 * model["epsilon"])
    difference = c33 - c44
    c13 = torch.sqrt(difference.square() + 2.0 * model["delta"] * c33 * difference) - c44
    coupling_squared = (c13 + c44).square()
    coefficient0 = c44 * c33
    coefficient1 = c44 * (c44 - c33) + (c11 - c44) * c33 - coupling_squared
    coefficient2 = (c11 - c44) * (c44 - c33) + coupling_squared
    endpoint0 = coefficient0
    endpoint1 = coefficient0 + coefficient1 + coefficient2
    convex = coefficient2 > 0.0
    vertex = -coefficient1 / torch.where(convex, 2.0 * coefficient2, torch.ones_like(coefficient2))
    interior_vertex = convex & (vertex > 0.0) & (vertex < 1.0)
    safe_vertex = torch.where(interior_vertex, vertex, torch.zeros_like(vertex))
    vertex_value = coefficient0 + coefficient1 * safe_vertex + coefficient2 * safe_vertex.square()
    minimum_determinant = torch.minimum(endpoint0, endpoint1)
    minimum_determinant = torch.minimum(
        minimum_determinant,
        torch.where(interior_vertex, vertex_value, minimum_determinant),
    )
    with torch.no_grad():
        minimum_stability = float(minimum_determinant.min())
    if minimum_stability < 0.0:
        raise WaveNumericsError(
            "invalid Wave numerical input",
            object_name=object_name,
            field="delta",
            expected="minimum x-z Christoffel determinant >= 0",
            actual=minimum_stability,
            hint="adjust Thomsen parameters so every phase-speed squared is non-negative",
        )


class AnisotropicElastic2D(_TimeDomainFacade):
    """Packed two-dimensional tilted transverse-isotropic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho", "epsilon", "delta", "theta"),
        output_keys=("seismic",),
        input_units={
            "vp": "m/s",
            "vs": "m/s",
            "rho": "kg/m^3",
            "epsilon": "1",
            "delta": "1",
            "theta": "rad",
        },
    )
    _dimension = 2
    _maturity = "production"
    _physics = "anisotropic-elastic"
    _model_fields = ("vp", "vs", "rho", "epsilon", "delta", "theta")
    _survey_type = Seismic2DSurvey
    _supports_boundary = False
    _boundary_unsupported_reason = (
        "boundary-memory reconstruction is not supported for anisotropic state"
    )
    _boundary_unsupported_remediation = (
        "select full, checkpoint, or recursive memory"
    )

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> ElasticTTI:
        return ElasticTTI(fd_order=config.discretization.fd_order)

    def _validate_constitutive(self, model: Mapping[str, torch.Tensor]) -> None:
        super()._validate_constitutive(model)
        _validate_ti_hyperbolicity(type(self).__name__, model)


class AnisotropicElastic3D(_TimeDomainFacade):
    """Packed three-dimensional tilted transverse-isotropic propagation.

    Args:
        survey: packed acquisition (:class:`Seismic2DSurvey` /
            :class:`Seismic3DSurvey`).
        wavelets: per-shot source time functions ``(n_shots, nt)`` (build
            shared ones with :func:`shared_wavelet`).
        config: optional :class:`WaveSimulationConfig` override.
    """

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("vp", "vs", "rho", "epsilon", "delta", "gamma", "theta"),
        output_keys=("seismic",),
        input_units={
            "vp": "m/s",
            "vs": "m/s",
            "rho": "kg/m^3",
            "epsilon": "1",
            "delta": "1",
            "gamma": "1",
            "theta": "rad",
        },
    )
    _dimension = 3
    _maturity = "production"
    _physics = "anisotropic-elastic"
    _model_fields = ("vp", "vs", "rho", "epsilon", "delta", "gamma", "theta")
    _survey_type = Seismic3DSurvey
    _supports_boundary = False
    _boundary_unsupported_reason = (
        "boundary-memory reconstruction is not supported for anisotropic state"
    )
    _boundary_unsupported_remediation = (
        "select full, checkpoint, or recursive memory"
    )

    @classmethod
    def _new_equation(cls, config: WaveSimulationConfig) -> ElasticTTI3D:
        return ElasticTTI3D(fd_order=config.discretization.fd_order)

    def _validate_constitutive(self, model: Mapping[str, torch.Tensor]) -> None:
        super()._validate_constitutive(model)
        _validate_ti_hyperbolicity(type(self).__name__, model)


__all__ = ["AnisotropicElastic2D", "AnisotropicElastic3D"]
