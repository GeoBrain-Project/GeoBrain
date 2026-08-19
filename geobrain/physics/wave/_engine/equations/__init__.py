"""Wave equations (the math layer): one time step, independent of time loop.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .acoustic import AcousticVelocityStress
from .acoustic3d import AcousticVelocityStress3D
from .base import FieldSpec, ModelSpec, WaveEquation
from .elastic import ElasticTTI, ElasticVelocityStress, ElasticVTI
from .elastic3d import ElasticVelocityStress3D, ElasticTTI3D, ElasticVTI3D
from .viscoacoustic import ViscoAcousticVelocityStress
from .viscoelastic import ViscoElasticVelocityStress

__all__ = [
    "AcousticVelocityStress",
    "AcousticVelocityStress3D",
    "ElasticVelocityStress",
    "ElasticVTI",
    "ElasticTTI",
    "ElasticVelocityStress3D",
    "ElasticVTI3D",
    "ElasticTTI3D",
    "ViscoAcousticVelocityStress",
    "ViscoElasticVelocityStress",
    "FieldSpec",
    "ModelSpec",
    "WaveEquation",
]
