"""
Time-domain EM operators.

E3b ships TEM1D (central-loop 1D-layered ``dB_z/dt``). E6e ships TEM3D
(3D time-domain magnetic-dipole) and the VTEM airborne facade.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .tem1d import (
    CentralLoopSource,
    TEM1D,
    TEM1DReceiver,
    TEM1DSurvey,
    WaveformTEM1D,
)
from .tem3d import (
    MagneticDipoleSource,
    TEM3D,
    TEM3DReceiver,
    TEM3DSurvey,
)
from .vtem import VTEM, VTEMSurvey

__all__ = [
    "CentralLoopSource",
    "TEM1D",
    "TEM1DReceiver",
    "TEM1DSurvey",
    "WaveformTEM1D",
    "TEM3D",
    "MagneticDipoleSource",
    "TEM3DReceiver",
    "TEM3DSurvey",
    "VTEM",
    "VTEMSurvey",
]
