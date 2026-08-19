"""
Frequency-domain EM operators.

E3a ships CSEM1D (marine controlled-source, 1D-layered, VMD source).
E6d ships FDEM3D (3D frequency-domain magnetic-dipole) + HEM facade.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .csem1d import CSEM1D, CSEMReceiver, MarineCSEM1DSurvey, VMDSource
from .fdem_cyl import FDEMCyl, FDEMCylSurvey
from .fdem3d import (
    FDEM3D,
    FDEM3DReceiver,
    FDEM3DSurvey,
    MagneticDipoleSource,
)
from .hem import HEM, HEMSurvey

__all__ = [
    "CSEM1D",
    "CSEMReceiver",
    "FDEM3D",
    "FDEM3DReceiver",
    "FDEM3DSurvey",
    "FDEMCyl",
    "FDEMCylSurvey",
    "HEM",
    "HEMSurvey",
    "MagneticDipoleSource",
    "MarineCSEM1DSurvey",
    "VMDSource",
]
