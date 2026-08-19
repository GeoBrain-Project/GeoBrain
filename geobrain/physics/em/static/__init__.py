"""Static EM family: DC resistivity, induced polarisation, spectral IP.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .dc2d import DC2D, DC2DSurvey, assemble_dc2d_system
from .dc25d import DC25D, DC25DSurvey, wavenumber_quadrature
from .dc3d import (
    DC3D,
    DC3DSurvey,
    BoundDipoleDipoleSurvey,
    DipoleDipoleSurvey,
    assemble_dc3d_system,
)
from .ip import (
    IP2D,
    IP3D,
    IPChargeabilityModel,
    IPSimulator,
)
from .self_potential import SelfPotential2D, SelfPotentialSurvey
from .sip  import SIP, SIPColeColeModel, SIPSurvey, cole_cole_eta

__all__ = [
    "DC2D",
    "DC2DSurvey",
    "DC25D",
    "DC25DSurvey",
    "wavenumber_quadrature",
    "DC3D",
    "DC3DSurvey",
    "DipoleDipoleSurvey",
    "BoundDipoleDipoleSurvey",
    "SelfPotential2D",
    "SelfPotentialSurvey",
    "IP2D",
    "IP3D",
    "IPChargeabilityModel",
    "IPSimulator",
    "SIP",
    "SIPColeColeModel",
    "SIPSurvey",
    "assemble_dc2d_system",
    "assemble_dc3d_system",
    "cole_cole_eta",
]
