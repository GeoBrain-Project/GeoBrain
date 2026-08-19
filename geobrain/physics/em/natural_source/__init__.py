"""Natural-source EM family: magnetotellurics (1D / 2D / 3D).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .mt1d import MT1D, MT1DSurvey
from .mt2d import MT2D, MT2DSurvey, assemble_mt2d_system
from .mt3d import MT3D, MT3DStation, MT3DSurvey

__all__ = [
    "MT1D",
    "MT2D",
    "MT2DSurvey",
    "MT3D",
    "MT3DStation",
    "MT3DSurvey",
    "MT1DSurvey",
    "assemble_mt2d_system",
]
