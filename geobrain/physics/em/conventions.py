"""Canonical production Fourier and phase conventions for the EM family.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Final

from geobrain.core.constants import MU_0 as _CORE_MU_0


# CODATA 2018 vacuum permeability: the platform canonical value, single
# sourced in geobrain.core.constants (the classical 4*pi*1e-7 differs at
# ~5.4e-10 relative and is exact only in pre-2019 SI).
# The CSEM quantized-tail literals were derived at 170 decimals against this
# constant; oracle tests must import it rather than re-declaring locally.
MU_0: Final[float] = _CORE_MU_0
EPSILON_0: Final[float] = 8.8541878128e-12
C_LIGHT: Final[float] = 2.99792458e8
ETA_0: Final[float] = 376.730313668


class EMFourierConvention(str, Enum):
    """The sole production Fourier convention: ``Re(F exp(+iωt))``."""

    PLUS_IWT = "+iwt"


class EMPhaseUnit(str, Enum):
    """The sole production phase unit."""

    RADIAN = "rad"


HOMOGENEOUS_HALFSPACE_PHASE_RAD: Final[float] = math.pi / 4.0


__all__ = [
    "EMFourierConvention",
    "EMPhaseUnit",
    "HOMOGENEOUS_HALFSPACE_PHASE_RAD",
    "MU_0",
    "EPSILON_0",
    "C_LIGHT",
    "ETA_0",
]
