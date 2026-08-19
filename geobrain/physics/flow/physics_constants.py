"""Immutable SI physical constants used by Flow production kernels.

``STANDARD_GRAVITY_M_S2`` re-exports the platform canon
(:mod:`geobrain.core.constants`) so the value is single-sourced.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from geobrain.core.constants import STANDARD_GRAVITY as STANDARD_GRAVITY_M_S2

STANDARD_ATMOSPHERE_PA: float = 101_325.0

__all__ = ["STANDARD_ATMOSPHERE_PA", "STANDARD_GRAVITY_M_S2"]
