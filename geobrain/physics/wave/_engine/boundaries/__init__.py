"""Absorbing boundary conditions.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .cpml import CPML, build_cpml, damping_profile_1d

__all__ = ["CPML", "build_cpml", "damping_profile_1d"]
