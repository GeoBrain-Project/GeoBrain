"""Physical constants shared across physics families.

Single source of truth for constants that more than one family consumes.
Family-local conventions modules re-export from here so call sites keep
their idiomatic import paths while the value stays single-sourced.

``MU_0`` is the CODATA-2018 vacuum permeability, used platform-wide in
place of the classical ``4*pi*1e-7`` H/m (exact only in the pre-2019 SI,
relative difference ~5.4e-10), so every EM family shares one value.
CODATA wins: post-2019 SI treats mu_0 as measured, and the csem kernels'
high-precision literals already commit the platform to it.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from typing import Final

MU_0: Final[float] = 1.25663706212e-6
"""Vacuum permeability [H/m], CODATA-2018."""

STANDARD_GRAVITY: Final[float] = 9.80665
"""Standard gravitational acceleration [m/s^2] (CGPM 1901 exact value).
Used by the flow kernels' SI formulation (replaces the FIELD
gravity constant M = 1/144 psi per lbm/ft^3/ft)."""

__all__ = ["MU_0", "STANDARD_GRAVITY"]
