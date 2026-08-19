"""Potential-fields helpers: constants, direction cosines, result dataclasses.

The ``AbstractPFModel`` / ``AbstractPFSimulator`` class hierarchy is intentionally
not provided here: the architecture uses the ``ForwardOperator`` layer instead.
Canonical station geometry lives in ``PotentialSurvey2D`` and
``PotentialSurvey3D`` (shared by the gravity and magnetic operators).

Contents:
    Constants:
        ``G_SI``, ``MU_0``, ``M_TO_MGAL``, ``M_TO_EOTVOS``, ``T_TO_NT``, ``EPS``
    Helper:
        ``field_direction_cosines``: public since downstream code in
        processing / corrections / operators relies on it.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import torch

from ...core.constants import MU_0 as _CORE_MU_0

# =========================================================================
# Section: Physical constants and unit conversions
# =========================================================================

# Newton's gravitational constant (m^3 / kg / s^2)
G_SI: float = 6.6743e-11

# Vacuum permeability (T * m / A): single-sourced from core.constants
# (CODATA-2018), which replaces the classical 4*pi*1e-7 value platform-wide
# (exact only in the pre-2019 SI; relative difference ~5.4e-10).
MU_0: float = _CORE_MU_0

# Unit conversions
M_TO_MGAL: float = 1.0e5  # m / s^2 -> mGal
M_TO_EOTVOS: float = 1.0e9  # 1 / s^2 -> Eotvos
T_TO_NT: float = 1.0e9  # Tesla -> nanoTesla

# Numerical-stability floor used by prism / log / arctan kernels.
EPS: float = 1.0e-30


# =========================================================================
# Section: Direction-cosine helper (canonical convention)
# =========================================================================
#
# Convention: x = East, y = North, z = Down (positive
# into earth). Inclination is the angle between the field vector and the
# horizontal plane, positive when the field dips downward. Declination is
# measured clockwise from geographic north (+y -> +x).
#
#     lx = cos(inc) * sin(dec)   # east
#     ly = cos(inc) * cos(dec)   # north
#     lz = sin(inc)              # down


def field_direction_cosines(
    inclination_deg: float,
    declination_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return ``(lx, ly, lz)`` direction cosines for the standard PF convention.

    Convention: ``x = East``, ``y = North``, ``z = Down``. Inclination is
    positive when the field dips downward. Declination is measured clockwise
    from +y (geographic north) toward +x (east). Inputs are in degrees.

    Returns each component as a scalar ``torch.float64`` tensor so callers can
    feed the result directly into autograd-tracked computations.

    Args:
        inclination_deg: field inclination [deg, down-positive].
        declination_deg: field declination [deg, east of north].
    """
    inc_rad = math.radians(inclination_deg)
    dec_rad = math.radians(declination_deg)
    cos_inc = math.cos(inc_rad)
    lx = torch.tensor(cos_inc * math.sin(dec_rad), dtype=torch.float64)
    ly = torch.tensor(cos_inc * math.cos(dec_rad), dtype=torch.float64)
    lz = torch.tensor(math.sin(inc_rad), dtype=torch.float64)
    return lx, ly, lz


__all__ = [
    "EPS",
    "G_SI",
    "M_TO_EOTVOS",
    "M_TO_MGAL",
    "MU_0",
    "T_TO_NT",
    "field_direction_cosines",
]
