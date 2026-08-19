"""Private numeric defaults shared by the flow kernels.

The kernels solve the SI Darcy forms directly (see
``adapters/UNIT_BOUNDARY.md`` for the unit boundary), so everything here is
unit-neutral: device/dtype defaults, saturation clamps, Newton controls, and
step-control factors, solver policy, not physics, shared by the kernels
through a private seam. (``unit_system == "FIELD"`` support in
``contracts.py`` is a live schema-declared feature at the ADAPTER boundary,
not a kernel unit system.)

This module is deliberately private and is not an Agent or public Flow
surface.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

# --- defaults --------------------------------------------------------------

DEVICE = torch.device("cpu")
DTYPE = torch.float64       # float64 for pressure-saturation residual stability


# --- Darcy / gravity unit conversions --------------------------------------
# SI:    q [m³/s]  = -k [m²]  / mu [Pa·s] · grad p [Pa/m]  · A [m²]
# Field: q [bbl/d] = -ALPHA · k [mD] / mu [cP] · grad p [psi/ft] · A [ft²]
EPS: float = 1e-12            # divide-by-zero guard


# --- Newton solver defaults -----------------------------------------------

MAX_NEWTON_ITER: int = 25
NEWTON_TOL: float = 1e-9      # max-residual convergence (CNV / MB equivalent).
# The absolute gate sits at a level meaningful for both residual unit
# systems: m³/s (TPFA) and kg/s (MPFA); for scale, the classic FIELD gate
# of 1e-3 STB/day ≈ 1.8e-9 m³/s.
NEWTON_TOL_REL: float = 1e-6  # relative residual ||r|| / ||r_0|| tolerance


# --- Time-step adaptivity -------------------------------------------------

DT_GROW_FACTOR: float = 1.5   # multiply dt on convergent step
DT_CUT_FACTOR: float = 0.5    # multiply dt on Newton failure
TARGET_ITER: int = 8          # Newton iters/step we aim for


# --- Saturation clamps ----------------------------------------------------

S_MIN: float = 1e-6
S_MAX: float = 1.0 - 1e-6


__all__ = [
    "DEVICE",
    "DTYPE",
    "DT_CUT_FACTOR",
    "DT_GROW_FACTOR",
    "EPS",
    "MAX_NEWTON_ITER",
    "NEWTON_TOL",
    "NEWTON_TOL_REL",
    "S_MAX",
    "S_MIN",
    "TARGET_ITER",
]
