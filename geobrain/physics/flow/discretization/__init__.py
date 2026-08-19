"""Spatial discretization: flux schemes (MPFA O-method 2D/3D, nonlinear FVM),
transmissibility, and boundary conditions for the flow solvers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .boundary import FlowBoundary, FlowBoundaryGroup
from .flux import (
    STANDARD_GRAVITY_M_S2,
    phase_potential,
    scatter_boundary_outflow,
    scatter_internal_face_flux,
    upwind_cell,
)
from .tpfa import tpfa_face_transmissibility, tpfa_phase_flux

__all__ = (
    "FlowBoundary",
    "FlowBoundaryGroup",
    "STANDARD_GRAVITY_M_S2",
    "phase_potential",
    "scatter_boundary_outflow",
    "scatter_internal_face_flux",
    "tpfa_face_transmissibility",
    "tpfa_phase_flux",
    "upwind_cell",
)
