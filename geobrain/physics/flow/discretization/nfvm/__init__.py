"""
Nonlinear finite-volume method (NFVM): monotone NTPFA / NMPFA flux for full-tensor
permeability.

A PyTorch implementation of the Aavatsmark / Nordbotten nonlinear finite-volume
method. The linear MPFA-O is consistent but loses the discrete maximum
principle on full-tensor / strongly-anisotropic problems; NFVM restores it by a
*nonlinear* two-point flux with non-negative effective transmissibility.

Construction, per cell-face ("half face"):

  * **Harmonic averaging points**: across each of the cell's faces a
    HAP is placed where harmonic flux continuity holds, with interpolation weights
    ``(w_self, w_other)`` (pressure at the HAP = ``w_self·p_self + w_other·p_other``).
  * **Positive basis**: the co-normal ``A·K·n`` is written as a
    **non-negative** combination of two (2-D) HAP directions ``Σ ω_i (hap_i − x_cell)``,
    a conical decomposition (``find_minimizing_basis``).
  * **One-sided discretization**: collapsing the HAP
    interpolation gives a one-sided flux ``t_l·p_l + t_r·p_r + Σ_c t_c·p_c`` whose
    transverse part (cells other than ``l, r``) is the remainder.

Per interior face the two one-sided fluxes are combined nonlinearly:
``F = μ_L q_L − μ_R q_R`` with ``μ_L = r_R/(r_L+r_R)``,
``μ_R = r_L/(r_L+r_R)`` (``:ntpfa`` uses signed remainders, ``:nmpfa`` their
magnitudes). The remainders cancel, leaving a two-point flux whose effective
transmissibility is non-negative (monotone) while staying consistent (a linear
field is reproduced exactly). The system is nonlinear in pressure (Newton).

This package provides the core NFVM operators (geometry-generic) plus a structured
Cartesian driver with ghost cells for Dirichlet boundaries. Units are SI.

Layout:

  * :mod:`.kernel`:     the reusable kernel island (HAP / positive basis /
    half-face decomposition, :class:`NFVMGeometry`, ``_build_nfvm``, ``nfvm_flux``,
    ``solve_nfvm``, ``_oriented_normal``).
  * :mod:`.steady`:     single-phase steady drivers (structured 2-D / 3-D + unstructured).
  * :mod:`.transient`: the physics solvers (two-phase + thermal single / two-phase /
    compositional) and the robustification helpers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .kernel import (
    NFVMGeometry,
    _basis_coefficients as _basis_coefficients,
    _build_nfvm as _build_nfvm,
    _duo_coefficients as _duo_coefficients,
    _extend_to_ghosts as _extend_to_ghosts,
    _onesided as _onesided,
    _oriented_normal as _oriented_normal,
    _two_point_trans as _two_point_trans,
    decompose_half_face,
    find_minimizing_basis,
    find_minimizing_basis_2d,
    harmonic_average_point,
    linear_discretization,
    nfvm_flux,
    solve_nfvm,
)
from .steady import (
    _CartNFVM as _CartNFVM,
    solve_nfvm_steady,
    solve_nfvm_steady_3d,
    solve_nfvm_unstructured,
)
from .transient import (
    _adaptive_march as _adaptive_march,
    _newton_solve as _newton_solve,
    _SAT_BOUND_TOL as _SAT_BOUND_TOL,
    nfvm_thermal_compositional,
    nfvm_thermal_conduction,
    nfvm_thermal_single_phase,
    nfvm_thermal_two_phase,
    nfvm_two_phase,
)

__all__ = ["harmonic_average_point", "find_minimizing_basis", "find_minimizing_basis_2d",
           "decompose_half_face", "linear_discretization", "nfvm_flux",
           "NFVMGeometry", "solve_nfvm", "solve_nfvm_steady",
           "solve_nfvm_unstructured", "solve_nfvm_steady_3d", "nfvm_two_phase",
           "nfvm_thermal_conduction", "nfvm_thermal_single_phase", "nfvm_thermal_two_phase",
           "nfvm_thermal_compositional"]
