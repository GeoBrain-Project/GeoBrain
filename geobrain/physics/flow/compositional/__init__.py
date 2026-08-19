"""
Compositional (multi-component EOS) flash for reservoir flow.

A PyTorch, end-to-end-differentiable cubic-EOS flash stack, built bottom-up:

    compositional/
    ├── mixture.py         # Component, Mixture, Wilson K-value estimate
    ├── rachford_rice.py   # solve_rachford_rice (vapor fraction)
    └── ...                # (cubic EOS, two-phase flash, stability, added next)

Units are SI (Pa, K, kg/mol, m³/mol).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from .cubic_eos import CubicEOS
from .flash import (
    FlashResult,
    StabilityFlashResult,
    flash,
    flash_2ph,
    require_flash_converged,
)
from .mixture import Component, Mixture
from .model import CompositionalModel
from .mpfa_model import MPFACompositionalModel, MPFACompositionalModel3D
from .mpfa_thermal_model import MPFAThermalCompositionalModel, MPFAThermalCompositionalModel3D
from .rachford_rice import rachford_rice_residual, solve_rachford_rice
from .stability import StabilityResult, stability_test
from .viscosity import lbc_viscosity

__all__ = (
    "Component",
    "CompositionalModel",
    "MPFACompositionalModel",
    "MPFACompositionalModel3D",
    "MPFAThermalCompositionalModel",
    "MPFAThermalCompositionalModel3D",
    "CubicEOS",
    "lbc_viscosity",
    "FlashResult",
    "StabilityFlashResult",
    "Mixture",
    "StabilityResult",
    "flash",
    "flash_2ph",
    "require_flash_converged",
    "rachford_rice_residual",
    "solve_rachford_rice",
    "stability_test",
)
