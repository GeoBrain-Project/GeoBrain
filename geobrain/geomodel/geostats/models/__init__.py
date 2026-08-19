"""
Variogram / covariance modelling.

Public symbols:

- :class:`VariogramKernel`: single nested variogram structure
  (one of 14 GSLIB families).
- :class:`CovarianceModel`: nugget + sum of structures, with
  ``variogram`` / ``covariance`` / ``correlogram`` evaluators and
  convenience constructors for the common Spherical / Exponential /
  Gaussian models.
- :func:`setup_rotation_matrix` / :func:`anisotropic_distance` /
  :func:`anisotropic_squared_distance`: GSLIB ``setrot`` /
  ``sqdist`` numerics for geometric anisotropy.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .covariance import CovarianceModel
from .cross_variogram import CrossVariogramCalculator, CrossVariogramResult
from .estimators import cressie_hawkins, dowd, genton, matheron
from .experimental import (
    ExperimentalVariogram,
    VariogramFitAttempt,
    VariogramFitDiagnostics,
    VariogramFitResult,
)
from .lmc import LinearModelOfCoregionalization
from .rotation import (
    anisotropic_distance,
    anisotropic_squared_distance,
    setup_rotation_matrix,
)
from .variogram_calculator import VariogramCalculator
from .variogram_kernel import VariogramKernel

# Backwards-compatible alias: within ``model/`` the name
# ``Variogram`` means the experimental variogram calculator.
Variogram = VariogramCalculator

__all__ = [
    "CovarianceModel",
    "CrossVariogramCalculator",
    "CrossVariogramResult",
    "ExperimentalVariogram",
    "LinearModelOfCoregionalization",
    "Variogram",
    "VariogramCalculator",
    "VariogramFitAttempt",
    "VariogramFitDiagnostics",
    "VariogramFitResult",
    "VariogramKernel",
    "anisotropic_distance",
    "anisotropic_squared_distance",
    "cressie_hawkins", "dowd", "genton", "matheron",
    "setup_rotation_matrix",
]
