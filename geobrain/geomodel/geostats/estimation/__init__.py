"""
Kriging-style estimators.

Public symbols:

- :class:`SimpleKriging`, :class:`OrdinaryKriging`,
  :class:`UniversalKriging`: point estimators.
- :func:`covariance_matrix`, :func:`covariance_vector`,
  anisotropic-aware pairwise covariance.
- :func:`drift_basis`: polynomial-drift basis used by UK.
- :func:`krige_loop`: low-level kernel loop shared by all three
  estimators.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .block_kriging import BlockKriging
from .cokriging import CollocatedCokriging
from .covariance_matrix import covariance_matrix, covariance_vector
from .drift import drift_basis
from .indicator_kriging import IndicatorKriging
from .kriging import KrigingSolvePolicy, OrdinaryKriging, SimpleKriging, UniversalKriging
from .kriging_kernel import constraint_count, krige_loop
from .soft_indicator_kriging import SoftIndicatorKriging

__all__ = [
    "BlockKriging",
    "CollocatedCokriging",
    "IndicatorKriging",
    "KrigingSolvePolicy",
    "OrdinaryKriging",
    "SimpleKriging",
    "SoftIndicatorKriging",
    "UniversalKriging",
    "constraint_count",
    "covariance_matrix",
    "covariance_vector",
    "drift_basis",
    "krige_loop",
]
