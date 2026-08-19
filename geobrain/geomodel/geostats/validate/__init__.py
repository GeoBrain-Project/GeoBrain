"""
Cross-validation and model diagnostics.

Public surface:

- :class:`CrossValidationResult`: dataclass holding actuals /
  estimates / variances / errors / fold-index.
- :func:`leave_one_out`, :func:`k_fold`: module-level
  convenience runners.
- :class:`LeaveOneOut`, :class:`KFold`: class wrappers exposing
  a ``.run()`` / ``.summary()`` API.
- :func:`cross_validation_report`, :func:`msdr`: summary statistics
  (RMSE / MAE / mean error / Pearson correlation / mean &
  variance of standardized error / MSDR).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .cross_validation import (
    CrossValidationResult,
    KFold,
    LeaveOneOut,
    SpatialEstimator,
    k_fold,
    leave_one_out,
)
from .diagnostics import cross_validation_report, msdr

__all__ = [
    "CrossValidationResult",
    "KFold",
    "LeaveOneOut",
    "SpatialEstimator",
    "cross_validation_report",
    "k_fold",
    "leave_one_out",
    "msdr",
]
