"""
GeoFrame column transforms.

Public symbols:

- :class:`Transform`: abstract base.
- :class:`Pipeline`: sequential composition (``t1 | t2``).
- :class:`NormalScore`: quantile-matching to standard normal,
  invertible via GSLIB tail-handling.
- :class:`IndicatorEncode`: cumulative indicator encoding at
  supplied thresholds; not invertible.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .base import Pipeline, Transform
from .decluster import Decluster
from .histogram import HistogramSmooth, ScatterSmooth
from .indicator import IndicatorEncode
from .normal_score import NormalScore
from .power_transforms import BoxCox, YeoJohnson
from .trend import Detrend

__all__ = [
    "BoxCox",
    "Decluster",
    "Detrend",
    "HistogramSmooth",
    "IndicatorEncode",
    "NormalScore",
    "Pipeline",
    "ScatterSmooth",
    "Transform",
    "YeoJohnson",
]
