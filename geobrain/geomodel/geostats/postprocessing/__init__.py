"""
Post-processing of geomodel outputs.

Public symbols:

- :class:`PostIndicatorKriging`: collapse a CCDF (``prob_0`` …
  ``prob_K-1`` columns from :class:`IndicatorKriging` /
  :class:`SoftIndicatorKriging`) into ``estimate`` (E-type mean) and
  ``variance`` per cell.
- :class:`PostSimulation`: aggregate a list of realisations
  into per-cell mean / variance / percentile / exceedance columns.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .maps import MAPSCleaning
from .postik import PostIndicatorKriging
from .postsim import PostSimulation

__all__ = ["MAPSCleaning", "PostIndicatorKriging", "PostSimulation"]
