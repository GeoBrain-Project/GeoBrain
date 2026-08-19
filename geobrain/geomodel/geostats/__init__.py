"""Geostatistics: variograms, kriging, simulation, transforms, validation.

This is the geostatistical algorithm family within :mod:`geobrain.geomodel`,
sibling to :mod:`geobrain.geomodel.implicit` (implicit modelling) and
:mod:`geobrain.geomodel.generative` (generative AI).

The top-level :mod:`geobrain.geomodel` re-exports the common entry points
(``OrdinaryKriging``, ``SGSIM``, …); reach in here directly for the full
catalog.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from . import (
    estimation,
    models,
    postprocessing,
    simulation,
    transform,
    validate,
)
from .estimation import *  # noqa: F401,F403
from .models import *  # noqa: F401,F403
from .postprocessing import *  # noqa: F401,F403
from .simulation import *  # noqa: F401,F403
from .transform import *  # noqa: F401,F403
from .validate import *  # noqa: F401,F403

__all__ = [
    "estimation",
    "models",
    "postprocessing",
    "simulation",
    "transform",
    "validate",
]
