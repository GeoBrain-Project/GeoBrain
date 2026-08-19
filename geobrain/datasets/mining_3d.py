"""
Mining-3D synthetic drillhole dataset.

200
drillhole-style 3-D points with a lognormal gold-grade column
``Au`` enriched near ``z = −100``. Deterministic via fixed seed
``123``.

Coordinates: ``x, y ∈ [0, 500]``, ``z ∈ [−200, 0]``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np

from geobrain.geomodel.frames._arrays import as_float_array, column_stack_float
from geobrain.geomodel.frames import GeoFrame, GeoPoints


def mining_3d() -> GeoFrame:
    """Return the mining-3D synthetic GeoFrame (200 rows, Au column)."""
    rng = np.random.default_rng(123)
    n = 200
    x = as_float_array(rng.uniform(0.0, 500.0, n))
    y = as_float_array(rng.uniform(0.0, 500.0, n))
    z = as_float_array(rng.uniform(-200.0, 0.0, n))

    base = as_float_array(np.exp(-0.01 * np.abs(z + 100.0)))
    Au = as_float_array(
        np.maximum(0.01, rng.lognormal(mean=-1.0, sigma=0.8, size=n) * base * 3.0)
    )
    coords = column_stack_float([x, y, z])
    return GeoFrame(GeoPoints(coords), properties={"Au": Au})


__all__ = ["mining_3d"]
