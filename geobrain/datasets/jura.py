"""
Jura dataset (synthetic approximation).

Heavy-metal-style
benchmark with 359 points and 7 metal columns (``Cd``, ``Co``,
``Cr``, ``Cu``, ``Ni``, ``Pb``, ``Zn``) plus an integer ``Rock``
class. Deterministic via fixed seed ``101``.

Like the other datasets here, this is a *synthetic
approximation* of the published Atteia et al. (1994) dataset;
shape and column names are matched, but values come from a
sinusoid + lognormal-noise process.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np

from geobrain.geomodel.frames._arrays import as_float_array, column_stack_float
from geobrain.geomodel.frames import GeoFrame, GeoPoints


def jura() -> GeoFrame:
    """Return the Jura synthetic GeoFrame (359 rows, 7 metals + Rock)."""
    rng = np.random.default_rng(101)
    n = 359
    x = as_float_array(rng.uniform(0.0, 6.0, n))
    y = as_float_array(rng.uniform(0.0, 6.0, n))
    base = as_float_array(np.sin(x * 1.5) * np.cos(y * 1.2))

    Cd = as_float_array(np.maximum(0.1, 1.5 + 0.8 * base + rng.normal(0.0, 0.5, n)))
    Co = as_float_array(np.maximum(0.5, 10.0 + 3.0 * base + rng.normal(0.0, 2.0, n)))
    Cr = as_float_array(np.maximum(1.0, 30.0 + 10.0 * base + rng.normal(0.0, 5.0, n)))
    Cu = as_float_array(np.maximum(0.5, 15.0 + 5.0 * base + rng.normal(0.0, 3.0, n)))
    Ni = as_float_array(np.maximum(1.0, 20.0 + 8.0 * base + rng.normal(0.0, 4.0, n)))
    Pb = as_float_array(np.maximum(1.0, 40.0 + 12.0 * base + rng.normal(0.0, 6.0, n)))
    Zn = as_float_array(np.maximum(5.0, 80.0 + 25.0 * base + rng.normal(0.0, 10.0, n)))
    Rock = as_float_array(np.round(rng.uniform(1.0, 5.0, n)))

    coords = column_stack_float([x, y])
    return GeoFrame(
        GeoPoints(coords),
        properties={
            "Cd": Cd, "Co": Co, "Cr": Cr, "Cu": Cu,
            "Ni": Ni, "Pb": Pb, "Zn": Zn, "Rock": Rock,
        },
    )


__all__ = ["jura"]
