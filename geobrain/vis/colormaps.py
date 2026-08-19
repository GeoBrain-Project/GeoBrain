"""Geophysics-specific colormaps.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from geobrain.core.errors import GeoBrainError


def _make_colormap(name, colors):
    """Create LinearSegmentedColormap from list of (r, g, b) tuples."""
    return LinearSegmentedColormap.from_list(name, colors, N=256)


GEO_COLORMAPS = {
    "geo_seismic": _make_colormap(
        "geo_seismic",
        [
            (0.0, 0.0, 0.8),
            (1.0, 1.0, 1.0),
            (0.8, 0.0, 0.0),
        ],
    ),
    "geo_velocity": _make_colormap(
        "geo_velocity",
        [
            (0.0, 0.0, 0.6),
            (0.0, 0.5, 0.3),
            (0.9, 0.9, 0.0),
            (0.8, 0.0, 0.0),
        ],
    ),
    "geo_resistivity": _make_colormap(
        "geo_resistivity",
        [
            (0.3, 0.0, 0.5),
            (0.0, 0.2, 0.8),
            (0.0, 0.7, 0.5),
            (0.9, 0.9, 0.0),
            (0.8, 0.2, 0.0),
            (0.5, 0.0, 0.0),
        ],
    ),
    "geo_density": _make_colormap(
        "geo_density",
        [
            (0.5, 0.3, 0.1),
            (1.0, 1.0, 1.0),
            (0.1, 0.3, 0.6),
        ],
    ),
    "geo_porosity": _make_colormap(
        "geo_porosity",
        [
            (1.0, 1.0, 1.0),
            (0.1, 0.4, 0.8),
        ],
    ),
}


_registered = False


def register_colormaps() -> None:
    """Register all geophysics colormaps with matplotlib.cm."""
    global _registered
    if _registered:
        return
    for name, cmap in GEO_COLORMAPS.items():
        try:
            plt.colormaps.register(cmap, name=name)
        except ValueError:
            pass  # already registered
    _registered = True


def get_colormap(name: str):
    """
    Get a registered geophysics colormap by name.

    Raises ValueError if name not found.
    """
    if name in GEO_COLORMAPS:
        return GEO_COLORMAPS[name]
    raise GeoBrainError(f"Colormap '{name}' not found. Available: {list(GEO_COLORMAPS.keys())}")
