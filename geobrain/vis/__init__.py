"""GeoBrain Visualization: 2-D matplotlib helpers + interactive 3-D.

This package provides general-purpose plotting routines that any geophysics
sub-discipline can reuse. The 2-D plot functions return matplotlib ``Axes``
(multi-panel helpers return ``list[Axes]``); :class:`Scene3D` is the
outward-facing interactive 3-D viewer (PyVista desktop / Jupyter, or Plotly
browser HTML, one fluent API). Domain-specific submodules must be imported
explicitly to avoid cluttering the cross-domain namespace.

Architecture::

    vis/
    ├── Spatial Plotting
    │   ├── field.py        # Scalar fields (grid, scatter, slice, tripcolor, polygon)
    │   ├── mesh.py         # Mesh geometry (gridlines, octree, triangular wireframes)
    │   ├── map.py          # Plan-view maps (anomaly, station)
    │   ├── slicer.py       # Orthogonal 3-D slice viewer (matplotlib panes)
    │   └── scene3d.py      # Interactive 3-D scene, Scene3D (PyVista / Plotly)
    ├── Analysis
    │   └── diagnostics.py  # Inversion diagnostics (convergence, sensitivity, DOI)
    ├── Shared
    │   ├── colormaps.py    # Geophysics palettes (auto-registered on import)
    │   └── _utils.py       # Internal plotting utilities (private)
    └── Physics-specific (import explicitly)
        ├── seismic.py      # plot_velocity_model, plot_gather, plot_section
        ├── em.py           # plot_pseudosection
        ├── flow.py         # plot_reservoir_state, plot_well_rates
        └── geomodel.py     # plot_geotable_2d

Colormaps auto-registered on import (idempotent):
    ``geo_seismic``, ``geo_velocity``, ``geo_resistivity``,
    ``geo_density``, ``geo_porosity``.

Quick Start:
    >>> from geobrain.vis import plot_field_2d, plot_convergence
    >>> ax = plot_field_2d(velocity_model, dx=10, dz=10, label='Vp (m/s)')
    >>> ax = plot_convergence(loss_history)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

# matplotlib is an optional dependency (the ``vis`` extra). A bare
# ``ModuleNotFoundError: No module named 'matplotlib'`` from a submodule
# import is unactionable, so probe it here and re-raise with the install
# hint (original exception preserved as the cause).
try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError as _exc:
    raise ModuleNotFoundError(
        "geobrain.vis requires matplotlib, which is not installed. "
        "Install the visualization extra with `pip install geobrain[vis]` "
        "(or `pip install matplotlib`).",
        name="matplotlib",
    ) from _exc

# =========================================================================
# Colormaps (auto-registered)
# =========================================================================
from geobrain.vis.colormaps import GEO_COLORMAPS, get_colormap, register_colormaps

# =========================================================================
# Inversion Diagnostics
# =========================================================================
from geobrain.vis.diagnostics import (
    plot_comparison,
    plot_convergence,
    plot_difference,
    plot_doi,
    plot_model_evolution,
    plot_sensitivity,
)

# =========================================================================
# Spatial: Scalar Fields
# =========================================================================
from geobrain.vis.field import (
    plot_field_2d,
    plot_field_polygon,
    plot_field_scatter,
    plot_field_slice,
    plot_field_tripcolor,
)

# =========================================================================
# Spatial: Plan-View Maps
# =========================================================================
from geobrain.vis.map import plot_anomaly_map, plot_station_map

# =========================================================================
# Spatial: Mesh Geometry
# =========================================================================
from geobrain.vis.mesh import (
    plot_mesh_2d,
    plot_mesh_quadtree,
    plot_mesh_triangles,
)

# =========================================================================
# Spatial: 3-D Slice Viewer (matplotlib orthogonal panes)
# =========================================================================
from geobrain.vis.slicer import Slicer, plot_3d_slicer

# =========================================================================
# Interactive 3-D scene (PyVista / Plotly dual backend): lazy: the heavy
# pyvista / plotly imports happen only when a Scene3D is constructed.
# =========================================================================
from geobrain.vis.scene3d import (
    Scene3D,
    set_jupyter_backend,
    view_geomodel,
    view_isosurface,
    view_octree,
    view_points,
    view_reservoir,
    view_slices,
    view_survey,
    view_timelapse,
    view_volume,
)

register_colormaps()

__all__ = [
    # Spatial: scalar fields
    "plot_field_2d",
    "plot_field_scatter",
    "plot_field_slice",
    "plot_field_tripcolor",
    "plot_field_polygon",
    # Spatial: mesh geometry
    "plot_mesh_2d",
    "plot_mesh_quadtree",
    "plot_mesh_triangles",
    # Spatial: plan-view maps
    "plot_anomaly_map",
    "plot_station_map",
    # Spatial: 3-D slices (matplotlib)
    "Slicer",
    "plot_3d_slicer",
    # Interactive 3-D scene (PyVista / Plotly)
    "Scene3D",
    "view_volume",
    "view_slices",
    "view_isosurface",
    "view_points",
    "view_octree",
    "view_reservoir",
    "view_survey",
    "view_geomodel",
    "view_timelapse",
    "set_jupyter_backend",
    # Inversion diagnostics
    "plot_convergence",
    "plot_sensitivity",
    "plot_doi",
    "plot_comparison",
    "plot_difference",
    "plot_model_evolution",
    # Colormaps
    "register_colormaps",
    "get_colormap",
    "GEO_COLORMAPS",
]
