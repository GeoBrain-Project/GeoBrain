"""
Geomodel (geostats) visualization helpers for geobrain.vis.

Promoted from the inline ``_pixel_map`` pattern that appeared verbatim
in the geomodel algorithm examples (kriging, Gaussian simulation, MPS),
which each define the same docstring ("Inline ``geostats.viz.pixel_map``
-- pcolormesh on a 2-D slice") and the same body: read a column from a
``GeoFrame``, reshape to ``(nx, ny, nz)`` via :meth:`GeoFrame.to_grid_array`,
take the first z-slice, transpose to ``(ny, nx)``, and ``pcolormesh`` it
on cell-edge coordinates built from the :class:`GeoGrid` GSLIB
metadata (``xmn`` / ``ymn`` / ``xsiz`` / ``ysiz``).

This module exposes a single helper, :func:`plot_geotable_2d`, that
fuses those three copies into one. The library version differs from
the inline copies in three small ways:

- Returns the matplotlib :class:`~matplotlib.axes.Axes` (callers can
  always reach the ``QuadMesh`` via ``ax.collections[-1]``); that
  matches the rest of :mod:`geobrain.vis` (every public helper returns
  ``Axes``).
- Adds an optional ``label`` parameter so callers attaching their own
  colorbar can label it via this helper rather than wiring
  ``plt.colorbar`` themselves.
- Does **not** auto-attach a colorbar (two of the three inline copies
  did, the 07 MPS version did not, leaving it off matches the lowest
  common denominator and avoids surprising layouts inside multi-panel
  figures).

If :meth:`GeoFrame.to_grid_array` ever stops returning a 3-D array
(``GeoFrame.to_grid_array`` requires a ``GeoGrid`` geometry), a tiny fallback
reshape path is provided.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes

from ._utils import add_colorbar, ensure_ax

if TYPE_CHECKING:
    from geobrain.geomodel import GeoFrame, GeoGrid


# ---------------------------------------------------------------------------
# plot_geotable_2d
# ---------------------------------------------------------------------------


def plot_geotable_2d(
    geotable: "GeoFrame",
    grid: "GeoGrid",
    column: str,
    *,
    cmap: str = "viridis",
    label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    colorbar: bool = False,
    figsize: tuple[float, float] = (6, 5),
    **kwargs: Any,
) -> Axes:
    """
    Render a ``GeoFrame`` column as a 2-D map on a :class:`GeoGrid`.

    Equivalent to the inline ``_pixel_map`` helper recurring across the
    geomodel algorithm examples. Pulls ``column``
    out of ``geotable`` via :meth:`GeoFrame.to_grid_array` (3-D
    ``(nx, ny, nz)``), takes ``[:, :, 0]``, transposes to ``(ny, nx)``,
    and draws it with :meth:`Axes.pcolormesh` using cell-edge
    coordinates derived from the grid's GSLIB metadata.

    Args:
        geotable: A :class:`geobrain.geomodel.GeoFrame` whose geometry
            is a :class:`GeoGrid` (the only geometry type
            ``to_grid_array`` supports).
        grid: The same :class:`GeoGrid` used to back ``geotable``.
            Passed explicitly so this helper can also be called with a
            free-standing flat array via the fallback path; in normal
            usage ``grid is geotable.geometry``.
        column: Name of the column to render.
        cmap: Matplotlib colormap name (default ``"viridis"``, the
            kriging / SGS examples' choice; pass ``"gray_r"`` for the
            categorical MPS examples).
        label: Optional colorbar label; only consulted when
            ``colorbar=True``.
        vmin: Lower colorbar bound. ``None`` lets matplotlib pick.
        vmax: Upper colorbar bound.
        title: Optional axes title.
        ax: Existing :class:`~matplotlib.axes.Axes` to draw into;
            created when ``None``.
        colorbar: When ``True`` attach a colorbar (the 02 / 05 inline
            convention). The MPS example (07) drew its own colorbar
            outside the helper, so we leave this off by default.
        figsize: Figure size when ``ax`` is ``None``.
        **kwargs: Forwarded to :meth:`Axes.pcolormesh`.

    Returns:
        The matplotlib :class:`~matplotlib.axes.Axes` the map was drawn
        into. The ``QuadMesh`` is the last entry of ``ax.collections``.

    Raises:
        ValueError: If ``column`` is missing from ``geotable``, or if
            the shape extracted from ``to_grid_array`` is not 3-D.
    """
    arr2d, xs, ys = _extract_slice(geotable, grid, column)

    _, ax = ensure_ax(ax, figsize=figsize)
    mesh = ax.pcolormesh(xs, ys, arr2d, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if colorbar:
        add_colorbar(mesh, ax, label=label)
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_slice(
    geotable: "GeoFrame",
    grid: "GeoGrid",
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pull the first z-slice out of ``geotable[column]`` and build edges.

    Centralises the ``to_grid_array`` call + transpose + edge-coord
    construction so :func:`plot_geotable_2d` and any future variants
    share the path. Uses :meth:`GeoFrame.to_grid_array` when available
    (the canonical API, :meth:`GeoFrame.to_grid_array`)
    and falls back to a flat-property reshape otherwise.
    """
    if hasattr(geotable, "to_grid_array"):
        arr3d = np.asarray(geotable.to_grid_array(column))
    else:
        # Fallback: dig the raw 1-D buffer out and reshape with GSLIB
        # F-order. ``_properties`` is the canonical storage
        # (see GeoFrame._coerce_property).
        if hasattr(geotable, "_properties") and column in geotable._properties:
            flat = np.asarray(geotable._properties[column])
        elif hasattr(geotable, column):
            flat = np.asarray(getattr(geotable, column))
        else:
            raise GeoBrainError(
                f"plot_geotable_2d: column {column!r} not found on geotable; "
                "geotable lacks both to_grid_array and a matching attribute."
            )
        arr3d = flat.reshape((grid.nx, grid.ny, grid.nz), order="F")

    if arr3d.ndim != 3:
        raise GeoBrainError(
            f"plot_geotable_2d: expected (nx, ny, nz) grid array, "
            f"got shape {tuple(arr3d.shape)}."
        )

    arr2d = arr3d[:, :, 0].T  # -> (ny, nx) for pcolormesh
    xs = np.linspace(grid.xmn, grid.xmn + grid.nx * grid.xsiz, grid.nx + 1)
    ys = np.linspace(grid.ymn, grid.ymn + grid.ny * grid.ysiz, grid.ny + 1)
    return arr2d, xs, ys


__all__ = [
    "plot_geotable_2d",
]
