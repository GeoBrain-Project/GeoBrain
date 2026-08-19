"""2D/3D scalar field visualization.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection

from geobrain.core.errors import GeoBrainError

from ._utils import (
    _require_2d,
    add_colorbar,
    ensure_ax,
    panel_edges,
    symmetric_clim,
    to_numpy,
)


def _auto_cmap_for(data, user_cmap):
    """
    Return a sensible cmap given the user choice and the data sign.

    If the user didn't override the default ``'viridis'`` but the data
    straddles zero, return a divergent colormap so the zero line reads
    correctly. Silent when the user explicitly picks a cmap.
    """
    if user_cmap != "viridis":
        return user_cmap
    try:
        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
    except (TypeError, ValueError):
        return user_cmap
    if dmin < 0 < dmax:
        return "RdBu_r"
    return user_cmap


def plot_field_2d(
    data,
    dx=1,
    dz=1,
    ox=0,
    oz=0,
    cmap="viridis",
    vmin=None,
    vmax=None,
    label=None,
    title=None,
    xlabel="X (m)",
    ylabel="Z (m)",
    invert_yaxis=True,
    ax=None,
    mesh=None,
    **kwargs,
):
    """
    2D scalar field via pcolormesh.

    Args:
        dx / dz: cell size along each axis. A single number is a uniform
            grid; a sequence is one width per cell, or the node
            positions themselves.
        mesh: take the geometry from a mesh instead. Required for a
            graded grid, whose cell boundaries a single ``dx`` / ``dz``
            cannot place: the widths, not their average, decide where
            every boundary below the first grading change lands.
        invert_yaxis: When True (default), flip the y-axis so depth
            increases downward, matching the seismic and geophysical
            convention. Pass ``False`` for plan views where ``z`` is
            actually a horizontal coordinate.
    """# explicit 2-D check: previously a 1-D input
    # crashed on ``nz, nx = data.shape`` with the unhelpful "not
    # enough values to unpack" error.
    data = _require_2d("data", to_numpy(data))
    fig, ax = ensure_ax(ax)
    x, z = panel_edges(data.shape, dx, dz, ox, oz, mesh)
    resolved_cmap = _auto_cmap_for(data, cmap)
    # When the data straddles zero and we auto-upgraded to a divergent
    # colormap, force symmetric vmin/vmax (unless the caller pinned
    # them) so the white midpoint lands on value 0: mirroring
    # ``plot_anomaly_map``. Otherwise matplotlib normalises over the
    # asymmetric [dmin, dmax] and a residual field paints 0 off-white.
    if resolved_cmap != cmap and vmin is None and vmax is None:
        try:
            vmin, vmax = symmetric_clim(data)
        except (ValueError, RuntimeError):
            # symmetric_clim can't bracket degenerate values
            # (all-NaN/empty) → fall back to matplotlib autoscaling.
            pass
    im = ax.pcolormesh(
        x, z, data, cmap=resolved_cmap, vmin=vmin, vmax=vmax, shading="flat", **kwargs
    )
    if label:
        add_colorbar(im, ax, label=label)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    if invert_yaxis and not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax


def plot_field_scatter(
    x, z, values, cmap="viridis", s=20, vmin=None, vmax=None, label=None, ax=None, **kwargs
):
    """Scatter plot of scalar values at arbitrary (x, z) locations.

    Args:
        x / z: point coordinates.
        values: per-point values mapped to colour.
        cmap: colormap.
        s: marker size.
        vmin / vmax: colour limits.
        label: colorbar label.
        ax: target axes (new figure when ``None``).
        **kwargs: forwarded to ``scatter``.
    """
    x, z, values = to_numpy(x), to_numpy(z), to_numpy(values)
    fig, ax = ensure_ax(ax)
    im = ax.scatter(x, z, c=values, cmap=cmap, s=s, vmin=vmin, vmax=vmax, **kwargs)
    if label:
        add_colorbar(im, ax, label=label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    return ax


def plot_field_slice(
    data_3d,
    axis="y",
    index=None,
    dx=1,
    dy=1,
    dz=1,
    ox=0,
    oy=0,
    oz=0,
    cmap="viridis",
    label=None,
    ax=None,
    **kwargs,
):
    """
    Extract and plot a 2D slice from a 3D volume.

    Raises:
        ValueError: If axis not in ('x', 'y', 'z').

    Args:
        data_3d: 3-D field to slice.
        axis: slicing axis (``'x'``/``'y'``/``'z'``).
        index: slice index (midplane when ``None``).
        dx / dy / dz, ox / oy / oz: cell sizes and origins per axis.
        cmap: colormap.
        label: colorbar label.
        ax: target axes (new figure when ``None``).
        **kwargs: forwarded to ``imshow``.
    """
    data_3d = to_numpy(data_3d)
    nz, nx, ny = data_3d.shape

    if axis == "y":
        # Constant-Y slice (X-Z). Y is axis-2, so hold it there.
        idx = index if index is not None else ny // 2
        slc = data_3d[:, :, idx]  # (nz, nx)
        return plot_field_2d(
            slc, dx=dx, dz=dz, ox=ox, oz=oz, cmap=cmap, label=label, ax=ax, **kwargs
        )
    elif axis == "x":
        # Constant-X slice (Y-Z). X is axis-1, so hold it there.
        idx = index if index is not None else nx // 2
        slc = data_3d[:, idx, :]  # (nz, ny), Y-Z slice
        return plot_field_2d(
            slc,
            dx=dy,
            dz=dz,
            ox=oy,
            oz=oz,
            xlabel="Y (m)",
            ylabel="Z (m)",
            cmap=cmap,
            label=label,
            ax=ax,
            **kwargs,
        )
    elif axis == "z":
        idx = index if index is not None else nz // 2
        # Plan view: data_3d[idx] is (nx, ny). Transpose to (ny, nx) so
        # plot_field_2d draws X (axis-1) horizontal and Y (axis-2) vertical.
        slc = data_3d[idx, :, :].T  # (nx, ny) -> (ny, nx), X-Y plan view
        # Plan views have Y as a horizontal coord; do NOT flip the axis.
        return plot_field_2d(
            slc,
            dx=dx,
            dz=dy,
            ox=ox,
            oz=oy,
            xlabel="X (m)",
            ylabel="Y (m)",
            invert_yaxis=False,
            cmap=cmap,
            label=label,
            ax=ax,
            **kwargs,
        )
    else:
        raise GeoBrainError(f"Unknown axis '{axis}'. Must be 'x', 'y', or 'z'.")


def plot_field_tripcolor(
    vertices, cells, values, cmap="viridis", label=None, invert_yaxis=True, ax=None, **kwargs
):
    """
    Scalar field on triangular mesh via tripcolor.

    Args:
        invert_yaxis: When True (default), flip the y-axis so depth
            increases downward, matches :func:`plot_field_2d` and the
            seismic / geophysical convention. Pass ``False`` for plan
            views where ``z`` is actually a horizontal coordinate.
    """
    vertices, cells = to_numpy(vertices), to_numpy(cells)
    values = to_numpy(values)
    fig, ax = ensure_ax(ax)
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], cells)
    im = ax.tripcolor(triang, values, cmap=cmap, **kwargs)
    if label:
        add_colorbar(im, ax, label=label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    if invert_yaxis and not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax


def plot_field_polygon(
    vertices, cells, values=None, cmap="viridis", invert_yaxis=True, ax=None, **kwargs
):
    """
    Scalar field on a polygonal unstructured mesh via PolyCollection.

    The polygon counterpart of :func:`plot_field_tripcolor`, same role
    (color cells by a scalar) but supports arbitrary N-gons instead of
    being triangle-only. Pass ``values=None`` to fall back to a single
    facecolor (useful for previewing mesh geometry without scalar data).

    Args:
        invert_yaxis: When True (default), flip the y-axis so depth
            increases downward, matches :func:`plot_field_2d` and the
            seismic / geophysical convention. Pass ``False`` for plan
            views where ``z`` is actually a horizontal coordinate.
    """
    vertices, cells = to_numpy(vertices), to_numpy(cells)
    fig, ax = ensure_ax(ax)

    polys = [vertices[cell] for cell in cells]
    pc = PolyCollection(polys, edgecolor="k", linewidth=0.5, **kwargs)

    if values is not None:
        values = to_numpy(values)
        pc.set_array(values.astype(float))
        pc.set_cmap(cmap)
        ax.add_collection(pc)
        add_colorbar(pc, ax)
    else:
        pc.set_facecolor("lightblue")
        ax.add_collection(pc)

    ax.autoscale_view()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    if invert_yaxis and not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax
