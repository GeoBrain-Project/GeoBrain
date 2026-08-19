"""Plan-view contour/pcolormesh maps for anomalies and stations.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import warnings

from geobrain.core.errors import GeoBrainError


from ._utils import _require_2d, add_colorbar, ensure_ax, symmetric_clim, to_numpy


def plot_anomaly_map(
    x, y, values, method="contourf", n_levels=20, cmap="RdBu_r", label=None, ax=None, **kwargs
):
    """
    Plan-view anomaly map (gravity, magnetic, etc.).

    Parameters
    ----------
    x, y : array-like
        1-D coordinate vectors of length ``nx`` and ``ny`` respectively.
    values : array-like
        2-D field with shape ``(len(y), len(x)) == (ny, nx)``, rows
        index ``y``, columns index ``x``. This row-major
        ``(ny, nx)`` convention is what ``contourf`` / ``pcolormesh``
        expect; a transposed ``(nx, ny)`` array is rejected with a
        named :class:`GeoBrainError` instead of a deep Matplotlib
        ``TypeError``.
    method : {"contourf", "pcolormesh"}
        Rendering backend.
    n_levels : int
        Number of filled contour levels. **Only applies to
        ``method="contourf"``**; it is ignored by ``pcolormesh``
        (which has no discrete levels), and passing it together with
        ``method="pcolormesh"`` emits a warning.

    Uses a symmetric vmin/vmax around zero when the caller hasn't
    supplied them so the white midpoint of a divergent colormap always
    lands on zero. Pass ``vmin=``/``vmax=`` explicitly to override.
    """
    # anomaly maps require a 2-D ``values`` array so
    # ``contourf`` / ``pcolormesh`` accept it. Previously a 1-D input
    # raised matplotlib's "Input z must be 2D".
    x, y = to_numpy(x), to_numpy(y)
    values = _require_2d("values", to_numpy(values))

    # Enforce the (ny, nx) == (len(y), len(x)) value-shape contract up
    # front so a transposed / wrong-shape array fails with a named,
    # library-level error rather than a deep Matplotlib ``TypeError``
    # ("Length of x ... must match number of columns in z" /
    # "Dimensions of C ... should be one smaller than X and Y").
    expected = (len(y), len(x))
    if values.shape != expected:
        raise GeoBrainError(
            f"values must have shape (len(y), len(x)) = {expected} "
            f"(ny, nx); got {tuple(values.shape)}. Rows index y, "
            f"columns index x; pass values.T if your array is "
            f"(nx, ny)."
        )

    # ``n_levels`` only controls ``contourf``'s discrete levels; it is
    # silently irrelevant to ``pcolormesh``. Warn rather than mislead a
    # caller who set it expecting it to take effect.
    if method == "pcolormesh" and n_levels != 20:
        warnings.warn(
            "n_levels only applies to method='contourf' and is ignored "
            "by method='pcolormesh'.",
            UserWarning,
            stacklevel=2,
        )

    fig, ax = ensure_ax(ax)

    # Default to symmetric clim for divergent colormaps so zero maps to
    # the midpoint colour.
    if "vmin" not in kwargs and "vmax" not in kwargs:
        try:
            vmin, vmax = symmetric_clim(values)
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax
        except (ValueError, RuntimeError):
            # symmetric_clim can't bracket degenerate values (all-NaN/empty)
            # → fall back to matplotlib's default autoscaling.
            pass

    if method == "contourf":
        im = ax.contourf(x, y, values, levels=n_levels, cmap=cmap, **kwargs)
    elif method == "pcolormesh":
        im = ax.pcolormesh(x, y, values, cmap=cmap, shading="auto", **kwargs)
    else:
        raise GeoBrainError(f"Unknown method '{method}'. Must be 'contourf' or 'pcolormesh'.")

    if label:
        add_colorbar(im, ax, label=label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    return ax


def plot_station_map(x, y, labels=None, ax=None, **kwargs):
    """Station/electrode location map.

    Args:
        x / y: station coordinates.
        labels: optional per-station text labels.
        ax: target axes (new figure when ``None``).
        **kwargs: forwarded to ``scatter``.
    """
    x, y = to_numpy(x), to_numpy(y)
    fig, ax = ensure_ax(ax)
    defaults = dict(marker="v", color="k", s=40)
    defaults.update(kwargs)
    ax.scatter(x, y, **defaults)

    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (x[i], y[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    return ax
