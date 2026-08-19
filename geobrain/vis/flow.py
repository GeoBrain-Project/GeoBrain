"""
Reservoir-flow visualization helpers for geobrain.vis.

The library form of the reservoir-state and well-rate figures the flow
examples build inline (state maps, perforation histories, rate panels).

All helpers:

- Accept ``torch.Tensor`` or ``numpy.ndarray`` via
  :func:`geobrain.vis._utils.to_numpy` (detach/cpu handled).
- Return matplotlib :class:`~matplotlib.axes.Axes` (or a list for
  multi-panel helpers) so callers can attach titles, overlays, savefig.
- Use ``origin="lower"`` for the reservoir maps to match the inline
  example convention (Y/depth-up).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

from typing import Mapping, Sequence

import numpy as np
from matplotlib.axes import Axes

from ._utils import _require_2d, add_colorbar, ensure_ax, to_numpy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_wells(wells):
    """
    Normalize ``wells`` into a list of ``(x_idx, z_idx, label)`` tuples.

    Accepts either:

    - a mapping ``{name: (i, j[, marker])}``, or
    - a sequence of ``(x_idx, z_idx, label)`` / ``(x_idx, z_idx)``.

    Returns an empty list when ``wells`` is ``None``.
    """
    if wells is None:
        return []
    out: list[tuple[float, float, str]] = []
    items: Sequence
    if isinstance(wells, Mapping):
        items = [(name, val) for name, val in wells.items()]
        for name, val in items:
            if len(val) >= 3:
                x_idx, z_idx, marker = val[0], val[1], val[2]
            else:
                x_idx, z_idx = val[0], val[1]
                marker = "w^"
            out.append((float(x_idx), float(z_idx), str(marker), str(name)))
    else:
        for w in wells:
            if len(w) >= 3:
                x_idx, z_idx, label = w[0], w[1], w[2]
            else:
                x_idx, z_idx = w[0], w[1]
                label = ""
            out.append((float(x_idx), float(z_idx), "w^", str(label)))
    return out


def _draw_well_markers(ax: Axes, wells_norm) -> None:
    """
    Draw well markers from a normalized list.

    Each entry is ``(x_idx, z_idx, marker, label)``. The marker string
    follows matplotlib's ``plot`` short format (e.g. ``"w^"`` for a
    white triangle); when callers pass plain ``(i, j, label)`` we use
    ``"w^"`` so the marker is still visible on a colored map.
    """
    for x_idx, z_idx, marker, _label in wells_norm:
        ax.plot(
            x_idx,
            z_idx,
            marker,
            markersize=10,
            markeredgecolor="k",
            markeredgewidth=0.8,
        )


# ---------------------------------------------------------------------------
# plot_reservoir_state
# ---------------------------------------------------------------------------


def plot_reservoir_state(
    pressure,
    sw=None,
    sg=None,
    *,
    wells=None,
    cmap_p: str = "RdYlBu_r",
    cmap_s: str = "Blues",
    cmap_sg: str = "Reds",
    figsize: tuple[float, float] = (12, 4),
    axes: list[Axes] | None = None,
    titles: Sequence[str] | None = None,
    label_p: str = "Pressure",
    origin: str = "lower",
    **kwargs,
) -> list[Axes]:
    """
    Combined pressure + (optional) Sw + (optional) Sg map panel.

    One panel per supplied state field (p / Sw / Sg), with optional well
    markers. The derived ``So`` panel is deliberately dropped; callers can
    compute ``1 - Sw - Sg`` and call :func:`plot_field_2d` for the
    extra panel if needed. The default ``cmap_p`` is ``"RdYlBu_r"``
    (matches both examples). The inline 02 example used ``"YlGnBu"``
    for Sw while 03 used ``"Blues"``; we default to ``"Blues"`` and
    let 02 override via ``cmap_s``.

    Args:
        pressure: 2-D pressure field, shape ``(nz, nx)`` (or
            ``(ny, nx)``: interpretation is purely cosmetic).
        sw: Optional 2-D water saturation field.
        sg: Optional 2-D gas saturation field.
        wells: Optional well-marker overlay. Either a mapping
            ``{name: (i, j[, marker])}`` (the inline convention) or a
            sequence of ``(x_idx, z_idx[, label])`` tuples. Indices are
            in pixel/cell coordinates (the inline examples plot wells
            with ``origin="lower"``, so ``(i, j)`` lands at column ``i``,
            row ``j``).
        cmap_p: Colormap for pressure (default ``"RdYlBu_r"``).
        cmap_s: Colormap for water saturation (default ``"Blues"``).
        cmap_sg: Colormap for gas saturation (default ``"Reds"``).
        figsize: Figure size when ``axes`` is ``None``.
        axes: Existing list of axes to draw into. Must be at least as
            long as the number of panels (1, 2, or 3). Created when
            ``None``.
        titles: Optional list of panel titles; defaults are
            ``["Pressure", "Water Saturation", "Gas Saturation"]``.
        label_p: Colorbar label for pressure.
        origin: Imshow origin (default ``"lower"`` to match the inline
            examples).
        **kwargs: Forwarded to each :meth:`Axes.imshow` call.

    Returns:
        A list of matplotlib ``Axes`` (length 1, 2, or 3 depending on
        which saturation fields were supplied).
    """
    p = _require_2d("pressure", to_numpy(pressure))
    panels: list[tuple[np.ndarray, str, str, str]] = [
        (p, cmap_p, label_p, "Pressure"),
    ]
    if sw is not None:
        sw_arr = _require_2d("sw", to_numpy(sw))
        panels.append((sw_arr, cmap_s, "", "Water Saturation"))
    if sg is not None:
        sg_arr = _require_2d("sg", to_numpy(sg))
        panels.append((sg_arr, cmap_sg, "", "Gas Saturation"))

    n_panels = len(panels)

    if axes is None:
        import matplotlib.pyplot as plt

        fig, ax_list = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            ax_list = [ax_list]
        else:
            ax_list = list(ax_list)
    else:
        if len(axes) < n_panels:
            raise GeoBrainError(
                f"axes must contain at least {n_panels} entries "
                f"(got {len(axes)})."
            )
        ax_list = list(axes[:n_panels])

    wells_norm = _normalize_wells(wells)

    for idx, (data, cmap, cbar_label, default_title) in enumerate(panels):
        ax = ax_list[idx]
        im = ax.imshow(data, cmap=cmap, origin=origin, aspect="auto", **kwargs)
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx])
        else:
            ax.set_title(default_title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        add_colorbar(im, ax, label=cbar_label if cbar_label else None)
        _draw_well_markers(ax, wells_norm)

    return ax_list


# ---------------------------------------------------------------------------
# plot_well_rates
# ---------------------------------------------------------------------------


def plot_well_rates(
    time,
    rates_dict: Mapping[str, "np.ndarray"],
    *,
    ax: Axes | None = None,
    ylabel: str = "Rate (m^3/s)",
    xlabel: str = "Time (Day)",
    title: str | None = None,
    fill: bool = True,
    colors: Sequence[str] | None = None,
    figsize: tuple[float, float] = (8, 4),
    **kwargs,
) -> Axes:
    """
    Per-well production/injection rate time series.

    Draws all supplied per-well rate series on a single axes (rather
    than one panel per well and phase) so 1-3 wells fit
    in one figure (the inline examples used three separate panels with
    separate y-labels, which doesn't generalize when wells are
    arbitrary in count).

    Args:
        time: 1-D time vector (length ``nt``).
        rates_dict: Mapping ``{well_name: rates_array}``. Each array
            must have length ``nt``.
        ax: Existing axes to draw into; created when ``None``.
        ylabel: Y-axis label. Default uses SI metric rate units; pass
            ``"Oil Rate (STB/day)"`` etc. for field-unit displays.
        xlabel: X-axis label.
        title: Optional axes title.
        fill: When ``True`` (the inline default), draw a translucent
            ``fill_between`` below each curve.
        colors: Optional list of color strings (one per well, cycled if
            shorter). Defaults to the inline palette
            ``#2ca02c / #1f77b4 / #d62728`` (oil-green / water-blue /
            inj-red), which matches the 02 example's per-panel choice.
        figsize: Figure size when ``ax`` is ``None``.
        **kwargs: Forwarded to :meth:`Axes.plot`.

    Returns:
        The matplotlib ``Axes`` the curves were drawn into.
    """
    if not rates_dict:
        raise GeoBrainError(
            "rates_dict must contain at least one well; got an empty mapping."
        )

    t = np.asarray(to_numpy(time)).reshape(-1)
    fig, ax = ensure_ax(ax, figsize=figsize)

    default_colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd",
                      "#ff7f0e", "#8c564b", "#e377c2"]
    palette = list(colors) if colors is not None else default_colors

    for idx, (name, rates) in enumerate(rates_dict.items()):
        r = np.asarray(to_numpy(rates)).reshape(-1)
        if r.shape[0] != t.shape[0]:
            raise GeoBrainError(
                f"rates for well {name!r} has length {r.shape[0]} but "
                f"time has length {t.shape[0]}."
            )
        color = palette[idx % len(palette)]
        if fill:
            ax.fill_between(t, r, alpha=0.15, color=color)
        ax.plot(t, r, "-", color=color, linewidth=1.5, label=str(name), **kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    if len(rates_dict) > 1:
        ax.legend()
    return ax


__all__ = [
    "plot_reservoir_state",
    "plot_well_rates",
]
