"""
Seismic-specific visualization helpers for geobrain.vis.

The library form of the inline plotting patterns the seismic and vis
gallery examples share.

All helpers:

- Accept ``torch.Tensor`` or ``numpy.ndarray`` via
  :func:`geobrain.vis._utils.to_numpy` (detach/cpu handled).
- Return a matplotlib :class:`~matplotlib.axes.Axes` for downstream
  customization (titles, overlays, savefig).
- Apply seismic-conventional axis orientation: depth / time increases
  downward, distance / receiver increases rightward.
- Use ``aspect="auto"`` and ``interpolation="bilinear"`` to match the
  inline style of the seismic gallery examples.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from geobrain.core.errors import GeoBrainError

from ._utils import (
    _require_2d,
    add_colorbar,
    ensure_ax,
    symmetric_clim,
    to_numpy,
)


# ---------------------------------------------------------------------------
# Internal helper: the imshow scaffold shared by the three depth-section
# plotters (velocity model / density section / wavefield snapshot). They all
# draw a 2-D array with seismic orientation (depth/Z down) on a metric extent,
# differing only in colour limits, axis labels, and whether a colorbar is
# attached. ``plot_gather`` deliberately stays separate: it uses a
# sample-position (pixel-centre) extent so trace ``i`` lands at ``i * dx``,
# and it normalizes per trace.
# ---------------------------------------------------------------------------


def _imshow_panel(
    data: np.ndarray,
    *,
    dx,
    dz,
    ox: float = 0.0,
    oz: float = 0.0,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    xlabel: str,
    ylabel: str,
    title: str | None,
    ax: Axes | None,
    **kwargs,
):
    """Draw ``data`` with seismic axes; return ``(im, ax)``.

    Centralises the ``extent`` math + ``imshow(aspect="auto",
    interpolation="bilinear", ...)`` call + axis labels + optional title
    that ``plot_velocity_model`` / ``plot_section`` / ``plot_wavefield``
    otherwise each repeat verbatim. Colour limits and the colorbar are
    left to the caller (they differ per plotter).
    """
    fig, ax = ensure_ax(ax)
    nz, nx = data.shape
    extent = [
        float(ox),
        float(ox) + nx * float(dx),
        float(oz) + nz * float(dz),
        float(oz),
    ]
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        **kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return im, ax


# ---------------------------------------------------------------------------
# plot_velocity_model
# ---------------------------------------------------------------------------


def plot_velocity_model(
    vp,
    dx,
    dz,
    *,
    cmap: str = "geo_velocity",
    label: str = "Vp (m/s)",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Display a 2D velocity (or generic model) map in seismic axes.

    The library form of the ``_imshow`` helper repeated across the
    seismic gallery examples (including their FWI and depth-slice
    variants).

    Args:
        vp: 2-D velocity model of shape ``(nz, nx)``. Accepts numpy
            array or torch tensor.
        dx, dz: Grid spacing in metres (used to set the imshow extent).
        cmap: Matplotlib colormap. ``viridis`` matches the inline
            default; pass another cmap for residuals / differences.
        label: Colorbar label. Defaults to ``"Vp (m/s)"``; override
            for density, Vs, etc.
        vmin, vmax: Optional fixed color limits (matches the inline
            ``vmin``/``vmax`` arguments threaded through ``_imshow``).
        title: Optional axes title.
        ax: Existing axes to draw into; created when ``None``.
        **kwargs: Forwarded to :meth:`~matplotlib.axes.Axes.imshow`.

    Returns:
        The matplotlib ``Axes`` the model was drawn into.
    """
    data = _require_2d("vp", to_numpy(vp))
    im, ax = _imshow_panel(
        data,
        dx=dx,
        dz=dz,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xlabel="Distance (m)",
        ylabel="Depth (m)",
        title=title,
        ax=ax,
        **kwargs,
    )
    add_colorbar(im, ax, label=label)
    return ax


# ---------------------------------------------------------------------------
# plot_gather
# ---------------------------------------------------------------------------


def plot_gather(
    traces,
    dt,
    dx,
    *,
    time_axis: int = 0,
    normalize: bool = True,
    clip_pct: float = 99,
    cmap: str = "gray",
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Display a shot/receiver gather (time x offset).

    The library form of the ``_gather`` helper and inline shot-gather
    imshow patterns repeated across the seismic gallery examples.

    Args:
        traces: 2-D gather laid out as ``(nt, n_traces)`` by default,
            the wave operator's per-shot output convention (time on
            axis 0, receivers on axis 1, e.g. ``rcv[i_shot]`` from an
            operator that returns ``(..., nt, n_rcv)``). If your gather
            instead stores receivers on axis 0 and time on axis 1
            (``(n_traces, nt)``), pass ``time_axis=1``. The orientation
            is taken *only* from ``time_axis``; the shape is never
            sniffed, so a gather with fewer time samples than
            receivers (``nt < n_traces``) still plots time on the
            y-axis.
        dt: Time sample interval (seconds).
        dx: Trace spacing in metres. The extent places sample column
            ``i`` at the pixel-centre ``i * dx`` so trace ``i`` lands at
            ``i * dx`` on the x-axis (Offset).
        time_axis: Which axis of ``traces`` is time. ``0`` (default)
            means ``(nt, n_traces)``; ``1`` means ``(n_traces, nt)`` and
            the array is transposed internally so time becomes the rows.
        normalize: When ``True`` (default), normalize each trace by its
            peak absolute value before display so weak / strong traces
            share the same dynamic range.
        clip_pct: Percentile used for symmetric color clipping (default
            99). Matches the inline ``np.percentile(|data|, 99) * 1.2``
            pattern (slightly relaxed: no 1.2x bump).
        cmap: Matplotlib colormap. ``gray`` is the seismic convention
            for shot gathers; pass ``viridis``/``seismic`` to match
            the inline examples directly.
        title: Optional axes title.
        ax: Existing axes to draw into; created when ``None``.
        **kwargs: Forwarded to ``imshow``.

    Returns:
        The matplotlib ``Axes`` the gather was drawn into.
    """
    data = _require_2d("traces", to_numpy(traces)).astype(float, copy=True)
    if time_axis not in (0, 1):
        raise GeoBrainError(
            f"time_axis must be 0 (time on rows, (nt, n_traces)) or 1 "
            f"(time on cols, (n_traces, nt)); got {time_axis!r}."
        )
    # Move the declared time axis to the rows so imshow draws time
    # downward. No shape sniffing: the orientation is the caller's
    # contract, not a guess from the aspect ratio.
    if time_axis == 1:
        data = data.T
    nt, n_traces = data.shape

    if normalize:
        peak = np.max(np.abs(data), axis=0, keepdims=True)
        peak[peak == 0] = 1.0
        data = data / peak

    finite = data[np.isfinite(data)]
    if finite.size == 0 or np.max(np.abs(finite)) == 0:
        clip = 1.0
    else:
        clip = float(np.percentile(np.abs(finite), clip_pct))
        if clip == 0.0:
            clip = float(np.max(np.abs(finite)) + 1e-12)

    fig, ax = ensure_ax(ax)
    # Sample-position extent: under imshow's edge-extent / pixel-centre
    # mapping this puts column i at x = i * dx (trace i at i * dx) and
    # row j at t = j * dt, with time increasing downward.
    extent = [
        -0.5 * float(dx),
        (n_traces - 0.5) * float(dx),
        (nt - 0.5) * float(dt),
        -0.5 * float(dt),
    ]
    ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=-clip,
        vmax=clip,
        extent=extent,
        interpolation="bilinear",
        **kwargs,
    )
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# plot_section
# ---------------------------------------------------------------------------


def plot_section(
    data,
    dx,
    dz,
    ox: float = 0.0,
    oz: float = 0.0,
    *,
    clip_pct: float = 99,
    cmap: str = "gray",
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Seismic density section with symmetric percentile clipping.

    The library form of the vis gallery's inline ``plot_section``.
    Where the inline version
    used ``(dt, dx)`` and labelled the y-axis ``"Time (s)"``, this
    library version uses ``(dz, dx)`` and labels it ``"Depth (m)"``
    so the helper covers depth-domain density sections too. Callers
    plotting time-domain sections can override via the returned
    ``Axes``.

    Args:
        data: 2-D section, shape ``(nz, nx)``.
        dx, dz: Grid spacing.
        ox, oz: Origin shifts for the extent (default 0).
        clip_pct: Percentile used for symmetric clipping (default 99).
        cmap: Matplotlib colormap; ``gray`` is the seismic convention.
        title: Optional axes title.
        ax: Existing axes to draw into; created when ``None``.
        **kwargs: Forwarded to ``imshow``.

    Returns:
        The matplotlib ``Axes`` the section was drawn into.
    """
    data = _require_2d("data", to_numpy(data))
    vmin, vmax = symmetric_clim(data, percentile=clip_pct)
    _, ax = _imshow_panel(
        data,
        dx=dx,
        dz=dz,
        ox=ox,
        oz=oz,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xlabel="Distance (m)",
        ylabel="Depth (m)",
        title=title,
        ax=ax,
        **kwargs,
    )
    return ax


# ---------------------------------------------------------------------------
# plot_wavefield
# ---------------------------------------------------------------------------


def plot_wavefield(
    snapshot,
    dx,
    dz,
    *,
    clip_pct: float = 99,
    cmap: str = "geo_seismic",
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    Single wavefield snapshot with symmetric color limits.

    The library form of the vis gallery's inline ``plot_wavefield``.
    The library version
    defaults to ``cmap="RdBu_r"`` (the standard diverging palette
    used elsewhere in :mod:`geobrain.vis`) instead of the inline
    ``"seismic"``: pass ``cmap="seismic"`` to recover the original.

    Args:
        snapshot: 2-D wavefield snapshot, shape ``(nz, nx)``.
        dx, dz: Grid spacing.
        clip_pct: Percentile used for symmetric clipping (default 99).
        cmap: Matplotlib diverging colormap.
        title: Optional axes title.
        ax: Existing axes to draw into; created when ``None``.
        **kwargs: Forwarded to ``imshow``.

    Returns:
        The matplotlib ``Axes`` the wavefield was drawn into.
    """
    data = _require_2d("snapshot", to_numpy(snapshot))
    vmin, vmax = symmetric_clim(data, percentile=clip_pct)
    _, ax = _imshow_panel(
        data,
        dx=dx,
        dz=dz,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        xlabel="X (m)",
        ylabel="Z (m)",
        title=title,
        ax=ax,
        **kwargs,
    )
    return ax


__all__ = [
    "plot_velocity_model",
    "plot_gather",
    "plot_section",
    "plot_wavefield",
]
