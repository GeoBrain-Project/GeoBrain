"""
EM-specific visualization helpers for geobrain.vis.

The library form of the inline ``plot_pseudosection`` helper the vis
gallery example uses for its DC/IP apparent-resistivity panel.

All helpers:

- Accept ``torch.Tensor`` or ``numpy.ndarray`` via
  :func:`geobrain.vis._utils.to_numpy` (detach/cpu handled).
- Return a matplotlib :class:`~matplotlib.axes.Axes` for downstream
  customization (titles, overlays, savefig).
- Apply EM-conventional axis orientation: pseudo-depth increases
  downward, x increases rightward.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

import numpy as np
from matplotlib.axes import Axes

from ._utils import _safe_log_norm, add_colorbar, ensure_ax, to_numpy


# ---------------------------------------------------------------------------
# plot_pseudosection
# ---------------------------------------------------------------------------


def plot_pseudosection(
    x,
    pseudo_depth,
    values,
    *,
    log_scale: bool = True,
    cmap: str = "viridis",
    label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """
    DC/IP-style pseudo-section scatter plot.

    Used to display apparent
    resistivity (or IP chargeability) at electrode midpoints versus
    a survey-geometry-derived pseudo-depth. The library version
    defaults to ``cmap="viridis"`` instead of the inline
    ``"Spectral_r"`` to match other :mod:`geobrain.vis` helpers; pass
    ``cmap="geo_resistivity"`` or ``"Spectral_r"`` to recover the
    inline look.

    Args:
        x: 1-D array of midpoint x-coordinates (length ``N``).
        pseudo_depth: 1-D array of pseudo-depths (length ``N``).
            Drawn on an inverted y-axis (depth increases downward).
        values: 1-D array of plotted values (e.g. apparent resistivity
            in Ω·m), length ``N``.
        log_scale: When ``True`` (the inline default), apply a
            ``LogNorm`` to the colorbar via
            :func:`geobrain.vis._utils._safe_log_norm`: degrades to
            linear with a warning when ``values`` has no
            strictly-positive entries.
        cmap: Matplotlib colormap.
        label: Optional colorbar label (e.g. ``"ρ_a [Ω·m]"``). When
            ``None`` the colorbar is skipped (matches the inline
            behavior).
        vmin, vmax: Optional linear color limits. Ignored when
            ``log_scale=True`` and a valid log norm is produced.
        title: Optional axes title.
        ax: Existing axes to draw into; created when ``None``.
        **kwargs: Forwarded to :meth:`~matplotlib.axes.Axes.scatter`
            (e.g. ``s``, ``edgecolors``, ``linewidths``).

    Returns:
        The matplotlib ``Axes`` the pseudo-section was drawn into.
    """
    x_arr = np.asarray(to_numpy(x)).reshape(-1)
    z_arr = np.asarray(to_numpy(pseudo_depth)).reshape(-1)
    v_arr = np.asarray(to_numpy(values)).reshape(-1)

    if not (x_arr.shape[0] == z_arr.shape[0] == v_arr.shape[0]):
        raise GeoBrainError(
            f"x, pseudo_depth, and values must have equal length; got "
            f"{x_arr.shape[0]}, {z_arr.shape[0]}, {v_arr.shape[0]}."
        )

    fig, ax = ensure_ax(ax)
    norm = _safe_log_norm(v_arr, log_scale)

    scatter_kwargs = {"s": 30, "edgecolors": "k", "linewidths": 0.3}
    scatter_kwargs.update(kwargs)

    # When a log norm is active, vmin/vmax must not be passed alongside.
    if norm is not None:
        im = ax.scatter(x_arr, z_arr, c=v_arr, cmap=cmap, norm=norm, **scatter_kwargs)
    else:
        im = ax.scatter(
            x_arr, z_arr, c=v_arr, cmap=cmap, vmin=vmin, vmax=vmax, **scatter_kwargs
        )

    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    if label:
        add_colorbar(im, ax, label=label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Pseudo-depth (m)")
    if title:
        ax.set_title(title)
    return ax


__all__ = [
    "plot_pseudosection",
]
