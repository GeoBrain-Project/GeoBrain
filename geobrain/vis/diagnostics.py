"""
Inversion diagnostics: convergence, sensitivity, depth of investigation,
and multi-panel model comparison.

Cross-physics plots that visualize the *result* of running an inversion
rather than the underlying physical data. Convergence curves, the
diagonal of ``JᵀJ``, depth-of-investigation indices, and side-by-side
model snapshots all share this concern, so they live in ONE module: a
split by plot *shape* (1D line vs 2D map vs multi-panel) would scatter
one workflow's diagnostics across modules and make the menu confusing.

Depth convention: every 2-D map in this module renders depth
*increasing downward* (the GeoBrain house convention, shared with
:func:`plot_field_2d`, :func:`plot_sensitivity`, ``vis/em.py``, and the
seismic sections). ``imshow``-based panels achieve this through an
inverted ``extent``; ``pcolormesh``-based panels (``_multi_panel`` /
:func:`plot_difference`) use increasing edges and call
``ax.invert_yaxis()`` so sibling panels of one workflow agree.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np

from geobrain.vis._utils import (
    _require_non_empty,
    _safe_log_norm,
    add_colorbar,
    edges_are_uniform,
    ensure_ax,
    panel_edges,
    symmetric_clim,
    to_numpy,
)

__all__ = [
    "plot_convergence",
    "plot_sensitivity",
    "plot_doi",
    "plot_comparison",
    "plot_difference",
    "plot_model_evolution",
]


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def plot_convergence(
    loss_history, log_scale=True, xlabel="Iteration", ylabel="Loss", ax=None, **kwargs
):
    """Loss/misfit convergence curve.

    Args:
        loss_history: per-iteration objective values.
        log_scale: log-y axis when True.
        xlabel / ylabel: axis labels.
        ax: target axes (new figure when ``None``).
        **kwargs: forwarded to ``plot``.
    """
    loss = to_numpy(loss_history)
    fig, ax = ensure_ax(ax)
    ax.plot(np.arange(1, len(loss) + 1), loss, **kwargs)
    if log_scale:
        # A signed history (e.g. a log-posterior / ELBO from the bayes
        # samplers) has no strictly-positive finite entry that ``log``
        # can show, leaving 0 visible points (an empty plot). Mirror the
        # ``_safe_log_norm`` guard: stay linear and warn instead.
        if np.any(np.isfinite(loss) & (loss > 0)):
            ax.set_yscale("log")
        else:
            warnings.warn(
                "log_scale=True but loss_history has no strictly-positive "
                "finite entry; falling back to linear scale.",
                UserWarning,
                stacklevel=2,
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Sensitivity / DOI
# ---------------------------------------------------------------------------


def _draw_panel(ax, x, z, data, **kwargs):
    """
    Draw one 2-D panel on the node lines ``x`` and ``z``.

    A uniform grid keeps going through ``imshow``, which is what these
    panels have always used and what their ``aspect="auto"`` behaviour
    depends on. Only a graded grid switches to ``pcolormesh``, which is
    the call that can place unequal cell boundaries at all.
    """
    if edges_are_uniform(x) and edges_are_uniform(z):
        extent = [x[0], x[-1], z[-1], z[0]]
        return ax.imshow(data, aspect="auto", extent=extent, **kwargs)
    image = ax.pcolormesh(x, z, data, shading="flat", **kwargs)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return image


def plot_sensitivity(
    jtj_diag, shape, dx=1.0, dz=1.0, ox=0, oz=0, log_scale=True,
    cmap="YlGnBu", ax=None, mesh=None, **kwargs,
):
    """
    Sensitivity map from J^T J diagonal.

    A J^T J diagonal commonly contains zeros (rows that
    saw no station coverage); the previous implementation used
    ``LogNorm`` directly on those and matplotlib raised
    ``Invalid vmin or vmax``. Routes through :func:`_safe_log_norm`
    which degrades to linear when no positive finite entries exist
    and masks non-positives in the plot otherwise.

    Args:
        jtj_diag: flattened diagonal of ``J^T J``.
        shape: 2-D field shape to reshape into.
        dx / dz: cell size along each axis, one size per cell, or the
            node positions; ox / oz: origin.
        log_scale: plot ``log10`` of the sensitivity when True.
        cmap: colormap. Sensitivity is a non-negative magnitude, so
            the default is light at zero and even in lightness,
            which ``hot`` is not.
        ax: target axes (new figure when ``None``).
        mesh: take the geometry from a mesh, which a graded grid needs.
        **kwargs: forwarded to the drawing call.
    """
    jtj_diag = to_numpy(jtj_diag)
    data = jtj_diag.reshape(shape)
    fig, ax = ensure_ax(ax)
    x, z = panel_edges(shape, dx, dz, ox, oz, mesh)

    norm = _safe_log_norm(data, log_scale)
    if norm is not None:
        data = np.where(data > 0, data, np.nan)

    im = _draw_panel(ax, x, z, data, cmap=cmap, norm=norm, **kwargs)
    add_colorbar(im, ax, label="Sensitivity")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    return ax


def plot_doi(doi_values, shape, dx=1.0, dz=1.0, ox=0, oz=0, threshold=0.3,
             ax=None, mesh=None, **kwargs):
    """Depth of investigation index with threshold contour.

    Args:
        doi_values: flattened depth-of-investigation values.
        shape: 2-D field shape to reshape into.
        dx / dz: cell size along each axis, one size per cell, or the
            node positions; ox / oz: origin.
        threshold: DOI contour level drawn on top.
        ax: target axes (new figure when ``None``).
        mesh: take the geometry from a mesh, which a graded grid needs.
        **kwargs: forwarded to the drawing call.
    """
    doi_values = to_numpy(doi_values)
    data = doi_values.reshape(shape)
    fig, ax = ensure_ax(ax)
    x, z = panel_edges(shape, dx, dz, ox, oz, mesh)

    im = _draw_panel(ax, x, z, data, cmap="RdYlGn_r", vmin=0, vmax=1, **kwargs)
    add_colorbar(im, ax, label="DOI")

    # Contour on cell centres, which the node lines give directly and
    # which stay correct when the cells are not all one size.
    X, Z = np.meshgrid(0.5 * (x[:-1] + x[1:]), 0.5 * (z[:-1] + z[1:]))
    ax.contour(X, Z, data, levels=[threshold], colors="k", linewidths=2, linestyles="--")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    return ax


# ---------------------------------------------------------------------------
# Multi-panel comparison
# ---------------------------------------------------------------------------


def _multi_panel(
    data_list, dx, dz, ox, oz, titles, cmap, shared_clim, vmin, vmax, label,
    figsize, ncols, mesh=None, **kwargs,
):
    """Shared logic for multi-panel 2D field grids."""
    _require_non_empty("data_list", data_list)
    n = len(data_list)
    if ncols is None:
        ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes_arr.flatten().tolist()

    # Whether color limits are common to every panel: either the caller
    # shared them (``shared_clim``) or pinned them explicitly (vmin/vmax).
    explicit_shared = vmin is not None or vmax is not None
    common_clim = shared_clim or explicit_shared

    if shared_clim and not explicit_shared:
        all_data = np.concatenate([to_numpy(d).ravel() for d in data_list])
        # ``min``/``max`` propagate NaN: one NaN cell sets vmin=vmax=nan and
        # blanks every panel. Use NaN-aware reductions (mirroring
        # ``utils.symmetric_clim``'s isfinite filtering) and guard the
        # all-NaN / all-equal case so vmin < vmax.
        finite = all_data[np.isfinite(all_data)]
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
            if vmin == vmax:
                pad = abs(vmin) * 1e-6 if vmin != 0.0 else 1e-6
                vmin, vmax = vmin - pad, vmax + pad

    result_axes = []
    for i in range(n):
        ax_i = axes_flat[i]
        data = to_numpy(data_list[i])
        x, z = panel_edges(data.shape, dx, dz, ox, oz, mesh)
        im = ax_i.pcolormesh(x, z, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", **kwargs)
        if titles and i < len(titles):
            ax_i.set_title(titles[i])
        ax_i.set_aspect("equal")
        # Depth increases downward (house convention): pcolormesh draws
        # with increasing edges, so flip to match the imshow siblings.
        if not ax_i.yaxis_inverted():
            ax_i.invert_yaxis()
        # With a common clim one shared colorbar is faithful; when each
        # panel autoscales independently, a single bar built from the last
        # panel's mappable misrepresents the others: give each its own.
        draw_cbar = (i == n - 1) if common_clim else True
        if label and draw_cbar:
            add_colorbar(im, ax_i, label=label)
        result_axes.append(ax_i)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return result_axes


def plot_comparison(
    data_list, dx=1.0, dz=1.0, ox=0, oz=0, titles=None, cmap="viridis",
    shared_clim=True, vmin=None, vmax=None, label=None, figsize=None, ncols=None,
    mesh=None, **kwargs,
):
    """Side-by-side comparison of N 2D fields. Returns list[Axes].

    Args:
        data_list: sequence of 2-D fields plotted side by side.
        dx / dz: cell size along each axis, one size per cell, or
            the node positions; ox / oz: origin.
        titles: per-panel titles.
        cmap: shared colormap.
        shared_clim / vmin / vmax: colour-limit policy.
        label: colorbar label.
        figsize / ncols: figure layout.
        mesh: take the geometry from a mesh, which a graded grid needs.
        **kwargs: forwarded to the drawing call.
    """
    return _multi_panel(
        data_list, dx, dz, ox, oz, titles, cmap, shared_clim,
        vmin, vmax, label, figsize, ncols, mesh=mesh, **kwargs,
    )


def plot_difference(
    data_a, data_b, dx=1.0, dz=1.0, ox=0, oz=0, cmap="RdBu_r",
    title="Difference", ax=None, mesh=None, **kwargs,
):
    """Difference map: data_a - data_b with symmetric color limits.

    Args:
        data_a / data_b: 2-D fields; the panel shows ``data_a - data_b``.
        dx / dz: cell size along each axis, one size per cell, or
            the node positions; ox / oz: origin.
        cmap: diverging colormap.
        title: panel title.
        ax: target axes (new figure when ``None``).
        mesh: take the geometry from a mesh, which a graded grid needs.
        **kwargs: forwarded to the drawing call.
    """
    a, b = to_numpy(data_a), to_numpy(data_b)
    diff = a - b
    fig, ax = ensure_ax(ax)
    x, z = panel_edges(diff.shape, dx, dz, ox, oz, mesh)
    vmin, vmax = symmetric_clim(diff, percentile=99)
    im = ax.pcolormesh(x, z, diff, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", **kwargs)
    add_colorbar(im, ax, label="Difference")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    # Depth increases downward (house convention), matching the imshow
    # siblings; pcolormesh draws with increasing edges, so flip.
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax


def plot_model_evolution(
    models, dx=1.0, dz=1.0, ox=0, oz=0, epochs=None, ncols=4, cmap="viridis",
    figsize=None, mesh=None, **kwargs,
):
    """Inversion model snapshots at different iterations. Returns list[Axes].

    Args:
        models: sequence of 2-D model snapshots.
        dx / dz: cell size along each axis, one size per cell, or
            the node positions; ox / oz: origin.
        epochs: per-snapshot labels.
        ncols: panels per row.
        cmap: shared colormap.
        figsize: figure size.
        mesh: take the geometry from a mesh, which a graded grid needs.
        **kwargs: forwarded to the drawing call.
    """
    if epochs is not None:
        titles = [f"Epoch {e}" for e in epochs]
    else:
        titles = [f"Step {i}" for i in range(len(models))]
    return _multi_panel(
        models, dx, dz, ox, oz, titles, cmap, True, None, None, None, figsize,
        ncols, mesh=mesh, **kwargs,
    )
