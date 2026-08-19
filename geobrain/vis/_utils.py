"""
Internal visualization utilities. Not part of public API.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def to_numpy(x) -> np.ndarray:
    """
    Convert torch.Tensor or array-like to numpy. Handles detach/cpu.

    Complex tensors / arrays are returned as-is (complex dtype).
    Matplotlib path handlers that need a real-valued input (e.g.
    ``pcolormesh``) will raise; callers plotting MT / EM spectra should
    pre-reduce via ``.real``, ``.imag``, or ``np.abs`` at the call site.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def ensure_ax(
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Return (fig, ax), creating new figure if ax is None."""
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def add_colorbar(im, ax, label=None, **kwargs):
    """Add colorbar with consistent sizing."""
    defaults = dict(fraction=0.046, pad=0.04)
    defaults.update(kwargs)
    cb = ax.figure.colorbar(im, ax=ax, **defaults)
    if label:
        cb.set_label(label)
    return cb


def symmetric_clim(data: np.ndarray, percentile: float = 99) -> tuple[float, float]:
    """Symmetric color limits for diverging colormaps. Returns (-vmax, vmax)."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return (-1.0, 1.0)
    vmax = float(np.percentile(np.abs(finite), percentile))
    if vmax == 0.0:
        # All-zero (after isfinite filtering): a (-0.0, 0.0) clim is
        # degenerate (vmin == vmax). Mirror plot_gather's own fallback so
        # the clim is always non-degenerate and consistent across helpers.
        return (-1.0, 1.0)
    return (-vmax, vmax)


def _safe_log_norm(values, log_scale: bool):
    """
    Return a ``LogNorm`` only when ``values`` has at least one
    strictly-positive finite entry. Several call sites
    used ``LogNorm`` unconditionally and crashed on all-zero or
    sign-mixed data; this helper is the unified guard.

    Behaviour:

    - ``log_scale=False`` → ``None`` (linear, caller stays as-is).
    - No finite positive entries → warn, return ``None`` (degrade to
      linear) so the figure still renders.
    - Otherwise → ``LogNorm(vmin=min_positive)`` with a warning if
      non-positive entries exist (callers should mask them).
    """
    if not log_scale:
        return None
    arr = np.asarray(values)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        warnings.warn(
            "log_scale=True but values have no finite entries; "
            "falling back to linear scale.",
            UserWarning,
            stacklevel=3,
        )
        return None
    positive = arr[finite_mask & (arr > 0)]
    if positive.size == 0:
        warnings.warn(
            "log_scale=True but values have no strictly-positive "
            "entries; falling back to linear scale.",
            UserWarning,
            stacklevel=3,
        )
        return None
    if np.any(~finite_mask) or np.any(arr <= 0):
        warnings.warn(
            "log_scale=True but values contain non-positive or non-finite "
            "entries; non-positive entries will be masked in the plot.",
            UserWarning,
            stacklevel=3,
        )
    vmin = float(np.min(positive))
    return mcolors.LogNorm(vmin=vmin)


# ---------------------------------------------------------------------------
# Lightweight shape validators
# ---------------------------------------------------------------------------


def _require_2d(name: str, arr) -> np.ndarray:
    """
    Return ``arr`` as a NumPy array after asserting ``ndim == 2``.

    Plotting helpers call this once at the top of their bodies so a
    wrong-shape input fails with a named, library-level error rather
    than ``not enough values to unpack`` / ``Input z must be 2D``
    deeper inside Matplotlib.
    """
    out = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
    if out.ndim != 2:
        raise GeoBrainError(
            f"{name} must be a 2-D array (got shape {tuple(out.shape)} "
            f"with ndim={out.ndim})."
        )
    return out


def _require_non_empty(name: str, seq) -> None:
    """
    Reject empty sequences with a named error.

    Multi-panel / multi-track plotting helpers used to fall through
    into ``matplotlib.subplots(0, ...)`` and surface as
    ``ValueError: Number of columns must be positive`` or
    ``ZeroDivisionError`` from the layout math.
    """
    if len(seq) == 0:
        raise GeoBrainError(
            f"{name} must contain at least one element; got an empty "
            f"sequence."
        )


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def axis_edges(n_cells: int, spacing, origin: float, name: str) -> np.ndarray:
    """
    Node positions along one axis of a 2-D panel.

    ``spacing`` may be a single cell width, one width per cell, or the
    ``n_cells + 1`` node positions themselves. The first form is what a
    uniform grid needs and is what every caller used before graded grids
    were supported; the other two are how a graded mesh states that its
    cells are not all the same size.
    """
    arr = np.asarray(to_numpy(spacing), dtype=float)
    if arr.ndim == 0:
        return np.arange(n_cells + 1) * float(arr) + origin
    if arr.ndim != 1:
        raise GeoBrainError(
            f"{name} must be a scalar cell size, one size per cell, or the "
            f"node positions; got an array with ndim={arr.ndim}."
        )
    if arr.size == n_cells:
        return origin + np.concatenate(([0.0], np.cumsum(arr)))
    if arr.size == n_cells + 1:
        return arr + origin
    raise GeoBrainError(
        f"{name} has {arr.size} entries, which is neither one per cell "
        f"({n_cells}) nor one per node ({n_cells + 1})."
    )


def panel_edges(shape, dx, dz, ox, oz, mesh=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve the (x, z) node lines a 2-D panel is drawn on.

    Pass ``mesh`` and the geometry is taken from the mesh itself, which is
    the only way a graded grid can be drawn without distortion: building
    the node lines from a single cell size puts every boundary below the
    first grading change in the wrong place. Without a mesh the ``dx`` /
    ``dz`` arguments carry the geometry, and they accept per-cell widths
    as well as one uniform size.
    """
    nz, nx = shape
    if mesh is None:
        return (axis_edges(nx, dx, ox, "dx"), axis_edges(nz, dz, oz, "dz"))

    lines = getattr(mesh, "node_lines", None)
    if lines is None:
        raise GeoBrainError(
            "mesh does not expose node lines, so a 2-D panel cannot be "
            "drawn on it; project the field onto a structured mesh first, "
            "or plot it with plot_field_tripcolor / plot_field_polygon."
            f" | object={type(mesh).__name__!r}"
        )
    node = [np.asarray(to_numpy(line), dtype=float) for line in lines()]
    if len(node) < 2:
        raise GeoBrainError(
            f"mesh must have at least two axes to draw a 2-D panel; got "
            f"{len(node)}."
        )
    if (node[0].size - 1, node[1].size - 1) != (nz, nx):
        raise GeoBrainError(
            "data shape does not match the mesh | object='panel_edges' "
            f"| field='data' | expected={(node[0].size - 1, node[1].size - 1)} "
            f"| actual={(nz, nx)}"
        )
    return node[1] + ox, node[0] + oz


def edges_are_uniform(edges: np.ndarray) -> bool:
    """True when the node line is evenly spaced, to floating-point noise."""
    if edges.size < 3:
        return True
    step = np.diff(edges)
    return bool(np.allclose(step, step[0], rtol=1e-9, atol=0.0))
