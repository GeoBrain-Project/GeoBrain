"""Shared figure toolkit for the GeoBrain example gallery.

Every script calls :func:`apply_style` once and builds its figure through
the helpers here, so the whole gallery reads as one document rather than
as thirty scripts by thirty authors. An identical copy of this file sits
in each part's directory, because each script is run from its own folder.

Colour has ROLES, and only four of them
---------------------------------------
Nine colour ramps in a gallery is nine things a reader has to learn. The
gallery uses four, chosen for perceptual uniformity and colour-blind
safety, and each one means something:

- :data:`CMAP_MODEL` (plasma) is for a property whose VALUE matters:
  velocity, porosity, resistivity, probability, a recovered model.
- :data:`CMAP_ANOMALY` (RdBu_r) is for a SIGNED quantity about zero: a
  gradient, a difference, a density contrast. Always drawn on limits
  symmetric about zero, via :func:`symmetric_limits`.
- :data:`CMAP_MAGNITUDE` (YlGnBu) is for a NON-NEGATIVE magnitude: absolute
  error, standard deviation, energy. Light at zero, so a mostly-empty
  field reads as mostly empty rather than as a black rectangle.
- :data:`CMAP_CATEGORY` is for discrete classes: rock units, facies. Built
  from the same colour-blind-safe palette as the line colours, because a
  rock unit is a name, not a number on a continuous ramp.

A quantity that is signed in principle but ONE-SIDED in the data (a
density deficit that never goes positive, a chargeability that never goes
negative) would waste half of :data:`CMAP_ANOMALY` on values that never
occur, so it gets :data:`CMAP_EXCESS` or :data:`CMAP_DEFICIT`. Those are
the two HALVES of the diverging ramp itself, so the sign convention is
identical (blue below, red above) and white is still zero, but the whole
bar is values the data can actually take. Note that this cannot be done
by clipping the diverging ramp to ``(min, 0)``, that puts zero at the
ramp's end rather than its white middle, and paints the background dark.

Lines have roles too: truth is black, starting models are dotted blue,
recovered results are orange, observations are black markers, and a
threshold to compare against is neutral grey. A role is a NAME, never an
index into :data:`PALETTE`. The palette is ordered for separation
between anonymous series and is free to be reordered; the roles are not.

Three exceptions, where the domain wins
---------------------------------------
Against all of the above, three quantities carry a convention so old that
a practitioner reads them faster in the convention than in anything
better behaved, and ``geobrain.vis`` already ships all three:

- :data:`CMAP_SEISMIC` (``geo_seismic``): a seismic amplitude is blue
  through white to red about zero. This one is simply correct: it is a
  proper diverging ramp, and it is what every gather in the industry is
  plotted in.
- :data:`CMAP_VELOCITY` (``geo_velocity``): a velocity model is the
  cool-to-warm rainbow.
- :data:`CMAP_RESISTIVITY` (``geo_resistivity``): a resistivity section
  is the resistivity rainbow, over decades on a log scale.

The last two are rainbows, and rainbows have a real defect: their
lightness rises and falls (measured on ``geo_velocity``: 0.07 up to 0.74
and back to 0.24), so two different values print as the same grey and the
bright band reads as an edge that is not in the data. They are used here
anyway, because a velocity model or a resistivity section in a
perceptually uniform ramp costs the reader more recognition than it buys
them accuracy. Matplotlib's ``jet`` is NOT used: it is the same rainbow
with sharper false edges and no domain provenance. Everything else (a
geostatistical grade, a porosity, a probability) has no such convention,
and keeps the role ramps above.

One colour bar per SCALE, not per panel
---------------------------------------
Four panels showing the same quantity on the same limits need one colour
bar between them, not four. :func:`shared_colorbar` does that, and
:func:`common_limits` is how the panels come to share a scale in the
first place, which is not cosmetic: panels drawn on different limits
cannot be compared by eye, however much the caption asks the reader to.

Where the bar GOES is not cosmetic either. Matplotlib takes the bar's
space out of the axes it is attached to and out of no others, so a bar
beside two panels of a row of three leaves those two narrower than the
third and the columns stop lining up from row to row. Unless the group is
the whole grid, the bar therefore goes underneath it, and it is drawn the
same thickness whatever it spans.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator

# Role ramps. Continuous ones are perceptually uniform; the categorical one
# is built below from the line palette.
CMAP_MODEL = "plasma"
CMAP_ANOMALY = "RdBu_r"
CMAP_MAGNITUDE = "YlGnBu"


def _half_of(cmap_name: str, side: str, name: str) -> ListedColormap:
    """One half of a diverging ramp, still white at the zero end."""
    base = plt.get_cmap(cmap_name)
    lo, hi = (0.5, 1.0) if side == "upper" else (0.0, 0.5)
    return ListedColormap(base(np.linspace(lo, hi, 128)), name=name)


#: One-sided anomalies: literally the halves of :data:`CMAP_ANOMALY`, so a
#: quantity that never changes sign is drawn in the same colours as one
#: that does, without spending half the bar on values that cannot occur.
CMAP_EXCESS = _half_of(CMAP_ANOMALY, "upper", "geobrain_excess")
CMAP_DEFICIT = _half_of(CMAP_ANOMALY, "lower", "geobrain_deficit")

# Domain ramps, shipped by geobrain.vis and registered by apply_style. See
# the module docstring: three quantities have a convention old enough and
# strong enough that a practitioner reads them faster in the convention
# than in anything better behaved.
CMAP_SEISMIC = "geo_seismic"
CMAP_VELOCITY = "geo_velocity"
CMAP_RESISTIVITY = "geo_resistivity"

# Colour-blind-safe line palette for ANONYMOUS series - the k-th of n
# things that have no meaning beyond being different from each other.
# Ordered so that the first four are as far apart as the gamut allows:
# two similar oranges next to each other in a four-well plot is the one
# mistake this palette exists to prevent.
PALETTE = ("#0173b2", "#d55e00", "#029e73", "#cc78bc",
           "#de8f05", "#56b4e9", "#949494")

CMAP_CATEGORY = ListedColormap(PALETTE, name="geobrain_category")

# Colours for lines that MEAN something. A role never reads an index out
# of PALETTE: the palette is free to be reordered for separation, and
# these stay put.
C_TRUTH = "black"
C_START = "#0173b2"
C_RECOVERED = "#d55e00"
C_OBSERVED = "black"
C_PREDICTED = "#d55e00"
C_SERIES = "#0173b2"
C_POSTERIOR = "#029e73"
#: A threshold the reader is asked to compare against - chi-squared = 1,
#: the sill, the score of the true model. Neutral, so that it never
#: competes with a data series for a hue.
C_LIMIT = "#525252"

#: Canonical panel size in inches. Every figure in the gallery is a grid of
#: these, so a 2x3 and a 2x4 have panels of the same size and the parts do
#: not look as though they came from different projects.
PANEL_W, PANEL_H = 4.3, 3.6
#: No figure-level title, so no strip reserved for one. The claim lives in
#: the script's docstring and in the part README; the panel titles carry
#: what each panel is and what it measured. A figure that also captions
#: itself says the same thing twice and steals the room to do it.
SUPTITLE_H = 0.0


def apply_style() -> None:
    """Set the gallery-wide matplotlib defaults (idempotent).

    Also registers ``geobrain.vis``'s domain ramps, so that the three
    quantities with a real convention can be drawn in it.
    """
    from geobrain.vis import register_colormaps

    register_colormaps()
    plt.rcParams.update({
        "figure.constrained_layout.use": True,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "figure.titlesize": 13,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        # A legend gets a white backing with no edge: invisible against the
        # white panel of a line plot, and the difference between readable
        # and unreadable when it lands on a map.
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(PALETTE)),
        "image.aspect": "auto",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


def figure(nrows: int = 1, ncols: int = 1, *, scale: float = 1.0,
           panel_w: float | None = None, panel_h: float | None = None,
           **kwargs):
    """A figure whose panels are a canonical size.

    Panels are NOT lettered. Letters are journal furniture: they exist so a
    caption can say "see (c)". These figures carry their claim in the panel
    title instead, and several of them are read as thumbnails in the
    project README, where a letter is one more thing in the way.

    Args:
        nrows, ncols: panel grid.
        scale: multiply the canonical panel size, for the rare figure whose
            panels need more room (a wide profile) or less (a bar chart).
        panel_w, panel_h: override the panel WIDTH or HEIGHT in inches. The
            canonical panel is landscape, which suits a map. A depth
            profile, a well display or a sounding is read DOWN the page,
            and squeezing it into a landscape panel flattens the one axis
            the reader came for; those figures set a narrow ``panel_w``
            and a tall ``panel_h``.
        **kwargs: forwarded to ``plt.subplots`` (``sharex``, ``height_ratios``).

    Returns:
        ``(fig, axes)`` exactly as ``plt.subplots`` returns them.
    """
    width = PANEL_W if panel_w is None else panel_w
    height = PANEL_H if panel_h is None else panel_h
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(scale * width * ncols,
                 scale * height * nrows + SUPTITLE_H),
        **kwargs)
    return fig, axes


def field(ax, values, *, extent=None, cmap: object = CMAP_MODEL,
          vmin: float | None = None, vmax: float | None = None,
          title: str | None = None, xlabel: str | None = None,
          ylabel: str | None = None, origin: str = "upper",
          aspect: str = "auto", **kwargs):
    """Draw a 2-D field the gallery's way; return the image for a colour bar.

    Turns the grid off (a grid over an image is noise), applies the role
    ramp, and sets the labels in one call so that thirty maps across the
    gallery are drawn the same way.
    """
    ax.grid(False)
    image = ax.imshow(np.asarray(values), extent=extent, cmap=cmap, vmin=vmin,
                      vmax=vmax, origin=origin, aspect=aspect, **kwargs)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return image


#: Colour bars are this thick, in inches, whatever they span. Matplotlib
#: sizes a bar by ASPECT, so a bar laid under three panels comes out three
#: times thicker than one under a single panel - which is why an otherwise
#: tidy grid ends up with a slab across the bottom of it.
BAR_THICKNESS = 0.15


def _bar_extent(flat):
    """``(n_rows, n_columns, rows, columns)`` the group occupies, or None."""
    rows: set[int] = set()
    columns: set[int] = set()
    geometry = None
    for ax in flat:
        spec = ax.get_subplotspec()
        if spec is None:
            return None
        geometry = spec.get_gridspec().get_geometry()
        rows.update(spec.rowspan)
        columns.update(spec.colspan)
    if geometry is None:
        return None
    return geometry[0], geometry[1], rows, columns


def _bar_span(fig, extent, location: str) -> float:
    """How long the bar will be, in inches, along its own axis."""
    n_rows, n_columns, rows, columns = extent
    width, height = fig.get_size_inches()
    if location in ("bottom", "top"):
        return float(width) * len(columns) / n_columns
    return float(height) * len(rows) / n_rows


def shared_colorbar(fig, image, axes, label: str, *,
                    location: str | None = None, **kwargs):
    """One colour bar for every panel that shares a scale.

    Pass the whole group of axes. Matplotlib steals the bar's space from
    the group and from nothing else, which is what makes a grid look
    ragged: a bar beside two panels of a row of three leaves those two
    narrower than their neighbour, and the columns stop lining up between
    rows. So unless the group is the WHOLE grid, the bar goes underneath
    it, where the space it takes is space no panel wanted. Pass
    ``location`` explicitly to override.

    The bar comes out the same thickness whether it serves one panel or
    four, so that a figure reads as a grid of panels with a rule beside
    them rather than as panels competing with their own colour bars, and
    it carries as many ticks as it has room for rather than a fixed count
    that runs its labels together on a short bar.
    """
    flat = list(np.atleast_1d(np.asarray(axes, dtype=object)).ravel())
    extent = _bar_extent(flat)
    if extent is None:
        return fig.colorbar(image, ax=flat, label=label,
                            location=location or "right", **kwargs)
    if location is None:
        n_rows, n_columns, rows, columns = extent
        whole_grid = len(rows) == n_rows and len(columns) == n_columns
        location = "right" if whole_grid else "bottom"
    span = _bar_span(fig, extent, location)
    kwargs.setdefault("aspect", max(8.0, span / BAR_THICKNESS))
    bar = fig.colorbar(image, ax=flat, label=label, location=location,
                       **kwargs)
    # A horizontal tick label is roughly three times as wide as a vertical
    # one is tall, so the two orientations do not get the same tick budget.
    room = 1.1 if location in ("bottom", "top") else 0.45
    if not isinstance(getattr(image, "norm", None), BoundaryNorm):
        bar.locator = MaxNLocator(nbins=max(3, int(span / room)),
                                  steps=[1, 2, 2.5, 5, 10])
        bar.update_ticks()
    return bar


def common_limits(*arrays, quantile: float | None = None) -> tuple[float, float]:
    """Limits that let a set of panels be compared by eye.

    Args:
        *arrays: the fields that must share a scale.
        quantile: if given, clip to this two-sided quantile (0.99 keeps the
            middle 99%), so one outlying cell cannot flatten every panel.
    """
    stacked = np.concatenate([np.asarray(a, dtype=float).ravel()
                              for a in arrays])
    stacked = stacked[np.isfinite(stacked)]
    if quantile is None:
        return float(stacked.min()), float(stacked.max())
    tail = 0.5 * (1.0 - quantile)
    return (float(np.quantile(stacked, tail)),
            float(np.quantile(stacked, 1.0 - tail)))


def symmetric_limits(*arrays, quantile: float | None = None) -> tuple[float, float]:
    """Limits symmetric about zero, so a diverging ramp means what it looks
    like: white is zero, and equal reds and blues are equal magnitudes."""
    lo, hi = common_limits(*arrays, quantile=quantile)
    bound = max(abs(lo), abs(hi))
    return -bound, bound


def category_norm(n_classes: int, colours: Sequence[str] | None = None):
    """``(cmap, norm)`` for ``n_classes`` discrete classes.

    A rock unit is a name, not a value: drawing units 0, 1, 2 on a
    continuous ramp invites the reader to interpolate between them.

    Args:
        n_classes: how many classes.
        colours: override the palette. Worth doing when one class is a
            BACKGROUND, since "nothing here" should be neutral, not a saturated
            hue competing with the classes that matter, or when the same
            classes appear in a neighbouring panel and must be the same
            colour there.
    """
    if colours is None:
        colours = [PALETTE[i % len(PALETTE)] for i in range(n_classes)]
    cmap = ListedColormap(list(colours))
    norm = BoundaryNorm(np.arange(-0.5, n_classes, 1.0), n_classes)
    return cmap, norm


def category_colorbar(fig, image, axes, label: str, names: Sequence[str],
                      **kwargs):
    """A colour bar with one tick per class, labelled by name."""
    bar = shared_colorbar(fig, image, axes, label, **kwargs)
    bar.set_ticks(np.arange(len(names)))
    bar.set_ticklabels(list(names))
    return bar


def outer_labels(axes, xlabel: str | None = None,
                 ylabel: str | None = None) -> None:
    """Label only the bottom row and the left column of a grid.

    Repeating "Distance [m]" under all eight panels of a figure is eight
    times the ink for one piece of information.
    """
    grid = np.atleast_2d(np.asarray(axes, dtype=object))
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            ax = grid[row, column]
            if xlabel is not None:
                ax.set_xlabel(xlabel if row == grid.shape[0] - 1 else "")
            if ylabel is not None:
                ax.set_ylabel(ylabel if column == 0 else "")


def animation(fig, draw, n_frames: int, path, *, fps: int = 8,
              dpi: int = 80) -> None:
    """Write a small looping GIF and close the figure.

    Animations earn their place only where TIME is the subject: a front
    advancing, a model sharpening band by band, an ensemble flickering
    through what it does not know. A still frame says those things badly
    or not at all; everything else in the gallery is a still, because a
    reader cannot scrub a GIF back to the frame they wanted.

    The defaults keep files small enough to live in a repository (a few
    hundred kilobytes): modest dpi, few frames, and no attempt at a
    high-fidelity render. Pass the frame INDEX to ``draw``.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    FuncAnimation(fig, draw, frames=n_frames, blit=False).save(
        path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"saved {path}")
