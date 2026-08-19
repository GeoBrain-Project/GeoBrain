"""Where should the next hole go, and how deep?

Everything before this part produces a model, or a distribution over
models. This one spends money. The question is not "what is down there"
but "which measurement, taken next, would change what I decide", and
that can be answered BEFORE the hole is drilled, because it depends only
on the prior ensemble and on what the decision is.

The currency is the EFFICACY OF INFORMATION: the gain in the probability
of deciding correctly, cell by cell, that a proposed borehole would buy.
It follows the treatment of Caers, Scheidt, Yin, Wang, Mukerji and House,
"Efficacy of Information in Mineral Exploration Drilling", Natural
Resources Research 31(3):1157 (2022), the spatial example of their
Figures 1 and 2, computed here with the platform's own
``SpatialDecisionAccuracy``.

Not the same map as the uncertainty
-----------------------------------
Variance says where you do not know. Efficacy says where not knowing
COSTS you a decision. A cell sitting at probability 0.5 is worth
resolving; a cell already at 0.99 is worth almost nothing to measure,
however expensive it was to model. And because a borehole informs its
neighbours through the spatial correlation, its efficacy is not confined
to the hole.

Two currencies, and one of them is a trap
-----------------------------------------
``SpatialDecisionAccuracy(normalize=False)`` reports the ABSOLUTE gain in
decision accuracy, the map to drill on. ``normalize=True`` reports each
cell's gain as a fraction of that cell's own perfect-information ceiling,
which is a useful diagnostic and a terrible target map: where the prior
had already decided, the ceiling is nearly zero, the hole captures
nearly all of it, and the normalised map goes bright. The script computes
both and measures the difference, because reading the wrong one sends
the rig to the wrong place.

What is measured
----------------
1. The prior ensemble, and where its uncertainty actually lives.
2. What the decision would be with no new data at all, and how often it
   would be right, the baseline every gain is measured against.
3. One proposed borehole, in both currencies: how much accuracy it buys,
   how far its influence reaches, and what each map says about the cells
   that were already decided.
4. The design scan over location and depth, repeated on an INDEPENDENT
   prior ensemble, because a scan computed from 250 realisations is
   itself an estimate.
5. The budget view: efficacy per cell drilled, whose optimum is nowhere
   near the optimum of efficacy itself.

APIs featured:
    - geobrain.geomodel.geostats.FFTMA for the prior ensemble
    - geobrain.decision.SpatialDecisionAccuracy (PCA + linear-Gaussian
      conditioning, Bayes-action accuracy) with normalize on and off
    - DecisionAccuracyResult.gain_map / prior_accuracy / mean_gain

Expected runtime: < 2 min.

Outputs:
    out/01_borehole_efficacy_of_information.png: the prior probability,
    and the efficacy of one borehole in both currencies: the absolute
    gain that is the map to drill on, and the normalised gain that is
    not. The design scan, the two-ensemble check and the cost view are
    printed as measurements rather than drawn.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    CMAP_MODEL,
    C_RECOVERED,
    apply_style,
    figure,
    shared_colorbar,
)
from geobrain.decision import SpatialDecisionAccuracy
from geobrain.geomodel import (
    FFTMA,
    CovarianceModel,
    GeoGrid,
    PropertyMetadata,
    SimulationExecutionConfig,
)

apply_style()
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

NZ, NX = 30, 50                       # depth x location, in cells
N_REAL = 250                          # prior realisations
RANGE = 10.0                          # variogram range, in cells
BORE_COL, BORE_DEPTH = 22, 14         # the borehole put forward for costing


def as_array(value) -> np.ndarray:
    """Numpy view of whatever the decision API hands back."""
    return np.asarray(value.detach().cpu() if hasattr(value, "detach")
                      else value)


# %% 1. The prior ensemble -------------------------------------------------
#
# A binary indicator: the body is present or it is not. Each realisation is
# an isotropic Gaussian field plus a corner-to-corner trend, truncated at
# zero. The trend is what makes the problem interesting - it puts the
# uncertain boundary on a diagonal, so "where is it worth drilling" has a
# non-trivial answer rather than "in the middle".
def prior_ensemble(seed: int) -> np.ndarray:
    """``(N_REAL, NZ, NX)`` binary realisations from one simulation seed."""
    grid = GeoGrid(shape=(NX, NZ, 1), origin=(0.0, 0.0, 0.0),
                   spacing=(1.0, 1.0, 1.0))
    simulator = FFTMA(
        CovarianceModel.spherical(sill=1.0, range_=RANGE), mean=0.0,
        property=PropertyMetadata("simulation", "continuous", "1"),
        execution=SimulationExecutionConfig(n_realizations=N_REAL, seed=seed))
    # The covariance is isotropic, so the (location, depth) axis order of the
    # grid carries no anisotropy to get backwards; the transpose below only
    # reaches the (depth, location) layout the decision code expects.
    depth = np.linspace(0.0, 1.0, NZ)[:, None]
    across = np.linspace(0.0, 1.0, NX)[None, :]
    trend = 2.5 * (across + depth - 0.85)
    out = np.empty((N_REAL, NZ, NX))
    for i, realisation in enumerate(simulator(None, grid).realizations):
        field = np.asarray(
            realisation.frame.to_grid_array("simulation", order="F")).squeeze()
        out[i] = ((field.T if field.shape == (NX, NZ) else field) + trend > 0.0)
    return out


t0 = time.time()
reals = prior_ensemble(seed=7)
probability = reals.mean(0)
variance = reals.var(0)
open_cells = float((variance > 0.2).mean())
settled = float((variance < 0.01).mean())
print(f"[1] {N_REAL} indicator realisations on {NZ}x{NX} cells in "
      f"{time.time() - t0:.1f} s; the body occupies "
      f"{100 * reals.mean():.0f}% of the section on average")
print(f"[1] {100 * open_cells:.0f}% of cells are still genuinely open "
      f"(ensemble variance above 0.2) and {100 * settled:.0f}% are settled "
      "either way; the open ones lie along the diagonal boundary, which is "
      "where a hole could change something")

# %% 2. What the decision would be with no new data ------------------------
#
# Identity utilities: at each cell the decision is "present" or "absent",
# and being right is worth one. With no new data the best you can do is
# take the more likely of the two and be right max(p, 1-p) of the time.
# Every gain below is measured against that.
absolute = SpatialDecisionAccuracy(reals, np.eye(2), threshold=0.5,
                                   variance_fraction=0.8, normalize=False)
fraction = SpatialDecisionAccuracy(reals, np.eye(2), threshold=0.5,
                                   variance_fraction=0.8, normalize=True)
prior_accuracy = np.maximum(probability, 1.0 - probability)
decided = float((prior_accuracy > 0.99).mean())
kept = int(np.searchsorted(
    np.cumsum(as_array(absolute.explained_variance_ratio)), 0.8) + 1)
ceiling = float((1.0 - prior_accuracy).mean())
print(f"[2] with no new data the best decision is right "
      f"{100 * prior_accuracy.mean():.1f}% of the time on average, so perfect "
      f"information anywhere is worth at most {ceiling:.3f} of accuracy - "
      "that is the ceiling every number below is a fraction of")
print(f"[2] {100 * decided:.0f}% of cells are already decided beyond 99%, and "
      f"the ensemble compresses to {kept} principal components (80% of its "
      "variance), which is what makes scanning hundreds of candidate holes "
      "affordable")

# %% 3. One borehole, in both currencies -----------------------------------
cells = absolute.flat_indices([(z, BORE_COL) for z in range(BORE_DEPTH)])
t0 = time.time()
result = absolute.compute(cells)
result_fraction = fraction.compute(cells)
gain = as_array(result.gain_map)
gain_fraction = as_array(result_fraction.gain_map)
print(f"[3] a vertical hole at location {BORE_COL}, {BORE_DEPTH} cells deep, "
      f"scored both ways in {1000 * (time.time() - t0):.0f} ms")
print(f"      absolute accuracy gain: mean {float(result.mean_gain):.3f} over "
      f"the section, up to {gain.max():.3f} at its best cell")
print(f"      that is {100 * float(result.mean_gain) / ceiling:.0f}% of the "
      "perfect-information ceiling, from one hole")

corr_absolute = float(np.corrcoef(variance.ravel(), gain.ravel())[0, 1])
corr_fraction = float(np.corrcoef(variance.ravel(), gain_fraction.ravel())[0, 1])
print("[3] and the two currencies disagree about where to drill:")
print(f"      {'':22s} {'settled cells':>14s} {'open cells':>12s} "
      f"{'corr with variance':>20s}")
for name, field, corr in (("absolute (drill on)", gain, corr_absolute),
                          ("normalised (fraction)", gain_fraction,
                           corr_fraction)):
    print(f"      {name:22s} {field[variance < 0.01].mean():14.3f} "
          f"{field[variance > 0.2].mean():12.3f} {corr:20.2f}")
print(f"[3] the normalised map is BRIGHT where the prior had already decided "
      f"({gain_fraction[variance < 0.01].mean():.3f}), because there the "
      "ceiling is nearly zero and the hole captures nearly all of it. It is a "
      "diagnostic of how much of the available accuracy was captured, not a "
      "map of where to drill")

# How far does one hole reach? Measured on the absolute map, over the
# columns where the prior still had something to say.
offsets = np.arange(0, 16)
reach = np.array([float(np.nanmean(np.where(variance[:, BORE_COL + d] > 0.05,
                                            gain[:, BORE_COL + d], np.nan)))
                  for d in offsets])
half = float(np.interp(0.5 * reach[0], reach[::-1], offsets[::-1].astype(float)))
print(f"[3] lateral reach: the gain falls to half its at-hole value "
      f"{half:.1f} cells away, against a variogram range of {RANGE:.0f} - a "
      "hole is worth more than the column it occupies, and less than the "
      "correlation length suggests")

# %% 4. The design scan, twice ---------------------------------------------
#
# A scan computed from 250 realisations is itself an estimate. Running it
# again on an INDEPENDENT prior ensemble is the cheapest honest check of
# whether a recommendation is a finding or a fluctuation.
LOCATIONS = np.arange(2, NX, 2)
DEPTHS = np.arange(2, NZ + 1, 2)


def design_scan(scorer: SpatialDecisionAccuracy) -> np.ndarray:
    """Mean accuracy gain for every (depth, location) candidate hole."""
    scan = np.zeros((DEPTHS.size, LOCATIONS.size))
    for j, location in enumerate(LOCATIONS):
        for i, depth in enumerate(DEPTHS):
            picked = scorer.flat_indices([(z, int(location))
                                          for z in range(int(depth))])
            scan[i, j] = float(scorer.compute(picked).mean_gain)
    return scan


t0 = time.time()
scan = design_scan(absolute)
by_depth, by_location = scan.mean(axis=1), scan.mean(axis=0)
depth_span = float(by_depth.max() - by_depth.min())
location_span = float(by_location.max() - by_location.min())
best = np.unravel_index(int(scan.argmax()), scan.shape)
print(f"[4] {DEPTHS.size * LOCATIONS.size} candidate boreholes scored in "
      f"{time.time() - t0:.0f} s")
print(f"      averaged over location, the gain runs {by_depth.min():.3f} to "
      f"{by_depth.max():.3f} with depth (a span of {depth_span:.3f})")
print(f"      averaged over depth, {by_location.min():.3f} to "
      f"{by_location.max():.3f} with location (a span of "
      f"{location_span:.3f})")
print(f"[4] depth matters {depth_span / location_span:.0f}x more than "
      "location: the design question here is how deep to drill, not where")

t0 = time.time()
scan_b = design_scan(SpatialDecisionAccuracy(
    prior_ensemble(seed=101), np.eye(2), threshold=0.5,
    variance_fraction=0.8, normalize=False))
best_b = np.unravel_index(int(scan_b.argmax()), scan_b.shape)
disagreement = float(np.abs(scan - scan_b).mean())
print(f"[4] the same scan on an independent prior ensemble, "
      f"{time.time() - t0:.0f} s:")
print(f"      ensemble A likes location {LOCATIONS[best[1]]}, depth "
      f"{DEPTHS[best[0]]} ({scan[best]:.3f})")
print(f"      ensemble B likes location {LOCATIONS[best_b[1]]}, depth "
      f"{DEPTHS[best_b[0]]} ({scan_b[best_b]:.3f})")
print(f"      the two scans differ by {disagreement:.3f} on average, against "
      f"a location-to-location spread of {location_span:.3f} and a "
      f"depth span of {depth_span:.3f}")
favourite = int(LOCATIONS[int(np.argmax(scan.mean(axis=0)))])
favourite_b = int(LOCATIONS[int(np.argmax(scan_b.mean(axis=0)))])
print(f"      averaged over depth they prefer location {favourite} and "
      f"{favourite_b} respectively, {abs(favourite - favourite_b)} cells "
      f"apart on a {NX}-cell section")
print("[4] " + ("so the lateral preference does not survive a fresh ensemble: "
                "read the depth axis, and pick the location on access and cost"
                if (disagreement > 0.5 * location_span
                    or abs(favourite - favourite_b) > 4) else
                "so both the depth ranking and the lateral preference survive "
                "a fresh ensemble, and can be acted on"))

# The normalised currency, on a coarse scan, as a check that the design
# conclusion does not depend on which of the two maps you read.
coarse_locations, coarse_depths = LOCATIONS[::3], DEPTHS[::3]
coarse = np.array([[float(fraction.compute(fraction.flat_indices(
    [(z, int(loc)) for z in range(int(dep))])).mean_gain)
    for loc in coarse_locations] for dep in coarse_depths])
coarse_ratio = ((coarse.mean(1).max() - coarse.mean(1).min())
                / (coarse.mean(0).max() - coarse.mean(0).min()))
print(f"[4] scored in the normalised currency instead, depth still matters "
      f"{coarse_ratio:.0f}x more than location - the currency changes the "
      "map you look at, not the design you choose")

# %% 5. Cells drilled are the budget ---------------------------------------
per_cell = by_depth / DEPTHS
cheapest = int(np.argmax(per_cell))
print(f"[5] the deepest hole buys the most accuracy ({by_depth[-1]:.3f} at "
      f"depth {DEPTHS[-1]}) and the least per cell drilled "
      f"({per_cell[-1]:.4f}); the best value per cell is at depth "
      f"{DEPTHS[cheapest]} ({per_cell[cheapest]:.4f}, "
      f"{per_cell[cheapest] / per_cell[-1]:.1f}x better)")
print("[5] which of those two numbers is the objective is a budget question, "
      "not a geophysical one - and the efficacy map answers either, before "
      "anything is drilled")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(3, 1, panel_w=6.4, panel_h=3.0, sharex=True)


def show(ax, values, title, cmap=CMAP_MODEL, vmin=None, vmax=None,
         bore=False, **kwargs):
    ax.grid(False)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax,
                      aspect="auto", **kwargs)
    if bore:
        ax.plot([BORE_COL, BORE_COL], [0, BORE_DEPTH - 1], color=C_RECOVERED,
                lw=3.0, solid_capstyle="butt")
    ax.set(title=title, ylabel="Depth [cells]")
    return image


# Neither a single prior realisation nor the map of max(p, 1-p) is drawn:
# the first says less than the probability it came from, and the second is
# a deterministic function of that same probability. The ceiling they
# define is printed above, where a number belongs.
image = show(axes[0], probability,
             f"Ensemble probability p(x) ({100 * open_cells:.0f}% of cells "
             "open)", vmin=0.0, vmax=1.0, bore=True)
shared_colorbar(fig, image, axes[0], "P(present)", location="right")

image = show(axes[1], gain,
             f"Absolute gain (mean {float(result.mean_gain):.3f}, "
             f"{100 * float(result.mean_gain) / ceiling:.0f}% of ceiling)",
             vmin=0.0, vmax=float(gain.max()), bore=True)
shared_colorbar(fig, image, axes[1], "Gain in P(correct decision)",
                location="right")

# The same hole in the normalised currency - the trap, drawn. The design
# scan over every candidate location and depth is not: its answers are
# five numbers, and they are printed above.
image = show(axes[2], gain_fraction,
             "Normalised gain "
             f"({gain_fraction[variance < 0.01].mean():.2f} where the prior "
             "had decided)",
             vmin=0.0, vmax=1.0, bore=True)
axes[2].set_xlabel("Location [cells]")
shared_colorbar(fig, image, axes[2], "Fraction of that cell's ceiling",
                location="right")

fig.savefig(OUT / "01_borehole_efficacy_of_information.png")
print(f"saved {OUT / '01_borehole_efficacy_of_information.png'}")
plt.show()
