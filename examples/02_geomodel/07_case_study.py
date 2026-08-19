"""A study end to end: from a sample table to a number a decision needs.

The six scripts before this one each isolate a step. This one runs the
whole sequence on one dataset, in the order a real study runs it, and
ends where studies actually end: not with a map, but with a probability
attached to a threshold.

    1. Look at the data. Histogram, location map, summary statistics,
       and notice that the sampling is not uniform.
    2. Decluster. Samples cluster in interesting places, so the naive
       sample mean can be biased toward whatever made them interesting.
       Cell declustering reweights them, and the script MEASURES whether
       that bias was actually present.
    3. Transform. Sequential Gaussian simulation needs a Gaussian
       variable, so the declustered values go through a normal-score
       transform and come back at the end.
    4. Fit the variogram, in normal-score space, where the sill
       should be near 1, and check whether it is.
    5. Simulate an ensemble.
    6. Post-process into the deliverable: the ensemble mean, the spread,
       and the probability of exceeding a cutoff.

About this dataset
------------------
``geobrain.datasets.walker_lake`` is a SYNTHETIC APPROXIMATION. Its
loader builds a table whose columns, count and general character mirror
the published Walker Lake benchmark, from seeded random processes rather
than the digitised source data. It is deterministic and it is a fair
exercise of the workflow; it is not the published data, and no number
here should be quoted as if it were.

The declustering step, and an honest negative result
---------------------------------------------------
Declustering is the least glamorous step and the one that most often
changes the answer. If high-grade ground was drilled more densely, which
is what happens, because that is where the interest was, then the
equal-weight mean overstates the deposit.

On THIS table it changes almost nothing, and the script says why rather
than glossing it. The weights span a factor of sixteen, so the sampling
is genuinely irregular; but the correlation between a sample's weight and
its value is about +0.01, so there is no density-value bias for the
weights to remove. That is a real answer, not a failed step: the reason
to decluster is to find out whether the bias is there, and measuring the
correlation is how you know which case you are in.

APIs featured:
    - geobrain.datasets.walker_lake
    - geobrain.geomodel.Decluster / NormalScore
    - geobrain.geomodel.VariogramCalculator, SGSIM, PostSimulation

Expected runtime: < 4 min, nearly all of it the simulation.

Outputs:
    out/07_case_study.png: what declustering does to the histogram, the
    normal-score variogram, the grade-tonnage curve, and one realisation
    against the ensemble mean and the exceedance map.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    C_LIMIT,
    CMAP_MODEL,
    PALETTE,
    C_RECOVERED,
    C_TRUTH,
    apply_style,
    figure,
    shared_colorbar,
)
from geobrain.datasets import walker_lake
from geobrain.geomodel import (
    SGSIM,
    Decluster,
    GeoFrame,
    GeoGrid,
    NeighbourhoodSpec,
    NormalScore,
    PostSimulation,
    PropertyMetadata,
    SimulationExecutionConfig,
    VariogramCalculator,
)

apply_style()
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

N_REAL = 12
CUTOFF = 60.0
COLUMN = "V"

# %% 1. Look at the data ---------------------------------------------------
data = walker_lake()
values = np.asarray(data[COLUMN], dtype=float)
coords = np.asarray(data.geometry.coords, dtype=float)[:, :2]
print(f"[1] {len(values)} samples of '{COLUMN}' over "
      f"{coords[:, 0].min():.0f}-{coords[:, 0].max():.0f} by "
      f"{coords[:, 1].min():.0f}-{coords[:, 1].max():.0f}")
skew = float(((values - values.mean()) ** 3).mean() / values.std() ** 3)
print(f"[1] {COLUMN}: min {values.min():.1f}, median "
      f"{np.median(values):.1f}, mean {values.mean():.1f}, max "
      f"{values.max():.1f}, skew {skew:+.2f}")
print(f"[1] that skew is essentially zero and {int((values < 0).sum())} values "
      "are negative, so this stand-in behaves like a Gaussian variable rather "
      "than like a grade. The workflow below is the one a grade study runs; "
      "the DATA is not a grade, and the normal-score step therefore has less "
      "to do here than it would on a real deposit")
print("[1] NOTE: this loader is a seeded synthetic approximation of the "
      "published benchmark, not the digitised source data. It exercises the "
      "workflow honestly; it is not a result to quote")

# %% 2. Decluster ----------------------------------------------------------
decluster = Decluster(COLUMN)
weighted = decluster.fit_transform(data)
weights = np.asarray(weighted["decluster_weight"], dtype=float)
naive_mean = float(values.mean())
declustered_mean = float(decluster.declustered_mean)
print(f"[2] cell declustering at an optimal cell size of "
      f"{float(decluster.optimal_cell_size):.0f}: naive mean "
      f"{naive_mean:.2f} -> declustered {declustered_mean:.2f} "
      f"({100 * (declustered_mean / naive_mean - 1):+.1f}%)")
bias = float(np.corrcoef(weights, values)[0, 1])
print(f"[2] weights span {weights.min():.2f} to {weights.max():.2f}, a factor "
      f"of {weights.max() / weights.min():.0f}, so the sampling really is "
      f"irregular, yet the mean barely moved. The reason is measurable: the "
      f"correlation between a sample's weight and its value is {bias:+.3f}")
print("[2] declustering corrects a bias only where one exists. Dense drilling "
      "biases the mean when it went where the values are high; here it "
      "demonstrably did not, and the honest move is to measure that "
      "correlation rather than to assume the correction either way")

# %% 3. Transform, and fit the variogram where the sill is 1 --------------
normal_score = NormalScore(COLUMN, weights="decluster_weight")
ns_data = normal_score.fit_transform(weighted)
ns_column = f"{COLUMN}_ns"
t0 = time.time()
fit = VariogramCalculator(n_lags=15).fit(ns_data, ns_column, kind="auto")
kernel = fit.structures[0]
print(f"[3] normal-score transform (declustering weights carried in), then a "
      f"{kernel.name} variogram: range {kernel.range_max:.0f}, sill "
      f"{fit.sill:.2f}, nugget {fit.nugget:.2f}. In normal-score space the "
      f"sill should sit near 1; this fit reads {fit.sill:.2f}, which is the "
      "automatic fit trading sill against range on a curve that has not "
      "flattened inside the sampled lags, worth noticing rather than hiding")

# %% 4. Simulate -----------------------------------------------------------
# The conditioning check is half-open, so a sample sitting exactly on the
# upper bound falls outside. Pad by one cell.
NX, NY = 50, 50
raw = (coords[:, 0].min(), coords[:, 0].max(),
       coords[:, 1].min(), coords[:, 1].max())
pad_x = (raw[1] - raw[0]) / NX
pad_y = (raw[3] - raw[2]) / NY
extent = (raw[0] - 0.5 * pad_x, raw[1] + 0.5 * pad_x,
          raw[2] - 0.5 * pad_y, raw[3] + 0.5 * pad_y)
spacing = ((extent[1] - extent[0]) / NX, (extent[3] - extent[2]) / NY)
grid = GeoGrid(shape=(NX, NY), origin=(extent[0], extent[2]), spacing=spacing)
radius = 1.6 * float(kernel.range_max)
t0 = time.time()
ensemble = SGSIM(fit, ktype=0, mean=0.0,
                 property=PropertyMetadata(ns_column, "continuous", "1"),
                 neighbourhood=NeighbourhoodSpec(radii_m=(radius, radius),
                                                 angles_deg=(0.0,),
                                                 min_neighbors=1,
                                                 max_neighbors=24),
                 execution=SimulationExecutionConfig(n_realizations=N_REAL,
                                                     seed=2026))(ns_data, grid)
realisations = [
    normal_score.inverse_transform(
        GeoFrame(r.frame.geometry,
                 properties={ns_column: r.frame["simulation"],
                             COLUMN: r.frame["simulation"]}))
    for r in ensemble.realizations]
draws = np.stack([np.asarray(r[COLUMN], dtype=float).reshape(-1)
                  for r in realisations])
print(f"[4] {N_REAL} realisations on a {NX}x{NY} grid in "
      f"{time.time() - t0:.0f} s; back-transformed, each spans "
      f"{draws.min():.1f} to {draws.max():.1f}")
print(f"[4] realisation means average {draws.mean(axis=1).mean():.2f} against "
      f"the declustered sample mean of {declustered_mean:.2f}, so the "
      "declustering survived the round trip, which is the point of carrying "
      "the weights into the transform")

# %% 5. Post-process into the deliverable ---------------------------------
post = PostSimulation(column=COLUMN, percentiles=[10, 50, 90],
                      exceed_cutoffs=[CUTOFF])(realisations)
mean_map = np.asarray(post["mean"], dtype=float).reshape(-1)
p10 = np.asarray(post["p10"], dtype=float).reshape(-1)
p90 = np.asarray(post["p90"], dtype=float).reshape(-1)
exceed_key = next(k for k in post.columns if k.startswith("p_gt"))
exceedance = np.asarray(post[exceed_key], dtype=float).reshape(-1)
print(f"[5] ensemble mean {mean_map.min():.1f} to {mean_map.max():.1f}; "
      f"P90 minus P10 spans {(p90 - p10).min():.1f} to {(p90 - p10).max():.1f}")
print(f"[5] P({COLUMN} > {CUTOFF:.0f}) reaches {exceedance.max():.2f} and "
      f"covers {100 * np.mean(exceedance > 0.5):.0f}% of the area above an "
      "even chance")

# The grade-tonnage curve: what the ensemble says about every cutoff, with
# the spread ACROSS realisations rather than from the mean map alone.
cutoffs = np.linspace(float(np.percentile(draws, 5)),
                      float(np.percentile(draws, 95)), 30)
tonnage = np.stack([[float(np.mean(d > c)) for c in cutoffs] for d in draws])
grade = np.stack([[float(d[d > c].mean()) if np.any(d > c) else np.nan
                   for c in cutoffs] for d in draws])
mean_tonnage_from_mean_map = np.array([float(np.mean(mean_map > c))
                                       for c in cutoffs])
at_cutoff = int(np.argmin(np.abs(cutoffs - CUTOFF)))
print(f"[6] at the {CUTOFF:.0f} cutoff the ensemble puts "
      f"{100 * tonnage[:, at_cutoff].mean():.1f}% of the area above it, "
      f"P10 to P90 {100 * np.percentile(tonnage[:, at_cutoff], 10):.1f}% to "
      f"{100 * np.percentile(tonnage[:, at_cutoff], 90):.1f}%")
print(f"[6] the SMOOTHED mean map would have said "
      f"{100 * mean_tonnage_from_mean_map[at_cutoff]:.1f}% and reading a "
      "grade-tonnage curve off an averaged map is the smoothing error of "
      "script 01 arriving in the economics, and it is why the curve is "
      "computed per realisation and only then summarised")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
plot_extent = (extent[0], extent[1], extent[2], extent[3])


def show(ax, field, title, style, label=None):
    im = ax.imshow(np.asarray(field).reshape(NY, NX), origin="lower",
                   extent=plot_extent, aspect="auto", **style)
    ax.set(title=title, xlabel="x", ylabel="y")
    ax.grid(False)
    return im


ax = axes[0, 0]
ax.hist(values, bins=30, color=PALETTE[0], alpha=0.75, density=True,
        label="Equal weight")
ax.hist(values, bins=30, weights=weights / weights.sum() * len(values),
        histtype="step", lw=2.2, color=C_RECOVERED, density=True,
        label="Declustered")
ax.axvline(naive_mean, color=PALETTE[0], ls="--", lw=1.5)
ax.axvline(declustered_mean, color=C_RECOVERED, ls="--", lw=1.5)
ax.set(title=f"Declustering: mean {naive_mean:.1f} to "
             f"{declustered_mean:.1f}", xlabel=COLUMN, ylabel="Density")
ax.legend(fontsize=8)

ax = axes[0, 1]
exp = VariogramCalculator(n_lags=15).compute(ns_data, ns_column)
ax.plot(exp.lags, exp.gammas, "o", ms=6, color=C_TRUTH, label="Experimental")
lag_axis = np.linspace(0.0, float(np.max(exp.lags)), 200)
ax.plot(lag_axis, fit.variogram(lag_axis), color=C_RECOVERED, lw=2.0,
        label=f"{kernel.name}, range {kernel.range_max:.0f}")
ax.axhline(1.0, color=C_LIMIT, ls="--", lw=1.3)
ax.annotate("sill = 1 in normal-score space", xy=(0.03, 1.0),
            xycoords=("axes fraction", "data"), xytext=(0, 4),
            textcoords="offset points", fontsize=8, va="bottom",
            color=C_LIMIT)
ax.set(title="Variogram, normal-score space", xlabel="Lag",
       ylabel="Semivariance")
ax.legend(fontsize=8, loc="lower right")

vlim = dict(vmin=float(np.percentile(draws, 1)),
            vmax=float(np.percentile(draws, 99)), cmap=CMAP_MODEL)
# Row 1 is the maps, in workflow order, so the two panels on the same V
# scale sit next to each other and share one bar.
im = show(axes[1, 0], draws[0], "One realisation", vlim)
show(axes[1, 1], mean_map, "Ensemble mean", vlim)
shared_colorbar(fig, im, axes[1, :2], COLUMN, location="bottom")
im = show(axes[1, 2], exceedance, f"P({COLUMN} > {CUTOFF:.0f})",
          dict(cmap=CMAP_MODEL, vmin=0.0, vmax=1.0))
shared_colorbar(fig, im, axes[1, 2], "Probability", location="bottom")

ax = axes[0, 2]
ax.plot(cutoffs, 100 * tonnage.mean(axis=0), color=PALETTE[0], lw=2.2,
        label="Ensemble mean of realisations")
ax.fill_between(cutoffs, 100 * np.percentile(tonnage, 10, axis=0),
                100 * np.percentile(tonnage, 90, axis=0), color=PALETTE[0],
                alpha=0.22, label="P10 to P90")
ax.plot(cutoffs, 100 * mean_tonnage_from_mean_map, color=C_RECOVERED, ls="--",
        lw=2.0, label="Read off the mean map (wrong)")
ax.axvline(CUTOFF, color=C_TRUTH, ls=":", lw=1.4)
ax.set(title="Grade-tonnage per realisation",
       xlabel=f"Cutoff on {COLUMN}", ylabel="Area above cutoff [%]")
ax.legend(fontsize=8)

fig.savefig(OUT / "07_case_study.png")
print(f"saved {OUT / '07_case_study.png'}")
plt.show()
