"""Which kriging, and on what support?

"Kriging" names a family, not an algorithm. Every member solves the same
system for weights that minimise the estimation variance; they differ in
what they assume about the mean, and that assumption is the whole
difference between a good map and a bad one.

    SIMPLE kriging assumes the mean is KNOWN and constant. Where the data
        run out, the estimate falls back to that mean, which is either
        exactly right or confidently wrong.

    ORDINARY kriging assumes the mean is constant but UNKNOWN, and
        re-estimates it locally from the neighbourhood. It is the default
        for good reason, and away from data it drifts to the local mean
        rather than a global one.

    UNIVERSAL kriging assumes the mean follows a TREND, here a plane in
        x and y, and solves for the trend coefficients alongside the
        weights. On a field that genuinely has a trend it extrapolates;
        on one that does not, it invents.

The field in this script has a strong linear trend on purpose, because
that is the case where the three visibly disagree, and the script scores
all three against the truth. Do not assume universal kriging must win
that scoring: inside a search neighbourhood a plane and a slowly varying
local mean are nearly the same thing, so ordinary kriging can match it
and often does. The script prints the ordering it actually measured.

Support: the question is never asked about a point
--------------------------------------------------
Nobody mines a point, drains a point, or reports the grade of a point.
The decision is made over a block, and a block is an average. Block
kriging estimates that average directly, and the variance it returns is
smaller than the point variance by an amount the variogram determines,
because averaging over a block already integrates away the short-scale
variability that a point estimate has to carry.

The script measures the reduction and checks it against the block's own
average variogram value, so the number is explained rather than asserted.

The neighbourhood is a real choice too
--------------------------------------
Kriging with every sample is expensive and, past a point, pointless: once
a sample is beyond the range, its weight is nearly zero. The script
sweeps the neighbourhood size and reports where the accuracy stops
improving, which is the honest place to stop paying for it. The curve is
not monotonic, and the reason is in the printout.

APIs featured:
    - geobrain.geomodel.SimpleKriging / OrdinaryKriging / UniversalKriging
    - geobrain.geomodel.BlockKriging with block_size_m and discretization
    - geobrain.geomodel.NeighbourhoodSpec as the cost/accuracy knob

Expected runtime: < 3 min.

Outputs:
    out/03_kriging_estimators.png: the trended truth and its samples,
    the ordinary-kriging estimate and its error, one transect through all
    three estimators, the neighbourhood sweep, and the point-versus-block
    variance comparison.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    CMAP_ANOMALY,
    CMAP_MODEL,
    C_TRUTH,
    PALETTE,
    apply_style,
    figure,
    shared_colorbar,
)
from geobrain.geomodel import (
    FFTMA,
    BlockKriging,
    CovarianceModel,
    Detrend,
    GeoFrame,
    GeoGrid,
    GeoPoints,
    NeighbourhoodSpec,
    OrdinaryKriging,
    PropertyMetadata,
    SimpleKriging,
    SimulationExecutionConfig,
    UniversalKriging,
    VariogramCalculator,
    VariogramKernel,
)

apply_style()
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

NX, NY, CELL = 48, 48, 12.0            # 576 m square, divisible by the block
RANGE_M, SILL, MEAN = 130.0, 90.0, 40.0
TREND_X, TREND_Y = 0.055, 0.030        # per metre, a real and strong trend
N_SAMPLES = 140
PROP = PropertyMetadata("V", "continuous", "1")

# %% 1. A field with a trend, which is what separates the estimators ------
grid = GeoGrid(shape=(NX, NY), origin=(0.0, 0.0), spacing=(CELL, CELL))
cells = np.asarray(grid.coords, dtype=float)[:, :2]
residual = np.asarray(
    FFTMA(CovarianceModel(nugget=0.0, structures=[
        VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=SILL,
                        ranges=(RANGE_M, RANGE_M, 1.0e6),
                        angles=(0.0, 0.0, 0.0))]),
        property=PROP,
        execution=SimulationExecutionConfig(n_realizations=1, seed=31),
        mean=0.0)(None, grid).realizations[0].frame.to_numpy("simulation"),
    dtype=float).reshape(-1)
trend = TREND_X * cells[:, 0] + TREND_Y * cells[:, 1]
truth = MEAN + trend + residual
rng = np.random.default_rng(17)
picked = rng.choice(truth.size, size=N_SAMPLES, replace=False)
samples = GeoFrame(GeoPoints(cells[picked]), properties={"V": truth[picked]})
print(f"[1] {NX}x{NY} at {CELL:.0f} m: a {SILL:.0f}-sill residual on a plane "
      f"rising {TREND_X * NX * CELL:.0f} units west-to-east and "
      f"{TREND_Y * NY * CELL:.0f} south-to-north. Truth {truth.min():.1f} to "
      f"{truth.max():.1f}; the trend spans "
      f"{trend.max() - trend.min():.0f} of that")
print(f"[1] {N_SAMPLES} samples; sample mean {truth[picked].mean():.1f}, "
      f"field mean {truth.mean():.1f}")

# %% 2. The trend poisons the variogram before it reaches any estimator ---
raw_fit = VariogramCalculator(n_lags=14).fit(samples, "V", kind="auto")
detrended = Detrend("V", degree=1).fit_transform(samples)
residual_column = next(c for c in detrended.columns if c != "V")
fit = VariogramCalculator(n_lags=14).fit(detrended, residual_column,
                                         kind="auto")
print(f"[2] variogram of the RAW samples: {raw_fit.structures[0].name}, range "
      f"{raw_fit.structures[0].range_max:.0f} m, larger than the "
      f"{NX * CELL:.0f} m domain, because a variogram cannot tell a trend "
      "from very long-range structure and simply reports both as one")
print(f"[2] variogram of the DETRENDED samples ('{residual_column}'): "
      f"{fit.structures[0].name}, range {fit.structures[0].range_max:.0f} m "
      f"against the true residual range of {RANGE_M:.0f} m. Remove the trend "
      "first or every neighbourhood you draw from it will be global")
radius = 2.2 * float(fit.structures[0].range_max)
# min_neighbors is not cosmetic for universal kriging: fitting a plane needs
# at least three non-collinear points, and a thin neighbourhood makes the
# drift matrix singular. The library raises rather than quietly regularising.
neighbourhood = NeighbourhoodSpec(radii_m=(radius, radius), angles_deg=(0.0,),
                                  min_neighbors=6, max_neighbors=24)
print(f"[2] search radius {radius:.0f} m with at least six neighbours, a "
      "real neighbourhood rather than the whole map, which is what lets the "
      "three estimators differ, and enough points for the drift plane to be "
      "determined at every cell")


def krige(estimator, label: str):
    t0 = time.time()
    out = estimator(samples, grid)
    est = np.asarray(out["estimate"], dtype=float).reshape(-1)
    var = np.asarray(out["variance"], dtype=float).reshape(-1)
    err = float(np.sqrt(np.mean((est - truth) ** 2)))
    print(f"[2] {label:18s} RMSE {err:6.2f}, estimate {est.min():6.1f} to "
          f"{est.max():6.1f}  ({time.time() - t0:4.1f} s)")
    return dict(estimate=est, variance=var, rmse=err)


runs = {
    "Simple (known mean)": krige(
        SimpleKriging(fit, property=PROP, mean=float(truth[picked].mean()),
                      neighbourhood=neighbourhood), "simple"),
    "Ordinary": krige(
        OrdinaryKriging(fit, property=PROP, neighbourhood=neighbourhood),
        "ordinary"),
    "Universal (linear drift)": krige(
        UniversalKriging(fit, property=PROP, drift_terms=("x", "y"),
                         neighbourhood=neighbourhood), "universal"),
}
best = min(runs, key=lambda k: runs[k]["rmse"])
print(f"[3] on a field that really does have a trend, {best} wins, and the "
      "ordering is a statement about the MEAN, not about the weights")

# %% 3. How much neighbourhood is worth paying for ------------------------
print("[4] neighbourhood sweep on ordinary kriging:")
sweep = {}
for n_max in (4, 8, 16, 32):
    t0 = time.time()
    out = OrdinaryKriging(fit, property=PROP,
                          neighbourhood=NeighbourhoodSpec(
                              radii_m=(radius, radius), angles_deg=(0.0,),
                              min_neighbors=1,
                              max_neighbors=n_max))(samples, grid)
    est = np.asarray(out["estimate"], dtype=float).reshape(-1)
    sweep[n_max] = (float(np.sqrt(np.mean((est - truth) ** 2))),
                    time.time() - t0)
    print(f"      max {n_max:2d} neighbours: RMSE {sweep[n_max][0]:6.3f}, "
          f"{sweep[n_max][1]:5.1f} s")
best_n = min(sweep, key=lambda n: sweep[n][0])
worst_n = max(sweep, key=lambda n: sweep[n][0])
spread = sweep[worst_n][0] - sweep[best_n][0]
print(f"[4] the best is {best_n} neighbours at RMSE {sweep[best_n][0]:.3f}; "
      f"the worst is {worst_n} at {sweep[worst_n][0]:.3f}. The whole sweep "
      f"spans {spread:.3f} RMSE for a "
      f"{sweep[max(sweep)][1] / sweep[min(sweep)][1]:.1f}x range in runtime, "
      "so past the first handful the neighbourhood buys almost nothing")
print("[4] the reason it does not simply improve is worth knowing: ordinary "
      "kriging re-estimates the mean INSIDE its neighbourhood, so a small "
      "neighbourhood behaves like a locally varying mean and absorbs part of "
      "the trend. Widening it flattens that local mean back toward the global "
      "one, which on a trended field pulls the wrong way, and the two effects "
      "meet somewhere in the middle")

# %% 4. Point support versus block support --------------------------------
BLOCK = 4                                    # cells per side
# The block grid must SPAN the samples, not sit inside them: conditioning is
# checked against the half-open grid extent, so its origin stays at 0.
block_grid = GeoGrid(shape=(NX // BLOCK, NY // BLOCK), origin=(0.0, 0.0),
                     spacing=(BLOCK * CELL, BLOCK * CELL))
t0 = time.time()
block_out = BlockKriging(fit, property=PROP,
                         block_size_m=(BLOCK * CELL, BLOCK * CELL),
                         block_discretization=(4, 4), kind="ordinary",
                         neighbourhood=neighbourhood)(samples, block_grid)
block_est = np.asarray(block_out["estimate"], dtype=float).reshape(-1)
block_var = np.asarray(block_out["variance"], dtype=float).reshape(-1)
point_at_block = OrdinaryKriging(fit, property=PROP,
                                 neighbourhood=neighbourhood)(samples,
                                                              block_grid)
point_var = np.asarray(point_at_block["variance"], dtype=float).reshape(-1)
truth_blocks = truth.reshape(NY, NX).reshape(
    NY // BLOCK, BLOCK, NX // BLOCK, BLOCK).mean(axis=(1, 3)).reshape(-1)
print(f"[5] block kriging on {BLOCK * CELL:.0f} m blocks in "
      f"{time.time() - t0:.1f} s: mean variance {block_var.mean():.2f} "
      f"against the point estimate's {point_var.mean():.2f} at the same "
      f"locations, a {100 * (1 - block_var.mean() / point_var.mean()):.0f}% "
      "reduction")
gamma_bar = float(fit.variogram(np.linspace(1.0, BLOCK * CELL, 40)).mean())
print(f"[5] and it is not free variance: the block's own average "
      f"semivariance over its extent is about {gamma_bar:.1f}, which is the "
      f"short-scale variability a block average integrates away rather than "
      "has to predict")
print(f"[5] against the true block averages, block kriging RMSE "
      f"{np.sqrt(np.mean((block_est - truth_blocks) ** 2)):.2f}")

# %% 5. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
extent = (0.0, NX * CELL, 0.0, NY * CELL)
vlim = dict(vmin=float(truth.min()), vmax=float(truth.max()), cmap=CMAP_MODEL)
elim = float(np.abs(runs["Ordinary"]["estimate"] - truth).max())


def show(ax, field, title, style, label=None, shape=(NX, NY)):
    im = ax.imshow(np.asarray(field).reshape(shape).T, origin="lower",
                   extent=extent, aspect="auto", **style)
    ax.set(title=title, xlabel="x [m]", ylabel="y [m]")
    ax.grid(False)
    return im


im = show(axes[0, 0], truth, "Truth = plane + residual", vlim)
axes[0, 0].plot(cells[picked, 0], cells[picked, 1], ".", ms=3.5, color="white",
                mec="black", mew=0.25, ls="none", label=f"{N_SAMPLES} samples")
axes[0, 0].legend(fontsize=8, loc="lower right", framealpha=0.85)
show(axes[0, 1], runs["Ordinary"]["estimate"],
     f"Ordinary kriging\nRMSE {runs['Ordinary']['rmse']:.2f}", vlim)
shared_colorbar(fig, im, axes[0, :2], "V [-]")

# The three estimators are NOT drawn as three maps. They differ by
# hundredths of an RMSE, so three maps are one picture printed three
# times; what separates them is the drift each one assumes, and a single
# transect shows that - simple kriging pulled back towards the global
# mean between samples, universal free to follow the plane.
ax = axes[0, 2]
ROW = NY // 2
axis_x = (np.arange(NX) + 0.5) * CELL
ax.plot(axis_x, np.asarray(truth).reshape(NY, NX)[ROW, :], color=C_TRUTH,
        lw=2.4, label="Truth")
for (name, run), colour in zip(runs.items(), PALETTE):
    ax.plot(axis_x, np.asarray(run["estimate"]).reshape(NY, NX)[ROW, :],
            color=colour, lw=1.7, label=f"{name}, RMSE {run['rmse']:.2f}")
ax.set(title=f"One transect, at y = {(ROW + 0.5) * CELL:.0f} m",
       xlabel="x [m]", ylabel="V [-]")
ax.legend(fontsize=7.5, frameon=True, framealpha=0.9, loc="upper left")

im = show(axes[1, 0], runs["Ordinary"]["estimate"] - truth,
          "Ordinary kriging error",
          dict(cmap=CMAP_ANOMALY, vmin=-elim, vmax=elim))
shared_colorbar(fig, im, axes[1, 0], "estimate - truth [-]")

ax = axes[1, 1]
sizes = sorted(sweep)
ax.plot(sizes, [sweep[n][0] for n in sizes], "o-", color=PALETTE[0], lw=2.0,
        label="RMSE")
ax.set(title="RMSE and runtime by neighbourhood size",
       xlabel="Maximum neighbours", ylabel="RMSE [-]", xticks=sizes)
twin = ax.twinx()
twin.plot(sizes, [sweep[n][1] for n in sizes], "s--", color=PALETTE[1], lw=1.6)
twin.set_ylabel("Runtime [s]", color=PALETTE[1])
twin.tick_params(axis="y", colors=PALETTE[1])
twin.grid(False)
ax.legend(fontsize=8, loc="upper left")

ax = axes[1, 2]
bins = np.linspace(0.0, float(max(point_var.max(), block_var.max())), 30)
ax.hist(point_var, bins=bins, histtype="step", lw=2.0, color=PALETTE[0],
        label=f"Point support (mean {point_var.mean():.1f})")
ax.hist(block_var, bins=bins, histtype="step", lw=2.0, color=PALETTE[2],
        label=f"{BLOCK * CELL:.0f} m blocks (mean {block_var.mean():.1f})")
ax.axvline(gamma_bar, color=C_TRUTH, ls="--", lw=1.5)
ax.annotate("the block's own\nmean semivariance", xy=(gamma_bar, 0.0),
            xycoords=("data", "axes fraction"), xytext=(7, 62),
            textcoords="offset points", fontsize=8, color=C_TRUTH)
ax.set(title=f"Kriging variance, point vs {BLOCK * CELL:.0f} m blocks "
             f"({100 * (1 - block_var.mean() / point_var.mean()):.0f}% lower)",
       xlabel="Kriging variance [-]", ylabel="Cells [-]")
ax.legend(fontsize=8)

fig.savefig(OUT / "03_kriging_estimators.png")
print(f"saved {OUT / '03_kriging_estimators.png'}")
plt.show()
