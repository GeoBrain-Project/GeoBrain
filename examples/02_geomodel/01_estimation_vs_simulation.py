"""Estimation and simulation answer different questions.

Given samples at scattered locations and a map to fill in, there are two
things you can do, and confusing them is the most expensive mistake in
geostatistics.

    KRIGING estimates. At every cell it returns the linear combination of
        the neighbouring samples with the smallest expected squared error,
        plus the variance of that error. It is the best single number per
        cell, and because averaging is smoothing, the map it draws is
        smoother than the earth ever was.

    SIMULATION draws. Every realisation reproduces the sample histogram
        and the variogram by construction, so it looks like the earth and
        has the right amount of variability. It is NOT the best estimate:
        any single realisation is worse than the kriged map, cell for
        cell, and that is the point: you generate many and let the spread
        answer questions about the outcome.

The script measures all of that against a known truth. A field is drawn
with FFT-MA, sampled at 160 scattered points, and the samples alone are
handed to ordinary kriging and to sequential Gaussian simulation.

What is measured
----------------
1. **Accuracy.** Kriging beats any single realisation on RMSE against the
   truth, and it also beats the ensemble mean by a little. If your
   deliverable is one map of best guesses, krige.

2. **Variability.** The kriged map's variance is far below the truth's,
   and its variogram sits below the model at every lag, and the smoothing is
   visible as missing sill. A realisation carries most of it back, though not
   all, because a realisation honours the FITTED variogram and the fit is
   itself estimated from 160 points. If your deliverable feeds a flow
   simulation, a mine plan
   or any nonlinear calculation, that missing variance is the answer being
   wrong in a way no error bar reports.

3. **Two different uncertainties.** Kriging variance is the uncertainty of
   the ESTIMATE and depends only on where the samples are, not on their
   values, so it is a map of data coverage. The ensemble's P90 minus P10 is
   the uncertainty of the OUTCOME. They are not interchangeable and the
   figure shows both.

APIs featured:
    - geobrain.geomodel.OrdinaryKriging with NeighbourhoodSpec
    - geobrain.geomodel.SGSIM + NormalScore + PostSimulation
    - geobrain.geomodel.VariogramCalculator for the diagnostic variograms
    - geobrain.geomodel.FFTMA to make a truth worth measuring against

Expected runtime: < 3 min, nearly all of it the sequential
simulation itself.

Outputs:
    out/01_estimation_vs_simulation.png: the truth and its samples, the
    kriged map and its variance, two realisations, the ensemble spread,
    and the variogram and histogram diagnostics.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _style import (
    CMAP_MAGNITUDE,
    CMAP_MODEL,
    PALETTE,
    C_LIMIT,
    C_TRUTH,
    animation,
    apply_style,
    figure,
    shared_colorbar,
)
from geobrain.geomodel import (
    FFTMA,
    SGSIM,
    CovarianceModel,
    GeoFrame,
    GeoGrid,
    GeoPoints,
    NeighbourhoodSpec,
    NormalScore,
    OrdinaryKriging,
    PostSimulation,
    PropertyMetadata,
    SimulationExecutionConfig,
    VariogramCalculator,
    VariogramKernel,
)

apply_style()
rng = np.random.default_rng(11)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

NX, NY, CELL = 84, 48, 10.0            # 840 x 480 m
RANGE_M, AZIMUTH = 150.0, 30.0
N_SAMPLES, N_REAL = 160, 12
MEAN, SILL = 60.0, 220.0
PROP = PropertyMetadata("V", "continuous", "1")
grid = GeoGrid(shape=(NX, NY), origin=(0.0, 0.0), spacing=(CELL, CELL))
model = CovarianceModel(nugget=0.0, structures=[
    VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=SILL,
                    ranges=(RANGE_M, 0.45 * RANGE_M, 1.0e6),
                    angles=(AZIMUTH, 0.0, 0.0))])

# %% 1. A truth, and the only thing the inversion is allowed to see -------
t0 = time.time()
truth_frame = FFTMA(model, property=PROP,
                    execution=SimulationExecutionConfig(n_realizations=1,
                                                        seed=2025),
                    mean=MEAN)(None, grid).realizations[0].frame
truth = np.asarray(truth_frame.to_numpy("simulation"), dtype=float).reshape(-1)
cells = np.asarray(grid.coords, dtype=float)[:, :2]
picked = rng.choice(truth.size, size=N_SAMPLES, replace=False)
samples = GeoFrame(GeoPoints(cells[picked]), properties={"V": truth[picked]})
print(f"[1] truth on {NX}x{NY} cells at {CELL:.0f} m, spherical variogram "
      f"{RANGE_M:.0f} m by {0.45 * RANGE_M:.0f} m at {AZIMUTH:.0f} deg: "
      f"V {truth.min():.1f} to {truth.max():.1f}, sd {truth.std():.1f} "
      f"({time.time() - t0:.1f} s)")
print(f"[1] {N_SAMPLES} scattered samples: {100 * N_SAMPLES / truth.size:.1f}% "
      "of the cells is everything either method gets")

# %% 2. The variogram both methods will be given --------------------------
fit = VariogramCalculator(n_lags=14).fit(samples, "V", kind="auto")
kernel = fit.structures[0]
print(f"[2] variogram fitted to the samples alone: {kernel.name}, range "
      f"{kernel.range_max:.0f} m, sill {fit.sill:.0f}, nugget {fit.nugget:.1f} "
      f"(truth was spherical, {RANGE_M:.0f} m, {SILL:.0f}, 0)")
radius = 1.6 * float(kernel.range_max)
neighbourhood = NeighbourhoodSpec(radii_m=(radius, radius), angles_deg=(0.0,),
                                  max_neighbors=24)

# %% 3. Krige once ---------------------------------------------------------
t0 = time.time()
kriged = OrdinaryKriging(fit, property=PROP,
                         neighbourhood=neighbourhood)(samples, grid)
estimate = np.asarray(kriged["estimate"], dtype=float).reshape(-1)
kriging_var = np.asarray(kriged["variance"], dtype=float).reshape(-1)
print(f"[3] ordinary kriging in {time.time() - t0:.1f} s: estimate "
      f"{estimate.min():.1f} to {estimate.max():.1f}, sd {estimate.std():.1f} "
      f"against the truth's {truth.std():.1f}, the smoothing in one number")

# %% 4. Simulate many -----------------------------------------------------
t0 = time.time()
normal_score = NormalScore("V")
samples_ns = normal_score.fit_transform(samples)
fit_ns = VariogramCalculator(n_lags=14).fit(samples_ns, "V_ns", kind="auto")
radius_ns = 1.6 * float(fit_ns.structures[0].range_max)
sgsim = SGSIM(fit_ns, ktype=0, mean=0.0,
              property=PropertyMetadata("V_ns", "continuous", "1"),
              neighbourhood=NeighbourhoodSpec(radii_m=(radius_ns, radius_ns),
                                              angles_deg=(0.0,),
                                              max_neighbors=24),
              execution=SimulationExecutionConfig(n_realizations=N_REAL,
                                                  seed=7))
# Each realisation arrives as a GeoFrame carrying "simulation" in normal-score
# space; the back-transform needs it under the original column name.
realisations = [
    normal_score.inverse_transform(
        GeoFrame(r.frame.geometry,
                 properties={"V_ns": r.frame["simulation"],
                             "V": r.frame["simulation"]}))
    for r in sgsim(samples_ns, grid).realizations]
draws = np.stack([np.asarray(r["V"], dtype=float).reshape(-1)
                  for r in realisations])
post = PostSimulation(column="V", percentiles=[10, 50, 90])(realisations)
spread = (np.asarray(post["p90"], dtype=float).reshape(-1)
          - np.asarray(post["p10"], dtype=float).reshape(-1))
ensemble_mean = draws.mean(axis=0)
print(f"[4] {N_REAL} SGSIM realisations in {time.time() - t0:.0f} s: each has "
      f"sd {draws.std(axis=1).mean():.1f} on average against the truth's "
      f"{truth.std():.1f}, which the kriged map missed by "
      f"{100 * (1 - estimate.std() / truth.std()):.0f}%")


# %% 5. Score them on the two different questions -------------------------
def rmse(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean((field - truth) ** 2)))


single = float(np.mean([rmse(d) for d in draws]))
print(f"[5] as an ESTIMATE, cell by cell: kriging RMSE {rmse(estimate):.2f}, "
      f"ensemble mean {rmse(ensemble_mean):.2f}, a single realisation "
      f"{single:.2f} on average. Kriging wins, and it is supposed to")
print(f"[5] as a MODEL OF THE EARTH: variance of the truth "
      f"{truth.var():.0f}, of a realisation {draws.var(axis=1).mean():.0f}, "
      f"of the kriged map {estimate.var():.0f}. Simulation wins, and it is "
      "supposed to")

# The diagnostic that shows WHY: the variogram of each map against the model
# the samples asked for. Kriging cannot put back variance it averaged away.
diag_grid = GeoPoints(cells)


def experimental(field: np.ndarray):
    frame = GeoFrame(diag_grid, properties={"V": field})
    exp = VariogramCalculator(n_lags=12).compute(frame, "V")
    return (np.asarray(exp.lags, dtype=float),
            np.asarray(exp.gammas, dtype=float))


lag_t, gam_t = experimental(truth)
lag_k, gam_k = experimental(estimate)
lag_s, gam_s = experimental(draws[0])
mid = slice(2, None)
print(f"[5] and the variogram says it: averaged over lags beyond the first "
      f"two, the kriged map carries {100 * np.mean(gam_k[mid] / gam_t[mid]):.0f}% "
      f"of the truth's semivariance, a realisation "
      f"{100 * np.mean(gam_s[mid] / gam_t[mid]):.0f}%")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(2, 3, panel_h=3.0)
extent = (0.0, NX * CELL, 0.0, NY * CELL)
vlim = dict(vmin=float(truth.min()), vmax=float(truth.max()), cmap=CMAP_MODEL)


def show(ax, field, title, style, label=None):
    im = ax.imshow(field.reshape(NY, NX), origin="lower", extent=extent,
                   aspect="auto", **style)
    ax.set(title=title, xlabel="x [m]", ylabel="y [m]")
    ax.grid(False)
    return im


# One scale, one bar for the three maps of V: the comparison between
# them - smooth estimate against textured realisation - is the point, and
# three separate bars would let each panel rescale away the difference.
im = show(axes[0, 0], truth, "Truth", vlim)
axes[0, 0].plot(cells[picked, 0], cells[picked, 1], ".", ms=3.5,
                color="white", mec="black", mew=0.25, ls="none",
                label=f"{N_SAMPLES} samples")
axes[0, 0].legend(fontsize=8, loc="upper right", framealpha=0.85)
show(axes[0, 1], estimate,
     f"Kriged estimate\nRMSE {rmse(estimate):.2f}, sd {estimate.std():.1f}",
     vlim)
show(axes[0, 2], draws[0],
     f"One realisation\nRMSE {rmse(draws[0]):.2f}, sd {draws[0].std():.1f}",
     vlim)
shared_colorbar(fig, im, axes[0, :], "V [-]")

im = show(axes[1, 0], np.sqrt(kriging_var),
          "Kriging standard error\n(a map of where the samples are)",
          dict(cmap=CMAP_MAGNITUDE))
axes[1, 0].plot(cells[picked, 0], cells[picked, 1], ".", ms=3.0,
                color="white", ls="none")
shared_colorbar(fig, im, axes[1, 0], "sd of the estimate [-]")
im = show(axes[1, 1], spread,
          "Ensemble P90 - P10\n(a map of what the outcome could be)",
          dict(cmap=CMAP_MAGNITUDE))
shared_colorbar(fig, im, axes[1, 1], "V [-]")

ax = axes[1, 2]
ax.plot(lag_t, gam_t, "o-", ms=4, color=C_TRUTH, lw=2.0,
        label=f"Truth (sd {truth.std():.1f})")
ax.plot(lag_s, gam_s, "s-", ms=4, color=PALETTE[2], lw=1.8,
        label=f"Realisation (sd {draws[0].std():.1f})")
ax.plot(lag_k, gam_k, "^-", ms=4, color=PALETTE[0], lw=1.8,
        label=f"Kriged map (sd {estimate.std():.1f})")
ax.axhline(SILL, color=C_LIMIT, ls="--", lw=1.3)
ax.annotate("model sill", xy=(0.40, SILL), xycoords=("axes fraction", "data"),
            xytext=(0, 5), textcoords="offset points", fontsize=8, va="bottom",
            color=C_LIMIT)
ax.set(title="Variogram of each map",
       xlabel="Lag [m]", ylabel="Semivariance [-]")
ax.legend(fontsize=8, loc="lower right")

# %% 7. The ensemble, as an animation --------------------------------------
#
# A still can show one realisation or an average of them. Neither conveys
# what an ensemble IS - the same data, the same variogram, and a different
# earth every time. Flicking through them does.
anim_fig, anim_ax = figure(1, 1, panel_w=6.8, panel_h=3.6)
anim_image = anim_ax.imshow(draws[0].reshape(NY, NX), origin="lower",
                            extent=extent, aspect="auto", **vlim)
anim_ax.grid(False)
anim_ax.set(xlabel="x [m]", ylabel="y [m]")
anim_fig.colorbar(anim_image, ax=anim_ax, label="V [-]")
anim_ax.plot(cells[picked, 0], cells[picked, 1], ".", ms=3.5, color="white",
             mec="black", mew=0.25, ls="none")


def draw_realisation(index: int) -> None:
    anim_image.set_data(draws[index].reshape(NY, NX))
    anim_ax.set_title(f"Realisation {index + 1} of {len(draws)}")


animation(anim_fig, draw_realisation, len(draws), OUT / "01_ensemble.gif",
          fps=3)

fig.savefig(OUT / "01_estimation_vs_simulation.png")
print(f"saved {OUT / '01_estimation_vs_simulation.png'}")
plt.show()
