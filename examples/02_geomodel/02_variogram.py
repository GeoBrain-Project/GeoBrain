"""The variogram: the one input everything downstream inherits.

Kriging weights, simulation textures, and every uncertainty number in this
section are computed from a variogram model. Fit it badly and nothing
downstream can recover, which makes this the least glamorous and most
consequential step in the workflow.

A variogram answers one question: how different are two values, on
average, as a function of the vector between them?

    gamma(h) = 0.5 * mean[ (z(x + h) - z(x))^2 ]

Three numbers describe the model fitted to it. The NUGGET is the
semivariance extrapolated back to zero separation: measurement error
plus structure finer than the closest sample spacing. The SILL is where
the curve flattens, which is the field's variance. The RANGE is the lag
at which it gets there, beyond which two samples tell you nothing about
each other.

Direction matters, and omnidirectional hides it
-----------------------------------------------
Geology is rarely isotropic. Bedding, channels and structural grain all
make the range longer along one azimuth than across it, and an
omnidirectional variogram averages that away into a single compromise
curve that fits neither direction. The fix is to bin pairs by azimuth as
well as by distance: this script computes the variogram along six
azimuths, reads the range off each, and recovers the anisotropy of a
field whose true ratio and orientation are known.

The cost of the choice is measured too. A directional variogram uses only
the pairs inside its angular tolerance, so a narrow tolerance gives a
cleaner direction and a noisier curve. The script prints the pair counts
so the trade-off is visible rather than assumed.

Fitting, and what the fit will not tell you
-------------------------------------------
``fit_model_with_diagnostics`` returns the model AND a record of how it
was reached: which kernel was requested, which was selected, what was
tried, and whether a fallback was used. That record is worth keeping,
because an automatic fit that quietly fell back to a different kernel
family is a different model from the one you think you have.

The honest limit is the last panel: refit the same field from 60, 120 and
400 samples and watch the range estimate move. The variogram is estimated
from data, its uncertainty is real, and no amount of curve-fitting
machinery converts 60 samples into a known range.

APIs featured:
    - geobrain.geomodel.VariogramCalculator: omnidirectional and
      directional (azimuth, azimuth_tol), .compute and .fit
    - ExperimentalVariogram.fit_model_with_diagnostics -> VariogramFitResult
    - geobrain.geomodel.CovarianceModel / VariogramKernel for the truth

Expected runtime: < 1 min.

Outputs:
    out/02_variogram.png: the field and its samples, the omnidirectional
    variogram and its fit, the directional set, the recovered anisotropy
    rose, and the sample-count sensitivity.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import math
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
from geobrain.geomodel import (
    FFTMA,
    CovarianceModel,
    GeoFrame,
    GeoGrid,
    GeoPoints,
    PropertyMetadata,
    SimulationExecutionConfig,
    VariogramCalculator,
    VariogramKernel,
)

apply_style()
rng = np.random.default_rng(4)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

NX, NY, CELL = 70, 70, 10.0
RANGE_MAJOR, RANGE_MINOR, AZIMUTH = 220.0, 70.0, 45.0
SILL, NUGGET, MEAN = 100.0, 12.0, 50.0
N_SAMPLES = 400
PROP = PropertyMetadata("V", "continuous", "1")

# %% 1. A field whose anisotropy is known ---------------------------------
grid = GeoGrid(shape=(NX, NY), origin=(0.0, 0.0), spacing=(CELL, CELL))
truth_model = CovarianceModel(nugget=NUGGET, structures=[
    VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=SILL,
                    ranges=(RANGE_MAJOR, RANGE_MINOR, 1.0e6),
                    angles=(AZIMUTH, 0.0, 0.0))])
t0 = time.time()
field = np.asarray(
    FFTMA(truth_model, property=PROP,
          execution=SimulationExecutionConfig(n_realizations=1, seed=99),
          mean=MEAN)(None, grid).realizations[0].frame.to_numpy("simulation"),
    dtype=float).reshape(-1)
cells = np.asarray(grid.coords, dtype=float)[:, :2]
print(f"[1] {NX}x{NY} field at {CELL:.0f} m from a spherical variogram: "
      f"range {RANGE_MAJOR:.0f} m along {AZIMUTH:.0f} deg by "
      f"{RANGE_MINOR:.0f} m across (ratio "
      f"{RANGE_MAJOR / RANGE_MINOR:.1f}), sill {SILL:.0f}, nugget "
      f"{NUGGET:.0f} ({time.time() - t0:.1f} s)")


def sample(n: int, seed: int) -> GeoFrame:
    idx = np.random.default_rng(seed).choice(field.size, size=n, replace=False)
    return GeoFrame(GeoPoints(cells[idx]), properties={"V": field[idx]})


samples = sample(N_SAMPLES, 4)

# %% 2. Omnidirectional: one curve, and what it costs ---------------------
LAG = 20.0
omni = VariogramCalculator(lag_distance=LAG, n_lags=14).compute(samples, "V")
omni_fit = omni.fit_model_with_diagnostics(kind="auto")
omni_kernel = omni_fit.model.structures[0]
print(f"[2] omnidirectional over {int(np.sum(omni.pairs))} pairs: fitted "
      f"{omni_fit.diagnostics.selected_kind}, range "
      f"{omni_kernel.range_max:.0f} m, sill {omni_fit.model.sill:.0f}, nugget "
      f"{omni_fit.model.nugget:.1f}")
print(f"[2] that single range sits between the true {RANGE_MINOR:.0f} and "
      f"{RANGE_MAJOR:.0f} m, a compromise that describes no direction. "
      f"Requested '{omni_fit.diagnostics.requested_kind}', selected "
      f"'{omni_fit.diagnostics.selected_kind}', fallback "
      f"{omni_fit.diagnostics.fallback_used}")

# %% 3. Directional: bin by azimuth as well as distance -------------------
AZIMUTHS = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
TOLERANCE = 22.5
directional = {}
for azimuth in AZIMUTHS:
    exp = VariogramCalculator(lag_distance=LAG, n_lags=12, azimuth=azimuth,
                              azimuth_tol=TOLERANCE).compute(samples, "V")
    fitted = exp.fit_model(kind="spherical")
    directional[azimuth] = (exp, float(fitted.structures[0].range_max))
ranges = np.array([directional[a][1] for a in AZIMUTHS])
long_axis = AZIMUTHS[int(ranges.argmax())]
short_axis = AZIMUTHS[int(ranges.argmin())]
print(f"[3] directional at +/-{TOLERANCE:.0f} deg, {int(np.sum(directional[AZIMUTHS[0]][0].pairs))}"
      f"-{int(np.sum(directional[AZIMUTHS[3]][0].pairs))} pairs per azimuth:")
for azimuth in AZIMUTHS:
    print(f"      {azimuth:5.0f} deg -> range {directional[azimuth][1]:5.0f} m")
print(f"[3] longest at {long_axis:.0f} deg, shortest at {short_axis:.0f} deg "
      f"and the azimuths are binned every {AZIMUTHS[1] - AZIMUTHS[0]:.0f} deg, so "
      f"the true {AZIMUTH:.0f} deg is bracketed rather than hit. The ratio is "
      f"the number that matters: {ranges.max() / ranges.min():.1f} against a "
      f"true {RANGE_MAJOR / RANGE_MINOR:.1f}")

# %% 4. What the tolerance buys and costs ---------------------------------
print("[4] the angular tolerance is a trade, and it does NOT simply get "
      "better as it narrows:")
for tol in (10.0, 22.5, 45.0):
    exp = VariogramCalculator(lag_distance=LAG, n_lags=12, azimuth=AZIMUTH,
                              azimuth_tol=tol).compute(samples, "V")
    got = float(exp.fit_model(kind="spherical").structures[0].range_max)
    print(f"      +/-{tol:4.1f} deg: {int(np.sum(exp.pairs)):6d} pairs, range "
          f"{got:5.0f} m (true {RANGE_MAJOR:.0f})")
print("[4] narrowing the cone isolates the direction but starves the fit: at "
      "10 deg it sees a fifth of the pairs and overshoots badly, at 45 deg it "
      "mixes in the short axis and undershoots. The middle is not a compromise "
      "for its own sake: it is where both errors are small")

# %% 5. The limit nobody can fit their way out of -------------------------
counts = (60, 120, 400)
by_count = {}
for n in counts:
    got = [float(VariogramCalculator(lag_distance=LAG, n_lags=12,
                                     azimuth=AZIMUTH, azimuth_tol=TOLERANCE)
                 .compute(sample(n, 100 + s), "V")
                 .fit_model(kind="spherical").structures[0].range_max)
           for s in range(5)]
    by_count[n] = np.array(got)
    print(f"[5] {n:3d} samples, five independent draws: major range "
          f"{by_count[n].mean():5.0f} +/- {by_count[n].std():4.0f} m "
          f"(true {RANGE_MAJOR:.0f})")
print("[5] the spread is the variogram's own uncertainty, and it propagates "
      "into every kriging weight and every realisation downstream")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
coords = np.asarray(samples.geometry.coords, dtype=float)

ax = axes[0, 0]
im = ax.imshow(field.reshape(NY, NX), origin="lower",
               extent=(0.0, NX * CELL, 0.0, NY * CELL), aspect="auto",
               cmap=CMAP_MODEL)
ax.plot(coords[:, 0], coords[:, 1], ".", ms=2.5, color="white",
        mec="black", mew=0.2, ls="none")
arrow = 0.5 * RANGE_MAJOR
cx, cy = 0.5 * NX * CELL, 0.5 * NY * CELL
theta = math.radians(90.0 - AZIMUTH)
ax.annotate("", xy=(cx + arrow * math.cos(theta), cy + arrow * math.sin(theta)),
            xytext=(cx - arrow * math.cos(theta), cy - arrow * math.sin(theta)),
            arrowprops=dict(arrowstyle="<->", lw=2.0, color="white"))
ax.set(title=f"The field and {N_SAMPLES} samples (long axis "
             f"{AZIMUTH:.0f} deg)", xlabel="x [m]", ylabel="y [m]")
ax.grid(False)
shared_colorbar(fig, im, ax, "V [-]")

ax = axes[0, 1]
ax.plot(omni.lags, omni.gammas, "o", ms=6, color=C_TRUTH, label="Experimental")
lag_axis = np.linspace(0.0, float(np.max(omni.lags)), 200)
ax.plot(lag_axis, omni_fit.model.variogram(lag_axis), color=C_RECOVERED, lw=2.0,
        label=f"Fit: {omni_fit.diagnostics.selected_kind}, "
              f"range {omni_kernel.range_max:.0f} m")
ax.axhline(SILL + NUGGET, color=C_LIMIT, ls="--", lw=1.3)
ax.annotate("true sill + nugget", xy=(0.03, SILL + NUGGET),
            xycoords=("axes fraction", "data"), xytext=(0, 4),
            textcoords="offset points", fontsize=8, va="bottom",
            color=C_LIMIT)
ax.set(title="Omnidirectional", xlabel="Lag [m]",
       ylabel="Semivariance [-]")
ax.legend(fontsize=8, loc="lower right")

ax = axes[0, 2]
colours = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(AZIMUTHS)))
for azimuth, colour in zip(AZIMUTHS, colours):
    exp, got = directional[azimuth]
    ax.plot(exp.lags, exp.gammas, "o-", ms=4, lw=1.5, color=colour,
            label=f"{azimuth:.0f} deg  ({got:.0f} m)")
ax.set(title=f"Directional, +/-{TOLERANCE:.1f} deg tolerance",
       xlabel="Lag [m]", ylabel="Semivariance [-]")
ax.legend(fontsize=7.5, ncol=2)

ax = axes[1, 0]
angles = np.radians(np.array(AZIMUTHS + [a + 180.0 for a in AZIMUTHS]))
radii = np.concatenate([ranges, ranges])
order = np.argsort(angles)
ax = fig.add_subplot(2, 3, 4, projection="polar")
axes[1, 0].remove()
ax.plot(np.append(angles[order], angles[order][0]),
        np.append(radii[order], radii[order][0]), "o-", color=PALETTE[0],
        lw=2.0, label="Fitted range")
for a, r, style in ((AZIMUTH, RANGE_MAJOR, "-"), (AZIMUTH + 90.0, RANGE_MINOR, "-")):
    ax.plot([math.radians(a), math.radians(a + 180.0)], [r, r], style,
            color=C_TRUTH, lw=2.2)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_title(f"Range by azimuth (recovered "
             f"{ranges.max() / ranges.min():.1f}, true "
             f"{RANGE_MAJOR / RANGE_MINOR:.1f})", fontsize=11)
ax.tick_params(labelsize=8)

ax = axes[1, 1]
positions = np.arange(len(counts))
ax.boxplot([by_count[n] for n in counts], positions=positions, widths=0.55)
ax.axhline(RANGE_MAJOR, color=C_TRUTH, ls="--", lw=1.6, label="True range")
ax.set(title="Five independent draws at each sample count",
       xlabel="Samples", ylabel="Fitted major range [m]",
       ylim=(0.0, 1.15 * float(max(v.max() for v in by_count.values()))))
ax.set_xticks(positions)
ax.set_xticklabels([str(n) for n in counts])
ax.legend(fontsize=8)

ax = axes[1, 2]
ax.axis("off")
lines = ["What the fit record says", ""]
lines.append(f"requested kind   {omni_fit.diagnostics.requested_kind}")
lines.append(f"selected kind    {omni_fit.diagnostics.selected_kind}")
lines.append(f"fallback used    {omni_fit.diagnostics.fallback_used}")
lines.append(f"kernels tried    {len(omni_fit.diagnostics.attempts)}")
lines.append("")
lines.append(f"nugget   {omni_fit.model.nugget:6.1f}   (true {NUGGET:.0f})")
lines.append(f"sill     {omni_fit.model.sill:6.1f}   (true {SILL + NUGGET:.0f})")
lines.append(f"range    {omni_kernel.range_max:6.0f}   (true "
             f"{RANGE_MINOR:.0f}-{RANGE_MAJOR:.0f})")
lines.append("")
lines.append("An automatic fit that quietly fell back")
lines.append("to another kernel family is a different")
lines.append("model from the one you think you have.")
ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
        family="monospace", transform=ax.transAxes)

fig.savefig(OUT / "02_variogram.png")
print(f"saved {OUT / '02_variogram.png'}")
plt.show()
