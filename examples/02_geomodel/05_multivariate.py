"""A cheap variable everywhere beats an expensive one nowhere.

The usual situation is not "too little data". It is data of two kinds:
the variable you care about, sampled expensively and sparsely (assays,
core plugs, well logs), and a proxy sampled cheaply and densely (a
geophysical attribute, a satellite band, a survey grid).

Kriging the primary alone throws the proxy away. Cokriging keeps it, and
the question this script answers is how much that is worth, in the only
terms that matter: error against a known truth.

Collocated cokriging is the practical form. It uses the secondary value
AT THE CELL BEING ESTIMATED and nowhere else, which is exactly the case
where the secondary is known everywhere, and it needs one number to
describe the relationship: the primary-secondary correlation. Under the
Markov screening assumption the full cross-variogram is not required,
which is what makes it usable on real projects.

What is measured
----------------
The same 40 primary samples are used three ways: ordinary kriging on the
primary alone, collocated cokriging with the dense secondary, and
co-simulation which adds the variability that both kriged maps smooth
away. All three are scored against the truth, and the improvement is
reported as a percentage of the kriging-only error.

The script also repeats the comparison at several correlation strengths.
The measured answer on this field is that the proxy pays at every one of
them, with a return that grows from a few percent at 0.3 to more than a
quarter at 0.9, because collocated cokriging weights the secondary by the
correlation it is given, so a weak proxy earns a small weight rather than
doing harm. The correlation supplied to the estimator is the one MEASURED
from the co-located samples, not the one used to build the truth, because
on a real project that is all you have, and the estimator will believe
whatever number you hand it.

APIs featured:
    - geobrain.geomodel.CollocatedCokriging with the Markov assumption
    - geobrain.geomodel.CoSGSIM for the co-simulated ensemble
    - geobrain.geomodel.OrdinaryKriging as the primary-only benchmark

Expected runtime: < 3 min.

Outputs:
    out/05_multivariate.png: the truth with its sparse samples, the two
    recoveries, the dense secondary, the correlation sweep, and the
    primary-secondary cross-plot.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _style import (
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
    CollocatedCokriging,
    CoSGSIM,
    CovarianceModel,
    GeoFrame,
    GeoGrid,
    GeoPoints,
    NeighbourhoodSpec,
    OrdinaryKriging,
    PropertyMetadata,
    SimulationExecutionConfig,
    VariogramCalculator,
    VariogramKernel,
)

apply_style()
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

NX, NY, CELL = 44, 44, 12.0
RANGE_M, SILL = 150.0, 1.0
N_PRIMARY, N_REAL = 40, 8
CORRELATION = 0.85
PRIMARY = PropertyMetadata("primary", "continuous", "1")
SECONDARY = PropertyMetadata("secondary", "continuous", "1")

# %% 1. One structure, two variables --------------------------------------
grid = GeoGrid(shape=(NX, NY), origin=(0.0, 0.0), spacing=(CELL, CELL))
cells = np.asarray(grid.coords, dtype=float)[:, :2]
model = CovarianceModel(nugget=0.0, structures=[
    VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=SILL,
                    ranges=(RANGE_M, RANGE_M, 1.0e6),
                    angles=(0.0, 0.0, 0.0))])


def draw(seed: int) -> np.ndarray:
    return np.asarray(
        FFTMA(model, property=PRIMARY,
              execution=SimulationExecutionConfig(n_realizations=1, seed=seed),
              mean=0.0)(None, grid).realizations[0].frame
        .to_numpy("simulation"), dtype=float).reshape(-1)


base, independent = draw(12), draw(77)
truth = base
secondary = (CORRELATION * base
             + np.sqrt(1.0 - CORRELATION ** 2) * independent)
rng = np.random.default_rng(9)
picked = rng.choice(truth.size, size=N_PRIMARY, replace=False)
print(f"[1] {NX}x{NY} at {CELL:.0f} m. Primary and secondary share one "
      f"spherical structure of range {RANGE_M:.0f} m, mixed to a true "
      f"correlation of {CORRELATION:.2f}")
print(f"[1] the secondary is known at ALL {truth.size} cells; the primary at "
      f"{N_PRIMARY} ({100 * N_PRIMARY / truth.size:.1f}%)")

# Collocated cokriging wants the secondary in BOTH places: on the samples, to
# measure the cross-relationship, and at every target, to use it there. The
# conditioning frame therefore carries two columns, not one.
primary_samples = GeoFrame(GeoPoints(cells[picked]),
                           properties={"primary": truth[picked],
                                       "secondary": secondary[picked]},
                           metadata={"primary": PRIMARY,
                                     "secondary": SECONDARY})
domain = GeoFrame(grid, properties={"secondary": secondary},
                  metadata={"secondary": SECONDARY})
sample_correlation = float(np.corrcoef(truth[picked], secondary[picked])[0, 1])
print(f"[1] correlation measured from the {N_PRIMARY} co-located samples: "
      f"{sample_correlation:.2f}, and that, not the true {CORRELATION:.2f}, is "
      "what the estimator is given")

# %% 2. Three ways to use the same 40 samples -----------------------------
fit = VariogramCalculator(n_lags=12).fit(primary_samples, "primary",
                                         kind="auto")
radius = 2.0 * float(fit.structures[0].range_max)
neighbourhood = NeighbourhoodSpec(radii_m=(radius, radius), angles_deg=(0.0,),
                                  min_neighbors=1, max_neighbors=24)


def rmse(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean((field - truth) ** 2)))


t0 = time.time()
kriged = np.asarray(
    OrdinaryKriging(fit, property=PRIMARY,
                    neighbourhood=neighbourhood)(primary_samples,
                                                 grid)["estimate"],
    dtype=float).reshape(-1)
print(f"[2] ordinary kriging, primary only:  RMSE {rmse(kriged):.4f}  "
      f"({time.time() - t0:.1f} s)")

t0 = time.time()
cokriged = np.asarray(
    CollocatedCokriging(fit, sample_correlation, primary_property=PRIMARY,
                        secondary_property=SECONDARY, kind="ordinary",
                        neighbourhood=neighbourhood)(primary_samples,
                                                     domain)["estimate"],
    dtype=float).reshape(-1)
print(f"[2] collocated cokriging:             RMSE {rmse(cokriged):.4f}  "
      f"({time.time() - t0:.1f} s)")

t0 = time.time()
cosim = CoSGSIM(fit, sample_correlation, property=PRIMARY,
                secondary_property=SECONDARY, neighbourhood=neighbourhood,
                execution=SimulationExecutionConfig(n_realizations=N_REAL,
                                                    seed=6))(primary_samples,
                                                             domain)
draws = np.stack([np.asarray(r.frame["simulation"], dtype=float).reshape(-1)
                  for r in cosim.realizations])
cosim_mean = draws.mean(axis=0)
print(f"[2] co-simulation, {N_REAL} realisations: RMSE "
      f"{np.mean([rmse(d) for d in draws]):.4f} each, {rmse(cosim_mean):.4f} "
      f"for the ensemble mean ({time.time() - t0:.0f} s)")
print(f"[3] the secondary is worth "
      f"{100 * (1 - rmse(cokriged) / rmse(kriged)):.0f}% of the "
      "primary-only error, from data that was already on the shelf")
print(f"[3] and the usual trade returns: a single co-simulated realisation is "
      f"worse than the cokriged map cell for cell, but carries variance "
      f"{draws.var(axis=1).mean():.3f} against the cokriged map's "
      f"{cokriged.var():.3f} and a true {truth.var():.3f}")

# %% 3. When is a proxy worth having? -------------------------------------
print("[4] the same comparison at several true correlations:")
sweep = {}
for rho in (0.3, 0.5, 0.7, 0.9):
    sec = rho * base + np.sqrt(1.0 - rho ** 2) * independent
    rho_hat = float(np.corrcoef(truth[picked], sec[picked])[0, 1])
    dom = GeoFrame(grid, properties={"secondary": sec},
                   metadata={"secondary": SECONDARY})
    pairs = GeoFrame(GeoPoints(cells[picked]),
                     properties={"primary": truth[picked],
                                 "secondary": sec[picked]},
                     metadata={"primary": PRIMARY, "secondary": SECONDARY})
    got = np.asarray(
        CollocatedCokriging(fit, rho_hat, primary_property=PRIMARY,
                            secondary_property=SECONDARY, kind="ordinary",
                            neighbourhood=neighbourhood)(pairs,
                                                         dom)["estimate"],
        dtype=float).reshape(-1)
    sweep[rho] = rmse(got)
    print(f"      true rho {rho:.1f} (measured {rho_hat:+.2f}): RMSE "
          f"{sweep[rho]:.4f}, {100 * (1 - sweep[rho] / rmse(kriged)):+5.1f}% "
          "against primary-only")
gains = {r: 100 * (1 - sweep[r] / rmse(kriged)) for r in sweep}
print("[4] on this field the proxy pays at EVERY correlation tested: "
      + ", ".join(f"{r:.1f} -> {gains[r]:+.0f}%" for r in sorted(gains))
      + ". There is no threshold to cross here, only a return that grows "
      "with the correlation, because collocated cokriging weights the "
      "secondary by the correlation it is told and a weak proxy simply gets "
      "a small weight")
print("[4] what that does NOT license is trusting a correlation you did not "
      "measure: the estimator believes the number you hand it, so the honest "
      "input is the one computed from your own co-located pairs, with as "
      "many pairs as you can find")

# %% 4. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
extent = (0.0, NX * CELL, 0.0, NY * CELL)
vlim = dict(vmin=float(truth.min()), vmax=float(truth.max()), cmap=CMAP_MODEL)


def show(ax, field, title, style, label=None):
    im = ax.imshow(np.asarray(field).reshape(NY, NX), origin="lower",
                   extent=extent, aspect="auto", **style)
    ax.set(title=title, xlabel="x [m]", ylabel="y [m]")
    ax.grid(False)
    return im


im = show(axes[0, 0], truth, "Truth (the primary)", vlim)
axes[0, 0].plot(cells[picked, 0], cells[picked, 1], "o", ms=4.5,
                color="white", mec="black", mew=0.6, ls="none",
                label=f"{N_PRIMARY} samples")
axes[0, 0].legend(fontsize=8, loc="lower right", framealpha=0.85)
show(axes[0, 1], kriged, f"Kriging, primary only\nRMSE {rmse(kriged):.4f}",
     vlim)
show(axes[0, 2], cokriged,
     f"Collocated cokriging\nRMSE {rmse(cokriged):.4f}", vlim)
shared_colorbar(fig, im, axes[0, :], "primary [-]", location="bottom")

im2 = show(axes[1, 0], secondary,
           f"Secondary, known everywhere\ntrue correlation {CORRELATION:.2f}",
           dict(cmap=CMAP_MODEL))
shared_colorbar(fig, im2, axes[1, 0], "secondary [-]", location="bottom")

ax = axes[1, 1]
rhos = sorted(sweep)
ax.plot(rhos, [sweep[r] for r in rhos], "o-", color=PALETTE[0], lw=2.0,
        label="Collocated cokriging")
ax.axhline(rmse(kriged), color=C_TRUTH, ls="--", lw=1.6,
           label="Primary only")
ax.set(title="RMSE by true correlation",
       xlabel="True primary-secondary correlation", ylabel="RMSE [-]")
ax.legend(fontsize=8)

ax = axes[1, 2]
ax.plot(secondary[picked], truth[picked], "o", ms=5, color=PALETTE[0],
        mec="black", mew=0.4, ls="none",
        label=f"{N_PRIMARY} co-located samples")
line = np.linspace(float(secondary.min()), float(secondary.max()), 20)
ax.plot(line, sample_correlation * line, color=C_RECOVERED, lw=2.0,
        label=f"slope = measured rho {sample_correlation:.2f}")
ax.set(title="The one number cokriging needs",
       xlabel="Secondary [-]", ylabel="Primary [-]")
ax.legend(fontsize=8)

fig.savefig(OUT / "05_multivariate.png")
print(f"saved {OUT / '05_multivariate.png'}")
plt.show()
