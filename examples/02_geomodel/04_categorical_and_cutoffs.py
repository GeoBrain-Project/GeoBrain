"""Categories and cutoffs, when the question is not "how much".

Two questions in this section are not about a value at all.

    "Is this cell above the cutoff?" covers the mine plan, the net-pay flag,
        the contamination threshold. The answer is a PROBABILITY, and it
        is not obtained by kriging the grade and comparing the estimate
        to the cutoff. A kriged estimate is smoothed, so thresholding it
        under-reports both the high tail and the low one.

    "Which rock is this?" covers facies, lithology, alteration. The answer is
        a LABEL, and a label has no arithmetic. Averaging sandstone and
        shale gives neither.

Indicator kriging answers the first. It transforms each sample into a 0/1
indicator at the cutoff and krige THAT, which estimates the conditional
probability directly and never smooths a grade it then thresholds.

Categorical sequential indicator simulation answers the second. It draws
labels cell by cell from the local conditional distribution, so every
realisation is a legal geology, with hard boundaries and no intermediate
rock, and the ensemble reproduces the target proportions.

What is measured
----------------
For the cutoff question the script compares two ways of getting a
probability map (indicator kriging, and thresholding an ordinary-kriged
grade) against the truth's own indicator, and scores both with a Brier
score. Thresholding the smoothed estimate is not a small approximation:
it is badly calibrated, over-flagging the map while the indicator route
lands within a few points of the true proportion above.

For the facies question it checks that the realisations honour the target
proportions, and reports the classification accuracy of the ensemble's
most-likely facies against the true geology.

A note on the contract
----------------------
``IndicatorKriging`` refuses a categorical property, and it is right to.
Thresholds are an ordering, categories have none, and a library that
accepted both silently would let you krige rock names. Facies go through
``SISIM(..., categorical=True)`` instead, and that path requires the
frame to carry explicit ``PropertyMetadata`` with its ``Category`` list:
the labels are part of the data, not a convention in the caller's head.

APIs featured:
    - geobrain.geomodel.IndicatorKriging on a continuous variable
    - geobrain.geomodel.SISIM(categorical=True) with marginal_probs
    - geobrain.geomodel.PropertyMetadata + Category as the facies contract

Expected runtime: < 4 min, most of it the categorical simulation.

Outputs:
    out/04_categorical_and_cutoffs.png: the truth above cutoff, both
    probability maps with their scores, the true facies against one
    categorical realisation, and the proportion check.

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
    apply_style,
    category_colorbar,
    category_norm,
    figure,
    shared_colorbar,
)
from geobrain.geomodel import (
    FFTMA,
    SISIM,
    Category,
    CovarianceModel,
    GeoFrame,
    GeoGrid,
    GeoPoints,
    IndicatorKriging,
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

NX, NY, CELL = 48, 48, 12.0
RANGE_M, SILL, MEAN = 140.0, 1.0, 0.0
N_SAMPLES, N_REAL = 180, 8
FACIES = (Category(0, "shale"), Category(1, "sand"), Category(2, "channel"))
BREAKS = (-0.5, 0.7)               # Gaussian breaks -> three facies
GRADE = PropertyMetadata("grade", "continuous", "1")
LITHO = PropertyMetadata("facies", "categorical", None, categories=FACIES)

# %% 1. One earth, read two ways ------------------------------------------
grid = GeoGrid(shape=(NX, NY), origin=(0.0, 0.0), spacing=(CELL, CELL))
cells = np.asarray(grid.coords, dtype=float)[:, :2]
gaussian = np.asarray(
    FFTMA(CovarianceModel(nugget=0.0, structures=[
        VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=SILL,
                        ranges=(RANGE_M, 0.6 * RANGE_M, 1.0e6),
                        angles=(60.0, 0.0, 0.0))]), property=GRADE,
        execution=SimulationExecutionConfig(n_realizations=1, seed=5),
        mean=MEAN)(None, grid).realizations[0].frame.to_numpy("simulation"),
    dtype=float).reshape(-1)
grade = np.exp(0.55 * gaussian)                      # a skewed grade
facies = np.digitize(gaussian, BREAKS).astype(float)  # the same earth, labelled
proportions = np.array([float(np.mean(facies == k)) for k in range(3)])
print(f"[1] {NX}x{NY} at {CELL:.0f} m from one Gaussian field, read twice: a "
      f"skewed grade {grade.min():.2f} to {grade.max():.2f}, and three facies "
      "cut from the same field at fixed breaks")
print("[1] true proportions " + ", ".join(
    f"{c.label} {100 * proportions[c.code]:.0f}%" for c in FACIES)
    )

rng = np.random.default_rng(23)
picked = rng.choice(grade.size, size=N_SAMPLES, replace=False)
grade_samples = GeoFrame(GeoPoints(cells[picked]),
                         properties={"grade": grade[picked]},
                         metadata={"grade": GRADE})
facies_samples = GeoFrame(GeoPoints(cells[picked]),
                          properties={"facies": facies[picked]},
                          metadata={"facies": LITHO})
print(f"[1] {N_SAMPLES} samples; sampled proportions " + ", ".join(
    f"{100 * np.mean(facies[picked] == c.code):.0f}%" for c in FACIES))

# %% 2. The cutoff question, two ways -------------------------------------
fit = VariogramCalculator(n_lags=14).fit(grade_samples, "grade", kind="auto")
radius = 2.0 * float(fit.structures[0].range_max)
neighbourhood = NeighbourhoodSpec(radii_m=(radius, radius), angles_deg=(0.0,),
                                  min_neighbors=1, max_neighbors=24)

# Indicator kriging estimates a whole conditional CDF, so it wants SEVERAL
# thresholds with a variogram fitted to each indicator: the indicator of a
# high cutoff is a different, patchier field than that of a low one, and one
# variogram cannot serve both. The library refuses a single threshold.
THRESHOLDS = [float(np.quantile(grade[picked], q)) for q in (0.4, 0.6, 0.8)]
CUTOFF = THRESHOLDS[1]
above = grade > CUTOFF
indicator_fits = []
for level in THRESHOLDS:
    ind = GeoFrame(GeoPoints(cells[picked]),
                   properties={"ind": (grade[picked] > level).astype(float)},
                   metadata={"ind": PropertyMetadata("ind", "continuous", "1")})
    indicator_fits.append(
        VariogramCalculator(n_lags=14).fit(ind, "ind", kind="auto"))
print("[2] indicator variograms, one per threshold:")
for level, ifit in zip(THRESHOLDS, indicator_fits):
    print(f"      cutoff {level:5.2f} ({100 * np.mean(grade > level):4.1f}% of "
          f"the map above): range {ifit.structures[0].range_max:5.0f} m")
print(f"[2] the grade's own variogram range is "
      f"{fit.structures[0].range_max:.0f} m, a different number again, which "
      "is why the indicator gets its own")

t0 = time.time()
ik = IndicatorKriging(indicator_fits, THRESHOLDS, property=GRADE,
                      neighbourhood=neighbourhood)(grade_samples, grid)
prob_columns = [c for c in ik.columns if c.startswith("prob")]
ik_column = prob_columns[1]
# IMPORTANT CONVENTION: the operator returns a CDF. Its indicator is built as
# (values <= threshold), so prob_i is P(Z <= z_i), the probability of being
# BELOW. Reading it as "above" inverts the map and still produces a plausible
# picture, which is why the complement is taken here explicitly.
cdf_below = np.clip(np.asarray(ik[ik_column], dtype=float).reshape(-1), 0.0, 1.0)
p_indicator = 1.0 - cdf_below
print(f"[2] indicator kriging in {time.time() - t0:.1f} s -> columns "
      f"{prob_columns}; reading '{ik_column}' at the {CUTOFF:.2f} cutoff as a "
      f"CDF and taking its complement (mean P(above) "
      f"{p_indicator.mean():.2f} against a true {above.mean():.2f})")

t0 = time.time()
kriged_grade = np.asarray(
    OrdinaryKriging(fit, property=GRADE,
                    neighbourhood=neighbourhood)(grade_samples,
                                                 grid)["estimate"],
    dtype=float).reshape(-1)
p_threshold = (kriged_grade > CUTOFF).astype(float)
print(f"[2] and the shortcut: krige the grade ({time.time() - t0:.1f} s) and "
      "threshold the estimate")


def score(probability: np.ndarray, label: str) -> tuple[float, float]:
    """Brier score, and the share of truly-above cells the map finds."""
    brier = float(np.mean((probability - above.astype(float)) ** 2))
    recall = float(np.mean(probability[above] > 0.5))
    print(f"[3] {label:26s} Brier {brier:.4f}, calls "
          f"{100 * recall:4.0f}% of the truly-above cells above, and predicts "
          f"{100 * np.mean(probability > 0.5):4.0f}% of the map above "
          f"(truth {100 * above.mean():.0f}%)")
    return brier, recall


brier_ik, recall_ik = score(p_indicator, "indicator kriging")
brier_th, recall_th = score(p_threshold, "threshold the estimate")
print(f"[3] indicator kriging is better CALIBRATED: Brier "
      f"{brier_ik:.3f} against {brier_th:.3f}, "
      f"{100 * (1 - brier_ik / brier_th):.0f}% lower, and it calls "
      f"{100 * np.mean(p_indicator > 0.5):.0f}% of the map above against a "
      f"true {100 * above.mean():.0f}%, where the shortcut calls "
      f"{100 * np.mean(p_threshold > 0.5):.0f}%")
print(f"[3] the shortcut does have the higher RECALL "
      f"({100 * recall_th:.0f}% against {100 * recall_ik:.0f}%), but it buys "
      "that by over-calling: a map that flags more of everything finds more "
      "of the truly-above cells and more of the truly-below ones too. Recall "
      "alone cannot separate a good probability map from a generous one, "
      "which is what the Brier score is for")

# %% 3. The facies question -----------------------------------------------
facies_models = []
for code in range(3):
    ind = GeoFrame(GeoPoints(cells[picked]),
                   properties={"i": (facies[picked] == code).astype(float)},
                   metadata={"i": PropertyMetadata("i", "continuous", "1")})
    facies_models.append(VariogramCalculator(n_lags=12).fit(ind, "i",
                                                            kind="auto"))
t0 = time.time()
ensemble = SISIM(facies_models, [float(c.code) for c in FACIES],
                 property=LITHO, neighbourhood=neighbourhood,
                 execution=SimulationExecutionConfig(n_realizations=N_REAL,
                                                     seed=3),
                 categorical=True,
                 marginal_probs=tuple(float(p) for p in proportions))(
    facies_samples, grid)
draws = np.stack([np.asarray(r.frame["simulation"], dtype=float).reshape(-1)
                  for r in ensemble.realizations])
print(f"[4] {N_REAL} categorical realisations in {time.time() - t0:.0f} s; "
      "each is a legal geology with hard boundaries")
for c in FACIES:
    got = np.array([float(np.mean(d == c.code)) for d in draws])
    print(f"      {c.label:8s} target {100 * proportions[c.code]:5.1f}%  "
          f"realisations {100 * got.mean():5.1f}% +/- {100 * got.std():.1f}")

votes = np.stack([(draws == c.code).mean(axis=0) for c in FACIES])
most_likely = votes.argmax(axis=0).astype(float)
accuracy = float(np.mean(most_likely == facies))
print(f"[4] the ensemble's most-likely facies matches the truth in "
      f"{100 * accuracy:.0f}% of cells, and unlike a kriged map it is a rock "
      "name everywhere, which is what the next step in the workflow needs")

# %% 4. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
extent = (0.0, NX * CELL, 0.0, NY * CELL)


def show(ax, field, title, style, label=None):
    im = ax.imshow(np.asarray(field).reshape(NY, NX), origin="lower",
                   extent=extent, aspect="auto", **style)
    ax.set(title=title, xlabel="x [m]", ylabel="y [m]")
    ax.grid(False)
    return im


# Three maps on one 0-1 scale and one bar: the whole comparison is which
# of the two estimators is closer to the truth panel beside them.
probability = dict(cmap=CMAP_MODEL, vmin=0.0, vmax=1.0)
im = show(axes[0, 0], above.astype(float), f"Truth: grade > {CUTOFF:.2f}",
          probability)
axes[0, 0].plot(cells[picked, 0], cells[picked, 1], ".", ms=3.0, color="white",
                mec="black", mew=0.2, ls="none")
show(axes[0, 1], p_indicator,
     f"Indicator kriging\nBrier {brier_ik:.4f}, recall {100 * recall_ik:.0f}%",
     probability)
show(axes[0, 2], p_threshold,
     f"Krige then threshold\nBrier {brier_th:.4f}, recall "
     f"{100 * recall_th:.0f}%", probability)
shared_colorbar(fig, im, axes[0, :], "P(above cutoff) [-]",
                location="bottom")

# A facies is a NAME, not a number: discrete colours and a bar that
# names them, rather than a ramp inviting the reader to interpolate.
_facies_cmap, _facies_norm = category_norm(3)
facies_style = dict(cmap=_facies_cmap, norm=_facies_norm)
im = show(axes[1, 0], facies, "True facies", facies_style)
show(axes[1, 1], draws[0], "One realisation - hard boundaries",
     facies_style)
category_colorbar(fig, im, axes[1, :2], "Facies",
                  [c.label for c in FACIES], location="bottom")

ax = axes[1, 2]
width = 0.35
positions = np.arange(3)
ax.bar(positions - width / 2, 100 * proportions, width, color=PALETTE[0],
       label="Target")
got = np.array([[float(np.mean(d == c.code)) for c in FACIES] for d in draws])
ax.bar(positions + width / 2, 100 * got.mean(axis=0), width,
       yerr=100 * got.std(axis=0), capsize=4, color=PALETTE[2],
       label=f"{N_REAL} realisations")
ax.set(title="Facies proportions, target vs realisations",
       xlabel="Facies", ylabel="Share of cells [%]", xticks=positions)
ax.set_xticklabels([c.label for c in FACIES])
ax.legend(fontsize=8)

fig.savefig(OUT / "04_categorical_and_cutoffs.png")
print(f"saved {OUT / '04_categorical_and_cutoffs.png'}")
plt.show()
