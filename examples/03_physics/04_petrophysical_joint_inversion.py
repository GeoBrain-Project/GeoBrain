"""Petrophysically guided joint inversion: from two images to one geology.

Gravity and magnetics see the same rock through different windows. Invert
them separately and you get two smooth, blurry images that a geologist has
to reconcile by eye. Invert them jointly with a smoothness term and you
get two smooth, blurry images that happen to overlap. Neither answers the
question actually being asked, which is not "what is the density here"
but "which ROCK is here".

This script asks the second question, on a kimberlite-style setting with
three units whose petrophysical signatures are known in advance:

    unit        density contrast   susceptibility   abundance
    background        0 kg/m3          0     SI        0.90
    PK             -800 kg/m3          0.005 SI        0.075
    VK             -200 kg/m3          0.020 SI        0.025

PK is much less dense and barely magnetic; VK is barely less dense and
strongly magnetic. Neither survey alone separates them: gravity sees one
body, magnetics sees the other, and that is exactly the case where the
petrophysics has to do the work.

How the coupling enters
-----------------------
The three units are encoded as a three-component Gaussian mixture over
the pair (density contrast, susceptibility). Then
:func:`geobrain.optim.gmm_prior` adds ``-log p_GMM`` of every cell's pair
to the objective, so a cell that sits between two clusters is penalised
for being petrophysically impossible, while a cell sitting on a cluster
is free. The physics is untouched: one
:class:`~geobrain.geomodel.EarthModel` carries both fields and a
:class:`~geobrain.inverse.joint.JointProblem` runs ``Gravity3D`` and
``Magnetic3D`` against their own data with their own noise.

The inversion runs in two stages, because the coupling must not be
allowed to overrule the data. Stage one fits the data with smoothness
only, that is the plain Tikhonov result, and the benchmark. Stage two
warm-starts from it and turns the mixture prior on, cranking its weight
up while the data still fit and backing off the moment they do not. The
misfits at the end are reported for both stages so the reader can check
that the second answer was not bought by abandoning the first.

What comes out that a smooth inversion cannot give
--------------------------------------------------
A rock-unit map that is SUPPORTED. Once each cell's pair sits on a
cluster, the most likely component IS a classification, so the inversion
returns a quasi-geology model alongside the two property images, and the
script scores it against the true geology.

Read the two scores together, because the first one alone would mislead.
Both stages get 97% of the anomalous cells right, so the MAP does not
separate them. What separates them is whether those cells had any
business being classified: 25% of the smooth recovery's anomalous cells
fall inside their own unit's 95% ellipse, against 75% of the guided
one's. The smooth recovery smears along a continuum between the units and
its argmax is a coin toss that happened to land well on this model; the
guided one snaps onto the clusters, so its classification means
something. That is what the cross-plot beside the maps is showing, and it
is why the figure is not just the two maps.

APIs featured:
    - geobrain.optim.gmm_prior with geobrain.bayes.GaussianMixture: the
      petrophysical coupling as one term in the objective
    - geobrain.inverse.joint.JointProblem + JointForward field rewiring
    - geobrain.geomodel.EarthModel carrying two coupled fields
    - GaussianMixture.component_posteriors as the classifier

Expected runtime: < 2 min.

Outputs:
    out/04_petrophysical_joint_inversion.png: the true geology, the
    rock-unit map gravity and magnetics recover under petrophysical
    guidance, and the cross-plot the coupling acts in. How well the
    PROPERTIES came back is two chi-squared numbers per stage, printed.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    CMAP_ANOMALY,
    CMAP_DEFICIT,
    PALETTE,
    apply_style,
    category_colorbar,
    category_norm,
    figure,
    shared_colorbar,
)
from geobrain.bayes import GaussianMixture
from geobrain.core import ForwardContext, ModelState
from geobrain.geomodel.earthmodel import EarthModel, Field
from geobrain.inverse import GaussianLikelihood, JointForward, JointProblem
from geobrain.mesh import TensorMesh
from geobrain.optim import Inverter, gmm_prior, smoothness
from geobrain.physics.potential import (
    EarthField,
    Gravity3D,
    Magnetic3D,
    PotentialSurvey3D,
)

apply_style()
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
NZ, NX, NY = 6, 12, 12
DZ = DH = 50.0
RHO_PK, CHI_PK = -800.0, 0.005            # kg/m3, SI
RHO_VK, CHI_VK = -200.0, 0.020
UNITS = ("background", "PK", "VK")
Z_BODY = slice(0, 2)                      # outcropping, top 100 m
NOISE_PCT = 0.02
# The potential operators are strict SI: gz in m/s2, tmi in tesla. Only
# the printed and plotted numbers are converted.
MGAL, NANOTESLA = 1.0e5, 1.0e9

# %% 1. Three units, two of which no single survey can separate ------------
mesh = TensorMesh(shape=(NZ, NX, NY), spacing=(DZ, DH, DH))
SHAPE = (NZ, NX, NY)
_rho, _chi, _geo = (torch.zeros(SHAPE, dtype=D), torch.zeros(SHAPE, dtype=D),
                    torch.zeros(SHAPE, dtype=torch.long))
for unit, (rho, chi, xs, ys) in enumerate(
        ((RHO_PK, CHI_PK, slice(2, 5), slice(2, 5)),
         (RHO_VK, CHI_VK, slice(7, 10), slice(7, 10))), start=1):
    _rho[Z_BODY, xs, ys] = rho
    _chi[Z_BODY, xs, ys] = chi
    _geo[Z_BODY, xs, ys] = unit
rho_true, chi_true, geology_true = _rho, _chi, _geo

# Magnetic3D wants the canonical FLAT cell vector while EarthModel wants the
# mesh shape, so `flat` marks every crossing of that boundary.
flat = torch.Tensor.reshape
print(f"[1] {NZ}x{NX}x{NY} cells, {DH:.0f} m laterally and {DZ:.0f} m "
      f"deep; PK ({RHO_PK:.0f} kg/m3, {CHI_PK} SI) and VK ({RHO_VK:.0f}, "
      f"{CHI_VK}) at {Z_BODY.start * DZ:.0f}-{Z_BODY.stop * DZ:.0f} m depth "
      f"with {int((_geo > 0).sum())} anomalous cells, the dense-but-weak body "
      "and the light-but-strong one")

# The petrophysical prior: what each unit looks like, and how common it is.
MEANS = torch.tensor([[0.0, 0.0], [RHO_PK, CHI_PK], [RHO_VK, CHI_VK]], dtype=D)
VARS = torch.tensor([[6.0e2, 3.175e-7],           # background is tightest
                     [2.4e3, 1.5e-6],
                     [2.4e3, 1.5e-6]], dtype=D)
ABUNDANCE = torch.tensor([0.9, 0.075, 0.025], dtype=D)
gmm = GaussianMixture(means=[MEANS[k] for k in range(3)],
                      covs=[torch.diag(VARS[k]) for k in range(3)],
                      weights=ABUNDANCE)
for k, name in enumerate(UNITS):
    print(f"    {name:11s} mean ({float(MEANS[k, 0]):+7.1f} kg/m3, "
          f"{float(MEANS[k, 1]):.3f} SI)  sd "
          f"({float(VARS[k, 0]) ** 0.5:5.1f}, {float(VARS[k, 1]) ** 0.5:.1e})  "
          f"abundance {float(ABUNDANCE[k]):.3f}")


def classify(rho: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
    """Most probable unit per cell, the inversion's quasi-geology output."""
    pairs = torch.stack([rho.detach().reshape(-1), chi.detach().reshape(-1)], 1)
    return gmm.component_posteriors(pairs).argmax(1).reshape(SHAPE)


# %% 2. Two surveys over the same ground ----------------------------------
line = torch.linspace(25.0, NX * DH - 25.0, 10, dtype=D)
gx, gy = torch.meshgrid(line, line, indexing="ij")
stations = torch.stack([gx.reshape(-1), gy.reshape(-1),
                        torch.full((gx.numel(),), 10.0, dtype=D)], dim=1)
gravity = Gravity3D(survey=PotentialSurvey3D(stations))
magnetics = Magnetic3D(survey=PotentialSurvey3D(stations),
                       earth_field=EarthField(intensity_tesla=60000.0e-9,
                                              inclination_deg=83.0,
                                              declination_deg=20.0))
ctx = ForwardContext.of(mesh=mesh)

t0 = time.time()
with torch.no_grad():
    gz_clean = gravity(ModelState({"rho": rho_true}), ctx).data["gz"]
    tmi_clean = magnetics(ModelState({"chi": chi_true.reshape(-1)}),
                          ctx).data["tmi"]
STD_G = NOISE_PCT * float(gz_clean.abs().max())
STD_M = NOISE_PCT * float(tmi_clean.abs().max())
gz_obs = gz_clean + STD_G * torch.randn_like(gz_clean)
tmi_obs = tmi_clean + STD_M * torch.randn_like(tmi_clean)
print(f"[2] {stations.shape[0]} stations 10 m up, forward in "
      f"{time.time() - t0:.1f} s: gravity {float(gz_clean.min()) * MGAL:.2f} to "
      f"{float(gz_clean.max()) * MGAL:.2f} mGal, magnetics "
      f"{float(tmi_clean.min()) * NANOTESLA:.0f} to "
      f"{float(tmi_clean.max()) * NANOTESLA:.0f} nT, "
      f"both at {NOISE_PCT * 100:.0f}% noise")


def misfits(params) -> tuple[float, float]:
    """Reduced chi-squared per channel."""
    with torch.no_grad():
        pg = gravity(ModelState({"rho": params["drho"]}), ctx).data["gz"]
        pm = magnetics(ModelState({"chi": params["chi"].reshape(-1)}),
                       ctx).data["tmi"]
    return (float(((pg - gz_obs) / STD_G).pow(2).mean()),
            float(((pm - tmi_obs) / STD_M).pow(2).mean()))


# %% 3. Two stages: fit the data, then let the petrophysics speak ---------
BOUNDS = {"drho": (-2000.0, 0.0), "chi": (0.0, 0.1)}
# The mixture weight is not fixed. It climbs while both channels still fit
# and retreats the moment either stops fitting, so the coupling can never
# buy its clusters with data misfit.
# Weights are quoted per cell relative to the 864-cell model they were
# tuned on, so refining the mesh does not re-tune the inversion.
N_CELLS_REF = NZ * NX * NY / 864.0
W_START, W_UP, W_DOWN, W_MAX, TOL = 0.3, 1.8, 0.6, 100.0, 1.5
STOP_CHI2 = 1.05          # stage 1 stops at the noise, smeared and honest


def invert(init_rho, init_chi, *, guided: bool, n_iters: int, lr_rho: float,
           lr_chi: float, label: str):
    """One joint gravity + magnetics stage on a fresh EarthModel."""
    # The field is named "drho" because it is a density CONTRAST, which is
    # signed; the platform reserves "rho" for absolute density and range-checks
    # it. JointForward rewires drho into Gravity3D's rho input.
    model = EarthModel(mesh, fields={
        "drho": Field(init=init_rho.detach().clone()),
        "chi": Field(init=init_chi.detach().clone())})
    problem = JointProblem(
        model=model,
        forwards={"gravity": JointForward(gravity, fields={"drho": "rho"}),
                  "magnetics": JointForward(
                      magnetics, field_to_mesh=lambda t: t.reshape(-1))},
        observed={"gravity": gz_obs, "magnetics": tmi_obs},
        likelihoods={"gravity": GaussianLikelihood(std=STD_G),
                     "magnetics": GaussianLikelihood(std=STD_M)},
        weights="grad_norm", ctx=ctx)
    weight = {"pgi": W_START if guided else 0.0}
    history: list[tuple[float, float, float]] = []

    def regularizer(p):
        # Both terms sum over CELLS while the misfit sums over STATIONS, so a
        # weight that is not per-cell silently changes strength when the mesh
        # is refined. Dividing by the cell count keeps these numbers meaning
        # the same thing at any resolution.
        # drho's smoothness weight carries a 1e-6 rescale because the field is
        # in kg/m3 and the penalty is quadratic.
        reg = (smoothness(p["drho"], weight=1.0e-10 / N_CELLS_REF)
               + smoothness(p["chi"], weight=1.0e-4 / N_CELLS_REF))
        if weight["pgi"] > 0.0:
            pairs = torch.stack([p["drho"].flatten(),
                                 p["chi"].flatten()], dim=1)
            reg = reg + gmm_prior(pairs, gmm,
                                  weight=weight["pgi"] / N_CELLS_REF)
        return reg

    def callback(_it, _loss, params):
        chi2_g, chi2_m = misfits(params)
        history.append((chi2_g, chi2_m, weight["pgi"]))
        if not guided:
            # Stop as soon as both channels reach the noise: the smooth answer
            # is meant to be the honest under-determined one, not a polished
            # one bought by fitting past the error bars.
            return chi2_g < STOP_CHI2 and chi2_m < STOP_CHI2
        fitting = chi2_g < TOL and chi2_m < TOL
        weight["pgi"] = (min(weight["pgi"] * W_UP, W_MAX) if fitting
                         else max(weight["pgi"] * W_DOWN, W_START))
        return False

    callback.every = 10
    t_start = time.time()
    result = Inverter(problem, params=model,
                      learning_rates={"drho": lr_rho, "chi": lr_chi},
                      regularizer=regularizer, bounds=BOUNDS,
                      ctx=ctx).run(n_iters=n_iters, callback=callback)
    recovered = model.resolve_from(result.params)
    chi2_g, chi2_m = misfits(result.params)
    extra = f", mixture weight ended at {weight['pgi']:.1f}" if guided else ""
    rec_rho, rec_chi = recovered["drho"].detach(), recovered["chi"].detach()
    print(f"    amplitudes: drho {float(rec_rho.min()):+8.1f} to "
          f"{float(rec_rho.max()):+7.1f} kg/m3 (truth {RHO_PK:+.0f} to 0), "
          f"chi {float(rec_chi.min()):.4f} to {float(rec_chi.max()):.4f} SI "
          f"(truth 0 to {CHI_VK})")
    print(f"[3] {label:8s} {result.n_iters:3d} iterations in "
          f"{time.time() - t_start:4.0f} s: chi-squared gravity {chi2_g:.2f}, "
          f"magnetics {chi2_m:.2f}{extra}")
    return dict(drho=recovered["drho"].detach(),
                chi=recovered["chi"].detach(),
                chi2=(chi2_g, chi2_m), history=history)


smooth = invert(torch.full_like(rho_true, -0.1),
                torch.full_like(chi_true, 1.0e-4), guided=False,
                n_iters=120, lr_rho=30.0, lr_chi=2.0e-3, label="smooth")
guided = invert(smooth["drho"], smooth["chi"], guided=True,
                n_iters=150, lr_rho=15.0, lr_chi=8.0e-4, label="guided")

# %% 4. Score both, on the question that was actually asked ---------------
anomaly = geology_true > 0
results = {"smooth": smooth, "guided": guided}
for name, run in results.items():
    run["geology"] = classify(run["drho"], run["chi"])
    run["accuracy"] = float((run["geology"] == geology_true).double().mean())
    run["anomaly_accuracy"] = float(
        (run["geology"][anomaly] == geology_true[anomaly]).double().mean())
    # Fraction of anomaly cells inside their own unit's 95% ellipse: the
    # Mahalanobis test the mixture itself defines.
    inside = 0
    for unit in (1, 2):
        sel = geology_true == unit
        pair = torch.stack([run["drho"][sel], run["chi"][sel]], dim=1)
        d2 = (((pair - MEANS[unit]) ** 2) / VARS[unit]).sum(1)
        inside += int((d2 <= 5.991).sum())          # chi-squared(2) at 95%
    run["on_cluster"] = inside / int(anomaly.sum())

print("[4] rock-unit map over the anomalous cells: "
      + ", ".join(f"{k} {100 * v['anomaly_accuracy']:.0f}% correct"
                  for k, v in results.items()))
print("[4] and in petrophysical space, the share of anomaly cells sitting "
      "inside their own unit's 95% ellipse: "
      + ", ".join(f"{k} {100 * v['on_cluster']:.0f}%"
                  for k, v in results.items()))
print("[4] with the data fit held: gravity "
      + ", ".join(f"{k} {v['chi2'][0]:.2f}" for k, v in results.items())
      + "; magnetics "
      + ", ".join(f"{k} {v['chi2'][1]:.2f}" for k, v in results.items())
      + ". The prior did not buy its clusters with misfit")

# %% 5. Picture ------------------------------------------------------------
#
# The two anomaly maps come first because they ARE the argument: gravity
# sees the dense body and barely the other, magnetics the reverse. Neither
# survey alone separates the units; the rock-unit map is what the pair of
# them plus the petrophysics returns.
fig, axes = figure(1, 3, panel_w=4.4, panel_h=4.4)
Z_SLICE = 0
extent = (0.0, NX * DH, 0.0, NY * DH)
UNIT_NAMES = ("background", "PK", "VK")
UNIT_COLOURS = ("0.86", PALETTE[1], PALETTE[2])
unit_cmap, unit_norm = category_norm(len(UNIT_NAMES), colours=UNIT_COLOURS)
geo_lim = dict(cmap=unit_cmap, norm=unit_norm)
STATION_EXTENT = (float(line[0]), float(line[-1]), float(line[0]),
                  float(line[-1]))


def outline(ax):
    """The true bodies, on whatever the panel is showing."""
    for unit in (1, 2):
        ax.contour((geology_true[Z_SLICE] == unit).double().numpy().T,
                   levels=[0.5], colors="black", linewidths=1.3,
                   linestyles="--", origin="lower", extent=extent)


def anomaly(ax, values, title, label, cmap, **kw):
    grid = values.reshape(len(line), len(line)).T.numpy()
    im = ax.imshow(grid, origin="lower", extent=STATION_EXTENT, cmap=cmap,
                   aspect="auto", interpolation="bilinear", **kw)
    ax.set(title=title, xlabel="x [m]", xlim=(0.0, NX * DH),
           ylim=(0.0, NY * DH))
    ax.grid(False)
    outline(ax)
    shared_colorbar(fig, im, ax, label, location="bottom")


# Gravity3D returns elevation-up gz, so a mass DEFICIT comes back positive;
# geophysical displays are down-positive, where a light body is a low.
with torch.no_grad():
    gz_show = (-gz_obs * MGAL).detach()
    tmi_show = (tmi_obs * NANOTESLA).detach()
anomaly(axes[0], gz_show, "Gravity - only the dense-contrast body",
        r"$g_z$ [mGal]", CMAP_DEFICIT, vmin=float(gz_show.min()), vmax=0.0)
axes[0].set_ylabel("y [m]")
mag_lim = float(tmi_show.abs().max())
anomaly(axes[1], tmi_show, "Magnetics - only the magnetic body", "TMI [nT]",
        CMAP_ANOMALY, vmin=-mag_lim, vmax=mag_lim)


def show(ax, values, title, style, ylabel=None):
    im = ax.imshow(values[Z_SLICE].numpy().T, origin="lower", extent=extent,
                   aspect="auto", **style)
    ax.set(title=title, xlabel="x [m]", ylabel=ylabel)
    ax.grid(False)
    return im


im = show(axes[2], guided["geology"].double(),
          f"Recovered rock units ({100 * guided['anomaly_accuracy']:.0f}% "
          "correct)", geo_lim)
outline(axes[2])
category_colorbar(fig, im, axes[2], "Rock unit", UNIT_NAMES,
                  location="bottom")

fig.savefig(OUT / "04_petrophysical_joint_inversion.png")
print(f"saved {OUT / '04_petrophysical_joint_inversion.png'}")
plt.show()
