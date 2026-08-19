"""DC resistivity: a survey over a ridge, and when to stop inverting.

Inject current between two electrodes, measure the voltage between two
others, walk the pair down the line and outward in separation. That is
the workhorse of near-surface geophysics, and this script runs it the way
it is actually run: over TOPOGRAPHY, and stopping when the data are fit
to the noise rather than after a round number of iterations.

Topography is not decoration. The ground here rises 70 m over a ridge,
and it is handled the way every DC code handles it: the mesh keeps going
above the ground and the cells above the surface are given the
conductivity of air. Electrodes are then draped onto the first ground
cell in their column. Nothing about :class:`DC2D` changes; the air is
just very resistive rock. Two consequences worth seeing:

    the air cells must be HELD FIXED during the inversion. Their gradient
    is not small; it is the largest in the model, because a tiny
    conductivity in the denominator makes the objective extremely
    sensitive there. Invert them and the air fills with current.

    apparent resistivity is distorted by the ridge before any geology is
    involved. The pseudosection is a display convention, not an image.

The stopping rule is the other half of the script. Data carry 3% noise,
so a model that drives the misfit to zero is fitting noise and will grow
structure to do it. The inversion therefore watches chi-squared, the
misfit measured in units of the noise, and stops at chi-squared = 1,
which is the statistically honest place to stop.

APIs featured:
    - geobrain.physics.em.DC2D, DC2DSurvey.from_grid_indices
    - an air mask + draped electrodes for topography
    - log-conductivity inversion restricted to the active cells
    - the gallery's shared figure toolkit in _style.py

Expected runtime: < 2 min.

Outputs:
    out/02_dc_resistivity.png: the section under topography, the model
    recovered at chi-squared 1, and the same inversion carried on until
    it fits noise.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from _models import correlated_fields
from _style import (
    CMAP_ANOMALY,
    apply_style,
    field,
    figure,
    shared_colorbar,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.optim.regularizers import smoothness
from geobrain.physics.em import DC2D, DC2DSurvey

apply_style()
torch.manual_seed(4)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
NZ, NX, DH = 36, 80, 12.0                     # 432 m tall (incl. air), ~1 km
mesh = TensorMesh(shape=(NZ, NX), spacing=(DH, DH))
ctx = ForwardContext.of(mesh=mesh)
zz = torch.arange(NZ, dtype=D)[:, None] * DH
xx = torch.arange(NX, dtype=D)[None, :] * DH

# %% 1. A ridge, and an earth under it -------------------------------------
AIR_THICKNESS = 120.0                         # mesh top above the lowest ground
RELIEF = 70.0
SIGMA_AIR, RHO_BG = 1e-8, 150.0
ridge = RELIEF / (1.0 + ((xx[0] - 480.0) / 250.0) ** 2)
z_surface = AIR_THICKNESS - ridge             # depth to ground, per column
air = zz < z_surface[None, :]
depth_below = zz - z_surface[None, :]         # depth BELOW the local surface

texture, _ = correlated_fields((NZ, NX), DH, seed=5, ranges=(200.0, 60.0),
                               correlation=0.0)
conductive = torch.exp(-(((xx - 300.0) / 62.0) ** 2
                         + ((depth_below - 75.0) / 32.0) ** 2))
resistive = torch.exp(-(((xx - 660.0) / 66.0) ** 2
                        + ((depth_below - 85.0) / 34.0) ** 2))
rho_ground = (torch.tensor(RHO_BG).log() + 0.25 * texture
              - 2.0 * conductive + 1.5 * resistive).exp()
sigma_true = torch.where(air, torch.tensor(SIGMA_AIR, dtype=D),
                         1.0 / rho_ground)
print(f"[1] {NZ}x{NX} cells at {DH:.0f} m; {RELIEF:.0f} m of relief, "
      f"{int(air.sum())} air cells; ground resistivity "
      f"{float(rho_ground[~air].min()):.0f}-"
      f"{float(rho_ground[~air].max()):.0f} ohm-m")

# %% 2. Electrodes draped on the ground ------------------------------------
ELEC_X = torch.arange(5, 76, 5, dtype=torch.long)
ELEC_Z = torch.tensor([int(air[:, int(c)].sum()) for c in ELEC_X],
                      dtype=torch.long)
N_LEVEL = 4
pairs = [(int(ELEC_X[i]), int(ELEC_X[i + 1])) for i in range(len(ELEC_X) - 1)]
layout: list[tuple[int, int, int]] = []
for ip in range(len(pairs)):
    for n in range(1, N_LEVEL + 1):
        im, jn = ip + 1 + n, ip + 2 + n
        if jn < len(ELEC_X):
            layout.append((ip, im, jn))
operators = [DC2D(DC2DSurvey.from_grid_indices(
    source_z=int(ELEC_Z[i]), source_x=a,
    sink_z=int(ELEC_Z[i + 1]), sink_x=b,
    rcv_z=ELEC_Z, rcv_x=ELEC_X, spacing=(DH, DH), current=1.0))
    for i, (a, b) in enumerate(pairs)]
print(f"[2] {len(ELEC_X)} electrodes draped on the ridge -> {len(pairs)} "
      f"injections, {len(layout)} quadrupoles (n = 1..{N_LEVEL})")


def measure(sigma: torch.Tensor) -> torch.Tensor:
    phi = [op(ModelState({"sigma": sigma}), ctx).data["voltage"]
           for op in operators]
    return torch.stack([phi[i][m] - phi[i][n] for i, m, n in layout])


t0 = time.time()
with torch.no_grad():
    v_clean = measure(sigma_true)
    flat = torch.where(air, torch.tensor(SIGMA_AIR, dtype=D),
                       torch.tensor(1.0 / RHO_BG, dtype=D))
    v_flat = measure(flat)
NOISE = 0.03
sigma_d = NOISE * v_clean.abs().clamp_min(v_clean.abs().median() * 1e-2)
v_obs = v_clean + sigma_d * torch.randn_like(v_clean)
rho_app_obs = RHO_BG * v_obs / v_flat
print(f"[2] forward in {time.time() - t0:.0f} s; {v_obs.numel()} readings "
      f"with {NOISE * 100:.0f}% noise. Over a FLAT earth of the same "
      f"{RHO_BG:.0f} ohm-m the ridge alone would move apparent resistivity "
      f"by {float((RHO_BG * v_clean / v_flat / RHO_BG - 1.0).abs().max() * 100):.0f}%")

# %% 3. Invert the active cells, watching chi-squared ----------------------
#
# Only the ground is a parameter. The air is held at its true value: its
# gradient is the largest in the model, and letting the optimizer touch it
# just fills the sky with current.
log_bg = float(torch.tensor(1.0 / RHO_BG).log())
log_sigma0 = torch.full((NZ, NX), log_bg, dtype=D)
log_sigma = log_sigma0.clone().requires_grad_(True)
ground = ~air
optimizer = torch.optim.LBFGS([log_sigma], lr=1.0, max_iter=30,
                              history_size=12,
                              line_search_fn="strong_wolfe")
chi2_history: list[float] = []
at_target: torch.Tensor | None = None
stopped_at = None
t0 = time.time()


def closure() -> torch.Tensor:
    global at_target, stopped_at
    optimizer.zero_grad()
    sigma = torch.where(air, torch.tensor(SIGMA_AIR, dtype=D), log_sigma.exp())
    chi2 = (((measure(sigma) - v_obs) / sigma_d) ** 2).mean()
    (chi2 + smoothness((log_sigma - log_sigma0) * ground, dx=DH, dz=DH,
                       weight=2.0)).backward()
    chi2_history.append(float(chi2.detach()))
    # the optimizer will happily keep going past the noise level; keep the
    # model from the moment the data were first fit, which is the answer
    if at_target is None and chi2_history[-1] <= 1.0:
        at_target = log_sigma.detach().clone()
        stopped_at = len(chi2_history)
    return chi2


optimizer.step(closure)
if at_target is None:
    at_target = log_sigma.detach().clone()
with torch.no_grad():
    rho_rec = 1.0 / torch.where(air, torch.tensor(SIGMA_AIR, dtype=D),
                                at_target.exp())
    rho_over = 1.0 / torch.where(air, torch.tensor(SIGMA_AIR, dtype=D),
                                 log_sigma.exp())
    rho_app_fit = RHO_BG * measure(1.0 / rho_rec) / v_flat
print(f"[3] inverted {int(ground.sum())} active cells in "
      f"{time.time() - t0:.0f} s over {len(chi2_history)} evaluations; "
      f"chi-squared {chi2_history[0]:.1f} -> {min(chi2_history):.2f}"
      + (f", crossing 1 at evaluation {stopped_at}" if stopped_at
         else " (never reached 1)"))
print(f"    at the noise level the ground spans "
      f"{float(rho_rec[ground].min()):.0f}-{float(rho_rec[ground].max()):.0f} "
      f"ohm-m; carried on to chi-squared {min(chi2_history):.2f} it spans "
      f"{float(rho_over[ground].min()):.0f}-{float(rho_over[ground].max()):.0f}"
      ", structure invented to fit noise")

# %% 4. Picture ------------------------------------------------------------
# Three sections stacked, not side by side: a section is 960 m wide and
# 430 m deep, so it wants a wide panel, and stacking puts the same
# distance axis under all three. The chi-squared history is not drawn -
# the two right-hand titles ARE the stopping rule, and the full history
# is printed above.
fig, axes = figure(3, 1, panel_w=7.2, panel_h=3.0, sharex=True)
EXTENT = (0.0, NX * DH, NZ * DH, 0.0)

# A diverging ramp on resistivity is only honest if white means something.
# It is centred on the BACKGROUND, so blue reads "more conductive than the
# host rock" and red "more resistive" - which is what a reader of a DC
# section is looking for. The two sides scale independently, because the
# conductor departs from the background further than the resistor does.
lognorm = dict(norm=TwoSlopeNorm(
    vcenter=float(np.log10(RHO_BG)),
    vmin=float(rho_ground[~air].log10().min()),
    vmax=float(rho_ground[~air].log10().max())))


def section(ax, values, title):
    shown = np.where(air.numpy(), np.nan, values.log10().numpy())
    image = field(ax, shown, extent=EXTENT, cmap=CMAP_ANOMALY, title=title,
                  ylabel="Depth below mesh top [m]", **lognorm)
    ax.plot(xx[0].numpy(), z_surface.numpy(), color="black", lw=1.2)
    ax.plot((ELEC_X.double() * DH).numpy(),
            (ELEC_Z.double() * DH).numpy(), "v", ms=5, color="black",
            mec="white", mew=0.6, ls="none")
    return image


image = section(axes[0], rho_ground, "True resistivity under the ridge")
section(axes[1], rho_rec, "Recovered, stopped at chi-squared = 1")
section(axes[2], rho_over,
        f"Carried on to chi-squared {min(chi2_history):.2f}")
axes[2].set_xlabel("Distance [m]")
shared_colorbar(fig, image, axes, r"$\log_{10}\rho$ [ohm-m]",
                location="bottom")

fig.savefig(OUT / "02_dc_resistivity.png")
print(f"saved {OUT / '02_dc_resistivity.png'}")
plt.show()
