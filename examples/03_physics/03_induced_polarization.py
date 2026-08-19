"""Induced polarization: the image resistivity cannot give you.

A DC survey asks how easily the ground conducts. An IP survey, run with
the same electrodes on the same wire, asks something independent: how
much charge the rock STORES while the current is on and gives back after
it is switched off. Disseminated sulphides, graphite and clays do that;
brine in clean sand does not.

The ground is the ridge from the DC script, and two bodies are buried
under it so the difference cannot be argued with:

    A is conductive but not chargeable: a brine-saturated lens. DC sees it;
        IP does not.

    B is chargeable, with NO resistivity contrast of its own. The
        disseminated-sulphide case, and the one that pays for the survey.
        DC is blind to it by construction.

Two practical points the script is built around.

Apparent chargeability is a RATIO of two DC solves, ``(V_eta - V_inf) /
V_eta``, so the dipole difference has to be taken on each potential
separately, BEFORE the ratio is formed. Differencing apparent
chargeability between receivers is not a thing you may do.

Chargeability lives in [0, 1), so it is inverted through a sigmoid rather
than as a free parameter: an unbounded step drives the effective
conductivity ``sigma_inf (1 - eta)`` negative and the solve fails. The
starting model matters too: start at eta = 0.2 everywhere and the
optimizer sits there; start near zero and it finds the body.

Like the DC script, this one stops when chi-squared reaches 1 rather than
after a fixed iteration count.

APIs featured:
    - geobrain.physics.em.IP2D (sigma_infty, chargeability ->
      chargeability_obs, with V_primary / V_steady as diagnostic fields)
    - the same topographic air mask and draped electrodes as script 02
    - sigmoid-bounded chargeability inversion on the active cells

Expected runtime: < 2 min.

Outputs:
    out/03_induced_polarization.png: the two truths, resistivity and
    chargeability, and the chargeability recovered from the IP data.

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
    CMAP_EXCESS,
    apply_style,
    field,
    figure,
    shared_colorbar,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.optim.regularizers import smoothness
from geobrain.physics.em import IP2D, DC2DSurvey

apply_style()
torch.manual_seed(6)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
NZ, NX, DH = 32, 76, 12.0
mesh = TensorMesh(shape=(NZ, NX), spacing=(DH, DH))
ctx = ForwardContext.of(mesh=mesh)
zz = torch.arange(NZ, dtype=D)[:, None] * DH
xx = torch.arange(NX, dtype=D)[None, :] * DH

# %% 1. The ridge, a conductor, and a chargeable body ----------------------
AIR_THICKNESS, RELIEF = 96.0, 55.0
SIGMA_AIR, RHO_BG, ETA_BODY = 1e-8, 200.0, 0.20
ridge = RELIEF / (1.0 + ((xx[0] - 450.0) / 240.0) ** 2)
z_surface = AIR_THICKNESS - ridge
air = zz < z_surface[None, :]
depth_below = zz - z_surface[None, :]

texture, _ = correlated_fields((NZ, NX), DH, seed=9, ranges=(170.0, 55.0),
                               correlation=0.0)
body_a = torch.exp(-(((xx - 260.0) / 58.0) ** 2
                     + ((depth_below - 70.0) / 30.0) ** 2))
body_b = torch.exp(-(((xx - 640.0) / 60.0) ** 2
                     + ((depth_below - 70.0) / 30.0) ** 2))
rho_ground = (torch.tensor(RHO_BG).log() + 0.22 * texture
              - 2.4 * body_a).exp()
eta_ground = ETA_BODY * body_b
sigma_true = torch.where(air, torch.tensor(SIGMA_AIR, dtype=D),
                         1.0 / rho_ground)
eta_true = torch.where(air, torch.zeros((), dtype=D), eta_ground)
ground = ~air
print(f"[1] {NZ}x{NX} cells at {DH:.0f} m, {RELIEF:.0f} m of relief")
print(f"    body A: down to {float(rho_ground[ground].min()):.0f} ohm-m, "
      "chargeability 0")
print(f"    body B: no resistivity contrast, chargeability up to "
      f"{float(eta_true.max()):.2f}")

# %% 2. The same line, draped ---------------------------------------------
ELEC_X = torch.arange(5, 72, 5, dtype=torch.long)
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
operators = [IP2D(DC2DSurvey.from_grid_indices(
    source_z=int(ELEC_Z[i]), source_x=a,
    sink_z=int(ELEC_Z[i + 1]), sink_x=b,
    rcv_z=ELEC_Z, rcv_x=ELEC_X, spacing=(DH, DH), current=1.0))
    for i, (a, b) in enumerate(pairs)]
print(f"[2] {len(ELEC_X)} electrodes -> {len(pairs)} injections, "
      f"{len(layout)} quadrupoles")


def potentials(sigma: torch.Tensor,
               eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dipole differences of V_inf and V_eta, taken BEFORE the ratio."""
    primary, steady = [], []
    for op in operators:
        out = op(ModelState({"sigma_infty": sigma, "chargeability": eta}), ctx)
        primary.append(out.fields["V_primary"])
        steady.append(out.fields["V_steady"])
    dv_p = torch.stack([primary[i][m] - primary[i][n] for i, m, n in layout])
    dv_s = torch.stack([steady[i][m] - steady[i][n] for i, m, n in layout])
    return dv_p, dv_s


t0 = time.time()
with torch.no_grad():
    dv_p, dv_s = potentials(sigma_true, eta_true)
    flat = torch.where(air, torch.tensor(SIGMA_AIR, dtype=D),
                       torch.tensor(1.0 / RHO_BG, dtype=D))
    dv_flat, _ = potentials(flat, torch.zeros_like(eta_true))
rho_app = RHO_BG * dv_p / dv_flat
m_clean = (dv_s - dv_p) / dv_s
NOISE = 0.02
sigma_m = torch.full_like(m_clean, NOISE * float(m_clean.abs().max()))
m_obs = m_clean + sigma_m * torch.randn_like(m_clean)
print(f"[2] forward in {time.time() - t0:.0f} s; apparent chargeability "
      f"{float(m_clean.min()):+.4f} to {float(m_clean.max()):+.4f}")

# %% 3. Each body answers exactly one of the two questions ----------------
midpoint = torch.tensor([0.5 * float(ELEC_X[m] + ELEC_X[n]) * DH
                         for _, m, n in layout], dtype=D)
level = torch.tensor([float(n - i - 2) for i, _, n in layout], dtype=D)
over_a = midpoint < 420.0
over_b = midpoint > 480.0
print(f"[3] over A: rho_app {float(rho_app[over_a].mean()):6.1f} ohm-m, "
      f"m_app {float(m_clean[over_a].mean()):+.4f}")
print(f"    over B: rho_app {float(rho_app[over_b].mean()):6.1f} ohm-m, "
      f"m_app {float(m_clean[over_b].mean()):+.4f}, so B is invisible to the "
      "resistivity image and obvious in the chargeability one")

# %% 4. Invert the chargeability, stopping at the noise level -------------
ETA_MAX = 0.4
raw0 = torch.full((NZ, NX), -3.66, dtype=D)          # eta ~ 0.01 everywhere
raw = raw0.clone().requires_grad_(True)
optimizer = torch.optim.LBFGS([raw], lr=1.0, max_iter=25, history_size=10,
                              line_search_fn="strong_wolfe")
chi2_history: list[float] = []
at_target: torch.Tensor | None = None
stopped_at = None
t0 = time.time()


def closure() -> torch.Tensor:
    global at_target, stopped_at
    optimizer.zero_grad()
    eta = torch.sigmoid(raw) * ETA_MAX * ground
    p_try, s_try = potentials(sigma_true, eta)
    chi2 = ((((s_try - p_try) / s_try - m_obs) / sigma_m) ** 2).mean()
    (chi2 + smoothness(eta, dx=DH, dz=DH, weight=4e-3)).backward()
    chi2_history.append(float(chi2.detach()))
    if at_target is None and chi2_history[-1] <= 1.0:
        at_target = raw.detach().clone()
        stopped_at = len(chi2_history)
    return chi2


optimizer.step(closure)
if at_target is None:
    at_target = raw.detach().clone()
eta_rec = (torch.sigmoid(at_target) * ETA_MAX * ground).detach()
target = eta_true > 0.5 * ETA_BODY
peak = (eta_rec == eta_rec.max()).nonzero()[0]
print(f"[4] inverted in {time.time() - t0:.0f} s over {len(chi2_history)} "
      f"evaluations; chi-squared {chi2_history[0]:.1f} -> "
      f"{min(chi2_history):.2f}"
      + (f", crossing 1 at evaluation {stopped_at}" if stopped_at else ""))
print(f"    chargeability {float(eta_rec[target].mean()):.3f} inside the body "
      f"(truth {ETA_BODY:.2f}), {float(eta_rec[ground & ~target].mean()):.3f} "
      f"outside; peak at {float(peak[1]) * DH:.0f} m")

# %% 5. Picture ------------------------------------------------------------
fig, axes = figure(3, 1, panel_w=7.2, panel_h=3.0, sharex=True)
EXTENT = (0.0, NX * DH, NZ * DH, 0.0)


def section(ax, values, title, cmap, **kw):
    shown = np.where(air.numpy(), np.nan, values.numpy())
    image = field(ax, shown, extent=EXTENT, cmap=cmap, title=title,
                  ylabel="Depth below mesh top [m]", **kw)
    ax.plot(xx[0].numpy(), z_surface.numpy(), color="black", lw=1.2)
    ax.plot((ELEC_X.double() * DH).numpy(), (ELEC_Z.double() * DH).numpy(),
            "v", ms=5, color="black", mec="white", mew=0.6, ls="none")
    for blob, name in ((body_a, "A"), (body_b, "B")):
        ax.contour(xx.expand(NZ, NX).numpy(), zz.expand(NZ, NX).numpy(),
                   (blob * ground).numpy(), levels=[0.4], colors="0.25",
                   linewidths=1.2, linestyles="--")
    return image


# Resistivity diverges about the HOST ROCK: white is the background, blue
# is more conductive than it, red more resistive. Centring anywhere else
# would make an unremarkable background read as an anomaly.
image = section(axes[0], rho_ground.log10(),
                "True resistivity", CMAP_ANOMALY,
                norm=TwoSlopeNorm(vcenter=float(np.log10(RHO_BG)),
                                  vmin=float(rho_ground[~air].log10().min()),
                                  vmax=float(rho_ground[~air].log10().max())))
shared_colorbar(fig, image, axes[0], r"$\log_{10}\rho$ [ohm-m]",
                location="bottom")

# Chargeability is the SAME ramp, but only the half of it that can occur:
# the background here is exactly zero and eta is never negative, so a full
# diverging bar would spend half its length on values the earth cannot
# take. CMAP_EXCESS is literally the upper half of CMAP_ANOMALY - same
# colours, same white-at-zero, twice the resolution.
image = section(axes[1], eta_true, "True chargeability",
                CMAP_EXCESS, vmin=0.0, vmax=ETA_BODY)
section(axes[2], eta_rec,
        f"Recovered, stopped at chi-squared {chi2_history[-1]:.2f}",
        CMAP_EXCESS, vmin=0.0, vmax=ETA_BODY)
axes[2].set_xlabel("Distance [m]")
shared_colorbar(fig, image, axes[1:3], r"$\eta$ [-]", location="bottom")

fig.savefig(OUT / "03_induced_polarization.png")
print(f"saved {OUT / '03_induced_polarization.png'}")
plt.show()
