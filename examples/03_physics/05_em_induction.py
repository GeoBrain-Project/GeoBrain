"""Frequency-domain induction: a sounding, and the depth it can defend.

A vertical magnetic dipole is flown 30 m above the ground and a vertical
receiver 10 m away records the field the ground induces. Airborne systems
report that as the secondary field in parts per million of the primary::

    ppm = (Bz - Bp) / Bp * 1e6,     Bp = -mu0 * m / (4 pi r^3)

Sweeping frequency sweeps depth, because the skin depth shrinks as the
frequency rises: over 10 ohm-m ground it is about 80 m at 380 Hz and 4 m
at 130 kHz. So one sounding across four decades of frequency carries a
depth profile, and :class:`CSEM1D` computes it with a 1-D Hankel
transform through a layered earth: no mesh, no boundary conditions, and
autograd straight through the transform to every layer's conductivity.

The earth is three layers: 10 ohm-m over a 1 ohm-m conductor between 20
and 60 m, over 10 ohm-m again. Ten frequencies, real and imaginary, is
twenty numbers.

Twenty data, eleven unknowns
---------------------------
The inversion solves for log-conductivity on eleven fixed layers whose
thicknesses grow geometrically from 3 m, a grid chosen for the physics,
not for the data, and finer than twenty numbers can resolve. So the
answer is not determined by the data alone, and the script says what
picks the rest: a first-difference smoothness term with a weak pull
toward the starting half-space.

Reading the profile honestly is the skill here, and the script prints the
numbers rather than admiring the curve. The misfit settles just above the
noise, near chi-squared 1.2. The conductor's peak comes back at the right
strength, about 1.1 S/m against a true 1.0. Its TOP is placed within a
layer, because the high frequencies bracket it. Its BASE comes back some
18 percent too shallow, because below the conductor the field has already
decayed and every layer down there is competing for the same handful of
decimal places. The excess conductance (the integral of conductivity
above background, which is the quantity least sensitive to where you
decide the conductor ends) is about a quarter low.

None of those is a defect in the inversion; they are what twenty numbers
buy. Quote a layered EM result as a conductance with a depth range, and
it is defensible. Quote it as a boundary at 49 m, and it is not.

APIs featured:
    - geobrain.physics.em.CSEM1D with VMDSource / CSEMReceiver, and
      autograd through the 1-D Hankel transform
    - geobrain.optim.regularizers.smoothness + smallness
    - L-BFGS with a chi-squared stop

Expected runtime: < 3 min, and the 1-D Hankel forward is nearly all of
it: about 2.5 s per evaluation on eleven layers, which is why this
example inverts once rather than sweeping.

Outputs:
    out/05_em_induction.png: both sounding components against one fit,
    and the recovered layered conductivity against the truth.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    C_RECOVERED,
    C_START,
    C_TRUTH,
    PALETTE,
    apply_style,
    figure,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.optim.regularizers import smallness, smoothness
from geobrain.physics.em import (
    CSEM1D,
    CSEMReceiver,
    FieldComponent,
    MarineCSEM1DSurvey,
    VMDSource,
)

apply_style()
torch.manual_seed(6)
torch.set_default_dtype(torch.float64)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
MU_0 = 4.0e-7 * math.pi
SIGMA_TRUE = torch.tensor([0.1, 1.0, 0.1], dtype=D)      # S/m
THICK_TRUE = torch.tensor([20.0, 40.0], dtype=D)         # m
N_FREQ = 10
FREQS = torch.logspace(math.log10(380.0), math.log10(130000.0), N_FREQ, dtype=D)
ALTITUDE, OFFSET, MOMENT = 30.0, 10.0, 1.0
NOISE_PCT, NOISE_FLOOR = 0.05, 1.0e-3

# %% 1. The sounding ------------------------------------------------------
B_PRIMARY = -MU_0 * MOMENT / (4.0 * math.pi * OFFSET ** 3)
survey = MarineCSEM1DSurvey(
    sources=(VMDSource(position=(0.0, 0.0, -ALTITUDE),
                       magnetic_moment_am2=MOMENT),),
    receivers=(CSEMReceiver(position=(OFFSET, 0.0, -ALTITUDE),
                            component=FieldComponent.BZ),),
    frequencies=tuple(float(f) for f in FREQS))
sounding = CSEM1D(survey)
ctx = ForwardContext()


def response(sigma: torch.Tensor, thickness: torch.Tensor) -> torch.Tensor:
    """Secondary field in ppm, stacked as [real over f, imaginary over f]."""
    bz = sounding(ModelState({"sigma": sigma, "thickness": thickness}),
                  ctx).data["bz"][0]
    ppm = (bz - B_PRIMARY) / B_PRIMARY * 1e6
    return torch.cat([ppm.real, ppm.imag])


RHO_TOP = 1.0 / float(SIGMA_TRUE[0])
skin = 503.0 * (RHO_TOP / FREQS) ** 0.5
t0 = time.time()
with torch.no_grad():
    clean = response(SIGMA_TRUE, THICK_TRUE)
sigma_d = NOISE_PCT * clean.abs() + NOISE_FLOOR
observed = clean + sigma_d * torch.randn_like(clean)
print(f"[1] {N_FREQ} frequencies, {float(FREQS[0]):.0f} Hz to "
      f"{float(FREQS[-1]) / 1e3:.0f} kHz: over {RHO_TOP:.0f} ohm-m the skin "
      f"depth runs {float(skin[-1]):.0f} m to {float(skin[0]):.0f} m, which "
      f"brackets the conductor at {float(THICK_TRUE[0]):.0f}-"
      f"{float(THICK_TRUE.sum()):.0f} m")
print(f"[1] forward {time.time() - t0:.1f} s; secondary field "
      f"{float(clean[:N_FREQ].min()):.0f}-{float(clean[:N_FREQ].max()):.0f} ppm "
      f"in phase, {float(clean[N_FREQ:].min()):.0f}-"
      f"{float(clean[N_FREQ:].max()):.0f} ppm quadrature; {observed.numel()} "
      f"data at {NOISE_PCT * 100:.0f}% noise")

# %% 2. A layer grid finer than the data can resolve ----------------------
widths, total, w = [], 0.0, 3.0
while total < 200.0:
    widths.append(w)
    total += w
    w *= 1.4
INV_THICK = torch.tensor(widths, dtype=D)
N_LAYER = len(widths) + 1
LOG_START = math.log10(float(SIGMA_TRUE[0]))
edges = torch.cat([torch.zeros(1, dtype=D), INV_THICK.cumsum(0)])
BETA, GAMMA, MAX_ITER = 2.5, 2.0e-3, 30
print(f"[2] {N_LAYER} layers from {float(INV_THICK[0]):.0f} m thick at the top "
      f"to {float(INV_THICK[-1]):.0f} m at {float(edges[-1]):.0f} m, "
      f"{N_LAYER} unknowns against {observed.numel()} data, so a smoothness "
      "term picks among the profiles that fit")

# %% 3. Invert -------------------------------------------------------------
m = torch.full((N_LAYER,), LOG_START, dtype=D, requires_grad=True)
optimizer = torch.optim.LBFGS([m], lr=1.0, max_iter=MAX_ITER, history_size=12,
                              line_search_fn="strong_wolfe")
history: list[float] = []
at_target: torch.Tensor | None = None
stopped_at: int | None = None


def closure() -> torch.Tensor:
    global at_target, stopped_at
    optimizer.zero_grad()
    chi2 = (((response(torch.pow(10.0, m), INV_THICK) - observed)
             / sigma_d) ** 2).mean()
    (chi2 + smoothness(m, weight=BETA)
     + smallness(m, m_ref=LOG_START, weight=GAMMA)).backward()
    history.append(float(chi2.detach()))
    if at_target is None and history[-1] <= 1.0:
        at_target, stopped_at = m.detach().clone(), len(history)
    return chi2


t0 = time.time()
optimizer.step(closure)
if at_target is None:
    at_target = m.detach().clone()
sigma_rec = torch.pow(10.0, at_target)
with torch.no_grad():
    predicted = response(sigma_rec, INV_THICK)
print(f"[3] {len(history)} evaluations in {time.time() - t0:.0f} s "
      f"({(time.time() - t0) / len(history):.1f} s each, the Hankel forward "
      f"is the cost); chi-squared {history[0]:.0f} -> {min(history):.2f}"
      + (f", crossing 1 at evaluation {stopped_at}" if stopped_at
         else " (never reached 1)"))

# %% 4. What the profile is worth -----------------------------------------
centres = 0.5 * (edges[:-1] + edges[1:])
peak = int(sigma_rec.argmax())
true_mid = float(THICK_TRUE[0] + THICK_TRUE[1] / 2)
conductive = (sigma_rec > 0.3).nonzero().flatten()
top = float(edges[int(conductive[0])]) if conductive.numel() else float("nan")
base = float(edges[int(conductive[-1]) + 1]) if conductive.numel() else float("nan")
# Excess conductance = integral of (sigma - background) dz over the whole
# profile. It is the closest thing to what an induction sounding measures
# directly, and unlike a boundary depth it does not depend on where you
# decide the conductor stops.
BACKGROUND = float(SIGMA_TRUE[0])
excess_true = float(((SIGMA_TRUE[:2] - BACKGROUND) * THICK_TRUE).sum())
excess_rec = float(((sigma_rec[:-1] - BACKGROUND).clamp(min=0.0)
                    * INV_THICK).sum())
print(f"[4] peak conductivity {float(sigma_rec[peak]):.2f} S/m at "
      f"{float(centres[peak]):.0f} m (truth 1.00 S/m centred {true_mid:.0f} m); "
      f"above 0.3 S/m from {top:.0f} to {base:.0f} m against a true "
      f"{float(THICK_TRUE[0]):.0f}-{float(THICK_TRUE.sum()):.0f} m")
print(f"[4] excess conductance over the whole profile {excess_rec:.0f} S "
      f"against a true {excess_true:.0f} S "
      f"({abs(excess_rec / excess_true - 1) * 100:.0f}% low), and the base of "
      f"the conductor {abs(base / float(THICK_TRUE.sum()) - 1) * 100:.0f}% too "
      "shallow. Both are what twenty numbers buy, so quote this as a "
      f"conductance of order {excess_rec:.0f} S somewhere in {top:.0f}-"
      f"{base:.0f} m and it is defensible; quote a boundary at {base:.0f} m "
      "and it is not")

# %% 5. Picture ------------------------------------------------------------
# A sounding and a conductivity profile are both read DOWN the page, so
# the panels are narrow and tall like the other well displays in the
# gallery rather than the landscape default.
fig, axes = figure(1, 2, panel_w=3.6, panel_h=6.6)
freq_np = FREQS.numpy()

ax = axes[0]
for sl, colour, name in ((slice(0, N_FREQ), PALETTE[0], "In-phase (real)"),
                         (slice(N_FREQ, 2 * N_FREQ), PALETTE[1],
                          "Quadrature (imaginary)")):
    ax.errorbar(freq_np, observed[sl].numpy(), yerr=sigma_d[sl].numpy(),
                fmt="o", ms=6, mfc="none", mew=1.4, color=colour,
                elinewidth=1.0, capsize=2.5, ls="none", label=f"{name}, observed")
    ax.semilogx(freq_np, predicted[sl].numpy(), color=colour, lw=2.0,
                label=f"{name}, recovered")
ax.set(title=f"Sounding fit (chi-squared {min(history):.2f})",
       xlabel="Frequency [Hz]", ylabel="Secondary / primary [ppm]",
       xscale="log", xlim=(0.8 * freq_np.min(), 1.3 * freq_np.max()))
ax.legend(fontsize=8)
ax.grid(which="both")


def staircase(thickness: torch.Tensor, values: torch.Tensor, bottom: float):
    """Layer values as a depth staircase, for a step plot."""
    depth = torch.cat([torch.zeros(1, dtype=D), thickness.cumsum(0),
                       torch.tensor([bottom], dtype=D)])
    return (values.repeat_interleave(2).numpy(),
            depth.repeat_interleave(2)[1:-1].numpy())


BOTTOM = float(INV_THICK.sum())
ax = axes[1]
s, d = staircase(THICK_TRUE, SIGMA_TRUE, BOTTOM)
ax.plot(s, d, color=C_TRUTH, lw=2.6, label="Truth")
s, d = staircase(INV_THICK, torch.full((N_LAYER,), 10.0 ** LOG_START, dtype=D),
                 BOTTOM)
ax.plot(s, d, color=C_START, ls=":", lw=1.6, label="Start (half-space)")
s, d = staircase(INV_THICK, sigma_rec, BOTTOM)
ax.plot(s, d, color=C_RECOVERED, lw=2.0,
        label=f"Recovered ({excess_rec:.0f} S excess vs {excess_true:.0f} S)")
ax.set(title="Recovered conductivity", xlabel="Conductivity [S/m]",
       ylabel="Depth [m]", xscale="log", ylim=(BOTTOM, 0.0),
       xlim=(0.6 * float(SIGMA_TRUE.min()), 2.0 * float(SIGMA_TRUE.max())))
ax.legend(fontsize=8, loc="lower right")
ax.grid(which="both")

fig.savefig(OUT / "05_em_induction.png")
print(f"saved {OUT / '05_em_induction.png'}")
plt.show()
