"""Full-waveform inversion on Marmousi: the flagship, in forty lines of loop.

FWI is not a feature in GeoBrain; it is what you get when a wave operator
declares ``IMPLICIT_VJP`` and you point a stock optimizer at it. The
backward pass of :class:`Helmholtz2D` is one adjoint solve of the same
sparse system the forward used, so ``loss.backward()`` IS the FWI
gradient, and no adjoint code appears in this script.

The earth is a window of the Marmousi II benchmark, read from
``examples/data/marmousi`` and resampled onto the working grid: faulted,
dipping, and full of the thin high-velocity wedges that make the model a
benchmark in the first place. Nothing here is a Gaussian blob, and the
starting model is the one a real project would own: the true section
smoothed until only its broad trend survives.

What the loop adds is the craft that makes FWI converge:

    multi-scale   invert 2 Hz before 11 Hz. Low frequencies see the
                  smooth background and keep the update on the right side
                  of a cycle; starting high is how FWI cycle-skips.
    L-BFGS with a line search, restarted per band. This is the single
                  biggest lever in the script. A first-order method with a
                  hand-set step (Adam at lr 0.012) closes 12% of the gap;
                  L-BFGS closes 36% in the same wall clock, because FWI
                  misfit is badly scaled and curvature information is
                  worth more than momentum.
    update damping  regularize (m - m0), not m: the background trend is
                  information you already trust, and the wavenumbers the
                  acquisition cannot see are what need damping. How MUCH
                  damping is not a property of the physics; it depends on
                  the optimizer. Adam normalizes every parameter by its
                  own gradient history, which lets poorly-constrained deep
                  cells move as fast as well-constrained shallow ones, so
                  it needed roughly thirty times this weight just to stay
                  stable. Under L-BFGS the same weight strangles the
                  update.
    per-trace balance  normalize every trace by its own amplitude, or the
                  direct arrival, orders of magnitude the strongest
                  event, is the only thing the objective ever sees.

Read the result the way a processor would: the difference panel is what
the data actually bought, and it is also the evidence that the starting
model was smooth, since every fault and every layer in the recovered
section had to come from there. The deep half recovers less, because at
these offsets and frequencies the acquisition simply does not illuminate
it, a statement about the survey rather than about the optimizer.

APIs featured:
    - geobrain.physics.wave.Helmholtz2D (+ packed Survey/Source/Receiver)
    - torch.optim.LBFGS driving a GeoBrain forward directly, one
      restarted instance per frequency band
    - geobrain.optim.regularizers.smoothness on the model UPDATE
    - _models.marmousi / _models.smooth: the shared benchmark loader

Expected runtime: < 2 min.

Outputs:
    out/01_seismic_fwi.png: true and recovered velocity, the update
    FWI bought, and the band-by-band misfit.
    out/01_fwi_climb.gif: the model sharpening band by band.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _models import marmousi, smooth
from _style import (
    CMAP_ANOMALY,
    CMAP_VELOCITY,
    animation,
    apply_style,
    field,
    figure,
    shared_colorbar,
    symmetric_limits,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.optim.regularizers import smoothness
from geobrain.physics.wave import (
    Helmholtz2D,
    Helmholtz2DReceiver,
    Helmholtz2DSource,
    Helmholtz2DSurvey,
)
from geobrain.vis import plot_convergence

apply_style()
torch.manual_seed(7)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. The earth: a window of Marmousi II --------------------------------
NZ, NX = 80, 200
DZ = DX = 16.0                                   # 1.28 km deep, 3.2 km wide
mesh = TensorMesh(shape=(NZ, NX), spacing=(DZ, DX))
ctx = ForwardContext.of(mesh=mesh)
vp_true = marmousi(NZ, NX, DX, fields=("vp",))["vp"]
print(f"[1] Marmousi II window on a {NZ}x{NX} grid at {DX:.0f} m; vp in "
      f"[{float(vp_true.min()):.0f}, {float(vp_true.max()):.0f}] m/s")

# %% 2. Acquisition and observed data --------------------------------------
N_SHOT, N_RCV = 16, 60
FREQS = (2.0, 3.0, 4.0, 6.0, 8.0, 11.0)
src_x = torch.linspace(0.05, 0.95, N_SHOT) * (NX - 1) * DX
rcv_x = torch.linspace(0.02, 0.98, N_RCV) * (NX - 1) * DX


def operator_at(freqs: tuple[float, ...]) -> Helmholtz2D:
    return Helmholtz2D(Helmholtz2DSurvey(
        sources=tuple(Helmholtz2DSource(position=(float(x), 1.5 * DZ),
                                        amplitude=1.0 + 0.0j, shot_id=s)
                      for s, x in enumerate(src_x)),
        receivers=tuple(Helmholtz2DReceiver(position=(float(x), 1.5 * DZ),
                                            shot_id=s)
                        for s in range(N_SHOT) for x in rcv_x),
        frequencies=freqs, n_pml=12,
    ))


with torch.no_grad():
    (d_obs,) = operator_at(FREQS)(
        ModelState(tensors={"vp": vp_true}), ctx).fetch("p")
d_obs = d_obs[..., 0]                            # (n_trace, n_freq), complex
# noise proportional to each trace's own amplitude, so far offsets keep
# their signal-to-noise instead of drowning under a global sigma
d_obs = d_obs * (1.0 + 0.01 * (torch.randn_like(d_obs.real)
                               + 1j * torch.randn_like(d_obs.real)))
print(f"[2] observed: {N_SHOT} shots x {N_RCV} receivers x {len(FREQS)} "
      f"frequencies = {d_obs.numel()} complex samples (1% noise)")

# %% 3. The starting model a real project would own ------------------------
#
# The true section smoothed over ~240 m: the trend a velocity analysis
# would give you, with every layer boundary destroyed, so each reflector
# in the answer was built by the data.
#
# The smoothing length is a real choice, not a cosmetic one. Smooth by
# much less and there is nothing left for FWI to legitimately add, since at
# 11 Hz and 2500 m/s the half-wavelength is already ~110 m, so a start
# smoothed at 100 m is ALREADY the answer at every recoverable
# wavenumber, and any apparent success is the starting model talking.
SMOOTH_CELLS = 15.0
vp_init = smooth(vp_true, sigma_cells=SMOOTH_CELLS)
log_v0 = vp_init.log()
log_v = log_v0.clone().requires_grad_(True)
print(f"[3] starting model: truth smoothed over {SMOOTH_CELLS * DZ:.0f} m, "
      f"rms difference {float((vp_true - vp_init).pow(2).mean().sqrt()):.0f} "
      "m/s")

# %% 4. Multi-scale FWI ----------------------------------------------------
DAMPING = 1.0
history: list[float] = []
band_start: list[int] = []
# One model snapshot per evaluation, for the animation at the end. Storing
# a 2-D field per iteration is cheap next to the wave solves.
snapshots: list[torch.Tensor] = []
t0 = time.time()
for k_f, f in enumerate(FREQS):
    op_f = operator_at((f,))
    d_f = d_obs[:, k_f].reshape(N_SHOT, N_RCV)
    trace_energy = d_f.abs().pow(2).clamp_min(1e-30)
    band_start.append(len(history))
    optimizer = torch.optim.LBFGS([log_v], lr=1.0, max_iter=30,
                                  history_size=12,
                                  line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        (d_pred,) = op_f(ModelState(tensors={"vp": log_v.exp()}), ctx).fetch("p")
        res = (d_pred[..., 0, 0].reshape(N_SHOT, N_RCV) - d_f).abs().pow(2)
        misfit = (res / trace_energy).mean()
        (misfit + smoothness(log_v - log_v0, dx=DX, dz=DZ,
                             weight=DAMPING)).backward()
        history.append(float(misfit.detach()))
        snapshots.append(log_v.exp().detach().clone())
        return misfit

    optimizer.step(closure)
    print(f"[4] {f:4.0f} Hz band: misfit {history[band_start[-1]]:.4f} "
          f"-> {history[-1]:.4f} in {len(history) - band_start[-1]} "
          "evaluations")
vp_rec = log_v.exp().detach()
print(f"[4] {len(history)} iterations over {len(FREQS)} bands in "
      f"{time.time() - t0:.0f} s")

# %% 5. How much of the section came back? --------------------------------
shallow = slice(0, NZ // 2)
deep = slice(NZ // 2, NZ)


def rms(a: torch.Tensor) -> float:
    return float(a.pow(2).mean().sqrt())


for name, band in (("shallow half", shallow), ("deep half", deep)):
    before = rms(vp_true[band] - vp_init[band])
    after = rms(vp_true[band] - vp_rec[band])
    print(f"[5] {name}: rms error {before:.0f} -> {after:.0f} m/s "
          f"({(1 - after / before) * 100:.0f}% of the gap closed)")

# %% 6. Picture ------------------------------------------------------------
#
# The two velocity panels sit on one scale under one colour bar - the
# comparison between them IS the result, and per-panel bars would let each
# rescale the difference away. The starting model is not drawn: the update
# panel below is what it was worth.
fig, axes = figure(2, 2)
EXTENT = (0.0, NX * DX, NZ * DZ, 0.0)
clim = dict(vmin=float(vp_true.min()), vmax=float(vp_true.max()))

for ax, values, title in ((axes[0, 0], vp_true, "Marmousi II - the truth"),
                          (axes[0, 1], vp_rec, "Recovered by FWI")):
    image = field(ax, values.numpy(), extent=EXTENT, cmap=CMAP_VELOCITY,
                  title=title, xlabel="Distance [m]", ylabel="Depth [m]",
                  **clim)
shared_colorbar(fig, image, axes[0, :], "vp [m/s]")
axes[0, 0].plot(src_x.numpy(), [1.5 * DZ] * N_SHOT, "*", ms=11, color="white",
                mec="black", mew=0.5, ls="none", clip_on=False, label="Shots")
axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.85)

update = (vp_rec - vp_init).numpy()
lim = symmetric_limits(update, quantile=0.995)
image = field(axes[1, 0], update, extent=EXTENT, cmap=CMAP_ANOMALY,
              vmin=lim[0], vmax=lim[1],
              title="Recovered - start",
              xlabel="Distance [m]", ylabel="Depth [m]")
shared_colorbar(fig, image, axes[1, 0], "vp update [m/s]",
                location="bottom")

ax = axes[1, 1]
plot_convergence(history, xlabel="Iteration", ylabel="Normalized misfit",
                 ax=ax)
for k, start in enumerate(band_start):
    ax.axvline(start, color="gray", ls=":", lw=1.0)
    ax.annotate(f"{FREQS[k]:.0f} Hz", xy=(start, max(history)), xytext=(3, -8),
                textcoords="offset points", fontsize=8, color="gray")
ax.set_title("Multi-scale convergence")
ax.grid(which="both")

# %% 7. The multi-scale climb, as an animation -----------------------------
#
# The point of a multi-scale schedule is that each band adds a finer layer
# of structure onto what the last one settled. That is a sequence, and the
# still above can only show its two endpoints.
anim_fig, anim_ax = figure(1, 1, panel_w=8.6, panel_h=3.6)
anim_image = field(anim_ax, snapshots[0].numpy(), extent=EXTENT,
                   cmap=CMAP_VELOCITY, xlabel="Distance [m]",
                   ylabel="Depth [m]", **clim)
anim_fig.colorbar(anim_image, ax=anim_ax, label="vp [m/s]")
step = max(1, len(snapshots) // 26)
anim_indices = list(range(0, len(snapshots), step)) + [len(snapshots) - 1]


def draw_fwi(index: int) -> None:
    k = anim_indices[index]
    band = sum(1 for b in band_start if b <= k)
    anim_image.set_data(snapshots[k].numpy())
    anim_ax.set_title(f"FWI iteration {k + 1} of {len(snapshots)} - "
                      f"{FREQS[band - 1]:.0f} Hz band")


animation(anim_fig, draw_fwi, len(anim_indices), OUT / "01_fwi_climb.gif")

fig.savefig(OUT / "01_seismic_fwi.png")
print(f"saved {OUT / '01_seismic_fwi.png'}")
plt.show()
