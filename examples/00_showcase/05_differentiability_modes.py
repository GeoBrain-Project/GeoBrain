"""Three ways to make a gradient, one contract, and all three are exact.

A differentiable platform is only useful if you can trust the derivative
and afford to compute it. GeoBrain does not have one gradient mechanism;
it has several, because different numerics need different ones. What it
does have is a single contract: every operator DECLARES which mechanism
it uses, in its :class:`DifferentiabilitySpec`, before you run it.

This script takes one buried anomaly, asks three operators for the
gradient of a data misfit, and checks every one against central finite
differences:

    FULL_AUTOGRAD   Acoustic2D: time-domain finite differences. Autograd
                    unrolls all 500 timesteps and walks back through them.

    IMPLICIT_VJP    Helmholtz2D: the same wave equation in the frequency
                    domain. Nothing is unrolled: at the solved linear
                    system the implicit function theorem replaces the whole
                    factorisation with ONE adjoint solve.

    CUSTOM_VJP      Gravity2D: a hand-written backward that is neither of
                    the above.

The pairing is the point. Acoustic2D and Helmholtz2D are the same physics,
so the mechanism is a property of the NUMERICS, not of the equation. And
because Gravity2D's level is decided when its execution policy binds, you
read it off the instance, not the class.

Where adjoint and finite difference disagree at all, the finite
difference is the one at fault, and the script proves it rather than
asserting it: shrink the step by three and the gap falls by nine, the
O(h²) signature of a central difference converging onto an exact
derivative. That is the honest way to read any gradient check: the
reference is the approximation.

The last panel is why any of this matters. All three gradients cost one
backward pass, whatever the mechanism. Finite differences would cost two
forward runs per parameter, so for the 3200-cell model here that is 6400
runs, and for the time-domain operator, hours instead of seconds.

APIs featured:
    - operator.differentiability: the declared level, trainable inputs
      and output keys, readable before running anything
    - geobrain.physics.wave.Acoustic2D / Helmholtz2D
    - geobrain.physics.potential.Gravity2D

Expected runtime: < 2 min.

Outputs:
    out/05_differentiability_modes.png: the model, the three gradients,
    the finite-difference check, and the cost of the alternative.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    CMAP_ANOMALY,
    CMAP_VELOCITY,
    C_RECOVERED,
    PALETTE,
    apply_style,
    field,
    figure,
    shared_colorbar,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D
from geobrain.physics.wave import (
    Acoustic2D,
    Helmholtz2D,
    Helmholtz2DReceiver,
    Helmholtz2DSource,
    Seismic2DSurvey,
    Helmholtz2DSurvey,
    ricker,
)

apply_style()
torch.manual_seed(0)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

D = torch.float64
NZ, NX, DH = 40, 80, 20.0                      # 800 x 1600 m
mesh = TensorMesh(shape=(NZ, NX), spacing=(DH, DH))
ctx = ForwardContext.of(mesh=mesh)

# %% 1. One anomaly, seen three ways -------------------------------------
zz = torch.arange(NZ, dtype=D)[:, None] * DH
xx = torch.arange(NX, dtype=D)[None, :] * DH
blob = torch.exp(-(((xx - 800.0) / 220.0) ** 2 + ((zz - 420.0) / 130.0) ** 2))
vp_start = 1800.0 + 0.6 * zz.expand(NZ, NX)            # m/s
vp_true = vp_start + 320.0 * blob
rho_start = 1800.0 + 0.20 * zz.expand(NZ, NX)          # kg/m^3
rho_true = rho_start + 260.0 * blob
RHO_ACOUSTIC = torch.full((NZ, NX), 2200.0, dtype=D)
print(f"[1] {NZ}x{NX} cells at {DH:.0f} m; one anomaly at 800 m offset, "
      f"420 m depth")
print(f"    vp {float(vp_true.min()):.0f}-{float(vp_true.max()):.0f} m/s, "
      f"rho {float(rho_true.min()):.0f}-{float(rho_true.max()):.0f} kg/m3")

# %% 2. The three operators ----------------------------------------------
SRC_X = (400.0, 1200.0)
REC_X = torch.linspace(100.0, 1500.0, 24, dtype=D)
NT, DT, F0 = 500, 0.002, 9.0

rec_xz = torch.stack([REC_X, torch.full((len(REC_X),), 20.0, dtype=D)], dim=1)
acoustic = Acoustic2D(
    Seismic2DSurvey(
        source_positions=torch.tensor([[x, 20.0] for x in SRC_X], dtype=D),
        source_shot_index=torch.arange(len(SRC_X)),
        receiver_positions=torch.cat([rec_xz] * len(SRC_X)),
        receiver_shot_index=torch.cat([
            torch.full((len(REC_X),), s, dtype=torch.long)
            for s in range(len(SRC_X))]),
        nt=NT, dt=DT),
    ricker(NT, DT, F0, causal=True, dtype=D).expand(len(SRC_X), NT))

FREQS = (4.0, 7.0)
helmholtz = Helmholtz2D(Helmholtz2DSurvey(
    sources=tuple(Helmholtz2DSource(position=(x, 20.0), shot_id=s)
                  for s, x in enumerate(SRC_X)),
    receivers=tuple(Helmholtz2DReceiver(position=(float(x), 20.0), shot_id=s)
                    for s in range(len(SRC_X)) for x in REC_X),
    frequencies=FREQS, n_pml=12))

gravity = Gravity2D(PotentialSurvey2D(torch.stack([
    torch.linspace(100.0, 1500.0, 36, dtype=D),
    torch.ones(36, dtype=D)], dim=1)))


def run_acoustic(vp: torch.Tensor) -> torch.Tensor:
    return acoustic(ModelState({"vp": vp, "rho": RHO_ACOUSTIC}),
                    ctx).data["seismic"]


def run_helmholtz(vp: torch.Tensor) -> torch.Tensor:
    return helmholtz(ModelState({"vp": vp}), ctx).data["p"]


def run_gravity(rho: torch.Tensor) -> torch.Tensor:
    return gravity(ModelState({"rho": rho}), ctx).data["gz"]


# The level is DECLARED, and for Gravity2D it is decided per instance,
# only once the execution policy has bound. Read it off the object.
MODES = (
    ("Acoustic2D", "Time-domain wave", acoustic, run_acoustic,
     vp_true, vp_start, 5),
    ("Helmholtz2D", "Frequency-domain wave", helmholtz, run_helmholtz,
     vp_true, vp_start, 8),
    ("Gravity2D", "Potential field", gravity, run_gravity,
     rho_true, rho_start, 8),
)
for name, _, op, _, _, _, _ in MODES:
    spec = op.differentiability
    print(f"[2] {name:12s} declares {str(spec.level).split('.')[-1]:14s} "
          f"trainable {spec.trainable_inputs} -> {spec.output_keys}")

# %% 3. One backward each, then a finite-difference audit ----------------
#
# Same objective everywhere: half the squared data residual against the
# anomaly's own response, evaluated at the smooth starting model.
FD_STEP = 1e-5
results = {}
for name, subtitle, op, run, truth, start, n_probe in MODES:
    with torch.no_grad():
        observed = run(truth)
    t0 = time.time()
    model = start.clone().requires_grad_(True)
    residual = run(model) - observed
    misfit = 0.5 * (residual.abs() ** 2).sum()
    (grad,) = torch.autograd.grad(misfit, model)
    t_adjoint = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        run(start)
    t_forward = time.time() - t0

    flat = start.reshape(-1)
    g_flat = grad.detach().reshape(-1)
    order = g_flat.abs().argsort(descending=True)
    # span decades of sensitivity rather than crowding the strongest cells
    picks = [int(order[k]) for k in
             torch.logspace(0, torch.tensor(float(order.numel() - 1)).log10(),
                            n_probe, dtype=D).long().clamp(max=order.numel() - 1)]
    pairs = []
    for cell in dict.fromkeys(picks):
        h = FD_STEP * abs(float(flat[cell]))
        shifted = []
        for sign in (1.0, -1.0):
            probe = flat.clone()
            probe[cell] += sign * h
            with torch.no_grad():
                r = run(probe.reshape(start.shape)) - observed
                shifted.append(float(0.5 * (r.abs() ** 2).sum()))
        pairs.append((float(g_flat[cell]), (shifted[0] - shifted[1]) / (2 * h)))

    worst = max(abs(a - f) / max(abs(f), 1e-300) for a, f in pairs)
    results[name] = dict(subtitle=subtitle, level=str(op.differentiability.level
                                                     ).split(".")[-1],
                         grad=grad.detach(), pairs=pairs, worst=worst,
                         t_adjoint=t_adjoint, t_forward=t_forward)
    print(f"[3] {name:12s} adjoint {t_adjoint:6.2f} s for "
          f"{start.numel()} derivatives; agrees with finite differences to "
          f"{worst:.1e} over {len(pairs)} probes")

# %% 4. Whose error is it? ------------------------------------------------
#
# The residual disagreement is the FINITE DIFFERENCES', not the adjoint's.
# Halving the step quarters the gap: the O(h²) signature of a central
# difference converging onto an exact derivative.
name, _, _, run, truth, start, _ = MODES[0]
with torch.no_grad():
    observed = run(truth)
flat = start.reshape(-1)
cell = int(results[name]["grad"].reshape(-1).abs().argmax())
exact = float(results[name]["grad"].reshape(-1)[cell])
print(f"[4] {name} step study at its most sensitive cell "
      f"(adjoint {exact:+.8e}):")
for rel_step in (1e-4, 3e-5, 1e-5):
    h = rel_step * abs(float(flat[cell]))
    shifted = []
    for sign in (1.0, -1.0):
        probe = flat.clone()
        probe[cell] += sign * h
        with torch.no_grad():
            r = run(probe.reshape(start.shape)) - observed
            shifted.append(float(0.5 * (r.abs() ** 2).sum()))
    fd = (shifted[0] - shifted[1]) / (2 * h)
    print(f"    step {h:7.4f} m/s -> finite difference {fd:+.8e}, "
          f"off by {abs(exact - fd) / abs(fd):.2e}")

# %% 5. What the alternative would have cost -----------------------------
N_PARAM = int(vp_start.numel())
for name in results:
    r = results[name]
    fd_cost = 2 * N_PARAM * r["t_forward"]
    r["t_fd"] = fd_cost
    print(f"[4] {name:12s} one forward {r['t_forward']:.3f} s -> finite "
          f"differences would need {2 * N_PARAM} of them: {fd_cost / 3600:.2f} h "
          f"against {r['t_adjoint']:.2f} s ({fd_cost / r['t_adjoint']:.0f}x)")

# %% 6. Picture -----------------------------------------------------------
fig, axes = figure(2, 3)
extent = (0.0, NX * DH, NZ * DH, 0.0)

ax = axes[1, 0]
im = field(ax, vp_true.numpy(), extent=extent, cmap=CMAP_VELOCITY)
ax.plot(list(SRC_X), [20.0] * len(SRC_X), "*", ms=14, color=C_RECOVERED,
        ls="none", clip_on=False, label="Sources")
ax.plot(REC_X.numpy(), [20.0] * len(REC_X), "v", ms=5, color="black",
        mec="white", mew=0.5, ls="none", clip_on=False, label="Receivers")
ax.set(title="One anomaly, three physics", xlabel="Distance [m]",
       ylabel="Depth [m]")
ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
shared_colorbar(fig, im, ax, "vp [m/s]")

ACQUISITION_ROWS = 3           # the source/receiver line saturates any scale
# The three gradients differ in magnitude by ten orders (a wave equation, a
# Helmholtz solve and a potential field), so their raw scales are not
# comparable and three colour bars would only report that. Each panel is
# normalised to its own subsurface maximum, which puts the SHAPES side by
# side - and the shapes are what "the gradient is exact" is about.
for ax, name in zip(axes[0, :], results):
    r = results[name]
    g = r["grad"].numpy()
    # scale to the subsurface, not to the acquisition line - otherwise the
    # perforated surface rows are the only thing the eye can see
    lim = float(abs(g[ACQUISITION_ROWS:]).max())
    im = field(ax, g / lim, extent=extent, cmap=CMAP_ANOMALY, vmin=-1.0,
               vmax=1.0, title=f"{r['level']} - {name}",
               xlabel="Distance [m]", ylabel="Depth [m]")
    ax.annotate(r["subtitle"], xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=8, color="white",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "black",
                      "alpha": 0.45, "edgecolor": "none"})
shared_colorbar(fig, im, axes[0, :],
                "Misfit gradient, normalised per panel", extend="both")

ax = axes[1, 1]
for k, name in enumerate(results):
    pairs = results[name]["pairs"]
    scale = max(abs(f) for _, f in pairs)
    ax.plot([f / scale for _, f in pairs], [a / scale for a, _ in pairs],
            "o", ms=7, color=PALETTE[k], ls="none", alpha=0.85,
            label=f"{name} ({results[name]['worst']:.0e})")
ax.plot([-1.1, 1.1], [-1.1, 1.1], color="gray", ls="--", lw=1.2,
        label="1 : 1")
ax.set(title="Adjoint against finite differences",
       xlabel="Finite-difference derivative (normalized)",
       ylabel="Adjoint derivative (normalized)",
       xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
ax.legend(fontsize=8, loc="upper left")

ax = axes[1, 2]
names = list(results)
width = 0.36
positions = torch.arange(len(names), dtype=D).numpy()
ax.bar(positions - width / 2, [results[n]["t_adjoint"] for n in names],
       width, color=PALETTE[0], label="One adjoint (all gradients)")
ax.bar(positions + width / 2, [results[n]["t_fd"] for n in names], width,
       color=PALETTE[1], label=f"Finite differences ({2 * N_PARAM} forwards)")
for k, name in enumerate(names):
    ratio = results[name]["t_fd"] / results[name]["t_adjoint"]
    ax.annotate(f"{ratio:,.0f}x", xy=(positions[k], results[name]["t_fd"]),
                xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=8, color=PALETTE[1])
top = max(results[n]["t_fd"] for n in names)
ax.set(title="What the adjoint buys", ylabel="Wall-clock [s]", yscale="log",
       ylim=(min(results[n]["t_adjoint"] for n in names) * 0.2, top * 300.0))
ax.set_xticks(positions)
ax.set_xticklabels(names, fontsize=9)
ax.legend(fontsize=8, loc="upper left")
ax.grid(which="both", axis="y")

fig.savefig(OUT / "05_differentiability_modes.png")
print(f"saved {OUT / '05_differentiability_modes.png'}")
plt.show()
