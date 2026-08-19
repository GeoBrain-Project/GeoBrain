"""The inversion toolbox: the knobs, and what each one buys you.

``InverseProblem`` says WHAT is being inverted; ``Inverter`` says HOW.
This script opens the second box. One gravity problem (two buried
bodies, 3% noise, 392 unknowns from 64 stations) is inverted four times,
changing nothing but the configuration, so the effect of each choice is
visible side by side:

    A  smoothness only          the classic L2 answer: smeared, shallow
    B  + depth weighting        Li & Oldenburg: the mass drops to depth
    C  + a reference model      interpretation enters as tikhonov(m_ref=...)
    D  + IRLS compactness       the industry recipe: compact, full amplitude

(``total_variation`` is in the library too, but it is worth knowing what
it does NOT do: on an under-determined potential-field problem with a
zero start there are no sharp contrasts yet for TV to preserve, so raising
its weight only damps the amplitude. TV earns its keep when the starting
model already carries edges, after a pass of D, for instance.)

Everything else is shared. The pieces you configure are:

    likelihood   how data misfit is scored (noise model)
    regularizer  ANY callable over the parameter dict: the library ships
                 smoothness, smallness, tikhonov, total_variation,
                 depth_weighting, l1/l2, cross_gradient, gmm_prior
    bounds       projected after every step; physical, not numerical
    optimizer    AdamConfig (robust, slow) or LBFGSConfig (fast, needs a
                 sane starting model)
    result       InversionResult carries best_params, best_loss, best_iter,
                 loss_history, data/reg split, wall_clock_sec, stop_reason

APIs featured:
    - geobrain.inverse.InverseProblem, GaussianLikelihood
    - geobrain.optim.Inverter, AdamConfig, LBFGSConfig, InversionResult
    - geobrain.optim.regularizers: smoothness / smallness / tikhonov /
      depth_weighting (and total_variation, l1, l2, cross_gradient, ...)

Expected runtime: < 90 s.

Outputs:
    out/05_inversion_toolbox.png: the truth, four recipes, and what the
    choice costs in data fit and convergence.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    CMAP_ANOMALY,
    PALETTE,
    apply_style,
    field,
    figure,
    shared_colorbar,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.inverse import GaussianLikelihood, InverseProblem
from geobrain.optim import AdamConfig, Inverter, LBFGSConfig
from geobrain.optim.regularizers import (
    depth_weighting,
    smallness,
    smoothness,
    tikhonov,
)
from geobrain.mesh import TensorMesh
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D

apply_style()
torch.manual_seed(7)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. One problem, defined once ------------------------------------------
NZ, NX, D = 14, 28, 25.0
mesh = TensorMesh(shape=(NZ, NX), spacing=(D, D))
xs = torch.linspace(-150.0, 850.0, 64, dtype=torch.float64)
gravity = Gravity2D(PotentialSurvey2D(torch.stack([xs, torch.ones_like(xs)], 1)))
ctx = ForwardContext.of(mesh=mesh)

rho_true = torch.zeros(mesh.shape, dtype=torch.float64)
rho_true[4:8, 6:12] = 400.0                    # dense body
rho_true[8:11, 16:22] = -300.0                 # deeper mass deficit

(gz_clean,) = gravity(ModelState(tensors={"rho": rho_true}), ctx).fetch("gz")
noise_std = 0.03 * float(gz_clean.abs().max())
gz_obs = gz_clean + noise_std * torch.randn_like(gz_clean)

problem = InverseProblem(
    forward=gravity,
    observed={"gz": gz_obs},
    likelihood=GaussianLikelihood(std=noise_std),   # the noise model
)
print(f"[1] {rho_true.numel()} unknowns from {gz_obs.numel()} stations, "
      f"noise σ = {noise_std * 1e5:.3f} mGal")

# Li & Oldenburg depth weight: potential-field kernels decay fast, so
# without this the objective is cheapest to satisfy at the surface.
depths = ((torch.arange(NZ, dtype=torch.float64) + 0.5) * D).reshape(NZ, 1)
w2 = depth_weighting(depths, reference_depth=D, exponent=2.0) ** 2
BOUNDS = {"rho": (-450.0, 450.0)}
EPS = 20.0                                     # IRLS compactness focus

# %% 2. Four recipes, one problem ------------------------------------------
def run(regularizer, *, passes: int = 1, label: str) -> tuple:
    rho = torch.zeros(mesh.shape, dtype=torch.float64)
    history: list[float] = []
    for _ in range(passes):
        wc = EPS**2 / (rho**2 + EPS**2)        # IRLS weight (1.0 when passes=1)
        result = Inverter(
            problem,
            params={"rho": rho.clone()},
            optimizer=LBFGSConfig(lr=0.8, max_iter=20),
            regularizer=lambda p, wc=wc: regularizer(p["rho"], wc),
            bounds=BOUNDS,
            ctx=ctx,
        ).run(n_iters=25)
        rho = result.best_params["rho"].detach()
        history.extend(float(v) for v in result.loss_history)
    (fit,) = gravity(ModelState(tensors={"rho": rho}), ctx).fetch("gz")
    rms = float((fit - gz_obs).pow(2).mean().sqrt() / noise_std)
    print(f"[2] {label:26s} RMS={rms:4.2f}  peak=[{float(rho.min()):+5.0f},"
          f" {float(rho.max()):+5.0f}]  wall={result.wall_clock_sec:4.1f}s  "
          f"stop={result.stop_reason}")
    return rho, rms, history

recipes = {}
# every weight below was tuned so the four recipes fit the data equally
# well (normalized RMS ~1), or this would compare apples to oranges
recipes["A · smoothness"] = run(
    lambda m, wc: smoothness(m, dx=D, dz=D, weight=4e-3),
    label="A smoothness only")
recipes["B · + depth weight"] = run(
    lambda m, wc: smoothness(m, dx=D, dz=D, weight=0.01, cell_weights=w2)
    + smallness(m, weight=0.002 * w2),
    label="B + depth weighting")
# a geologist's guess: the dense body roughly where the map says, the
# deficit not yet interpreted. tikhonov pulls the answer toward it exactly
# where the data have nothing to say.
m_ref = torch.zeros(mesh.shape, dtype=torch.float64)
m_ref[4:8, 6:12] = 250.0
recipes["C · + reference model"] = run(
    lambda m, wc: smoothness(m, dx=D, dz=D, weight=0.01, cell_weights=w2)
    + tikhonov(m, m_ref=m_ref, weight=0.0012),
    label="C + reference model")
recipes["D · + IRLS compact"] = run(
    lambda m, wc: smoothness(m, dx=D, dz=D, weight=0.01, cell_weights=w2)
    + smallness(m, weight=0.002 * w2 * wc),
    passes=4, label="D + IRLS compactness")

# %% 3. Optimizer choice is a separate knob --------------------------------
adam = Inverter(problem, params={"rho": torch.zeros(mesh.shape,
                                                    dtype=torch.float64)},
                optimizer=AdamConfig(lr=8.0),
                regularizer=lambda p: smoothness(p["rho"], dx=D, dz=D,
                                                 weight=0.01, cell_weights=w2)
                + smallness(p["rho"], weight=0.002 * w2),
                bounds=BOUNDS, ctx=ctx).run(n_iters=400)
print(f"[3] same recipe B under Adam: best loss {adam.best_loss:.1f} in "
      f"{adam.completed_iters} iterations ({adam.wall_clock_sec:.1f} s) vs "
      f"L-BFGS {recipes['B · + depth weight'][2][-1]:.1f} in 25")

# %% 4. Picture ------------------------------------------------------------
fig, axes = figure(2, 3)
extent = (0.0, NX * D, NZ * D, 0.0)
vlim = 450.0

# Truth and four recipes on ONE scale and ONE bar - they are the same
# quantity, and the comparison between them is the entire point.
im = field(axes[0, 0], rho_true.numpy(), extent=extent, cmap=CMAP_ANOMALY,
           vmin=-vlim, vmax=vlim, title="Truth", ylabel="Depth [m]")

panels = [(0, 1), (0, 2), (1, 0), (1, 1)]
for (r, c), (label, (rho, rms, _)) in zip(panels, recipes.items()):
    ax = axes[r, c]
    field(ax, rho.numpy(), extent=extent, cmap=CMAP_ANOMALY, vmin=-vlim,
          vmax=vlim, title=f"{label}  (RMS {rms:.2f})",
          ylabel="Depth [m]" if c == 0 else None,
          xlabel="Distance [m]" if r == 1 else None)
shared_colorbar(fig, im, [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0],
                          axes[1, 1]], r"$\Delta\rho$ [kg/m$^3$]")

for k, (label, (_, _, history)) in enumerate(recipes.items()):
    axes[1, 2].semilogy(history, color=PALETTE[k], lw=1.8, label=label)
axes[1, 2].set(title="Convergence by regularizer",
               xlabel="Evaluation", ylabel="Objective")
axes[1, 2].grid(which="both")
axes[1, 2].legend(fontsize=7)

fig.savefig(OUT / "05_inversion_toolbox.png")
print(f"saved {OUT / '05_inversion_toolbox.png'}")
plt.show()
