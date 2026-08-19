"""Hello, GeoBrain: the whole platform on one screen.

Every workflow in GeoBrain is the same three objects wired together:

1. a FORWARD: physics as a differentiable operator;
2. an :class:`InverseProblem`: forward + observed data + noise model
   bound together; it knows nothing about optimizers;
3. an :class:`Inverter`: a solver configuration pointed at the problem.

The demo recovers a 2-D density image from one noisy gravity profile,
and does it with the full industry recipe, because the pieces are all
stock parts: Li & Oldenburg depth weighting (without it, least squares
plates all mass at the surface), physical bound constraints, and
IRLS compactness, a six-line outer loop, since a regularizer is just a
callable rebuilt from the previous iterate.

APIs featured:
    - geobrain.physics.potential.Gravity2D, PotentialSurvey2D
    - geobrain.inverse.InverseProblem, GaussianLikelihood
    - geobrain.optim.Inverter, LBFGSConfig
    - geobrain.optim.regularizers.depth_weighting / smoothness / smallness

Expected runtime: < 2 min.

Outputs:
    out/01_gravity_inversion.png: truth, recovery, data fit, convergence.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    C_OBSERVED,
    C_PREDICTED,
    C_SERIES,
    CMAP_ANOMALY,
    apply_style,
    field,
    figure,
    outer_labels,
    shared_colorbar,
    symmetric_limits,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.inverse import GaussianLikelihood, InverseProblem
from geobrain.mesh import TensorMesh
from geobrain.optim import Inverter, LBFGSConfig
from geobrain.optim.regularizers import depth_weighting, smallness, smoothness
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D

apply_style()
torch.manual_seed(7)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

# %% 1. The FORWARD: physics bound to an acquisition -----------------------
NZ, NX, D = 14, 28, 25.0
mesh = TensorMesh(shape=(NZ, NX), spacing=(D, D))   # 350 m x 700 m section
xs = torch.linspace(-150.0, 850.0, 64, dtype=torch.float64)   # flanking coverage
gravity = Gravity2D(PotentialSurvey2D(torch.stack([xs, torch.ones_like(xs)], 1)))
ctx = ForwardContext.of(mesh=mesh)

# %% 2. Observed data: two buried bodies, 3% noise -------------------------
rho_true = torch.zeros(mesh.shape, dtype=torch.float64)
rho_true[4:8, 6:12] = 400.0                    # dense body
rho_true[8:11, 16:22] = -300.0                 # deeper mass deficit

(gz_clean,) = gravity(ModelState(tensors={"rho": rho_true}), ctx).fetch("gz")
noise_std = 0.03 * float(gz_clean.abs().max())
gz_obs = gz_clean + noise_std * torch.randn_like(gz_clean)
print(f"observed: {gz_obs.numel()} stations, noise σ = {noise_std * 1e5:.3f} mGal")

# %% 3. The PROBLEM: physics + data + noise model --------------------------
problem = InverseProblem(
    forward=gravity,
    observed={"gz": gz_obs},
    likelihood=GaussianLikelihood(std=noise_std),
)

# %% 4. The SOLVER: depth weighting + bounds + IRLS compactness ------------
#
# Gravity kernels decay fast with depth, so the objective needs the
# standard Li & Oldenburg weight (β = 2) or everything plates at the
# surface. The compactness loop is plain IRLS: after each pass the
# smallness weight is rebuilt as ε²/(m² + ε²), cheap where the model has
# committed to a body, expensive in the background, which sharpens the
# blobs of the L2 answer into compact bodies at full amplitude.
depths = ((torch.arange(NZ, dtype=torch.float64) + 0.5) * D).reshape(NZ, 1)
w2 = depth_weighting(depths, reference_depth=D, exponent=2.0) ** 2
EPS = 20.0                                     # compactness focus [kg/m^3]

rho = torch.zeros(mesh.shape, dtype=torch.float64)
history: list[float] = []
passes: list[int] = []
for irls_pass in range(4):
    wc = EPS**2 / (rho**2 + EPS**2)
    result = Inverter(
        problem,
        params={"rho": rho.clone()},
        optimizer=LBFGSConfig(lr=0.8, max_iter=20),
        regularizer=lambda p, wc=wc: (
            smoothness(p["rho"], dx=D, dz=D, weight=0.1, cell_weights=w2)
            + smallness(p["rho"], weight=0.02 * w2 * wc)),
        bounds={"rho": (-450.0, 450.0)},
        ctx=ctx,
    ).run(n_iters=25)
    rho = result.best_params["rho"].detach()
    passes.append(len(history))
    history.extend(float(v) for v in result.loss_history)
    print(f"IRLS pass {irls_pass + 1}: best loss {result.best_loss:.2f}")

(gz_fit,) = gravity(ModelState(tensors={"rho": rho}), ctx).fetch("gz")
rms = float((gz_fit - gz_obs).pow(2).mean().sqrt() / noise_std)
print(f"normalized data RMS = {rms:.2f} (1 = fit to the noise level)")

# %% 5. Picture ------------------------------------------------------------
fig, axes = figure(2, 2)
extent = (0.0, NX * D, NZ * D, 0.0)
vlim = symmetric_limits(rho_true.numpy(), rho.numpy())

for ax, values, title in ((axes[0, 0], rho_true.numpy(), "Truth"),
                          (axes[0, 1], rho.numpy(),
                           "Recovered: depth-weighted IRLS, one profile")):
    image = field(ax, values, extent=extent, cmap=CMAP_ANOMALY, vmin=vlim[0],
                  vmax=vlim[1], title=title)
shared_colorbar(fig, image, axes[0, :], r"$\Delta\rho$ [kg/m$^3$]")

# Gravity2D returns elevation-up gz; geophysical displays are down-positive
axes[1, 0].plot(xs.numpy(), (-gz_obs * 1e5).numpy(), ".", ms=5,
                color=C_OBSERVED, label="Observed")
axes[1, 0].plot(xs.numpy(), (-gz_fit * 1e5).numpy(), color=C_PREDICTED,
                lw=2.2, label="Predicted")
axes[1, 0].axvspan(0.0, NX * D, color="gray", alpha=0.08)
axes[1, 0].set(title=f"Data fit (normalized RMS {rms:.2f})",
               xlabel="Distance [m]",
               ylabel="Downward attraction [mGal]")
axes[1, 0].legend()

axes[1, 1].semilogy(history, color=C_SERIES, lw=2.0)
for k, b in enumerate(passes[1:], start=2):
    axes[1, 1].axvline(b, color="gray", ls="--", lw=0.8)
    axes[1, 1].text(b + 1, max(history) * 0.5, f"IRLS {k}", rotation=90,
                    fontsize=8, color="gray", va="top")
axes[1, 1].set(title="Convergence across IRLS passes",
               xlabel="Evaluation", ylabel="Loss")
axes[1, 1].grid(which="both")
outer_labels(axes[0, :], "Distance [m]", "Depth [m]")

fig.savefig(OUT / "01_gravity_inversion.png")
print(f"saved {OUT / '01_gravity_inversion.png'}")
plt.show()
