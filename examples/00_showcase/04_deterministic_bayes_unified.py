"""Deterministic and Bayesian AVO inversion: one API, a one-line pivot.

The platform does not have a deterministic stack and a Bayesian stack.
It has ONE object and two Tier-1 doors on it. The entire workflow of
this script is::

    problem   = InverseProblem(forward=..., observed=..., likelihood=..., prior=...)

    # deterministic inversion
    result    = problem.create_inverter(params={"phi": phi0},
                                        optimizer=LBFGSConfig(...)).run(n_iters=60)

    # Bayesian uncertainty quantification: the pivot is ONE line
    posterior = problem.as_posterior(transforms={"phi": IntervalTransform(...)})
    draws     = posterior.sample("nuts", params={"phi": phi_map}, n_iters=300, ...)
    # ...and "hmc" / "langevin" / "svgd" are the same one-keyword swap.

The physics is the full rock-physics-driven AVO problem: a POROSITY
profile drives Voigt–Reuss–Hill mineral moduli, Hertz–Mindlin contacts,
the soft-sand line, Gassmann fluid substitution and a density model to
(vp, vs, rho), which an Aki–Richards kernel convolves into a nine-angle
gather. All of it sits one PropertyTransform link ahead of the AVO
operator, so the porosity gradient flows through five rock-physics laws
without a single hand-written derivative. The Inverter DESCENDS
``-log posterior`` (its loss at the MAP matches
``-problem.log_posterior`` to the digit, printed below); NUTS EXPLORES
the same function through the same object, bounded by
``IntervalTransform`` so porosity can never leave [0.05, 0.40], and the
posterior on porosity propagates through the same rock physics into a
posterior on velocity for free.

APIs featured:
    - InverseProblem.create_inverter: Tier-1 deterministic door
    - InverseProblem.as_posterior + IntervalTransform: Tier-1 pivot
    - Posterior.sample("nuts"): one posterior, any sampler
    - rock physics as a chain link: VRH / hertz_mindlin_moduli /
      SoftSand / gassmann_k_sat / DensityModel / velocities_from_moduli
    - geobrain.physics.wave.ConvolutionalAVO (Aki–Richards + wavelet)

Expected runtime: < 3 min.

Outputs:
    out/04_deterministic_bayes_unified.png: a well display with the observed
    gather with the MAP residual quoted on it, the porosity the two doors
    recover, and the velocity the same posterior propagates into.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    CMAP_SEISMIC,
    C_POSTERIOR,
    C_RECOVERED,
    C_TRUTH,
    apply_style,
    field,
    figure,
    shared_colorbar,
)
from geobrain.bayes import IntervalTransform, ess
from geobrain.core import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
    ForwardContext,
    ModelState,
    PropertyTransform,
)
from geobrain.inverse import GaussianLikelihood, InverseProblem
from geobrain.optim import LBFGSConfig
from geobrain.physics.rock import hertz_mindlin_moduli, velocities_from_moduli
from geobrain.physics.rock.models import VRH, DensityModel, SoftSand
from geobrain.physics.rock.models._fluid_gassmann import gassmann_k_sat
from geobrain.physics.wave import ConvolutionalAVO, ricker

apply_style()
torch.manual_seed(7)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
C_BAYES = C_POSTERIOR

# %% 1. Rock physics as ONE chain link -------------------------------------
#
# porosity -> VRH mineral moduli -> Hertz-Mindlin contacts -> soft-sand
# line -> Gassmann brine substitution -> density -> (vp, vs, rho).
# Declaring it a PropertyTransform is all it takes to sit in a chain.
PHI_LO, PHI_HI = 0.05, 0.40

class PorosityToElastic(PropertyTransform):
    """phi -> (vp, vs, rho) through five differentiable rock-physics laws."""

    differentiability = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("phi",),
        output_keys=("vp", "vs", "rho"),
    )

    def __init__(self) -> None:
        super().__init__()
        f64 = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
        vrh = VRH()
        k_min = vrh(f64(36.6e9), f64(21.0e9), f64(0.9))   # 90% quartz, 10% clay
        mu_min = vrh(f64(44.0e9), f64(9.0e9), f64(0.9))
        hm = hertz_mindlin_moduli(f64(20.0e6), k_min, mu_min, f64(PHI_HI),
                                  f64(7.0))               # 20 MPa, C = 7
        self.k_min = float(k_min)
        self.k_hm, self.mu_hm = hm.k_dry, hm.mu_dry
        self.soft_sand = SoftSand(K_mineral=self.k_min,
                                  mu_mineral=float(mu_min),
                                  phi_critical=PHI_HI)
        self.density = DensityModel()
        self.k_brine, self.rho_brine = 3.06e9, 1080.0

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        (phi,) = state.fetch("phi")
        k_dry, mu_dry = self.soft_sand(self.k_hm, self.mu_hm, phi)
        k_sat = gassmann_k_sat(k_dry, self.k_min, self.k_brine, phi)
        rho = self.density(phi, torch.tensor(2635.0, dtype=torch.float64),
                           torch.tensor(self.rho_brine, dtype=torch.float64))
        vp, vs = velocities_from_moduli(k_sat, mu_dry, rho)
        return state.with_tensors(vp=vp, vs=vs, rho=rho)

# %% 2. ONE problem: porosity -> rock physics -> a nine-angle gather -------
NT, DZ = 96, 1.0
z_idx = torch.arange(NT, dtype=torch.float64)
# a smooth multi-scale profile whose wavelengths sit INSIDE the seismic
# band (11-34 samples), so band-limited data can actually resolve it
phi_true = (0.18
            + 0.055 * torch.sin(2 * math.pi * z_idx / 34.0 + 0.5)
            + 0.035 * torch.sin(2 * math.pi * z_idx / 19.0 + 2.0)
            + 0.020 * torch.sin(2 * math.pi * z_idx / 11.0 + 4.0)
            ).clamp(PHI_LO, PHI_HI)

ANGLES = [float(a) for a in range(4, 45, 5)]       # 4° .. 44°, nine stacks
wavelet = ricker(31, 0.004, 30.0, dtype=torch.float64)
forward = ConvolutionalAVO(ANGLES, wavelet) @ PorosityToElastic()
ctx = ForwardContext()
print(f"forward: {forward}")
print(f"  contract: {forward.differentiability.trainable_inputs} -> "
      f"{forward.differentiability.output_keys}")

(gather_clean,) = forward(ModelState(tensors={"phi": phi_true}), ctx).fetch("trace")
noise_std = 0.05 * float(gather_clean.std())
gather_obs = gather_clean + noise_std * torch.randn_like(gather_clean)

PRIOR_MEAN, PRIOR_STD, SMOOTH_STD = 0.18, 0.10, 0.025

class PorosityPrior:
    """Marginal N(0.2, 0.1²) plus smoothness on adjacent samples: the
    deterministic world's damping + smoothing regularizers, renamed."""

    def log_prob(self, state: ModelState) -> torch.Tensor:
        (phi,) = state.fetch("phi")
        marginal = -0.5 * (((phi - PRIOR_MEAN) / PRIOR_STD) ** 2).sum()
        smooth = -0.5 * (((phi[1:] - phi[:-1]) / SMOOTH_STD) ** 2).sum()
        return marginal + smooth

    def sample(self, generator: torch.Generator | None = None) -> ModelState:
        draw = torch.randn((NT,), generator=generator, dtype=torch.float64)
        return ModelState(tensors={"phi": PRIOR_MEAN + PRIOR_STD * draw})

problem = InverseProblem(
    forward=forward,
    observed={"trace": gather_obs},
    likelihood=GaussianLikelihood(std=noise_std),
    prior=PorosityPrior(),
)
print(f"ONE problem: {NT} porosity unknowns, {gather_obs.numel()} samples "
      f"across {len(ANGLES)} angle stacks, likelihood x prior")

# %% 3. Door one, deterministic: create_inverter().run() -------------------
phi0 = torch.full((NT,), PRIOR_MEAN, dtype=torch.float64)
result = problem.create_inverter(
    params={"phi": phi0.clone()},
    optimizer=LBFGSConfig(lr=0.5, max_iter=10),
    bounds={"phi": (PHI_LO, PHI_HI)},
    ctx=ctx,
).run(n_iters=60)
phi_map = result.best_params["phi"].detach()

with torch.no_grad():
    lp_map = float(problem.log_posterior(
        ModelState(tensors={"phi": phi_map}), ctx))
print(f"Inverter best loss           = {result.best_loss:12.6f}")
print(f"-log_posterior at the MAP    = {-lp_map:12.6f}")
print("  -> same object, same surface: the MAP is the posterior's mode")

# %% 4. Door two, Bayesian: as_posterior() + sample("nuts") ----------------
posterior = problem.as_posterior(
    transforms={"phi": IntervalTransform(PHI_LO, PHI_HI)})
draws = posterior.sample(
    "nuts", params={"phi": phi_map.clone()}, n_iters=400, ctx=ctx,
    step_size=0.1, max_depth=6, warmup=250, target_accept=0.8,
    generator=torch.Generator().manual_seed(1),
)
chain_phi = draws.samples["phi"].detach()
print(f'posterior.sample("nuts"): {chain_phi.shape[0]} draws, median ESS '
      f"{float(ess(chain_phi).median()):.0f} "
      "(bounded by IntervalTransform, one keyword from the MAP)")

# the porosity posterior propagates through the SAME rock physics
rock = PorosityToElastic()
with torch.no_grad():
    vp_draws = torch.stack([
        rock(ModelState(tensors={"phi": chain_phi[i]})).fetch("vp")[0]
        for i in range(chain_phi.shape[0])])
    (vp_true_prof,) = rock(ModelState(tensors={"phi": phi_true})).fetch("vp")
    (vp_map_prof,) = rock(ModelState(tensors={"phi": phi_map})).fetch("vp")
    (gather_map,) = forward(ModelState(tensors={"phi": phi_map}),
                            ctx).fetch("trace")

# %% 5. Picture: a four-panel well display ---------------------------------
# Three panels read DOWN the page, so they are narrow and tall: depth is
# the axis the reader came for, and the canonical landscape panel flattens
# it. The gather keeps its own aspect through sharey.
fig, (ax_obs, ax_phi, ax_vp) = figure(
    1, 3, panel_w=3.3, panel_h=7.0, sharey=True,
    width_ratios=(1.0, 1.15, 1.15))
depth = (z_idx * DZ).numpy()
extent = (ANGLES[0] - 2.5, ANGLES[-1] + 2.5, depth[-1], 0.0)
lim = float(gather_obs.abs().max())

# The MAP synthetic is indistinguishable from the data at this scale, which
# is the whole point and also why drawing it twice teaches nothing. The
# residual is not drawn either: on the data's own scale it is a white
# rectangle, and a white rectangle is a NUMBER pretending to be a panel.
# The number goes in the title of the data it is a residual of.
residual = (gather_obs - gather_map).numpy()
image = field(ax_obs, gather_obs.numpy(), extent=extent, cmap=CMAP_SEISMIC,
              vmin=-lim, vmax=lim, interpolation="bilinear",
              title="Observed gather (MAP residual "
                    f"{100 * abs(residual).max() / lim:.0f}% of peak)",
              xlabel="Angle [deg]", ylabel="Depth [m]")
shared_colorbar(fig, image, ax_obs, "Reflectivity", location="bottom")

lo, hi = chain_phi.quantile(0.05, dim=0), chain_phi.quantile(0.95, dim=0)
ax_phi.fill_betweenx(depth, lo.numpy(), hi.numpy(), color=C_BAYES,
                     alpha=0.25, lw=0, label="Posterior 90%")
ax_phi.plot(phi_map.numpy(), depth, color=C_RECOVERED, lw=2.0, label="MAP")
ax_phi.plot(phi_true.numpy(), depth, color=C_TRUTH, lw=1.6, ls="--",
            label="Truth")
ax_phi.set(title="Porosity - the unknown", xlabel="Porosity", xlim=(0.02, 0.30))
ax_phi.legend(loc="lower left")

vlo = vp_draws.quantile(0.05, dim=0)
vhi = vp_draws.quantile(0.95, dim=0)
ax_vp.fill_betweenx(depth, vlo.numpy(), vhi.numpy(), color=C_BAYES,
                    alpha=0.25, lw=0)
ax_vp.plot(vp_map_prof.numpy(), depth, color=C_RECOVERED, lw=2.0)
ax_vp.plot(vp_true_prof.numpy(), depth, color=C_TRUTH, lw=1.6, ls="--")
ax_vp.set(title="vp", xlabel="vp [m/s]")

fig.savefig(OUT / "04_deterministic_bayes_unified.png")
print(f"saved {OUT / '04_deterministic_bayes_unified.png'}")
plt.show()
