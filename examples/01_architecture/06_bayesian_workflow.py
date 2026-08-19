"""The Bayesian workflow, and the diagnostics that decide if you believe it.

Sampling a posterior is the easy part; knowing whether the samples mean
anything is the job. This script runs the full workflow on a small,
deliberately correlated problem: three-term AVO inversion for the
(vp, vs, rho) of a target layer, and then interrogates the chains the
way you should before quoting a number:

    1. build the posterior      problem.as_posterior(transforms=...)
    2. run SEVERAL chains       run_chains(...) from dispersed starts
    3. split R-hat              did the chains find the SAME distribution?
    4. effective sample size    how many independent draws did you buy?
    5. trace plots              eyeball the warmup and the mixing
    6. posterior predictive     do draws reproduce the data they were fit to?

The physics is chosen because it is informatively correlated: at these
angles vp and vs trade off almost one-for-one, so the posterior is a
narrow diagonal ridge rather than a ball. The chains still converge
(R-hat ≈ 1.00) and buy a healthy number of independent draws, but a
single point estimate cannot express that ridge, which is the whole
reason to sample rather than only optimise.

APIs featured:
    - InverseProblem.as_posterior + geobrain.bayes.PositiveTransform
    - geobrain.bayes.run_chains (multi-chain, dispersed starts)
    - geobrain.bayes.split_rhat / ess / summarize
    - posterior predictive checks through the same forward operator

Expected runtime: < 2 min.

Outputs:
    out/06_bayesian_workflow.png: traces, R-hat/ESS, the vp-vs ridge,
    and the posterior predictive fan against the data.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from _style import (
    C_OBSERVED,
    C_POSTERIOR,
    C_RECOVERED,
    C_TRUTH,
    PALETTE,
    apply_style,
    figure,
)
from geobrain.bayes import (
    NUTS,
    PositiveTransform,
    ess,
    run_chains,
    split_rhat,
    summarize,
)
from geobrain.core import ForwardContext, ModelState
from geobrain.inverse import GaussianLikelihood, InverseProblem
from geobrain.optim import LBFGSConfig
from geobrain.physics.wave import AkiRichards

apply_style()
torch.manual_seed(5)
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
C_BAYES = C_POSTERIOR

# %% 1. A small problem with a known pathology -----------------------------
TRUE = {"vp": (2600.0, 2250.0), "vs": (1150.0, 1300.0), "rho": (2400.0, 2050.0)}
PRIOR_MEAN = {"vp": (2600.0, 2500.0), "vs": (1150.0, 1200.0),
              "rho": (2400.0, 2300.0)}
PRIOR_STD = {"vp": (25.0, 250.0), "vs": (15.0, 150.0), "rho": (25.0, 200.0)}
FIELDS = ("vp", "vs", "rho")

class ElasticPrior:
    """Independent Gaussians: tight on the logged layer, broad on the target."""

    def log_prob(self, state: ModelState) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64)
        for name in FIELDS:
            (val,) = state.fetch(name)
            mean = torch.tensor(PRIOR_MEAN[name], dtype=torch.float64)
            std = torch.tensor(PRIOR_STD[name], dtype=torch.float64)
            total = total - 0.5 * (((val - mean) / std) ** 2).sum() - std.log().sum()
        return total

    def sample(self, generator: torch.Generator | None = None) -> ModelState:
        return ModelState(tensors={
            name: torch.tensor(PRIOR_MEAN[name], dtype=torch.float64)
            + torch.tensor(PRIOR_STD[name], dtype=torch.float64)
            * torch.randn(2, generator=generator, dtype=torch.float64)
            for name in FIELDS})

angles = torch.linspace(2.0, 40.0, 14, dtype=torch.float64)
avo = AkiRichards(angles_deg=angles)
ctx = ForwardContext()

truth = ModelState(tensors={k: torch.tensor(v, dtype=torch.float64)
                            for k, v in TRUE.items()})
(r_clean,) = avo(truth, ctx).fetch("reflectivity")
noise_std = 0.005
r_obs = r_clean + noise_std * torch.randn_like(r_clean)

problem = InverseProblem(forward=avo, observed={"reflectivity": r_obs},
                         likelihood=GaussianLikelihood(std=noise_std),
                         prior=ElasticPrior())
print(f"[1] {len(FIELDS) * 2} unknowns, {r_obs.numel()} angle samples, "
      f"noise σ = {noise_std}")

# %% 2. A MAP point to start the chains from -------------------------------
init = {k: torch.tensor(v, dtype=torch.float64) for k, v in PRIOR_MEAN.items()}
map_result = problem.create_inverter(
    params={k: v.clone() for k, v in init.items()},
    optimizer=LBFGSConfig(lr=0.5, max_iter=10), ctx=ctx).run(n_iters=40)
map_state = {k: v.detach() for k, v in map_result.best_params.items()}
print(f"[2] MAP found in {map_result.completed_iters} iterations "
      f"(-log posterior {map_result.best_loss:.3f})")

# %% 3. Several chains, from DISPERSED starts ------------------------------
#
# One chain can look converged and be wrong. run_chains builds one sampler
# per chain; the chain index is there so each starts somewhere different,
# that dispersion is what makes split R-hat meaningful.
posterior = problem.as_posterior(
    transforms={name: PositiveTransform() for name in FIELDS})

N_CHAINS, N_DRAWS = 4, 500

def make_chain(index: int, generator: torch.Generator | None):
    spread = 1.0 + 0.05 * (index - (N_CHAINS - 1) / 2)      # ±5% dispersion
    start = {k: v * spread for k, v in map_state.items()}
    return NUTS(posterior, params=start, step_size=0.02, max_depth=7,
                warmup=300, target_accept=0.8, generator=generator)

result = run_chains(make_chain, n_chains=N_CHAINS, n_iters=N_DRAWS, ctx=ctx,
                    seeds=[11, 12, 13, 14])
chains = {k: v.detach() for k, v in result.samples.items()}
print(f"[3] {N_CHAINS} chains x {N_DRAWS} draws; "
      f"stacked shape {tuple(chains['vp'].shape)} (chain, draw, layer)")

# %% 4. The diagnostics ----------------------------------------------------
#
# split R-hat compares the first and second half of every chain against
# each other AND across chains: ~1.00 means they agree, > 1.01 means they
# do not. ESS answers "how many INDEPENDENT draws is this worth?".
print("[4] convergence report for the target layer")
print(f"    {'field':6s} {'R-hat':>8s} {'ESS':>8s} {'mean':>9s} {'sd':>8s}")
rhat, ess_target = {}, {}
for name in FIELDS:
    target = chains[name][..., 1]                       # layer 2
    rhat[name] = float(split_rhat(target.unsqueeze(-1)))
    ess_target[name] = float(ess(target.unsqueeze(-1), chains=True))
    print(f"    {name:6s} {rhat[name]:8.3f} {ess_target[name]:8.0f} "
          f"{float(target.mean()):9.1f} {float(target.std()):8.1f}")
print("    (rule of thumb: R-hat < 1.01 and ESS > 100 per quantity you "
      "plan to quote)")

flat = {name: chains[name][..., 1].reshape(-1) for name in FIELDS}
summary = summarize(result)
print(f"[4] summarize() also returns {sorted(next(iter(summary.values())))} "
      "per field")

# %% 5. Posterior predictive check -----------------------------------------
#
# Draws are only credible if the data they were fit to fall inside their
# predictions. Push a subsample back through the SAME operator, and add
# the measurement noise, because the PREDICTIVE distribution is
# p(y_new | y) = f(m) + eps, not the spread of f(m) alone. Forgetting the
# noise term is the classic way to declare a model overconfident.
idx = torch.randperm(flat["vp"].numel())[:200]
with torch.no_grad():
    fan = torch.stack([
        avo(ModelState(tensors={
            "vp": torch.stack([chains["vp"].reshape(-1, 2)[i, 0], flat["vp"][i]]),
            "vs": torch.stack([chains["vs"].reshape(-1, 2)[i, 0], flat["vs"][i]]),
            "rho": torch.stack([chains["rho"].reshape(-1, 2)[i, 0], flat["rho"][i]]),
        }), ctx).fetch("reflectivity")[0].reshape(-1) for i in idx])
    fan = fan + noise_std * torch.randn_like(fan)
    (r_map,) = avo(ModelState(tensors=map_state), ctx).fetch("reflectivity")
inside = ((r_obs.reshape(-1) >= fan.quantile(0.05, dim=0))
          & (r_obs.reshape(-1) <= fan.quantile(0.95, dim=0))).double().mean()
print(f"[5] posterior predictive: {float(inside) * 100:.0f}% of the observed "
      "angles fall inside the 5-95% band (≈90% is healthy)")

# %% 6. Picture ------------------------------------------------------------
fig, axes = figure(2, 2, scale=1.15)

# (a) traces: thin lines for the draws, thick lines for the running mean.
#     R-hat is exactly the question "do those thick lines meet?"
for c in range(N_CHAINS):
    trace = chains["vp"][c, :, 1]
    axes[0, 0].plot(trace.numpy(), lw=0.5, alpha=0.45, color=PALETTE[c])
    running = trace.cumsum(0) / torch.arange(1, trace.numel() + 1)
    axes[0, 0].plot(running.numpy(), lw=2.0, color=PALETTE[c],
                    label=f"Chain {c + 1}")
axes[0, 0].axhline(TRUE["vp"][1], color=C_TRUTH, lw=1.8, ls="--",
                   label="Truth")
axes[0, 0].set(title="Traces and running means, vp",
               xlabel="Draw (post-warmup)", ylabel="vp [m/s]")
axes[0, 0].legend(ncol=2, fontsize=7, loc="upper right")

# (b) one scale per panel: ESS as bars, R-hat printed on each bar
names = list(FIELDS)
bars = axes[0, 1].barh(range(len(names)), [ess_target[n] for n in names],
                       color=PALETTE[0], alpha=0.9)
axes[0, 1].axvline(100.0, color=C_RECOVERED, ls="--", lw=1.5)
axes[0, 1].annotate("ESS ≥ 100", xy=(100.0, 0.5),
                    xycoords=("data", "axes fraction"), rotation=90,
                    ha="center", va="center", fontsize=8, color=C_RECOVERED,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.9))
for rect, name in zip(bars, names):
    axes[0, 1].text(rect.get_width() * 0.98, rect.get_y() + rect.get_height() / 2,
                    f"R-hat {rhat[name]:.3f}", ha="right", va="center",
                    color="white", fontsize=9, fontweight="semibold")
axes[0, 1].set_yticks(range(len(names)), names)
axes[0, 1].invert_yaxis()
axes[0, 1].set(title=f"Effective sample size (of {N_CHAINS * N_DRAWS} draws)",
               xlabel="Effective sample size")

# (c) the ridge a point estimate cannot express
axes[1, 0].scatter(flat["vp"].numpy(), flat["vs"].numpy(), s=6, alpha=0.18,
                   color=C_BAYES, label="Posterior draws")
axes[1, 0].plot(float(map_state["vp"][1]), float(map_state["vs"][1]), "*",
                ms=17, color=C_RECOVERED, mec="k", label="MAP")
axes[1, 0].plot(TRUE["vp"][1], TRUE["vs"][1], "s", ms=9, color=C_TRUTH,
                label="Truth")
corr = float(torch.corrcoef(torch.stack([flat["vp"], flat["vs"]]))[0, 1])
axes[1, 0].set(title=f"The vp–vs ridge (correlation {corr:+.2f})",
               xlabel="Target vp [m/s]", ylabel="Target vs [m/s]")
axes[1, 0].legend(loc="upper left")

# (d) do the draws reproduce the data they were fit to?
ang = angles.numpy()
axes[1, 1].fill_between(ang, fan.quantile(0.05, dim=0).numpy(),
                        fan.quantile(0.95, dim=0).numpy(), color=C_BAYES,
                        alpha=0.25, label="Predictive 5–95%")
axes[1, 1].plot(ang, r_map.reshape(-1).numpy(), color=C_RECOVERED, lw=2.0,
                label="MAP prediction")
axes[1, 1].errorbar(ang, r_obs.reshape(-1).numpy(), yerr=noise_std, fmt=".",
                    ms=7, color=C_OBSERVED, ecolor="gray", elinewidth=1.0,
                    capsize=2, label="Observed ± σ")
axes[1, 1].set(title=f"Posterior predictive check "
                     f"({float(inside) * 100:.0f}% of points inside)",
               xlabel="Incidence angle [°]", ylabel="Reflectivity")
axes[1, 1].legend(loc="lower left")

fig.savefig(OUT / "06_bayesian_workflow.png")
print(f"saved {OUT / '06_bayesian_workflow.png'}")
plt.show()
