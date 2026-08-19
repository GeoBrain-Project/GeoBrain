"""GeoBrain Bayesian Inference: posterior sampling for geophysical inverse problems.

This package implements Tier 2 of the inversion architecture: posterior samplers
(HMC, NUTS, LangevinDynamics, SVGD) that consume an ``InverseProblem`` or any
callable
log-posterior via each sampler's ``from_callable`` classmethod (e.g.
``HMC.from_callable(log_post, ...)``). For Tier 1 canonical usage, call
``problem.as_posterior().sample("hmc", ...)``. For the deterministic path,
see ``geobrain.optim``.

Architecture:
    bayes/
    ├── base.py             # Sampler ABC and target protocol
    ├── execution.py        # chain config, stop reason, run accounting
    ├── results.py          # owned immutable InferenceResult
    ├── diagnostics.py      # ESS, split-R-hat, summarize, run_chains
    ├── distributions.py    # Parametric distributions (Gaussian, mixtures)
    ├── posterior.py         # Posterior facade (wraps InverseProblem)
    ├── transforms.py        # Named prior-space transforms (IdentityTransform,
    │                        #   PositiveTransform, IntervalTransform), thin
    │                        #   subclasses of the core bijectors in
    │                        #   geobrain.core.transforms
    ├── samplers/            # Concrete sampler implementations
    │   ├── hmc.py              # Hamiltonian Monte Carlo
    │   ├── nuts.py             # No-U-Turn Sampler
    │   ├── langevin.py         # Langevin Dynamics Sampler
    │   ├── svgd.py             # Stein Variational Gradient Descent
    │   └── _hamiltonian.py, _nuts_{tree,warmup,execution,results}.py
    │                        # Private leapfrog/tree/warmup machinery behind
    │                        #   HMC and NUTS (~2.1k lines; touched only by
    │                        #   bayes' own tests)
    └── _callable_problem.py # Internal adapter for callable log-posteriors

Quick Start:
    >>> from geobrain.bayes import Posterior
    >>> posterior = Posterior(inverse_problem)
    >>> result = posterior.sample(
    ...     "hmc", params={"vp": v0}, n_iters=500, step_size=0.01, n_leapfrog=20,
    ... )

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

# =========================================================================
# Posterior Facade
# =========================================================================
from .posterior import Posterior

# =========================================================================
# Invertible transforms: change-of-variables for constrained Bayesian sampling
# =========================================================================
from .transforms import (
    IdentityTransform,
    IntervalTransform,
    InvertibleTransform,
    PositiveTransform,
)

# =========================================================================
# Samplers
# =========================================================================
from .samplers import HMC, LangevinDynamics, NUTS, SVGD

# =========================================================================
# Core ABC & Result Container
# =========================================================================
from .base import LogPosteriorTarget, Sampler
from .execution import ChainConfig, RunAccounting, SamplerStopReason
from .results import InferenceResult

# =========================================================================
# Chain diagnostics: ESS, split-R-hat, summaries, multi-chain runner
# =========================================================================
from .diagnostics import ess, split_rhat, summarize, run_chains

# =========================================================================
# Probability Distributions
# =========================================================================
from .distributions import (
    Distribution,
    Gaussian,
    GaussianMixture,
)

__all__ = [
    # Posterior facade
    "Posterior",
    # Invertible transforms
    "InvertibleTransform",
    "IdentityTransform",
    "PositiveTransform",
    "IntervalTransform",
    # Samplers
    "HMC",
    "NUTS",
    "LangevinDynamics",
    "SVGD",
    # Base
    "Sampler",
    "LogPosteriorTarget",
    "InferenceResult",
    "ChainConfig",
    "SamplerStopReason",
    "RunAccounting",
    # Diagnostics
    "ess",
    "split_rhat",
    "summarize",
    "run_chains",
    # Distributions
    "Distribution",
    "Gaussian",
    "GaussianMixture",
]
