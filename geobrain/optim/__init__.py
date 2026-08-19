"""GeoBrain Optimization: deterministic inversion and bare optimizer wrappers.

This package consolidates the deterministic inversion machinery, organized into
a 3-tier API that balances convenience with control. All tiers consume the same
``InverseProblem`` from ``geobrain.inverse``.

Three-Tier API:
    Tier 1 (canonical):  ``problem.create_inverter(params={"vp": v0},
    optimizer=AdamConfig(lr=1e-3)).run(n_iters=200)``
    Tier 2 (explicit):   ``Inverter(problem, params={"vp": v0},
    optimizer=AdamConfig(lr=1e-3)).run(n_iters=200)``
    Tier 3 (bare loop):  ``Adam(problem, params={"vp": v0},
    config=AdamConfig(lr=1e-3)).run(n_iters=200)``

Architecture:
    optim/
    ├── inverter.py         # Tier 2 Inverter facade (regularizers, the hook contract)
    ├── config.py           # Immutable Adam / L-BFGS configuration records
    ├── execution.py        # Cancellation, stop reasons, iteration records
    ├── results.py          # Immutable optimizer / inversion result snapshots
    ├── solvers/
    │   ├── base.py         # Optimizer ABC and shared execution seam
    │   ├── adam.py         # Tier 3 Adam optimizer
    │   └── lbfgs.py        # Tier 3 L-BFGS optimizer
    ├── regularizers.py     # l1, l2, smallness, smoothness, tikhonov,
    │                       #   total_variation, cross_gradient, gmm_prior
    ├── processing.py       # THE hook contract: GradientProcessor / StepProjection
    │                       #   built-ins (NaNGuard, Mask, GaussianSmooth, NormClip,
    │                       #   Weight, DepthWeight, BoundsClamp, Freeze) + log_every
    │                       #:    the old scalar/per-parameter clamp-callback module
    │                       #   was deleted outright; BoundsClamp/
    │                       #   Freeze are the sole constraint seam now.
    ├── parameterization.py # ParameterBlock, Parameterization (flat vectors)
    ├── _parameterized_problem.py
    │                       # Private physical-state adapter used by
    │                       # Inverter.from_parameterization
    ├── _normalize.py       # Internal lr / bounds normalization helpers
    ├── _factories.py       # Internal factory helper (Inverter.from_function)
    ├── _loss_evaluator.py  # Internal loss/grad evaluation seam shared by
    │                       #   Inverter and the Tier 3 bare optimizers
    ├── _solver_execution.py# Internal shared solver run loop
    ├── _result_assembly.py # Internal result construction helpers
    └── _validation.py      # Private scalar coercion for the execution
                            #   contracts (unrelated to core/validation.py)

Quick Start:
    >>> from geobrain.optim import AdamConfig, Inverter, smoothness
    >>> inv = Inverter(
    ...     problem, params={"vp": v0}, optimizer=AdamConfig(lr=1e-3),
    ...     regularizer=lambda p: smoothness(p["vp"], weight=1e-2),
    ... )
    >>> result = inv.run(n_iters=200)

Flat-vector optimizers use the same facade through
``Inverter.from_parameterization``; the factory is a method on the existing
public ``Inverter`` symbol and adds no package-level export.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

# =========================================================================
# Immutable configuration and execution contracts
# =========================================================================
from .config import AdamConfig, LBFGSConfig
from .execution import CancellationToken, IterationRecord, StopReason
from .results import InversionResult, OptimizationResult

# =========================================================================
# Optimizer ABC
# =========================================================================
from .solvers.base import Optimizer

# =========================================================================
# Bare Optimizers (Tier 3)
# =========================================================================
from .solvers.adam import Adam
from .solvers.lbfgs import LBFGS

# =========================================================================
# Inverter Facade (Tier 2)
# =========================================================================
from .inverter import Inverter

# =========================================================================
# Regularizers
# =========================================================================
from .regularizers import (
    cross_gradient,
    depth_weighting,
    gmm_prior,
    l1,
    l2,
    smallness,
    smoothness,
    smoothness_second_order,
    tikhonov,
    total_variation,
    total_variation_second_order,
)

# =========================================================================
# Hook contract: gradient processors, step projections, the
# lazy-clone run(callback=) observer helper. THE single seam for gradient
# masking/filtering and post-step constraints (the old scalar/per-parameter
# clamp-callback module was deleted outright: see BoundsClamp /
# Freeze below).
# =========================================================================
from .processing import (
    BoundsClamp,
    DepthWeight,
    Freeze,
    GaussianSmooth,
    GradientProcessor,
    Mask,
    NaNGuard,
    NormClip,
    StepProjection,
    Weight,
    log_every,
)

# =========================================================================
# Parameterization
# =========================================================================
from .parameterization import ParameterBlock, Parameterization

__all__ = [
    # Inverter facade (Tier 2)
    "Inverter",
    "InversionResult",
    # Configuration and execution contracts
    "AdamConfig",
    "LBFGSConfig",
    "CancellationToken",
    "StopReason",
    "IterationRecord",
    # Bare optimizers (Tier 3)
    "Adam",
    "LBFGS",
    # Optimizer ABC
    "Optimizer",
    "OptimizationResult",
    # Regularizers (single-tensor)
    "l1",
    "l2",
    "smallness",
    "smoothness", "smoothness_second_order",
    "depth_weighting",
    "tikhonov",
    "total_variation", "total_variation_second_order",
    # Regularizers (inter-parameter coupling, for joint inversion)
    "cross_gradient",
    "gmm_prior",
    # Hook contract: gradient processors, step projections, log_every
    "GradientProcessor",
    "StepProjection",
    "NaNGuard",
    "Mask",
    "GaussianSmooth",
    "NormClip",
    "Weight",
    "DepthWeight",
    "BoundsClamp",
    "Freeze",
    "log_every",
    # Parameterization
    "ParameterBlock",
    "Parameterization",
]
