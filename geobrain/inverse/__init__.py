"""Inverse-problem definition: the InverseProblem keystone + the likelihood
and prior it binds. Top-level layer (L4) sitting above geobrain.core, binding
it to the solvers in geobrain.optim / geobrain.bayes.

Also home to the waveform data-misfit family (:mod:`geobrain.inverse.misfit`),
the frequency-continuation filtering support
(:mod:`geobrain.inverse.filtering`), and :class:`ChannelLikelihood`
(:mod:`geobrain.inverse.channel_likelihood`), all ``Likelihood``-protocol
implementations that drop into ``InverseProblem(likelihood=...)`` alongside
:class:`GaussianLikelihood`.

Module layout::

    inverse/
    ├── inverse_problem.py    # InverseProblem keystone (+ Tier-1 factories
    │                         #   create_inverter / as_posterior, which import
    │                         #   optim / bayes lazily; those layers sit above)
    ├── likelihood.py         # Likelihood protocol + GaussianLikelihood
    ├── prior.py              # Prior protocol + Gaussian/TV/smoothness priors
    ├── evaluation.py         # ObjectiveEvaluation record
    ├── misfit.py             # Waveform data-misfit family (envelope, x-corr, …)
    ├── filtering.py          # Frequency-continuation filtered misfit
    ├── channel_likelihood.py # Per-channel composite likelihood
    ├── joint.py              # JointProblem (multi-physics objective)
    ├── joint_weights.py      # Joint-term weighting policies (used via joint)
    ├── joint_binding.py      # JointForward binding of shared model state
    ├── model_space.py        # JointModelSpace (+ cross-gradient via optim,
    │                         #   imported lazily for the same layering reason)
    └── _units.py             # Private declared-unit checks for observed data

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .channel_likelihood import ChannelLikelihood  # noqa: F401
from .evaluation import ObjectiveEvaluation, ObjectiveEvaluationLike  # noqa: F401
from .filtering import FrequencyFilteredMisfit, butterworth_lowpass  # noqa: F401
from .inverse_problem import InverseProblem, InverseProblemLike  # noqa: F401
from .joint import JointProblem  # noqa: F401
from .joint_binding import JointForward  # noqa: F401
from .model_space import JointModelSpace  # noqa: F401
from .likelihood import GaussianLikelihood, Likelihood  # noqa: F401
from .misfit import (  # noqa: F401
    EnvelopeWaveform,
    GlobalCorrelationWaveform,
    HuberWaveform,
    L1Waveform,
    L2Waveform,
    NormalizedIntegrationWaveform,
    StudentTWaveform,
    TravelTimeWaveform,
    WassersteinWaveform,
)
from .prior import GaussianPrior, Prior  # noqa: F401

__all__ = [
    "JointModelSpace",
    "InverseProblem",
    "InverseProblemLike",
    "ObjectiveEvaluation",
    "ObjectiveEvaluationLike",
    "Likelihood",
    "GaussianLikelihood",
    "ChannelLikelihood",
    "JointProblem",
    "JointForward",
    "Prior",
    "GaussianPrior",
    # Waveform data-misfit family (FWI objectives)
    "EnvelopeWaveform",
    "GlobalCorrelationWaveform",
    "HuberWaveform",
    "L1Waveform",
    "L2Waveform",
    "NormalizedIntegrationWaveform",
    "StudentTWaveform",
    "TravelTimeWaveform",
    "WassersteinWaveform",
    # Frequency-continuation filtering
    "FrequencyFilteredMisfit",
    "butterworth_lowpass",
]
