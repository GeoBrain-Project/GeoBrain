"""Shared wavelet amplitude kernels: convention-agnostic.

The amplitude formula is the part that is genuinely shared across a wavelet's
zero-phase and causal forms (and across its free-function and class APIs); only
the *time axis* ``t`` differs between them, and that is the caller's job. So the
single source of truth for "the Ricker shape" lives here, evaluated at whatever
``t`` the caller built.

(Only Ricker is shared this way, the Gaussian and Ormsby *generators* and
*source functions* use genuinely different formulas, so they are not unified.)

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

import math

import torch
from torch import Tensor


def ricker_amplitude(t: Tensor, f0: float) -> Tensor:
    """Ricker (Mexican-hat) amplitude ``(1 - 2a)·exp(-a)`` with ``a = (π·f0·t)²``,
    evaluated at the given time samples ``t``."""
    a = (math.pi * f0 * t).pow(2)
    return (1.0 - 2.0 * a) * torch.exp(-a)
