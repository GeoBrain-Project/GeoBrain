"""
Frequency-domain (time-harmonic) wave solvers.

Unlike the packed time-domain facades in :mod:`geobrain.physics.wave.acoustic`
(which march a wavefield through time), these solve a frequency-domain
*boundary-value problem*: assemble a complex sparse system per frequency and
solve it, with an implicit-function-theorem adjoint (``IMPLICIT_VJP``). The
architecture mirrors the frequency-domain EM solvers (``em/frequency_domain``,
``em/natural_source``) rather than the FDTD propagators, hence its own home
here rather than under ``propagators/``.

Currently: 2-D constant-density acoustic Helmholtz (``Helmholtz2D``).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from .helmholtz import (
    Helmholtz2D,
    Helmholtz2DReceiver,
    Helmholtz2DSource,
    Helmholtz2DSurvey,
)
from .helmholtz_2d import build_helmholtz_2d_coo

__all__ = [
    "Helmholtz2D",
    "Helmholtz2DReceiver",
    "Helmholtz2DSource",
    "Helmholtz2DSurvey",
    "build_helmholtz_2d_coo",
]
