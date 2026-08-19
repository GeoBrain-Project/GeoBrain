"""Invertible transforms: differentiable change-of-variables for constrained Bayesian sampling.

**Semantic wrapper module.** The implementation lives at
:mod:`geobrain.core.transforms`: it is pure ``torch`` math with no
Bayesian-inference dependency, so it was moved down to the core layer (L0)
where :mod:`geobrain.geomodel.earthmodel` (L3, ``Field(bounds=...)`` sugar) can reuse it
without an upward import into ``geobrain.bayes`` (L4). This module gives those
L0 implementations Bayesian role-specific public names while leaving the core
transform class names unchanged; see :mod:`geobrain.core.transforms` for the
implementation and full change-of-variables background.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from geobrain.core import GeoBrainError

if TYPE_CHECKING:
    from geobrain.core.transforms import InvertibleTransform

    class _Identity:
        def forward(self, u: torch.Tensor) -> torch.Tensor: ...

        def inverse(self, x: torch.Tensor) -> torch.Tensor: ...

        def log_abs_det_jacobian(self, u: torch.Tensor) -> torch.Tensor: ...

    class _Exp(_Identity):
        pass

    class _AffineSigmoid(_Identity):
        lo: float
        hi: float

        def __init__(self, lo: float, hi: float) -> None: ...

else:
    from geobrain.core.transforms import (
        AffineSigmoid as _AffineSigmoid,
        Exp as _Exp,
        Identity as _Identity,
        InvertibleTransform,
    )

__all__ = [
    "InvertibleTransform",
    "IdentityTransform",
    "PositiveTransform",
    "IntervalTransform",
]


class IdentityTransform(_Identity):
    """Identity Bayesian change of variables.

    ``forward`` and ``inverse`` return their tensor input unchanged, and
    ``log_abs_det_jacobian`` returns a scalar zero on the input dtype/device.
    Use this explicit no-op when a transform mapping should retain a field in
    unconstrained space.
    """


class PositiveTransform(_Exp):
    """Map an unconstrained parameter to the positive real line.

    ``forward(u)`` computes ``exp(u)``, ``inverse(x)`` requires every value of
    ``x`` to be strictly positive, and ``log_abs_det_jacobian(u)`` returns
    ``u.sum()`` for change-of-variables correction.
    """

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map strictly positive values back to unconstrained space."""
        if torch.any(x <= 0):
            raise GeoBrainError(
                "PositiveTransform inverse (log) requires strictly positive values; "
                "pass an initial constrained value > 0",
                object_name="PositiveTransform", field="x",
                expected="x > 0", actual=float(x.detach().min()),
            )
        return super().inverse(x)


class IntervalTransform(_AffineSigmoid):
    """Map an unconstrained parameter to an open bounded interval.

    Args:
        lo: Exclusive lower bound.
        hi: Exclusive upper bound; must be greater than ``lo``.

    ``forward(u)`` applies the affine sigmoid map into ``(lo, hi)``;
    ``inverse(x)`` requires values strictly inside that interval; and
    ``log_abs_det_jacobian(u)`` returns the summed change-of-variables
    correction.
    """

    def __init__(self, lo: float, hi: float) -> None:
        if not hi > lo:
            raise GeoBrainError(
                "IntervalTransform requires hi > lo",
                object_name="IntervalTransform", field="(lo, hi)",
                expected="hi > lo", actual=(lo, hi),
            )
        super().__init__(lo, hi)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Map values strictly inside this interval to unconstrained space."""
        if torch.any(x <= self.lo) or torch.any(x >= self.hi):
            raise GeoBrainError(
                "IntervalTransform inverse requires values strictly inside (lo, hi); "
                "pass an initial constrained value in the open interval",
                object_name="IntervalTransform", field="x",
                expected=f"{self.lo} < x < {self.hi}",
                actual=(float(x.detach().min()), float(x.detach().max())),
            )
        return super().inverse(x)

    def __repr__(self) -> str:
        return f"IntervalTransform(lo={self.lo}, hi={self.hi})"
