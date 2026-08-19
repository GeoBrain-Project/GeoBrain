"""
Immutable configuration records for deterministic optimizers.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from geobrain.core import GeoBrainError

from ._validation import _coerce_integral_scalar, _coerce_real_scalar

__all__ = ["AdamConfig", "LBFGSConfig"]


def _normalize_finite_float(
    value: object,
    *,
    owner: str,
    field: str,
    minimum: float,
) -> float:
    """Normalize one numeric option without accepting ``bool`` as a number."""
    return float(
        _coerce_real_scalar(
            value,
            owner=owner,
            field=field,
            finite=True,
            minimum=minimum,
        )
    )


@dataclass(frozen=True)
class AdamConfig:
    """Validated, frozen options for :class:`torch.optim.Adam`.

    ``lr``, ``eps``, and ``weight_decay`` are finite and non-negative.
    ``betas`` contains exactly two finite coefficients in ``[0, 1)``. Boolean
    values are rejected for every numeric field instead of being normalized as
    Python's numeric ``False``/``True`` values.

    Attributes:
        lr: learning rate.
        betas: Adam ``(beta1, beta2)`` moment decay pair.
        eps: numerical stabiliser in the denominator.
        weight_decay: decoupled L2 penalty coefficient.
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        """Normalize scalar values and reject invalid Adam options."""
        lr = _normalize_finite_float(
            self.lr, owner="AdamConfig", field="lr", minimum=0.0
        )
        eps = _normalize_finite_float(
            self.eps, owner="AdamConfig", field="eps", minimum=0.0
        )
        weight_decay = _normalize_finite_float(
            self.weight_decay,
            owner="AdamConfig",
            field="weight_decay",
            minimum=0.0,
        )
        try:
            beta1_raw, beta2_raw = self.betas
        except (TypeError, ValueError):
            raise GeoBrainError(
                "AdamConfig.betas must contain exactly two coefficients",
                object_name="AdamConfig",
                field="betas",
                expected="two finite coefficients in [0, 1)",
                actual=self.betas,
            ) from None
        beta1 = _coerce_real_scalar(
            beta1_raw,
            owner="AdamConfig",
            field="betas[0]",
            finite=True,
            minimum=0.0,
            maximum_exclusive=1.0,
        )
        beta2 = _coerce_real_scalar(
            beta2_raw,
            owner="AdamConfig",
            field="betas[1]",
            finite=True,
            minimum=0.0,
            maximum_exclusive=1.0,
        )
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "betas", (beta1, beta2))
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "weight_decay", weight_decay)


@dataclass(frozen=True)
class LBFGSConfig:
    """Validated, frozen canonical options for L-BFGS.

    ``lr``, ``tolerance_grad``, and ``tolerance_change`` are finite and
    non-negative; ``history_size`` and ``max_iter`` are positive integers.
    Boolean values are not accepted at either numeric boundary.
    ``line_search_fn`` is ``"strong_wolfe"`` or ``None``.

    Attributes:
        lr: step size handed to ``torch.optim.LBFGS``.
        history_size: number of curvature pairs kept.
        max_iter: inner L-BFGS iterations per optimizer step.
        tolerance_grad: gradient-norm stopping tolerance.
        tolerance_change: parameter/loss change stopping tolerance.
        line_search_fn: ``None`` or ``'strong_wolfe'``.
    """

    lr: float = 1.0
    history_size: int = 10
    max_iter: int = 20
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9
    line_search_fn: Literal["strong_wolfe"] | None = "strong_wolfe"

    def __post_init__(self) -> None:
        """Normalize scalar values and reject invalid L-BFGS options."""
        lr = _normalize_finite_float(
            self.lr, owner="LBFGSConfig", field="lr", minimum=0.0
        )
        history_size = _coerce_integral_scalar(
            self.history_size,
            owner="LBFGSConfig",
            field="history_size",
            minimum=1,
        )
        max_iter = _coerce_integral_scalar(
            self.max_iter,
            owner="LBFGSConfig",
            field="max_iter",
            minimum=1,
        )
        tolerance_grad = _normalize_finite_float(
            self.tolerance_grad,
            owner="LBFGSConfig",
            field="tolerance_grad",
            minimum=0.0,
        )
        tolerance_change = _normalize_finite_float(
            self.tolerance_change,
            owner="LBFGSConfig",
            field="tolerance_change",
            minimum=0.0,
        )
        if self.line_search_fn not in (None, "strong_wolfe"):
            raise GeoBrainError(
                "LBFGSConfig.line_search_fn must be None or 'strong_wolfe'",
                object_name="LBFGSConfig",
                field="line_search_fn",
                expected="None or 'strong_wolfe'",
                actual=self.line_search_fn,
            )
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "history_size", history_size)
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "tolerance_grad", tolerance_grad)
        object.__setattr__(self, "tolerance_change", tolerance_change)


OptimizerConfig: TypeAlias = AdamConfig | LBFGSConfig
