"""Power-transform normalisers: Box-Cox and Yeo-Johnson.

Both auto-fit the optimal ``λ`` parameter from data
via ``scipy.stats`` when ``lmbda`` is ``None``.

- :class:`BoxCox`: strictly-positive data.
- :class:`YeoJohnson`: handles zeros / negative data via the
  piecewise formula
  ``y = ((x+1)^λ − 1) / λ`` (``x ≥ 0``) etc.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array
from .base import Transform

if TYPE_CHECKING:
    from ...frames import GeoFrame

__all__ = ["BoxCox", "YeoJohnson"]


class BoxCox(Transform):
    """
    Box-Cox power transform with auto-fit ``λ``.

    ::

        y =  (x^λ − 1) / λ   if λ ≠ 0
        y =  log(x)          if λ = 0

    Requires strictly positive ``x``.

    Args:
        column: input column.
        output_column: default ``f"{column}_bc"``.
        lmbda: λ. ``None`` → auto-estimate via ``scipy.stats.boxcox``
            during :meth:`fit`.
    """

    def __init__(
        self,
        column: str,
        *,
        output_column: str | None = None,
        lmbda: float | None = None,
    ) -> None:
        self.column = column
        self.output_column = output_column or f"{column}_bc"
        self.lmbda = lmbda
        self._fitted_lambda: float | None = None
        self._fitted = False

    def fit(self, data: "GeoFrame") -> "BoxCox":
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="BoxCox.fit", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )
        values = as_float_array(data[self.column])
        if np.any(values <= 0):
            raise GeoBrainError(
                "BoxCox requires strictly positive data",
                object_name="BoxCox.fit", field=self.column,
                expected="all > 0",
                actual=f"min = {float(np.min(values))}",
            )
        if self.lmbda is not None:
            self._fitted_lambda = float(self.lmbda)
        else:
            boxcox_fn = getattr(import_module("scipy.stats"), "boxcox")
            result = cast(tuple[object, object], boxcox_fn(values))
            self._fitted_lambda = float(result[1])
        self._fitted = True
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._fitted_lambda is None:
            raise GeoBrainError(
                "BoxCox.transform called before fit()",
                object_name="BoxCox.transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        values = as_float_array(data[self.column])
        lam = self._fitted_lambda
        if abs(lam) < 1.0e-12:
            transformed = as_float_array(np.log(values))
        else:
            transformed = as_float_array((values ** lam - 1.0) / lam)
        out = data.copy()
        out[self.output_column] = transformed
        return out

    def inverse_transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._fitted_lambda is None:
            raise GeoBrainError(
                "BoxCox.inverse_transform called before fit()",
                object_name="BoxCox.inverse_transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        values = as_float_array(data[self.output_column])
        lam = self._fitted_lambda
        if abs(lam) < 1.0e-12:
            original = as_float_array(np.exp(values))
        else:
            original = as_float_array((lam * values + 1.0) ** (1.0 / lam))
        out = data.copy()
        out[self.column] = original
        return out

    @property
    def fitted_lambda(self) -> float | None:
        return self._fitted_lambda

    def __repr__(self) -> str:
        lam = f", λ={self._fitted_lambda:.3f}" if self._fitted else ""
        return f"BoxCox({self.column!r}{lam})"


def _yj_forward(values: FloatArray, lam: float) -> FloatArray:
    """Yeo-Johnson forward transform, vectorised."""
    out = np.empty_like(values, dtype=np.float64)
    pos = values >= 0
    neg = ~pos
    if abs(lam) < 1.0e-12:
        out[pos] = np.log(values[pos] + 1.0)
    else:
        out[pos] = ((values[pos] + 1.0) ** lam - 1.0) / lam
    if abs(lam - 2.0) < 1.0e-12:
        out[neg] = -np.log(-values[neg] + 1.0)
    else:
        out[neg] = -((-values[neg] + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam)
    return as_float_array(out)


def _yj_inverse(y: FloatArray, lam: float) -> FloatArray:
    """Yeo-Johnson inverse, vectorised."""
    out = np.empty_like(y, dtype=np.float64)
    pos = y >= 0
    neg = ~pos
    if abs(lam) < 1.0e-12:
        out[pos] = np.exp(y[pos]) - 1.0
    else:
        out[pos] = (lam * y[pos] + 1.0) ** (1.0 / lam) - 1.0
    if abs(lam - 2.0) < 1.0e-12:
        out[neg] = 1.0 - np.exp(-y[neg])
    else:
        out[neg] = 1.0 - (-(2.0 - lam) * y[neg] + 1.0) ** (1.0 / (2.0 - lam))
    return as_float_array(out)


class YeoJohnson(Transform):
    """
    Yeo-Johnson power transform with auto-fit ``λ``.

    Handles zero and negative data through the piecewise formula
    (see module docstring). Falls back to Box-Cox behaviour when all
    data is positive.

    Args:
        column: input column.
        output_column: default ``f"{column}_yj"``.
        lmbda: λ. ``None`` → auto-estimate via ``scipy.stats.yeojohnson``.
    """

    def __init__(
        self,
        column: str,
        *,
        output_column: str | None = None,
        lmbda: float | None = None,
    ) -> None:
        self.column = column
        self.output_column = output_column or f"{column}_yj"
        self.lmbda = lmbda
        self._fitted_lambda: float | None = None
        self._fitted = False

    def fit(self, data: "GeoFrame") -> "YeoJohnson":
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="YeoJohnson.fit", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )
        values = as_float_array(data[self.column])
        if self.lmbda is not None:
            self._fitted_lambda = float(self.lmbda)
        else:
            yj_fn = getattr(import_module("scipy.stats"), "yeojohnson")
            result = cast(tuple[object, object], yj_fn(values))
            self._fitted_lambda = float(result[1])
        self._fitted = True
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._fitted_lambda is None:
            raise GeoBrainError(
                "YeoJohnson.transform called before fit()",
                object_name="YeoJohnson.transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        values = as_float_array(data[self.column])
        out = data.copy()
        out[self.output_column] = _yj_forward(values, self._fitted_lambda)
        return out

    def inverse_transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._fitted_lambda is None:
            raise GeoBrainError(
                "YeoJohnson.inverse_transform called before fit()",
                object_name="YeoJohnson.inverse_transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        values = as_float_array(data[self.output_column])
        out = data.copy()
        out[self.column] = _yj_inverse(values, self._fitted_lambda)
        return out

    @property
    def fitted_lambda(self) -> float | None:
        return self._fitted_lambda

    def __repr__(self) -> str:
        lam = f", λ={self._fitted_lambda:.3f}" if self._fitted else ""
        return f"YeoJohnson({self.column!r}{lam})"
