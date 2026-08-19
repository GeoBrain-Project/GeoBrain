"""
Polynomial detrending transform.

Fits a
linear (``degree=1``) or quadratic (``degree=2``) trend surface to a
column and subtracts it. The fitted coefficients are kept, so
:meth:`inverse_transform` can add the trend back to a residual /
realisation column.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array
from .base import Transform

if TYPE_CHECKING:
    from ...frames import GeoFrame

__all__ = ["Detrend"]


class Detrend(Transform):
    """
    Polynomial-trend fit / subtract / add-back.

    Args:
        column: input data column.
        degree: ``1`` (linear basis ``1, x, y``) or ``2`` (quadratic
            basis ``1, x, y, x², xy, y²``).
        output_column: residual column name (default
            ``f"{column}_detrended"``).
    """

    def __init__(
        self,
        column: str,
        degree: int = 1,
        output_column: str | None = None,
    ) -> None:
        if degree not in (1, 2):
            raise GeoBrainError(
                "degree must be 1 or 2",
                object_name="Detrend", field="degree",
                expected="1 or 2", actual=degree,
            )
        self.column = column
        self.degree = int(degree)
        self.output_column = output_column or f"{column}_detrended"
        self._coefficients: FloatArray | None = None
        self._fitted = False

    def _design(self, coords: FloatArray) -> FloatArray:
        x = as_float_array(coords[:, 0])
        y = as_float_array(coords[:, 1])
        if self.degree == 1:
            return as_float_array(np.column_stack([np.ones(x.size), x, y]))
        return as_float_array(
            np.column_stack([np.ones(x.size), x, y, x * x, x * y, y * y])
        )

    def fit(self, data: "GeoFrame") -> "Detrend":
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="Detrend.fit", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )
        coords = as_float_array(data.geometry.coords)
        values = as_float_array(data[self.column])
        A = self._design(coords)
        coeffs, *_ = np.linalg.lstsq(A, values, rcond=None)
        self._coefficients = as_float_array(coeffs)
        self._fitted = True
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._coefficients is None:
            raise GeoBrainError(
                "Detrend.transform called before fit()",
                object_name="Detrend.transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        coords = as_float_array(data.geometry.coords)
        values = as_float_array(data[self.column])
        trend = as_float_array(self._design(coords) @ self._coefficients)
        out = data.copy()
        out[self.output_column] = as_float_array(values - trend)
        return out

    def inverse_transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted or self._coefficients is None:
            raise GeoBrainError(
                "Detrend.inverse_transform called before fit()",
                object_name="Detrend.inverse_transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        if self.output_column not in data.columns:
            raise GeoBrainError(
                f"residual column {self.output_column!r} is missing",
                object_name="Detrend.inverse_transform",
                field="output_column",
                expected=f"one of {data.columns}", actual=self.output_column,
            )
        coords = as_float_array(data.geometry.coords)
        residuals = as_float_array(data[self.output_column])
        trend = as_float_array(self._design(coords) @ self._coefficients)
        out = data.copy()
        out[self.column] = as_float_array(residuals + trend)
        return out

    @property
    def coefficients(self) -> FloatArray | None:
        return self._coefficients

    def __repr__(self) -> str:
        return f"Detrend({self.column!r}, degree={self.degree})"
