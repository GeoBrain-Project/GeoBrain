"""
Histogram and scatter smoothing transforms.

Two transforms:

- :class:`HistogramSmooth`: kernel-density smoothing of a column's
  empirical CDF (Gaussian KDE with Silverman's-rule bandwidth).
- :class:`ScatterSmooth`: moving-average or polynomial fit of one
  column against another (denoising / proxy generation).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_bool_array, as_float_array, zeros_float
from .base import Transform

if TYPE_CHECKING:
    from ...frames import GeoFrame

__all__ = ["HistogramSmooth", "ScatterSmooth"]


class HistogramSmooth(Transform):
    """
    Gaussian-KDE smoothing of a column's empirical CDF.

    Args:
        column: input column.
        n_bins: resolution of the CDF evaluation grid.
        bandwidth: Gaussian bandwidth ``h``. ``None`` → Silverman's rule
            of thumb ``0.9 · min(σ, IQR/1.349) · n^{-1/5}``.
        output_column: output column name (default ``f"{column}_cdf"``).
    """

    def __init__(
        self,
        column: str,
        *,
        n_bins: int = 50,
        bandwidth: float | None = None,
        output_column: str | None = None,
    ) -> None:
        if n_bins < 2:
            raise GeoBrainError(
                "n_bins must be >= 2",
                object_name="HistogramSmooth", field="n_bins",
                expected=">= 2", actual=n_bins,
            )
        if bandwidth is not None and bandwidth <= 0:
            raise GeoBrainError(
                "bandwidth must be positive",
                object_name="HistogramSmooth", field="bandwidth",
                expected="> 0", actual=bandwidth,
            )
        self.column = column
        self.n_bins = int(n_bins)
        self.bandwidth = bandwidth
        self.output_column = output_column or f"{column}_cdf"
        self._eval_points: FloatArray | None = None
        self._cdf_values: FloatArray | None = None
        self._bandwidth_used: float | None = None
        self._fitted = False

    def fit(self, data: "GeoFrame") -> "HistogramSmooth":
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="HistogramSmooth.fit", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )
        values = as_float_array(data[self.column])
        finite = as_bool_array(np.isfinite(values))
        valid = as_float_array(values[finite])
        if valid.size < 2:
            raise GeoBrainError(
                "need >= 2 finite values for KDE",
                object_name="HistogramSmooth.fit", field="column",
                expected=">= 2 finite values", actual=valid.size,
            )

        n = valid.size
        sample_std = float(np.std(valid, ddof=1))
        std = sample_std if sample_std > 0 else 1.0
        iqr = float(np.percentile(valid, 75) - np.percentile(valid, 25))
        spread = min(std, iqr / 1.349) if iqr > 0 else std
        silverman = 0.9 * spread * n ** (-0.2)
        h = self.bandwidth if self.bandwidth is not None else max(silverman, 1.0e-12)
        self._bandwidth_used = h

        vmin = float(np.min(valid)) - 3.0 * h
        vmax = float(np.max(valid)) + 3.0 * h
        eval_points = as_float_array(np.linspace(vmin, vmax, self.n_bins))
        self._eval_points = eval_points

        diff = eval_points[:, None] - valid[None, :]
        kernel = np.exp(-0.5 * (diff / h) ** 2) / (h * np.sqrt(2 * np.pi))
        pdf = as_float_array(np.mean(kernel, axis=1))

        dx = np.diff(eval_points)
        cdf_raw = np.concatenate(
            [[0.0], np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)]
        )
        cdf_max = float(cdf_raw[-1])
        self._cdf_values = as_float_array(
            cdf_raw / cdf_max if cdf_max > 0
            else np.linspace(0.0, 1.0, self.n_bins)
        )
        self._fitted = True
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if (
            not self._fitted
            or self._eval_points is None
            or self._cdf_values is None
        ):
            raise GeoBrainError(
                "HistogramSmooth.transform called before fit()",
                object_name="HistogramSmooth.transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        values = as_float_array(data[self.column])
        cdf_out = as_float_array(
            np.interp(values, self._eval_points, self._cdf_values)
        )
        out = data.copy()
        out[self.output_column] = cdf_out
        return out

    @property
    def bandwidth_used(self) -> float | None:
        return self._bandwidth_used

    @property
    def eval_points(self) -> FloatArray | None:
        return self._eval_points

    @property
    def cdf_values(self) -> FloatArray | None:
        return self._cdf_values

    def __repr__(self) -> str:
        return f"HistogramSmooth({self.column!r})"


class ScatterSmooth(Transform):
    """
    Bivariate smoothing of ``y_column`` vs ``x_column``.

    Args:
        x_column / y_column: predictor / response columns.
        method: ``"moving_average"`` (sorted-X k-NN mean) or
            ``"polynomial"`` (degree-``degree`` least squares).
        window: half-width for the moving-average mode.
        degree: degree for the polynomial mode.
        output_column: name of the smoothed-Y column (default
            ``f"{y_column}_smooth"``).
    """

    def __init__(
        self,
        x_column: str,
        y_column: str,
        *,
        output_column: str | None = None,
        method: str = "moving_average",
        window: int = 5,
        degree: int = 2,
    ) -> None:
        if method not in ("moving_average", "polynomial"):
            raise GeoBrainError(
                "method must be 'moving_average' or 'polynomial'",
                object_name="ScatterSmooth", field="method",
                expected="'moving_average' or 'polynomial'", actual=method,
            )
        if window < 1:
            raise GeoBrainError(
                "window must be >= 1",
                object_name="ScatterSmooth", field="window",
                expected=">= 1", actual=window,
            )
        if degree < 0:
            raise GeoBrainError(
                "degree must be >= 0",
                object_name="ScatterSmooth", field="degree",
                expected=">= 0", actual=degree,
            )
        self.x_column = x_column
        self.y_column = y_column
        self.output_column = output_column or f"{y_column}_smooth"
        self.method = method
        self.window = int(window)
        self.degree = int(degree)
        self._poly_coeffs: FloatArray | None = None
        self._sorted_x: FloatArray | None = None
        self._smoothed_y: FloatArray | None = None
        self._fitted = False

    def fit(self, data: "GeoFrame") -> "ScatterSmooth":
        for col in (self.x_column, self.y_column):
            if col not in data.columns:
                raise GeoBrainError(
                    f"column {col!r} is missing",
                    object_name="ScatterSmooth.fit", field="column",
                    expected=f"one of {data.columns}", actual=col,
                )
        x = as_float_array(data[self.x_column])
        y = as_float_array(data[self.y_column])

        if self.method == "polynomial":
            self._poly_coeffs = as_float_array(np.polyfit(x, y, self.degree))
        else:
            order = np.argsort(x, kind="mergesort")
            sx = as_float_array(x[order])
            sy = as_float_array(y[order])
            n = sx.size
            hw = max(self.window // 2, 1)
            smoothed = zeros_float(n)
            for i in range(n):
                lo = max(0, i - hw)
                hi = min(n, i + hw + 1)
                smoothed[i] = float(np.mean(sy[lo:hi]))
            self._sorted_x = sx
            self._smoothed_y = smoothed
        self._fitted = True
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if not self._fitted:
            raise GeoBrainError(
                "ScatterSmooth.transform called before fit()",
                object_name="ScatterSmooth.transform", field="_fitted",
                expected="True", actual=self._fitted,
            )
        x = as_float_array(data[self.x_column])
        if self.method == "polynomial":
            if self._poly_coeffs is None:
                raise GeoBrainError(
                    "polynomial fit state missing",
                    object_name="ScatterSmooth.transform",
                    field="_poly_coeffs", expected="FloatArray", actual=None,
                )
            y_smooth = as_float_array(np.polyval(self._poly_coeffs, x))
        else:
            if self._sorted_x is None or self._smoothed_y is None:
                raise GeoBrainError(
                    "moving-average fit state missing",
                    object_name="ScatterSmooth.transform",
                    field="_sorted_x", expected="FloatArray", actual=None,
                )
            y_smooth = as_float_array(
                np.interp(x, self._sorted_x, self._smoothed_y)
            )
        out = data.copy()
        out[self.output_column] = y_smooth
        return out

    def __repr__(self) -> str:
        return (
            f"ScatterSmooth({self.x_column!r}, {self.y_column!r}, "
            f"method={self.method!r})"
        )
