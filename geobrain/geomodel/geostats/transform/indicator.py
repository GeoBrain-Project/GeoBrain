"""
Indicator encoding (continuous → binary cumulative indicators).

For each threshold ``tᵢ``, adds a column ``ind_i`` with values
``1.0`` where ``data[column] <= tᵢ`` and ``0.0`` otherwise.

This is the input layer for indicator kriging / soft-IK. The
encoding is not invertible.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import as_float_array
from .base import Transform

if TYPE_CHECKING:
    from ...frames import GeoFrame


class IndicatorEncode(Transform):
    """
    Cumulative-indicator encoding at supplied thresholds.

    Args:
        column: input column name.
        thresholds: ordered list of cutoff values (need not be
            monotone; results are independent per threshold).
        prefix: column-name prefix (default ``"ind"``), output
            columns are ``f"{prefix}_{i}"`` for ``i = 0…len-1``.
    """

    def __init__(
        self,
        column: str,
        thresholds: list[float],
        *,
        prefix: str = "ind",
    ) -> None:
        if not thresholds:
            raise GeoBrainError(
                "thresholds must contain at least one cutoff",
                object_name="IndicatorEncode", field="thresholds",
                expected="non-empty", actual=thresholds,
            )
        self.column = column
        self.thresholds: list[float] = [float(t) for t in thresholds]
        self.prefix = prefix

    def fit(self, data: "GeoFrame") -> "IndicatorEncode":
        """No fitting needed: column existence checked upfront."""
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="IndicatorEncode.fit", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )
        return self

    def transform(self, data: "GeoFrame") -> "GeoFrame":
        if self.column not in data.columns:
            raise GeoBrainError(
                f"column {self.column!r} is missing",
                object_name="IndicatorEncode.transform", field="column",
                expected=f"one of {data.columns}", actual=self.column,
            )
        values = as_float_array(data[self.column])
        out = data.copy()
        for i, t in enumerate(self.thresholds):
            out[f"{self.prefix}_{i}"] = as_float_array(
                (values <= t).astype(np.float64)
            )
        return out

    @property
    def output_columns(self) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(len(self.thresholds))]

    def __repr__(self) -> str:
        return f"IndicatorEncode({self.column!r}, thresholds={self.thresholds})"


__all__ = ["IndicatorEncode"]
