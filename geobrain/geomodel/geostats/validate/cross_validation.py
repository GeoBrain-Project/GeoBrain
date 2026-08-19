"""Spatial cross-validation: leave-one-out and k-fold.

Public surface:

- :class:`CrossValidationResult`: dataclass bundling per-point
  actuals / estimates / variances / errors plus a :meth:`summary`
  method (see :func:`cross_validation_report` for fields).
- :func:`leave_one_out`: drop each point in turn, refit the
  estimator on the rest, score the dropped point.
- :func:`k_fold`: partition points into ``n_folds`` (optionally
  with spatial blocking), score each test fold from a train fit.
- class wrappers :class:`LeaveOneOut` / :class:`KFold` that expose
  a class-based form.

The estimator argument follows the 0.2 kriging protocol; it is callable as
``estimator(data: GeoFrame, domain) -> GeoFrame`` and owns immutable
``PropertyMetadata``. Cross-validation verifies that identity and never
mutates estimator configuration.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import BoolArray, FloatArray, as_bool_array, as_float_array
from ...frames import GeoFrame, GeoPoints, PropertyMetadata
from .diagnostics import cross_validation_report

__all__ = [
    "CrossValidationResult",
    "KFold",
    "LeaveOneOut",
    "SpatialEstimator",
    "k_fold",
    "leave_one_out",
]


class SpatialEstimator(Protocol):
    """
    Protocol for an estimator usable with cross-validation.

    Must expose immutable ``property`` metadata and behave as a
    callable ``(train_data, domain) -> result GeoFrame`` that
    returns at least an ``estimate`` and ``variance`` column.
    """

    property: PropertyMetadata

    def __call__(self, data: GeoFrame, domain: Any) -> GeoFrame: ...


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CrossValidationResult:
    """
    Result of a cross-validation run.

    Attributes:
        coords:    ``(n, ndim)`` array of the held-out point coordinates.
        actual:    ``(n,)`` true values at the held-out points.
        estimate:  ``(n,)`` kriging estimates at the held-out points.
        variance:  ``(n,)`` kriging variances at the held-out points.
        error:     ``(n,)`` ``actual − estimate``.
        fold:      ``(n,)`` fold index per point (LOO uses 0..n-1; k-fold
            uses 0..n_folds-1). Always present so downstream code can
            re-aggregate by fold uniformly.
    """

    coords: FloatArray
    actual: FloatArray
    estimate: FloatArray
    variance: FloatArray
    error: FloatArray
    fold: FloatArray

    def summary(self) -> dict[str, float]:
        """Return cross-validation summary statistics: see
        :func:`geobrain.geomodel.geostats.validate.diagnostics.cross_validation_report`.
        """
        summary: dict[str, float] = cross_validation_report(
            self.actual,
            self.estimate,
            self.variance,
        )
        return summary

    def to_geotable(self) -> GeoFrame:
        """
        Repackage as a :class:`GeoFrame` (same geometry as the
        held-out GeoPoints, with ``actual`` / ``estimate`` / ``variance``
        / ``error`` / ``fold`` columns)."""
        return GeoFrame(
            GeoPoints(self.coords),
            properties={
                "actual": self.actual,
                "estimate": self.estimate,
                "variance": self.variance,
                "error": self.error,
                "fold": self.fold,
            },
        )

    def __repr__(self) -> str:
        return (
            f"CrossValidationResult(n={self.actual.size}, "
            f"rmse={float(np.sqrt(np.mean(self.error**2))):.4g})"
        )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _validate_inputs(estimator: SpatialEstimator, data: GeoFrame, column: str) -> None:
    if not isinstance(data, GeoFrame):
        raise GeoBrainError(
            "data must be a GeoFrame",
            object_name="cross_validation",
            field="data",
            expected="GeoFrame",
            actual=type(data).__name__,
        )
    if column not in data.columns:
        raise GeoBrainError(
            f"column {column!r} is missing",
            object_name="cross_validation",
            field="column",
            expected=f"one of {data.columns}",
            actual=column,
        )
    if not callable(estimator):
        raise GeoBrainError(
            "estimator must be callable",
            object_name="cross_validation",
            field="estimator",
            expected="callable",
            actual=type(estimator).__name__,
        )
    property_metadata = getattr(estimator, "property", None)
    if not isinstance(property_metadata, PropertyMetadata):
        raise GeoBrainError(
            "estimator does not expose PropertyMetadata",
            object_name="cross_validation",
            field="estimator.property",
            expected="PropertyMetadata",
            actual=type(property_metadata).__name__,
        )
    if property_metadata.name != column or data.metadata_for(column) != property_metadata:
        raise GeoBrainError(
            "cross-validation property does not match the estimator",
            object_name="cross_validation",
            field="column",
            expected=property_metadata.to_dict(),
            actual=data.metadata_for(column).to_dict(),
        )


# ----------------------------------------------------------------------
# Leave-one-out
# ----------------------------------------------------------------------


def leave_one_out(
    estimator: SpatialEstimator,
    data: GeoFrame,
    column: str,
) -> CrossValidationResult:
    """
    Run leave-one-out cross-validation.

    For each data point, drop it, refit the estimator on the remaining
    ``n − 1`` points, and score the dropped point. Returns a
    :class:`CrossValidationResult` aligned to the original ordering.

    Args:
        estimator: the spatial estimator to cross-validate.
        data: conditioning frame.
        column: data column to validate.
    """
    _validate_inputs(estimator, data, column)
    n = len(data)
    if n < 2:
        raise GeoBrainError(
            "leave-one-out requires at least 2 data points",
            object_name="leave_one_out",
            field="data",
            expected=">= 2",
            actual=n,
        )

    coords = as_float_array(data.geometry.coords)
    actuals = as_float_array(data[column])
    estimates: FloatArray = as_float_array(np.full(n, np.nan, dtype=np.float64))
    variances: FloatArray = as_float_array(np.full(n, np.nan, dtype=np.float64))

    for i in range(n):
        mask: BoolArray = as_bool_array(np.ones(n, dtype=bool))
        mask[i] = False
        train = data.where(as_bool_array(mask))
        target = GeoPoints(coords[i : i + 1])
        result = estimator(train, target)
        estimates[i] = float(result["estimate"][0])
        variances[i] = float(result["variance"][0])

    errors = as_float_array(actuals - estimates)
    fold: FloatArray = as_float_array(np.arange(n, dtype=np.float64))
    return CrossValidationResult(
        coords=coords,
        actual=actuals,
        estimate=as_float_array(estimates),
        variance=as_float_array(variances),
        error=errors,
        fold=as_float_array(fold),
    )


class LeaveOneOut:
    """Class wrapper around :func:`leave_one_out`."""

    def __init__(self, estimator: SpatialEstimator) -> None:
        self.estimator = estimator
        self._result: CrossValidationResult | None = None

    def run(self, data: GeoFrame, column: str) -> CrossValidationResult:
        self._result = leave_one_out(self.estimator, data, column)
        return self._result

    def summary(self) -> dict[str, float]:
        if self._result is None:
            raise GeoBrainError(
                "LeaveOneOut.summary() called before run()",
                object_name="LeaveOneOut.summary",
                field="_result",
                expected="CrossValidationResult",
                actual=None,
            )
        return self._result.summary()


# ----------------------------------------------------------------------
# k-fold
# ----------------------------------------------------------------------


def _assign_folds(
    coords: FloatArray,
    n_folds: int,
    block_size: float | None,
    rng: np.random.Generator,
) -> FloatArray:
    n = coords.shape[0]
    if block_size is not None:
        if block_size <= 0:
            raise GeoBrainError(
                "block_size must be positive",
                object_name="k_fold",
                field="block_size",
                expected="> 0",
                actual=block_size,
            )
        # Block-index each available spatial dimension (x, y, and z when
        # present). A collision-free block key is obtained by running
        # ``np.unique(..., axis=0, return_inverse=True)`` on the stacked
        # per-dimension block indices: mirroring the stride-based
        # collision-free cell id used by ``decluster._cell_weights``. The
        # earlier ``x*100000 + y`` scheme dropped z entirely, collapsing
        # 3-D data onto the xy plane so vertically-separated points shared
        # folds; ``axis=0`` uniqueness keeps every dimension distinct
        # without any hand-tuned stride.
        ndim = coords.shape[1]
        block_index = np.floor(coords[:, :ndim] / block_size).astype(np.int64)
        # ``return_inverse`` gives each point the index of its unique block
        # row: a dense, collision-free block id in ``0..n_blocks-1``.
        unique_blocks, block_inverse = np.unique(block_index, axis=0, return_inverse=True)
        block_inverse = np.asarray(block_inverse).reshape(-1)
        n_blocks = unique_blocks.shape[0]
        # Round-robin the shuffled blocks across folds (block at shuffled
        # position ``i`` gets fold ``i % n_folds``): balanced folds, same
        # convention as the previous implementation.
        order = np.arange(n_blocks)
        rng.shuffle(order)
        fold_of_block = np.empty(n_blocks, dtype=np.int64)
        fold_of_block[order] = np.arange(n_blocks) % n_folds
        fold = fold_of_block[block_inverse].astype(np.float64)
    else:
        fold = (np.arange(n) % n_folds).astype(np.float64)
        rng.shuffle(fold)
    return as_float_array(fold)


def k_fold(
    estimator: SpatialEstimator,
    data: GeoFrame,
    column: str,
    *,
    n_folds: int = 10,
    block_size: float | None = None,
    seed: int | None = None,
) -> CrossValidationResult:
    """
    Run k-fold cross-validation.

    Args:
        estimator: any callable matching :class:`SpatialEstimator`.
        data: training data with ``column``.
        column: target column to score.
        n_folds: number of folds (must be ≥ 2 and ≤ ``len(data)``).
        block_size: if given, assign folds by spatial block of that
            (Euclidean) size, useful for spatially-correlated data
            to avoid optimistic CV scores. ``None`` (default) means
            random assignment.
        seed: RNG seed for reproducible fold assignment.
    """
    _validate_inputs(estimator, data, column)
    n = len(data)
    if n_folds < 2:
        raise GeoBrainError(
            "k_fold requires n_folds >= 2",
            object_name="k_fold",
            field="n_folds",
            expected=">= 2",
            actual=n_folds,
        )
    if n_folds > n:
        raise GeoBrainError(
            "k_fold requires n_folds <= len(data)",
            object_name="k_fold",
            field="n_folds",
            expected=f"<= {n}",
            actual=n_folds,
        )

    rng = np.random.default_rng(seed)
    coords = as_float_array(data.geometry.coords)
    actuals = as_float_array(data[column])
    fold_assignment = _assign_folds(coords, n_folds, block_size, rng)

    estimates: FloatArray = as_float_array(np.full(n, np.nan, dtype=np.float64))
    variances: FloatArray = as_float_array(np.full(n, np.nan, dtype=np.float64))

    for k in range(n_folds):
        test_mask = as_bool_array(fold_assignment == k)
        train_mask = as_bool_array(~test_mask)
        if not bool(np.any(test_mask)) or not bool(np.any(train_mask)):
            continue
        train = data.where(train_mask)
        test_coords = coords[test_mask]
        target = GeoPoints(test_coords)
        result = estimator(train, target)
        estimates[test_mask] = as_float_array(result["estimate"])
        variances[test_mask] = as_float_array(result["variance"])

    errors = as_float_array(actuals - estimates)
    return CrossValidationResult(
        coords=coords,
        actual=actuals,
        estimate=as_float_array(estimates),
        variance=as_float_array(variances),
        error=errors,
        fold=fold_assignment,
    )


class KFold:
    """Class wrapper around :func:`k_fold`.

    Args:
        estimator: the spatial estimator to cross-validate.
        n_folds: fold count.
        block_size: optional spatial-block fold assignment [m].
        seed: fold-assignment seed.
    """

    def __init__(
        self,
        estimator: SpatialEstimator,
        n_folds: int = 10,
        block_size: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_folds = int(n_folds)
        self.block_size = block_size
        self.seed = seed
        self._result: CrossValidationResult | None = None

    def run(self, data: GeoFrame, column: str) -> CrossValidationResult:
        self._result = k_fold(
            self.estimator,
            data,
            column,
            n_folds=self.n_folds,
            block_size=self.block_size,
            seed=self.seed,
        )
        return self._result

    def summary(self) -> dict[str, float]:
        if self._result is None:
            raise GeoBrainError(
                "KFold.summary() called before run()",
                object_name="KFold.summary",
                field="_result",
                expected="CrossValidationResult",
                actual=None,
            )
        return self._result.summary()
