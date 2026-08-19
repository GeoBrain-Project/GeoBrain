"""
Experimental-variogram calculator.

The
calculator scans all point pairs in a :class:`GeoFrame`, bins them by
lag distance, and aggregates semivariance with a choice of estimator
(``matheron``, ``cressie``, ``dowd``, ``genton``).

Supports:

- automatic lag spacing (``lag_distance=None`` → data extent / 30);
- directional filtering by ``azimuth``, optional ``dip``,
  ``azimuth_tol``, and lateral ``bandwidth``;
- user-supplied or default lag tolerance (half the lag spacing).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import BoolArray, FloatArray, IntArray, as_bool_array, as_float_array, as_int_array
from .estimators import cressie_hawkins, dowd, genton

if TYPE_CHECKING:
    from ...frames import GeoFrame
    from .experimental import ExperimentalVariogram


def _zeros_float(n: int) -> FloatArray:
    return as_float_array(np.zeros(n, dtype=np.float64))


def _zeros_int(n: int) -> IntArray:
    return as_int_array(np.zeros(n, dtype=np.int64))


def _direction_mask(
    coords: FloatArray,
    azimuth: float,
    dip: float | None,
    azimuth_tol: float,
    bandwidth: float | None,
) -> BoolArray:
    """
    Boolean mask over ``pdist``-ordered pairs for directional filtering.

    The mask is ``True`` when the *unsigned* azimuth of the pair lies
    within ``azimuth_tol`` of ``azimuth`` (modulo 180°), and (if
    ``bandwidth`` is given) the perpendicular offset is ≤ bandwidth.
    A ``dip`` ≠ 0 enforces a similar tolerance on the pair's dip
    angle (3-D only).
    """
    n = int(coords.shape[0])
    n_pairs = n * (n - 1) // 2

    x = as_float_array(coords[:, 0])
    y = as_float_array(coords[:, 1])
    has_z = coords.shape[1] >= 3

    dx_vec = _zeros_float(n_pairs)
    dy_vec = _zeros_float(n_pairs)
    dz_vec: FloatArray | None = _zeros_float(n_pairs) if has_z and dip is not None else None

    idx = 0
    for i in range(n):
        count = n - i - 1
        dx_vec[idx : idx + count] = x[i + 1 :] - x[i]
        dy_vec[idx : idx + count] = y[i + 1 :] - y[i]
        if dz_vec is not None:
            dz_vec[idx : idx + count] = coords[i + 1 :, 2] - coords[i, 2]
        idx += count

    pair_azimuth = as_float_array(np.degrees(np.arctan2(dx_vec, dy_vec)) % 360)
    azimuth_mod = azimuth % 360

    angle_diff = as_float_array(np.abs(pair_azimuth - azimuth_mod) % 360)
    angle_diff = as_float_array(np.minimum(angle_diff, 360 - angle_diff))

    reverse_azimuth = as_float_array((pair_azimuth + 180) % 360)
    angle_diff_rev = as_float_array(np.abs(reverse_azimuth - azimuth_mod) % 360)
    angle_diff_rev = as_float_array(np.minimum(angle_diff_rev, 360 - angle_diff_rev))

    mask = as_bool_array((angle_diff <= azimuth_tol) | (angle_diff_rev <= azimuth_tol))

    if bandwidth is not None:
        az_rad = float(np.radians(azimuth_mod))
        dir_x = float(np.sin(az_rad))
        dir_y = float(np.cos(az_rad))
        perp_dist = as_float_array(np.abs(dx_vec * dir_y - dy_vec * dir_x))
        mask = as_bool_array(mask & (perp_dist <= bandwidth))

    if dz_vec is not None and dip is not None and dip != 0.0:
        horiz_dist = as_float_array(np.sqrt(dx_vec**2 + dy_vec**2))
        pair_dip = as_float_array(np.degrees(np.arctan2(dz_vec, horiz_dist)))
        dip_diff = as_float_array(np.abs(pair_dip - dip))
        dip_diff_rev = as_float_array(np.abs(-pair_dip - dip))
        dip_tol = azimuth_tol
        mask = as_bool_array(mask & ((dip_diff <= dip_tol) | (dip_diff_rev <= dip_tol)))

    return mask


class VariogramCalculator:
    """
    Experimental-variogram calculator (binned semivariance).

    Args:
        lag_distance: spacing between lag bins. ``None`` → auto:
            ``max(extent) / (2 · n_lags)``.
        n_lags: number of lag bins (default 15).
        tolerance: lag tolerance (default ``lag_distance / 2``).
        bandwidth: directional bandwidth (lateral offset cap).
        estimator: one of ``"matheron"``, ``"cressie"``, ``"dowd"``,
            ``"genton"``.
        azimuth: azimuth in degrees (clockwise from North). ``None`` →
            omnidirectional.
        dip: dip in degrees (3-D only). Default ``None`` (no dip
            filtering even when 3-D coords are present).
        azimuth_tol: angular tolerance in degrees (default 22.5).
    """

    ESTIMATORS = ("matheron", "cressie", "dowd", "genton")

    def __init__(
        self,
        lag_distance: float | None = None,
        n_lags: int = 15,
        tolerance: float | None = None,
        bandwidth: float | None = None,
        estimator: str = "matheron",
        azimuth: float | None = None,
        dip: float | None = None,
        azimuth_tol: float = 22.5,
    ) -> None:
        if estimator not in self.ESTIMATORS:
            raise GeoBrainError(
                "unknown estimator",
                object_name="VariogramCalculator", field="estimator",
                expected=f"one of {self.ESTIMATORS}", actual=estimator,
            )
        if n_lags < 1:
            raise GeoBrainError(
                "n_lags must be >= 1",
                object_name="VariogramCalculator", field="n_lags",
                expected=">= 1", actual=n_lags,
            )
        if lag_distance is not None and lag_distance <= 0:
            raise GeoBrainError(
                "lag_distance must be positive",
                object_name="VariogramCalculator", field="lag_distance",
                expected="> 0 or None", actual=lag_distance,
            )
        self.lag_distance = lag_distance
        self.n_lags = int(n_lags)
        self.tolerance = tolerance
        self.bandwidth = bandwidth
        self.estimator = estimator
        self.azimuth = azimuth
        self.dip = dip
        self.azimuth_tol = float(azimuth_tol)

    # ------------------------------------------------------------------

    def compute(
        self,
        data: "GeoFrame",
        column: str,
    ) -> "ExperimentalVariogram":
        """Return an :class:`ExperimentalVariogram` for ``column``."""
        from .experimental import ExperimentalVariogram

        coords = as_float_array(data.geometry.coords)
        values = as_float_array(data[column])
        n = int(values.size)

        lag_dist = self.lag_distance
        if lag_dist is None:
            extent = as_float_array(np.max(coords, axis=0) - np.min(coords, axis=0))
            if np.any(extent > 0):
                max_extent = float(np.max(extent[extent > 0]))
            else:
                max_extent = 1.0
            lag_dist = max_extent / (self.n_lags * 2)
        lag_dist = float(lag_dist)

        tol = self.tolerance if self.tolerance is not None else lag_dist * 0.5

        lags = _zeros_float(self.n_lags)
        gammas = _zeros_float(self.n_lags)
        pairs = _zeros_int(self.n_lags)

        direction_info = None
        if self.azimuth is not None:
            direction_info = (self.azimuth, self.dip if self.dip is not None else 0.0)

        if n < 2:
            lags[:] = (np.arange(self.n_lags) + 1) * lag_dist
            return ExperimentalVariogram(lags, gammas, pairs, direction=direction_info)

        pdist = cast(
            Callable[..., object],
            getattr(import_module("scipy.spatial.distance"), "pdist"),
        )
        dist_vec = as_float_array(pdist(coords, metric="euclidean"))
        # Absolute increments |vᵢ − vⱼ| (matheron/cressie/dowd).
        diff_vec = as_float_array(pdist(values.reshape(-1, 1), metric="euclidean"))
        # Signed increments vᵢ − vⱼ in the same pdist pair ordering, needed by
        # the Genton estimator (folding to |Δ| before forming its Qₙ scale
        # biases γ low: the absolute value destroys the increment spread).
        signed_vec: FloatArray | None = None
        if self.estimator == "genton":
            n_pairs = n * (n - 1) // 2
            signed_vec = _zeros_float(n_pairs)
            idx = 0
            for i in range(n):
                count_i = n - i - 1
                signed_vec[idx : idx + count_i] = values[i] - values[i + 1 :]
                idx += count_i

        dir_mask = None
        if self.azimuth is not None:
            dir_mask = _direction_mask(
                coords, self.azimuth, self.dip, self.azimuth_tol, self.bandwidth
            )

        # GSLIB overlapping-window binning: a pair contributes to bin k when
        # its distance is within ``tol`` of the bin centre (lag·k). The default
        # ``tol = lag/2`` gives contiguous, non-overlapping bins; ``tol > lag/2``
        # widens (overlaps) adjacent windows and smooths the experimental γ.
        for lag_idx in range(self.n_lags):
            nominal = (lag_idx + 1) * lag_dist
            bin_mask = as_bool_array(np.abs(dist_vec - nominal) <= tol)
            if dir_mask is not None:
                bin_mask = as_bool_array(bin_mask & dir_mask)
            count = int(np.sum(bin_mask))
            if count > 0:
                diffs = as_float_array(diff_vec[bin_mask])
                lags[lag_idx] = float(np.sum(dist_vec[bin_mask]))
                pairs[lag_idx] = count

                if self.estimator == "matheron":
                    gammas[lag_idx] = float(np.sum(diffs**2))
                elif self.estimator == "cressie":
                    gammas[lag_idx] = cressie_hawkins(diffs, count)
                elif self.estimator == "dowd":
                    gammas[lag_idx] = dowd(diffs)
                elif self.estimator == "genton":
                    assert signed_vec is not None
                    gammas[lag_idx] = genton(as_float_array(signed_vec[bin_mask]))

        nonempty = as_bool_array(pairs > 0)
        if self.estimator == "matheron":
            gammas[nonempty] /= 2.0 * pairs[nonempty]
        lags[nonempty] /= pairs[nonempty]
        # For empty bins, report the nominal lag centre so users can plot.
        empty = ~nonempty
        lags[empty] = (np.arange(self.n_lags)[empty] + 1) * lag_dist

        return ExperimentalVariogram(lags, gammas, pairs, direction=direction_info)

    # ------------------------------------------------------------------

    @staticmethod
    def fit(
        data: "GeoFrame",
        column: str,
        *,
        kind: str = "auto",
        strict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """One-step convenience: compute the experimental variogram then fit a model."""
        calc = VariogramCalculator(**kwargs)
        exp_var = calc.compute(data, column)
        return exp_var.fit_model(kind=kind, strict=strict)

    def __repr__(self) -> str:
        parts = [
            f"lag={self.lag_distance}",
            f"n_lags={self.n_lags}",
            f"estimator={self.estimator!r}",
        ]
        if self.azimuth is not None:
            parts.append(f"azimuth={self.azimuth}")
        if self.dip is not None:
            parts.append(f"dip={self.dip}")
        return f"VariogramCalculator({', '.join(parts)})"


__all__ = ["VariogramCalculator"]
