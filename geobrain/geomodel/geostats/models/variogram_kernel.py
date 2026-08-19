"""Variogram kernels: single nested structures.

Implements
all 14 GSLIB variogram families::

     1 SPHERICAL        8 RATIONAL
     2 EXPONENTIAL      9 CUBIC
     3 GAUSSIAN        10 LINEAR
     4 POWER           11 CIRCULAR
     5 HOLE_EFFECT     12 JBESSEL
     6 MATERN          13 SUPERSPHERICAL
     7 STABLE          14 HYPERSPHERICAL

Each :class:`VariogramKernel` evaluates the isotropic semivariogram
:math:`\\gamma(h)` along the *major* range direction. Anisotropy is
handled upstream by :func:`setup_rotation_matrix` (the kernel
receives the GSLIB *reduced* lag ``h`` and ignores the angles;
they live on the kernel only for bookkeeping / ``to_core_params``).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special  # type: ignore[import-untyped]

from ....core import GeoBrainError
from ...errors import GeomodelNumericsError
from ...frames._arrays import FloatArray, as_bool_array, as_float_array


class VariogramKernel:
    """
    Single variogram structure (one nested component).

    Args:
        kind: integer kernel id (1–14); use one of the class
            constants ``SPHERICAL``…``HYPERSPHERICAL``.
        contribution: sill contribution ``C`` of this structure
            (i.e. variance contribution; must be ``> 0``).
        ranges: range parameters. May be a scalar (isotropic) or a
            tuple ``(a_max, a_mid, a_min)``.
        angles: GSLIB rotation angles in degrees
            ``(azimuth, dip, plunge)``. Default ``(0, 0, 0)``.
        **params: extra family-specific parameters. Common ones:

            - ``nu``:     Matern smoothness (default 1.5),
                          J-Bessel order (default 0.5),
                          SuperSpherical shape (default 1.0)
            - ``alpha``: Stable exponent (default 1.5),
                          Rational shape (default 1.0)
            - ``dim``:    SuperSpherical / HyperSpherical dimension
                          (default 3)
    """

    SPHERICAL = 1
    EXPONENTIAL = 2
    GAUSSIAN = 3
    POWER = 4
    HOLE_EFFECT = 5
    MATERN = 6
    STABLE = 7
    RATIONAL = 8
    CUBIC = 9
    LINEAR = 10
    CIRCULAR = 11
    JBESSEL = 12
    SUPERSPHERICAL = 13
    HYPERSPHERICAL = 14

    _NAMES = {
        1: "Spherical",
        2: "Exponential",
        3: "Gaussian",
        4: "Power",
        5: "HoleEffect",
        6: "Matern",
        7: "Stable",
        8: "Rational",
        9: "Cubic",
        10: "Linear",
        11: "Circular",
        12: "JBessel",
        13: "SuperSpherical",
        14: "HyperSpherical",
    }

    def __init__(
        self,
        kind: int,
        contribution: float,
        ranges: float | tuple[float, ...],
        angles: tuple[float, ...] = (0.0, 0.0, 0.0),
        **params: Any,
    ) -> None:
        if kind not in self._NAMES:
            raise GeoBrainError(
                "unknown variogram kernel kind",
                object_name="VariogramKernel",
                field="kind",
                expected=f"one of {list(self._NAMES)}",
                actual=kind,
            )
        if contribution <= 0:
            raise GeoBrainError(
                "variogram contribution must be positive",
                object_name="VariogramKernel",
                field="contribution",
                expected="> 0",
                actual=contribution,
            )
        ranges_t: tuple[float, ...]
        if isinstance(ranges, (int, float)):
            ranges_t = (float(ranges), float(ranges), float(ranges))
        else:
            ranges_t = tuple(float(r) for r in ranges)
            if len(ranges_t) not in (2, 3):
                raise GeoBrainError(
                    "ranges must be a scalar or a length-2/3 tuple",
                    object_name="VariogramKernel",
                    field="ranges",
                    expected="scalar or (a_max,a_mid[,a_min])",
                    actual=ranges,
                )
            if len(ranges_t) == 2:
                ranges_t = (ranges_t[0], ranges_t[1], ranges_t[1])
        for r in ranges_t:
            if r < 0:
                raise GeoBrainError(
                    "ranges must be non-negative",
                    object_name="VariogramKernel",
                    field="ranges",
                    expected=">= 0",
                    actual=ranges_t,
                )
        angles_t = tuple(float(a) for a in angles)
        if len(angles_t) != 3:
            raise GeoBrainError(
                "angles must have length 3 (azimuth, dip, plunge)",
                object_name="VariogramKernel",
                field="angles",
                expected="length 3",
                actual=len(angles_t),
            )
        self.kind = int(kind)
        self.contribution = float(contribution)
        self.ranges = ranges_t
        self.angles = angles_t
        self.params: dict[str, Any] = dict(params)

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._NAMES[self.kind]

    @property
    def range_max(self) -> float:
        return self.ranges[0]

    @property
    def anis1(self) -> float:
        """range_mid / range_max (GSLIB convention)."""
        return self.ranges[1] / self.ranges[0] if self.ranges[0] > 0 else 1.0

    @property
    def anis2(self) -> float:
        """range_min / range_max (GSLIB convention)."""
        return self.ranges[2] / self.ranges[0] if self.ranges[0] > 0 else 1.0

    def require_stationary_covariance(self, *, object_name: str) -> None:
        """Reject generalized variograms that do not define bounded covariance."""
        if self.kind in (self.POWER, self.LINEAR):
            raise GeomodelNumericsError(
                "variogram kernel does not define a stationary covariance",
                object_name=object_name,
                field="variogram.kind",
                expected="bounded stationary covariance kernel",
                actual=self.name,
            )
        numeric = (self.contribution, *self.ranges, *self.angles)
        if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
            raise GeomodelNumericsError(
                "variogram kernel parameters must be finite",
                object_name=object_name,
                field="variogram",
                expected="finite covariance parameters",
                actual=numeric,
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, h: object) -> FloatArray:
        """Isotropic γ(h) for this single structure (no nugget)."""
        h_arr = as_float_array(h)
        a = self.ranges[0]
        c = self.contribution
        k = self.kind

        if k == self.SPHERICAL:
            return self._spherical(h_arr, a, c)
        if k == self.EXPONENTIAL:
            return self._exponential(h_arr, a, c)
        if k == self.GAUSSIAN:
            return self._gaussian(h_arr, a, c)
        if k == self.POWER:
            return self._power(h_arr, a, c)
        if k == self.HOLE_EFFECT:
            return self._hole_effect(h_arr, a, c)
        if k == self.MATERN:
            return self._matern(h_arr, a, c)
        if k == self.STABLE:
            return self._stable(h_arr, a, c)
        if k == self.RATIONAL:
            return self._rational(h_arr, a, c)
        if k == self.CUBIC:
            return self._cubic(h_arr, a, c)
        if k == self.LINEAR:
            return self._linear(h_arr, a, c)
        if k == self.CIRCULAR:
            return self._circular(h_arr, a, c)
        if k == self.JBESSEL:
            return self._jbessel(h_arr, a, c)
        if k == self.SUPERSPHERICAL:
            return self._superspherical(h_arr, a, c)
        if k == self.HYPERSPHERICAL:
            return self._hyperspherical(h_arr, a, c)
        return as_float_array(np.zeros_like(h_arr, dtype=np.float64))

    # ------------------------------------------------------------------
    # Individual kernels
    # ------------------------------------------------------------------

    @staticmethod
    def _step_to_c(h: FloatArray, c: float) -> FloatArray:
        """Helper: 0 at h=0, c elsewhere: used when range collapses."""
        return as_float_array(np.where(h > 0, c, 0.0))

    def _spherical(self, h: FloatArray, a: float, c: float) -> FloatArray:
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(np.minimum(h / a, 1.0))
        return as_float_array(c * (1.5 * hr - 0.5 * hr**3))

    def _exponential(self, h: FloatArray, a: float, c: float) -> FloatArray:
        if a <= 0:
            return self._step_to_c(h, c)
        return as_float_array(c * (1.0 - np.exp(-3.0 * h / a)))

    def _gaussian(self, h: FloatArray, a: float, c: float) -> FloatArray:
        if a <= 0:
            return self._step_to_c(h, c)
        return as_float_array(c * (1.0 - np.exp(-3.0 * (h / a) ** 2)))

    def _power(self, h: FloatArray, a: float, c: float) -> FloatArray:
        # Power model: 'a' is the exponent, 'c' the slope; ``range_max``
        # is reinterpreted as ``a`` per GSLIB convention.
        return as_float_array(c * np.power(h, a))

    def _hole_effect(self, h: FloatArray, a: float, c: float) -> FloatArray:
        if a <= 0:
            return self._step_to_c(h, c)
        return as_float_array(c * (1.0 - np.cos(h / a * np.pi)))

    def _matern(self, h: FloatArray, a: float, c: float) -> FloatArray:
        nu = float(self.params.get("nu", 1.5))
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(h / a)
        result = as_float_array(np.zeros_like(h, dtype=np.float64))
        nonzero = as_bool_array(hr > 0)
        if not np.any(nonzero):
            return result
        s = as_float_array(np.sqrt(2.0 * nu) * hr[nonzero])
        coef = 2.0 ** (1.0 - nu) / float(special.gamma(nu))
        bess = as_float_array(special.kv(nu, s))
        corr = as_float_array(np.clip(coef * (s**nu) * bess, 0.0, 1.0))
        result[nonzero] = c * (1.0 - corr)
        return result

    def _stable(self, h: FloatArray, a: float, c: float) -> FloatArray:
        alpha = float(self.params.get("alpha", 1.5))
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(h / a)
        return as_float_array(c * (1.0 - np.exp(-(hr**alpha))))

    def _rational(self, h: FloatArray, a: float, c: float) -> FloatArray:
        alpha = float(self.params.get("alpha", 1.0))
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(h / a)
        return as_float_array(c * (1.0 - (1.0 + hr**2 / alpha) ** (-alpha)))

    def _cubic(self, h: FloatArray, a: float, c: float) -> FloatArray:
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(np.minimum(h / a, 1.0))
        return as_float_array(c * (7.0 * hr**2 - 8.75 * hr**3 + 3.5 * hr**5 - 0.75 * hr**7))

    def _linear(self, h: FloatArray, a: float, c: float) -> FloatArray:
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(np.minimum(h / a, 1.0))
        return as_float_array(c * hr)

    def _circular(self, h: FloatArray, a: float, c: float) -> FloatArray:
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(h / a)
        result = as_float_array(np.full_like(h, c, dtype=np.float64))
        inside = as_bool_array(hr < 1.0)
        if np.any(inside):
            hr_in = as_float_array(hr[inside])
            term = as_float_array(np.arccos(hr_in) - hr_in * np.sqrt(1.0 - hr_in**2))
            result[inside] = c * (1.0 - (2.0 / np.pi) * term)
        result[h == 0] = 0.0
        return result

    def _jbessel(self, h: FloatArray, a: float, c: float) -> FloatArray:
        nu = float(self.params.get("nu", 0.5))
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(h / a)
        result = as_float_array(np.zeros_like(h, dtype=np.float64))
        nonzero = as_bool_array(hr > 0)
        if not np.any(nonzero):
            return result
        s = as_float_array(hr[nonzero])
        coef = float(special.gamma(nu + 1.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = as_float_array(special.jv(nu, s) / (s / 2.0) ** nu)
        corr = as_float_array(np.clip(coef * ratio, -1.0, 1.0))
        result[nonzero] = c * (1.0 - corr)
        return result

    def _superspherical(self, h: FloatArray, a: float, c: float) -> FloatArray:
        nu = float(self.params.get("nu", 1.0))
        dim = float(self.params.get("dim", 3))
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(h / a)
        result = as_float_array(np.full_like(h, c, dtype=np.float64))
        inside = as_bool_array(hr < 1.0)
        if np.any(inside):
            hr_in = as_float_array(hr[inside])
            hyp = as_float_array(special.hyp2f1(-nu, 0.5, dim / 2.0, hr_in**2))
            result[inside] = c * (1.0 - hyp)
        result[h == 0] = 0.0
        return result

    def _hyperspherical(self, h: FloatArray, a: float, c: float) -> FloatArray:
        dim = float(self.params.get("dim", 3))
        if a <= 0:
            return self._step_to_c(h, c)
        hr = as_float_array(h / a)
        result = as_float_array(np.full_like(h, c, dtype=np.float64))
        inside = as_bool_array(hr < 1.0)
        if np.any(inside):
            hr_in = as_float_array(hr[inside])
            hyp = as_float_array(special.hyp2f1((dim - 1.0) / 2.0, -0.5, dim / 2.0, hr_in**2))
            result[inside] = c * (1.0 - hyp)
        result[h == 0] = 0.0
        return result

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        extra = ""
        if self.params:
            extra = ", " + ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"VariogramKernel({self.name}, C={self.contribution}, range={self.ranges[0]}{extra})"


__all__ = ["VariogramKernel"]
