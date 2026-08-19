"""
Polynomial drift basis for Universal Kriging.

Supported terms (case-insensitive, monomial):

- ``"x"``, ``"y"``, ``"z"``: linear
- ``"xy"``, ``"xz"``, ``"yz"``: bilinear
- ``"x2"``, ``"y2"``, ``"z2"``: quadratic

The constant term ``1`` is implicit in Ordinary / Universal Kriging
(the ``Σ wᵢ = 1`` unbiasedness row), so it is **not** listed here.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array

__all__ = ["drift_basis", "drift_difference_basis", "ALLOWED_DRIFT_TERMS"]

ALLOWED_DRIFT_TERMS = {
    "x",
    "y",
    "z",
    "xy",
    "xz",
    "yz",
    "x2",
    "y2",
    "z2",
}
_DEKKER_SPLITTER = 134_217_729.0
_PRODUCT_CHUNK_ROWS = 65_536


def _scaled_axis(
    values: FloatArray,
    target: float,
) -> tuple[FloatArray, float, FloatArray]:
    """Return dimensionless coordinates and centred differences without overflow."""
    magnitude = max(float(np.max(np.abs(values))), abs(target))
    if magnitude == 0.0:
        zeros = as_float_array(np.zeros_like(values, dtype=np.float64))
        return zeros, 0.0, zeros
    _, exponent = np.frexp(magnitude)
    scale = float(np.ldexp(0.5, exponent))
    normalized = as_float_array(values / scale)
    normalized_target = target / scale
    # Scaling by a binary power is exact.  Subtracting only afterwards avoids
    # both endpoint overflow and the loss of a target ULP caused by dividing an
    # already-rounded physical difference by an arbitrary column maximum.
    scaled_differences = as_float_array(normalized - normalized_target)
    return normalized, normalized_target, scaled_differences


def _unit_column(values: FloatArray) -> FloatArray:
    """Scale one finite drift constraint to unit maximum magnitude."""
    scale = float(np.max(np.abs(values)))
    return values if scale == 0.0 else as_float_array(values / scale)


def _two_sum(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return vectorized rounded sums and their exact residuals."""
    total = left + right
    right_virtual = total - left
    error = (left - (total - right_virtual)) + (right - right_virtual)
    return total, error


def _product_parts(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact two-word products of normalized mantissas and exponents."""
    left_mantissa, left_exponent = np.frexp(left)
    right_mantissa, right_exponent = np.frexp(right)
    product = left_mantissa * right_mantissa
    left_split = _DEKKER_SPLITTER * left_mantissa
    left_high = left_split - (left_split - left_mantissa)
    left_low = left_mantissa - left_high
    right_split = _DEKKER_SPLITTER * right_mantissa
    right_high = right_split - (right_split - right_mantissa)
    right_low = right_mantissa - right_high
    error = (
        (left_high * right_high - product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    return product, error, left_exponent + right_exponent


def _product_difference_representations(
    left: np.ndarray,
    right: np.ndarray,
    target_product: float,
    target_error: float,
    target_exponent: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leading binary representations for exact product differences."""
    product, product_error, product_exponent = _product_parts(left, right)
    common_exponent = np.maximum(product_exponent, target_exponent)
    product_shift = product_exponent - common_exponent
    target_shift = target_exponent - common_exponent
    product_high = np.ldexp(product, product_shift)
    product_low = np.ldexp(product_error, product_shift)
    target_high = np.ldexp(target_product, target_shift)
    target_low = np.ldexp(target_error, target_shift)

    difference_high, subtraction_error = _two_sum(product_high, -target_high)
    low_high, low_error = _two_sum(product_low, -target_low)
    middle, middle_error = _two_sum(subtraction_error, low_high)
    difference_high, carry = _two_sum(difference_high, middle)
    tail_high, tail_error = _two_sum(low_error, middle_error)
    tail_high, tail_carry = _two_sum(tail_high, carry)
    difference_high, final_error = _two_sum(difference_high, tail_high)
    difference = difference_high + (tail_error + tail_carry + final_error)
    mantissa, local_exponent = np.frexp(difference)
    return mantissa, common_exponent + local_exponent


def _normalised_product_difference(
    left: FloatArray,
    right: FloatArray,
    target_left: float,
    target_right: float,
) -> FloatArray:
    """Return a finite unit-scaled product-difference column over all float64 exponents."""
    target_product, target_error, target_exponent_array = _product_parts(
        np.asarray([target_left], dtype=np.float64),
        np.asarray([target_right], dtype=np.float64),
    )
    target_exponent = int(target_exponent_array[0])
    scale_mantissa = 0.0
    scale_exponent = 0
    for start in range(0, left.size, _PRODUCT_CHUNK_ROWS):
        stop = min(start + _PRODUCT_CHUNK_ROWS, left.size)
        mantissa, exponent = _product_difference_representations(
            left[start:stop],
            right[start:stop],
            float(target_product[0]),
            float(target_error[0]),
            target_exponent,
        )
        nonzero = mantissa != 0.0
        if not np.any(nonzero):
            continue
        chunk_exponent = int(np.max(exponent[nonzero]))
        chunk_mantissa = float(np.max(np.abs(mantissa[nonzero & (exponent == chunk_exponent)])))
        if (
            scale_mantissa == 0.0
            or chunk_exponent > scale_exponent
            or (chunk_exponent == scale_exponent and chunk_mantissa > scale_mantissa)
        ):
            scale_exponent = chunk_exponent
            scale_mantissa = chunk_mantissa
    if scale_mantissa == 0.0:
        return as_float_array(np.zeros_like(left, dtype=np.float64))

    output = np.zeros_like(left, dtype=np.float64)
    for start in range(0, left.size, _PRODUCT_CHUNK_ROWS):
        stop = min(start + _PRODUCT_CHUNK_ROWS, left.size)
        mantissa, exponent = _product_difference_representations(
            left[start:stop],
            right[start:stop],
            float(target_product[0]),
            float(target_error[0]),
            target_exponent,
        )
        output[start:stop] = np.ldexp(
            mantissa / scale_mantissa,
            exponent - scale_exponent,
        )
    return as_float_array(output)


def drift_basis(
    coords: FloatArray,
    drift_terms: list[str],
    *,
    target: FloatArray | None = None,
) -> FloatArray:
    """
    Evaluate the drift basis at ``coords``.

    Args:
        coords: dimension-preserving ``(n, 2)`` or ``(n, 3)`` array.
        drift_terms: list of term names; see module docstring.
        target: optional aligned target. When supplied, return finite,
            independently column-scaled ``f(data) - f(target)`` constraints,
            exactly matching Universal Kriging's numerical basis.

    Returns:
        ``(n, len(drift_terms))`` float64 array.
    """
    if target is not None:
        return drift_difference_basis(coords, target, drift_terms)
    arr = as_float_array(coords)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise GeoBrainError(
            "drift_basis: coords must be (n, 2) or (n, 3)",
            object_name="drift_basis",
            field="coords",
            expected="(n, 2) or (n, 3)",
            actual=arr.shape,
        )
    if not np.isfinite(arr).all():
        raise GeoBrainError(
            "drift coordinates must be finite",
            object_name="drift_basis",
            field="coords",
            expected="finite coordinates",
            actual="contains NaN or infinity",
        )
    n = arr.shape[0]
    nterms = len(drift_terms)
    F: FloatArray = as_float_array(np.zeros((n, nterms), dtype=np.float64))
    x = arr[:, 0]
    y = arr[:, 1]
    z = None if arr.shape[1] == 2 else arr[:, 2]
    for k, raw in enumerate(drift_terms):
        term = raw.lower()
        if term not in ALLOWED_DRIFT_TERMS:
            raise GeoBrainError(
                f"unknown drift term {raw!r}",
                object_name="drift_basis",
                field="drift_terms",
                expected=f"one of {sorted(ALLOWED_DRIFT_TERMS)}",
                actual=raw,
            )
        if z is None and "z" in term:
            raise GeoBrainError(
                f"drift term {raw!r} requires three-dimensional coordinates",
                object_name="drift_basis",
                field="drift_terms",
                expected="x/y-only terms for a 2-D domain",
                actual=raw,
            )
        if term == "x":
            F[:, k] = x
        elif term == "y":
            F[:, k] = y
        elif term == "z":
            assert z is not None
            F[:, k] = z
        elif term == "xy":
            with np.errstate(over="ignore", invalid="ignore"):
                F[:, k] = x * y
        elif term == "xz":
            assert z is not None
            with np.errstate(over="ignore", invalid="ignore"):
                F[:, k] = x * z
        elif term == "yz":
            assert z is not None
            with np.errstate(over="ignore", invalid="ignore"):
                F[:, k] = y * z
        elif term == "x2":
            with np.errstate(over="ignore", invalid="ignore"):
                F[:, k] = x * x
        elif term == "y2":
            with np.errstate(over="ignore", invalid="ignore"):
                F[:, k] = y * y
        elif term == "z2":
            assert z is not None
            with np.errstate(over="ignore", invalid="ignore"):
                F[:, k] = z * z
    if not np.isfinite(F).all():
        raise GeoBrainError(
            "unscaled polynomial drift basis overflowed; pass target for a finite centred basis",
            object_name="drift_basis",
            field="coords/drift_terms",
            expected="finite raw basis or an explicit target",
            actual="contains NaN or infinity",
        )
    return as_float_array(F)


def drift_difference_basis(
    coords: FloatArray,
    target: FloatArray,
    drift_terms: list[str],
) -> FloatArray:
    """Evaluate scaled ``f(data) - f(target)`` constraints without cancellation."""
    arr = as_float_array(coords)
    point = as_float_array(target)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3) or point.shape != (arr.shape[1],):
        raise GeoBrainError(
            "drift differences require aligned 2-D or 3-D coordinates",
            object_name="drift_difference_basis",
            field="coords/target",
            expected="(n, 2|3) data and one aligned target",
            actual={"coords": arr.shape, "target": point.shape},
        )
    if not np.isfinite(arr).all() or not np.isfinite(point).all():
        raise GeoBrainError(
            "drift coordinates and target must be finite",
            object_name="drift_difference_basis",
            field="coords/target",
            expected="finite coordinates and target",
            actual="contains NaN or infinity",
        )
    _, _, dx = _scaled_axis(arr[:, 0], float(point[0]))
    _, _, dy = _scaled_axis(arr[:, 1], float(point[1]))
    z_terms = None if arr.shape[1] == 2 else _scaled_axis(arr[:, 2], float(point[2]))
    raw_x = as_float_array(arr[:, 0])
    raw_y = as_float_array(arr[:, 1])
    raw_z = None if arr.shape[1] == 2 else as_float_array(arr[:, 2])
    basis: FloatArray = as_float_array(np.empty((arr.shape[0], len(drift_terms)), dtype=np.float64))

    for index, raw in enumerate(drift_terms):
        term = raw.lower()
        if term not in ALLOWED_DRIFT_TERMS:
            raise GeoBrainError(
                f"unknown drift term {raw!r}",
                object_name="drift_difference_basis",
                field="drift_terms",
                expected=f"one of {sorted(ALLOWED_DRIFT_TERMS)}",
                actual=raw,
            )
        if z_terms is None and "z" in term:
            raise GeoBrainError(
                f"drift term {raw!r} requires three-dimensional coordinates",
                object_name="drift_difference_basis",
                field="drift_terms",
                expected="x/y-only terms for a 2-D domain",
                actual=raw,
            )
        if term == "x":
            column = dx
        elif term == "y":
            column = dy
        elif term == "z":
            assert z_terms is not None
            _, _, dz = z_terms
            column = dz
        elif term == "xy":
            column = _normalised_product_difference(
                raw_x,
                raw_y,
                float(point[0]),
                float(point[1]),
            )
        elif term == "xz":
            assert raw_z is not None
            column = _normalised_product_difference(
                raw_x,
                raw_z,
                float(point[0]),
                float(point[2]),
            )
        elif term == "yz":
            assert raw_z is not None
            column = _normalised_product_difference(
                raw_y,
                raw_z,
                float(point[1]),
                float(point[2]),
            )
        elif term == "x2":
            column = _normalised_product_difference(
                raw_x,
                raw_x,
                float(point[0]),
                float(point[0]),
            )
        elif term == "y2":
            column = _normalised_product_difference(
                raw_y,
                raw_y,
                float(point[1]),
                float(point[1]),
            )
        else:
            assert raw_z is not None
            column = _normalised_product_difference(
                raw_z,
                raw_z,
                float(point[2]),
                float(point[2]),
            )
        basis[:, index] = _unit_column(as_float_array(column))
    return as_float_array(basis)


def _exact_monomial(point: np.ndarray, term: str) -> Fraction:
    """Evaluate one supported monomial exactly over supplied float64 coordinates."""
    axes = {"x": 0, "y": 1, "z": 2}
    if len(term) == 1:
        return Fraction.from_float(float(point[axes[term]]))
    left = Fraction.from_float(float(point[axes[term[0]]]))
    right_axis = term[0] if term[1] == "2" else term[1]
    right = Fraction.from_float(float(point[axes[right_axis]]))
    return left * right


def _kriging_drift_basis(
    coords: FloatArray,
    target: FloatArray,
    drift_terms: list[str],
) -> FloatArray:
    """Return ratio-faithful constraints for a small UK neighbourhood.

    The public basis remains vectorized for large exploratory arrays.  A kriging
    neighbourhood is intentionally bounded, so exact binary rational arithmetic
    can retain subtraction residuals that no independently rounded float column
    can encode.  Each returned column still differs from its physical monomial
    constraints by one nonzero scalar only.
    """
    basis = drift_difference_basis(coords, target, drift_terms)
    arr = as_float_array(coords)
    point = as_float_array(target)
    for column_index, raw in enumerate(drift_terms):
        term = raw.lower()
        target_value = _exact_monomial(point, term)
        differences = [_exact_monomial(row, term) - target_value for row in arr]
        scale = max((abs(value) for value in differences), default=Fraction(0))
        if scale != 0:
            basis[:, column_index] = [float(value / scale) for value in differences]
    return as_float_array(basis)
