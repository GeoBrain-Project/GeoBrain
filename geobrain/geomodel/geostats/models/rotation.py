"""
Anisotropy rotation matrices (numpy port of GSLIB ``setrot``).

Provides:

- :func:`setup_rotation_matrix` (GSLIB ``setrot``): converts the
  three GSLIB angles (azimuth, dip, plunge) and the two anisotropy
  ratios (``anis1=range_mid/range_max``, ``anis2=range_min/range_max``)
  into a 3×3 matrix that, applied to a coordinate difference vector
  ``(dx, dy, dz)``, yields the *reduced* coordinates whose Euclidean
  norm is the GSLIB anisotropic distance.
- :func:`anisotropic_distance`: convenience scalar form.

Angle convention (GSLIB / Deutsch & Journel 1998):
``azimuth`` is measured clockwise from North (``+y``) in degrees,
``dip`` is positive downwards, ``plunge`` is rotation about the
major axis. The 2-D special case is ``ang2 = ang3 = 0``,
``anis2 = 1``.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import numpy as np

from ....core import GeoBrainError
from ...frames._arrays import FloatArray, as_float_array

_DEG2RAD = np.pi / 180.0
_DEKKER_SPLITTER = 134_217_729.0
_FloatRep = tuple[float, int]


def _normalise_rep(mantissa: float, exponent: int) -> _FloatRep:
    """Normalize a signed binary floating representation without materializing it."""
    if mantissa == 0.0:
        return 0.0, 0
    normalized, shift = math.frexp(mantissa)
    return normalized, exponent + shift


def _negate_rep(value: _FloatRep) -> _FloatRep:
    return -value[0], value[1]


def _product_rep(*values: _FloatRep) -> _FloatRep:
    mantissa = 1.0
    exponent = 0
    for value_mantissa, value_exponent in values:
        if value_mantissa == 0.0:
            return 0.0, 0
        mantissa *= value_mantissa
        exponent += value_exponent
        mantissa, exponent = _normalise_rep(mantissa, exponent)
    return mantissa, exponent


def _rep_to_float(value: _FloatRep) -> float:
    try:
        return math.ldexp(value[0], value[1])
    except OverflowError:
        return math.copysign(math.inf, value[0])


def _divide_rep(value: _FloatRep, divisor: float) -> _FloatRep:
    """Divide a binary representation by a positive finite scalar."""
    return _normalise_rep(value[0] / divisor, value[1])


def _divide_reps(value: _FloatRep, divisor: _FloatRep) -> _FloatRep:
    """Divide two non-materialized binary representations."""
    return _normalise_rep(value[0] / divisor[0], value[1] - divisor[1])


def _split_scalar(value: float) -> tuple[float, float]:
    """Dekker-split a normalized scalar into non-overlapping words."""
    intermediate = _DEKKER_SPLITTER * value
    high = intermediate - (intermediate - value)
    return high, value - high


def _two_product_scalar(left: float, right: float) -> tuple[float, float]:
    """Return a normalized product and its exact float64 residual."""
    product = left * right
    left_high, left_low = _split_scalar(left)
    right_high, right_low = _split_scalar(right)
    error = (
        (left_high * right_high - product)
        + left_high * right_low
        + left_low * right_high
        + left_low * right_low
    )
    return product, error


def _factor_product_expansion(factors: tuple[_FloatRep, ...]) -> tuple[list[float], int]:
    """Return an exact short expansion and exponent for a product of factors."""
    expansion = [1.0]
    exponent = 0
    for mantissa, factor_exponent in factors:
        multiplied: list[float] = []
        for component in expansion:
            product, error = _two_product_scalar(component, mantissa)
            multiplied.extend((product, error))
        expansion = multiplied
        exponent += factor_exponent
    return expansion, exponent


def _sum_factor_products_rep(
    terms: tuple[tuple[_FloatRep, ...], ...],
) -> _FloatRep:
    """Return a compensated short sum of binary products without materializing it.

    Rotation construction has at most three factors per term.  Expanding every
    normalized product into exact non-overlapping float64 words prevents a lost
    multiplication residual or a subnormal combined angle from disappearing
    before an extreme anisotropy scale is applied.
    """
    expansions: list[tuple[list[float], int]] = []
    for factors in terms:
        expansion, exponent = _factor_product_expansion(factors)
        if any(component != 0.0 for component in expansion):
            expansions.append((expansion, exponent))
    if not expansions:
        return 0.0, 0
    maximum_exponent = max(exponent for _, exponent in expansions)
    scaled = math.fsum(
        math.ldexp(component, exponent - maximum_exponent)
        for expansion, exponent in expansions
        for component in expansion
    )
    if scaled == 0.0:
        return 0.0, 0
    scaled_mantissa, scaled_exponent = math.frexp(scaled)
    return _normalise_rep(scaled_mantissa, maximum_exponent + scaled_exponent)


def _sum_factor_products_divided(
    terms: tuple[tuple[_FloatRep, ...], ...],
    divisor: float,
) -> float:
    """Round a compensated short sum of products after scaled division."""
    total_mantissa, total_exponent = _sum_factor_products_rep(terms)
    if total_mantissa == 0.0:
        return 0.0
    divisor_mantissa, divisor_exponent = math.frexp(divisor)
    return _rep_to_float(
        _normalise_rep(
            total_mantissa / divisor_mantissa,
            total_exponent - divisor_exponent,
        )
    )


def _hypot_reps(left: _FloatRep, right: _FloatRep) -> _FloatRep:
    """Return a scale-safe hypotenuse without materializing either operand."""
    nonzero = tuple(value for value in (left, right) if value[0] != 0.0)
    if not nonzero:
        return 0.0, 0
    maximum_exponent = max(value[1] for value in nonzero)
    scaled_left = math.ldexp(left[0], left[1] - maximum_exponent)
    scaled_right = math.ldexp(right[0], right[1] - maximum_exponent)
    norm_mantissa, norm_exponent = math.frexp(math.hypot(scaled_left, scaled_right))
    return _normalise_rep(norm_mantissa, maximum_exponent + norm_exponent)


def _two_sum_scalar(left: float, right: float) -> tuple[float, float]:
    """Return a rounded sum and its exact float64 residual."""
    total = left + right
    right_virtual = total - left
    error = (left - (total - right_virtual)) + (right - right_virtual)
    return total, error


def _combined_degree_expansion(
    left: float,
    right: float,
    *,
    subtract: bool,
) -> tuple[float, float, int]:
    """Reduce an exact two-float angle sum/difference to one quadrant.

    Reducing each original binary64 degree value modulo a full turn first
    prevents overflow for otherwise valid finite angles.  ``_two_sum_scalar``
    then retains the exact low word that an ordinary ``left +/- right`` would
    round away.  The returned high/low pair is an error-free representation of
    the residual angle, and ``quadrant`` records the removed multiple of 90°.
    """
    left_reduced = math.remainder(left, 360.0)
    right_reduced = math.remainder(right, 360.0)
    if subtract:
        right_reduced = -right_reduced
    combined_high, combined_low = _two_sum_scalar(left_reduced, right_reduced)
    quadrant = int(round(combined_high / 90.0))

    def residual_for(candidate: int) -> tuple[float, float]:
        residual = combined_high - 90.0 * candidate
        return _two_sum_scalar(residual, combined_low)

    residual_high, residual_low = residual_for(quadrant)
    if residual_high > 45.0 or (residual_high == 45.0 and residual_low > 0.0):
        quadrant += 1
        residual_high, residual_low = residual_for(quadrant)
    elif residual_high < -45.0 or (residual_high == -45.0 and residual_low < 0.0):
        quadrant -= 1
        residual_high, residual_low = residual_for(quadrant)
    return residual_high, residual_low, quadrant


def _residual_sincos_reps(
    degree_high: float,
    degree_low: float,
) -> tuple[_FloatRep, _FloatRep]:
    """Evaluate sine/cosine once from an exact two-word degree residual."""
    if degree_high == 0.0 and degree_low == 0.0:
        return (0.0, 0), (0.5, 1)

    if abs(degree_high) == 45.0:
        diagonal = math.frexp(math.sqrt(0.5))
        sine_high = diagonal if degree_high > 0.0 else _negate_rep(diagonal)
        cosine_high = diagonal
    else:
        degree_rep = math.frexp(degree_high)
        radians_high = _product_rep(degree_rep, math.frexp(float(_DEG2RAD)))
        radians = _rep_to_float(radians_high)
        sinc = 1.0 if radians == 0.0 else math.sin(radians) / radians
        sine_high = _product_rep(radians_high, math.frexp(sinc))
        cosine_high = math.frexp(math.cos(radians))

    if degree_low == 0.0:
        return sine_high, cosine_high

    radians_low = _product_rep(
        math.frexp(degree_low),
        math.frexp(float(_DEG2RAD)),
    )
    # The low word is at most half an ulp of the high word.  Its quadratic
    # correction is therefore below the rounded high-word sine/cosine; retain
    # the first-order term as a binary representation so extreme anisotropy
    # can scale it before float64 materialization.
    sine = _sum_factor_products_rep(
        (
            (sine_high,),
            (cosine_high, radians_low),
        )
    )
    cosine = _sum_factor_products_rep(
        (
            (cosine_high,),
            (_negate_rep(sine_high), radians_low),
        )
    )
    return sine, cosine


def _combined_sincos_reps(
    left_degrees: float,
    right_degrees: float,
    *,
    subtract: bool,
) -> tuple[_FloatRep, _FloatRep]:
    """Return sine/cosine after combining the original degree floats exactly."""
    degree_high, degree_low, quadrant = _combined_degree_expansion(
        left_degrees,
        right_degrees,
        subtract=subtract,
    )
    sine, cosine = _residual_sincos_reps(degree_high, degree_low)
    position = quadrant % 4
    if position == 1:
        sine, cosine = cosine, _negate_rep(sine)
    elif position == 2:
        sine, cosine = _negate_rep(sine), _negate_rep(cosine)
    elif position == 3:
        sine, cosine = _negate_rep(cosine), sine
    norm = _hypot_reps(sine, cosine)
    if norm[0] == 0.0:
        raise GeoBrainError(
            "combined rotation angles do not admit a finite unit direction",
            object_name="setup_rotation_matrix",
            field="ang1/ang3",
            expected="finite exactly combined angles",
            actual=(_rep_to_float(sine), _rep_to_float(cosine)),
        )
    return _divide_reps(sine, norm), _divide_reps(cosine, norm)


def _sincos_degree_reps(angle: float) -> tuple[_FloatRep, _FloatRep]:
    """Return sine/cosine binary representations after exact quadrant reduction."""
    if not math.isfinite(angle):
        raise GeoBrainError(
            "rotation angles must be finite",
            object_name="setup_rotation_matrix",
            field="ang1/ang2/ang3",
            expected="finite angles in degrees",
            actual=angle,
        )
    # IEEE remainder retains the sign and magnitude of a subnormal displacement
    # from zero, unlike ``angle % 360`` which can round ``-min_subnormal`` to 360.
    reduced = math.remainder(angle, 360.0)
    quadrant = int(round(reduced / 90.0))
    residual_degrees = reduced - 90.0 * quadrant

    if residual_degrees == 0.0:
        sine_residual: _FloatRep = (0.0, 0)
        cosine_residual: _FloatRep = (0.5, 1)
    elif abs(residual_degrees) == 45.0:
        # The two exact magnitudes coincide on an octant diagonal.  A typical
        # libm rounds sin(pi/4) and cos(pi/4) to adjacent floats; retaining that
        # artificial mismatch before a 1e300 row scale creates a false 1e284
        # matrix coefficient.  Use their shared correctly rounded magnitude.
        diagonal = math.frexp(math.sqrt(0.5))
        sine_residual = diagonal if residual_degrees > 0.0 else _negate_rep(diagonal)
        cosine_residual = diagonal
    else:
        degree_mantissa, degree_exponent = math.frexp(residual_degrees)
        radian_mantissa, radian_exponent = math.frexp(float(_DEG2RAD))
        radians_rep = _normalise_rep(
            degree_mantissa * radian_mantissa,
            degree_exponent + radian_exponent,
        )
        radians = _rep_to_float(radians_rep)
        sinc = 1.0 if radians == 0.0 else math.sin(radians) / radians
        sine_residual = _product_rep(radians_rep, math.frexp(sinc))
        cosine_residual = math.frexp(math.cos(radians))

    position = quadrant % 4
    if position == 0:
        return sine_residual, cosine_residual
    if position == 1:
        return cosine_residual, _negate_rep(sine_residual)
    if position == 2:
        return _negate_rep(sine_residual), _negate_rep(cosine_residual)
    return _negate_rep(cosine_residual), sine_residual


def _sincos_degrees(angle: float) -> tuple[float, float]:
    """Return sine/cosine, exact only for exactly cardinal degree inputs."""
    sine, cosine = _sincos_degree_reps(angle)
    return _rep_to_float(sine), _rep_to_float(cosine)


def setup_rotation_matrix(
    ang1: float = 0.0,
    ang2: float = 0.0,
    ang3: float = 0.0,
    anis1: float = 1.0,
    anis2: float = 1.0,
) -> FloatArray:
    """
    3×3 rotation+scaling matrix in GSLIB ``setrot`` convention.

    The returned matrix ``R`` is such that for any coordinate
    difference ``Δ = (dx, dy, dz)``, the anisotropic squared
    distance is ``‖R · Δ‖²`` and the anisotropic distance is
    ``√‖R · Δ‖²``.

    Args:
        ang1: azimuth (deg, 0–360, clockwise from North).
        ang2: dip (deg, −90 to 90).
        ang3: plunge (deg, −90 to 90).
        anis1: range_mid / range_max (must be > 0).
        anis2: range_min / range_max (must be > 0).

    Raises:
        GeoBrainError: if ``anis1 <= 0`` or ``anis2 <= 0``.
    """
    if not np.isfinite((anis1, anis2)).all() or anis1 <= 0 or anis2 <= 0:
        raise GeoBrainError(
            "anisotropy ratios must be positive",
            object_name="setup_rotation_matrix",
            field="anis1/anis2",
            expected="> 0",
            actual=(anis1, anis2),
        )

    # GSLIB alpha is 90° - ang1 modulo a full turn.  Evaluating the
    # equivalent co-functions directly avoids losing a real near-cardinal
    # ``ang1`` in the subtraction from 90 degrees.
    sin_ang1, cos_ang1 = _sincos_degree_reps(ang1)
    sin_ang2, cos_ang2 = _sincos_degree_reps(ang2)
    sint, cost = _sincos_degree_reps(ang3)
    sina, cosa = cos_ang1, sin_ang1
    sinb, cosb = _negate_rep(sin_ang2), cos_ang2

    rot: FloatArray = as_float_array(np.zeros((3, 3), dtype=np.float64))
    if abs(_rep_to_float(sinb)) == 1.0:
        # At, or within float64 rounding distance of, a vertical dip, the
        # general formulas reduce to one angle sum/difference plus a stable
        # half-angle correction.  ``sinb`` itself may already equal +/-1 while
        # nonzero ``cosb`` still resolves 1 - abs(sinb) at second order:
        #
        #   q = 1 - abs(sinb) = cosb**2 / (1 + sqrt(1 - cosb**2)).
        #
        # Keeping q as a binary representation prevents its underflow before
        # an extreme reciprocal anisotropy scale is applied.
        cosb_float = _rep_to_float(cosb)
        denominator = 1.0 + math.sqrt(max(0.0, 1.0 - cosb_float * cosb_float))
        q = _divide_rep(_product_rep(cosb, cosb), denominator)
        rot[0, 0] = _sum_factor_products_divided(((cosb, cosa),), 1.0)
        rot[0, 1] = _sum_factor_products_divided(((cosb, sina),), 1.0)
        rot[0, 2] = -_rep_to_float(sinb)
        rot[1, 2] = _sum_factor_products_divided(((sint, cosb),), anis1)
        rot[2, 2] = _sum_factor_products_divided(((cost, cosb),), anis2)
        if sinb[0] > 0.0:
            sine, cosine = _combined_sincos_reps(
                ang1,
                ang3,
                subtract=False,
            )
            rot[1, 0] = _sum_factor_products_divided(
                (
                    (_negate_rep(cosine),),
                    (_negate_rep(sint), q, cosa),
                ),
                anis1,
            )
            rot[1, 1] = _sum_factor_products_divided(
                (
                    (sine,),
                    (_negate_rep(sint), q, sina),
                ),
                anis1,
            )
            rot[2, 0] = _sum_factor_products_divided(
                (
                    (sine,),
                    (_negate_rep(cost), q, cosa),
                ),
                anis2,
            )
            rot[2, 1] = _sum_factor_products_divided(
                (
                    (cosine,),
                    (_negate_rep(cost), q, sina),
                ),
                anis2,
            )
        else:
            sine, cosine = _combined_sincos_reps(
                ang1,
                ang3,
                subtract=True,
            )
            rot[1, 0] = _sum_factor_products_divided(
                (
                    (_negate_rep(cosine),),
                    (sint, q, cosa),
                ),
                anis1,
            )
            rot[1, 1] = _sum_factor_products_divided(
                (
                    (sine,),
                    (sint, q, sina),
                ),
                anis1,
            )
            rot[2, 0] = _sum_factor_products_divided(
                (
                    (_negate_rep(sine),),
                    (cost, q, cosa),
                ),
                anis2,
            )
            rot[2, 1] = _sum_factor_products_divided(
                (
                    (_negate_rep(cosine),),
                    (cost, q, sina),
                ),
                anis2,
            )
        if not np.isfinite(rot).all():
            raise GeoBrainError(
                "anisotropy ratios do not admit a finite float64 rotation matrix",
                object_name="setup_rotation_matrix",
                field="anis1/anis2",
                expected="positive ratios with representable reciprocal rotation coefficients",
                actual=(anis1, anis2),
            )
        return as_float_array(rot)

    rot[0, 0] = _sum_factor_products_divided(((cosb, cosa),), 1.0)
    rot[0, 1] = _sum_factor_products_divided(((cosb, sina),), 1.0)
    rot[0, 2] = _sum_factor_products_divided(((_negate_rep(sinb),),), 1.0)
    rot[1, 0] = _sum_factor_products_divided(
        (
            (_negate_rep(cost), sina),
            (sint, sinb, cosa),
        ),
        anis1,
    )
    rot[1, 1] = _sum_factor_products_divided(
        (
            (cost, cosa),
            (sint, sinb, sina),
        ),
        anis1,
    )
    rot[1, 2] = _sum_factor_products_divided(((sint, cosb),), anis1)
    rot[2, 0] = _sum_factor_products_divided(
        (
            (sint, sina),
            (cost, sinb, cosa),
        ),
        anis2,
    )
    rot[2, 1] = _sum_factor_products_divided(
        (
            (_negate_rep(sint), cosa),
            (cost, sinb, sina),
        ),
        anis2,
    )
    rot[2, 2] = _sum_factor_products_divided(((cost, cosb),), anis2)
    if not np.isfinite(rot).all():
        raise GeoBrainError(
            "anisotropy ratios do not admit a finite float64 rotation matrix",
            object_name="setup_rotation_matrix",
            field="anis1/anis2",
            expected="positive ratios with representable reciprocal rotation coefficients",
            actual=(anis1, anis2),
        )
    return as_float_array(rot)


def _split(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Dekker-split normalized float64 values into high and low words."""
    intermediate = as_float_array(_DEKKER_SPLITTER * values)
    high = as_float_array(intermediate - (intermediate - values))
    return high, as_float_array(values - high)


def _two_product(left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return a product and its exact float64 rounding residual."""
    product = as_float_array(left * right)
    left_high, left_low = _split(left)
    right_high, right_low = _split(right)
    error = as_float_array(
        ((left_high * right_high - product) + left_high * right_low + left_low * right_high)
        + left_low * right_low
    )
    return product, error


def _two_sum(left: FloatArray, right: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return a sum and its exact float64 rounding residual."""
    total = as_float_array(left + right)
    right_virtual = as_float_array(total - left)
    error = as_float_array((left - (total - right_virtual)) + (right - right_virtual))
    return total, error


def _compensated_projection(values: FloatArray, direction: FloatArray) -> FloatArray:
    """Project normalized rows without thresholding any nonzero component."""
    high = as_float_array(np.zeros(values.shape[0], dtype=np.float64))
    low = as_float_array(np.zeros(values.shape[0], dtype=np.float64))
    for axis in range(3):
        product, product_error = _two_product(values[:, axis], direction[axis])
        high, sum_error = _two_sum(high, product)
        low = as_float_array(low + product_error + sum_error)
    high, sum_error = _two_sum(high, low)
    return as_float_array(high + sum_error)


def _scaled_product(
    normalized_projection: FloatArray,
    coordinate_scale: FloatArray,
    row_scale: float,
) -> FloatArray:
    """Multiply three float64 factors without premature overflow or underflow."""
    projection_mantissa, projection_exponent = np.frexp(normalized_projection)
    coordinate_mantissa, coordinate_exponent = np.frexp(coordinate_scale)
    row_mantissa, row_exponent = np.frexp(row_scale)
    mantissa = projection_mantissa * coordinate_mantissa * row_mantissa
    exponent = projection_exponent + coordinate_exponent + row_exponent
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return as_float_array(np.ldexp(mantissa, exponent))


def _power_of_two_scales(magnitudes: FloatArray) -> FloatArray:
    """Return the largest finite power of two not exceeding each magnitude."""
    scales = as_float_array(np.ones_like(magnitudes, dtype=np.float64))
    nonzero = magnitudes > 0.0
    if np.any(nonzero):
        _, exponent = np.frexp(magnitudes[nonzero])
        scales[nonzero] = np.ldexp(np.ones(exponent.shape, dtype=np.float64), exponent - 1)
    return scales


def _anisotropic_norm(delta: FloatArray, rotmat: FloatArray) -> FloatArray:
    """Return vectorized reduced-coordinate norms with scale-aware projections."""
    arr = as_float_array(delta)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise GeoBrainError(
            "delta must have shape (n, 2) or (n, 3)",
            object_name="anisotropic_squared_distance",
            field="delta",
            expected="(n, 2) or (n, 3)",
            actual=tuple(arr.shape),
        )
    if arr.shape[1] == 2:
        n = arr.shape[0]
        arr = as_float_array(np.column_stack([arr, np.zeros(n, dtype=np.float64)]))
    rot = as_float_array(rotmat)
    if rot.shape != (3, 3):
        raise GeoBrainError(
            "rotmat must be 3x3",
            object_name="anisotropic_squared_distance",
            field="rotmat",
            expected="(3, 3)",
            actual=tuple(rot.shape),
        )
    if np.isnan(arr).any():
        raise GeoBrainError(
            "delta must not contain NaN",
            object_name="anisotropic_squared_distance",
            field="delta",
            expected="finite values or an overflowed infinite lag",
            actual="contains NaN",
        )
    finite = np.isfinite(arr).all(axis=1)
    output = as_float_array(np.full(arr.shape[0], np.inf, dtype=np.float64))
    if not np.any(finite):
        return output
    finite_arr = as_float_array(arr[finite])
    coordinate_magnitude = as_float_array(np.max(np.abs(finite_arr), axis=1))
    coordinate_scale = _power_of_two_scales(coordinate_magnitude)
    normalized_delta = as_float_array(finite_arr / coordinate_scale[:, None])

    reduced_components = as_float_array(np.empty_like(finite_arr, dtype=np.float64))
    for axis in range(3):
        row_magnitude = float(np.max(np.abs(rot[axis])))
        if not np.isfinite(row_magnitude) or row_magnitude <= 0.0:
            raise GeoBrainError(
                "rotmat rows must have positive finite norms",
                object_name="anisotropic_squared_distance",
                field="rotmat",
                expected="three positive finite row norms",
                actual=row_magnitude,
            )
        row_scale = float(_power_of_two_scales(as_float_array([row_magnitude]))[0])
        direction = as_float_array(rot[axis] / row_scale)
        projection = _compensated_projection(normalized_delta, direction)
        reduced_components[:, axis] = _scaled_product(
            projection,
            coordinate_scale,
            row_scale,
        )
    output[finite] = np.hypot.reduce(reduced_components, axis=1)
    return output


def anisotropic_squared_distance(delta: FloatArray, rotmat: FloatArray) -> FloatArray:
    """
    Vectorised anisotropic squared distance.

    Args:
        delta: ``(n, 3)`` coordinate differences (or ``(n, 2)``, Δz=0
            is inferred).
        rotmat: 3×3 rotation+scaling matrix from
            :func:`setup_rotation_matrix`.

    Returns:
        ``(n,)`` array of squared anisotropic distances. Values whose true
        square exceeds float64 are represented by positive infinity.
    """
    distance = _anisotropic_norm(delta, rotmat)
    with np.errstate(over="ignore"):
        return as_float_array(distance * distance)


def anisotropic_distance(p1: FloatArray, p2: FloatArray, rotmat: FloatArray) -> float | FloatArray:
    """
    Scalar or vectorised anisotropic distance between two points / point sets.

    Args:
        p1, p2: point coordinates. May be ``(3,)`` / ``(2,)`` arrays
            (scalar form) or matching ``(n, 3)`` / ``(n, 2)`` arrays.
        rotmat: 3×3 rotation+scaling matrix.

    Returns:
        Scalar float for scalar inputs; ``(n,)`` array otherwise.
    """
    a = as_float_array(p1)
    b = as_float_array(p2)
    if a.shape != b.shape:
        raise GeoBrainError(
            "p1 and p2 must have the same shape",
            object_name="anisotropic_distance",
            field="p1/p2",
            expected=f"shape == {a.shape}",
            actual=b.shape,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        delta = as_float_array(b - a)
    if a.ndim == 1:
        return float(_anisotropic_norm(as_float_array(delta.reshape(1, -1)), rotmat)[0])
    return _anisotropic_norm(delta, rotmat)


__all__ = [
    "setup_rotation_matrix",
    "anisotropic_squared_distance",
    "anisotropic_distance",
]
