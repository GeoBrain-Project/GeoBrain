"""
Cross-validation summary statistics.

Computes a cross-validation report, extended with the **Mean Squared Deviation Ratio** (MSDR), a
standard variogram-calibration check: a well-calibrated model has
``MSDR = mean(eᵢ² / σᵢ²) ≈ 1``. Under-shoot (MSDR < 1) means the
model over-estimates uncertainty; over-shoot (MSDR > 1) means the
model is over-confident.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np

from ...frames._arrays import FloatArray, as_float_array

__all__ = ["cross_validation_report", "msdr"]


def msdr(error: FloatArray, variance: FloatArray) -> float:
    """
    Mean Squared Deviation Ratio: ``mean(e² / σ²)``.

    A well-calibrated variogram model has MSDR ≈ 1.

    Args:
        error: estimation errors.
        variance: estimation variances.
    """
    e = as_float_array(error)
    v = as_float_array(np.maximum(variance, 1.0e-20))
    return float(np.mean(e * e / v))


def cross_validation_report(
    actual: FloatArray,
    estimate: FloatArray,
    variance: FloatArray,
) -> dict[str, float]:
    """
    Compute summary diagnostics from a cross-validation run.

    Returns a dict with:

    - ``n``: number of points scored
    - ``mean_error``, ``mae``, ``rmse``: error magnitude
    - ``correlation``: Pearson(actual, estimate)
    - ``mean_standardized_error``: mean of ``eᵢ/σᵢ``
    - ``variance_of_standardized_error``: variance of ``eᵢ/σᵢ``
    - ``msdr``: :func:`msdr`

    Args:
        actual / estimate: held-out truths and estimates.
        variance: estimation variances.
    """
    a = as_float_array(actual)
    e_hat = as_float_array(estimate)
    var = as_float_array(variance)
    error = as_float_array(a - e_hat)
    n = int(a.size)

    me = float(np.mean(error)) if n > 0 else 0.0
    mae = float(np.mean(np.abs(error))) if n > 0 else 0.0
    rmse = float(np.sqrt(np.mean(error * error))) if n > 0 else 0.0

    if n > 1 and float(np.std(a)) > 0.0 and float(np.std(e_hat)) > 0.0:
        corr = float(np.corrcoef(a, e_hat)[0, 1])
    else:
        corr = 0.0

    std_var = np.sqrt(np.maximum(var, 1.0e-20))
    std_err = error / std_var
    mse_std = float(np.mean(std_err)) if n > 0 else 0.0
    vse_std = float(np.var(std_err, ddof=1)) if n > 1 else 0.0

    return {
        "n": n,
        "mean_error": me,
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "mean_standardized_error": mse_std,
        "variance_of_standardized_error": vse_std,
        "msdr": msdr(error, var),
    }
