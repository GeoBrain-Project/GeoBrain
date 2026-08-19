"""
Anisotropic covariance matrices for kriging.

The :class:`CovarianceModel` evaluates the *isotropic* semivariogram
γ(h) along the major-range direction; for kriging we need the
*anisotropic* covariance ``C(p, q) = sill − γ(‖p − q‖_anis)``, where
the rotated lag norm depends on each nested structure's
``(angles, anis1, anis2)``.

The two functions here compute that pairwise (matrix form) and
target-to-data (vector form):

- :func:`covariance_matrix`: ``C(coords_a, coords_b)``,
  shape ``(na, nb)``.
- :func:`covariance_vector`: ``C(coords_a, single_target)``,
  shape ``(na,)``.

Both add the nugget contribution only at exactly-coincident pairs so the
diagonal of the data-data block carries the full sill. Near-coordinate
collision policy is handled before covariance assembly and is deliberately
not duplicated here.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core.errors import GeoBrainError

import numpy as np

from ...frames._arrays import FloatArray, as_float_array
from ..models.covariance import CovarianceModel
from ..models.rotation import _anisotropic_norm, setup_rotation_matrix

__all__ = ["covariance_matrix", "covariance_vector"]


def _coordinates(coords: FloatArray) -> FloatArray:
    """Validate coordinates without changing their declared dimension."""
    arr = as_float_array(coords)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise GeoBrainError(f"coords must be (n, 2) or (n, 3); got {arr.shape}")
    return arr


def covariance_matrix(
    model: CovarianceModel,
    coords_a: FloatArray,
    coords_b: FloatArray,
) -> FloatArray:
    """
    Pairwise covariance matrix ``C(coords_a, coords_b)``.

    Each nested structure contributes ``Cᵢ − γᵢ(‖h‖_anisᵢ)`` (where
    ``Cᵢ`` is the structure's contribution and the rotated norm uses
    the structure's anisotropy / orientation). The nugget is added
    only at coincident pairs.

    Args:
        model: covariance model.
        coords_a / coords_b: coordinate tables the matrix spans.
    """
    a = _coordinates(coords_a)
    b = _coordinates(coords_b)
    if a.shape[1] != b.shape[1]:
        raise GeoBrainError(
            "covariance coordinates must have the same declared dimension",
            object_name="covariance_matrix",
            field="coords_a/coords_b",
            expected=f"both {a.shape[1]}-D",
            actual=f"{a.shape[1]}-D and {b.shape[1]}-D",
        )
    na, nb = a.shape[0], b.shape[0]

    deltas = a[:, None, :] - b[None, :, :]  # (na, nb, 3)
    is_zero = np.all(deltas == 0.0, axis=-1)

    result = as_float_array(np.zeros((na, nb), dtype=np.float64))
    for struct in model.structures:
        anisotropy = np.asarray((struct.anis1, struct.anis2), dtype=np.float64)
        if not np.isfinite(anisotropy).all() or np.any(anisotropy <= 0.0):
            raise GeoBrainError(
                "covariance anisotropy ratios must be positive and finite",
                object_name="covariance_matrix",
                field="variogram.ranges",
                expected="strictly positive finite anisotropy ratios",
                actual=anisotropy.tolist(),
            )
        R = setup_rotation_matrix(
            struct.angles[0],
            struct.angles[1],
            struct.angles[2],
            float(anisotropy[0]),
            float(anisotropy[1]),
        )
        flattened = as_float_array(deltas.reshape(na * nb, a.shape[1]))
        reduced_distance = as_float_array(_anisotropic_norm(flattened, R).reshape(na, nb))
        gamma = struct.evaluate(reduced_distance)
        result = as_float_array(result + struct.contribution - gamma)

    if model.nugget > 0:
        result[is_zero] += model.nugget
    return result


def covariance_vector(
    model: CovarianceModel,
    coords_a: FloatArray,
    target: FloatArray,
) -> FloatArray:
    """Convenience wrapper: covariance to a single 3-D target point.

    Args:
        model: covariance model.
        coords_a: data coordinates.
        target: single target coordinate.
    """
    t = as_float_array(target).reshape(1, -1)
    return as_float_array(covariance_matrix(model, coords_a, t).reshape(-1))
