"""
Linear Model of Coregionalization (LMC).

The LMC parameterises an n-variable multivariate covariance model as
a sum of ``n_structures`` shared spatial kernels weighted by per-
structure coregionalization matrices ``B_s``:

::

    C_ij(h) = nugget_ij · 𝟙[h=0]
              + Σ_s B_s[i, j] · ρ_s(h)

where ``ρ_s`` is the *normalised* (correlogram-scaled) form of the
``s``-th structure. Each ``B_s`` must be positive-semi-definite for
the LMC to be a valid multivariate covariance.

``get_model(i, j)`` extracts a single :class:`CovarianceModel` for a
direct (``j == i``) or cross-variable (``i ≠ j``) pair, suitable for
plugging into kriging / cokriging.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np

from ....core import GeoBrainError
from ...errors import GeomodelNumericsError
from ...frames._arrays import FloatArray, as_float_array
from .covariance import CovarianceModel
from .variogram_kernel import VariogramKernel

__all__ = ["LinearModelOfCoregionalization"]


class _SignedCrossCovarianceModel(CovarianceModel):  # type: ignore[misc]
    """Read-only signed LMC cross covariance that cannot drive kriging."""

    def __init__(
        self,
        nugget: float,
        components: list[tuple[VariogramKernel, float]],
    ) -> None:
        self.nugget = float(nugget)
        self.structures: list[VariogramKernel] = []
        self._signed_components = tuple(
            (
                VariogramKernel(
                    kernel.kind,
                    1.0,
                    kernel.ranges,
                    kernel.angles,
                    **kernel.params,
                ),
                float(coefficient),
            )
            for kernel, coefficient in components
        )

    @property
    def n_structures(self) -> int:
        return len(self._signed_components)

    @property
    def sill(self) -> float:
        return float(self.nugget + sum(value for _, value in self._signed_components))

    def variogram(self, h: object) -> FloatArray:
        h_arr = as_float_array(h)
        result = as_float_array(np.where(h_arr > 0.0, self.nugget, 0.0))
        for kernel, coefficient in self._signed_components:
            result = as_float_array(result + coefficient * kernel.evaluate(h_arr))
        return result

    def covariance(self, h: object) -> FloatArray:
        return as_float_array(self.sill - self.variogram(h))

    def correlogram(self, h: object) -> FloatArray:
        h_arr = as_float_array(h)
        if self.sill == 0.0:
            return as_float_array(np.ones_like(h_arr, dtype=np.float64))
        return as_float_array(self.covariance(h_arr) / self.sill)

    def require_stationary_covariance(self, *, object_name: str) -> None:
        raise GeomodelNumericsError(
            "signed LMC cross covariance cannot drive a univariate kriging system",
            object_name=object_name,
            field="variogram",
            expected="positive-semidefinite direct covariance model",
            actual="signed cross covariance",
        )


def _check_psd(matrix: FloatArray, *, object_name: str, field: str) -> None:
    """
    Raise :class:`GeoBrainError` unless ``matrix`` (assumed symmetric) is
    positive-semi-definite.

    A symmetric-but-indefinite coregionalization matrix violates the
    Cauchy-Schwarz bound on cross covariances and yields an invalid LMC,
    so we require the smallest eigenvalue to be non-negative within a
    scale-aware tolerance.
    """
    if not np.isfinite(matrix).all():
        raise GeomodelNumericsError(
            f"{field} must be finite",
            object_name=object_name,
            field=field,
            expected="finite symmetric positive-semidefinite matrix",
            actual="contains NaN or infinity",
        )
    diagonal = as_float_array(np.diag(matrix))
    if np.any(diagonal < 0.0):
        raise GeomodelNumericsError(
            f"{field} diagonal must be non-negative",
            object_name=object_name,
            field=field,
            expected="non-negative marginal variances",
            actual=diagonal.tolist(),
        )

    positive = diagonal > 0.0
    zero = ~positive
    if np.any(zero) and (np.any(matrix[zero, :] != 0.0) or np.any(matrix[:, zero] != 0.0)):
        raise GeomodelNumericsError(
            f"{field} zero-variance rows must have zero covariance",
            object_name=object_name,
            field=field,
            expected="zero row and column for every zero marginal variance",
            actual="nonzero covariance coupled to a zero variance",
        )
    if not np.any(positive):
        return

    positive_block = as_float_array(matrix[np.ix_(positive, positive)])
    standard_deviations = as_float_array(np.sqrt(diagonal[positive]))
    scale_products = as_float_array(np.multiply.outer(standard_deviations, standard_deviations))
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = as_float_array(positive_block / scale_products)
    if not np.isfinite(normalized).all():
        raise GeomodelNumericsError(
            f"{field} must be positive-semi-definite",
            object_name=object_name,
            field=field,
            expected="finite marginal correlation matrix",
            actual="correlation scaling overflowed",
        )
    transpose = as_float_array(normalized.T)
    normalized_bits = normalized.view(np.uint64)
    transpose_bits = transpose.view(np.uint64)
    sign_bit = np.uint64(1 << 63)
    normalized_keys = np.where(
        normalized_bits & sign_bit,
        ~normalized_bits,
        normalized_bits | sign_bit,
    )
    transpose_keys = np.where(
        transpose_bits & sign_bit,
        ~transpose_bits,
        transpose_bits | sign_bit,
    )
    lower_keys = np.minimum(normalized_keys, transpose_keys)
    upper_keys = np.maximum(normalized_keys, transpose_keys)
    if np.any(upper_keys - lower_keys > np.uint64(8)):
        with np.errstate(over="ignore", invalid="ignore"):
            difference = as_float_array(np.abs(normalized - transpose))
        raise GeomodelNumericsError(
            f"{field} must be symmetric",
            object_name=object_name,
            field=field,
            expected="pairwise asymmetry within 8 float64 ULPs",
            actual=float(np.max(difference)),
        )
    # Half-sum avoids overflowing when two equal finite entries are max_float.
    symmetric = as_float_array(0.5 * normalized + 0.5 * transpose)
    if not np.isfinite(symmetric).all():
        raise GeomodelNumericsError(
            f"{field} must be positive-semi-definite",
            object_name=object_name,
            field=field,
            expected="finite normalized symmetric matrix",
            actual="symmetrization produced NaN or infinity",
        )
    entry_scale = float(np.max(np.abs(symmetric)))
    eigensystem_matrix = (
        symmetric if entry_scale <= 1.0 else as_float_array(symmetric / entry_scale)
    )
    try:
        eigvals = np.linalg.eigvalsh(eigensystem_matrix)
    except np.linalg.LinAlgError as exc:
        raise GeomodelNumericsError(
            f"{field} eigenvalue validation failed",
            object_name=object_name,
            field=field,
            expected="finite positive-semidefinite matrix",
            actual="eigendecomposition failed",
        ) from exc
    if not np.isfinite(eigvals).all():
        raise GeomodelNumericsError(
            f"{field} eigenvalue validation failed",
            object_name=object_name,
            field=field,
            expected="finite eigenvalues for a normalized symmetric matrix",
            actual="eigendecomposition returned NaN or infinity",
        )
    eig_min = float(eigvals.min())
    dimension = eigensystem_matrix.shape[0]
    matrix_scale = max(1.0, float(np.linalg.norm(eigensystem_matrix, ord=np.inf)))
    # Symmetric eigensolvers are backward stable.  This bound is deliberately
    # tied to float64 roundoff, matrix order, and norm rather than a fixed
    # scientific tolerance that could conceal a resolved negative mode.
    tol = 32.0 * np.finfo(np.float64).eps * dimension * matrix_scale
    if eig_min < -tol:
        raise GeomodelNumericsError(
            f"{field} must be positive-semi-definite",
            object_name=object_name,
            field=field,
            expected="positive-semi-definite (min eigenvalue >= 0)",
            actual=f"min eigenvalue {eig_min:.6g}",
        )


class LinearModelOfCoregionalization:
    """
    Multivariate LMC with ``n_vars`` variables and ``n_structures`` shared kernels.

    Args:
        n_vars: number of variables.
        nugget_matrix: ``(n_vars, n_vars)`` PSD nugget covariance.
            Default: zeros.
    """

    def __init__(
        self,
        n_vars: int,
        nugget_matrix: FloatArray | None = None,
    ) -> None:
        if n_vars < 1:
            raise GeoBrainError(
                "n_vars must be >= 1",
                object_name="LinearModelOfCoregionalization",
                field="n_vars",
                expected=">= 1",
                actual=n_vars,
            )
        if nugget_matrix is None:
            nugget = as_float_array(np.zeros((n_vars, n_vars), dtype=np.float64))
        else:
            nugget = as_float_array(nugget_matrix)
            if nugget.shape != (n_vars, n_vars):
                raise GeoBrainError(
                    "nugget_matrix shape mismatch",
                    object_name="LinearModelOfCoregionalization",
                    field="nugget_matrix",
                    expected=f"({n_vars}, {n_vars})",
                    actual=nugget.shape,
                )
            _check_psd(
                nugget,
                object_name="LinearModelOfCoregionalization",
                field="nugget_matrix",
            )
        self.n_vars = int(n_vars)
        self.nugget_matrix: FloatArray = nugget
        self.structures: list[tuple[VariogramKernel, FloatArray]] = []

    # ------------------------------------------------------------------

    def add_structure(
        self,
        kernel: VariogramKernel,
        coregion_matrix: FloatArray,
    ) -> "LinearModelOfCoregionalization":
        """
        Add a shared kernel + its per-pair coregion matrix.

        Returns ``self`` so calls chain.
        """
        if not isinstance(kernel, VariogramKernel):
            raise GeoBrainError(
                "kernel must be a VariogramKernel",
                object_name="add_structure",
                field="kernel",
                expected="VariogramKernel",
                actual=type(kernel).__name__,
            )
        B = as_float_array(coregion_matrix)
        if B.shape != (self.n_vars, self.n_vars):
            raise GeoBrainError(
                "coregion_matrix shape mismatch",
                object_name="add_structure",
                field="coregion_matrix",
                expected=f"({self.n_vars}, {self.n_vars})",
                actual=B.shape,
            )
        _check_psd(B, object_name="add_structure", field="coregion_matrix")
        self.structures.append((kernel, B))
        return self

    @property
    def n_structures(self) -> int:
        return len(self.structures)

    # ------------------------------------------------------------------

    def cross_variogram(self, var_i: int, var_j: int, h: object) -> FloatArray:
        """``γᵢⱼ(h)`` between variables ``i`` and ``j`` (isotropic ``h``)."""
        self._check_indices(var_i, var_j)
        h_arr = as_float_array(h)
        result = as_float_array(np.where(h_arr > 0, self.nugget_matrix[var_i, var_j], 0.0))
        for kernel, B in self.structures:
            if kernel.contribution > 0:
                normalised = as_float_array(kernel.evaluate(h_arr) / kernel.contribution)
            else:
                normalised = as_float_array(np.where(h_arr > 0, 1.0, 0.0))
            result = as_float_array(result + float(B[var_i, var_j]) * normalised)
        return result

    def cross_covariance(self, var_i: int, var_j: int, h: object) -> FloatArray:
        """``Cᵢⱼ(h) = sillᵢⱼ − γᵢⱼ(h)``."""
        self._check_indices(var_i, var_j)
        sill_ij = float(self.nugget_matrix[var_i, var_j])
        for _, B in self.structures:
            sill_ij += float(B[var_i, var_j])
        return as_float_array(sill_ij - self.cross_variogram(var_i, var_j, h))

    def get_model(self, var_i: int, var_j: int | None = None) -> CovarianceModel:
        """
        Extract the direct (``j == i``) or cross :class:`CovarianceModel`.

        The cross model is valid for evaluation but cannot drive kriging
        directly; it is meant for diagnostics / inspection.
        """
        self._check_indices(var_i, var_i if var_j is None else var_j)
        j = var_i if var_j is None else var_j
        nugget = float(self.nugget_matrix[var_i, j])
        signed_components = [(kernel, float(B[var_i, j])) for kernel, B in self.structures]
        if nugget < 0.0 or any(value < 0.0 for _, value in signed_components):
            return _SignedCrossCovarianceModel(nugget, signed_components)
        structures: list[VariogramKernel] = []
        for kernel, contribution in signed_components:
            if contribution <= 0:
                continue
            structures.append(
                VariogramKernel(
                    kernel.kind,
                    contribution,
                    kernel.ranges,
                    kernel.angles,
                    **kernel.params,
                )
            )
        return CovarianceModel(max(nugget, 0.0), structures)

    # ------------------------------------------------------------------

    def _check_indices(self, var_i: int, var_j: int) -> None:
        for label, v in (("var_i", var_i), ("var_j", var_j)):
            if not 0 <= v < self.n_vars:
                raise GeoBrainError(
                    f"{label} out of range",
                    object_name="LinearModelOfCoregionalization",
                    field=label,
                    expected=f"0..{self.n_vars - 1}",
                    actual=v,
                )

    def __repr__(self) -> str:
        return (
            f"LinearModelOfCoregionalization("
            f"n_vars={self.n_vars}, n_structures={self.n_structures})"
        )
