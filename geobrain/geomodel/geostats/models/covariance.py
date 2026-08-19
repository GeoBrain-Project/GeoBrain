"""
Nested covariance / variogram model.

Implements the standard GSLIB convention of *nugget plus a sum of variogram
structures*.

Convenience constructors :meth:`CovarianceModel.spherical`,
:meth:`CovarianceModel.exponential`, :meth:`CovarianceModel.gaussian`
build a single-structure model from a sill / range pair.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np

from ....core import GeoBrainError
from ...errors import GeomodelNumericsError
from ...frames._arrays import FloatArray, as_float_array
from .variogram_kernel import VariogramKernel


class CovarianceModel:
    """
    ``γ(h) = nugget·𝟙[h>0] + Σᵢ γᵢ(h)`` over nested structures.

    Args:
        nugget: nugget effect ``C₀ >= 0``.
        structures: list of :class:`VariogramKernel`, at least one
            is required *unless* the caller intends a pure-nugget
            model (passing ``[]`` is allowed).
    """

    def __init__(
        self,
        nugget: float = 0.0,
        structures: list[VariogramKernel] | None = None,
    ) -> None:
        if nugget < 0:
            raise GeoBrainError(
                "nugget must be non-negative",
                object_name="CovarianceModel",
                field="nugget",
                expected=">= 0",
                actual=nugget,
            )
        structs = list(structures) if structures else []
        for i, s in enumerate(structs):
            if not isinstance(s, VariogramKernel):
                raise GeoBrainError(
                    "structures must contain VariogramKernel instances",
                    object_name="CovarianceModel",
                    field=f"structures[{i}]",
                    expected="VariogramKernel",
                    actual=type(s).__name__,
                )
        self.nugget = float(nugget)
        self.structures: list[VariogramKernel] = structs

    # ------------------------------------------------------------------
    # Aggregate properties
    # ------------------------------------------------------------------

    @property
    def n_structures(self) -> int:
        return len(self.structures)

    @property
    def sill(self) -> float:
        """``C₀ + Σᵢ Cᵢ``."""
        return float(self.nugget + sum(s.contribution for s in self.structures))

    def require_stationary_covariance(self, *, object_name: str) -> None:
        """Validate that this model is finite and usable as covariance."""
        if not np.isfinite(self.nugget) or not np.isfinite(self.sill) or self.sill <= 0.0:
            raise GeomodelNumericsError(
                "covariance sill must be positive and finite",
                object_name=object_name,
                field="variogram",
                expected="positive finite stationary covariance model",
                actual={"nugget": self.nugget, "sill": self.sill},
            )
        for structure in self.structures:
            structure.require_stationary_covariance(object_name=object_name)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def variogram(self, h: object) -> FloatArray:
        """``γ(h)`` for the nested model (isotropic: pre-rotate the lag)."""
        h_arr = as_float_array(h)
        result = as_float_array(np.where(h_arr > 0, self.nugget, 0.0))
        for struct in self.structures:
            result = as_float_array(result + struct.evaluate(h_arr))
        return result

    def covariance(self, h: object) -> FloatArray:
        """``C(h) = sill − γ(h)``."""
        return as_float_array(self.sill - self.variogram(h))

    def correlogram(self, h: object) -> FloatArray:
        """``ρ(h) = C(h) / sill``. Returns ones only for a zero sill."""
        h_arr = as_float_array(h)
        c0 = self.sill
        if c0 == 0.0:
            return as_float_array(np.ones_like(h_arr, dtype=np.float64))
        return as_float_array(self.covariance(h_arr) / c0)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def _single_kernel(
        cls,
        kind: int,
        sill: float,
        range_: float,
        nugget: float,
        angles: tuple[float, ...],
        anis: tuple[float, ...] | None = None,
        **params: object,
    ) -> "CovarianceModel":
        if sill <= nugget:
            raise GeoBrainError(
                "sill must exceed nugget",
                object_name="CovarianceModel",
                field="sill",
                expected=f"> {nugget}",
                actual=sill,
            )
        if range_ <= 0:
            raise GeoBrainError(
                "range_ must be positive",
                object_name="CovarianceModel",
                field="range_",
                expected="> 0",
                actual=range_,
            )
        contribution = sill - nugget
        if anis is None:
            ranges: tuple[float, ...] = (range_, range_, range_)
        else:
            ranges = (range_, range_ * anis[0], range_ * anis[1])
        return cls(
            nugget,
            [VariogramKernel(kind, contribution, ranges, angles, **params)],
        )

    @classmethod
    def spherical(
        cls,
        sill: float,
        range_: float,
        nugget: float = 0.0,
        angles: tuple[float, ...] = (0.0, 0.0, 0.0),
        anis: tuple[float, ...] | None = None,
    ) -> "CovarianceModel":
        return cls._single_kernel(VariogramKernel.SPHERICAL, sill, range_, nugget, angles, anis)

    @classmethod
    def exponential(
        cls,
        sill: float,
        range_: float,
        nugget: float = 0.0,
        angles: tuple[float, ...] = (0.0, 0.0, 0.0),
        anis: tuple[float, ...] | None = None,
    ) -> "CovarianceModel":
        return cls._single_kernel(VariogramKernel.EXPONENTIAL, sill, range_, nugget, angles, anis)

    @classmethod
    def gaussian(
        cls,
        sill: float,
        range_: float,
        nugget: float = 0.0,
        angles: tuple[float, ...] = (0.0, 0.0, 0.0),
        anis: tuple[float, ...] | None = None,
    ) -> "CovarianceModel":
        return cls._single_kernel(VariogramKernel.GAUSSIAN, sill, range_, nugget, angles, anis)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"nugget={self.nugget}"]
        parts.extend(repr(s) for s in self.structures)
        return f"CovarianceModel({', '.join(parts)})"


__all__ = ["CovarianceModel"]
