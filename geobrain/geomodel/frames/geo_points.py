"""GeoPoints: unstructured 2-D / 3-D point geometry (numpy).

Stores an unstructured point cloud:

- coords stored as ``np.ndarray`` of float64;
- accepts any tensor-like input (lists, tuples, torch.Tensor) and
  coerces via ``as_float_array``;
- validation via :class:`GeoBrainError`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import numpy as np

from ..errors import GeomodelContractError
from ._arrays import FloatArray, as_float_array
from .geometry import Geometry


class GeoPoints(Geometry):
    """
    Unstructured set of points in 2-D or 3-D space.

    Args:
        coords: ``(n, 2)`` or ``(n, 3)`` array-like (list, tuple,
            ndarray, or torch.Tensor, coerced via
            :func:`as_float_array`).
    """

    def __init__(self, coords: object) -> None:
        try:
            arr = as_float_array(coords)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GeomodelContractError(
                "GeoPoints coords must be numeric",
                object_name="GeoPoints",
                field="coords",
                expected="finite float64 array with shape (n, 2) or (n, 3)",
                actual=type(coords).__name__,
            ) from exc
        if arr.ndim != 2:
            raise GeomodelContractError(
                "GeoPoints coords must be 2-D of shape (n, 2) or (n, 3)",
                object_name="GeoPoints",
                field="coords",
                expected="2D",
                actual=tuple(arr.shape),
            )
        if arr.shape[1] not in (2, 3):
            raise GeomodelContractError(
                "GeoPoints supports 2-D or 3-D coords only",
                object_name="GeoPoints",
                field="coords.shape[1]",
                expected="2 or 3",
                actual=int(arr.shape[1]),
            )
        if arr.shape[0] == 0:
            raise GeomodelContractError(
                "GeoPoints must contain at least one point",
                object_name="GeoPoints",
                field="coords.shape[0]",
                expected=">= 1",
                actual=0,
            )
        if not np.isfinite(arr).all():
            raise GeomodelContractError(
                "GeoPoints coords must be finite (no NaN / inf)",
                object_name="GeoPoints",
                field="coords",
                expected="finite",
                actual="contains NaN or inf",
            )
        # Copy on store: ``as_float_array`` aliases a caller-supplied
        # float64 ndarray (``np.asarray`` does not copy), which would let
        # ``.coords`` be mutated post-construction and bypass the finiteness
        # check above. Mirror ``GeoFrame._coerce_property``'s column copy.
        owned = np.array(arr, dtype=np.float64, copy=True, order="C")
        # Back the public array with immutable bytes. Merely clearing the
        # writeable flag on an owning ndarray is reversible through
        # ``setflags(write=True)``; a bytes-backed buffer makes the read-only
        # contract non-recoverable through the public ndarray API.
        readonly = np.frombuffer(owned.tobytes(order="C"), dtype=np.float64).reshape(owned.shape)
        self._coords: FloatArray = readonly

    @property
    def coords(self) -> FloatArray:
        return self._coords

    def __repr__(self) -> str:
        return f"GeoPoints({self.npoints} points, {self.ndim}D)"


__all__ = ["GeoPoints"]
