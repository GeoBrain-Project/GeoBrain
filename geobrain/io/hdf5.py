"""
HDF5 reader / writer (h5py backend).

A thin convenience layer for storing ``numpy.ndarray`` blobs with optional
attribute metadata. Useful for caching intermediate inversion results,
sharing velocity models, or exchanging data with non-geophysics tooling.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

import numpy as np

from ..core.errors import GeoBrainError

from ._common import _to_numpy


class HDF5Data(NamedTuple):
    """Return type of :func:`read_hdf5`.

    Attributes:
        data: NumPy array read from the HDF5 dataset.
        attrs: Dataset attributes as a plain dict.
    """

    data: Any  # np.ndarray at runtime
    attrs: dict

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


def _require_h5py() -> Any:
    if h5py is None:
        raise ImportError("h5py not installed. Install with: pip install h5py")
    return h5py


def read_hdf5(path: str | os.PathLike, key: str = "data") -> HDF5Data:
    """
    Read a dataset at ``key`` and return an :class:`HDF5Data` named tuple
    ``(data, attrs)``.

    Attributes attached to the dataset (h5py "attrs") are returned as a
    plain dict; non-array attrs are coerced to Python scalars when possible.

    Tuple-unpacking is backward-compatible::

        data, attrs = read_hdf5("model.h5", "vp")

    Args:
        path: HDF5 file to read.
        key: dataset key inside the file (default ``'data'``).
    """
    h5 = _require_h5py()
    with h5.File(str(path), "r") as f:
        if key not in f:
            raise GeoBrainError(
                "HDF5 file has no such dataset",
                object_name="read_hdf5", field="key",
                expected=f"one of {list(f.keys())}", actual=key,
            )
        ds = f[key]
        data = np.asarray(ds[...])
        attrs: dict = {}
        for k, v in ds.attrs.items():
            if isinstance(v, np.ndarray) and v.ndim == 0:
                attrs[k] = v.item()
            elif isinstance(v, bytes):
                attrs[k] = v.decode("utf-8", errors="replace")
            else:
                attrs[k] = v
    return HDF5Data(data, attrs)


def write_hdf5(
    path: str | os.PathLike,
    key: str,
    data: np.ndarray,
    attrs: dict | None = None,
    *,
    compression: str | None = "gzip",
) -> None:
    """
    Write ``data`` under ``key`` with optional ``attrs``.

    Tensor inputs are detached + moved to cpu before conversion to numpy.
    Existing files are appended to (the key is overwritten if present).

    Args:
        path: output HDF5 file.
        key: dataset key to create.
        data: array payload.
        attrs: optional attribute dict stored on the dataset.
        compression: HDF5 compression filter (``None`` disables).
    """
    h5 = _require_h5py()

    arr = _to_numpy(data)

    with h5.File(str(path), "a") as f:
        if key in f:
            del f[key]
        ds = f.create_dataset(key, data=arr, compression=compression)
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v
