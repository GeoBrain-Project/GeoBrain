# pyright: reportPrivateImportUsage=false
"""
Shared helpers for the IO backends.

IO is the *detached* boundary, the operator graph and the file-format
backends never share a live autograd graph. The :func:`_to_numpy`
helper is the single chokepoint that enforces this: every backend that
opts in (HDF5 / LAS / SEG-Y / VTK) routes user-supplied tensors through
it before handing them to the format library.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover; torch is a hard dep, but stay defensive
    torch = None  # type: ignore[assignment]


__all__ = ["_to_numpy"]


def _to_numpy(data: Any, *, dtype=None) -> np.ndarray:
    """
    Convert ``data`` to a numpy array, detaching torch tensors.

    - :class:`torch.Tensor` (any device, ``requires_grad=True`` OK) →
      ``data.detach().cpu().numpy()``.
    - :class:`numpy.ndarray` / array-like → ``np.asarray(data)``.

    Always returns a numpy array. The optional ``dtype`` is applied via
    ``np.asarray`` after detachment.
    """
    if torch is not None and isinstance(data, torch.Tensor):
        arr = data.detach().cpu().numpy()
    else:
        arr = np.asarray(data)
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    return arr
