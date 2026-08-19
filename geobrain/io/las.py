"""
LAS (well-log ASCII) reader (lasio backend).

Returns the curves as a dict of 1D ``numpy.ndarray`` keyed by curve mnemonic
(e.g., ``"DEPT"``, ``"GR"``, ``"VP"``). Depth is included as one of the
curves rather than as a separate axis.

Writers are not provided: scientific use rarely overwrites LAS, and the few
who need it can call ``lasio`` directly.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

import numpy as np


class LASData(NamedTuple):
    """Return type of :func:`read_las`.

    Attributes:
        curves: ``mnemonic → 1D numpy.ndarray``, all of equal length.
        meta: Well info from the LAS header as a shallow dict.
    """

    curves: dict
    meta: dict

try:
    import lasio
except ImportError:  # pragma: no cover
    lasio = None  # type: ignore[assignment]


def _require_lasio() -> Any:
    if lasio is None:
        raise ImportError(
            "lasio not installed. Install with: pip install lasio"
        )
    return lasio


def read_las(path: str | os.PathLike) -> LASData:
    """
    Read a LAS file.

    Returns a :class:`LASData` named tuple ``(curves, meta)``:

    - ``curves``: ``mnemonic → 1D numpy.ndarray``, all of equal length.
    - ``meta``: well info from the LAS header (well name, UWI, etc.) as a
      shallow dict (``well.section`` items only).

    Tuple-unpacking is backward-compatible::

        curves, meta = read_las("well.las")
    """
    ls = _require_lasio()
    las = ls.read(str(path))
    curves: dict[str, np.ndarray] = {}
    for curve in las.curves:
        curves[curve.mnemonic] = np.asarray(curve.data, dtype=np.float64)

    meta: dict = {}
    if hasattr(las, "well"):
        for item in las.well:
            meta[item.mnemonic] = item.value
    return LASData(curves, meta)
