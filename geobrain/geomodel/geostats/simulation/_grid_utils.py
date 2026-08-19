"""
Shared grid-conditioning helpers for the simulation algorithms.

Pure-numpy utilities used across the sequential (SGSIM/SISIM/DSSIM/CoSGSIM) and
patch-based (DirectSampling/FILTERSIM/SNESIM/ImageQuilting) simulators:

- :func:`flatten_gslib`: ``(nx, ny, nz)`` → flat GSLIB x-fastest ordering.
- :func:`fill_hard_data`: paint conditioning ("hard") data onto a grid array
  at each point's containing cell.
- :func:`snap_to_hard_data`: force simulated nodes coincident with a
  conditioning point to that point's exact value (reproduction property).

Factored here so the individual algorithm modules don't each carry a private
copy of identical code.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...frames._arrays import FloatArray, as_float_array

if TYPE_CHECKING:
    from ...frames import GeoFrame, GeoGrid


def flatten_gslib(arr: FloatArray) -> FloatArray:
    """Convert a declared-rank or adapter array to flat x-fastest ordering."""
    return as_float_array(np.asarray(arr).ravel(order="F"))


def fill_hard_data(
    grid: GeoGrid,
    data: GeoFrame,
    col: str,
    out: np.ndarray,
) -> None:
    """Paint conditioning ("hard") data onto ``out`` at each point's grid cell."""
    coords = as_float_array(data.geometry.coords)
    values = as_float_array(data[col])
    # Map each data point to its containing grid cell. Cell centres sit at
    # ``(i + 0.5)·xsiz + xmn`` (see GeoGrid.coords); inverting gives
    # ``i = floor((x - xmn)/xsiz)``: same convention as ``fft_ma.py``.
    for k in range(coords.shape[0]):
        i = int(np.floor((coords[k, 0] - grid.xmn) / grid.xsiz))
        j = int(np.floor((coords[k, 1] - grid.ymn) / grid.ysiz))
        if grid.ndim == 2:
            if 0 <= i < grid.nx and 0 <= j < grid.ny:
                if out.ndim == 2:
                    out[i, j] = float(values[k])
                else:
                    out[i, j, 0] = float(values[k])
            continue
        kk = int(np.floor((coords[k, 2] - grid.zmn) / grid.zsiz))
        if 0 <= i < grid.nx and 0 <= j < grid.ny and 0 <= kk < grid.nz:
            out[i, j, kk] = float(values[k])


def snap_to_hard_data(
    sim: np.ndarray,
    targets: np.ndarray,
    cond_coords: np.ndarray,
    cond_values: np.ndarray,
) -> None:
    """Force simulated nodes coincident with a conditioning point to its value.

    Even with regularisation a kriging system only gets *close* at a hard-data
    location; this nails the reproduction property. Mutates ``sim`` in place.
    """
    for k in range(cond_coords.shape[0]):
        dist = np.sum((targets - cond_coords[k]) ** 2, axis=1)
        coincident = np.flatnonzero(dist < 1.0e-12)
        for idx in coincident:
            sim[idx] = float(cond_values[k])
