"""
Ricker (Mexican-hat) and Gaussian wavelet generators.

The Ricker wavelet is the second derivative of a Gaussian; it is the de-facto
default source-time function in exploration seismic FDTD.

Both generators auto-size the time window from ``f0`` / ``dt`` / ``length_factor``
and return a symmetric zero-phase wavelet on a window ``[-nc·dt, +nc·dt]`` with
``nw = 2·nc + 1`` samples (always odd so the peak lands on a sample).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._kernels import ricker_amplitude
from .base import WaveletGenerator


def _odd_window(f0: float, dt: float, length_factor: float) -> int:
    nw = length_factor / f0 / dt
    return 2 * int(nw / 2) + 1


class RickerWavelet(WaveletGenerator):
    """Zero-phase Ricker wavelet: ``(1 - 2π²f₀²t²)·exp(-π²f₀²t²)``."""

    def __init__(self, length_factor: float = 2.2) -> None:
        super().__init__()
        self.length_factor = length_factor
        self._name = "ricker"

    def forward(
        self,
        f0: float,
        dt: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(f0, Tensor):
            f0 = float(f0.item())
        if isinstance(dt, Tensor):
            dt = float(dt.item())
        device = device or "cpu"

        nw = _odd_window(f0, dt, self.length_factor)
        nc = nw // 2

        t = (torch.arange(nw, device=device, dtype=dtype) - nc) * dt
        wavelet = ricker_amplitude(t, f0)
        return wavelet, t

    def get_length(self, f0: float, dt: float) -> int:
        return _odd_window(f0, dt, self.length_factor)


class GaussianWavelet(WaveletGenerator):
    """Gaussian pulse: ``exp(-4·f₀²·t²)``. Useful as the integrated-Ricker primitive."""

    def __init__(self, length_factor: float = 2.0) -> None:
        super().__init__()
        self.length_factor = length_factor
        self._name = "gaussian"

    def forward(
        self,
        f0: float,
        dt: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(f0, Tensor):
            f0 = float(f0.item())
        if isinstance(dt, Tensor):
            dt = float(dt.item())
        device = device or "cpu"

        nw = _odd_window(f0, dt, self.length_factor)
        nc = nw // 2

        t = (torch.arange(nw, device=device, dtype=dtype) - nc) * dt
        wavelet = torch.exp(-4.0 * f0 ** 2 * t ** 2)
        return wavelet, t


__all__ = ["RickerWavelet", "GaussianWavelet"]
