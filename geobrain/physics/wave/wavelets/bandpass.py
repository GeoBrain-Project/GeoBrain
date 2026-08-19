"""
Bandpass wavelets: Ormsby and Klauder.

- :class:`OrmsbyWavelet` produces a trapezoidal-spectrum wavelet with four corner
  frequencies ``f1 < f2 < f3 < f4`` (zero outside, flat between f2-f3, ramped on
  the edges).
- :class:`KlauderWavelet` is the autocorrelation of a linear-sweep vibroseis signal
  between ``f_low`` and ``f_high`` over duration ``T``.

Both classes use the same ``(f0, dt) → (wavelet, t)`` interface as
:class:`~geobrain.physics.wave.wavelets.base.WaveletGenerator`. The ``f0`` argument
is unused for these (their spectrum is set by the corner / sweep frequencies) but
is kept in the signature for polymorphism with :class:`RickerWavelet`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from geobrain.core import GeoBrainError

import math

import torch
from torch import Tensor

from .base import WaveletGenerator


class OrmsbyWavelet(WaveletGenerator):
    """Trapezoidal-spectrum Ormsby wavelet.

    Args:
        f1 / f2 / f3 / f4: trapezoid corner frequencies [Hz]
            (``f1 < f2 < f3 < f4``).
    """

    def __init__(self, f1: float, f2: float, f3: float, f4: float) -> None:
        super().__init__()
        if not (f1 < f2 < f3 < f4):
            raise GeoBrainError(
                f"Frequencies must satisfy f1 < f2 < f3 < f4, got ({f1}, {f2}, {f3}, {f4})"
            )
        if f1 <= 0:
            raise GeoBrainError(f"All corner frequencies must be positive, got f1={f1}")
        self.f1, self.f2, self.f3, self.f4 = f1, f2, f3, f4
        self._name = "ormsby"

    def forward(
        self,
        f0: float,
        dt: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(dt, Tensor):
            dt = float(dt.item())
        device = device or "cpu"

        nw = int(2.0 / self.f1 / dt)
        nw = 2 * (nw // 2) + 1
        nc = nw // 2

        t = (torch.arange(nw, device=device, dtype=dtype) - nc) * dt
        eps = torch.tensor(1e-10, device=device, dtype=dtype)
        t_safe = torch.where(torch.abs(t) < 1e-10, eps, t)

        def sinc_sq(f: float) -> Tensor:
            arg = math.pi * f * t_safe
            return (torch.sin(arg) / arg).pow(2) * f ** 2

        wavelet = (sinc_sq(self.f4) - sinc_sq(self.f3)) / (self.f4 - self.f3) - (
            sinc_sq(self.f2) - sinc_sq(self.f1)
        ) / (self.f2 - self.f1)

        center = torch.tensor(
            (self.f4 + self.f3) - (self.f2 + self.f1), device=device, dtype=dtype
        )
        wavelet = torch.where(torch.abs(t) < 1e-10, center, wavelet)
        wavelet = wavelet / torch.max(torch.abs(wavelet))
        return wavelet, t

    def __repr__(self) -> str:
        return f"OrmsbyWavelet(f1={self.f1}, f2={self.f2}, f3={self.f3}, f4={self.f4})"


class KlauderWavelet(WaveletGenerator):
    """Klauder vibroseis-sweep wavelet.

    Args:
        f_low / f_high: sweep band edges [Hz].
        T: sweep duration [s].
    """

    def __init__(self, f_low: float, f_high: float, T: float) -> None:
        super().__init__()
        if f_low >= f_high:
            raise GeoBrainError(f"f_low must be less than f_high: {f_low} >= {f_high}")
        if T <= 0:
            raise GeoBrainError(f"Sweep duration must be positive: {T}")
        self.f_low, self.f_high, self.T = f_low, f_high, T
        self._name = "klauder"

    def forward(
        self,
        f0: float,
        dt: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        if isinstance(dt, Tensor):
            dt = float(dt.item())
        device = device or "cpu"

        bandwidth = self.f_high - self.f_low
        f_center = 0.5 * (self.f_low + self.f_high)

        nw = int(self.T / dt)
        nw = 2 * (nw // 2) + 1
        nc = nw // 2

        t = (torch.arange(nw, device=device, dtype=dtype) - nc) * dt
        eps = torch.tensor(1e-10, device=device, dtype=dtype)
        t_safe = torch.where(torch.abs(t) < 1e-10, eps, t)

        arg1 = math.pi * bandwidth * t_safe
        arg2 = 2.0 * math.pi * f_center * t
        wavelet = torch.sin(arg1) / arg1 * torch.cos(arg2)

        one = torch.tensor(1.0, device=device, dtype=dtype)
        wavelet = torch.where(torch.abs(t) < 1e-10, one, wavelet)
        return wavelet, t

    def __repr__(self) -> str:
        return f"KlauderWavelet(f_low={self.f_low}, f_high={self.f_high}, T={self.T})"


__all__ = ["OrmsbyWavelet", "KlauderWavelet"]
