"""
Abstract base for source-wavelet generators.

Wavelets are stateless math factories (no trainable parameters), so the base is a
plain ``ABC``; ``nn.Module`` would only add unused machinery. Concrete generators
implement :meth:`forward(f0, dt, device, dtype)` returning ``(wavelet, time_axis)``
as 1-D torch tensors.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class WaveletGenerator(ABC):
    """Abstract source-wavelet generator."""

    def __init__(self) -> None:
        self._name = "base"

    @abstractmethod
    def forward(
        self,
        f0: float,
        dt: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(wavelet, time_axis)`` as 1-D tensors of the same length."""

    def __call__(
        self,
        f0: float,
        dt: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        return self.forward(f0, dt, device=device, dtype=dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def name(self) -> str:
        return self._name


__all__ = ["WaveletGenerator"]
