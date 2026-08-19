"""
Source-wavelet generators for FDTD / convolutional modeling.

All wavelet code lives here. Two coexisting APIs, both built on the shared
amplitude kernels in :mod:`._kernels`:

- **Class hierarchy**: :class:`WaveletGenerator` ABC + concrete
  :class:`RickerWavelet`, :class:`GaussianWavelet`, :class:`OrmsbyWavelet`,
  :class:`KlauderWavelet`. Each generator auto-sizes the time window from
  ``f0`` / ``dt`` and returns ``(wavelet, time_axis)``, **zero-phase** (centred).
  Useful when you want a wavelet sized to its natural support.

- **Free function** (:func:`ricker`, also exported as ``wave.ricker``):
  ``ricker(nt, dt, f0, *, causal=False) → Tensor`` on a caller-specified
  ``nt``-sample window. **Zero-phase** by default (the convolutional-synthetic
  form); ``causal=True`` gives the delayed form a step-off / explosive FDTD
  source wants. Preferred when ``nt`` is fixed by the simulation.

(Only the Ricker *amplitude* is shared between the two, Gaussian and Ormsby use
genuinely different formulas in their generator vs source forms.)

The :func:`create_wavelet` factory dispatches to a generator by string
name, optionally pads / truncates to ``nt``, and is the convenience
entry-point for example scripts.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations
from geobrain.core import GeoBrainError

import torch
from torch import Tensor

from .bandpass import KlauderWavelet, OrmsbyWavelet
from .base import WaveletGenerator
from .functions import ricker
from .impulse import GaussianWavelet, RickerWavelet


def create_wavelet(
    wavelet_type: str,
    f0: float,
    dt: float,
    nt: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    **kwargs: float,
) -> tuple[Tensor, Tensor]:
    """
    Build a wavelet by string name, optionally pad/truncate to ``nt``.

    Args:
        wavelet_type: ``"ricker"`` / ``"gaussian"`` / ``"ormsby"`` /
            ``"klauder"``. Case-insensitive.
        f0: Dominant frequency (Hz). Unused for Ormsby/Klauder but still
            required by the polymorphic signature.
        dt: Time-sampling interval (s).
        nt: If given, pad with zeros or truncate to exactly ``nt``
            samples. If ``None``, returns the generator's natural window.
        device, dtype: torch device / dtype.
        **kwargs: Extra arguments forwarded to the generator constructor
            (e.g. ``f1, f2, f3, f4`` for Ormsby; ``f_low, f_high, T`` for
            Klauder). For Ormsby these default to ``(0.2·f0, 0.5·f0,
            1.5·f0, 2.0·f0)``; for Klauder to ``(10, 80, 8.0)``.

    Returns:
        ``(wavelet, time_axis)`` 1-D tensors of equal length.
    """
    wavelet_type = wavelet_type.lower()

    if wavelet_type in ("ricker", "mexican_hat"):
        generator: WaveletGenerator = RickerWavelet()
    elif wavelet_type == "gaussian":
        generator = GaussianWavelet()
    elif wavelet_type == "ormsby":
        f1 = kwargs.pop("f1", f0 * 0.2)
        f2 = kwargs.pop("f2", f0 * 0.5)
        f3 = kwargs.pop("f3", f0 * 1.5)
        f4 = kwargs.pop("f4", f0 * 2.0)
        generator = OrmsbyWavelet(f1, f2, f3, f4)
    elif wavelet_type == "klauder":
        f_low = kwargs.pop("f_low", 10.0)
        f_high = kwargs.pop("f_high", 80.0)
        T = kwargs.pop("T", 8.0)
        generator = KlauderWavelet(f_low, f_high, T)
    else:
        raise GeoBrainError(
            f"Unknown wavelet type: '{wavelet_type}'. "
            f"Choose from: 'ricker', 'gaussian', 'ormsby', 'klauder'"
        )

    wavelet, time_axis = generator(f0, dt, device=device, dtype=dtype)

    if nt is not None:
        cur = wavelet.numel()
        if cur < nt:
            pad = torch.zeros(nt - cur, device=wavelet.device, dtype=wavelet.dtype)
            wavelet = torch.cat([wavelet, pad])
            extended_time = time_axis[-1] + dt * torch.arange(
                1,
                nt - cur + 1,
                device=time_axis.device,
                dtype=time_axis.dtype,
            )
            time_axis = torch.cat([time_axis, extended_time])
        elif cur > nt:
            wavelet = wavelet[:nt]
            time_axis = time_axis[:nt]
    return wavelet, time_axis


__all__ = [
    "GaussianWavelet",
    "KlauderWavelet",
    "OrmsbyWavelet",
    "RickerWavelet",
    "WaveletGenerator",
    "create_wavelet",
    "ricker",
]
