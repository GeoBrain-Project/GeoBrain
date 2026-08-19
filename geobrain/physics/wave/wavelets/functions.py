"""
Free-function wavelet builders (alongside the class generators in this package).

``ricker`` is the canonical Ricker free function on a caller-specified
``nt``-sample window:

- **zero-phase** (``causal=False``, default): centred on the window; this is the
  public ``wave.ricker`` used for convolutional synthetics;
- **causal** (``causal=True``): delayed by ``delay`` (default ``1/f0``); the form
  a step-off / explosive FDTD source wants.

Both forms share :func:`ricker_amplitude`; only the time axis differs.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

import torch
from torch import Tensor

from geobrain.core import GeoBrainError
from ._kernels import ricker_amplitude


def ricker(
    nt: int,
    dt: float,
    f0: float,
    *,
    causal: bool = False,
    delay: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Ricker (Mexican-hat) wavelet of length ``nt``.

    Args:
        nt: number of samples.
        dt: sample interval (s).
        f0: dominant (peak) frequency (Hz).
        causal: if ``False`` (default) the wavelet is **zero-phase**, centred on
            the ``nt`` window. If ``True`` it is **causal**, delayed by ``delay``.
        delay: causal time shift (s); defaults to ``1/f0`` (only used when
            ``causal``).
        device, dtype: output tensor device / dtype.

    Returns:
        1-D tensor of shape ``(nt,)``.
    """
    if nt <= 0:
        raise GeoBrainError(
            "ricker: nt must be positive",
            object_name="ricker", field="nt", expected="> 0", actual=nt,
        )
    if f0 <= 0:
        raise GeoBrainError(f"ricker: f0 must be positive: {f0}")
    if causal:
        if delay is None:
            delay = 1.0 / f0
        t = torch.arange(nt, device=device, dtype=dtype) * dt - delay
    else:
        t = (torch.arange(nt, device=device, dtype=dtype) - (nt - 1) / 2.0) * dt
    return ricker_amplitude(t, f0)
