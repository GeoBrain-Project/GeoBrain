"""
Multi-scale (frequency-continuation) FWI support.

Low-frequency-first inversion is the standard cure for FWI cycle-skipping
(Bunks et al. 1995; Virieux & Operto 2009): invert a low-pass-filtered gather
at an increasing sequence of cut-off frequencies, each stage warm-started from
the previous model, so the long wavelengths are recovered before the short
ones are trusted.

This module provides the two pieces GeoBrain needs for that on top of the
existing solver stack:

- :func:`butterworth_lowpass`: a DIFFERENTIABLE zero-phase Butterworth
  low-pass along the time axis, implemented in the frequency domain
  (``irfft(|H(f)|·rfft(x))`` with the analog Butterworth magnitude
  ``|H|² = 1/(1+(f/fc)^{2n})``). Zero-phase and autograd-friendly, the
  differentiable analog of the reference's ``scipy.filtfilt`` (which is also
  zero-phase; interior response matches, edge handling differs).
- :class:`FrequencyFilteredMisfit`: a :class:`~geobrain.inverse.
  likelihood.Likelihood`-protocol wrapper that low-pass-filters BOTH the
  predicted and observed gathers at ``cutoff`` before delegating to an inner
  waveform misfit. Drops into ``InverseProblem(likelihood=...)``, so a
  frequency-continuation cascade is a loop that rebuilds it per stage.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ..core.errors import GeoBrainError
from .likelihood import Likelihood, _validate_common_channel_tensors

__all__ = ["butterworth_lowpass", "FrequencyFilteredMisfit"]


def butterworth_lowpass(
    x: torch.Tensor,
    cutoff: float,
    fs: float,
    *,
    order: int = 4,
    time_axis: int = 1,
) -> torch.Tensor:
    """Zero-phase Butterworth low-pass along ``time_axis`` (differentiable).

    Args:
        x: Gather tensor (real). Any shape; the transform acts on
            ``time_axis`` (default 1, the ``(n_src, nt, n_rcv)`` time axis).
        cutoff: Cut-off frequency in Hz (``0 < cutoff < fs/2``).
        fs: Sampling frequency in Hz (``1/dt``).
        order: Butterworth order (steeper roll-off for larger).
        time_axis: Axis to filter.

    Returns:
        The low-pass-filtered tensor, same shape/dtype as ``x``.
    """
    if not cutoff > 0:
        raise GeoBrainError(
            "butterworth_lowpass cutoff must be > 0",
            object_name="butterworth_lowpass", field="cutoff",
            expected="> 0", actual=cutoff,
        )
    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        raise GeoBrainError(
            "butterworth_lowpass cutoff must be below the Nyquist (fs/2)",
            object_name="butterworth_lowpass", field="cutoff",
            expected=f"< {nyquist}", actual=cutoff,
        )
    nt = x.shape[time_axis]
    freqs = torch.fft.rfftfreq(nt, d=1.0 / fs, device=x.device, dtype=x.dtype)
    # Analog Butterworth zero-phase magnitude |H(f)|² = 1/(1+(f/fc)^{2n}).
    resp = 1.0 / (1.0 + (freqs / cutoff) ** (2 * order))
    shape = [1] * x.ndim
    shape[time_axis] = resp.shape[0]
    xf = torch.fft.rfft(x, dim=time_axis)
    return torch.fft.irfft(xf * resp.reshape(shape), n=nt, dim=time_axis)


@dataclass(frozen=True)
class FrequencyFilteredMisfit:
    """Low-pass-filter both gathers at ``cutoff``, then delegate to ``inner``.

    Satisfies the :class:`~geobrain.inverse.likelihood.Likelihood`
    protocol (``misfit`` / ``log_likelihood``), so it slots into
    ``InverseProblem(likelihood=...)``.
    A frequency-continuation cascade rebuilds it per stage with an increasing
    ``cutoff``, warm-starting the model each time.

    Attributes:
        inner: The base misfit to delegate to after filtering, any
            :class:`~geobrain.inverse.likelihood.Likelihood`-protocol object
            (typically one of the ``geobrain.inverse`` waveform misfits, e.g.
            :class:`~geobrain.inverse.misfit.L2Waveform`). Filter-first
            ordering (filter, then whatever ``inner`` does, including its
            own ``normalize``) falls out for free: this wrapper filters both
            gathers and delegates, so a ``normalize=True`` inner misfit
            normalizes the ALREADY-filtered gathers by the filtered
            observed amplitude.
        cutoff: Low-pass cut-off in Hz for this stage.
        fs: Sampling frequency in Hz (``1/dt``).
        order: Butterworth order.
        channel: Gather channel to filter (default ``"seismic"``).
    """

    inner: Likelihood
    cutoff: float
    fs: float
    order: int = 4
    channel: str = "seismic"

    def _filtered(
        self,
        mapping: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out = dict(mapping)
        if self.channel in out:
            out[self.channel] = butterworth_lowpass(
                out[self.channel], self.cutoff, self.fs, order=self.order
            )
        return out

    @property
    def gradient_singular(self) -> bool:
        """Proxy the inner misfit's declaration so the Inverter's automatic
        ``nan_policy="guard"`` protection covers filtered stages too (found
        during the P2c example migrations: without this proxy, every
        multiscale cascade stage silently lost the auto-guard)."""
        return bool(getattr(self.inner, "gradient_singular", False))

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        _validate_common_channel_tensors(
            predicted,
            observed,
            (self.channel,),
            object_name="FrequencyFilteredMisfit.misfit",
        )
        return self.inner.misfit(self._filtered(predicted), self._filtered(observed))

    def log_likelihood(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return -self.misfit(predicted, observed)
