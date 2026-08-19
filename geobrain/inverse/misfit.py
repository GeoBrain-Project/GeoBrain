"""
Waveform data-misfit functionals for full-waveform inversion.

Each misfit is a frozen dataclass that satisfies the
:class:`~geobrain.inverse.likelihood.Likelihood` protocol
(``misfit`` + ``log_likelihood``), so it drops straight into
``InverseProblem(likelihood=...)`` in place of the default
:class:`~geobrain.inverse.GaussianLikelihood`. They operate on a seismic gather
channel of shape ``(n_src, nt, n_rcv)``, GeoBrain's wave convention, time
on axis 1, and are fully vectorized (no per-shot / per-trace Python loops).

Catalog (Brossier/Virieux-lineage FWI objectives; cross-validated against
an independent open-source FWI implementation):

- :class:`L2Waveform`: per-(shot,receiver) energy ``Σ √(dt·Σ_t r²)``
  (Tarantola 1984). NOTE this is the FWI-standard L2, NOT the χ² of
  :class:`GaussianLikelihood`; use the latter for a Gaussian-noise posterior.
- :class:`L1Waveform`: ``Σ|r|·dt`` (robust to spiky noise; Laplace-flavoured).
- :class:`HuberWaveform`: smooth-L1 / Huber, quadratic near zero, linear in
  the tails (``β`` knee).
- :class:`StudentTWaveform`: heavy-tailed robust misfit (outlier-tolerant).
- :class:`EnvelopeWaveform`: analytic-signal envelope difference (Hilbert via
  FFT); low-frequency, cycle-skip-robust (Wu/Luo/Bozdağ).
- :class:`GlobalCorrelationWaveform`: negative zero-lag normalized
  cross-correlation, amplitude-invariant (Routh 2011).
- :class:`NormalizedIntegrationWaveform`: optimal-transport-flavoured misfit
  on trace CDFs after a positivity transform (Engquist/Yang).
- :class:`TravelTimeWaveform`: cross-correlation travel-time-shift misfit,
  differentiable soft-argmax (Luo & Schuster 1991).
- :class:`WassersteinWaveform`: 1-D optimal-transport (Wasserstein-p) misfit
  on trace distributions (Engquist & Froese 2014; Métivier et al. 2016).

Which are true likelihoods: L2 ↔ Gaussian, L1 ↔ Laplace, StudentT ↔ Student-t
noise. The others (Envelope, GlobalCorrelation, NormalizedIntegration,
TravelTime, Wasserstein) are OBJECTIVE functionals, not noise models;
``log_likelihood`` returns ``−misfit`` so they still drive
:class:`~geobrain.optim.Inverter` (MAP/deterministic FWI), but do not
interpret them as a calibrated data-noise likelihood in a Bayesian chain.

Protocol attrs (read via ``getattr(obj, name, default)``, a bare
protocol-conforming misfit with neither attribute stays first-class):

- ``gradient_singular`` (``ClassVar[bool]``, default ``False``): flags
  misfits whose gradient has a removable 0/0 singularity at a perfectly-fit
  trace (``L2Waveform``'s ``d√E/dr = r/√E``, ``WassersteinWaveform``'s
  integer-``searchsorted`` quantile detachment / OT singularities,
  ``EnvelopeWaveform``'s analytic-magnitude denominators). Consumed by
  :class:`~geobrain.optim.Inverter`'s NaN-guard policy; it is a ``ClassVar``,
  NOT a dataclass field, so it is never a constructor kwarg on these frozen
  dataclasses.
- ``normalize`` (``bool`` dataclass field, default ``False``): when
  ``True``, ``_pull`` divides BOTH gathers by the per-shot amplitude of the
  OBSERVED gather (``obs.abs().amax(dim=(1, 2), keepdim=True)``, clamped
  ``>= 1e-12``), stateless per call. Objective-equivalent to the manual
  ``obs.abs().amax(dim=(1, 2), keepdim=True)`` normalization used by the
  Marmousi FWI examples.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping

import torch
import torch.nn.functional as F

from ..core.errors import GeoBrainError, MissingFieldError
from .likelihood import _validate_common_channel_tensors

__all__ = [
    "L2Waveform",
    "L1Waveform",
    "HuberWaveform",
    "StudentTWaveform",
    "EnvelopeWaveform",
    "GlobalCorrelationWaveform",
    "NormalizedIntegrationWaveform",
    "TravelTimeWaveform",
    "WassersteinWaveform",
]


def _pull(
    predicted: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
    channel: str,
    owner: str,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fetch (syn, obs) and validate the shared gather tensor contract.

    When ``normalize`` is True, both gathers are divided by the per-shot
    amplitude of the OBSERVED gather (``obs.abs().amax(dim=(1, 2),
    keepdim=True)``, clamped ``>= 1e-12``), stateless per call, observed-side
    divisor only (the predicted gather never influences the divisor). This is
    the ``normalize`` protocol attr's implementation, shared by every misfit
    that routes through ``_pull``.
    """
    if channel not in observed:
        raise MissingFieldError(
            f"{owner} observed is missing the misfit channel",
            object_name=owner, field=channel,
            expected=f"present in observed {sorted(observed)}",
        )
    if channel not in predicted:
        raise MissingFieldError(
            f"{owner} predicted is missing the misfit channel",
            object_name=owner, field=channel,
            expected=f"present in predicted {sorted(predicted)}",
        )
    _validate_common_channel_tensors(
        predicted,
        observed,
        (channel,),
        object_name=owner,
    )
    syn, obs = predicted[channel], observed[channel]
    if obs.ndim != 3:
        raise GeoBrainError(
            f"{owner} expects a gather of shape (n_src, nt, n_rcv)",
            object_name=owner, field=channel,
            expected="3-D (n_src, nt, n_rcv)", actual=tuple(obs.shape),
        )
    if normalize:
        divisor = obs.detach().abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        syn = syn / divisor
        obs = obs / divisor
    return syn, obs


def _live_mask(obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
    """(n_src, n_rcv) bool: traces where BOTH gathers aren't all-zero.

    Dead-trace masking: muted / absent shot-receiver pairs
    do not contribute. For fully live data this is all-True.
    """
    dead = (obs.abs().sum(dim=1) == 0) & (syn.abs().sum(dim=1) == 0)
    return ~dead


def _analytic(x: torch.Tensor) -> torch.Tensor:
    """Analytic signal along the time axis (axis 1) via the FFT Hilbert.

    Zero-padded to ``2·nt`` (matching the reference) so circular-convolution
    wrap-around does not contaminate the envelope.
    """
    nt = x.shape[1]
    n = 2 * nt
    xf = torch.fft.fft(x, n=n, dim=1)
    h = torch.zeros(n, dtype=xf.dtype, device=x.device)
    h[0] = 1.0
    h[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    shape = [1, n] + [1] * (x.ndim - 2)
    return torch.fft.ifft(xf * h.reshape(shape), dim=1)[:, :nt]


@dataclass(frozen=True)
class _WaveMisfitBase:
    """Shared base for the waveform-misfit family.

    Attributes:
        channel: Gather channel key (default ``"seismic"``).
        dt: Sample interval, used to scale time-integral misfits.
        normalize: When True, ``_pull`` divides both gathers by the per-shot
            amplitude of the OBSERVED gather (see :func:`_pull`) before the
            misfit is formed, objective-equivalent to the manual
            ``obs.abs().amax(dim=(1, 2), keepdim=True)`` normalization used by
            the Marmousi FWI examples.
        gradient_singular: ``ClassVar``, NOT a dataclass field (never a
            constructor kwarg), True on misfits whose gradient has a
            removable 0/0 singularity at a perfectly-fit trace.
    """

    channel: str = "seismic"
    dt: float = 1.0
    normalize: bool = False

    gradient_singular: ClassVar[bool] = False

    if TYPE_CHECKING:
        def misfit(
            self,
            predicted: Mapping[str, torch.Tensor],
            observed: Mapping[str, torch.Tensor],
        ) -> torch.Tensor: ...

    def log_likelihood(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return -self.misfit(predicted, observed)


@dataclass(frozen=True)
class L2Waveform(_WaveMisfitBase):
    """FWI L2: ``Σ_{s,r} √(dt · Σ_t r²)`` over live traces (Tarantola 1984).

    ``gradient_singular = True``: ``d√E/dr = r/√E`` is a 0/0 at a
    perfectly-fit trace (``E == 0``).

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
    """

    gradient_singular: ClassVar[bool] = True

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "L2Waveform",
                         normalize=self.normalize)
        rsd = obs - syn
        energy = (rsd * rsd * self.dt).sum(dim=1)          # (n_src, n_rcv)
        mask = _live_mask(obs, syn)
        return torch.sqrt(energy[mask]).sum()


@dataclass(frozen=True)
class L1Waveform(_WaveMisfitBase):
    """``Σ |obs − syn| · dt`` (robust to spiky outliers).

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
    """

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "L1Waveform",
                         normalize=self.normalize)
        return (obs - syn).abs().sum() * self.dt


@dataclass(frozen=True)
class HuberWaveform(_WaveMisfitBase):
    """Smooth-L1 / Huber misfit (``reduction='sum'``) scaled by ``dt``.

    Quadratic for ``|r| < beta``, linear beyond, the standard robust knee.

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
        beta: Huber knee, quadratic below, linear above.
    """

    beta: float = 1.0

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "HuberWaveform",
                         normalize=self.normalize)
        return F.smooth_l1_loss(syn, obs, reduction="sum", beta=self.beta) * self.dt


@dataclass(frozen=True)
class StudentTWaveform(_WaveMisfitBase):
    """Heavy-tailed Student-t misfit ``Σ 0.5(s+1)·log(1 + r²/(s·σ²))·dt``.

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
        sigma: residual scale of the Student-t model.
        s: degrees of freedom (smaller = heavier tails).
    """

    sigma: float = 1.0
    s: float = 1.0

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "StudentTWaveform",
                         normalize=self.normalize)
        rsd = syn - obs
        term = 0.5 * (self.s + 1.0) * torch.log(
            1.0 + rsd * rsd / (self.s * self.sigma * self.sigma)
        )
        return (term * self.dt).sum()


@dataclass(frozen=True)
class EnvelopeWaveform(_WaveMisfitBase):
    """Analytic-signal envelope-difference misfit (cycle-skip-robust).

    ``0.5·Σ (E_syn^p − E_obs^p)²·dt`` (``norm='L2'``) or ``Σ|·|`` (``norm='L1'``),
    with ``E = |Hilbert(trace)|`` the instantaneous amplitude.

    ``gradient_singular = True``: the analytic-magnitude denominators (``|·|``
    of the Hilbert transform) are singular at a zero-amplitude sample.

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
        p: power applied to the envelopes before differencing.
        norm: ``'L1'`` or ``'L2'`` envelope-difference norm.
    """

    p: float = 1.0
    norm: str = "L2"

    gradient_singular: ClassVar[bool] = True

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "EnvelopeWaveform",
                         normalize=self.normalize)
        if self.norm not in ("L1", "L2"):
            raise GeoBrainError(
                "EnvelopeWaveform norm must be 'L1' or 'L2'",
                object_name="EnvelopeWaveform", field="norm",
                expected="'L1' | 'L2'", actual=self.norm,
            )
        e_obs = _analytic(obs).abs()
        e_syn = _analytic(syn).abs()
        rsd = e_syn.pow(self.p) - e_obs.pow(self.p)
        mask = _live_mask(obs, syn).unsqueeze(1)           # (n_src, 1, n_rcv)
        rsd = rsd * mask
        if self.norm == "L1":
            return rsd.abs().sum()
        return 0.5 * (rsd * rsd * self.dt).sum()


@dataclass(frozen=True)
class GlobalCorrelationWaveform(_WaveMisfitBase):
    """Negative zero-lag normalized cross-correlation, summed over live traces.

    Each trace is L2-normalized, then the Pearson correlation over time is
    formed and NEGATED (maximizing correlation = minimizing misfit).
    Amplitude-invariant (Routh et al. 2011).

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
        eps: floor guarding the normalization denominators.
    """

    eps: float = 1e-8

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "GlobalCorrelationWaveform",
                         normalize=self.normalize)
        # per-trace L2 normalization along time (axis 1)
        o = obs / obs.norm(dim=1, keepdim=True).clamp_min(self.eps)
        s = syn / syn.norm(dim=1, keepdim=True).clamp_min(self.eps)
        cov = (o * s).mean(dim=1)                           # (n_src, n_rcv)
        var_o = o.var(dim=1, unbiased=True)
        var_s = s.var(dim=1, unbiased=True)
        corr = cov / (torch.sqrt(var_o * var_s) + self.eps)
        corr = torch.nan_to_num(corr, nan=0.0)
        mask = _live_mask(obs, syn)
        return -(corr * mask).sum() * self.dt


@dataclass(frozen=True)
class NormalizedIntegrationWaveform(_WaveMisfitBase):
    """Optimal-transport-flavoured misfit on trace CDFs (Engquist/Yang).

    Applies a positivity transform, normalizes each trace to a probability
    over time, forms the cumulative distributions ``F, G`` and returns
    ``Σ |F − G|^p``. ``trans='linear'`` shifts by the global minimum
    (scaled by ``theta``); ``'abs'`` / ``'square'`` / ``'softplus'`` are
    the other admissible non-negativity maps.

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
        p: exponent of the CDF-difference norm.
        trans: positivity transform (``'linear'``/``'abs'``/``'square'``/``'softplus'``).
        theta: scale parameter of the positivity transform.
        eps: floor guarding the mass normalization.
    """

    p: float = 2.0
    trans: str = "linear"
    theta: float = 1.0
    eps: float = 1e-18

    def _positivity(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.trans == "linear":
            return x + c
        if self.trans == "abs":
            return x.abs()
        if self.trans == "square":
            return x * x
        if self.trans == "softplus":
            return F.softplus(x)
        raise GeoBrainError(
            "NormalizedIntegrationWaveform trans must be "
            "'linear' | 'abs' | 'square' | 'softplus'",
            object_name="NormalizedIntegrationWaveform", field="trans",
            expected="linear|abs|square|softplus", actual=self.trans,
        )

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel,
                         "NormalizedIntegrationWaveform", normalize=self.normalize)
        if self.trans == "linear":
            m = torch.minimum(obs.detach().min(), syn.detach().min())
            c = (-m).clamp_min(0.0) * self.theta
        else:
            c = torch.zeros((), dtype=obs.dtype, device=obs.device)
        mu = self._positivity(syn, c)
        nu = self._positivity(obs, c)
        # normalize each trace to a probability over time (axis 1)
        mu = mu / mu.sum(dim=1, keepdim=True).clamp_min(self.eps)
        nu = nu / nu.sum(dim=1, keepdim=True).clamp_min(self.eps)
        cdf = torch.cumsum(mu, dim=1) - torch.cumsum(nu, dim=1)
        return (cdf.abs() ** self.p).sum()


@dataclass(frozen=True)
class TravelTimeWaveform(_WaveMisfitBase):
    """Cross-correlation travel-time-shift misfit ``Σ_{s,r} |Δt|·dt``.

    Each trace is peak-normalized, the obs/syn cross-correlation is formed
    (``conv1d``, full padding), and a differentiable soft-argmax
    ``Δt = Σ softmax(β·xcorr)·lag`` estimates the time shift (Luo & Schuster
    1991; the soft-argmax makes it autograd-friendly). Larger ``beta`` sharpens
    toward the hard argmax.

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
        beta: soft-argmax sharpness (larger = closer to hard argmax).
    """

    beta: float = 10.0

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "TravelTimeWaveform",
                         normalize=self.normalize)
        n_src, nt, n_rcv = obs.shape
        # peak-normalize each trace along time
        o = obs / obs.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        s = syn / syn.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        # batched full cross-correlation: conv1d(input=o, weight=s) per trace
        # rearrange to (n_traces, 1, nt)
        ot = o.permute(0, 2, 1).reshape(-1, 1, nt)          # (T, 1, nt)
        st = s.permute(0, 2, 1).reshape(-1, 1, nt)
        # grouped conv: each trace convolved with its own kernel
        T = ot.shape[0]
        xc = F.conv1d(
            ot.reshape(1, T, nt), st.reshape(T, 1, nt),
            padding=nt - 1, groups=T,
        ).reshape(T, 2 * nt - 1)
        lags = torch.arange(
            -nt + 1, nt, device=obs.device, dtype=obs.dtype
        )
        weights = torch.softmax(self.beta * xc, dim=1)
        shift = (weights * lags).sum(dim=1) * self.dt        # (T,)
        mask = _live_mask(obs, syn).reshape(-1)
        return (shift.abs() * mask).sum()


@dataclass(frozen=True)
class WassersteinWaveform(_WaveMisfitBase):
    """1-D optimal-transport (Wasserstein-p) misfit on trace distributions.

    Each trace is made a probability over time (a positivity transform then
    sum-normalization) and the exact 1-D Wasserstein-p distance between the
    obs/syn distributions on the shared time support is summed over traces
    (Engquist & Froese 2014; Métivier et al. 2016). ``p=1`` is the exact
    CDF-L1 form ``∫|F−G|·dt``; general ``p`` uses the quantile-difference
    integral (the standard sorted-support 1-D OT).

    This is the textbook 1-D OT computed in-tree, with no external OT
    dependency. ``trans`` selects the positivity map (traces must be turned
    into non-negative densities): ``'square'`` (default), ``'abs'``,
    ``'softplus'``, or ``'linear'`` (global-min shift).

    ``gradient_singular = True``: the general-``p`` quantile lookup routes
    through an integer ``searchsorted`` index, which detaches the gradient
    from the quantile location, a source of OT-flavoured singularities.

    Attributes:
        channel: ``ForwardOutput`` data key holding the gather.
        dt: time step used in the misfit's ``dt`` scaling.
        normalize: per-shot max-normalize both gathers before the misfit.
        p: Wasserstein order of the 1-D transport distance.
        trans: positivity transform applied to traces first.
        eps: floor guarding the mass normalization.
    """

    p: float = 2.0
    trans: str = "square"
    eps: float = 1e-18

    gradient_singular: ClassVar[bool] = True

    def _density(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.trans == "square":
            d = x * x
        elif self.trans == "abs":
            d = x.abs()
        elif self.trans == "softplus":
            d = F.softplus(x)
        elif self.trans == "linear":
            d = x + c
        else:
            raise GeoBrainError(
                "WassersteinWaveform trans must be "
                "'square' | 'abs' | 'softplus' | 'linear'",
                object_name="WassersteinWaveform", field="trans",
                expected="square|abs|softplus|linear", actual=self.trans,
            )
        return d / d.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def misfit(
        self,
        predicted: Mapping[str, torch.Tensor],
        observed: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        syn, obs = _pull(predicted, observed, self.channel, "WassersteinWaveform",
                         normalize=self.normalize)
        n_src, nt, n_rcv = obs.shape
        if self.trans == "linear":
            m = torch.minimum(obs.detach().min(), syn.detach().min())
            c = (-m).clamp_min(0.0)
        else:
            c = torch.zeros((), dtype=obs.dtype, device=obs.device)
        u = self._density(syn, c)                            # (n_src, nt, n_rcv)
        v = self._density(obs, c)
        x = torch.arange(nt, device=obs.device, dtype=obs.dtype) * self.dt
        # CDFs along time
        U = torch.cumsum(u, dim=1)
        V = torch.cumsum(v, dim=1)
        if self.p == 1:
            # exact: W_1 = ∫ |F − G| dx  (constant support spacing dt)
            return (U - V).abs().sum(dim=1).sum() * self.dt
        # general p: ∫_0^1 |Q_u(q) − Q_v(q)|^p dq via a merged-CDF quantile.
        # Merge the CDF breakpoints of u and v (per trace), evaluate quantiles.
        Uf = U.permute(0, 2, 1).reshape(-1, nt).contiguous()   # (T, nt)
        Vf = V.permute(0, 2, 1).reshape(-1, nt).contiguous()
        qs, _ = torch.sort(torch.cat([Uf, Vf], dim=1), dim=1)  # (T, 2nt)
        widths = torch.diff(qs, dim=1, prepend=torch.zeros_like(qs[:, :1]))
        mids = (qs - 0.5 * widths).clamp(max=1.0).contiguous()  # interval midpoints
        # quantile: first support index whose CDF >= level
        iu = torch.searchsorted(Uf, mids)
        iv = torch.searchsorted(Vf, mids)
        iu = iu.clamp(max=nt - 1)
        iv = iv.clamp(max=nt - 1)
        qu = x[iu]
        qv = x[iv]
        wp = (widths * (qu - qv).abs() ** self.p).sum(dim=1)   # (T,) = W_p^p
        return wp.sum()
