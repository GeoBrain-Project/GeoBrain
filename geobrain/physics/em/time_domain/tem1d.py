"""
1D central-loop TEM forward.

Central-loop transmitter over a 1D layered earth; observes ``dB_z/dt``
at the loop center as a function of time. Differentiability:
FULL_AUTOGRAD. Frequency-kernel time convention: ``e^{+iωt}``, the
``layered_te_reflection`` kernel uses ``+iωμ₀σ``, the SAME as MT1D / MT3D /
CSEM1D (they all agree; none conjugate their output). The opposite ``e^{-iωt}``
convention (used by several 1-D EM codes) flips the ``dB/dt`` sign (see below).

Source: the ``time_domain/tem1d/`` 7-file subpackage, collapsed
into this one file.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from geobrain.core import GeoBrainError

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from geobrain.physics.em.capabilities import EMOperatorDiscovery

from geobrain.core import (
    ForwardContext, ModelState, ForwardOperator, ForwardOutput,
)
from geobrain.core.differentiability import (
    DifferentiabilityLevel, DifferentiabilitySpec,
)
from geobrain.physics.em.conventions import MU_0
from geobrain.physics.em.surveys import (
    CentralLoopSource, EMReceiver, TimeDomainSurvey,
)
from geobrain.physics.em.numerics.layered import layered_te_reflection
from geobrain.physics.em.numerics.hankel.dlf import (
    dlf_sincos, load_hankel_filter,
)


@dataclass(frozen=True)
class TEM1DReceiver(EMReceiver):
    """
    TEM1D receiver. By convention coincident with loop center on surface.

    The geometry constraint is enforced by ``TEM1DSurvey.__post_init__``;
    this dataclass just holds a position.
    """
    pass


@dataclass(frozen=True)
class TEM1DSurvey(TimeDomainSurvey):
    """
    1D central-loop TEM survey (single source, single receiver, on-surface).

    Inherits ``sources``, ``receivers``, ``times``, ``waveform`` from
    ``TimeDomainSurvey``. ``__post_init__`` enforces:

        - exactly one source (must be a ``CentralLoopSource``)
        - exactly one receiver (must be a ``TEM1DReceiver``)
        - receiver xy position coincides with source xy
        - receiver z = 0 (on the surface)
        - times non-empty and strictly positive

    Attributes:
        sources / receivers: acquisition tables.
        times: gate times [s].
        waveform: transmitter :class:`TimeWaveform`.
    """

    def __post_init__(self) -> None:
        # Delegate the base abstractness check first
        super().__post_init__()

        if len(self.sources) != 1:
            raise GeoBrainError(
                f"TEM1DSurvey requires exactly one source, got {len(self.sources)}",
            )
        if len(self.receivers) != 1:
            raise GeoBrainError(
                f"TEM1DSurvey requires exactly one receiver, got {len(self.receivers)}",
            )

        src = self.sources[0]
        rcv = self.receivers[0]

        if not isinstance(src, CentralLoopSource):
            raise GeoBrainError(
                "TEM1DSurvey source must be CentralLoopSource, "
                f"got {type(src).__name__}",
            )
        if not isinstance(rcv, TEM1DReceiver):
            raise GeoBrainError(
                "TEM1DSurvey receiver must be TEM1DReceiver, "
                f"got {type(rcv).__name__}",
            )

        sx, sy, _sz = src.position
        rx, ry, rz = rcv.position
        if not (math.isclose(rx, sx, abs_tol=1e-9)
                and math.isclose(ry, sy, abs_tol=1e-9)):
            raise GeoBrainError(
                "TEM1DSurvey receiver xy must coincide with source xy "
                f"(got receiver=({rx}, {ry}), source=({sx}, {sy}))",
            )
        if not math.isclose(rz, 0.0, abs_tol=1e-9):
            raise GeoBrainError(
                f"TEM1DSurvey receiver must be on the surface (z=0), got z={rz}",
            )

        if len(self.times) == 0:
            raise GeoBrainError("TEM1DSurvey requires non-empty times")
        if any(t <= 0 for t in self.times):
            raise GeoBrainError("TEM1DSurvey times must be strictly positive")


# ---------------------------------------------------------------------------
# Private kernel (ported from the time_domain/tem1d/kernel.py)
# # Engineering ``e^{+iωt}`` time convention: wavenumber uses ``+1j ω μ₀ σ``
# (same complex sqrt as CSEM1D's port). The dB/dt formula carries a leading
# ``-(2/π) · Im[...]`` factor, which is what realizes the convention sign
# (the opposite ``e^{-iωt}`` convention flips the dB/dt sign).
#
# The layered TE reflection is computed
# with broadcasting over arbitrary ``(lam, omega)`` shapes rather than the
# fixed ``(n_freq, n_lambda)`` layout CSEM1D's port uses. This lets us nest
# the J1 Hankel inside the cosine sin/cos DLF without materializing a
# ``(n_t, n_filter_sc, n_filter_h)`` intermediate via two separate calls.
# ---------------------------------------------------------------------------




def _hz_secondary_at_loop_center(
    omega: torch.Tensor,
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    loop_radius: float,
    moment: float,
    *,
    hankel_filter: str = "key_201_2012",
) -> torch.Tensor:
    """
    Secondary vertical H-field at the center of a circular loop.

    Evaluates

        H_z^(s)(ω) = (I a / 2)
            · ∫₀^∞ r_TE(λ, ω) J₁(λ a) dλ

    via the J1 DLF in pure torch, jointly across all entries of
    ``omega``. The Hankel filter nodes ``λ_k = base[k] / a`` are
    independent of ``omega`` (single fixed offset ``a``), so the kernel
    is built once and broadcast against ``omega`` for batched
    evaluation.

    Args:
        omega: Real torch tensor of angular frequencies; any shape.
        sigma, thickness: Layered earth (1D torch tensors).
        loop_radius: Loop radius ``a`` in metres.
        moment: Loop current ``I`` in amperes.
        hankel_filter: Filter name; default is the shipped Key-2012 201-point set.

    Returns:
        Complex128 tensor with the same shape as ``omega``.
    """
    a = float(loop_radius)
    I_current = float(moment)

    filter_dict = load_hankel_filter(hankel_filter)
    # Device-preserving: the Hankel filter coefficients ship on CPU; move them
    # onto the input (omega) device so the layered TE kernel, which mixes them
    # with the on-device sigma/omega: runs on cuda inputs too. ``.to`` is a
    # no-op when omega is on CPU. Mirrors the dlf_hankel / dlf_sincos pattern
    # (numerics/hankel/dlf.py) that already keeps CSEM1D device-preserving.
    base = filter_dict["base"].to(omega.device)   # (n_filter,)
    j1_w = filter_dict["j1"].to(omega.device)     # (n_filter,)

    lam_nodes = base / a                          # (n_filter,)
    # Broadcast omega to (..., 1) so r_TE comes back as (..., n_filter).
    r_te = layered_te_reflection(
        lam_nodes, omega.unsqueeze(-1), sigma, thickness, safe_tanh=True,
    )
    weights_c = j1_w.to(torch.complex128)
    # Ward & Hohmann eq. 4.81: integrand carries an explicit ``λ`` factor,
    # i.e. ∫ r_TE(λ) · λ · J₁(λa) dλ. The bare DLF identity is then
    # ∫ f(λ) J₁(λa) dλ ≈ (1/a) Σ f(λ_k) j1_w[k] with f(λ_k) = r_TE(λ_k) · λ_k.
    integral = (r_te * lam_nodes * weights_c).sum(dim=-1) / a   # (...,) complex
    return (I_current * a / 2.0) * integral


def _dbdt_central_loop(
    times: torch.Tensor,
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    loop_radius: float,
    moment: float,
    *,
    hankel_filter: str = "key_201_2012",
    sincos_filter: str = "sincos_201",
) -> torch.Tensor:
    """
    Step-off impulse response ``dB_z/dt(t)`` at the loop center.

    Computes::

        f(ω)       = Im[μ₀ · H_z^(s)(ω)]
        dB_z/dt(t) = -(2/π) · ∫₀^∞ f(ω) cos(ω t) dω

    via a J1-Hankel (over the layered TE response) nested inside a
    cosine sin/cos DLF. Both stages are pure-torch on the cached filter
    constants, so ``loss.backward()`` populates ``sigma.grad`` and
    ``thickness.grad`` automatically with no custom backward.

    The leading ``-(2/π)`` factor (negative, not positive) is what
    implements the ``e^{+iωt}`` engineering convention; the opposite
    ``e^{-iωt}`` convention flips the sign.

    Args:
        times: 1D ``torch.float64`` tensor of strictly-positive observation
            times (s).
        sigma, thickness: Layered earth (1D torch tensors).
        loop_radius: Loop radius ``a`` in metres.
        moment: Loop current ``I`` in amperes.
        hankel_filter, sincos_filter: Filter names; defaults are the shipped Key-2012 201-point sets.

    Returns:
        Real ``(n_times,)`` torch tensor of ``dB_z/dt(t)`` values in
        V / (A · m²), inheriting the autograd graph of ``sigma`` /
        ``thickness``.
    """
    def integrand(omega_grid: torch.Tensor) -> torch.Tensor:
        # ``omega_grid`` arrives shaped (n_times, n_filter_sc).
        Hz_s = _hz_secondary_at_loop_center(
            omega_grid, sigma, thickness, loop_radius, moment,
            hankel_filter=hankel_filter,
        )                                          # (n_times, n_filter_sc)
        return (MU_0 * Hz_s).real.to(torch.float64)

    F = dlf_sincos(integrand, times, kind="cos", filter_name=sincos_filter)
    return -(2.0 / math.pi) * F


class TEM1D(EMOperatorDiscovery, ForwardOperator):
    """1D central-loop TEM forward: dB_z/dt at loop center.

    Reads ``sigma`` (n_layer,) and ``thickness`` (n_layer-1,) from
    ``ModelState``. Returns ``ForwardOutput.data["dbdt_z"]`` real tensor
    of shape ``(n_time_gates,)`` in V/(A·m²).

    Time convention: ``e^{+iωt}``, the SAME as CSEM1D / MT1D / MT3D (the
    ``layered_te_reflection`` kernel uses ``+iωμ₀σ``; none conjugate their
    output; see the module docstring). Differentiability: ``FULL_AUTOGRAD``.

    Args:
        survey: central-loop TEM sounding acquisition.
        hankel_filter: digital Hankel filter id.
        sincos_filter: sine/cosine transform filter id.
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("sigma", "thickness"),
        output_keys=("dbdt_z",),
    )

    def __init__(
        self,
        survey: TEM1DSurvey,
        *,
        hankel_filter: str = "key_201_2012",
        sincos_filter: str = "sincos_201",
    ) -> None:
        super().__init__()
        if not isinstance(survey, TEM1DSurvey):
            raise GeoBrainError(
                f"TEM1D survey must be TEM1DSurvey, got {type(survey).__name__}",
            )
        self.survey = survey
        self.hankel_filter = hankel_filter
        self.sincos_filter = sincos_filter

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        sigma, thickness = state.fetch("sigma", "thickness")

        src = self.survey.sources[0]
        # src is CentralLoopSource (validated at survey __post_init__)
        loop_radius = float(src.radius_m)
        moment = float(src.current_a) * int(src.turns)

        times = torch.tensor(self.survey.times, dtype=torch.float64,
                              device=sigma.device)

        dbdt_z = _dbdt_central_loop(
            times=times,
            sigma=sigma.to(torch.float64),
            thickness=thickness.to(torch.float64),
            loop_radius=loop_radius,
            moment=moment,
            hankel_filter=self.hankel_filter,
            sincos_filter=self.sincos_filter,
        )

        # The DLF math runs in float64 for accuracy; return the caller's dtype
        # (do not silently upcast a float32 sigma to a float64 output). ``.to``
        # is a no-op when sigma is already float64.
        if dbdt_z.dtype != sigma.dtype:
            dbdt_z = dbdt_z.to(sigma.dtype)

        return ForwardOutput(data={"dbdt_z": dbdt_z})


class WaveformTEM1D(EMOperatorDiscovery, ForwardOperator):
    """1D central-loop TEM with an arbitrary transmitter waveform.

    The step-off response :func:`_dbdt_central_loop` is the system's response to
    an instantaneous current switch-off. For a finite transmitter waveform the
    measured ``dB_z/dt`` follows by Duhamel superposition (the diffusion is LTI
    at fixed σ):

        V(t) = −∫ (dI/dτ) s(t − τ) dτ
             ≈ −Σ_j ΔI_j · s(t − τ_mid,j)

    with ``s`` the step-off response, ``ΔI_j`` the (normalised) current change
    over waveform segment ``j`` and ``τ_mid,j`` its midpoint. A pure step-off
    waveform reproduces :class:`TEM1D` exactly; a finite ramp-off averages the
    step response over the ramp (the physical early-time reduction real
    VTEM/SkyTEM/ground-loop systems show).

    Differentiable (``FULL_AUTOGRAD``) through ``sigma``/``thickness`` (via the
    step response) **and** through ``waveform_currents``, so the transmitter
    waveform itself is an autograd / inversion axis.

    Args:
        survey: ``TEM1DSurvey`` whose ``times`` are the measurement gates.
        waveform_times: ``(M,)`` ascending segment times ≤ 0 (current is on for
            ``t < 0`` and switches off by ``t = 0``).
        waveform_currents: ``(M,)`` *normalised* transmitter current (e.g. 1 → 0);
            the physical scale lives in the loop ``moment``. May require grad.
    """

    differentiability: ClassVar[DifferentiabilitySpec] = DifferentiabilitySpec(
        level=DifferentiabilityLevel.FULL_AUTOGRAD,
        trainable_inputs=("sigma", "thickness"),
        output_keys=("dbdt_z",),
    )

    def __init__(
        self,
        survey: TEM1DSurvey,
        waveform_times: torch.Tensor,
        waveform_currents: torch.Tensor,
        *,
        hankel_filter: str = "key_201_2012",
        sincos_filter: str = "sincos_201",
    ) -> None:
        super().__init__()
        if not isinstance(survey, TEM1DSurvey):
            raise GeoBrainError(
                f"WaveformTEM1D survey must be TEM1DSurvey, got {type(survey).__name__}",
            )
        wt = torch.as_tensor(waveform_times, dtype=torch.float64)
        wc = torch.as_tensor(waveform_currents, dtype=torch.float64)
        if wt.ndim != 1 or wc.ndim != 1 or wt.numel() != wc.numel() or wt.numel() < 2:
            raise GeoBrainError(
                "waveform_times / waveform_currents must be 1D, equal length ≥ 2",
            )
        self.survey = survey
        self.waveform_times = wt
        self.waveform_currents = wc
        self.hankel_filter = hankel_filter
        self.sincos_filter = sincos_filter

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        sigma, thickness = state.fetch("sigma", "thickness")
        src = self.survey.sources[0]
        loop_radius = float(src.radius_m)
        moment = float(src.current_a) * int(src.turns)

        gates = torch.tensor(self.survey.times, dtype=torch.float64, device=sigma.device)
        wt = self.waveform_times.to(sigma.device)
        wc = self.waveform_currents.to(sigma.device)

        tau_mid = 0.5 * (wt[1:] + wt[:-1])          # (M-1,) segment midpoints
        dI = wc[1:] - wc[:-1]                        # (M-1,) current change / segment

        # convolution times u[g, j] = gate_g − τ_mid,j (all > 0 since τ_mid ≤ 0)
        u = gates.unsqueeze(1) - tau_mid.unsqueeze(0)        # (G, M-1)
        s = _dbdt_central_loop(
            times=u.reshape(-1),
            sigma=sigma.to(torch.float64),
            thickness=thickness.to(torch.float64),
            loop_radius=loop_radius,
            moment=moment,
            hankel_filter=self.hankel_filter,
            sincos_filter=self.sincos_filter,
        ).reshape(u.shape)                                   # (G, M-1)

        dbdt_z = -(dI.unsqueeze(0) * s).sum(dim=1)           # (G,) Duhamel sum
        # Preserve the caller's input dtype (the DLF math is float64 internally);
        # ``.to`` is a no-op when sigma is already float64.
        if dbdt_z.dtype != sigma.dtype:
            dbdt_z = dbdt_z.to(sigma.dtype)
        return ForwardOutput(data={"dbdt_z": dbdt_z})
