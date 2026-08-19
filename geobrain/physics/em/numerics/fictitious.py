"""
Fictitious-wave-domain EM: the diffusive 1-D MT problem by wave propagation.

The correspondence principle (Mittet, GEOPHYSICS 75(1), 2010; the libEMM
lineage): the quasi-static diffusive equation ``d²E/dz² = iωμ₀σE`` maps onto
a WAVE equation ``∂²E'/∂t'² = c² ∂²E'/∂z²`` with the fictitious velocity

    c(z) = sqrt(2 ω₀ / (μ₀ σ(z))),

under the complex-frequency substitution ``ω'² = −2 i ω ω₀``, i.e. the
diffusive response at ω is the wave response evaluated at
``ω' = sqrt(ω ω₀) · (1 − i)`` (decaying branch). The payoff in higher
dimensions is replacing per-frequency sparse solves with ONE explicit
time-stepping run; this module delivers the validated 1-D core:

- explicit staggered leapfrog of the fictitious wave equation (pure torch,
  autograd flows through the time loop, so ``dZ/dσ`` works),
- on-the-fly DFT of the surface fields at each ``ω'_k`` (the ``e^{−iω't}``
  kernel decays, which self-truncates the record),
- the exact impedance map back to the diffusive domain. The transform factor
  is fixed analytically by uniform-medium exactness: with the simulated
  wave-domain ratio ``R(ω') = E'(ω') / H'(ω')`` (``H' = v/μ₀``; for a uniform
  downgoing wave ``R = μ₀ c``), requiring ``Z = sqrt(iωμ₀/σ)`` gives

      Z(ω) = C(ω) · R(ω'),   C(ω) = sqrt(i ω / (2 ω₀)),

  independent of σ, so the SAME factor maps the full layered transfer
  function (that is the correspondence principle).

Gate: :func:`~geobrain.physics.em.numerics.layered.mt1d_wait_impedance`
(Wait's recursion, the platform's analytic ``e^{+iωt}`` reference).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math

import torch

from geobrain.core import GeoBrainError

from .layered import MU_0

__all__ = ["fictitious_wave_impedance_1d"]


def fictitious_wave_impedance_1d(
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    omega0: float | None = None,
    points_per_wavelength: int = 40,
    mu0: float = MU_0,
) -> torch.Tensor:
    """Surface MT impedance of a 1-D layered earth via the fictitious wave domain.

    Args:
        sigma: ``(n_layer,)`` conductivities (S/m), top → bottom; the last
            layer is the basement half-space. Differentiable input.
        thickness: ``(n_layer - 1,)`` thicknesses (m).
        frequencies: ``(n_freq,)`` frequencies (Hz).
        omega0: Fictitious angular scaling frequency. ``None`` picks
            ``2π · max(frequencies)`` (maps the band to comfortable wave
            frequencies ``f'_k = sqrt(f_k · f_max)``).
        points_per_wavelength: Spatial resolution of the shortest mapped
            wavelength (also capped by layer thickness / 10).
        mu0: Vacuum permeability.

    Returns:
        ``(n_freq,)`` complex128 surface impedance in the ``e^{+iωt}`` family
        (half-space phase +45°), matching ``mt1d_wait_impedance``.
    """
    if sigma.ndim != 1 or sigma.numel() == 0:
        raise GeoBrainError(
            "fictitious_wave_impedance_1d sigma must be a non-empty 1-D tensor",
            object_name="fictitious_wave_impedance_1d",
            field="sigma",
            expected="(n_layer,)",
            actual=tuple(sigma.shape),
        )
    if not bool((sigma.detach() > 0).all()):
        raise GeoBrainError(
            "fictitious_wave_impedance_1d sigma must be strictly positive",
            object_name="fictitious_wave_impedance_1d",
            field="sigma",
            expected="> 0",
            actual=sigma.detach().tolist(),
        )
    if thickness.numel() != max(0, sigma.numel() - 1):
        raise GeoBrainError(
            "thickness must have n_layer - 1 entries",
            object_name="fictitious_wave_impedance_1d",
            field="thickness",
            expected=max(0, sigma.numel() - 1),
            actual=thickness.numel(),
        )
    freqs = frequencies.to(torch.float64)
    omega = 2.0 * math.pi * freqs
    w0 = 2.0 * math.pi * float(freqs.max()) if omega0 is None else float(omega0)

    sig = sigma.to(torch.float64)
    c_layers = torch.sqrt(2.0 * w0 / (mu0 * sig))            # (n_layer,)
    c_min = float(c_layers.detach().min())
    c_max = float(c_layers.detach().max())

    # Mapped wave-domain band: ω'_real = sqrt(ω ω₀).
    wprime_real = torch.sqrt(omega * w0)                      # (n_freq,)
    f_prime_max = float(wprime_real.max()) / (2.0 * math.pi)
    f_prime_min = float(wprime_real.min()) / (2.0 * math.pi)

    # Grid: resolve the shortest mapped wavelength AND the thinnest layer.
    # The layer cap is /40 (not /10): the staggered interface lands between
    # an E and a v node, so the effective thickness carries an O(dz/2) bias;
    # at /10 that is a systematic ~2.5% impedance error, at /40 it is ~0.6%.
    dz = c_min / (f_prime_max * points_per_wavelength)
    if thickness.numel():
        dz = min(dz, float(thickness.detach().min()) / 40.0)

    # Record long enough for the e^{−sqrt(ω ω₀) t} kernel to die (8 e-folds)
    # and for several periods of the slowest mapped component.
    t_end = max(8.0 / float(wprime_real.min()), 4.0 / f_prime_min)
    dt = 0.4 * dz / c_max
    nt = int(math.ceil(t_end / dt))

    # Geometry: [top pad (layer-1 medium) | SOURCE a few nodes above |
    # RECEIVER at the stack surface | layers | basement pad]. Each pad is
    # sized with ITS OWN region's velocity so no boundary reflection reaches
    # the receiver inside the record: the top medium travels at c_layer1,
    # not c_max (sizing the top pad by c_max starves a slow first layer: the
    # wave never arrives and the transfer ratio degenerates to 0/0).
    c_top = float(c_layers.detach()[0])
    c_bot = float(c_layers.detach()[-1])
    gap = 8                                                   # source→receiver
    n_top = int(math.ceil(c_top * t_end * 0.55 / dz)) + gap + 8
    j_rcv = n_top                                             # stack surface
    j_src = j_rcv - gap
    seg_nodes = [
        max(1, int(round(float(h) / dz))) for h in thickness.detach()
    ]
    n_bot = int(math.ceil(c_bot * t_end * 0.55 / dz)) + 8
    n_nodes = n_top + sum(seg_nodes) + n_bot

    # Per-node c² profile (differentiable in sigma).
    c2_nodes = torch.empty(n_nodes, dtype=torch.float64)
    c2_layers = 2.0 * w0 / (mu0 * sig)
    pieces = [c2_layers[0].expand(n_top)]
    for i, n_seg in enumerate(seg_nodes):
        pieces.append(c2_layers[i].expand(n_seg))
    pieces.append(c2_layers[-1].expand(n_bot))
    c2_nodes = torch.cat(pieces)

    # Source wavelet: first-derivative-of-Gaussian centred at the top of the
    # mapped band (broadband over [f'_min, f'_max]).
    f_src = f_prime_max
    t = torch.arange(nt, dtype=torch.float64) * dt
    t0 = 1.5 / f_src
    tau = (t - t0) * (2.0 * math.pi * f_src / 2.0)
    wavelet = -tau * torch.exp(-0.5 * tau * tau)

    # On-the-fly DFT kernels at ω' = sqrt(ω ω₀)(1 − i): e^{−iω't} decays.
    wprime = wprime_real.to(torch.complex128) * (1.0 - 1.0j)
    kern = torch.exp(-1.0j * wprime.unsqueeze(1) * t.unsqueeze(0)) * dt  # (nf, nt)

    # Leapfrog: v lives on half nodes (n_nodes-1), E on nodes.
    E = torch.zeros(n_nodes, dtype=torch.float64)
    v = torch.zeros(n_nodes - 1, dtype=torch.float64)
    E_rec = torch.empty(nt, dtype=torch.float64)
    v_rec = torch.empty(nt, dtype=torch.float64)
    r = dt / dz
    c2_int = c2_nodes[1:-1]
    for it in range(nt):
        v = v + r * (E[1:] - E[:-1])
        E = torch.cat(
            [E[:1], E[1:-1] + r * c2_int * (v[1:] - v[:-1]), E[-1:]]
        )
        E = E.clone()
        E[j_src] = E[j_src] + wavelet[it]
        E_rec[it] = E[j_rcv]
        v_rec[it] = 0.5 * (v[j_rcv - 1] + v[j_rcv])

    # v holds ∫ ∂z E dt at half steps; H' = v / μ₀ up to the leapfrog's
    # half-step stagger (absorbed by the analytic calibration below at the
    # resolutions this routine chooses).
    E_hat = (kern * E_rec.to(torch.complex128)).sum(dim=1)    # (n_freq,)
    v_hat = (kern * v_rec.to(torch.complex128)).sum(dim=1)
    # Exact leapfrog-stagger correction: inside iteration ``it`` the recorded
    # E is E^{it+1} (time (it+1)·dt) while v is v^{it+1/2} (time (it+1/2)·dt);
    # both were DFT'd with the it·dt kernel, so the ratio carries a spurious
    # e^{+iω' dt/2}: remove it. (At f'_max this is a ~5% magnitude / ~3°
    # phase bias; the residual O((ω'dt)²) dispersion is <0.1% here.)
    R = mu0 * (E_hat / v_hat) * torch.exp(-0.5j * wprime * dt)  # E'/H'

    # Exact diffusive map, calibrated by uniform-medium exactness:
    # Z(ω) = sqrt(iω/(2ω₀)) · R(ω'). Branch: e^{+iωt} family (phase +45°).
    C = torch.sqrt(1.0j * omega.to(torch.complex128) / (2.0 * w0))
    Z = C * R
    # Sign convention of the staggered v-update (v ~ +∫∂zE) puts the
    # downgoing-wave ratio at −μ₀c; fold the sign so the uniform half-space
    # lands on +sqrt(iωμ₀/σ) (the Wait/e^{+iωt} branch, asserted by the gate).
    if float(Z.real.detach().sum()) < 0.0:
        Z = -Z
    return Z
