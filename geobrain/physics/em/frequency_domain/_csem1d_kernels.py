"""CSEM1D Hankel-domain kernels: TE reflection recursion, scale-safe
tanh-thickness jets, dimensionless integrands and the VMD primary
(split from csem1d.py).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

from __future__ import annotations
import math
import torch
from geobrain.physics.em.conventions import MU_0
from geobrain.physics.em.errors import (
    EMContractError,
    EMNumericsError,
)
from geobrain.physics.em.numerics.layered import layered_te_reflection

# Kernel-tuning constants (moved with their consumers from csem1d.py).
_LOW_INDUCTION_RESCALE_LOG = math.log(1.0e-100)
_TANH_ANALYTIC_GRADIENT_REAL = 4.0
_TANH_ASYMPTOTIC_REAL = 16.0


def _integrand_bz(
    lam: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    source_z: float,
    receiver_z: float,
    surface_z_m: float = 0.0,
    mu0: float = MU_0,
) -> torch.Tensor:
    """
    VMD ``H_z`` J0-Hankel SECONDARY integrand on a layered earth.

    Returns the layered-earth reflected (secondary) part only

    ``f_sec(λ) = r_TE(λ, ω) * exp(λ (z_r + z_s)) * λ²``

    such that the air-domain vertical H-field secondary is

    ``H_z^sec(r) = (m / 4π) * ∫₀^∞ f_sec(λ) J_0(λ r) dλ``.

    The free-space primary contribution

    ``∫₀^∞ exp(-λ |z_r - z_s|) λ² J_0(λ r) dλ = (2 Δz² - r²) / R⁵``

    (with ``Δz = z_r - z_s``, ``R = √(r² + Δz²)``) is now evaluated
    in closed form by :func:`_vmd_primary_bz` at the operator level
    rather than handed to the digital linear filter; this eliminates
    the DLF accuracy floor for primary-dominated configurations
    (e.g. airborne HCP / HEM). ``B_z = μ₀ H_z`` is applied by the
    operator alongside the source moment and ``1/(4π)``.

    Both ``source_z`` and ``receiver_z`` are at or above ``surface_z_m`` in
    the public positive-down coordinate frame.

    Shapes:
    lam        : ``(n_lambda,)``        real
    omega      : ``(n_freq,)``          real
    sigma      : ``(n_layers,)``        real
    thickness  : ``(n_layers - 1,)``    real
    return     : ``(n_freq, n_lambda)`` complex128
    """
    lam_c = lam.to(torch.complex128).reshape(1, -1)  # (1, n_l)
    r_te = layered_te_reflection(
        lam.reshape(1, -1),
        omega.reshape(-1, 1),
        sigma,
        thickness,
        mu0,
    )
    # In air ``u_0 = λ``. Shift both public z coordinates to the recorded
    # surface before applying the upward-decaying reflected propagator.
    air_path = receiver_z + source_z - 2.0 * surface_z_m
    reflected = r_te * torch.exp(lam_c * air_path)
    return reflected * lam_c**2


def _finite_complex_polar_from_log(
    log_magnitude: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Return a finite complex value from log-magnitude and unit phase.

    The propagation-tail derivative may be representable only after its
    dimensional ``1/L`` factor is included.  Forming the unscaled exponential
    first would lose it, while directly exponentiating an inactive extreme
    branch could overflow.  Values outside binary64 are therefore represented
    by zero; selected representable branches retain normal and subnormal
    magnitudes.
    """
    log_max = math.log(torch.finfo(torch.float64).max)
    log_min = math.log(math.ulp(0.0))
    finite = torch.isfinite(log_magnitude)
    representable = finite & (log_magnitude >= log_min) & (log_magnitude <= log_max)
    safe_log_magnitude = torch.where(
        finite,
        torch.clamp(log_magnitude, min=log_min, max=log_max),
        torch.zeros_like(log_magnitude),
    )
    magnitude = torch.where(
        representable,
        torch.exp(safe_log_magnitude),
        torch.zeros_like(log_magnitude),
    )
    return phase * magnitude.to(torch.complex128)


def _scale_safe_tanh_thickness_jet(
    local: torch.Tensor,
    log_argument_magnitude: torch.Tensor,
    layer_thickness: torch.Tensor,
    finite_argument: torch.Tensor,
    asymptotic_tail: torch.Tensor,
    asymptotic_tanh: torch.Tensor,
    replace_thickness_graph: torch.Tensor,
    *,
    length_scale_m: float,
) -> torch.Tensor:
    """Replace quantized tanh thickness derivatives by an analytic local jet.

    Complex ``exp`` reverse mode multiplies its subnormal output before the
    later dimensional ``1/L`` chain factor.  Near the normal/subnormal
    boundary that ordering quantizes or erases a derivative which is still
    representable in physical metres.  Forward mode applies the factors in
    the opposite order and does not suffer that loss.

    On the analytic asymptotic branch, the ordinary thickness graph is frozen
    at the local anchor and replaced by a zero-primal second-order carrier.
    Its first and second derivatives are the analytic complex derivatives

    ``sech(p*h/L)^2 * p/L`` and
    ``-2*tanh(p*h/L)*sech(p*h/L)^2 * (p/L)^2``.

    The ``exp(-2*p*h/L)`` magnitude is combined with ``1/L`` or ``1/L**2``
    before exponentiation, so representable SI derivatives survive even when
    the dimensionless tail is coarsely quantized or rounds to zero.  The direct
    nonsaturated branch retains the ordinary graph.  The anchored base retains
    the local propagation graph, so only the ill-conditioned
    physical-thickness edge is replaced and its derivative is not counted
    twice.
    """
    local_anchor = local.detach()
    local_magnitude = torch.abs(local)
    # Freeze only h/L.  Rebuilding the argument log and phase from the active
    # propagation constant retains conductivity and induction derivatives in
    # the scale-safe carrier, including representable mixed second derivatives.
    log_thickness_ratio_anchor = (log_argument_magnitude - torch.log(local_magnitude)).detach()
    log_argument_anchor = torch.log(local_magnitude) + log_thickness_ratio_anchor
    thickness_delta = layer_thickness - layer_thickness.detach()

    local_phase = local / local_magnitude.to(torch.complex128)
    # Reconstruct the true argument rather than the finite forward surrogate,
    # which is capped at 700 solely to protect inactive tanh branches.  If the
    # true magnitude itself exceeds 1e300, every first/second binary64 jet is
    # necessarily zero even after the largest finite dimensional rate.
    reconstructable = log_argument_anchor <= math.log(1.0e300)
    argument_magnitude = torch.exp(torch.clamp(log_argument_anchor, max=math.log(1.0e300)))
    argument_anchor = local_phase * argument_magnitude.to(torch.complex128)
    argument_real = torch.where(
        reconstructable,
        argument_anchor.real,
        torch.full_like(argument_anchor.real, math.inf),
    )
    log_rate_magnitude = torch.log(local_magnitude) - math.log(length_scale_m)
    tail_log_magnitude = -2.0 * argument_real
    first_log_magnitude = math.log(4.0) + log_rate_magnitude + tail_log_magnitude
    second_log_magnitude = math.log(8.0) + 2.0 * log_rate_magnitude + tail_log_magnitude
    log_min = math.log(math.ulp(0.0))
    log_max = math.log(torch.finfo(torch.float64).max)
    phase_is_needed = reconstructable & (
        (
            torch.isfinite(first_log_magnitude)
            & (first_log_magnitude >= log_min)
            & (first_log_magnitude <= log_max)
        )
        | (
            torch.isfinite(second_log_magnitude)
            & (second_log_magnitude >= log_min)
            & (second_log_magnitude <= log_max)
        )
    )
    safe_argument_imag = torch.where(
        phase_is_needed,
        argument_anchor.imag,
        torch.zeros_like(argument_anchor.imag),
    )
    tail_phase = torch.polar(torch.ones_like(safe_argument_imag), -2.0 * safe_argument_imag)

    analytic_tail = _finite_complex_polar_from_log(tail_log_magnitude, tail_phase)
    tail_denominator = 1.0 + analytic_tail

    first = _finite_complex_polar_from_log(
        first_log_magnitude,
        local_phase * tail_phase,
    ) / (tail_denominator * tail_denominator)
    second = (
        _finite_complex_polar_from_log(
            second_log_magnitude,
            -(local_phase * local_phase) * tail_phase,
        )
        * (1.0 - analytic_tail)
        / (tail_denominator * tail_denominator * tail_denominator)
    )
    carrier = first * thickness_delta + 0.5 * second * thickness_delta * thickness_delta

    # Keep propagation-constant derivatives while removing only the thickness
    # edge from the asymptotic branch.  The straight-through anchor preserves
    # the exact ordinary forward value.
    argument_per_local = (finite_argument.detach() / local_anchor).detach()
    anchored_argument = local * argument_per_local
    anchored_tail = torch.exp(-2.0 * anchored_argument)
    anchored_tanh_graph = (1.0 - anchored_tail) / (1.0 + anchored_tail)
    anchored_tanh = asymptotic_tanh.detach() + anchored_tanh_graph - anchored_tanh_graph.detach()
    return torch.where(
        replace_thickness_graph,
        anchored_tanh + carrier,
        asymptotic_tanh,
    )


def _dimensionless_te_reflection(
    q: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    *,
    length_scale_m: float,
    log_induction_shift: torch.Tensor | None = None,
    mu0: float = MU_0,
) -> torch.Tensor:
    """Stable layered TE reflection in ``q = lambda * length_scale_m``.

    The recursion carries ``delta = p_eff - q`` rather than subtracting two
    nearly equal admittances at low induction. This keeps zero-contrast layers
    and very small induction numbers differentiable without manufacturing a
    roundoff tail that a relative quadrature criterion mistakes for physics.
    """
    log_length = math.log(length_scale_m)
    log_alpha = (
        torch.log(omega).reshape(-1, 1)
        + math.log(mu0)
        + torch.log(sigma).reshape(1, -1)
        + 2.0 * log_length
    )
    if log_induction_shift is None:
        log_induction_shift = torch.zeros(
            (omega.numel(), 1),
            dtype=torch.float64,
            device=omega.device,
        )
    shifted_log_alpha = log_alpha + log_induction_shift
    log_max = math.log(torch.finfo(torch.float64).max)
    log_min = math.log(math.ulp(0.0))
    if (
        not bool(torch.isfinite(log_alpha).all())
        or bool((log_alpha > log_max).any())
        or not bool(torch.isfinite(shifted_log_alpha).all())
        or bool((shifted_log_alpha > log_max).any())
        or bool((shifted_log_alpha < log_min).any())
    ):
        raise EMNumericsError(
            "CSEM1D dimensionless induction number is not representable",
            details={
                "operator": "CSEM1D",
                "stage": "dimensionless_induction",
                "length_scale_m": str(length_scale_m),
            },
            object_name="CSEM1D",
            field="sigma",
        )
    # Form alpha from a detached conductivity scale rather than differentiating
    # through log(sigma) -> exp(log_alpha).  The latter has a finite analytic
    # derivative but can create an overflowing exp-backward intermediate when
    # both sigma and alpha are extreme.
    sigma_scale = torch.max(sigma.detach())
    normalized_sigma = sigma / sigma_scale
    log_alpha_scale = (
        torch.log(omega).reshape(-1, 1)
        + math.log(mu0)
        + torch.log(sigma_scale)
        + 2.0 * log_length
        + log_induction_shift
    )
    alpha = torch.exp(log_alpha_scale) * normalized_sigma.reshape(1, -1)

    q_complex = q.to(torch.complex128).reshape(1, -1)
    q_squared = q_complex * q_complex
    propagation: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    for layer in range(int(sigma.numel())):
        induction = 1j * alpha[:, layer].reshape(-1, 1)
        local = torch.sqrt(q_squared + induction)
        propagation.append(local)
        deltas.append(induction / (local + q_complex))

    effective_delta = deltas[-1]
    if len(propagation) > 1:
        log_thickness_ratio = torch.log(thickness) - log_length
        for layer in range(len(propagation) - 2, -1, -1):
            local = propagation[layer]
            local_delta = deltas[layer]
            local_magnitude = torch.abs(local)
            safe_local_magnitude = local_magnitude.clamp_min(torch.finfo(torch.float64).tiny)
            log_argument_magnitude = torch.log(safe_local_magnitude) + log_thickness_ratio[layer]
            finite_argument_magnitude = torch.exp(
                torch.clamp(
                    log_argument_magnitude,
                    min=log_min,
                    max=math.log(700.0),
                )
            )
            finite_argument = (
                local / safe_local_magnitude.to(torch.complex128)
            ) * finite_argument_magnitude.to(torch.complex128)
            # ``torch.tanh`` rounds to exactly one once Re(p*h) is only a few
            # tens, and its backward then returns an exact zero.  That is not
            # safe here: the dimensionless derivative can remain representable
            # near Re(p*h)=350 and the final L**-3 field scale can amplify it
            # by hundreds of decades.  The positive-real propagation branch
            # permits the non-overflowing exp(-2*z) form.  The analytic jet
            # below carries first/second physical-thickness derivatives in an
            # order that remains valid when only the later SI scale makes a
            # quantized or underflowed tail derivative representable.
            asymptotic_tail = torch.exp(-2.0 * finite_argument)
            asymptotic_tanh = (1.0 - asymptotic_tail) / (1.0 + asymptotic_tail)
            use_asymptotic = finite_argument.real > _TANH_ASYMPTOTIC_REAL
            asymptotic_tanh = _scale_safe_tanh_thickness_jet(
                local,
                log_argument_magnitude,
                thickness[layer],
                finite_argument,
                asymptotic_tail,
                asymptotic_tanh,
                use_asymptotic,
                length_scale_m=length_scale_m,
            )
            direct_argument = torch.where(
                use_asymptotic,
                torch.zeros_like(finite_argument),
                finite_argument,
            )
            direct_tanh = torch.tanh(direct_argument)
            # ``torch.tanh`` is the most accurate forward formula below the
            # asymptotic cutoff, but its backward computes ``1 - tanh(z)**2``.
            # That subtraction already loses thickness-derivative precision
            # well before the forward value saturates (about 1e-4 relatively
            # at Re(z)=16).  From Re(z)>4, retain the exact direct forward bits
            # while taking derivatives through the cancellation-free
            # ``exp(-2*z)`` graph.  At the earlier handoff the two derivative
            # formulas agree to binary64 roundoff.
            use_analytic_gradient = (
                finite_argument.real > _TANH_ANALYTIC_GRADIENT_REAL
            )
            stable_direct_tanh = direct_tanh.detach() + (
                asymptotic_tanh - asymptotic_tanh.detach()
            )
            direct_tanh = torch.where(
                use_analytic_gradient,
                stable_direct_tanh,
                direct_tanh,
            )
            tanh_term = torch.where(
                use_asymptotic,
                asymptotic_tanh,
                direct_tanh,
            )
            effective = q_complex + effective_delta
            denominator = local + effective * tanh_term
            numerator = effective_delta * (local - q_complex * tanh_term) + tanh_term * (
                local_delta * (2.0 * q_complex + local_delta)
            )
            effective_delta = numerator / denominator

    return -effective_delta / (2.0 * q_complex + effective_delta)


def _dimensionless_integrand_bz(
    q: torch.Tensor,
    omega: torch.Tensor,
    sigma: torch.Tensor,
    thickness: torch.Tensor,
    *,
    length_scale_m: float,
    air_path_ratio: float,
    log_induction_shift: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the dimensionless CSEM secondary integrand in scaled wavenumber."""
    reflection = _dimensionless_te_reflection(
        q,
        omega,
        sigma,
        thickness,
        length_scale_m=length_scale_m,
        log_induction_shift=log_induction_shift,
    )
    q_complex = q.to(torch.complex128).reshape(1, -1)
    # Group the geometry-only tail before multiplying by the differentiable
    # reflection.  In the reverse pass this avoids forming grad*q**2 first and
    # only then multiplying by an underflowed exp(-q), an inf*0 NaN at extreme
    # but representable dimensional prefactors.
    propagation_weight = torch.exp(q_complex * air_path_ratio) * q_complex**2
    return reflection * propagation_weight


def _scale_dimensionless_secondary(
    value: torch.Tensor,
    *,
    length_scale_m: float,
    magnetic_moment_am2: float,
    log_amplitude_adjustment: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply ``mu0*m/(4*pi*length**3)`` without unsafe intermediate scale loss."""
    magnitude = torch.abs(value).detach()
    nonzero = magnitude > 0.0
    safe_magnitude = torch.where(nonzero, magnitude, torch.ones_like(magnitude))
    log_amplitude = (
        math.log(MU_0)
        + math.log(magnetic_moment_am2)
        - math.log(4.0 * math.pi)
        - 3.0 * math.log(length_scale_m)
        + torch.log(safe_magnitude)
    )
    if log_amplitude_adjustment is not None:
        log_amplitude = log_amplitude + log_amplitude_adjustment
    log_max = math.log(torch.finfo(torch.float64).max)
    log_min = math.log(math.ulp(0.0))
    invalid = nonzero & ((log_amplitude > log_max) | (log_amplitude < log_min))
    if not bool(torch.isfinite(log_amplitude).all()) or bool(invalid.any()):
        raise EMNumericsError(
            "CSEM1D secondary-field scaling is not representable",
            details={
                "operator": "CSEM1D",
                "stage": "secondary_scaling",
                "length_scale_m": str(length_scale_m),
                "magnetic_moment_am2": str(magnetic_moment_am2),
            },
            object_name="CSEM1D",
            field="bz",
        )
    amplitude = torch.exp(log_amplitude)
    phase = value / safe_magnitude
    return torch.where(nonzero, phase * amplitude, torch.zeros_like(value))


def _dimensionless_induction_log_shift(
    omega: torch.Tensor,
    sigma: torch.Tensor,
    *,
    length_scale_m: float,
) -> torch.Tensor:
    """Return a detached per-frequency shift that keeps tiny induction normal.

    The shifted reflection is evaluated at a safely normal induction number;
    its inverse shift is applied only when the final dimensional field is
    scaled.  The target is so deep in the linear-induction regime that the
    rescaling is exact to binary64 throughout the quadrature grid.
    """
    log_alpha = (
        torch.log(omega).reshape(-1, 1)
        + math.log(MU_0)
        + torch.log(sigma).reshape(1, -1)
        + 2.0 * math.log(length_scale_m)
    )
    maximum_log_alpha = torch.max(log_alpha, dim=1, keepdim=True).values
    shift = torch.clamp(_LOW_INDUCTION_RESCALE_LOG - maximum_log_alpha, min=0.0)
    return shift.detach()


def _needs_scale_safe_mixed_hessian(
    omega: torch.Tensor,
    sigma: torch.Tensor,
    thickness: torch.Tensor,
) -> bool:
    """Return whether an axial mixed Hessian approaches subnormal precision.

    At zero dimensionless wavenumber the finite-layer propagation constant is
    ``sqrt(i*omega*mu*sigma)``.  Its saturated ``tanh`` mixed
    thickness/conductivity derivative has asymptotic log-magnitude

    ``log(4) + log|p| - 2*Re(p*h) - log(sigma) + log|p*h|``.

    Once that coefficient is within ``sqrt(eps)`` of the binary64 normal
    boundary, applying the conductivity chain factor before the later Hankel
    and SI scales can quantize otherwise representable derivatives.  The
    check is detached, allocation-bounded, and depends only on machine
    precision rather than a model-specific numerical threshold.
    """
    if (
        not torch.is_grad_enabled()
        or not sigma.requires_grad
        or not thickness.requires_grad
        or thickness.numel() == 0
    ):
        return False

    with torch.no_grad():
        finite_sigma = sigma[:-1]
        log_propagation = 0.5 * (
            torch.log(omega).reshape(-1, 1)
            + math.log(MU_0)
            + torch.log(finite_sigma).reshape(1, -1)
        )
        log_argument = log_propagation + torch.log(thickness).reshape(1, -1)
        argument_magnitude = torch.exp(
            torch.clamp(log_argument, max=math.log(1.0e300))
        )
        argument_real = argument_magnitude / math.sqrt(2.0)
        mixed_log_magnitude = (
            math.log(4.0)
            + log_propagation
            - 2.0 * argument_real
            - torch.log(finite_sigma).reshape(1, -1)
            + log_argument
        )
        float_info = torch.finfo(torch.float64)
        precision_risk_ceiling = math.log(float_info.tiny) - 0.5 * math.log(float_info.eps)
        at_risk = (
            (argument_real > _TANH_ASYMPTOTIC_REAL)
            & (mixed_log_magnitude <= precision_risk_ceiling)
        )
        return bool(at_risk.any())


def _vmd_primary_bz(
    source_pos: tuple[float, float, float],
    receiver_pos: tuple[float, float, float],
    magnetic_moment_am2: float,
    mu0: float = MU_0,
) -> float:
    """Free-space VMD vertical B-field: closed form, no Hankel.

    A vertical magnetic dipole of moment ``m`` at ``(x_s, y_s, z_s)``
    produces a free-space (non-conducting, no displacement currents)
    vertical magnetic flux density at ``(x_r, y_r, z_r)``

    ``B_z = (μ₀ m / 4π) * (2 Δz² - r²) / R⁵``

    where ``Δz = z_r - z_s``, ``r = √(Δx² + Δy²)``,
    ``R = √(r² + Δz²)``. This is the exact value of the "direct" part
    of the layered-earth Hankel integral

    ``∫₀^∞ exp(-λ |Δz|) λ² J_0(λ r) dλ = (2 Δz² - r²) / R⁵``

    for ``R > 0``. We evaluate it analytically rather than via DLF to
    eliminate filter accuracy floor on primary-dominated geometries
    (low-induction-number airborne / HEM).

    The primary is frequency-independent in free space (quasi-static
    induction regime, displacement currents neglected), so we return a
    single Python ``float`` that the caller broadcasts across the
    frequency sweep.

    Raises:
        EMContractError: if source and receiver coincide (``R = 0``).
        EMNumericsError: if finite inputs produce a non-finite intermediate.
    """
    sx, sy, sz = source_pos
    rx, ry, rz = receiver_pos
    try:
        dx = float(rx) - float(sx)
        dy = float(ry) - float(sy)
        dz = float(rz) - float(sz)
    except (OverflowError, TypeError, ValueError) as exc:
        raise EMNumericsError(
            "VMD primary coordinate differences are not representable",
            details={"operator": "_vmd_primary_bz", "stage": "coordinate_difference"},
            object_name="_vmd_primary_bz",
            field="receiver_pos",
        ) from exc
    if not all(math.isfinite(value) for value in (dx, dy, dz)):
        raise EMNumericsError(
            "VMD primary coordinate differences are not representable",
            details={
                "operator": "_vmd_primary_bz",
                "stage": "coordinate_difference",
                "difference": [str(dx), str(dy), str(dz)],
            },
            object_name="_vmd_primary_bz",
            field="receiver_pos",
        )
    horizontal_distance = math.hypot(dx, dy)
    distance = math.hypot(horizontal_distance, dz)
    if distance == 0.0:
        raise EMContractError(
            "VMD primary undefined at coincident source/receiver "
            f"(source ({sx}, {sy}, {sz}) == receiver ({rx}, {ry}, {rz})).",
            details={
                "source_position": [sx, sy, sz],
                "receiver_position": [rx, ry, rz],
            },
            object_name="_vmd_primary_bz",
            field="receiver_pos",
            expected="R > 0 (non-coincident source/receiver)",
            actual=distance,
        )
    if not math.isfinite(distance):
        raise EMNumericsError(
            "VMD primary distance is not representable",
            details={
                "operator": "_vmd_primary_bz",
                "stage": "distance",
                "distance": str(distance),
            },
            object_name="_vmd_primary_bz",
            field="receiver_pos",
        )

    radial_fraction = horizontal_distance / distance
    vertical_fraction = dz / distance
    angular_factor = 2.0 * vertical_fraction * vertical_fraction - radial_fraction * radial_fraction
    if angular_factor == 0.0:
        return 0.0
    try:
        log_magnitude = (
            math.log(mu0)
            + math.log(magnetic_moment_am2)
            - math.log(4.0 * math.pi)
            + math.log(abs(angular_factor))
            - 3.0 * math.log(distance)
        )
        if log_magnitude < math.log(math.ulp(0.0)):
            raise OverflowError("primary magnitude is below float subnormal range")
        primary = math.copysign(math.exp(log_magnitude), angular_factor)
    except (OverflowError, TypeError, ValueError) as exc:
        raise EMNumericsError(
            "VMD primary scaling is not representable",
            details={
                "operator": "_vmd_primary_bz",
                "stage": "primary_scaling",
                "distance": str(distance),
            },
            object_name="_vmd_primary_bz",
            field="receiver_pos",
        ) from exc
    if not math.isfinite(primary):
        raise EMNumericsError(
            "VMD primary evaluation produced a non-finite field",
            details={
                "operator": "_vmd_primary_bz",
                "stage": "primary_field",
                "distance": str(distance),
            },
            object_name="_vmd_primary_bz",
            field="receiver_pos",
        )
    return primary

