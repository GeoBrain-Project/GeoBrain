"""Validated FFT map processing for regular Potential grids.

Every public function consumes one finite real ``(ny, nx)`` tensor and uses
keyword-only SI spacings ``dx_m`` and ``dy_m``.  Axis ``-1`` is east/x and axis
``-2`` is north/y.  The transforms use the natural periodic
``torch.fft.fft2`` convention: no implicit padding, tapering, or cropping is
performed, and every result has the input shape, dtype, device, and autograd
lineage.

The vertical derivative uses positive-down z.  ``upward_continue`` therefore
applies ``exp(-|k| height_m)`` for a non-negative upward height.  RTP angles use
geophysical inclination (positive down) and declination clockwise from north.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
import numbers
from typing import NoReturn, cast

import torch

from geobrain.core import ErrorCode

from .errors import PotentialContractError, PotentialNumericsError


_MAX_DERIVATIVE_ORDER = (1 << 63) - 1


def _contract_error(
    message: str,
    *,
    object_name: str,
    field: str,
    expected: object,
    actual: object,
    code: ErrorCode = ErrorCode.CONFIG_INVALID,
) -> NoReturn:
    raise PotentialContractError(
        message,
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
        code=code,
        hint="Provide a finite value that satisfies the documented SI processing contract.",
    )


def _validate_grid(data: object, *, object_name: str) -> torch.Tensor:
    """Return a validated finite real strided ``(ny, nx)`` tensor unchanged."""
    if not isinstance(data, torch.Tensor):
        _contract_error(
            "Potential map processing requires a torch.Tensor grid.",
            object_name=object_name,
            field="data",
            expected="finite real floating torch.Tensor [ny, nx]",
            actual=data,
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if data.is_nested:
        _contract_error(
            "Potential map processing does not support nested tensor grids.",
            object_name=object_name,
            field="data.layout",
            expected="regular strided tensor",
            actual={"layout": data.layout, "is_nested": True},
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
        )
    if data.layout != torch.strided:
        _contract_error(
            "Potential map processing requires a strided grid.",
            object_name=object_name,
            field="data.layout",
            expected=str(torch.strided),
            actual=data.layout,
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
        )
    if data.dtype not in {torch.float32, torch.float64}:
        _contract_error(
            "Potential map processing supports only real float32/float64 grids.",
            object_name=object_name,
            field="data.dtype",
            expected=[str(torch.float32), str(torch.float64)],
            actual=data.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
        )
    if data.device.type == "meta":
        _contract_error(
            "Potential map processing requires a materialized device.",
            object_name=object_name,
            field="data.device",
            expected="materialized CPU or accelerator tensor",
            actual=data.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )
    if data.ndim != 2 or data.numel() == 0 or 0 in data.shape:
        _contract_error(
            "Potential map processing requires a non-empty two-dimensional grid.",
            object_name=object_name,
            field="data",
            expected="non-empty [ny, nx]",
            actual=tuple(data.shape),
            code=ErrorCode.SHAPE_MISMATCH,
        )
    if not bool(torch.isfinite(data).all()):
        _contract_error(
            "Potential map processing requires finite grid values.",
            object_name=object_name,
            field="data",
            expected="all finite values",
            actual="contains non-finite values",
        )
    return data


def _finite_real(
    value: object,
    *,
    object_name: str,
    field: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _contract_error(
            "Potential processing scalar parameters must be real numbers.",
            object_name=object_name,
            field=field,
            expected="finite real scalar",
            actual=value,
        )
    try:
        resolved = float(value)
    except (OverflowError, ValueError):
        _contract_error(
            "Potential processing scalar parameter is not representable as a finite float.",
            object_name=object_name,
            field=field,
            expected="finite float-representable real scalar",
            actual=value,
        )
    if not math.isfinite(resolved):
        _contract_error(
            "Potential processing scalar parameters must be finite.",
            object_name=object_name,
            field=field,
            expected="finite real scalar",
            actual=value,
        )
    if positive and resolved <= 0.0:
        _contract_error(
            "Potential processing parameter must be strictly positive.",
            object_name=object_name,
            field=field,
            expected="> 0",
            actual=resolved,
        )
    if nonnegative and resolved < 0.0:
        _contract_error(
            "Potential processing parameter must be non-negative.",
            object_name=object_name,
            field=field,
            expected=">= 0",
            actual=resolved,
        )
    return resolved


def _spacings(*, object_name: str, dx_m: object, dy_m: object) -> tuple[float, float]:
    return (
        _finite_real(dx_m, object_name=object_name, field="dx_m", positive=True),
        _finite_real(dy_m, object_name=object_name, field="dy_m", positive=True),
    )


def _wavenumbers(
    ny: int,
    nx: int,
    *,
    dy_m: float,
    dx_m: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
    object_name: str = "_wavenumbers",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return angular ``(kx, ky)`` grids in natural FFT order."""

    def axis_wavenumbers(size: int, spacing_m: float, *, field: str) -> torch.Tensor:
        period_m = size * spacing_m
        if math.isfinite(period_m):
            result = (
                2.0
                * math.pi
                * torch.fft.fftfreq(
                    size,
                    d=spacing_m,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            unit_bins = torch.fft.fftfreq(
                size,
                d=1.0,
                device=device,
                dtype=dtype,
            )
            result = (2.0 * math.pi * unit_bins) / spacing_m
        invalid = not bool(torch.isfinite(result).all()) or (
            size > 1 and bool(torch.any(result[1:] == 0.0))
        )
        if invalid and dtype == torch.float32:
            wide_unit_bins = torch.fft.fftfreq(
                size,
                d=1.0,
                device=device,
                dtype=torch.float64,
            )
            result = ((2.0 * math.pi * wide_unit_bins) / spacing_m).to(dtype=dtype)
        if not bool(torch.isfinite(result).all()) or (
            size > 1 and bool(torch.any(result[1:] == 0.0))
        ):
            raise PotentialNumericsError(
                "Potential FFT non-DC wavenumbers are not representable in the grid dtype.",
                object_name=object_name,
                field=field,
                expected="every non-DC FFT bin is non-zero and representable",
                actual={"spacing_m": spacing_m, "size": size, "dtype": dtype},
                hint="Use float64 data or a smaller grid spacing.",
            )
        return cast(torch.Tensor, result)

    kx_1d = axis_wavenumbers(nx, dx_m, field="dx_m")
    ky_1d = axis_wavenumbers(ny, dy_m, field="dy_m")
    return cast(
        tuple[torch.Tensor, torch.Tensor],
        torch.meshgrid(kx_1d, ky_1d, indexing="xy"),
    )


def _dimensionless_radial_wavenumber(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    scale_m: float,
    object_name: str,
) -> torch.Tensor:
    """Return ``|k| * scale_m`` without first materializing physical ``k``."""

    def scaled_axis(size: int, spacing_m: float, *, field: str) -> torch.Tensor:
        unit_bins = torch.fft.fftfreq(
            size,
            d=1.0,
            device=data.device,
            dtype=data.dtype,
        )
        try:
            ratio = scale_m / spacing_m
        except OverflowError:
            ratio = math.inf
        magnitude = torch.abs(2.0 * math.pi * unit_bins)
        if math.isfinite(ratio) and ratio <= torch.finfo(data.dtype).max:
            result = magnitude * ratio
        else:
            result = torch.where(
                magnitude == 0.0,
                torch.zeros_like(magnitude),
                torch.full_like(magnitude, math.inf),
            )
        if size > 1 and bool(torch.any(result[1:] == 0.0)):
            raise PotentialNumericsError(
                "Potential FFT non-DC continuation exponents are not representable in the grid dtype.",
                object_name=object_name,
                field=field,
                expected="every non-DC FFT continuation exponent is non-zero or infinite",
                actual={
                    "spacing_m": spacing_m,
                    "scale_m": scale_m,
                    "size": size,
                    "dtype": data.dtype,
                },
                hint="Use float64 data or a less extreme height-to-spacing ratio.",
            )
        return result

    ny, nx = data.shape
    scaled_x = scaled_axis(nx, dx_m, field="dx_m")
    scaled_y = scaled_axis(ny, dy_m, field="dy_m")
    dimensionless_x, dimensionless_y = torch.meshgrid(
        scaled_x,
        scaled_y,
        indexing="xy",
    )
    return torch.hypot(dimensionless_x, dimensionless_y)


def _stable_vector_norm(*components: torch.Tensor) -> torch.Tensor:
    """Return a pointwise norm without squaring unscaled physical values."""
    stacked = torch.stack(components)
    scale = torch.amax(torch.abs(stacked), dim=0)
    safe_scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)
    normalized = stacked / safe_scale
    norm = cast(torch.Tensor, torch.linalg.vector_norm(normalized, dim=0))
    return scale * norm


_MAX_SPECTRAL_WORK_BYTES = 256 * 1024 * 1024
_MAX_SPECTRAL_TRANSFORMS = 65_536
_MAX_DIRECT_SPECTRAL_PRODUCTS = 1 << 24
_ZERO_EXPONENT_SENTINEL = -(1 << 60)


def _spectral_resource_error(
    *,
    object_name: str,
    transform_count: int,
    estimated_bytes: int,
) -> NoReturn:
    raise PotentialNumericsError(
        "Potential spectral working set exceeds the bounded processing policy.",
        object_name=object_name,
        field="spectral_working_set",
        expected={
            "maximum_transforms": _MAX_SPECTRAL_TRANSFORMS,
            "maximum_bytes": _MAX_SPECTRAL_WORK_BYTES,
        },
        actual={
            "transform_count": transform_count,
            "estimated_bytes": estimated_bytes,
        },
        code=ErrorCode.EXECUTION_FAILED,
        hint="Reduce the grid dynamic range/order or process the map in bounded tiles.",
    )


def _odd_kernel_precision_resource_error(
    *,
    object_name: str,
    direct_products: int,
    dtype: torch.dtype,
) -> NoReturn:
    raise PotentialNumericsError(
        "An extreme-scale odd spectral kernel requires exact bounded convolution.",
        object_name=object_name,
        field="spectral_working_set",
        expected={"maximum_direct_products": _MAX_DIRECT_SPECTRAL_PRODUCTS},
        actual={"direct_products": direct_products, "dtype": dtype},
        code=ErrorCode.EXECUTION_FAILED,
        hint="Reduce the grid size, use a less extreme spacing, or process the map in bounded tiles.",
    )


def _preflight_spectral_work(
    *,
    data: torch.Tensor,
    transform_count: int,
    tensor_count: int,
    object_name: str,
) -> None:
    """Reject a spectral expansion before any proportional tensor is allocated."""
    real_bytes = torch.empty((), dtype=data.dtype).element_size()
    work_real_bytes = 8 if data.dtype == torch.float32 and data.device.type != "mps" else real_bytes
    # Covers the complex spectrum, real terms, mantissa/exponent/order work
    # arrays, masks, and one backend FFT scratch copy at their peak overlap.
    estimated_bytes = tensor_count * data.numel() * (10 * work_real_bytes + 1)
    if transform_count > _MAX_SPECTRAL_TRANSFORMS or estimated_bytes > _MAX_SPECTRAL_WORK_BYTES:
        _spectral_resource_error(
            object_name=object_name,
            transform_count=transform_count,
            estimated_bytes=estimated_bytes,
        )


def _compensated_sum_last(values: torch.Tensor) -> torch.Tensor:
    """Neumaier-sum the final dimension using only same-device Torch ops."""
    total = torch.zeros(values.shape[:-1], dtype=values.dtype, device=values.device)
    compensation = torch.zeros_like(total)
    for index in range(values.shape[-1]):
        value = values[..., index]
        updated = total + value
        correction = torch.where(
            torch.abs(total) >= torch.abs(value),
            (total - updated) + value,
            (value - updated) + total,
        )
        compensation = compensation + correction
        total = updated
    return total + compensation


def _scaled_spatial_layers(
    data: torch.Tensor,
    *,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized exponent layers, their scales, and the exact mean."""
    nonzero = data != 0.0
    _, value_exponents = torch.frexp(data)
    band_exponents = torch.unique(value_exponents[nonzero], sorted=True).to(torch.int64)
    if band_exponents.numel() == 0:
        band_exponents = torch.zeros(1, dtype=torch.int64, device=data.device)

    layer_count = band_exponents.numel()
    _preflight_spectral_work(
        data=data,
        transform_count=layer_count,
        tensor_count=layer_count,
        object_name=object_name,
    )
    minimum_exponent = int(math.log2(torch.finfo(data.dtype).tiny))
    transform_guard = math.ceil(0.5 * math.log2(data.numel()))
    maximum_exponent = math.frexp(torch.finfo(data.dtype).max)[1] - 1 - transform_guard
    scale_exponents = torch.clamp(
        band_exponents,
        min=minimum_exponent,
        max=maximum_exponent,
    )
    scales = torch.ldexp(
        torch.ones(layer_count, dtype=data.dtype, device=data.device),
        scale_exponents,
    )
    masks = nonzero.unsqueeze(0) & (value_exponents.unsqueeze(0) == band_exponents[:, None, None])
    normalized = torch.where(
        masks,
        data.unsqueeze(0) / scales[:, None, None],
        torch.zeros((), dtype=data.dtype, device=data.device),
    )
    promote_float32 = data.dtype == torch.float32 and data.device.type != "mps"
    work = normalized.to(dtype=torch.float64) if promote_float32 else normalized
    normalized_sums = _compensated_sum_last(work.reshape(layer_count, -1))
    normalized_means = normalized_sums / data.numel()
    mean, _ = _combine_power_two_terms(
        normalized_means[:, None, None],
        scale_exponents,
        target_dtype=data.dtype,
    )
    return work, scale_exponents, mean.reshape(())


def _fft_normalized_layers(
    work: torch.Tensor,
    *,
    data: torch.Tensor,
) -> torch.Tensor:
    """Transform normalized layers and replace their DC by a compensated sum."""
    spectra = torch.fft.fft2(work, norm="ortho")
    normalized_sums = _compensated_sum_last(work.reshape(work.shape[0], -1))
    selector = torch.zeros(data.shape, dtype=spectra.dtype, device=data.device)
    selector[0, 0] = 1.0
    accurate_dc = normalized_sums / math.sqrt(data.numel())
    return cast(
        torch.Tensor,
        spectra + (accurate_dc - spectra[:, 0, 0])[:, None, None] * selector,
    )


def _spectral_log2_geometry(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed unit bins and log2 physical axis/radial wavenumbers.

    Axis logs are formed independently.  A singleton or extremely fine axis
    can therefore never make a representable non-DC bin on another axis vanish.
    """
    ny, nx = data.shape
    angular_x = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            nx,
            d=1.0,
            device=data.device,
            dtype=data.dtype,
        )
    )
    angular_y = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            ny,
            d=1.0,
            device=data.device,
            dtype=data.dtype,
        )
    )
    unit_x, unit_y = torch.meshgrid(angular_x, angular_y, indexing="xy")
    abs_x = torch.abs(unit_x)
    abs_y = torch.abs(unit_y)
    negative_infinity = torch.full_like(abs_x, -math.inf)
    log2_x = torch.where(
        abs_x == 0.0,
        negative_infinity,
        torch.log2(abs_x) - math.log2(dx_m),
    )
    log2_y = torch.where(
        abs_y == 0.0,
        negative_infinity,
        torch.log2(abs_y) - math.log2(dy_m),
    )
    active = torch.isfinite(log2_x) | torch.isfinite(log2_y)
    maximum = torch.maximum(log2_x, log2_y)
    safe_maximum = torch.where(active, maximum, torch.zeros_like(maximum))
    scaled_square = torch.where(
        torch.isfinite(log2_x),
        torch.exp2(2.0 * (log2_x - safe_maximum)),
        torch.zeros_like(log2_x),
    )
    scaled_square = scaled_square + torch.where(
        torch.isfinite(log2_y),
        torch.exp2(2.0 * (log2_y - safe_maximum)),
        torch.zeros_like(log2_y),
    )
    log2_radius = torch.where(
        active,
        safe_maximum + 0.5 * torch.log2(scaled_square),
        negative_infinity,
    )
    return unit_x, unit_y, log2_x, log2_y, log2_radius


def _combine_power_two_terms(
    terms: torch.Tensor,
    exponents: torch.Tensor,
    *,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round a bounded power-of-two expansion only after per-cell cancellation.

    Terms are sorted by their *actual pointwise* exponent.  Strong comparable
    contributions therefore cancel before weaker terms are introduced.  The
    accumulator remains a normalized mantissa/exponent pair until the final
    target-dtype rounding, so separately subnormal terms may form a
    representable result.
    """
    if terms.shape[0] == 0:
        zero = torch.zeros(terms.shape[1:], dtype=target_dtype, device=terms.device)
        return zero, torch.zeros_like(zero, dtype=torch.bool)
    mantissas, local_exponents = torch.frexp(terms)
    actual_exponents = exponents[:, None, None] + local_exponents.to(torch.int64)
    actual_exponents = torch.where(
        mantissas == 0.0,
        torch.full_like(actual_exponents, _ZERO_EXPONENT_SENTINEL),
        actual_exponents,
    )
    order = torch.argsort(actual_exponents, dim=0, descending=True)
    sorted_mantissas = torch.gather(mantissas, 0, order)
    sorted_exponents = torch.gather(actual_exponents, 0, order)

    current_mantissa = sorted_mantissas[0]
    current_exponent = sorted_exponents[0]
    work_minimum = -149 if terms.dtype == torch.float32 else -1074
    work_maximum = math.frexp(torch.finfo(terms.dtype).max)[1]
    for index in range(1, terms.shape[0]):
        term_mantissa = sorted_mantissas[index]
        term_exponent = sorted_exponents[index]
        common_exponent = torch.maximum(current_exponent, term_exponent)
        current_shift = torch.clamp(
            current_exponent - common_exponent,
            min=work_minimum,
            max=0,
        )
        term_shift = torch.clamp(
            term_exponent - common_exponent,
            min=work_minimum,
            max=0,
        )
        combined = torch.ldexp(current_mantissa, current_shift)
        combined = combined + torch.ldexp(term_mantissa, term_shift)
        current_mantissa, renormalization = torch.frexp(combined)
        current_exponent = common_exponent + renormalization.to(torch.int64)
        current_exponent = torch.where(
            current_mantissa == 0.0,
            torch.full_like(current_exponent, _ZERO_EXPONENT_SENTINEL),
            current_exponent,
        )

    finite_exponent = torch.where(
        current_mantissa == 0.0,
        torch.zeros_like(current_exponent),
        current_exponent,
    )
    low = finite_exponent < work_minimum
    high = finite_exponent > work_maximum
    shifted = torch.ldexp(
        current_mantissa,
        torch.clamp(finite_exponent, min=work_minimum, max=work_maximum),
    )
    shifted = torch.where(low, torch.zeros_like(shifted), shifted)
    shifted = torch.where(
        high & (current_mantissa != 0.0),
        torch.copysign(torch.full_like(shifted, math.inf), current_mantissa),
        shifted,
    )
    output = shifted.to(dtype=target_dtype)
    lost_nonzero = (current_mantissa != 0.0) & (output == 0.0)
    return output, lost_nonzero


def _zero_mean_spatial_kernels(
    transfers: torch.Tensor,
    *,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    """Return real convolution kernels with an exactly corrected zero mean."""
    complex_dtype = torch.complex128 if work_dtype == torch.float64 else torch.complex64
    kernels = torch.fft.ifft2(
        transfers.to(dtype=complex_dtype),
        norm="backward",
    ).real
    reversed_kernels = torch.roll(
        torch.flip(kernels, dims=(-2, -1)),
        shifts=(1, 1),
        dims=(-2, -1),
    )
    odd_kernels = 0.5 * (kernels - reversed_kernels)
    even_kernels = 0.5 * (kernels + reversed_kernels)
    if transfers.is_complex():
        purely_imaginary = torch.all(transfers.real == 0.0)
        purely_real = torch.all(transfers.imag == 0.0)
    else:
        purely_imaginary = torch.zeros((), dtype=torch.bool, device=transfers.device)
        purely_real = torch.ones((), dtype=torch.bool, device=transfers.device)
    kernels = torch.where(
        purely_imaginary,
        odd_kernels,
        torch.where(purely_real, even_kernels, kernels),
    )
    row_selector = torch.zeros(
        kernels.shape[-2:],
        dtype=kernels.dtype,
        device=kernels.device,
    )
    row_selector[0, :] = 1.0
    row_invariant = torch.all(transfers == transfers[..., :1, :])
    kernels = torch.where(row_invariant, kernels * row_selector, kernels)
    column_selector = torch.zeros(
        kernels.shape[-2:],
        dtype=kernels.dtype,
        device=kernels.device,
    )
    column_selector[:, 0] = 1.0
    column_invariant = torch.all(transfers == transfers[..., :, :1])
    kernels = torch.where(column_invariant, kernels * column_selector, kernels)
    flattened = kernels.reshape(*kernels.shape[:-2], -1)
    kernel_sums = _compensated_sum_last(flattened)
    selector = torch.zeros(
        kernels.shape[-2:],
        dtype=kernels.dtype,
        device=kernels.device,
    )
    selector[0, 0] = 1.0
    return kernels - kernel_sums[..., None, None] * selector


def _log2_transfer_has_structurally_zero_real_kernel(
    log2_amplitude: torch.Tensor,
    phase: torch.Tensor,
) -> bool:
    """Return whether a log-scaled transfer has no real spatial support.

    A real inverse FFT only observes the Hermitian projection
    ``(T(k) + conj(T(-k))) / 2``.  In particular, an imaginary transfer on a
    self-conjugate Nyquist bin is a structural zero, independent of the input
    grid or its scale.  Compare log-amplitudes and phases directly so this test
    runs before input scaling, layer construction, and every resource guard.
    """
    active = torch.isfinite(log2_amplitude) & (torch.abs(phase) != 0.0)
    reversed_active = torch.roll(
        torch.flip(active, dims=(-2, -1)),
        shifts=(1, 1),
        dims=(-2, -1),
    )
    reversed_log2_amplitude = torch.roll(
        torch.flip(log2_amplitude, dims=(-2, -1)),
        shifts=(1, 1),
        dims=(-2, -1),
    )
    reversed_phase = torch.roll(
        torch.flip(phase, dims=(-2, -1)),
        shifts=(1, 1),
        dims=(-2, -1),
    )
    paired_support = active == reversed_active
    paired_amplitude = ~active | (log2_amplitude == reversed_log2_amplitude)
    anti_hermitian_phase = ~active | (phase == -torch.conj(reversed_phase))
    return bool(torch.all(paired_support & paired_amplitude & anti_hermitian_phase))


def _direct_circular_convolution(
    layers: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Convolve small normalized layers without contaminating them by FFT roundoff."""
    total = torch.zeros_like(layers)
    compensation = torch.zeros_like(layers)
    ny, nx = kernel.shape
    for y_index in range(ny):
        for x_index in range(nx):
            shifted_kernel = torch.roll(
                kernel,
                shifts=(y_index, x_index),
                dims=(0, 1),
            )
            term = layers[:, y_index, x_index, None, None] * shifted_kernel.unsqueeze(0)
            updated = total + term
            correction = torch.where(
                torch.abs(total) >= torch.abs(term),
                (total - updated) + term,
                (term - updated) + total,
            )
            compensation = compensation + correction
            total = updated
    return total + compensation


def _remove_layer_means(layers: torch.Tensor) -> torch.Tensor:
    """Project normalized spatial layers off DC before direct convolution."""
    layer_sums = _compensated_sum_last(layers.reshape(layers.shape[0], -1))
    layer_means = layer_sums / layers[0].numel()
    return layers - layer_means[:, None, None]


def _bounded_spectral_core(
    data: torch.Tensor,
    transfer: torch.Tensor,
    *,
    object_name: str,
) -> torch.Tensor:
    dc_selector = torch.zeros_like(transfer)
    dc_selector[0, 0] = 1.0
    non_dc_transfer = transfer * (1.0 - dc_selector)
    layers, scale_exponents, accurate_mean = _scaled_spatial_layers(
        data,
        object_name=object_name,
    )
    direct_products = layers.shape[0] * data.numel() * data.numel()
    if direct_products <= _MAX_DIRECT_SPECTRAL_PRODUCTS:
        kernel = _zero_mean_spatial_kernels(
            non_dc_transfer,
            work_dtype=layers.dtype,
        )
        terms = _direct_circular_convolution(_remove_layer_means(layers), kernel)
    else:
        spectra = _fft_normalized_layers(layers, data=data)
        terms = torch.fft.ifft2(
            spectra * non_dc_transfer.unsqueeze(0),
            norm="ortho",
        ).real
    output, _ = _combine_power_two_terms(
        terms,
        scale_exponents,
        target_dtype=data.dtype,
    )
    output = output + accurate_mean
    return _finite_output(output, object_name=object_name)


def _log2_spectral_core(
    data: torch.Tensor,
    log2_amplitude: torch.Tensor,
    phase: torch.Tensor,
    *,
    object_name: str,
) -> torch.Tensor:
    if _log2_transfer_has_structurally_zero_real_kernel(log2_amplitude, phase):
        return torch.zeros_like(data)
    layers, scale_exponents, _ = _scaled_spatial_layers(
        data,
        object_name=object_name,
    )
    layer_count = layers.shape[0]
    active = torch.isfinite(log2_amplitude) & (torch.abs(phase) != 0.0)
    raw_frequency_exponents = torch.floor(log2_amplitude[active])
    if raw_frequency_exponents.numel() == 0:
        return torch.zeros_like(data)
    if bool(
        torch.any(~torch.isfinite(raw_frequency_exponents))
        | torch.any(torch.abs(raw_frequency_exponents) > (1 << 62))
    ):
        raise PotentialNumericsError(
            "Potential derivative spectral exponent is outside backend range.",
            object_name=object_name,
            field="order",
            expected="finite spectral exponent with absolute value <= 2**62",
            actual={"dtype": log2_amplitude.dtype},
            hint="Use a lower derivative order or physically representable spacing.",
        )
    frequency_exponents = torch.unique(raw_frequency_exponents, sorted=True).to(torch.int64)
    frequency_count = frequency_exponents.numel()
    pair_count = layer_count * frequency_count
    _preflight_spectral_work(
        data=data,
        transform_count=pair_count,
        tensor_count=pair_count,
        object_name=object_name,
    )
    use_direct_convolution = (
        pair_count * data.numel() * data.numel() <= _MAX_DIRECT_SPECTRAL_PRODUCTS
    )

    frequency_grid = frequency_exponents[:, None, None]
    masks = active.unsqueeze(0) & (torch.floor(log2_amplitude).unsqueeze(0) == frequency_grid)
    normalized_amplitude = torch.where(
        masks,
        torch.exp2(log2_amplitude.unsqueeze(0) - frequency_grid),
        torch.zeros((), dtype=log2_amplitude.dtype, device=log2_amplitude.device),
    )
    normalized_transfer = normalized_amplitude * phase.unsqueeze(0)
    direct_products = pair_count * data.numel() * data.numel()
    minimum_normal_exponent = math.floor(math.log2(torch.finfo(data.dtype).tiny))
    purely_imaginary = torch.all(normalized_transfer.real == 0.0)
    reaches_subnormal_scale = torch.any(frequency_exponents <= minimum_normal_exponent)
    if not use_direct_convolution and bool(purely_imaginary & reaches_subnormal_scale):
        _odd_kernel_precision_resource_error(
            object_name=object_name,
            direct_products=direct_products,
            dtype=data.dtype,
        )
    pair_exponents = (scale_exponents[:, None] + frequency_exponents[None, :]).reshape(-1)
    if use_direct_convolution:
        kernels = _zero_mean_spatial_kernels(
            normalized_transfer,
            work_dtype=layers.dtype,
        )
        zero_mean_layers = _remove_layer_means(layers)
        direct_terms = [
            _direct_circular_convolution(zero_mean_layers, kernel) for kernel in kernels
        ]
        # Match the layer-major order used by ``pair_exponents``.
        terms = torch.stack(direct_terms, dim=1).reshape(pair_count, *data.shape)
    else:
        spectra = _fft_normalized_layers(layers, data=data)
        pair_spectra = spectra[:, None, :, :] * normalized_transfer[None, :, :, :]
        terms = torch.fft.ifft2(pair_spectra, norm="ortho").real.reshape(
            pair_count,
            *data.shape,
        )
    output, lost_nonzero = _combine_power_two_terms(
        terms,
        pair_exponents,
        target_dtype=data.dtype,
    )
    if bool(torch.all(output == 0.0) & torch.any(lost_nonzero)):
        raise PotentialNumericsError(
            "Potential derivative non-DC output is not representable in the grid dtype.",
            object_name=object_name,
            field="output",
            expected="representable non-zero derivative contribution",
            actual={"dtype": output.dtype},
            hint="Use float64 data or a less extreme grid spacing.",
        )
    return _finite_output(output, object_name=object_name)


class _BoundedSpectralFunction(torch.autograd.Function):
    """Linear bounded-transfer primitive with independently scaled tangents."""

    @staticmethod
    def forward(
        data: torch.Tensor,
        transfer: torch.Tensor,
        object_name: str,
    ) -> torch.Tensor:
        return _bounded_spectral_core(data, transfer, object_name=object_name)

    @staticmethod
    def setup_context(
        ctx: object,
        inputs: tuple[torch.Tensor, torch.Tensor, str],
        output: torch.Tensor,
    ) -> None:
        context = cast("_SpectralContext", ctx)
        _, transfer, object_name = inputs
        context.save_for_backward(transfer)
        context.save_for_forward(transfer)
        context.object_name = object_name

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        context = cast("_SpectralContext", ctx)
        (transfer,) = context.saved_tensors
        gradient = cast(
            torch.Tensor,
            _BoundedSpectralFunction.apply(  # type: ignore[no-untyped-call]
                grad_output,
                torch.conj(transfer),
                context.object_name,
            ),
        )
        return gradient, None, None

    @staticmethod
    def jvp(
        ctx: object,
        grad_data: torch.Tensor,
        grad_transfer: torch.Tensor | None,
        grad_object_name: None,
    ) -> torch.Tensor:
        context = cast("_SpectralContext", ctx)
        (transfer,) = context.saved_tensors
        return _bounded_spectral_core(
            grad_data,
            transfer,
            object_name=context.object_name,
        )


class _Log2SpectralFunction(torch.autograd.Function):
    """Linear log-transfer primitive with scale-safe forward and adjoint paths."""

    @staticmethod
    def forward(
        data: torch.Tensor,
        log2_amplitude: torch.Tensor,
        phase: torch.Tensor,
        object_name: str,
    ) -> torch.Tensor:
        return _log2_spectral_core(
            data,
            log2_amplitude,
            phase,
            object_name=object_name,
        )

    @staticmethod
    def setup_context(
        ctx: object,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        output: torch.Tensor,
    ) -> None:
        context = cast("_SpectralContext", ctx)
        _, log2_amplitude, phase, object_name = inputs
        context.save_for_backward(log2_amplitude, phase)
        context.save_for_forward(log2_amplitude, phase)
        context.object_name = object_name

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        context = cast("_SpectralContext", ctx)
        log2_amplitude, phase = context.saved_tensors
        gradient = cast(
            torch.Tensor,
            _Log2SpectralFunction.apply(  # type: ignore[no-untyped-call]
                grad_output,
                log2_amplitude,
                torch.conj(phase),
                context.object_name,
            ),
        )
        return gradient, None, None, None

    @staticmethod
    def jvp(
        ctx: object,
        grad_data: torch.Tensor,
        grad_log2_amplitude: torch.Tensor | None,
        grad_phase: torch.Tensor | None,
        grad_object_name: None,
    ) -> torch.Tensor:
        context = cast("_SpectralContext", ctx)
        log2_amplitude, phase = context.saved_tensors
        return _log2_spectral_core(
            grad_data,
            log2_amplitude,
            phase,
            object_name=context.object_name,
        )


class _SpectralContext:
    """Static typing view of the attributes supplied by autograd contexts."""

    object_name: str
    saved_tensors: tuple[torch.Tensor, ...]

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...

    def save_for_forward(self, *tensors: torch.Tensor) -> None: ...


def _scaled_ifft2(
    data: torch.Tensor,
    *,
    transfer: torch.Tensor,
    object_name: str,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        _BoundedSpectralFunction.apply(  # type: ignore[no-untyped-call]
            data, transfer, object_name
        ),
    )


def _log2_transfer_ifft2(
    data: torch.Tensor,
    *,
    log2_amplitude: torch.Tensor,
    phase: torch.Tensor,
    object_name: str,
) -> torch.Tensor:
    return cast(
        torch.Tensor,
        _Log2SpectralFunction.apply(  # type: ignore[no-untyped-call]
            data, log2_amplitude, phase, object_name
        ),
    )


def _finite_output(value: torch.Tensor, *, object_name: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise PotentialNumericsError(
            "Potential map processing produced non-finite output.",
            object_name=object_name,
            field="output",
            expected="all finite values",
            actual={"dtype": value.dtype, "device": value.device},
            hint="Use physically representable spacings, orders, and filter parameters.",
        )
    return value


def _angle(
    value: object,
    *,
    object_name: str,
    field: str,
    lower: float,
    upper: float,
) -> float:
    angle = _finite_real(value, object_name=object_name, field=field)
    if not lower <= angle <= upper:
        _contract_error(
            "Potential field direction angle is outside its canonical range.",
            object_name=object_name,
            field=field,
            expected=f"{lower} <= value <= {upper} degrees",
            actual=angle,
        )
    return angle


def _direction_cosines(
    inclination_deg: float, declination_deg: float
) -> tuple[float, float, float]:
    inclination = math.radians(inclination_deg)
    declination = math.radians(declination_deg)
    horizontal = math.cos(inclination)
    return (
        horizontal * math.sin(declination),
        horizontal * math.cos(declination),
        math.sin(inclination),
    )


def reduce_to_pole(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    inclination_deg: float,
    declination_deg: float,
    magnetization_inclination_deg: float | None = None,
    magnetization_declination_deg: float | None = None,
    stabilization: float,
) -> torch.Tensor:
    """Reduce TMI to the pole with explicit dimensionless damping.

    The filter is ``k² conj(Af Am) / (|Af Am|² + (s k²)²)`` away from DC,
    where ``s`` is ``stabilization``.  This Tikhonov form has no silent epsilon
    substitution and remains bounded near low-latitude directional zeros.
    Exact equatorial inducing or magnetization directions are rejected because
    RTP then loses an entire horizontal frequency direction.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
        inclination_deg / declination_deg: inducing-field direction [deg].
        magnetization_inclination_deg / magnetization_declination_deg:
            remanent magnetization direction (defaults to induced).
        stabilization: low-latitude stabilization factor.
    """
    object_name = "reduce_to_pole"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    inclination = _angle(
        inclination_deg,
        object_name=object_name,
        field="inclination_deg",
        lower=-90.0,
        upper=90.0,
    )
    declination = _angle(
        declination_deg,
        object_name=object_name,
        field="declination_deg",
        lower=-180.0,
        upper=180.0,
    )
    magnetization_inclination = (
        inclination
        if magnetization_inclination_deg is None
        else _angle(
            magnetization_inclination_deg,
            object_name=object_name,
            field="magnetization_inclination_deg",
            lower=-90.0,
            upper=90.0,
        )
    )
    magnetization_declination = (
        declination
        if magnetization_declination_deg is None
        else _angle(
            magnetization_declination_deg,
            object_name=object_name,
            field="magnetization_declination_deg",
            lower=-180.0,
            upper=180.0,
        )
    )
    damping = _finite_real(
        stabilization,
        object_name=object_name,
        field="stabilization",
        positive=True,
    )
    minimum_effective_damping = math.sqrt(torch.finfo(grid.dtype).tiny)
    if damping >= 1.0 or damping < minimum_effective_damping:
        _contract_error(
            "RTP stabilization must be dimensionless and representable in the grid dtype.",
            object_name=object_name,
            field="stabilization",
            expected=f"{minimum_effective_damping} <= value < 1",
            actual=damping,
        )
    field_x, field_y, field_z = _direction_cosines(inclination, declination)
    mag_x, mag_y, mag_z = _direction_cosines(
        magnetization_inclination,
        magnetization_declination,
    )
    if abs(field_z) <= damping or abs(mag_z) <= damping:
        _contract_error(
            "Reduction to the pole is singular at the requested low-latitude direction.",
            object_name=object_name,
            field="inclination_deg/magnetization_inclination_deg",
            expected=f"absolute vertical direction cosine > stabilization ({damping})",
            actual={"field_vertical": field_z, "magnetization_vertical": mag_z},
        )

    unit_x, unit_y, log2_x, log2_y, log2_radius = _spectral_log2_geometry(
        grid,
        dx_m=dx,
        dy_m=dy,
    )
    dc = ~torch.isfinite(log2_radius)
    unit_kx = torch.where(
        torch.isfinite(log2_x),
        torch.sign(unit_x) * torch.exp2(log2_x - log2_radius),
        torch.zeros_like(unit_x),
    )
    unit_ky = torch.where(
        torch.isfinite(log2_y),
        torch.sign(unit_y) * torch.exp2(log2_y - log2_radius),
        torch.zeros_like(unit_y),
    )
    field_factor = 1j * (field_x * unit_kx + field_y * unit_ky) + field_z
    magnetization_factor = 1j * (mag_x * unit_kx + mag_y * unit_ky) + mag_z
    directional_product = field_factor * magnetization_factor
    filter_denominator = directional_product.abs().square() + damping**2
    transfer = directional_product.conj() / filter_denominator
    transfer = torch.where(dc, torch.ones_like(transfer), transfer)
    if bool(torch.all(transfer == 1.0)):
        return grid.clone()
    return _scaled_ifft2(
        grid,
        transfer=transfer,
        object_name=object_name,
    )


def _vertical_derivative_from_grid(
    grid: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    order: int,
    object_name: str,
) -> torch.Tensor:
    """Evaluate a validated vertical derivative for one public operation."""
    _, _, _, _, log2_radius = _spectral_log2_geometry(
        grid,
        dx_m=dx_m,
        dy_m=dy_m,
    )
    log2_amplitude = log2_radius * order
    phase = torch.where(
        torch.isfinite(log2_radius),
        torch.ones_like(log2_radius),
        torch.zeros_like(log2_radius),
    )
    return _log2_transfer_ifft2(
        grid,
        log2_amplitude=log2_amplitude,
        phase=phase,
        object_name=object_name,
    )


def _horizontal_gradient_from_grid(
    grid: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    object_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a validated horizontal gradient for one public operation."""
    unit_x, unit_y, log2_x, log2_y, _ = _spectral_log2_geometry(
        grid,
        dx_m=dx_m,
        dy_m=dy_m,
    )
    derivative_x = _log2_transfer_ifft2(
        grid,
        log2_amplitude=log2_x,
        phase=1j * torch.sign(unit_x),
        object_name=object_name,
    )
    derivative_y = _log2_transfer_ifft2(
        grid,
        log2_amplitude=log2_y,
        phase=1j * torch.sign(unit_y),
        object_name=object_name,
    )
    amplitude = _stable_vector_norm(derivative_x, derivative_y)
    return (
        derivative_x,
        derivative_y,
        _finite_output(amplitude, object_name=object_name),
    )


def vertical_derivative(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    order: int = 1,
) -> torch.Tensor:
    """Return the positive-down vertical derivative ``|k|**order``.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
        order: derivative order (1 or 2).
    """
    object_name = "vertical_derivative"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or order < 1
        or order > _MAX_DERIVATIVE_ORDER
    ):
        _contract_error(
            "Vertical derivative order must be a backend-representable positive integer.",
            object_name=object_name,
            field="order",
            expected=f"non-boolean int in [1, {_MAX_DERIVATIVE_ORDER}]",
            actual=order,
        )
    return _vertical_derivative_from_grid(
        grid,
        dx_m=dx,
        dy_m=dy,
        order=order,
        object_name=object_name,
    )


def horizontal_gradient(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return east derivative, north derivative, and horizontal amplitude.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
    """
    object_name = "horizontal_gradient"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    return _horizontal_gradient_from_grid(
        grid,
        dx_m=dx,
        dy_m=dy,
        object_name=object_name,
    )


def analytic_signal_amplitude(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
) -> torch.Tensor:
    """Return ``sqrt((dT/dx)² + (dT/dy)² + (dT/dz)²)``.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
    """
    object_name = "analytic_signal_amplitude"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    derivative_x, derivative_y, _ = _horizontal_gradient_from_grid(
        grid,
        dx_m=dx,
        dy_m=dy,
        object_name=object_name,
    )
    derivative_z = _vertical_derivative_from_grid(
        grid,
        dx_m=dx,
        dy_m=dy,
        order=1,
        object_name=object_name,
    )
    amplitude = _stable_vector_norm(derivative_x, derivative_y, derivative_z)
    return _finite_output(amplitude, object_name=object_name)


def tilt_derivative(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
) -> torch.Tensor:
    """Return ``atan2(dT/dz, sqrt((dT/dx)² + (dT/dy)²))`` radians.

    A point with exactly zero vertical and horizontal derivatives has zero tilt
    and the finite zero subgradient; no epsilon is added to non-zero values.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
    """
    object_name = "tilt_derivative"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    _, _, horizontal_amplitude = _horizontal_gradient_from_grid(
        grid,
        dx_m=dx,
        dy_m=dy,
        object_name=object_name,
    )
    derivative_z = _vertical_derivative_from_grid(
        grid,
        dx_m=dx,
        dy_m=dy,
        order=1,
        object_name=object_name,
    )
    undefined = (derivative_z == 0.0) & (horizontal_amplitude == 0.0)
    safe_vertical = torch.where(undefined, torch.zeros_like(derivative_z), derivative_z)
    safe_horizontal = torch.where(
        undefined,
        torch.ones_like(horizontal_amplitude),
        horizontal_amplitude,
    )
    return _finite_output(
        torch.atan2(safe_vertical, safe_horizontal),
        object_name=object_name,
    )


def upward_continue(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    height_m: float,
) -> torch.Tensor:
    """Continue a periodic map upward by non-negative ``height_m`` metres.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
        height_m: continuation height [m].
    """
    object_name = "upward_continue"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    height = _finite_real(
        height_m,
        object_name=object_name,
        field="height_m",
        nonnegative=True,
    )
    if height == 0.0:
        return grid.clone()
    exponent = -_dimensionless_radial_wavenumber(
        grid,
        dx_m=dx,
        dy_m=dy,
        scale_m=height,
        object_name=object_name,
    )
    return _scaled_ifft2(
        grid,
        transfer=torch.exp(exponent),
        object_name=object_name,
    )


def regional_residual_lowpass(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
    cutoff_wavelength_m: float,
    soft: bool = True,
    soft_width_fraction: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a map into long-wavelength regional and residual components.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
        cutoff_wavelength_m: regional/residual split wavelength [m].
        soft: cosine-tapered (True) vs hard cutoff.
        soft_width_fraction: taper width as a fraction of the cutoff.
    """
    object_name = "regional_residual_lowpass"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    cutoff = _finite_real(
        cutoff_wavelength_m,
        object_name=object_name,
        field="cutoff_wavelength_m",
        positive=True,
    )
    if type(soft) is not bool:
        _contract_error(
            "Low-pass soft selector must be bool.",
            object_name=object_name,
            field="soft",
            expected="bool",
            actual=soft,
        )
    width = _finite_real(
        soft_width_fraction,
        object_name=object_name,
        field="soft_width_fraction",
        positive=True,
    )
    if width >= 1.0:
        _contract_error(
            "Low-pass transition width must keep an ordered positive pass band.",
            object_name=object_name,
            field="soft_width_fraction",
            expected="0 < value < 1",
            actual=width,
        )
    _, _, _, _, log2_radius = _spectral_log2_geometry(
        grid,
        dx_m=dx,
        dy_m=dy,
    )
    dc = ~torch.isfinite(log2_radius)
    log2_cutoff = math.log2(2.0 * math.pi) - math.log2(cutoff)
    if soft:
        lower = 1.0 - width
        upper = 1.0 + width
        log2_lower = math.log2(lower)
        log2_upper = math.log2(upper)
        if bool(torch.all(dc | (log2_radius <= log2_cutoff + log2_lower))):
            regional = grid.clone()
            return regional, grid - regional
        log2_ratio = log2_radius - log2_cutoff
        normalized_wavenumber = torch.exp2(log2_ratio.clamp(log2_lower, log2_upper))
        phase = (normalized_wavenumber - lower) / (upper - lower)
        taper = 0.5 * (1.0 + torch.cos(math.pi * phase.clamp(0.0, 1.0)))
        transfer = torch.where(
            dc | (log2_ratio <= log2_lower),
            torch.ones_like(log2_radius),
            torch.where(
                log2_ratio >= log2_upper,
                torch.zeros_like(log2_radius),
                taper,
            ),
        )
    else:
        transfer = (dc | (log2_radius <= log2_cutoff)).to(grid.dtype)
        if bool(torch.all(transfer == 1.0)):
            regional = grid.clone()
            return regional, grid - regional
    regional = _scaled_ifft2(
        grid,
        transfer=transfer,
        object_name=object_name,
    )
    residual = _finite_output(grid - regional, object_name=object_name)
    return regional, residual


def field_gradient_tensor(
    data: torch.Tensor,
    *,
    dx_m: float,
    dy_m: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return ``(Txx, Tyy, Tzz, Txy, Txz, Tyz)`` in positive-down z.

    Args:
        data: 2-D gridded field ``(ny, nx)``.
        dx_m / dy_m: grid spacings [m].
    """
    object_name = "field_gradient_tensor"
    grid = _validate_grid(data, object_name=object_name)
    dx, dy = _spacings(object_name=object_name, dx_m=dx_m, dy_m=dy_m)
    unit_x, unit_y, log2_x, log2_y, log2_radius = _spectral_log2_geometry(
        grid,
        dx_m=dx,
        dy_m=dy,
    )
    transfers = (
        (2.0 * log2_x, -torch.ones_like(log2_x)),
        (2.0 * log2_y, -torch.ones_like(log2_y)),
        (2.0 * log2_radius, torch.ones_like(log2_radius)),
        (log2_x + log2_y, -torch.sign(unit_x * unit_y)),
        (log2_x + log2_radius, 1j * torch.sign(unit_x)),
        (log2_y + log2_radius, 1j * torch.sign(unit_y)),
    )
    outputs = tuple(
        _log2_transfer_ifft2(
            grid,
            log2_amplitude=log2_amplitude,
            phase=phase,
            object_name=object_name,
        )
        for log2_amplitude, phase in transfers
    )
    return cast(
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        outputs,
    )


__all__ = [
    "analytic_signal_amplitude",
    "field_gradient_tensor",
    "horizontal_gradient",
    "reduce_to_pole",
    "regional_residual_lowpass",
    "tilt_derivative",
    "upward_continue",
    "vertical_derivative",
]
