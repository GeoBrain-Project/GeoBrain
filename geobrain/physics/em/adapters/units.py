"""Explicit tensor-only SI/display unit adapters for the EM family.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core.errors import ErrorCode

from ..errors import EMContractError


def _require_scalable_tensor(value: object, *, adapter: str) -> torch.Tensor:
    """Require a live floating/complex tensor without casting or moving it."""
    if not isinstance(value, torch.Tensor):
        raise EMContractError(
            "EM unit adapter input must be a torch.Tensor",
            object_name=adapter,
            field="value",
            expected="torch.Tensor",
            actual=type(value).__qualname__,
            details={
                "adapter": adapter,
                "received_type": type(value).__qualname__,
                "remediation": "pass a floating or native-complex torch.Tensor",
            },
        )
    if not (value.is_floating_point() or value.is_complex()):
        raise EMContractError(
            "EM unit adapter dtype is unsupported",
            object_name=adapter,
            field="value.dtype",
            expected="floating or native-complex tensor",
            actual=str(value.dtype),
            code=ErrorCode.DTYPE_UNSUPPORTED,
            details={
                "adapter": adapter,
                "dtype": str(value.dtype),
                "remediation": "pass a floating or native-complex torch.Tensor",
            },
        )
    return value


def _scale_with_representability(
    value: torch.Tensor,
    *,
    factor: float,
    adapter: str,
) -> torch.Tensor:
    """Scale in the native dtype or reject a lossy representability boundary."""
    real_dtype = value.real.dtype if value.is_complex() else value.dtype
    native_factor = torch.tensor(factor, dtype=real_dtype, device=value.device)
    gradient_scale = (
        "nonfinite"
        if not bool(torch.isfinite(native_factor))
        else "zero"
        if not bool(native_factor != 0)
        else None
    )
    if gradient_scale is not None:
        failure = "overflow" if gradient_scale == "nonfinite" else "underflow"
        raise EMContractError(
            "EM adapter gradient scale is not representable in the input dtype",
            object_name=adapter,
            field="value.dtype",
            expected="finite nonzero gradient scale in the input real dtype",
            actual=str(value.dtype),
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="use a wider supported dtype with a representable gradient scale",
            details={
                "adapter": adapter,
                "dtype": str(value.dtype),
                "failure": failure,
                "gradient_scale": gradient_scale,
                "scale_factor": factor,
                "remediation": (
                    "use a wider supported dtype whose native gradient scale is "
                    "finite and nonzero"
                ),
            },
        )
    result = value * native_factor
    source_components = (value.real, value.imag) if value.is_complex() else (value,)
    result_components = (result.real, result.imag) if result.is_complex() else (result,)
    overflow = any(
        bool(torch.any(torch.isfinite(source) & ~torch.isfinite(converted)))
        for source, converted in zip(source_components, result_components)
    )
    underflow = any(
        bool(
            torch.any(
                torch.isfinite(source)
                & (source != 0)
                & (converted == 0)
            )
        )
        for source, converted in zip(source_components, result_components)
    )
    if overflow or underflow:
        failure = (
            "overflow_and_underflow"
            if overflow and underflow
            else "overflow"
            if overflow
            else "underflow"
        )
        raise EMContractError(
            "EM adapter result is not representable in the input dtype",
            object_name=adapter,
            field="value.dtype",
            expected="finite nonzero scaled values representable in the input dtype",
            actual=str(value.dtype),
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="use a wider supported dtype or a representable value range",
            details={
                "adapter": adapter,
                "dtype": str(value.dtype),
                "failure": failure,
                "scale_factor": factor,
                "remediation": (
                    "use a wider supported dtype or values whose scaled result "
                    "is representable"
                ),
            },
        )
    return result


def tesla_to_nanotesla(value: torch.Tensor) -> torch.Tensor:
    """Convert magnetic flux density from T to nT at a display boundary."""
    tensor = _require_scalable_tensor(value, adapter="tesla_to_nanotesla")
    return _scale_with_representability(
        tensor,
        factor=1.0e9,
        adapter="tesla_to_nanotesla",
    )


def nanotesla_to_tesla(value: torch.Tensor) -> torch.Tensor:
    """Convert magnetic flux density from nT to canonical T."""
    tensor = _require_scalable_tensor(value, adapter="nanotesla_to_tesla")
    return _scale_with_representability(
        tensor,
        factor=1.0e-9,
        adapter="nanotesla_to_tesla",
    )


def chargeability_to_mv_per_v(value: torch.Tensor) -> torch.Tensor:
    """Convert dimensionless chargeability to the industry mV/V display unit."""
    tensor = _require_scalable_tensor(value, adapter="chargeability_to_mv_per_v")
    return _scale_with_representability(
        tensor,
        factor=1.0e3,
        adapter="chargeability_to_mv_per_v",
    )


def mv_per_v_to_chargeability(value: torch.Tensor) -> torch.Tensor:
    """Convert industry mV/V to canonical dimensionless chargeability."""
    tensor = _require_scalable_tensor(value, adapter="mv_per_v_to_chargeability")
    return _scale_with_representability(
        tensor,
        factor=1.0e-3,
        adapter="mv_per_v_to_chargeability",
    )


__all__ = [
    "chargeability_to_mv_per_v",
    "mv_per_v_to_chargeability",
    "nanotesla_to_tesla",
    "tesla_to_nanotesla",
]
