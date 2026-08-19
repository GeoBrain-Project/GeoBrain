"""Explicit Potential display-unit adapters at the SI presentation boundary.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core import ErrorCode

from .errors import (
    PotentialCapabilityError,
    PotentialContractError,
    PotentialNumericsError,
)


def _convert_display_units(
    value: object,
    *,
    factor: int,
    field: str,
    output_field: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise PotentialContractError(
            "Display-unit adapters require a torch.Tensor.",
            object_name="PotentialDisplayAdapter",
            field=field,
            expected="finite real floating torch.Tensor",
            actual=value,
            hint="Pass the SI result tensor directly to the explicit display adapter.",
        )
    if value.layout != torch.strided:
        raise PotentialCapabilityError(
            "Display-unit adapters require a strided tensor layout.",
            object_name="PotentialDisplayAdapter",
            field=field,
            expected="torch.strided",
            actual=value.layout,
            code=ErrorCode.CAPABILITY_UNAVAILABLE,
            hint="Materialize SI values in a strided tensor before display conversion.",
        )
    if not value.is_floating_point():
        raise PotentialContractError(
            "Display-unit adapters require a real floating tensor.",
            object_name="PotentialDisplayAdapter",
            field=field,
            expected="torch.float32 or torch.float64",
            actual=value.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Keep physical SI results in a real floating tensor before conversion.",
        )
    if value.dtype not in {torch.float32, torch.float64}:
        raise PotentialCapabilityError(
            "Display-unit adapters support only float32 and float64 tensors.",
            object_name="PotentialDisplayAdapter",
            field=field,
            expected=["torch.float32", "torch.float64"],
            actual=value.dtype,
            code=ErrorCode.DTYPE_UNSUPPORTED,
            hint="Use dtype=torch.float32 or dtype=torch.float64 for SI computation and display conversion.",
        )
    if value.device.type == "meta":
        raise PotentialContractError(
            "Display-unit adapters require a materialized tensor device.",
            object_name="PotentialDisplayAdapter",
            field=field,
            expected="materialized CPU or accelerator tensor",
            actual=value.device,
            code=ErrorCode.DEVICE_UNAVAILABLE,
            hint="Materialize the SI tensor on its execution device before conversion.",
        )
    if not bool(torch.isfinite(value).all()):
        raise PotentialContractError(
            "Display-unit adapters require finite SI values.",
            object_name="PotentialDisplayAdapter",
            field=field,
            expected="all finite values",
            actual="contains non-finite values",
            hint="Resolve non-finite physical results before presentation conversion.",
        )
    if value.numel() > 0:
        maximum_input_magnitude = float(value.detach().abs().max().item())
        maximum_supported_magnitude = torch.finfo(value.dtype).max / factor
        if maximum_input_magnitude > maximum_supported_magnitude:
            raise PotentialNumericsError(
                "Display-unit conversion would overflow the output dtype.",
                object_name="PotentialDisplayAdapter",
                field=output_field,
                expected=f"finite {value.dtype} output after scaling by {factor}",
                actual={
                    "dtype": value.dtype,
                    "factor": factor,
                    "observed_maximum_input_magnitude": maximum_input_magnitude,
                },
                hint="Reduce the SI input magnitude or select a supported dtype with sufficient range.",
            )
    converted = value * factor
    if not bool(torch.isfinite(converted).all()):
        raise PotentialNumericsError(
            "Display-unit conversion produced non-finite output.",
            object_name="PotentialDisplayAdapter",
            field=output_field,
            expected=f"finite {value.dtype} output after scaling by {factor}",
            actual={"dtype": value.dtype, "factor": factor},
            hint="Reduce the SI input magnitude so the converted values remain finite.",
        )
    return converted


def gravity_to_mgal(value_m_per_s2: torch.Tensor) -> torch.Tensor:
    """Convert gravity acceleration from m/s² to mGal without detaching."""
    return _convert_display_units(
        value_m_per_s2,
        factor=100_000,
        field="value_m_per_s2",
        output_field="value_mgal",
    )


def magnetic_to_nt(value_tesla: torch.Tensor) -> torch.Tensor:
    """Convert magnetic flux density from tesla to nT without detaching."""
    return _convert_display_units(
        value_tesla,
        factor=1_000_000_000,
        field="value_tesla",
        output_field="value_nt",
    )


def gravity_gradient_to_eotvos(value_per_s2: torch.Tensor) -> torch.Tensor:
    """Convert gravity gradient from s⁻² to Eotvos without detaching."""
    return _convert_display_units(
        value_per_s2,
        factor=1_000_000_000,
        field="value_per_s2",
        output_field="value_eotvos",
    )


__all__ = [
    "gravity_gradient_to_eotvos",
    "gravity_to_mgal",
    "magnetic_to_nt",
]
