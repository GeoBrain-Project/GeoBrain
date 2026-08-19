"""Explicit differentiable conversions between industry units and SI.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import torch

from geobrain.core import ErrorCode

from .contracts import require_compatible_tensors
from .errors import RockContractError


def _validated_value(object_name: str, value: torch.Tensor) -> torch.Tensor:
    (result,) = require_compatible_tensors(object_name, ("value", value))
    if result.layout is not torch.strided:
        raise RockContractError(
            "Rock unit conversion requires a strided tensor",
            object_name=object_name,
            field="value",
            expected="torch.strided layout",
            actual=str(result.layout),
        )
    if result.device.type == "meta":
        raise RockContractError(
            "Rock unit conversion requires materialized values",
            object_name=object_name,
            field="value",
            expected="a materialized CPU or accelerator tensor",
            actual=str(result.device),
            code=ErrorCode.DEVICE_UNAVAILABLE,
        )
    if not bool(torch.isfinite(result).all()):
        raise RockContractError(
            "Rock unit conversion requires finite values",
            object_name=object_name,
            field="value",
            expected="finite values",
            actual="non-finite value(s)",
        )
    return result


def celsius_to_kelvin(value: torch.Tensor) -> torch.Tensor:
    """Convert degrees Celsius to kelvin without changing dtype or device."""

    value = _validated_value("celsius_to_kelvin", value)
    absolute_zero = value.new_tensor(-273.15)
    if bool(torch.any(value < absolute_zero)):
        raise RockContractError(
            "temperature is below absolute zero",
            object_name="celsius_to_kelvin",
            field="value",
            expected=">= -273.15 degrees Celsius",
            actual={"minimum": value.amin().item()},
        )
    return value + value.new_tensor(273.15)


def kelvin_to_celsius(value: torch.Tensor) -> torch.Tensor:
    """Convert kelvin to degrees Celsius without changing dtype or device."""

    value = _validated_value("kelvin_to_celsius", value)
    if bool(torch.any(value < 0.0)):
        raise RockContractError(
            "absolute temperature cannot be negative",
            object_name="kelvin_to_celsius",
            field="value",
            expected=">= 0 K",
            actual={"minimum": value.amin().item()},
        )
    return value - value.new_tensor(273.15)


def megapascal_to_pascal(value: torch.Tensor) -> torch.Tensor:
    """Convert megapascals to pascals."""

    value = _validated_value("megapascal_to_pascal", value)
    return value * value.new_tensor(1.0e6)


def pascal_to_megapascal(value: torch.Tensor) -> torch.Tensor:
    """Convert pascals to megapascals for explicit display boundaries."""

    value = _validated_value("pascal_to_megapascal", value)
    return value / value.new_tensor(1.0e6)


def gigapascal_to_pascal(value: torch.Tensor) -> torch.Tensor:
    """Convert gigapascals to pascals."""

    value = _validated_value("gigapascal_to_pascal", value)
    return value * value.new_tensor(1.0e9)


def pascal_to_gigapascal(value: torch.Tensor) -> torch.Tensor:
    """Convert pascals to gigapascals for explicit display boundaries."""

    value = _validated_value("pascal_to_gigapascal", value)
    return value / value.new_tensor(1.0e9)


def gram_per_cubic_centimetre_to_kilogram_per_cubic_metre(
    value: torch.Tensor,
) -> torch.Tensor:
    """Convert g/cm³ to kg/m³."""  # SI-EXEMPT: explicit boundary converter

    value = _validated_value(
        "gram_per_cubic_centimetre_to_kilogram_per_cubic_metre", value
    )
    return value * value.new_tensor(1.0e3)


def kilogram_per_cubic_metre_to_gram_per_cubic_centimetre(
    value: torch.Tensor,
) -> torch.Tensor:
    """Convert kg/m³ to g/cm³ for explicit display boundaries."""  # SI-EXEMPT

    value = _validated_value(
        "kilogram_per_cubic_metre_to_gram_per_cubic_centimetre", value
    )
    return value / value.new_tensor(1.0e3)


__all__ = [
    "celsius_to_kelvin",
    "gigapascal_to_pascal",
    "gram_per_cubic_centimetre_to_kilogram_per_cubic_metre",
    "kelvin_to_celsius",
    "kilogram_per_cubic_metre_to_gram_per_cubic_centimetre",
    "megapascal_to_pascal",
    "pascal_to_gigapascal",
    "pascal_to_megapascal",
]
