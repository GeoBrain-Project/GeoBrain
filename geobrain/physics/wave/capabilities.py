"""Pure-data capability and unsupported-combination records for Wave.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn

from .errors import WaveContractError


def _invalid(
    field: str,
    expected: object,
    actual: object,
    *,
    object_name: str,
) -> NoReturn:
    """Raise a consistent structured error for invalid report data."""
    raise WaveContractError(
        "invalid Wave capability report value",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _tuple_of_strings(
    value: object, *, field_name: str, object_name: str
) -> tuple[str, ...]:
    """Own a string sequence as an immutable tuple."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(field_name, "sequence of strings", value, object_name=object_name)
    result = tuple(value)
    if not all(isinstance(item, str) for item in result):
        _invalid(field_name, "sequence of strings", value, object_name=object_name)
    return result


def _tuple_of_string_pairs(
    value: object, *, field_name: str, object_name: str, unique_keys: bool = False
) -> tuple[tuple[str, str], ...]:
    """Own ordered string pairs and optionally require JSON-object-safe keys."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(field_name, "sequence of string pairs", value, object_name=object_name)
    result = tuple(value)
    if any(
        not isinstance(pair, Sequence)
        or isinstance(pair, (str, bytes, bytearray))
        or len(pair) != 2
        or not all(isinstance(item, str) for item in pair)
        for pair in result
    ):
        _invalid(field_name, "sequence of string pairs", value, object_name=object_name)
    pairs = tuple((pair[0], pair[1]) for pair in result)
    if unique_keys and len({name for name, _ in pairs}) != len(pairs):
        _invalid(field_name, "string pairs with unique names", value, object_name=object_name)
    return pairs


@dataclass(frozen=True, slots=True)
class WaveUnsupportedCombination:
    """A deterministic capability-selection diagnosis and its remediation.

    Attributes:
        selection: the requested feature combination.
        reason: why it is refused.
        remediation: what to change to make it runnable.
    """

    selection: tuple[tuple[str, str], ...]
    reason: str
    remediation: str

    def __post_init__(self) -> None:
        """Own selection pairs so the diagnostic remains immutable and JSON-safe."""
        object.__setattr__(
            self,
            "selection",
            _tuple_of_string_pairs(
                self.selection,
                field_name="selection",
                object_name=type(self).__name__,
                unique_keys=True,
            ),
        )
        if not isinstance(self.reason, str):
            _invalid("reason", "string", self.reason, object_name=type(self).__name__)
        if not isinstance(self.remediation, str):
            _invalid("remediation", "string", self.remediation, object_name=type(self).__name__)

    def to_dict(self) -> dict[str, object]:
        """Return an insertion-ordered JSON object for the unsupported selection."""
        return {
            "selection": {name: value for name, value in self.selection},
            "reason": self.reason,
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class WaveCapabilityReport:
    """Stable pure-data description of Wave support and known exclusions.

    Attributes:
        physics / equation / dimension / maturity: operator identity.
        required_model_fields: ModelState fields the kernel consumes.
        components: emitted output components.
        dtypes / devices / backends / boundaries / memory_strategies:
            supported execution axes.
        differentiable_model_fields / differentiable_wavelets: gradient
            surface declarations.
        mesh_capabilities: required ctx-mesh capability markers.
        resource_estimate_supported: preflight estimation availability.
        unsupported: explicit refused combinations.
    """

    physics: str
    equation: str
    dimension: int
    maturity: Literal["production", "experimental"]
    required_model_fields: tuple[tuple[str, str], ...]
    components: tuple[str, ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    backends: tuple[str, ...]
    boundaries: tuple[str, ...]
    memory_strategies: tuple[str, ...]
    differentiable_model_fields: tuple[str, ...]
    differentiable_wavelets: bool
    mesh_capabilities: tuple[str, ...]
    resource_estimate_supported: bool
    unsupported: tuple[WaveUnsupportedCombination, ...]

    def __post_init__(self) -> None:
        """Own all ordered values and reject malformed report records."""
        name = type(self).__name__
        if not isinstance(self.physics, str):
            _invalid("physics", "string", self.physics, object_name=name)
        if not isinstance(self.equation, str):
            _invalid("equation", "string", self.equation, object_name=name)
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool) or self.dimension <= 0:
            _invalid("dimension", "positive integer", self.dimension, object_name=name)
        if self.maturity not in ("production", "experimental"):
            _invalid("maturity", "'production' or 'experimental'", self.maturity, object_name=name)
        object.__setattr__(
            self,
            "required_model_fields",
            _tuple_of_string_pairs(
                self.required_model_fields,
                field_name="required_model_fields",
                object_name=name,
                unique_keys=True,
            ),
        )
        for field_name in (
            "components",
            "dtypes",
            "devices",
            "backends",
            "boundaries",
            "memory_strategies",
            "differentiable_model_fields",
            "mesh_capabilities",
        ):
            object.__setattr__(
                self,
                field_name,
                _tuple_of_strings(getattr(self, field_name), field_name=field_name, object_name=name),
            )
        if not isinstance(self.differentiable_wavelets, bool):
            _invalid(
                "differentiable_wavelets", "boolean", self.differentiable_wavelets, object_name=name
            )
        if not isinstance(self.resource_estimate_supported, bool):
            _invalid(
                "resource_estimate_supported",
                "boolean",
                self.resource_estimate_supported,
                object_name=name,
            )
        if isinstance(self.unsupported, (str, bytes, bytearray)) or not isinstance(
            self.unsupported, Sequence
        ):
            _invalid("unsupported", "sequence of WaveUnsupportedCombination", self.unsupported, object_name=name)
        unsupported = tuple(self.unsupported)
        if not all(isinstance(item, WaveUnsupportedCombination) for item in unsupported):
            _invalid("unsupported", "sequence of WaveUnsupportedCombination", self.unsupported, object_name=name)
        object.__setattr__(self, "unsupported", unsupported)

    def to_dict(self) -> dict[str, object]:
        """Return report fields in their declared order using only JSON primitives."""
        return {
            "physics": self.physics,
            "equation": self.equation,
            "dimension": self.dimension,
            "maturity": self.maturity,
            "required_model_fields": [
                {"name": name, "unit": unit} for name, unit in self.required_model_fields
            ],
            "components": list(self.components),
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "backends": list(self.backends),
            "boundaries": list(self.boundaries),
            "memory_strategies": list(self.memory_strategies),
            "differentiable_model_fields": list(self.differentiable_model_fields),
            "differentiable_wavelets": self.differentiable_wavelets,
            "mesh_capabilities": list(self.mesh_capabilities),
            "resource_estimate_supported": self.resource_estimate_supported,
            "unsupported": [item.to_dict() for item in self.unsupported],
        }


__all__ = ["WaveCapabilityReport", "WaveUnsupportedCombination"]
