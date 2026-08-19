"""Immutable capability reports for the rock family (family-template module).

The family-template counterpart of ``contracts.py``, matching the
platform spine: ``capabilities.py`` holds
``RockCapabilityReport`` / ``RockUnsupportedCombination``, ``contracts.py``
keeps ``RockFieldSpec``, tensor contracts, and iteration reports, the
shape ``flow/`` already has.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from geobrain.core import DifferentiabilityLevel

from .contracts import (
    RockFieldSpec,
    _nonempty_string,
    _pair_tuple,
    _string_tuple,
)
from .errors import RockContractError

__all__ = ["RockCapabilityReport", "RockUnsupportedCombination"]


@dataclass(frozen=True, slots=True)
class RockUnsupportedCombination:
    """One immutable capability selection that is explicitly unsupported."""

    selection: tuple[tuple[str, str], ...]
    reason: str
    remediation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection",
            _pair_tuple(
                self.selection,
                "selection",
                object_name="RockUnsupportedCombination",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "reason",
            _nonempty_string(
                self.reason,
                "reason",
                object_name="RockUnsupportedCombination",
            ),
        )
        object.__setattr__(
            self,
            "remediation",
            _nonempty_string(
                self.remediation,
                "remediation",
                object_name="RockUnsupportedCombination",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-ready unsupported-selection metadata."""

        return {
            "selection": [list(pair) for pair in self.selection],
            "reason": self.reason,
            "remediation": self.remediation,
        }


def _field_specs(
    values: object,
    field: str,
    *,
    object_name: str,
    allow_empty: bool,
) -> tuple[RockFieldSpec, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RockContractError(
            "invalid Rock field declaration",
            object_name=object_name,
            field=field,
            expected="sequence of RockFieldSpec entries",
            actual=values,
        )
    owned: list[RockFieldSpec] = []
    for item in values:
        if not isinstance(item, RockFieldSpec):
            raise RockContractError(
                "invalid Rock field declaration",
                object_name=object_name,
                field=field,
                expected="RockFieldSpec entries",
                actual=values,
            )
        owned.append(item)
    result = tuple(owned)
    if not allow_empty and not result:
        raise RockContractError(
            "Rock field declaration cannot be empty",
            object_name=object_name,
            field=field,
            expected="at least one RockFieldSpec",
            actual=result,
        )
    names = tuple(item.name for item in result)
    if len(names) != len(set(names)):
        raise RockContractError(
            "Rock field declaration contains duplicate names",
            object_name=object_name,
            field=field,
            expected="unique field names",
            actual=names,
        )
    return result


@dataclass(frozen=True, slots=True)
class RockCapabilityReport:
    """Validated, immutable, JSON-safe description of one Rock model."""

    physics: Literal["rock"]
    model: str
    subfamily: str
    maturity: Literal["production", "experimental"]
    required_model_fields: tuple[RockFieldSpec, ...]
    output_fields: tuple[RockFieldSpec, ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    broadcast: bool
    complex_outputs: tuple[str, ...]
    differentiability: DifferentiabilityLevel
    differentiable_model_fields: tuple[str, ...]
    iterative: bool
    resource_estimate_supported: bool
    citation: str
    calibrated_domain: tuple[tuple[str, str], ...]
    unsupported: tuple[RockUnsupportedCombination, ...]

    def __post_init__(self) -> None:
        if self.physics != "rock":
            raise RockContractError(
                "invalid Rock capability physics",
                object_name="RockCapabilityReport",
                field="physics",
                expected="rock",
                actual=self.physics,
            )
        object.__setattr__(
            self,
            "model",
            _nonempty_string(
                self.model,
                "model",
                object_name="RockCapabilityReport",
            ),
        )
        object.__setattr__(
            self,
            "subfamily",
            _nonempty_string(
                self.subfamily,
                "subfamily",
                object_name="RockCapabilityReport",
            ),
        )
        if self.maturity != "production" and self.maturity != "experimental":
            raise RockContractError(
                "invalid Rock capability maturity",
                object_name="RockCapabilityReport",
                field="maturity",
                expected="production or experimental",
                actual=self.maturity,
            )
        required = _field_specs(
            self.required_model_fields,
            "required_model_fields",
            object_name="RockCapabilityReport",
            allow_empty=False,
        )
        outputs = _field_specs(
            self.output_fields,
            "output_fields",
            object_name="RockCapabilityReport",
            allow_empty=False,
        )
        object.__setattr__(self, "required_model_fields", required)
        object.__setattr__(self, "output_fields", outputs)

        dtypes = _string_tuple(
            self.dtypes,
            "dtypes",
            object_name="RockCapabilityReport",
            allow_empty=False,
        )
        if any(dtype not in {"float32", "float64"} for dtype in dtypes):
            raise RockContractError(
                "invalid Rock capability dtype",
                object_name="RockCapabilityReport",
                field="dtypes",
                expected="float32 and/or float64",
                actual=dtypes,
            )
        object.__setattr__(self, "dtypes", dtypes)
        object.__setattr__(
            self,
            "devices",
            _string_tuple(
                self.devices,
                "devices",
                object_name="RockCapabilityReport",
                allow_empty=False,
            ),
        )
        if type(self.broadcast) is not bool:
            raise RockContractError(
                "invalid Rock broadcast capability",
                object_name="RockCapabilityReport",
                field="broadcast",
                expected="Boolean",
                actual=self.broadcast,
            )

        complex_outputs = _string_tuple(
            self.complex_outputs,
            "complex_outputs",
            object_name="RockCapabilityReport",
        )
        output_names = {item.name for item in outputs}
        if not set(complex_outputs) <= output_names:
            raise RockContractError(
                "complex Rock outputs must be declared outputs",
                object_name="RockCapabilityReport",
                field="complex_outputs",
                expected=sorted(output_names),
                actual=complex_outputs,
            )
        object.__setattr__(self, "complex_outputs", complex_outputs)

        if not isinstance(self.differentiability, DifferentiabilityLevel):
            raise RockContractError(
                "invalid Rock differentiability level",
                object_name="RockCapabilityReport",
                field="differentiability",
                expected="DifferentiabilityLevel",
                actual=self.differentiability,
            )
        differentiable = _string_tuple(
            self.differentiable_model_fields,
            "differentiable_model_fields",
            object_name="RockCapabilityReport",
        )
        required_names = {item.name for item in required}
        if not set(differentiable) <= required_names:
            raise RockContractError(
                "differentiable Rock fields must be required model fields",
                object_name="RockCapabilityReport",
                field="differentiable_model_fields",
                expected=sorted(required_names),
                actual=differentiable,
            )
        object.__setattr__(self, "differentiable_model_fields", differentiable)

        for field, value in (
            ("iterative", self.iterative),
            ("resource_estimate_supported", self.resource_estimate_supported),
        ):
            if type(value) is not bool:
                raise RockContractError(
                    "invalid Rock Boolean capability",
                    object_name="RockCapabilityReport",
                    field=field,
                    expected="Boolean",
                    actual=value,
                )
        object.__setattr__(
            self,
            "citation",
            _nonempty_string(
                self.citation,
                "citation",
                object_name="RockCapabilityReport",
            ),
        )
        object.__setattr__(
            self,
            "calibrated_domain",
            _pair_tuple(
                self.calibrated_domain,
                "calibrated_domain",
                object_name="RockCapabilityReport",
            ),
        )
        if not isinstance(self.unsupported, Sequence) or isinstance(
            self.unsupported,
            (str, bytes),
        ):
            raise RockContractError(
                "invalid unsupported Rock combination",
                object_name="RockCapabilityReport",
                field="unsupported",
                expected="sequence of RockUnsupportedCombination entries",
                actual=self.unsupported,
            )
        unsupported_items: list[RockUnsupportedCombination] = []
        for item in self.unsupported:
            if not isinstance(item, RockUnsupportedCombination):
                raise RockContractError(
                    "invalid unsupported Rock combination",
                    object_name="RockCapabilityReport",
                    field="unsupported",
                    expected="RockUnsupportedCombination entries",
                    actual=self.unsupported,
                )
            unsupported_items.append(item)
        unsupported = tuple(unsupported_items)
        object.__setattr__(self, "unsupported", unsupported)

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-ready capability metadata."""

        return {
            "physics": self.physics,
            "model": self.model,
            "subfamily": self.subfamily,
            "maturity": self.maturity,
            "required_model_fields": [
                field.to_dict() for field in self.required_model_fields
            ],
            "output_fields": [field.to_dict() for field in self.output_fields],
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "broadcast": self.broadcast,
            "complex_outputs": list(self.complex_outputs),
            "differentiability": self.differentiability.value,
            "differentiable_model_fields": list(self.differentiable_model_fields),
            "iterative": self.iterative,
            "resource_estimate_supported": self.resource_estimate_supported,
            "citation": self.citation,
            "calibrated_domain": [list(pair) for pair in self.calibrated_domain],
            "unsupported": [item.to_dict() for item in self.unsupported],
        }

