"""Allocation-free capability contracts for Potential execution.

The resource-estimation half lives in the sibling ``resources.py``
(family-template alignment): this module keeps the capability report and
the operator input schema.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from .config import PotentialStrategy
from .errors import PotentialContractError


SelectedStrategy = Literal["dense", "tiled", "store"]
PotentialOperatorId = Literal[
    "gravity-2d",
    "gravity-3d",
    "magnetic-3d",
    "magnetic-vector-3d",
]
_CAPABILITY_SCHEMA: Literal["geobrain.potential.capability/1.0"] = (
    "geobrain.potential.capability/1.0"
)

@dataclass(frozen=True, slots=True)
class PotentialCapabilityReport:
    """Stable JSON-safe metadata for one public Potential operator.

    Attributes:
        schema_version: report schema tag.
        operator_id: which operator the report describes.
        supported_components / output_units: emission surface.
        supported_devices / supported_dtypes: execution axes.
        strategies / differentiability_by_strategy: evaluation strategies
            and their gradient levels.
        required_state_fields: ModelState fields the forward consumes.
        geometry_contract: mesh/survey frame requirements.
        deterministic: bitwise-reproducibility declaration.
    """

    schema_version: Literal["geobrain.potential.capability/1.0"]
    operator_id: PotentialOperatorId
    supported_components: tuple[str, ...]
    output_units: Mapping[str, str]
    supported_devices: tuple[str, ...]
    supported_dtypes: tuple[str, ...]
    strategies: tuple[PotentialStrategy, ...]
    differentiability_by_strategy: Mapping[str, str]
    required_state_fields: Mapping[str, str]
    geometry_contract: Mapping[str, object]
    deterministic: bool

    def to_dict(self) -> dict[str, object]:
        """Return the closed transport form used by Agent clients."""
        return {
            "schema_version": self.schema_version,
            "operator_id": self.operator_id,
            "supported_components": list(self.supported_components),
            "output_units": dict(self.output_units),
            "supported_devices": list(self.supported_devices),
            "supported_dtypes": list(self.supported_dtypes),
            "strategies": list(self.strategies),
            "differentiability_by_strategy": dict(self.differentiability_by_strategy),
            "required_state_fields": dict(self.required_state_fields),
            "geometry_contract": dict(self.geometry_contract),
            "deterministic": self.deterministic,
        }


def potential_capability_report(
    *,
    operator_id: PotentialOperatorId,
    components: tuple[str, ...],
    output_units: Mapping[str, str],
    state_fields: Mapping[str, str],
    survey_width: int,
) -> PotentialCapabilityReport:
    """Build one consistently ordered public capability report."""
    return PotentialCapabilityReport(
        schema_version=_CAPABILITY_SCHEMA,
        operator_id=operator_id,
        supported_components=components,
        output_units=dict(output_units),
        supported_devices=("cpu", "cuda"),
        supported_dtypes=("float32", "float64"),
        strategies=("auto", "dense", "tiled", "store"),
        differentiability_by_strategy={
            "auto": "custom_vjp",
            "dense": "full_autograd",
            "tiled": "custom_vjp",
            "store": "full_autograd",
        },
        required_state_fields=dict(state_fields),
        geometry_contract={
            "mesh": "TensorMesh in core depth,x,y order",
            "survey_columns": ["x", "z"] if survey_width == 2 else ["x", "y", "z"],
            "coordinate_unit": "m",
            "vertical_axis": "elevation-positive-up",
        },
        deterministic=True,
    )


def potential_input_schema(
    *,
    operator_id: PotentialOperatorId,
    components: tuple[str, ...],
    tensor_name: Literal["rho", "chi", "magnetization"],
    survey_width: int,
    field_name: Literal["earth_field", "projection_field"] | None = None,
) -> dict[str, object]:
    """Return the closed Draft 2020-12 input schema for one operator."""
    finite_number: dict[str, object] = {"type": "number", "x-geobrain-finite": True}
    positive_number = {**finite_number, "exclusiveMinimum": 0.0}
    coordinate_row = {
        "type": "array",
        "minItems": survey_width,
        "maxItems": survey_width,
        "items": finite_number,
        "x-geobrain-unit": "m",
        "x-geobrain-finite": True,
    }
    if tensor_name == "magnetization":
        tensor_items: dict[str, object] = {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": finite_number,
            "x-geobrain-finite": True,
        }
        tensor_unit = "A/m"
    else:
        tensor_items = finite_number
        tensor_unit = "kg/m^3" if tensor_name == "rho" else "1"
    properties: dict[str, object] = {
        "mesh": {
            "type": "object",
            "additionalProperties": False,
            "required": ["origin_m", "cell_widths_m"],
            "properties": {
                "origin_m": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": finite_number,
                    "x-geobrain-unit": "m",
                    "x-geobrain-finite": True,
                },
                "cell_widths_m": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": positive_number,
                        "x-geobrain-unit": "m",
                        "x-geobrain-finite": True,
                    },
                },
            },
        },
        "survey": {
            "type": "object",
            "additionalProperties": False,
            "required": ["positions_m"],
            "properties": {
                "positions_m": {
                    "type": "array",
                    "minItems": 1,
                    "items": coordinate_row,
                    "x-geobrain-unit": "m",
                    "x-geobrain-finite": True,
                }
            },
        },
        "tensors": {
            "type": "object",
            "additionalProperties": False,
            "required": [tensor_name],
            "properties": {
                tensor_name: {
                    "type": "array",
                    "minItems": 1,
                    "items": tensor_items,
                    "x-geobrain-unit": tensor_unit,
                    "x-geobrain-finite": True,
                }
            },
        },
        "components": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "enum": list(components)},
        },
        "execution": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "strategy",
                "budget_bytes",
                "observation_tile_size",
                "cell_tile_size",
            ],
            "properties": {
                "strategy": {"enum": ["auto", "dense", "tiled", "store"]},
                "budget_bytes": {"type": "integer", "minimum": 1},
                "observation_tile_size": {
                    "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
                },
                "cell_tile_size": {
                    "anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]
                },
            },
        },
    }
    if field_name is not None:
        properties[field_name] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["intensity_tesla", "inclination_deg", "declination_deg"],
            "properties": {
                "intensity_tesla": {**positive_number, "x-geobrain-unit": "T"},
                "inclination_deg": {**finite_number, "minimum": -90.0, "maximum": 90.0},
                "declination_deg": finite_number,
            },
        }
    required = ["mesh", "survey", "tensors", "components", "execution"]
    if field_name is not None:
        required.append(field_name)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"geobrain.potential.input/{operator_id}/1.0",
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


__all__ = [
    "PotentialCapabilityReport",
    "potential_capability_report",
    "potential_input_schema",
]
