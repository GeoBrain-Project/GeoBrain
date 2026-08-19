"""Immutable unit-aware model schemas for the Flow family.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .errors import FlowCapabilityError, FlowContractError

_MODEL_SCHEMA_VERSION = "geobrain.flow.model-schema/1.0"
_FIELD_ROLES = frozenset({"primary", "property", "source", "result"})
_BINDING_ENTRYPOINTS = frozenset({"constructor", "residual"})
_RESIDUAL_UNITS = frozenset({"kg/s", "mol/s", "W", "m³/s", "STB/day", "scf/day"})
_CONSERVATION_KINDS = frozenset({"mass", "molar", "energy", "surface-volume"})
_UNITS_BY_CONSERVATION = {
    "mass": frozenset({"kg/s"}),
    "molar": frozenset({"mol/s"}),
    "energy": frozenset({"W"}),
    "surface-volume": frozenset({"m³/s", "STB/day", "scf/day"}),
}
_SI_UNITS = frozenset(
    {
        "1",
        "Pa",
        "m",
        "m²",
        "m³",
        "m²/s",
        "Pa·s",
        "s",
        "kg",
        "kg/m³",
        "kg/s",
        "mol",
        "mol/m³",
        "mol/s",
        "Pa⁻¹",
        "K",
        "J",
        "J/(kg·K)",
        "J/(m³·K)",
        "W",
        "W/(m·K)",
        "m³/s",
    }
)
_FIELD_UNITS = frozenset({"1", "psi", "mD", "STB/day", "scf/day"})
_KNOWN_UNITS = _SI_UNITS | _FIELD_UNITS
_UNITS_BY_SYSTEM = {"SI": _SI_UNITS, "FIELD": _FIELD_UNITS}


def _nonempty_string(value: object, *, object_name: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FlowContractError(
            "Flow schema string must be non-empty",
            object_name=object_name,
            field=field,
            expected="non-empty string",
            actual=value,
        )
    return value


def _string_tuple(
    values: object,
    *,
    object_name: str,
    field: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise FlowContractError(
            "Flow schema field must be a string sequence",
            object_name=object_name,
            field=field,
            expected="sequence of non-empty strings",
            actual=values,
        )
    result = tuple(_nonempty_string(item, object_name=object_name, field=field) for item in values)
    if not allow_empty and not result:
        raise FlowContractError(
            "Flow schema sequence cannot be empty",
            object_name=object_name,
            field=field,
            expected="at least one entry",
            actual=result,
        )
    if len(result) != len(set(result)):
        raise FlowContractError(
            "Flow schema sequence contains duplicates",
            object_name=object_name,
            field=field,
            expected="unique entries",
            actual=result,
        )
    return result


@dataclass(frozen=True, slots=True)
class FlowFieldBinding:
    """One explicit route from a schema field to an executable parameter."""

    entrypoint: Literal["constructor", "residual"]
    parameter: str
    attribute_path: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        if self.entrypoint not in _BINDING_ENTRYPOINTS:
            raise FlowContractError(
                "invalid Flow field binding entrypoint",
                object_name=object_name,
                field="entrypoint",
                expected=sorted(_BINDING_ENTRYPOINTS),
                actual=self.entrypoint,
            )
        object.__setattr__(
            self,
            "parameter",
            _nonempty_string(self.parameter, object_name=object_name, field="parameter"),
        )
        object.__setattr__(
            self,
            "attribute_path",
            _string_tuple(
                self.attribute_path,
                object_name=object_name,
                field="attribute_path",
                allow_empty=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-native binding metadata."""
        return {
            "entrypoint": self.entrypoint,
            "parameter": self.parameter,
            "attribute_path": list(self.attribute_path),
        }


def _binding_tuple(values: object, *, object_name: str) -> tuple[FlowFieldBinding, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise FlowContractError(
            "Flow field bindings must be a sequence",
            object_name=object_name,
            field="bindings",
            expected="sequence of FlowFieldBinding",
            actual=values,
        )
    result = tuple(values)
    for item in result:
        if not isinstance(item, FlowFieldBinding):
            raise FlowContractError(
                "invalid Flow field execution binding",
                object_name=object_name,
                field="bindings",
                expected="FlowFieldBinding",
                actual=item,
            )
    keys = tuple((item.entrypoint, item.parameter, item.attribute_path) for item in result)
    if len(keys) != len(set(keys)):
        raise FlowContractError(
            "duplicate Flow field execution binding",
            object_name=object_name,
            field="bindings",
            expected="unique bindings",
            actual=keys,
        )
    return result


@dataclass(frozen=True, slots=True)
class FlowFieldSpec:
    """One named tensor field plus its explicit executable bindings.

    Attributes:
        name: field name.
        unit: declared SI unit string.
        axes: layout axes tag.
        role: primary / property / source classification.
        components: per-phase or per-component labels.
        bindings: where the field binds in the residual assembly.
    """

    name: str
    unit: str
    axes: tuple[str, ...]
    role: Literal["primary", "property", "source", "result"]
    components: tuple[str, ...] = ()
    bindings: tuple[FlowFieldBinding, ...] = ()

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        object.__setattr__(
            self,
            "name",
            _nonempty_string(self.name, object_name=object_name, field="name"),
        )
        unit = _nonempty_string(self.unit, object_name=object_name, field="unit")
        if unit not in _KNOWN_UNITS:
            raise FlowContractError(
                "Flow production field uses an unknown unit",
                object_name=object_name,
                field="unit",
                expected=sorted(_KNOWN_UNITS),
                actual=unit,
            )
        object.__setattr__(self, "unit", unit)
        object.__setattr__(
            self,
            "axes",
            _string_tuple(
                self.axes,
                object_name=object_name,
                field="axes",
                allow_empty=False,
            ),
        )
        if self.role not in _FIELD_ROLES:
            raise FlowContractError(
                "invalid Flow field role",
                object_name=object_name,
                field="role",
                expected=sorted(_FIELD_ROLES),
                actual=self.role,
            )
        object.__setattr__(
            self,
            "components",
            _string_tuple(
                self.components,
                object_name=object_name,
                field="components",
                allow_empty=True,
            ),
        )
        bindings = _binding_tuple(self.bindings, object_name=object_name)
        if self.role in {"property", "source"} and not bindings:
            raise FlowContractError(
                "Flow property/source field requires an execution binding",
                object_name=object_name,
                field="bindings",
                expected="at least one FlowFieldBinding",
                actual=bindings,
            )
        if self.role == "source" and any(item.entrypoint != "residual" for item in bindings):
            raise FlowContractError(
                "Flow source field bindings must target the residual entrypoint",
                object_name=object_name,
                field="bindings",
                expected="residual",
                actual=tuple(item.entrypoint for item in bindings),
            )
        object.__setattr__(self, "bindings", bindings)

    def to_dict(self) -> dict[str, object]:
        """Return detached JSON-native field metadata."""
        return {
            "name": self.name,
            "unit": self.unit,
            "axes": list(self.axes),
            "role": self.role,
            "components": list(self.components),
            "bindings": [item.to_dict() for item in self.bindings],
        }


@dataclass(frozen=True, slots=True)
class FlowResidualBlockSpec:
    """One conservative residual block and its associated primary field.

    Attributes:
        name: block name.
        unit: residual unit string.
        conservation: which quantity the block conserves.
        primary_field: the primary unknown the block solves for.
    """

    name: str
    unit: Literal["kg/s", "mol/s", "W", "m³/s", "STB/day", "scf/day"]
    conservation: Literal["mass", "molar", "energy", "surface-volume"]
    primary_field: str

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        for field in ("name", "primary_field"):
            object.__setattr__(
                self,
                field,
                _nonempty_string(getattr(self, field), object_name=object_name, field=field),
            )
        if self.unit not in _RESIDUAL_UNITS:
            raise FlowContractError(
                "invalid Flow residual unit",
                object_name=object_name,
                field="unit",
                expected=sorted(_RESIDUAL_UNITS),
                actual=self.unit,
            )
        if self.conservation not in _CONSERVATION_KINDS:
            raise FlowContractError(
                "invalid Flow residual conservation kind",
                object_name=object_name,
                field="conservation",
                expected=sorted(_CONSERVATION_KINDS),
                actual=self.conservation,
            )
        canonical_units = _UNITS_BY_CONSERVATION[self.conservation]
        if self.unit not in canonical_units:
            raise FlowContractError(
                "Flow residual conservation has the wrong unit",
                object_name=object_name,
                field="unit",
                expected=sorted(canonical_units),
                actual=self.unit,
            )

    def to_dict(self) -> dict[str, str]:
        """Return detached JSON-native residual metadata."""
        return {
            "name": self.name,
            "unit": self.unit,
            "conservation": self.conservation,
            "primary_field": self.primary_field,
        }


def _field_tuple(
    values: object,
    *,
    object_name: str,
    field: str,
    role: str,
    allow_empty: bool,
) -> tuple[FlowFieldSpec, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise FlowContractError(
            "Flow model field declaration must be a sequence",
            object_name=object_name,
            field=field,
            expected="sequence of FlowFieldSpec",
            actual=values,
        )
    result = tuple(values)
    if not allow_empty and not result:
        raise FlowContractError(
            "Flow model field declaration cannot be empty",
            object_name=object_name,
            field=field,
            expected="at least one FlowFieldSpec",
            actual=result,
        )
    for item in result:
        if not isinstance(item, FlowFieldSpec):
            raise FlowContractError(
                "invalid Flow model field declaration",
                object_name=object_name,
                field=field,
                expected="FlowFieldSpec",
                actual=item,
            )
        if item.role != role:
            raise FlowContractError(
                "Flow field role does not match its schema collection",
                object_name=object_name,
                field=item.name,
                expected=role,
                actual=item.role,
            )
    return result


def _residual_tuple(values: object, *, object_name: str) -> tuple[FlowResidualBlockSpec, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise FlowContractError(
            "Flow residual declaration must be a sequence",
            object_name=object_name,
            field="residual_blocks",
            expected="sequence of FlowResidualBlockSpec",
            actual=values,
        )
    result = tuple(values)
    for item in result:
        if not isinstance(item, FlowResidualBlockSpec):
            raise FlowContractError(
                "invalid Flow residual declaration",
                object_name=object_name,
                field="residual_blocks",
                expected="FlowResidualBlockSpec",
                actual=item,
            )
    return result


@dataclass(frozen=True, slots=True)
class FlowModelSchema:
    """Closed, immutable discovery contract for one executable Flow model.

    Attributes:
        schema_version: schema tag.
        model_name: which flow model this describes.
        primary_fields / property_fields / source_fields: the
            :class:`FlowFieldSpec` groups.
        residual_blocks: conservation-equation block specs.
        grid_kinds: supported grid kinds.
        phases / components: fluid system composition.
        unit_system: declared unit system of the schema.
    """

    schema_version: str
    model_name: str
    primary_fields: tuple[FlowFieldSpec, ...]
    property_fields: tuple[FlowFieldSpec, ...]
    source_fields: tuple[FlowFieldSpec, ...]
    residual_blocks: tuple[FlowResidualBlockSpec, ...]
    grid_kinds: tuple[str, ...]
    phases: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    unit_system: Literal["SI", "FIELD"] = "SI"

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        if self.unit_system not in _UNITS_BY_SYSTEM:
            raise FlowContractError(
                "invalid Flow model unit system",
                object_name=object_name,
                field="unit_system",
                expected=sorted(_UNITS_BY_SYSTEM),
                actual=self.unit_system,
            )
        if self.schema_version != _MODEL_SCHEMA_VERSION:
            raise FlowContractError(
                "unsupported Flow model schema version",
                object_name=object_name,
                field="schema_version",
                expected=_MODEL_SCHEMA_VERSION,
                actual=self.schema_version,
            )
        object.__setattr__(
            self,
            "model_name",
            _nonempty_string(self.model_name, object_name=object_name, field="model_name"),
        )
        object.__setattr__(
            self,
            "primary_fields",
            _field_tuple(
                self.primary_fields,
                object_name=object_name,
                field="primary_fields",
                role="primary",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "property_fields",
            _field_tuple(
                self.property_fields,
                object_name=object_name,
                field="property_fields",
                role="property",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "source_fields",
            _field_tuple(
                self.source_fields,
                object_name=object_name,
                field="source_fields",
                role="source",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "residual_blocks",
            _residual_tuple(self.residual_blocks, object_name=object_name),
        )
        for field in ("grid_kinds", "phases", "components"):
            object.__setattr__(
                self,
                field,
                _string_tuple(
                    getattr(self, field),
                    object_name=object_name,
                    field=field,
                    allow_empty=field != "grid_kinds",
                ),
            )
        all_fields = self.primary_fields + self.property_fields + self.source_fields
        allowed_units = _UNITS_BY_SYSTEM[self.unit_system]
        for item in all_fields:
            if item.unit not in allowed_units:
                raise FlowContractError(
                    f"{self.unit_system} Flow schema contains an incompatible unit",
                    object_name=self.model_name,
                    field=item.name,
                    expected=sorted(allowed_units),
                    actual=item.unit,
                )
        for block in self.residual_blocks:
            if block.unit not in allowed_units:
                raise FlowContractError(
                    f"{self.unit_system} Flow schema contains an incompatible residual unit",
                    object_name=self.model_name,
                    field=block.name,
                    expected=sorted(allowed_units),
                    actual=block.unit,
                )
        names = tuple(item.name for item in all_fields)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise FlowContractError(
                f"duplicate Flow field name: {duplicates[0]}",
                object_name=object_name,
                field="fields",
                expected="globally unique field names",
                actual=duplicates,
            )
        block_names = tuple(block.name for block in self.residual_blocks)
        if len(block_names) != len(set(block_names)):
            raise FlowContractError(
                "duplicate Flow residual block name",
                object_name=object_name,
                field="residual_blocks",
                expected="unique names",
                actual=block_names,
            )
        primary_names = {item.name for item in self.primary_fields}
        for block in self.residual_blocks:
            if block.primary_field not in primary_names:
                raise FlowContractError(
                    f"undeclared residual primary field: {block.primary_field}",
                    object_name=object_name,
                    field="primary_field",
                    expected=sorted(primary_names),
                    actual=block.primary_field,
                )
        covered = {block.primary_field for block in self.residual_blocks}
        uncovered = sorted(primary_names - covered)
        if uncovered:
            raise FlowContractError(
                f"state field has no residual block: {uncovered[0]}",
                object_name=object_name,
                field="residual_blocks",
                expected="at least one block per primary field",
                actual=uncovered,
            )

    def to_dict(self) -> dict[str, object]:
        """Return deterministic detached JSON-native model metadata."""
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "unit_system": self.unit_system,
            "primary_fields": [item.to_dict() for item in self.primary_fields],
            "property_fields": [item.to_dict() for item in self.property_fields],
            "source_fields": [item.to_dict() for item in self.source_fields],
            "residual_blocks": [item.to_dict() for item in self.residual_blocks],
            "grid_kinds": list(self.grid_kinds),
            "phases": list(self.phases),
            "components": list(self.components),
        }

    def require_canonical_si(self, *, object_name: str | None = None) -> None:
        """Reject execution paths that only accept the canonical SI contract."""
        if self.unit_system != "SI":
            raise FlowCapabilityError(
                "Flow execution requires a canonical SI model schema",
                object_name=object_name or self.model_name,
                field="unit_system",
                expected="SI",
                actual=self.unit_system,
                hint="Use an SI-native kernel or add an explicit SI-to-kernel adapter.",
            )


def _property_fields(
    *,
    unit_system: Literal["SI", "FIELD"],
    grid_kinds: tuple[str, ...],
    phases: tuple[str, ...],
    primary_names: frozenset[str],
) -> list[FlowFieldSpec]:
    property_unit = "mD" if unit_system == "FIELD" else "m²"
    uses_rock_parameter = grid_kinds == ("cartesian",)
    permeability_bindings: tuple[FlowFieldBinding, ...]
    porosity_bindings: tuple[FlowFieldBinding, ...]
    if uses_rock_parameter:
        permeability_bindings = (
            FlowFieldBinding("constructor", "rock", ("permeability_m2",)),
        )
        porosity_bindings = (FlowFieldBinding("constructor", "rock", ("porosity",)),)
    else:
        permeability_bindings = (FlowFieldBinding("constructor", "perm_tensor"),)
        porosity_bindings = (FlowFieldBinding("constructor", "porosity"),)

    is_mpfa_thermal = not uses_rock_parameter and "temperature" in primary_names
    if is_mpfa_thermal:
        permeability_bindings += (FlowFieldBinding("residual", "perm"),)
        porosity_bindings += (FlowFieldBinding("residual", "porosity"),)

    properties = [
        FlowFieldSpec(
            "permeability",
            property_unit,
            ("cell",),
            "property",
            bindings=permeability_bindings,
        ),
        FlowFieldSpec(
            "porosity",
            "1",
            ("cell",),
            "property",
            bindings=porosity_bindings,
        ),
    ]
    if uses_rock_parameter and "temperature" in primary_names:
        properties.extend(
            (
                FlowFieldSpec(
                    "rock_thermal_conductivity",
                    "W/(m·K)",
                    ("scalar",),
                    "property",
                    bindings=(
                        FlowFieldBinding(
                            "constructor",
                            "rock_thermal_conductivity_w_m_k",
                        ),
                    ),
                ),
                FlowFieldSpec(
                    "fluid_thermal_conductivity",
                    "W/(m·K)",
                    ("scalar",),
                    "property",
                    bindings=(
                        FlowFieldBinding(
                            "constructor",
                            "fluid_thermal_conductivity_w_m_k",
                        ),
                    ),
                ),
            )
        )
        return properties

    conductivity_parameters: tuple[tuple[str, str], ...] = ()
    if is_mpfa_thermal and phases == ("fluid",):
        conductivity_parameters = (("fluid_thermal_conductivity", "lam_fluid"),)
    elif is_mpfa_thermal and phases == ("water", "oil"):
        conductivity_parameters = (
            ("water_thermal_conductivity", "lam_w"),
            ("oil_thermal_conductivity", "lam_o"),
        )
    elif is_mpfa_thermal and phases == ("water", "oil", "gas"):
        conductivity_parameters = (
            ("water_thermal_conductivity", "lam_w"),
            ("oil_thermal_conductivity", "lam_o"),
            ("gas_thermal_conductivity", "lam_g"),
        )
    elif is_mpfa_thermal and phases == ("liquid", "vapor"):
        conductivity_parameters = (
            ("liquid_thermal_conductivity", "lam_l"),
            ("vapor_thermal_conductivity", "lam_v"),
        )
    if conductivity_parameters:
        properties.append(
            FlowFieldSpec(
                "rock_thermal_conductivity",
                "W/(m·K)",
                ("cell",),
                "property",
                bindings=(FlowFieldBinding("residual", "lam_rock"),),
            )
        )
        properties.extend(
            FlowFieldSpec(
                field_name,
                "W/(m·K)",
                ("cell",),
                "property",
                bindings=(FlowFieldBinding("residual", parameter),),
            )
            for field_name, parameter in conductivity_parameters
        )
    return properties


def _source_field(
    name: str,
    unit: str,
    axes: tuple[str, ...],
    components: tuple[str, ...] = (),
    *,
    parameter: str | None = None,
    attribute_path: tuple[str, ...] = (),
) -> FlowFieldSpec:
    resolved_parameter = name if parameter is None else parameter
    return FlowFieldSpec(
        name,
        unit,
        axes,
        "source",
        components,
        (FlowFieldBinding("residual", resolved_parameter, attribute_path),),
    )


def _flow_model_schema(
    *,
    model_name: str,
    primary_fields: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...],
    residual_blocks: tuple[
        tuple[
            str,
            Literal["kg/s", "mol/s", "W", "STB/day", "scf/day"],
            Literal["mass", "molar", "energy", "surface-volume"],
            str,
        ],
        ...,
    ],
    grid_kinds: tuple[str, ...],
    phases: tuple[str, ...] = (),
    components: tuple[str, ...] = (),
    unit_system: Literal["SI", "FIELD"] = "SI",
    structured_sources: bool = False,
) -> FlowModelSchema:
    """Build one production schema without probing or executing a model.

    This is deliberately Flow-private: concrete model classes declare their
    immutable schema at construction or class definition time, while the three
    public dataclasses remain the complete Agent-facing contract.
    """
    primary_names = frozenset(name for name, _, _, _ in primary_fields)
    properties = _property_fields(
        unit_system=unit_system,
        grid_kinds=grid_kinds,
        phases=phases,
        primary_names=primary_names,
    )

    residual_names = {name for name, _, _, _ in residual_blocks}
    conservations = {conservation for _, _, conservation, _ in residual_blocks}
    sources: list[FlowFieldSpec] = []
    if unit_system == "FIELD":
        if residual_names == {"fluid_surface_volume"}:
            sources.append(_source_field("source_rates", "STB/day", ("cell",)))
        else:
            for phase, unit in (("water", "STB/day"), ("oil", "STB/day"), ("gas", "scf/day")):
                if (
                    f"{phase}_surface_volume" in residual_names
                    or f"{phase}_standard_volume" in residual_names
                ):
                    sources.append(_source_field(f"source_{phase}_rates", unit, ("cell",)))
    elif conservations == {"surface-volume"}:
        # SI surface-volume kernels (the TPFA family):
        # sources are surface-volume rates in m³/s, same shape as FIELD.
        if residual_names == {"fluid_surface_volume"}:
            sources.append(_source_field("source_rates", "m³/s", ("cell",)))
        else:
            for phase in ("water", "oil", "gas"):
                if (
                    f"{phase}_surface_volume" in residual_names
                    or f"{phase}_standard_volume" in residual_names
                ):
                    sources.append(
                        _source_field(f"source_{phase}_rates", "m³/s", ("cell",))
                    )
    elif "component_molar" in residual_names:
        sources.append(
            _source_field(
                "source_component_molar",
                "mol/s",
                ("cell", "component"),
                components,
                parameter="sources",
            )
        )
    else:
        for block_name, _, conservation, _ in residual_blocks:
            if conservation != "mass":
                continue
            phase = block_name.removesuffix("_mass")
            source_name = "source_mass" if phase == "fluid" else f"source_{phase}"
            source_parameter = (
                "source"
                if residual_names == {"fluid_mass"} and grid_kinds != ("cartesian",)
                else source_name
            )
            sources.append(
                _source_field(
                    source_name,
                    "kg/s",
                    ("cell",),
                    parameter="sources" if structured_sources else source_parameter,
                    attribute_path=("phase_mass_kg_s", phase)
                    if structured_sources
                    else (),
                )
            )
    if "energy" in residual_names:
        sources.append(
            _source_field(
                "source_energy",
                "W",
                ("cell",),
                parameter="sources" if structured_sources else None,
                attribute_path=("energy_w",) if structured_sources else (),
            )
        )

    return FlowModelSchema(
        _MODEL_SCHEMA_VERSION,
        model_name,
        tuple(
            FlowFieldSpec(name, unit, axes, "primary", field_components)
            for name, unit, axes, field_components in primary_fields
        ),
        tuple(properties),
        tuple(sources),
        tuple(
            FlowResidualBlockSpec(name, unit, conservation, primary_field)
            for name, unit, conservation, primary_field in residual_blocks
        ),
        grid_kinds,
        phases,
        components,
        unit_system,
    )


__all__ = ["FlowFieldBinding", "FlowFieldSpec", "FlowModelSchema", "FlowResidualBlockSpec"]
