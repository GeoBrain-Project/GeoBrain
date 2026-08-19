"""Deterministic Draft 2020-12 schemas and validation for the EM family.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
from types import MappingProxyType
from typing import NoReturn, TypeAlias, cast

from .errors import EMContractError


JSONScalar: TypeAlias = str | int | float | bool | None
FrozenJSON: TypeAlias = JSONScalar | tuple["FrozenJSON", ...] | Mapping[str, "FrozenJSON"]

_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_SCHEMA_TYPES = frozenset({"array", "boolean", "integer", "number", "object", "string"})
_COMMON_DECLARATION_KEYWORDS = frozenset(
    {"description", "enum", "examples", "type", "x-geobrain-unit"}
)
_TYPE_DECLARATION_KEYWORDS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "array": frozenset(
            {
                "items",
                "maxItems",
                "minItems",
                "uniqueItems",
                "x-geobrain-strictly-increasing",
            }
        ),
        "boolean": frozenset(),
        "integer": frozenset({"exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum"}),
        "number": frozenset({"exclusiveMaximum", "exclusiveMinimum", "maximum", "minimum"}),
        "object": frozenset({"additionalProperties", "properties", "required"}),
        "string": frozenset(),
    }
)


def _invalid(
    message: str,
    *,
    object_name: str,
    field: str,
    expected: object,
    actual: object,
    remediation: str,
) -> NoReturn:
    """Raise one deterministic structured schema contract error."""
    actual_value = (
        actual
        if type(actual) in (str, int, float, bool) or actual is None
        else type(actual).__qualname__
    )
    raise EMContractError(
        message,
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual_value,
        hint=remediation,
        details={
            "field": field,
            "received_type": type(actual).__qualname__,
            "remediation": remediation,
        },
    )


def _freeze_json(value: object, active: set[int], path: str) -> FrozenJSON:
    """Deep-copy strict finite JSON into immutable, sorted containers."""
    if value is None or type(value) in (str, int, bool):
        return cast(JSONScalar, value)
    if type(value) is float:
        number = value
        if not math.isfinite(number):
            _invalid(
                "EM schema contains a non-finite number",
                object_name="EMSchemaField",
                field=path,
                expected="finite JSON number",
                actual=number,
                remediation="replace NaN or infinity with a finite schema bound",
            )
        return number
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            _invalid(
                "EM schema contains a cycle",
                object_name="EMSchemaField",
                field=path,
                expected="acyclic JSON mapping",
                actual=value,
                remediation="remove the cyclic schema reference",
            )
        if any(type(key) is not str for key in value):
            _invalid(
                "EM schema mapping keys must be strings",
                object_name="EMSchemaField",
                field=path,
                expected="string keys",
                actual=value,
                remediation="replace every schema key with a string",
            )
        active.add(identity)
        try:
            source = cast(Mapping[str, object], value)
            frozen = {
                key: _freeze_json(source[key], active, f"{path}.{key}") for key in sorted(source)
            }
        finally:
            active.remove(identity)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in active:
            _invalid(
                "EM schema contains a cycle",
                object_name="EMSchemaField",
                field=path,
                expected="acyclic JSON sequence",
                actual=value,
                remediation="remove the cyclic schema reference",
            )
        active.add(identity)
        try:
            return tuple(
                _freeze_json(item, active, f"{path}[{index}]") for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)
    _invalid(
        "EM schema contains a non-JSON value",
        object_name="EMSchemaField",
        field=path,
        expected="finite JSON scalar, mapping, or sequence",
        actual=value,
        remediation="replace the value with strict JSON data",
    )


def _thaw_json(value: FrozenJSON) -> object:
    """Return detached ordinary JSON containers at the public boundary."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _json_schema_equal(left: object, right: object) -> bool:
    """Compare JSON values using Draft 2020-12 equality semantics."""
    if left is None or right is None:
        return left is None and right is None
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return left == right
    if type(left) is str or type(right) is str:
        return type(left) is str and type(right) is str and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_mapping = cast(Mapping[object, object], left)
        right_mapping = cast(Mapping[object, object], right)
        if set(left_mapping) != set(right_mapping):
            return False
        return all(
            _json_schema_equal(left_mapping[key], right_mapping[key]) for key in left_mapping
        )
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes, bytearray))
        and isinstance(right, Sequence)
        and not isinstance(right, (str, bytes, bytearray))
    ):
        return len(left) == len(right) and all(
            _json_schema_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return False


def _validate_enum(declaration: Mapping[str, object], value: object, path: str) -> None:
    """Apply the common enum keyword to every supported JSON type."""
    enum = declaration.get("enum")
    if (
        isinstance(enum, Sequence)
        and not isinstance(enum, (str, bytes, bytearray))
        and not any(_json_schema_equal(value, candidate) for candidate in enum)
    ):
        _validation_error(path, list(enum), value, "select one declared enum value")


def _strings(value: object, *, field: str, object_name: str) -> tuple[str, ...]:
    """Own a unique sorted sequence of canonical identifiers."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(
            "invalid EM cross-field declaration",
            object_name=object_name,
            field=field,
            expected="sequence of canonical identifiers",
            actual=value,
            remediation="provide unique lowercase identifier names",
        )
    items = tuple(cast(Sequence[object], value))
    if any(type(item) is not str or _IDENTIFIER.fullmatch(item) is None for item in items):
        _invalid(
            "invalid EM cross-field declaration",
            object_name=object_name,
            field=field,
            expected="sequence of canonical identifiers",
            actual=value,
            remediation="provide unique lowercase identifier names",
        )
    result = cast(tuple[str, ...], items)
    if len(set(result)) != len(result):
        _invalid(
            "duplicate EM cross-field name",
            object_name=object_name,
            field=field,
            expected="unique identifiers",
            actual=result,
            remediation="remove duplicate field names",
        )
    return tuple(sorted(result))


def _pairs(
    value: object,
    *,
    field: str,
    object_name: str,
    json_values: bool,
) -> tuple[tuple[str, object], ...]:
    """Own unique-key pairs used by conditions and equality constraints."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(
            "invalid EM cross-field pairs",
            object_name=object_name,
            field=field,
            expected="sequence of pairs",
            actual=value,
            remediation="provide a sequence of unique field-name pairs",
        )
    pairs: list[tuple[str, object]] = []
    for index, pair in enumerate(cast(Sequence[object], value)):
        if (
            isinstance(pair, (str, bytes, bytearray))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
            or type(pair[0]) is not str
            or _IDENTIFIER.fullmatch(pair[0]) is None
        ):
            _invalid(
                "invalid EM cross-field pair",
                object_name=object_name,
                field=f"{field}[{index}]",
                expected="canonical field-name pair",
                actual=pair,
                remediation="provide a pair beginning with a lowercase field name",
            )
        left = pair[0]
        right = pair[1]
        if json_values:
            right = _freeze_json(right, set(), f"{field}[{index}][1]")
        elif type(right) is not str or _IDENTIFIER.fullmatch(right) is None:
            _invalid(
                "invalid EM equality field pair",
                object_name=object_name,
                field=f"{field}[{index}]",
                expected="two canonical field names",
                actual=pair,
                remediation="provide two lowercase field names",
            )
        pairs.append((left, right))
    if len({left for left, _ in pairs}) != len(pairs):
        _invalid(
            "duplicate EM cross-field pair key",
            object_name=object_name,
            field=field,
            expected="unique left-hand field names",
            actual=value,
            remediation="remove duplicate cross-field pair keys",
        )
    return tuple(sorted(pairs, key=lambda pair: pair[0]))


def _validate_declaration(declaration: Mapping[str, FrozenJSON], path: str) -> None:
    """Validate the supported deterministic JSON Schema declaration subset."""
    schema_type = declaration.get("type")
    if type(schema_type) is not str or schema_type not in _SCHEMA_TYPES:
        _invalid(
            "invalid EM schema type declaration",
            object_name="EMSchemaField",
            field=f"{path}.type",
            expected=tuple(sorted(_SCHEMA_TYPES)),
            actual=schema_type,
            remediation="declare one supported JSON Schema type",
        )
    allowed_keywords = _COMMON_DECLARATION_KEYWORDS | _TYPE_DECLARATION_KEYWORDS[schema_type]
    unsupported = tuple(sorted(set(declaration) - allowed_keywords))
    if unsupported:
        keyword = unsupported[0]
        _invalid(
            "unsupported EM schema declaration keyword",
            object_name="EMSchemaField",
            field=f"{path}.{keyword}",
            expected=tuple(sorted(allowed_keywords)),
            actual=keyword,
            remediation="remove keywords not implemented by EM runtime validation",
        )
    for keyword in ("description", "x-geobrain-unit"):
        if keyword in declaration:
            text = declaration[keyword]
            if type(text) is not str or not text.strip():
                _invalid(
                    "invalid EM schema annotation",
                    object_name="EMSchemaField",
                    field=f"{path}.{keyword}",
                    expected="non-empty string",
                    actual=text,
                    remediation="provide a stable non-empty schema annotation",
                )
    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in declaration and type(declaration[keyword]) not in (int, float):
            _invalid(
                "invalid EM numeric schema bound",
                object_name="EMSchemaField",
                field=f"{path}.{keyword}",
                expected="finite non-Boolean number",
                actual=declaration[keyword],
                remediation="provide a finite numeric bound",
            )
    if "enum" in declaration:
        enum = declaration["enum"]
        if not isinstance(enum, tuple) or not enum:
            _invalid(
                "invalid EM schema enum",
                object_name="EMSchemaField",
                field=f"{path}.enum",
                expected="non-empty array of unique JSON values",
                actual=enum,
                remediation="provide at least one unique enum value",
            )
        if any(
            _json_schema_equal(enum[index], enum[other_index])
            for index in range(len(enum))
            for other_index in range(index)
        ):
            _invalid(
                "duplicate EM schema enum value",
                object_name="EMSchemaField",
                field=f"{path}.enum",
                expected="unique JSON values",
                actual=enum,
                remediation="remove duplicate enum values",
            )
    if "examples" in declaration and not isinstance(declaration["examples"], tuple):
        _invalid(
            "invalid EM schema examples",
            object_name="EMSchemaField",
            field=f"{path}.examples",
            expected="JSON array",
            actual=declaration["examples"],
            remediation="provide examples as a JSON array",
        )
    if schema_type == "object":
        if (
            "additionalProperties" in declaration
            and declaration["additionalProperties"] is not False
        ):
            _invalid(
                "EM object declarations must be closed",
                object_name="EMSchemaField",
                field=f"{path}.additionalProperties",
                expected=False,
                actual=declaration["additionalProperties"],
                remediation="set additionalProperties to false or omit it",
            )
        properties = declaration.get("properties", MappingProxyType({}))
        if not isinstance(properties, Mapping):
            _invalid(
                "invalid EM object properties declaration",
                object_name="EMSchemaField",
                field=f"{path}.properties",
                expected="mapping",
                actual=properties,
                remediation="declare object properties as a JSON mapping",
            )
        for key, child in properties.items():
            if _IDENTIFIER.fullmatch(key) is None or not isinstance(child, Mapping):
                _invalid(
                    "invalid nested EM schema property",
                    object_name="EMSchemaField",
                    field=f"{path}.properties.{key}",
                    expected="canonical name and schema mapping",
                    actual=child,
                    remediation="declare each nested property with a lowercase name and schema",
                )
            _validate_declaration(child, f"{path}.properties.{key}")
        required = declaration.get("required", ())
        required_names = _strings(required, field=f"{path}.required", object_name="EMSchemaField")
        if any(name not in properties for name in required_names):
            _invalid(
                "EM schema requires an undeclared property",
                object_name="EMSchemaField",
                field=f"{path}.required",
                expected=tuple(sorted(properties)),
                actual=required_names,
                remediation="declare every required property",
            )
    if schema_type == "array":
        for keyword in ("minItems", "maxItems"):
            if keyword in declaration:
                count = declaration[keyword]
                if type(count) is not int or count < 0:
                    _invalid(
                        "invalid EM array size bound",
                        object_name="EMSchemaField",
                        field=f"{path}.{keyword}",
                        expected="non-negative integer",
                        actual=count,
                        remediation="provide a non-negative array size bound",
                    )
        for keyword in ("uniqueItems", "x-geobrain-strictly-increasing"):
            if keyword in declaration and type(declaration[keyword]) is not bool:
                _invalid(
                    "invalid EM array validation flag",
                    object_name="EMSchemaField",
                    field=f"{path}.{keyword}",
                    expected="bool",
                    actual=declaration[keyword],
                    remediation="set the array validation flag to true or false",
                )
        items = declaration.get("items")
        if not isinstance(items, Mapping):
            _invalid(
                "EM array schema requires one item schema",
                object_name="EMSchemaField",
                field=f"{path}.items",
                expected="schema mapping",
                actual=items,
                remediation="declare the array item type",
            )
        _validate_declaration(items, f"{path}.items")
        if declaration.get("x-geobrain-strictly-increasing") is True and items.get("type") not in (
            "integer",
            "number",
        ):
            _invalid(
                "strict ordering requires numeric EM array items",
                object_name="EMSchemaField",
                field=f"{path}.x-geobrain-strictly-increasing",
                expected="array items.type of integer or number",
                actual=items.get("type"),
                remediation="remove strict ordering or declare numeric array items",
            )


@dataclass(frozen=True, slots=True)
class EMSchemaField:
    """One deeply immutable top-level EM input-schema property."""

    name: str
    schema: Mapping[str, FrozenJSON]
    required: bool = False

    def __post_init__(self) -> None:
        """Own and validate one finite strict-JSON schema declaration."""
        if type(self.name) is not str or _IDENTIFIER.fullmatch(self.name) is None:
            _invalid(
                "invalid EM schema field name",
                object_name=type(self).__name__,
                field="name",
                expected="lowercase identifier",
                actual=self.name,
                remediation="use a lowercase canonical field name",
            )
        if type(self.required) is not bool:
            _invalid(
                "invalid EM schema required flag",
                object_name=type(self).__name__,
                field="required",
                expected="bool",
                actual=self.required,
                remediation="set required to true or false",
            )
        frozen = _freeze_json(self.schema, set(), f"properties.{self.name}")
        if not isinstance(frozen, Mapping):
            _invalid(
                "invalid EM schema field declaration",
                object_name=type(self).__name__,
                field="schema",
                expected="schema mapping",
                actual=self.schema,
                remediation="provide a JSON Schema mapping",
            )
        typed = frozen
        _validate_declaration(typed, f"properties.{self.name}")
        object.__setattr__(self, "schema", typed)


@dataclass(frozen=True, slots=True)
class EMCrossFieldRule:
    """A deterministic conditional requirement not fully expressible in JSON Schema."""

    rule_id: str
    when: tuple[tuple[str, object], ...]
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    equal_fields: tuple[tuple[str, str], ...] = ()
    reason: str = ""
    remediation: str = ""

    def __post_init__(self) -> None:
        """Deeply own the rule and reject ambiguous requirements."""
        name = type(self).__name__
        if type(self.rule_id) is not str or _IDENTIFIER.fullmatch(self.rule_id) is None:
            _invalid(
                "invalid EM cross-field rule identifier",
                object_name=name,
                field="rule_id",
                expected="lowercase identifier",
                actual=self.rule_id,
                remediation="use a stable lowercase rule identifier",
            )
        when = _pairs(self.when, field="when", object_name=name, json_values=True)
        if not when:
            _invalid(
                "EM cross-field rule requires a condition",
                object_name=name,
                field="when",
                expected="at least one condition pair",
                actual=self.when,
                remediation="declare the selection that activates this rule",
            )
        required = _strings(self.required, field="required", object_name=name)
        forbidden = _strings(self.forbidden, field="forbidden", object_name=name)
        equal_raw = _pairs(
            self.equal_fields, field="equal_fields", object_name=name, json_values=False
        )
        equal_fields = tuple((left, cast(str, right)) for left, right in equal_raw)
        if set(required) & set(forbidden):
            _invalid(
                "EM cross-field rule both requires and forbids a field",
                object_name=name,
                field="required",
                expected="disjoint required and forbidden names",
                actual=tuple(sorted(set(required) & set(forbidden))),
                remediation="remove the conflicting field from one side of the rule",
            )
        for field in ("reason", "remediation"):
            text = getattr(self, field)
            if type(text) is not str or not text.strip() or text.strip() != text:
                _invalid(
                    "EM cross-field rule requires actionable text",
                    object_name=name,
                    field=field,
                    expected="non-empty canonical string",
                    actual=text,
                    remediation="describe the reason and a concrete remediation",
                )
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "required", required)
        object.__setattr__(self, "forbidden", forbidden)
        object.__setattr__(self, "equal_fields", equal_fields)


def _closed_declaration(value: FrozenJSON) -> FrozenJSON:
    """Recursively add closed-object semantics and canonicalize schema arrays."""
    if isinstance(value, Mapping):
        mapped = {key: _closed_declaration(value[key]) for key in sorted(value)}
        if mapped.get("type") == "object":
            mapped["additionalProperties"] = False
        for key in ("enum", "required"):
            member = mapped.get(key)
            if isinstance(member, tuple):
                mapped[key] = tuple(
                    sorted(member, key=lambda item: json.dumps(_thaw_json(item), sort_keys=True))
                )
        return MappingProxyType({key: mapped[key] for key in sorted(mapped)})
    if isinstance(value, tuple):
        return tuple(_closed_declaration(item) for item in value)
    return value


def _rule_payload(rule: EMCrossFieldRule) -> dict[str, object]:
    """Return one detached extension record."""
    return {
        "rule_id": rule.rule_id,
        "when": {key: _thaw_json(cast(FrozenJSON, value)) for key, value in rule.when},
        "required": list(rule.required),
        "forbidden": list(rule.forbidden),
        "equal_fields": [[left, right] for left, right in rule.equal_fields],
        "reason": rule.reason,
        "remediation": rule.remediation,
    }


def _rule_schema(rule: EMCrossFieldRule) -> dict[str, object]:
    """Emit the Draft 2020-12 portion of one conditional rule."""
    then: dict[str, object] = {}
    if rule.required:
        then["required"] = list(rule.required)
    if rule.forbidden:
        then["not"] = {"anyOf": [{"required": [name]} for name in rule.forbidden]}
    return {
        "if": {
            "properties": {
                key: {"const": _thaw_json(cast(FrozenJSON, value))} for key, value in rule.when
            },
            "required": [key for key, _ in rule.when],
        },
        "then": then,
    }


def build_em_input_schema(
    operator: str,
    fields: tuple[EMSchemaField, ...],
    rules: tuple[EMCrossFieldRule, ...],
) -> dict[str, object]:
    """Build one canonical, recursively closed Draft 2020-12 schema."""
    if type(operator) is not str or _IDENTIFIER.fullmatch(operator) is None:
        _invalid(
            "invalid EM schema operator identifier",
            object_name="build_em_input_schema",
            field="operator",
            expected="lowercase identifier",
            actual=operator,
            remediation="use the canonical lowercase operator identifier",
        )
    if isinstance(fields, (str, bytes, bytearray)) or not isinstance(fields, Sequence):
        _invalid(
            "invalid EM schema field collection",
            object_name="build_em_input_schema",
            field="fields",
            expected="sequence of exact EMSchemaField records",
            actual=fields,
            remediation="provide immutable EMSchemaField declarations",
        )
    owned_fields = tuple(cast(Sequence[object], fields))
    if any(type(item) is not EMSchemaField for item in owned_fields):
        _invalid(
            "invalid EM schema field member",
            object_name="build_em_input_schema",
            field="fields",
            expected="exact EMSchemaField records",
            actual=fields,
            remediation="remove subclasses and non-field members",
        )
    typed_fields = cast(tuple[EMSchemaField, ...], owned_fields)
    field_names = tuple(item.name for item in typed_fields)
    if len(set(field_names)) != len(field_names):
        _invalid(
            "duplicate EM schema field name",
            object_name="build_em_input_schema",
            field="fields",
            expected="unique names",
            actual=field_names,
            remediation="remove duplicate field declarations",
        )
    if isinstance(rules, (str, bytes, bytearray)) or not isinstance(rules, Sequence):
        _invalid(
            "invalid EM schema rule collection",
            object_name="build_em_input_schema",
            field="rules",
            expected="sequence of exact EMCrossFieldRule records",
            actual=rules,
            remediation="provide immutable EMCrossFieldRule declarations",
        )
    owned_rules = tuple(cast(Sequence[object], rules))
    if any(type(item) is not EMCrossFieldRule for item in owned_rules):
        _invalid(
            "invalid EM schema rule member",
            object_name="build_em_input_schema",
            field="rules",
            expected="exact EMCrossFieldRule records",
            actual=rules,
            remediation="remove subclasses and non-rule members",
        )
    typed_rules = cast(tuple[EMCrossFieldRule, ...], owned_rules)
    rule_ids = tuple(item.rule_id for item in typed_rules)
    if len(set(rule_ids)) != len(rule_ids):
        _invalid(
            "duplicate EM schema rule identifier",
            object_name="build_em_input_schema",
            field="rules",
            expected="unique rule identifiers",
            actual=rule_ids,
            remediation="remove duplicate cross-field rules",
        )
    declared = set(field_names)
    for rule in typed_rules:
        referenced = {
            *(key for key, _ in rule.when),
            *rule.required,
            *rule.forbidden,
            *(left for left, _ in rule.equal_fields),
            *(right for _, right in rule.equal_fields),
        }
        missing = tuple(sorted(referenced - declared))
        if missing:
            _invalid(
                "EM cross-field rule references undeclared fields",
                object_name="build_em_input_schema",
                field="rules",
                expected=tuple(sorted(declared)),
                actual=missing,
                remediation="declare every field referenced by a cross-field rule",
            )
    sorted_fields = tuple(sorted(typed_fields, key=lambda item: item.name))
    sorted_rules = tuple(sorted(typed_rules, key=lambda item: item.rule_id))
    return {
        "$schema": _DRAFT_2020_12,
        "$id": f"https://geobrain.dev/schema/0.2.0/em/{operator}.schema.json",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            item.name: _thaw_json(_closed_declaration(cast(FrozenJSON, item.schema)))
            for item in sorted_fields
        },
        "required": sorted(item.name for item in sorted_fields if item.required),
        "allOf": [_rule_schema(rule) for rule in sorted_rules],
        "x-geobrain-cross-field-rules": [_rule_payload(rule) for rule in sorted_rules],
    }


def _validation_error(path: str, expected: object, actual: object, remediation: str) -> NoReturn:
    """Map every runtime validation failure to deterministic EM context."""
    _invalid(
        "invalid EM request",
        object_name="validate_em_request",
        field=path,
        expected=expected,
        actual=actual,
        remediation=remediation,
    )


def _numeric_bound(value: object) -> int | float | None:
    """Return one exact non-Boolean numeric bound when present."""
    if type(value) is int or type(value) is float:
        return value
    return None


def _validate_value(declaration: Mapping[str, object], value: object, path: str) -> None:
    """Validate one value against the emitted strict schema subset."""
    schema_type = declaration.get("type")
    if schema_type == "object":
        if not isinstance(value, Mapping):
            _validation_error(path, "object", value, "provide a JSON object")
        request = cast(Mapping[object, object], value)
        if any(type(key) is not str for key in request):
            _validation_error(path, "object with string keys", value, "use string property names")
        typed_request = cast(Mapping[str, object], request)
        properties = declaration.get("properties", {})
        if not isinstance(properties, Mapping):
            _validation_error(path, "valid object schema", declaration, "regenerate the EM schema")
        typed_properties = cast(Mapping[str, object], properties)
        required = declaration.get("required", [])
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes, bytearray)):
            _validation_error(
                path, "valid required declaration", declaration, "regenerate the EM schema"
            )
        for name in cast(Sequence[object], required):
            if type(name) is not str:
                _validation_error(
                    path, "valid required declaration", declaration, "regenerate the EM schema"
                )
            if name not in typed_request:
                _validation_error(f"{path}.{name}", "required property", None, f"provide {name}")
        if declaration.get("additionalProperties") is not False:
            _validation_error(path, "closed object schema", declaration, "regenerate the EM schema")
        for key in sorted(typed_request):
            if key not in typed_properties:
                _validation_error(f"{path}.{key}", "declared property", key, f"remove {key}")
        for key in sorted(typed_properties):
            child = typed_properties[key]
            if not isinstance(child, Mapping):
                _validation_error(path, "valid property schema", child, "regenerate the EM schema")
            if key in typed_request:
                _validate_value(
                    cast(Mapping[str, object], child), typed_request[key], f"{path}.{key}"
                )
        _validate_enum(declaration, value, path)
        return
    if schema_type == "array":
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            _validation_error(path, "array", value, "provide a JSON array")
        sequence = tuple(cast(Sequence[object], value))
        min_items = declaration.get("minItems")
        max_items = declaration.get("maxItems")
        if type(min_items) is int and len(sequence) < min_items:
            _validation_error(
                path, f"at least {min_items} items", len(sequence), "add required array items"
            )
        if type(max_items) is int and len(sequence) > max_items:
            _validation_error(
                path, f"at most {max_items} items", len(sequence), "remove excess array items"
            )
        items = declaration.get("items")
        if not isinstance(items, Mapping):
            _validation_error(path, "valid item schema", declaration, "regenerate the EM schema")
        for index, item in enumerate(sequence):
            _validate_value(cast(Mapping[str, object], items), item, f"{path}[{index}]")
        if declaration.get("uniqueItems") is True:
            for index, item in enumerate(sequence):
                if any(_json_schema_equal(item, previous) for previous in sequence[:index]):
                    _validation_error(path, "unique array items", value, "remove duplicate values")
        if declaration.get("x-geobrain-strictly-increasing") is True and any(
            cast(float, left) >= cast(float, right) for left, right in zip(sequence, sequence[1:])
        ):
            _validation_error(
                path, "strictly increasing values", value, "sort values and remove duplicates"
            )
        _validate_enum(declaration, value, path)
        return
    valid_type = (
        (schema_type == "string" and type(value) is str)
        or (schema_type == "boolean" and type(value) is bool)
        or (schema_type == "integer" and type(value) is int)
        or (schema_type == "number" and type(value) in (int, float))
    )
    if not valid_type:
        _validation_error(path, schema_type, value, f"provide a value of JSON type {schema_type}")
    if type(value) is float and not math.isfinite(value):
        _validation_error(
            path, "finite number", value, "replace NaN or infinity with a finite number"
        )
    _validate_enum(declaration, value, path)
    if type(value) in (int, float):
        numeric = cast(float, value)
        minimum = _numeric_bound(declaration.get("minimum"))
        maximum = _numeric_bound(declaration.get("maximum"))
        exclusive_minimum = _numeric_bound(declaration.get("exclusiveMinimum"))
        exclusive_maximum = _numeric_bound(declaration.get("exclusiveMaximum"))
        if minimum is not None and numeric < minimum:
            _validation_error(
                path,
                f"greater than or equal to {minimum}",
                value,
                "provide a value within the declared range",
            )
        if maximum is not None and numeric > maximum:
            _validation_error(
                path,
                f"less than or equal to {maximum}",
                value,
                "provide a value within the declared range",
            )
        if exclusive_minimum is not None and numeric <= exclusive_minimum:
            _validation_error(
                path,
                f"greater than {exclusive_minimum}",
                value,
                "provide a value within the declared range",
            )
        if exclusive_maximum is not None and numeric >= exclusive_maximum:
            _validation_error(
                path,
                f"less than {exclusive_maximum}",
                value,
                "provide a value within the declared range",
            )


def _schema_mapping(value: object, path: str) -> Mapping[str, object]:
    """Require a JSON-style mapping with string keys."""
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _validation_error(
            path, "schema mapping with string keys", value, "regenerate the EM schema"
        )
    return cast(Mapping[str, object], value)


def _validate_runtime_declarations(properties: object) -> Mapping[str, object]:
    """Validate detached property schemas before interpreting request values."""
    property_map = _schema_mapping(properties, "$schema.properties")
    for name in sorted(property_map):
        if _IDENTIFIER.fullmatch(name) is None:
            _validation_error(
                f"$schema.properties.{name}",
                "canonical property name",
                name,
                "regenerate the EM schema",
            )
        path = f"properties.{name}"
        try:
            frozen = _freeze_json(property_map[name], set(), path)
        except EMContractError as error:
            error_field = error.field if type(error.field) is str else path
            _validation_error(
                f"$schema.{error_field}",
                "supported EM schema declaration",
                property_map[name],
                "regenerate the EM schema with build_em_input_schema",
            )
        if not isinstance(frozen, Mapping):
            _validation_error(
                f"$schema.{path}",
                "schema mapping",
                property_map[name],
                "regenerate the EM schema",
            )
        try:
            _validate_declaration(frozen, path)
        except EMContractError as error:
            error_field = error.field if type(error.field) is str else path
            _validation_error(
                f"$schema.{error_field}",
                "supported EM schema declaration",
                property_map[name],
                "regenerate the EM schema with build_em_input_schema",
            )
    return property_map


def validate_em_request(schema: Mapping[str, object], request: Mapping[str, object]) -> None:
    """Validate an EM JSON request without mutation, solving, or allocation."""
    schema_map = _schema_mapping(schema, "$schema")
    if schema_map.get("$schema") != _DRAFT_2020_12:
        _validation_error(
            "$schema",
            _DRAFT_2020_12,
            schema_map.get("$schema"),
            "use the emitted Draft 2020-12 schema",
        )
    if schema_map.get("type") != "object" or schema_map.get("additionalProperties") is not False:
        _validation_error("$schema", "closed object schema", schema, "use build_em_input_schema")
    _validate_runtime_declarations(schema_map.get("properties"))
    _validate_value(schema_map, request, "$")
    rules = schema_map.get("x-geobrain-cross-field-rules", [])
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes, bytearray)):
        _validation_error(
            "$schema.x-geobrain-cross-field-rules", "array", rules, "use build_em_input_schema"
        )
    request_map = _schema_mapping(request, "$")
    for raw_rule in cast(Sequence[object], rules):
        rule = _schema_mapping(raw_rule, "$schema.x-geobrain-cross-field-rules[]")
        when = _schema_mapping(rule.get("when"), "$schema.x-geobrain-cross-field-rules[].when")
        if not all(
            key in request_map and _json_schema_equal(request_map[key], value)
            for key, value in when.items()
        ):
            continue
        remediation = rule.get("remediation")
        if type(remediation) is not str or not remediation:
            _validation_error(
                "$schema.x-geobrain-cross-field-rules[].remediation",
                "non-empty string",
                remediation,
                "use build_em_input_schema",
            )
        required = rule.get("required", [])
        forbidden = rule.get("forbidden", [])
        equal_fields = rule.get("equal_fields", [])
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes, bytearray)):
            _validation_error(
                "$schema.x-geobrain-cross-field-rules[].required",
                "array",
                required,
                "use build_em_input_schema",
            )
        if not isinstance(forbidden, Sequence) or isinstance(forbidden, (str, bytes, bytearray)):
            _validation_error(
                "$schema.x-geobrain-cross-field-rules[].forbidden",
                "array",
                forbidden,
                "use build_em_input_schema",
            )
        for name in cast(Sequence[object], required):
            if type(name) is not str:
                _validation_error(
                    "$schema.x-geobrain-cross-field-rules[].required",
                    "string array",
                    required,
                    "use build_em_input_schema",
                )
            if name not in request_map:
                _validation_error(f"$.{name}", "conditionally required property", None, remediation)
        for name in cast(Sequence[object], forbidden):
            if type(name) is not str:
                _validation_error(
                    "$schema.x-geobrain-cross-field-rules[].forbidden",
                    "string array",
                    forbidden,
                    "use build_em_input_schema",
                )
            if name in request_map:
                _validation_error(
                    f"$.{name}", "property absent for selected mode", request_map[name], remediation
                )
        if not isinstance(equal_fields, Sequence) or isinstance(
            equal_fields, (str, bytes, bytearray)
        ):
            _validation_error(
                "$schema.x-geobrain-cross-field-rules[].equal_fields",
                "pair array",
                equal_fields,
                "use build_em_input_schema",
            )
        for pair in cast(Sequence[object], equal_fields):
            if (
                isinstance(pair, (str, bytes, bytearray))
                or not isinstance(pair, Sequence)
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
            ):
                _validation_error(
                    "$schema.x-geobrain-cross-field-rules[].equal_fields",
                    "string pair array",
                    equal_fields,
                    "use build_em_input_schema",
                )
            left, right = pair[0], pair[1]
            if not _json_schema_equal(request_map.get(left), request_map.get(right)):
                _validation_error(
                    f"$.{right}", f"equal to $.{left}", request_map.get(right), remediation
                )


__all__ = [
    "EMCrossFieldRule",
    "EMSchemaField",
    "build_em_input_schema",
    "validate_em_request",
]
