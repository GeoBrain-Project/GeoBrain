"""Pure-data Agent discovery records for GeoBrain Flow.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, NoReturn

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import (  # type: ignore[import-untyped]
    ValidationError as JsonSchemaValidationError,
)

from .errors import FlowContractError

FLOW_CAPABILITY_SCHEMA_VERSION = "geobrain.flow.capability/1.0"
FLOW_INPUT_SCHEMA_VERSION = "geobrain.flow.input/1.0"
FLOW_RUNTIME_CONSTRAINT_SCHEMA_VERSION = "geobrain.flow.runtime-constraints/1.0"


def _invalid(
    field: str,
    expected: object,
    actual: object,
    *,
    object_name: str,
) -> NoReturn:
    raise FlowContractError(
        "invalid Flow capability record",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _strings(
    value: object,
    *,
    field: str,
    object_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(field, "sequence of strings", value, object_name=object_name)
    result = tuple(value)
    if (not allow_empty and not result) or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        _invalid(field, "non-empty strings", value, object_name=object_name)
    if len(result) != len(set(result)):
        _invalid(field, "unique strings", value, object_name=object_name)
    return result


def _selection(value: object, *, object_name: str) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid("selection", "sequence of string pairs", value, object_name=object_name)
    result = tuple(value)
    pairs: list[tuple[str, str]] = []
    for item in result:
        if (
            isinstance(item, (str, bytes, bytearray))
            or not isinstance(item, Sequence)
            or len(item) != 2
            or any(not isinstance(part, str) or not part.strip() for part in item)
        ):
            _invalid("selection", "sequence of string pairs", value, object_name=object_name)
        pairs.append((item[0], item[1]))
    if len({key for key, _ in pairs}) != len(pairs):
        _invalid("selection", "pairs with unique keys", value, object_name=object_name)
    return tuple(pairs)


def _required_text(value: object, *, field: str, object_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(field, "non-empty string", value, object_name=object_name)
    return value


@dataclass(frozen=True, slots=True)
class FlowUnsupportedCombination:
    """One unsupported selection with a machine-readable recovery hint.

    Attributes:
        selection: the requested feature combination.
        reason: why it is refused.
        remediation: what to change to make it runnable.
    """

    selection: tuple[tuple[str, str], ...]
    reason: str
    remediation: str

    def __post_init__(self) -> None:
        name = type(self).__name__
        object.__setattr__(self, "selection", _selection(self.selection, object_name=name))
        object.__setattr__(
            self,
            "reason",
            _required_text(self.reason, field="reason", object_name=name),
        )
        object.__setattr__(
            self,
            "remediation",
            _required_text(self.remediation, field="remediation", object_name=name),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a detached strict-JSON representation."""
        return {
            "selection": {key: value for key, value in self.selection},
            "reason": self.reason,
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class FlowCapabilityReport:
    """Stable discovery record for one accepted Flow operator facade.

    Attributes:
        schema_version: report schema tag.
        name / maturity: family identity and maturity label.
        model_schemas: declared :class:`FlowModelSchema` ids.
        grid_kinds / dtypes / devices: supported execution axes.
        autograd_modes / history_modes / linear_solvers / well_controls /
            adapters: supported feature axes.
        resource_estimate_supported: whether preflight estimation works.
        unsupported: explicit :class:`FlowUnsupportedCombination` rows.
    """

    schema_version: str
    name: str
    maturity: Literal["production", "experimental"]
    model_schemas: tuple[str, ...]
    grid_kinds: tuple[str, ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    autograd_modes: tuple[str, ...]
    history_modes: tuple[str, ...]
    linear_solvers: tuple[str, ...]
    well_controls: tuple[str, ...]
    adapters: tuple[str, ...]
    resource_estimate_supported: bool
    unsupported: tuple[FlowUnsupportedCombination, ...]

    def __post_init__(self) -> None:
        name = type(self).__name__
        if self.schema_version != FLOW_CAPABILITY_SCHEMA_VERSION:
            _invalid(
                "schema_version",
                FLOW_CAPABILITY_SCHEMA_VERSION,
                self.schema_version,
                object_name=name,
            )
        object.__setattr__(
            self,
            "name",
            _required_text(self.name, field="name", object_name=name),
        )
        if self.maturity not in ("production", "experimental"):
            _invalid(
                "maturity",
                "'production' or 'experimental'",
                self.maturity,
                object_name=name,
            )
        for field in (
            "model_schemas",
            "grid_kinds",
            "dtypes",
            "devices",
            "autograd_modes",
            "history_modes",
            "linear_solvers",
            "well_controls",
            "adapters",
        ):
            object.__setattr__(
                self,
                field,
                _strings(
                    getattr(self, field),
                    field=field,
                    object_name=name,
                    allow_empty=field == "well_controls",
                ),
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
            _invalid(
                "unsupported",
                "sequence of FlowUnsupportedCombination",
                self.unsupported,
                object_name=name,
            )
        unsupported = tuple(self.unsupported)
        if any(not isinstance(item, FlowUnsupportedCombination) for item in unsupported):
            _invalid(
                "unsupported",
                "sequence of FlowUnsupportedCombination",
                self.unsupported,
                object_name=name,
            )
        object.__setattr__(self, "unsupported", unsupported)

    def to_dict(self) -> dict[str, object]:
        """Return report fields in stable order using only JSON primitives."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "maturity": self.maturity,
            "model_schemas": list(self.model_schemas),
            "grid_kinds": list(self.grid_kinds),
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "autograd_modes": list(self.autograd_modes),
            "history_modes": list(self.history_modes),
            "linear_solvers": list(self.linear_solvers),
            "well_controls": list(self.well_controls),
            "adapters": list(self.adapters),
            "resource_estimate_supported": self.resource_estimate_supported,
            "unsupported": [item.to_dict() for item in self.unsupported],
        }


_PRESSURE_ONLY_MODELS = ("MPFASinglePhaseModel",)
_PRESSURE_SATURATION_MODELS = (
    "MPFATwoPhaseModel",
    "MPFATwoPhaseModel3D",
)
_PRESSURE_TWO_SATURATION_MODELS = (
    "MPFAThreePhaseModel",
    "MPFAThreePhaseModel3D",
)
_THERMAL_MODELS = (
    "MPFAThermalSinglePhaseModel",
    "MPFAThermalSinglePhaseModel3D",
    "ThermalSinglePhaseModel",
)
_THERMAL_SATURATION_MODELS = (
    "MPFAThermalTwoPhaseModel",
    "MPFAThermalTwoPhaseModel3D",
)
_THERMAL_TWO_SATURATION_MODELS = (
    "MPFAThermalThreePhaseModel",
    "MPFAThermalThreePhaseModel3D",
)
_COMPOSITIONAL_MODELS = (
    "CompositionalModel",
    "MPFACompositionalModel",
    "MPFACompositionalModel3D",
)
_THERMAL_COMPOSITIONAL_MODELS = (
    "MPFAThermalCompositionalModel",
    "MPFAThermalCompositionalModel3D",
)
_CARTESIAN_PRODUCTION_MODELS = (
    "CompositionalModel",
    "ThermalSinglePhaseModel",
)
_MPFA_2D_PRODUCTION_MODELS = (
    "MPFACompositionalModel",
    "MPFASinglePhaseModel",
    "MPFAThermalCompositionalModel",
    "MPFAThermalSinglePhaseModel",
    "MPFAThermalThreePhaseModel",
    "MPFAThermalTwoPhaseModel",
    "MPFAThreePhaseModel",
    "MPFATwoPhaseModel",
)
_MPFA_3D_PRODUCTION_MODELS = (
    "MPFACompositionalModel3D",
    "MPFAThermalCompositionalModel3D",
    "MPFAThermalSinglePhaseModel3D",
    "MPFAThermalThreePhaseModel3D",
    "MPFAThermalTwoPhaseModel3D",
    "MPFAThreePhaseModel3D",
    "MPFATwoPhaseModel3D",
)
_PRODUCTION_MODEL_SCHEMAS = tuple(
    sorted(
        (*_CARTESIAN_PRODUCTION_MODELS, *_MPFA_2D_PRODUCTION_MODELS, *_MPFA_3D_PRODUCTION_MODELS)
    )
)


def _production_unsupported() -> tuple[FlowUnsupportedCombination, ...]:
    return (
        FlowUnsupportedCombination(
            selection=(("autograd_mode", "full"), ("linear_solver", "!=dense_direct")),
            reason="full differentiation requires a graph-preserving dense solve",
            remediation="select dense_direct or use implicit differentiation",
        ),
        FlowUnsupportedCombination(
            selection=(("wells", "enabled"),),
            reason="this cartesian profile has no typed WellGroup coupling",
            remediation="omit wells or select the production TPFA well profiles",
        ),
        FlowUnsupportedCombination(
            selection=(("device", "cuda|mps"),),
            reason="accelerator execution is not published without per-model acceptance evidence",
            remediation="select cpu; direct Python clients may use separately tested device paths",
        ),
    )


def _production_capability(
    *,
    name: str,
    model_schemas: tuple[str, ...],
    grid_kind: str,
    adapters: tuple[str, ...],
) -> FlowCapabilityReport:
    return FlowCapabilityReport(
        schema_version=FLOW_CAPABILITY_SCHEMA_VERSION,
        name=name,
        maturity="production",
        model_schemas=model_schemas,
        grid_kinds=(grid_kind,),
        dtypes=("float32", "float64"),
        devices=("cpu",),
        autograd_modes=("detached", "full", "implicit"),
        history_modes=("all", "checkpoint", "final", "recompute", "report"),
        linear_solvers=("bicgstab", "dense_direct", "gmres"),
        well_controls=(),
        adapters=adapters,
        resource_estimate_supported=True,
        unsupported=_production_unsupported(),
    )


def flow_evolution_capabilities() -> FlowCapabilityReport:
    """Return the default truthful Cartesian production profile."""
    return _production_capability(
        name="FlowEvolutionOperator.production.cartesian",
        model_schemas=_CARTESIAN_PRODUCTION_MODELS,
        grid_kind="cartesian",
        adapters=("field_units",),
    )


def _mpfa_2d_production_capabilities() -> FlowCapabilityReport:
    return _production_capability(
        name="FlowEvolutionOperator.production.mpfa_2d",
        model_schemas=_MPFA_2D_PRODUCTION_MODELS,
        grid_kind="mpfa-2d",
        adapters=("field_units",),
    )


def _mpfa_3d_production_capabilities() -> FlowCapabilityReport:
    return _production_capability(
        name="FlowEvolutionOperator.production.mpfa_3d",
        model_schemas=_MPFA_3D_PRODUCTION_MODELS,
        grid_kind="mpfa-3d",
        adapters=("field_units", "grdecl"),
    )


def _tpfa_dense_well_capabilities() -> FlowCapabilityReport:
    """Describe dense SI/TPFA execution with typed well controls.

    Production tier: the frame anchor matches the SI prediction and the
    slot map is literal (see ``adapters/UNIT_BOUNDARY.md``).
    """
    return FlowCapabilityReport(
        schema_version=FLOW_CAPABILITY_SCHEMA_VERSION,
        name="FlowEvolutionOperator.production.tpfa_dense_wells",
        maturity="production",
        model_schemas=("BlackOilModel", "OilWaterModel", "SinglePhaseModel"),
        grid_kinds=("cartesian",),
        dtypes=("float32", "float64"),
        devices=("cpu",),
        autograd_modes=("detached", "full", "implicit"),
        history_modes=("all", "checkpoint", "final", "recompute", "report"),
        linear_solvers=("bicgstab", "dense_direct", "gmres"),
        well_controls=("BHP", "GRAT", "LRAT", "ORAT", "RESV", "WRAT"),
        adapters=("field_units",),
        resource_estimate_supported=True,
        unsupported=(
            FlowUnsupportedCombination(
                selection=(("autograd_mode", "full"), ("linear_solver", "!=dense_direct")),
                reason="full differentiation requires the graph-preserving dense solve",
                remediation="select dense_direct or use implicit/detached differentiation",
            ),
            FlowUnsupportedCombination(
                selection=(("jacobian_layout", "dense"), ("linear_solver", "sparse_direct")),
                reason="sparse_direct requires an enabled sparse Jacobian",
                remediation="select dense_direct/gmres/bicgstab or enable sparse Jacobian first",
            ),
            FlowUnsupportedCombination(
                selection=(("device", "!=cpu"),),
                reason="the dense TPFA well path is accepted on CPU only",
                remediation="select cpu",
            ),
        ),
    )


def _tpfa_sparse_well_capabilities() -> FlowCapabilityReport:
    """Describe sparse SI/TPFA execution after explicit sparsity setup."""
    return FlowCapabilityReport(
        schema_version=FLOW_CAPABILITY_SCHEMA_VERSION,
        name="FlowEvolutionOperator.production.tpfa_sparse_wells",
        maturity="production",
        model_schemas=("BlackOilModel", "OilWaterModel", "SinglePhaseModel"),
        grid_kinds=("cartesian",),
        dtypes=("float32", "float64"),
        devices=("cpu",),
        autograd_modes=("detached", "implicit"),
        history_modes=("all", "checkpoint", "final", "recompute", "report"),
        linear_solvers=("bicgstab", "gmres", "sparse_direct"),
        well_controls=("BHP", "GRAT", "LRAT", "ORAT", "RESV", "WRAT"),
        adapters=("field_units",),
        resource_estimate_supported=True,
        unsupported=(
            FlowUnsupportedCombination(
                selection=(("jacobian_layout", "sparse"), ("linear_solver", "dense_direct")),
                reason="dense_direct is unavailable after sparse Jacobian setup",
                remediation="select sparse_direct, gmres, or bicgstab",
            ),
            FlowUnsupportedCombination(
                selection=(("autograd_mode", "full"), ("jacobian_layout", "sparse")),
                reason="the sparse Jacobian path does not preserve the full iteration graph",
                remediation="select implicit or detached differentiation",
            ),
            FlowUnsupportedCombination(
                selection=(
                    ("jacobian_layout", "sparse"),
                    ("model.sparse_jacobian", "disabled"),
                ),
                reason="the sparse profile requires an installed model sparsity pattern",
                remediation="call model.enable_sparse_jacobian() before operator construction",
            ),
            FlowUnsupportedCombination(
                selection=(("device", "!=cpu"),),
                reason="sparse_direct and the sparse TPFA profile are accepted on CPU only",
                remediation="select cpu",
            ),
        ),
    )


def _tpfa_varswitch_dense_capabilities() -> FlowCapabilityReport:
    """Describe dense SI variable switching separately from typed wells."""
    return FlowCapabilityReport(
        schema_version=FLOW_CAPABILITY_SCHEMA_VERSION,
        name="FlowEvolutionOperator.production.tpfa_varswitch_dense",
        maturity="production",
        model_schemas=("BlackOilVarSwitchModel",),
        grid_kinds=("cartesian",),
        dtypes=("float32", "float64"),
        devices=("cpu",),
        autograd_modes=("detached", "full", "implicit"),
        history_modes=("all", "checkpoint", "final", "recompute", "report"),
        linear_solvers=("bicgstab", "dense_direct", "gmres"),
        well_controls=(),
        adapters=("field_units",),
        resource_estimate_supported=True,
        unsupported=(
            FlowUnsupportedCombination(
                selection=(("wells", "enabled"),),
                reason="typed WellGroup coupling is unavailable for variable switching",
                remediation="omit wells or use an accepted fixed-variable TPFA model",
            ),
            FlowUnsupportedCombination(
                selection=(("autograd_mode", "full"), ("linear_solver", "!=dense_direct")),
                reason="full differentiation requires the graph-preserving dense solve",
                remediation="select dense_direct or use implicit/detached differentiation",
            ),
            FlowUnsupportedCombination(
                selection=(("jacobian_layout", "dense"), ("linear_solver", "sparse_direct")),
                reason="the variable-switch profile has no sparse Jacobian implementation",
                remediation="select dense_direct, gmres, or bicgstab",
            ),
        ),
    )


_CapabilityFactory = Callable[[], FlowCapabilityReport]
_PRODUCTION_CAPABILITY_FACTORIES: tuple[_CapabilityFactory, ...] = (
    flow_evolution_capabilities,
    _mpfa_2d_production_capabilities,
    _mpfa_3d_production_capabilities,
    # TPFA family: SI production tier.
    _tpfa_dense_well_capabilities,
    _tpfa_sparse_well_capabilities,
    _tpfa_varswitch_dense_capabilities,
)
_EXPERIMENTAL_CAPABILITY_FACTORIES: tuple[_CapabilityFactory, ...] = ()


def discover_flow_capabilities(
    include_experimental: bool = False,
) -> tuple[FlowCapabilityReport, ...]:
    """Return fresh reports in deterministic facade-name order."""
    if not isinstance(include_experimental, bool):
        raise FlowContractError(
            "include_experimental must be a boolean",
            object_name="discover_flow_capabilities",
            field="include_experimental",
            expected="boolean",
            actual=include_experimental,
        )
    factories = _PRODUCTION_CAPABILITY_FACTORIES
    if include_experimental:
        factories = (
            *_PRODUCTION_CAPABILITY_FACTORIES,
            *_EXPERIMENTAL_CAPABILITY_FACTORIES,
        )
    reports = tuple(factory() for factory in factories)
    production_count = len(_PRODUCTION_CAPABILITY_FACTORIES)
    if any(report.maturity != "production" for report in reports[:production_count]):
        raise RuntimeError("production Flow capability registry contains an unaccepted facade")
    if any(report.maturity != "experimental" for report in reports[production_count:]):
        raise RuntimeError("experimental Flow capability registry contains a production facade")
    return tuple(sorted(reports, key=lambda report: report.name))


def _vector_schema(
    *,
    unit: str,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {"type": "number"}
    if minimum is not None:
        item["minimum"] = minimum
    if exclusive_minimum is not None:
        item["exclusiveMinimum"] = exclusive_minimum
    if maximum is not None:
        item["maximum"] = maximum
    return {
        "type": "array",
        "items": item,
        "minItems": 1,
        "unit": unit,
        "axes": ["cell"],
    }


def _state_schema(
    required: tuple[str, ...],
) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required),
        "properties": {
            "pressure": _vector_schema(unit="Pa", exclusive_minimum=0.0),
            "sw": _vector_schema(unit="1", minimum=0.0, maximum=1.0),
            "sg": _vector_schema(unit="1", minimum=0.0, maximum=1.0),
            "temperature": _vector_schema(unit="K", exclusive_minimum=0.0),
            "composition": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "minItems": 2,
                },
                "minItems": 1,
                "unit": "1",
                "axes": ["cell", "component"],
            },
        },
    }


def _runtime_constraints() -> dict[str, object]:
    """Return detached machine rules that JSON Schema cannot express portably."""
    return {
        "schema_version": FLOW_RUNTIME_CONSTRAINT_SCHEMA_VERSION,
        "constraints": [
            {
                "id": "composition-simplex",
                "fields": ["state.composition"],
                "relation": "row_sum_equals",
                "target": 1.0,
                "absolute_tolerance": 1.0e-12,
            },
            {
                "id": "phase-saturation-simplex",
                "fields": ["state.sw", "state.sg"],
                "relation": "pairwise_sum_less_than_or_equal",
                "maximum": 1.0,
                "absolute_tolerance": 1.0e-12,
            },
            {
                "id": "report-times-strictly-increasing",
                "fields": ["execution.history.report_times_s"],
                "relation": "strictly_increasing",
                "minimum_inclusive": 0.0,
            },
            {
                "id": "state-axis-length-alignment",
                "fields": [
                    "state.pressure",
                    "state.sw",
                    "state.sg",
                    "state.temperature",
                    "state.composition",
                ],
                "axis": "cell",
                "relation": "equal_length_for_present_fields",
            },
        ],
    }


def _runtime_failure(constraint_id: str, expected: object, actual: object) -> NoReturn:
    raise FlowContractError(
        "Flow Agent input violates a runtime constraint",
        object_name="FlowEvolutionOperator.validate_input",
        field=constraint_id,
        expected=expected,
        actual=actual,
    )


def _sequence(value: object, *, constraint_id: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _runtime_failure(constraint_id, "JSON array", value)
    return value


def _finite_number(value: object, *, constraint_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        _runtime_failure(constraint_id, "finite JSON number", value)
    return float(value)


def _constraint_number(rule: Mapping[str, object], key: str) -> float:
    value = rule.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeError(f"invalid built-in Flow runtime constraint field {key!r}")
    return float(value)


def validate_flow_evolution_input(payload: Mapping[str, object]) -> None:
    """Validate the complete JSON contract, then enforce cross-array rules."""
    try:
        Draft202012Validator(flow_evolution_input_schema()).validate(payload)
    except JsonSchemaValidationError as exc:
        field = ".".join(str(part) for part in exc.absolute_path) or "$"
        raise FlowContractError(
            "Flow Agent input violates its Draft 2020-12 JSON Schema",
            object_name="FlowEvolutionOperator.validate_input",
            field=field,
            expected={str(exc.validator): exc.validator_value},
            actual=exc.instance,
        ) from exc
    if not isinstance(payload, Mapping):
        _runtime_failure("input-object", "JSON object", payload)
    state_value = payload.get("state")
    if not isinstance(state_value, Mapping):
        _runtime_failure("state-axis-length-alignment", "state object", state_value)
    state = state_value

    lengths: dict[str, int] = {}
    for field in ("pressure", "sw", "sg", "temperature", "composition"):
        if field in state:
            lengths[field] = len(
                _sequence(state[field], constraint_id="state-axis-length-alignment")
            )
    if not lengths or len(set(lengths.values())) != 1:
        _runtime_failure(
            "state-axis-length-alignment",
            "all present state fields share the cell-axis length",
            lengths,
        )

    constraint_rows = _runtime_constraints()["constraints"]
    if not isinstance(constraint_rows, list):
        raise RuntimeError("invalid built-in Flow runtime constraint registry")
    constraints: dict[str, Mapping[str, object]] = {}
    for row in constraint_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("invalid built-in Flow runtime constraint row")
        constraint_id = row.get("id")
        if not isinstance(constraint_id, str):
            raise RuntimeError("invalid built-in Flow runtime constraint row")
        constraints[constraint_id] = row
    composition = state.get("composition")
    if composition is not None:
        rule = constraints["composition-simplex"]
        tolerance = _constraint_number(rule, "absolute_tolerance")
        target = _constraint_number(rule, "target")
        for cell, row_value in enumerate(
            _sequence(composition, constraint_id="composition-simplex")
        ):
            row = _sequence(row_value, constraint_id="composition-simplex")
            values = tuple(
                _finite_number(value, constraint_id="composition-simplex") for value in row
            )
            if any(value < 0.0 or value > 1.0 for value in values) or not math.isclose(
                math.fsum(values),
                target,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                _runtime_failure(
                    "composition-simplex",
                    f"each composition row sums to {target} ± {tolerance}",
                    {"cell": cell, "values": values, "sum": math.fsum(values)},
                )

    if "sw" in state and "sg" in state:
        rule = constraints["phase-saturation-simplex"]
        tolerance = _constraint_number(rule, "absolute_tolerance")
        maximum = _constraint_number(rule, "maximum")
        sw = _sequence(state["sw"], constraint_id="phase-saturation-simplex")
        sg = _sequence(state["sg"], constraint_id="phase-saturation-simplex")
        for cell, (sw_value, sg_value) in enumerate(zip(sw, sg, strict=True)):
            total = _finite_number(
                sw_value, constraint_id="phase-saturation-simplex"
            ) + _finite_number(sg_value, constraint_id="phase-saturation-simplex")
            if total > maximum + tolerance:
                _runtime_failure(
                    "phase-saturation-simplex",
                    f"sw + sg <= {maximum} + {tolerance}",
                    {"cell": cell, "sum": total},
                )

    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        _runtime_failure("report-times-strictly-increasing", "execution object", execution)
    history = execution.get("history")
    if not isinstance(history, Mapping):
        _runtime_failure("report-times-strictly-increasing", "history object", history)
    report_times = history.get("report_times_s")
    if report_times is not None:
        times = tuple(
            _finite_number(value, constraint_id="report-times-strictly-increasing")
            for value in _sequence(
                report_times,
                constraint_id="report-times-strictly-increasing",
            )
        )
        if any(value < 0.0 for value in times) or any(
            later <= earlier for earlier, later in zip(times, times[1:], strict=False)
        ):
            _runtime_failure(
                "report-times-strictly-increasing",
                "finite seconds starting at or above zero in strictly increasing order",
                times,
            )


def _model_state_branch(
    models: tuple[str, ...],
    fields: tuple[str, ...],
) -> dict[str, object]:
    return {
        "properties": {
            "model_schema": {"enum": list(models)},
            "state": _state_schema(fields),
        },
        "required": ["model_schema", "state"],
    }


def flow_evolution_input_schema() -> Mapping[str, object]:
    """Return a fresh strict Draft 2020-12 schema for Agent/UI requests."""
    history = {
        "type": "object",
        "additionalProperties": False,
        "required": ["mode"],
        "properties": {
            "mode": {"enum": ["all", "checkpoint", "final", "recompute", "report"]},
            "report_times_s": {
                "type": "array",
                "items": {"type": "number", "minimum": 0.0},
                "uniqueItems": True,
            },
            "checkpoint_interval": {"type": "integer", "minimum": 1},
            "recompute_segments": {"type": "integer", "minimum": 1},
        },
        "allOf": [
            {
                "if": {
                    "required": ["mode"],
                    "properties": {"mode": {"const": "report"}},
                },
                "then": {
                    "required": ["report_times_s"],
                    "properties": {"report_times_s": {"minItems": 1}},
                },
                "else": {"properties": {"report_times_s": {"maxItems": 0}}},
            }
        ],
    }
    execution = {
        "type": "object",
        "additionalProperties": False,
        "required": ["autograd_mode", "linear_solver", "history"],
        "properties": {
            "autograd_mode": {"enum": ["detached", "full", "implicit"]},
            "linear_solver": {"enum": ["bicgstab", "dense_direct", "gmres"]},
            "history": history,
            "resource_budget_bytes": {"type": ["integer", "null"], "minimum": 1},
            "nonlinear": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_iterations": {"type": "integer", "minimum": 1},
                    "residual_tolerance": {"type": "number", "exclusiveMinimum": 0.0},
                    "update_tolerance": {"type": "number", "exclusiveMinimum": 0.0},
                    "line_search_max_iterations": {"type": "integer", "minimum": 1},
                },
            },
        },
        "allOf": [
            {
                "if": {
                    "required": ["autograd_mode"],
                    "properties": {"autograd_mode": {"const": "full"}},
                },
                "then": {"properties": {"linear_solver": {"const": "dense_direct"}}},
            }
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": FLOW_INPUT_SCHEMA_VERSION,
        "x-geobrain-runtime-constraints": _runtime_constraints(),
        "title": "GeoBrain FlowEvolutionOperator input",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "model_schema",
            "dtype",
            "device",
            "state",
            "time_step_s",
            "execution",
        ],
        "properties": {
            "model_schema": {"enum": list(_PRODUCTION_MODEL_SCHEMAS)},
            "dtype": {"enum": ["float32", "float64"]},
            "device": {"const": "cpu"},
            "state": {"type": "object"},
            "time_step_s": {"type": "number", "exclusiveMinimum": 0.0, "unit": "s"},
            "execution": execution,
        },
        "oneOf": [
            _model_state_branch(_PRESSURE_ONLY_MODELS, ("pressure",)),
            _model_state_branch(
                _PRESSURE_SATURATION_MODELS,
                ("pressure", "sw"),
            ),
            _model_state_branch(
                _PRESSURE_TWO_SATURATION_MODELS,
                ("pressure", "sw", "sg"),
            ),
            _model_state_branch(_THERMAL_MODELS, ("pressure", "temperature")),
            _model_state_branch(
                _THERMAL_SATURATION_MODELS,
                ("pressure", "sw", "temperature"),
            ),
            _model_state_branch(
                _THERMAL_TWO_SATURATION_MODELS,
                ("pressure", "sw", "sg", "temperature"),
            ),
            _model_state_branch(
                _COMPOSITIONAL_MODELS,
                ("pressure", "composition"),
            ),
            _model_state_branch(
                _THERMAL_COMPOSITIONAL_MODELS,
                ("pressure", "composition", "temperature"),
            ),
        ],
    }


__all__ = (
    "FlowCapabilityReport",
    "FlowUnsupportedCombination",
    "discover_flow_capabilities",
    "validate_flow_evolution_input",
)
