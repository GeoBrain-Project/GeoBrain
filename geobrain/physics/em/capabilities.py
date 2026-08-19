"""Immutable allocation-free EM capability discovery records.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import ClassVar, Literal, NoReturn, cast

from geobrain.core import DifferentiabilityLevel

from .adapters.channels import EM_CHANNEL_SPECS, EMChannelSpec
from .errors import EMCapabilityError, EMContractError
from .resources import EMResourceEstimate, EMResourceRequest, estimate_em_resources
from .schemas import EMSchemaField, build_em_input_schema


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_DOMAINS = frozenset({"static", "frequency", "time", "magnetotelluric"})
_MATURITIES = frozenset({"preview", "production"})
_MESH_FORMULATIONS = frozenset(
    {
        "edge_fem",
        "layered_1d",
        "structured_fem",
        "structured_mimetic",
        "structured_yee",
        "triangular_fem",
    }
)
_DTYPES = frozenset({"complex128", "complex64", "float32", "float64"})
_DEVICES = frozenset({"cpu", "cuda", "mps"})
_SOLVERS = frozenset({"analytic", "dlf", "direct", "iterative", "pcg", "pcg_gpu", "sparse_lu"})
_BOUNDARIES = frozenset({"absorbing", "dirichlet", "natural", "neumann", "none", "pec"})
_RECEIVER_LAYOUTS = frozenset({"cartesian", "paired"})
_MODEL_FIELDS = frozenset(
    {
        "chargeability",
        "conductivity_s_per_m",
        "resistivity_ohm_m",
        "thickness_m",
    }
)
_SELECTION_VALUES: dict[str, frozenset[str]] = {
    "boundary": _BOUNDARIES,
    "device": _DEVICES,
    "dtype": _DTYPES,
    "mesh_formulation": _MESH_FORMULATIONS,
    "receiver_layout": _RECEIVER_LAYOUTS,
    "requires_gradient": frozenset({"false", "true"}),
    "solver": _SOLVERS,
}
_CANONICAL_CHANNELS = {item.name: item for item in EM_CHANNEL_SPECS}


def _invalid(
    field: str,
    expected: object,
    actual: object,
    *,
    object_name: str,
) -> NoReturn:
    """Raise one stable JSON-safe capability contract error."""
    actual_value = (
        actual if type(actual) in (str, int, bool) or actual is None else type(actual).__qualname__
    )
    raise EMContractError(
        "invalid EM capability declaration",
        object_name=object_name,
        field=field,
        expected=expected,
        actual=actual_value,
        hint="use canonical EM capability values",
        details={
            "field": field,
            "received_type": type(actual).__qualname__,
            "remediation": "use canonical EM capability values",
        },
    )


def _strings(
    value: object,
    *,
    field: str,
    object_name: str,
    allowed: frozenset[str] | None = None,
) -> tuple[str, ...]:
    """Own, validate, de-duplicate, and canonically sort string members."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(field, "sequence of canonical strings", value, object_name=object_name)
    source = tuple(cast(Sequence[object], value))
    if any(type(item) is not str or not item or item.strip() != item for item in source):
        _invalid(field, "sequence of canonical strings", value, object_name=object_name)
    result = cast(tuple[str, ...], source)
    if len(set(result)) != len(result):
        _invalid(field, "unique canonical strings", result, object_name=object_name)
    if any(_IDENTIFIER.fullmatch(item) is None for item in result):
        _invalid(field, "lowercase identifier strings", result, object_name=object_name)
    if allowed is not None and any(item not in allowed for item in result):
        _invalid(field, tuple(sorted(allowed)), result, object_name=object_name)
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class EMUnsupportedCombination:
    """One immutable unsupported selection with an actionable remedy."""

    selection: tuple[tuple[str, str], ...]
    reason: str
    remediation: str

    def __post_init__(self) -> None:
        """Deeply own and validate selection pairs."""
        name = type(self).__name__
        value = self.selection
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            _invalid("selection", "sequence of string pairs", value, object_name=name)
        pairs: list[tuple[str, str]] = []
        for pair in cast(Sequence[object], value):
            if (
                isinstance(pair, (str, bytes, bytearray))
                or not isinstance(pair, Sequence)
                or len(pair) != 2
                or type(pair[0]) is not str
                or type(pair[1]) is not str
            ):
                _invalid("selection", "sequence of string pairs", value, object_name=name)
            axis = pair[0]
            selected = pair[1]
            allowed = _SELECTION_VALUES.get(axis)
            if allowed is None or selected not in allowed:
                _invalid(
                    "selection",
                    {key: sorted(values) for key, values in sorted(_SELECTION_VALUES.items())},
                    (axis, selected),
                    object_name=name,
                )
            pairs.append((axis, selected))
        if not pairs or len({axis for axis, _ in pairs}) != len(pairs):
            _invalid("selection", "non-empty pairs with unique axes", value, object_name=name)
        object.__setattr__(self, "selection", tuple(sorted(pairs)))
        for field in ("reason", "remediation"):
            text = getattr(self, field)
            if type(text) is not str or not text.strip() or text.strip() != text:
                _invalid(field, "non-empty canonical string", text, object_name=name)

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe representation."""
        return {
            "selection": {key: value for key, value in self.selection},
            "reason": self.reason,
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class EMCapabilityReport:
    """Stable pure-data report of one EM operator's public support surface."""

    family: Literal["em"]
    operator: str
    domain: Literal["static", "frequency", "time", "magnetotelluric"]
    dimension: Literal[1, 2, 3]
    maturity: Literal["preview", "production"]
    model_fields: tuple[str, ...]
    output_channels: tuple[EMChannelSpec, ...]
    mesh_formulations: tuple[str, ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    solvers: tuple[str, ...]
    boundaries: tuple[str, ...]
    receiver_layouts: tuple[Literal["cartesian", "paired"], ...]
    differentiability: DifferentiabilityLevel
    resource_estimate: bool
    unsupported_combinations: tuple[EMUnsupportedCombination, ...]

    def __post_init__(self) -> None:
        """Own all sequences and reject overstated or malformed capabilities."""
        name = type(self).__name__
        if self.family != "em" or type(self.family) is not str:
            _invalid("family", "em", self.family, object_name=name)
        if type(self.operator) is not str or _IDENTIFIER.fullmatch(self.operator) is None:
            _invalid("operator", "lowercase identifier", self.operator, object_name=name)
        if type(self.domain) is not str or self.domain not in _DOMAINS:
            _invalid("domain", tuple(sorted(_DOMAINS)), self.domain, object_name=name)
        if type(self.dimension) is not int or self.dimension not in (1, 2, 3):
            _invalid("dimension", (1, 2, 3), self.dimension, object_name=name)
        if type(self.maturity) is not str or self.maturity not in _MATURITIES:
            _invalid("maturity", tuple(sorted(_MATURITIES)), self.maturity, object_name=name)

        model_fields = _strings(
            self.model_fields,
            field="model_fields",
            object_name=name,
            allowed=_MODEL_FIELDS,
        )
        if not model_fields:
            _invalid(
                "model_fields",
                "non-empty sequence of canonical identifiers",
                self.model_fields,
                object_name=name,
            )
        object.__setattr__(self, "model_fields", model_fields)
        for field, allowed in (
            ("mesh_formulations", _MESH_FORMULATIONS),
            ("dtypes", _DTYPES),
            ("devices", _DEVICES),
            ("solvers", _SOLVERS),
            ("boundaries", _BOUNDARIES),
            ("receiver_layouts", _RECEIVER_LAYOUTS),
        ):
            object.__setattr__(
                self,
                field,
                _strings(getattr(self, field), field=field, object_name=name, allowed=allowed),
            )

        channels_value = self.output_channels
        if isinstance(channels_value, (str, bytes, bytearray)) or not isinstance(
            channels_value, Sequence
        ):
            _invalid("output_channels", "EMChannelSpec sequence", channels_value, object_name=name)
        channels = tuple(cast(Sequence[object], channels_value))
        if any(
            type(item) is not EMChannelSpec
            or _CANONICAL_CHANNELS.get(cast(EMChannelSpec, item).name) != item
            for item in channels
        ):
            _invalid(
                "output_channels",
                "canonical EMChannelSpec entries",
                channels_value,
                object_name=name,
            )
        typed_channels = cast(tuple[EMChannelSpec, ...], channels)
        if len({item.name for item in typed_channels}) != len(typed_channels):
            _invalid("output_channels", "unique channel names", channels_value, object_name=name)
        object.__setattr__(
            self, "output_channels", tuple(sorted(typed_channels, key=lambda item: item.name))
        )

        if not isinstance(self.differentiability, DifferentiabilityLevel):
            _invalid(
                "differentiability",
                "DifferentiabilityLevel",
                self.differentiability,
                object_name=name,
            )
        if type(self.resource_estimate) is not bool:
            _invalid("resource_estimate", "bool", self.resource_estimate, object_name=name)

        unsupported_value = self.unsupported_combinations
        if isinstance(unsupported_value, (str, bytes, bytearray)) or not isinstance(
            unsupported_value, Sequence
        ):
            _invalid(
                "unsupported_combinations",
                "EMUnsupportedCombination sequence",
                unsupported_value,
                object_name=name,
            )
        unsupported = tuple(cast(Sequence[object], unsupported_value))
        if any(type(item) is not EMUnsupportedCombination for item in unsupported):
            _invalid(
                "unsupported_combinations",
                "exact EMUnsupportedCombination entries",
                unsupported_value,
                object_name=name,
            )
        typed_unsupported = cast(tuple[EMUnsupportedCombination, ...], unsupported)
        selections = tuple(item.selection for item in typed_unsupported)
        if len(set(selections)) != len(selections):
            _invalid(
                "unsupported_combinations",
                "unique normalized selections",
                unsupported_value,
                object_name=name,
            )
        object.__setattr__(
            self,
            "unsupported_combinations",
            tuple(sorted(typed_unsupported, key=lambda item: item.selection)),
        )
        if self.differentiability in (
            DifferentiabilityLevel.NON_DIFFERENTIABLE,
            DifferentiabilityLevel.FORWARD_ONLY,
        ) and not any(
            ("requires_gradient", "true") in item.selection for item in typed_unsupported
        ):
            _invalid(
                "unsupported_combinations",
                "an explicit requires_gradient=true exclusion",
                unsupported_value,
                object_name=name,
            )

    def to_dict(self) -> dict[str, object]:
        """Return report fields using only detached JSON primitives."""
        return {
            "family": self.family,
            "operator": self.operator,
            "domain": self.domain,
            "dimension": self.dimension,
            "maturity": self.maturity,
            "model_fields": list(self.model_fields),
            "output_channels": [item.to_dict() for item in self.output_channels],
            "mesh_formulations": list(self.mesh_formulations),
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "solvers": list(self.solvers),
            "boundaries": list(self.boundaries),
            "receiver_layouts": list(self.receiver_layouts),
            "differentiability": self.differentiability.value,
            "resource_estimate": self.resource_estimate,
            "unsupported_combinations": [item.to_dict() for item in self.unsupported_combinations],
        }


@dataclass(frozen=True, slots=True)
class _EMDiscoveryProfile:
    operator: str
    domain: Literal["static", "frequency", "time", "magnetotelluric"]
    dimension: Literal[1, 2, 3]
    model_fields: tuple[str, ...]
    output_channels: tuple[str, ...]
    mesh_formulations: tuple[str, ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    solvers: tuple[str, ...]
    boundaries: tuple[str, ...]
    receiver_layouts: tuple[Literal["cartesian", "paired"], ...]
    differentiability: DifferentiabilityLevel


_DISCOVERY_PROFILES: dict[str, _EMDiscoveryProfile] = {
    "CSEM1D": _EMDiscoveryProfile("csem1d", "frequency", 1, ("conductivity_s_per_m", "thickness_m"), ("bz",), ("layered_1d",), ("complex128",), ("cpu",), ("dlf",), ("none",), ("cartesian",), DifferentiabilityLevel.FULL_AUTOGRAD),
    "FDEM3D": _EMDiscoveryProfile("fdem3d", "frequency", 3, ("conductivity_s_per_m",), ("bx", "by", "bz", "ex", "ey", "ez"), ("edge_fem", "structured_yee"), ("complex128",), ("cpu",), ("pcg", "sparse_lu"), ("natural", "pec"), ("cartesian", "paired"), DifferentiabilityLevel.IMPLICIT_VJP),
    "FDEMCyl": _EMDiscoveryProfile("fdem_cyl", "frequency", 2, ("conductivity_s_per_m",), ("bz",), ("structured_fem",), ("complex128",), ("cpu",), ("sparse_lu",), ("natural",), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "HEM": _EMDiscoveryProfile("hem", "frequency", 3, ("conductivity_s_per_m",), ("bz",), ("edge_fem", "structured_yee"), ("complex128",), ("cpu",), ("pcg", "sparse_lu"), ("natural", "pec"), ("paired",), DifferentiabilityLevel.IMPLICIT_VJP),
    "MT1D": _EMDiscoveryProfile("mt1d", "magnetotelluric", 1, ("conductivity_s_per_m", "thickness_m"), ("apparent_resistivity", "phase", "zxy"), ("layered_1d",), ("complex128",), ("cpu",), ("analytic",), ("none",), ("cartesian",), DifferentiabilityLevel.FULL_AUTOGRAD),
    "MT2D": _EMDiscoveryProfile("mt2d", "magnetotelluric", 2, ("conductivity_s_per_m",), ("apparent_resistivity", "phase", "zxy", "zyx"), ("structured_mimetic", "triangular_fem"), ("complex128",), ("cpu",), ("sparse_lu",), ("dirichlet", "natural"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "MT3D": _EMDiscoveryProfile("mt3d", "magnetotelluric", 3, ("conductivity_s_per_m",), ("zxy", "zyx"), ("edge_fem", "structured_yee"), ("complex128",), ("cpu",), ("pcg", "sparse_lu"), ("natural", "pec"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "TEM1D": _EMDiscoveryProfile("tem1d", "time", 1, ("conductivity_s_per_m", "thickness_m"), ("dbdt_z",), ("layered_1d",), ("float64",), ("cpu",), ("dlf",), ("none",), ("paired",), DifferentiabilityLevel.FULL_AUTOGRAD),
    "WaveformTEM1D": _EMDiscoveryProfile("waveform_tem1d", "time", 1, ("conductivity_s_per_m", "thickness_m"), ("dbdt_z",), ("layered_1d",), ("float64",), ("cpu",), ("dlf",), ("none",), ("paired",), DifferentiabilityLevel.FULL_AUTOGRAD),
    "TEM3D": _EMDiscoveryProfile("tem3d", "time", 3, ("conductivity_s_per_m",), ("dbdt_x", "dbdt_y", "dbdt_z", "ex", "ey", "ez"), ("edge_fem", "structured_yee"), ("float64",), ("cpu",), ("pcg", "sparse_lu"), ("natural", "pec"), ("cartesian", "paired"), DifferentiabilityLevel.IMPLICIT_VJP),
    "VTEM": _EMDiscoveryProfile("vtem", "time", 3, ("conductivity_s_per_m",), ("dbdt_z",), ("edge_fem", "structured_yee"), ("float64",), ("cpu",), ("pcg", "sparse_lu"), ("natural", "pec"), ("paired",), DifferentiabilityLevel.IMPLICIT_VJP),
    "DC2D": _EMDiscoveryProfile("dc2d", "static", 2, ("conductivity_s_per_m",), ("voltage",), ("structured_fem",), ("float64",), ("cpu",), ("sparse_lu",), ("dirichlet", "neumann"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "DC25D": _EMDiscoveryProfile("dc25d", "static", 2, ("conductivity_s_per_m",), ("voltage",), ("structured_mimetic", "triangular_fem"), ("float64",), ("cpu",), ("sparse_lu",), ("dirichlet", "neumann"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "DC3D": _EMDiscoveryProfile("dc3d", "static", 3, ("conductivity_s_per_m",), ("voltage",), ("structured_mimetic", "triangular_fem"), ("float64",), ("cpu",), ("sparse_lu",), ("dirichlet", "neumann"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "IP2D": _EMDiscoveryProfile("ip2d", "static", 2, ("chargeability", "conductivity_s_per_m"), ("chargeability",), ("structured_fem",), ("float64",), ("cpu",), ("sparse_lu",), ("dirichlet", "neumann"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "IP3D": _EMDiscoveryProfile("ip3d", "static", 3, ("chargeability", "conductivity_s_per_m"), ("chargeability",), ("structured_mimetic", "triangular_fem"), ("float64",), ("cpu",), ("sparse_lu",), ("dirichlet", "neumann"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "SIP": _EMDiscoveryProfile("sip", "frequency", 3, ("chargeability", "conductivity_s_per_m"), ("chargeability",), ("structured_mimetic", "triangular_fem"), ("complex128",), ("cpu",), ("sparse_lu",), ("dirichlet", "neumann"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
    "SelfPotential2D": _EMDiscoveryProfile("self_potential_2d", "static", 2, ("conductivity_s_per_m",), ("voltage",), ("structured_fem",), ("float64",), ("cpu",), ("sparse_lu",), ("dirichlet", "neumann"), ("cartesian",), DifferentiabilityLevel.IMPLICIT_VJP),
}


def _schema_fields(profile: _EMDiscoveryProfile) -> tuple[EMSchemaField, ...]:
    """Build the common strict SI request surface without tensors or solvers."""
    return (
        EMSchemaField("operator", {"type": "string", "enum": [profile.operator]}, required=True),
        EMSchemaField("mesh_formulation", {"type": "string", "enum": list(profile.mesh_formulations)}, required=True),
        EMSchemaField("dtype", {"type": "string", "enum": list(profile.dtypes)}, required=True),
        EMSchemaField("device", {"type": "string", "enum": list(profile.devices)}, required=True),
        EMSchemaField("solver", {"type": "string", "enum": list(profile.solvers)}, required=True),
        EMSchemaField("boundary", {"type": "string", "enum": list(profile.boundaries)}, required=True),
        EMSchemaField("receiver_layout", {"type": "string", "enum": list(profile.receiver_layouts)}, required=True),
        EMSchemaField("recording", {"type": "string", "enum": ["checkpoint_recompute", "gate_states", "output_only"]}, required=True),
        EMSchemaField("requires_gradient", {"type": "boolean"}, required=True),
        EMSchemaField("model_fields", {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "enum": list(profile.model_fields)}}, required=True),
        EMSchemaField("channels", {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "enum": list(profile.output_channels)}}, required=True),
        EMSchemaField("frequencies_hz", {"type": "array", "x-geobrain-unit": "Hz", "minItems": 1, "uniqueItems": True, "x-geobrain-strictly-increasing": True, "items": {"type": "number", "exclusiveMinimum": 0.0}}),
        EMSchemaField("times_s", {"type": "array", "x-geobrain-unit": "s", "minItems": 1, "uniqueItems": True, "x-geobrain-strictly-increasing": True, "items": {"type": "number", "exclusiveMinimum": 0.0}}),
        EMSchemaField("n_cells", {"type": "integer", "minimum": 0}, required=True),
        EMSchemaField("n_edges", {"type": "integer", "minimum": 0}, required=True),
        EMSchemaField("n_faces", {"type": "integer", "minimum": 0}, required=True),
        EMSchemaField("n_sources", {"type": "integer", "minimum": 0}, required=True),
        EMSchemaField("n_receivers", {"type": "integer", "minimum": 0}, required=True),
        EMSchemaField("n_samples", {"type": "integer", "minimum": 0}, required=True),
        EMSchemaField("resource_budget_bytes", {"type": "integer", "minimum": 0}),
    )


def _capability_error(profile: _EMDiscoveryProfile, field: str, actual: str, supported: tuple[str, ...]) -> NoReturn:
    raise EMCapabilityError(
        "unsupported EM discovery selection",
        object_name=profile.operator,
        field=field,
        expected=supported,
        actual=actual,
        hint=f"select one of {', '.join(supported)}",
        details={
            "operator": profile.operator,
            "selection": {field: actual},
            "unsupported_axis": field,
            "supported_values": supported,
            "remediation": f"select one of {', '.join(supported)}",
        },
    )


class EMOperatorDiscovery:
    """Class-level discovery/resource mixin shared by every public EM operator."""

    _em_discovery_profiles: ClassVar[dict[str, _EMDiscoveryProfile]] = _DISCOVERY_PROFILES

    @classmethod
    def _em_discovery_profile(cls) -> _EMDiscoveryProfile:
        try:
            return cls._em_discovery_profiles[cls.__name__]
        except KeyError as error:
            raise EMContractError(
                "EM operator has no discovery profile",
                object_name=cls.__name__,
                field="operator",
                expected=tuple(sorted(cls._em_discovery_profiles)),
                actual=cls.__name__,
                hint="register the operator in the family-local discovery inventory",
                details={"operator": cls.__name__, "remediation": "add one exact discovery profile"},
            ) from error

    @classmethod
    def input_schema(cls) -> dict[str, object]:
        profile = cls._em_discovery_profile()
        return build_em_input_schema(profile.operator, _schema_fields(profile), ())

    @classmethod
    def capability_report(cls) -> EMCapabilityReport:
        profile = cls._em_discovery_profile()
        unsupported = (
            EMUnsupportedCombination(
                selection=(("device", "cuda"),),
                reason="this preview profile has no executed CUDA evidence",
                remediation="select cpu until the CUDA science and dtype rows pass",
            ),
            EMUnsupportedCombination(
                selection=(("device", "mps"),),
                reason="this preview profile has no executed MPS evidence",
                remediation="select cpu until the MPS science and dtype rows pass",
            ),
        )
        return EMCapabilityReport(
            family="em",
            operator=profile.operator,
            domain=profile.domain,
            dimension=profile.dimension,
            maturity="preview",
            model_fields=profile.model_fields,
            output_channels=tuple(_CANONICAL_CHANNELS[name] for name in profile.output_channels),
            mesh_formulations=profile.mesh_formulations,
            dtypes=profile.dtypes,
            devices=profile.devices,
            solvers=profile.solvers,
            boundaries=profile.boundaries,
            receiver_layouts=profile.receiver_layouts,
            differentiability=profile.differentiability,
            resource_estimate=True,
            unsupported_combinations=unsupported,
        )

    @classmethod
    def estimate_resources(cls, request: EMResourceRequest) -> EMResourceEstimate:
        if type(request) is not EMResourceRequest:
            raise EMContractError(
                "invalid operator resource request",
                object_name=cls.__name__,
                field="request",
                expected="exact EMResourceRequest",
                actual=type(request).__qualname__,
                hint="construct EMResourceRequest before preflight",
                details={"operator": cls.__name__, "remediation": "construct EMResourceRequest"},
            )
        profile = cls._em_discovery_profile()
        for field, supported in (
            ("mesh_formulation", profile.mesh_formulations),
            ("dtype", profile.dtypes),
            ("device", profile.devices),
            ("receiver_layout", profile.receiver_layouts),
        ):
            selected = cast(str, getattr(request, field))
            if selected not in supported:
                _capability_error(profile, field, selected, cast(tuple[str, ...], supported))
        return estimate_em_resources(request)


__all__ = ["EMCapabilityReport", "EMOperatorDiscovery", "EMUnsupportedCombination"]
