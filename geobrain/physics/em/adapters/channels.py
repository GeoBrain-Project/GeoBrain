"""Immutable family-local SI channel and source specifications for the EM family.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from geobrain.core.errors import ErrorCode

from ..errors import EMContractError


_QUANTITY_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "apparent_resistivity": "Ω·m",
        "chargeability": "1",
        "electric_field": "V/m",
        "impedance": "Ω",
        "loop_current": "A",
        "loop_radius": "m",
        "loop_turns": "1",
        "magnetic_dipole_moment": "A·m²",
        "magnetic_field_strength": "A/m",
        "magnetic_flux_density": "T",
        "magnetic_flux_density_time_derivative": "T/s",
        "phase": "rad",
        "voltage": "V",
    }
)


def _invalid_spec(
    field: str,
    expected: object,
    actual: object,
    *,
    code: ErrorCode = ErrorCode.CONFIG_INVALID,
) -> EMContractError:
    """Build a consistent structured channel-declaration failure."""
    return EMContractError(
        "invalid EM channel specification",
        object_name="EMChannelSpec",
        field=field,
        expected=expected,
        actual=actual,
        code=code,
        hint="use the canonical family-local SI channel table",
        details={
            "field": field,
            "received_type": type(actual).__qualname__,
            "remediation": "use the canonical family-local SI channel table",
        },
    )


@dataclass(frozen=True, slots=True)
class EMChannelSpec:
    """One immutable channel/source name, physical quantity, and SI unit."""

    name: str
    quantity: str
    unit: str
    complex_valued: bool

    def __post_init__(self) -> None:
        """Reject blank, type-tricked, unsupported, or unit-ambiguous declarations."""
        for field, value in (("name", self.name), ("quantity", self.quantity), ("unit", self.unit)):
            if type(value) is not str or not value or value.strip() != value:
                raise _invalid_spec(field, "non-empty canonical string", value)
        if type(self.complex_valued) is not bool:
            raise _invalid_spec("complex_valued", "Boolean", self.complex_valued)
        expected_unit = _QUANTITY_UNITS.get(self.quantity)
        if expected_unit is None:
            raise _invalid_spec(
                "quantity",
                tuple(sorted(_QUANTITY_UNITS)),
                self.quantity,
            )
        if self.unit != expected_unit:
            raise _invalid_spec(
                "unit",
                expected_unit,
                self.unit,
                code=ErrorCode.UNIT_MISMATCH,
            )

    def to_dict(self) -> dict[str, object]:
        """Return the exact JSON-safe scalar representation."""
        return {
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "complex_valued": self.complex_valued,
        }


def build_em_channel_table(
    specifications: Iterable[EMChannelSpec],
) -> tuple[EMChannelSpec, ...]:
    """Own an immutable table and reject wrong members or duplicate names."""
    table = tuple(specifications)
    invalid_types = tuple(
        sorted(
            {
                type(item).__qualname__
                for item in table
                if type(item) is not EMChannelSpec
            }
        )
    )
    if invalid_types:
        raise EMContractError(
            "EM channel table contains invalid members",
            object_name="build_em_channel_table",
            field="specifications",
            expected="EMChannelSpec entries",
            actual=invalid_types,
            details={
                "invalid_types": invalid_types,
                "remediation": "construct every entry as EMChannelSpec",
            },
        )
    counts: dict[str, int] = {}
    for item in table:
        counts[item.name] = counts.get(item.name, 0) + 1
    duplicates = tuple(sorted(name for name, count in counts.items() if count > 1))
    if duplicates:
        raise EMContractError(
            "EM channel names must be unique",
            object_name="build_em_channel_table",
            field="specifications.name",
            expected="unique names",
            actual=duplicates,
            details={
                "duplicate_names": duplicates,
                "remediation": "remove or rename duplicate channel specifications",
            },
        )
    return table


EM_CHANNEL_SPECS = build_em_channel_table(
    (
        EMChannelSpec("ex", "electric_field", "V/m", True),
        EMChannelSpec("ey", "electric_field", "V/m", True),
        EMChannelSpec("ez", "electric_field", "V/m", True),
        EMChannelSpec("hx", "magnetic_field_strength", "A/m", True),
        EMChannelSpec("hy", "magnetic_field_strength", "A/m", True),
        EMChannelSpec("hz", "magnetic_field_strength", "A/m", True),
        EMChannelSpec("bx", "magnetic_flux_density", "T", True),
        EMChannelSpec("by", "magnetic_flux_density", "T", True),
        EMChannelSpec("bz", "magnetic_flux_density", "T", True),
        EMChannelSpec(
            "dbdt_x",
            "magnetic_flux_density_time_derivative",
            "T/s",
            False,
        ),
        EMChannelSpec(
            "dbdt_y",
            "magnetic_flux_density_time_derivative",
            "T/s",
            False,
        ),
        EMChannelSpec(
            "dbdt_z",
            "magnetic_flux_density_time_derivative",
            "T/s",
            False,
        ),
        EMChannelSpec("voltage", "voltage", "V", False),
        EMChannelSpec("zxy", "impedance", "Ω", True),
        EMChannelSpec("zyx", "impedance", "Ω", True),
        EMChannelSpec(
            "apparent_resistivity",
            "apparent_resistivity",
            "Ω·m",
            False,
        ),
        EMChannelSpec("phase", "phase", "rad", False),
        EMChannelSpec("chargeability", "chargeability", "1", False),
    )
)

EM_SOURCE_SPECS = build_em_channel_table(
    (
        EMChannelSpec(
            "magnetic_moment_am2",
            "magnetic_dipole_moment",
            "A·m²",
            False,
        ),
        EMChannelSpec("current_a", "loop_current", "A", False),
        EMChannelSpec("radius_m", "loop_radius", "m", False),
        EMChannelSpec("turns", "loop_turns", "1", False),
    )
)


def get_em_channel_spec(name: str) -> EMChannelSpec:
    """Return one canonical channel specification or a structured error."""
    if type(name) is not str or not name or name.strip() != name:
        raise EMContractError(
            "EM channel name must be a non-empty canonical string",
            object_name="get_em_channel_spec",
            field="name",
            expected="canonical channel name",
            actual=name,
            details={
                "received_type": type(name).__qualname__,
                "remediation": "select a name from EM_CHANNEL_SPECS",
            },
        )
    for specification in EM_CHANNEL_SPECS:
        if specification.name == name:
            return specification
    raise EMContractError(
        "unknown EM channel",
        object_name="get_em_channel_spec",
        field="name",
        expected=tuple(item.name for item in EM_CHANNEL_SPECS),
        actual=name,
        details={
            "name": name,
            "remediation": "select a name from EM_CHANNEL_SPECS",
        },
    )


__all__ = [
    "EM_CHANNEL_SPECS",
    "EM_SOURCE_SPECS",
    "EMChannelSpec",
    "build_em_channel_table",
    "get_em_channel_spec",
]
