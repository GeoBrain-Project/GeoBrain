"""Strict JSON-safe capability records for Geomodel Agent discovery.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NoReturn

from .errors import GeomodelContractError


def _invalid(field: str, expected: object, actual: object, *, name: str) -> NoReturn:
    raise GeomodelContractError(
        "invalid Geomodel capability record",
        object_name=name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _strings(value: object, *, field: str, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid(field, "sequence of strings", value, name=name)
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        _invalid(field, "sequence of non-empty strings", value, name=name)
    return result


def _property_units(
    value: object,
    *,
    name: str,
) -> tuple[str | None, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        _invalid("property_units", "sequence of unit strings or None", value, name=name)
    result = tuple(value)
    if not result or not all(
        item is None or (isinstance(item, str) and bool(item.strip())) for item in result
    ):
        _invalid(
            "property_units",
            "non-empty sequence of non-empty unit strings or None",
            value,
            name=name,
        )
    return result


@dataclass(frozen=True, slots=True)
class GeomodelUnsupportedCombination:
    """One unsupported capability selection and deterministic remediation.

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
        if isinstance(self.selection, (str, bytes, bytearray)) or not isinstance(
            self.selection, Sequence
        ):
            _invalid(
                "selection",
                "sequence of non-empty string pairs",
                self.selection,
                name=name,
            )
        pairs = tuple(self.selection)
        if any(
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes, bytearray))
            or len(pair) != 2
            or not all(isinstance(item, str) and item for item in pair)
            for pair in pairs
        ):
            _invalid("selection", "sequence of non-empty string pairs", self.selection, name=name)
        normalized = tuple((pair[0], pair[1]) for pair in pairs)
        if len({key for key, _ in normalized}) != len(normalized):
            _invalid("selection", "unique selection keys", self.selection, name=name)
        object.__setattr__(self, "selection", normalized)
        if not isinstance(self.reason, str) or not self.reason:
            _invalid("reason", "non-empty string", self.reason, name=name)
        if not isinstance(self.remediation, str) or not self.remediation:
            _invalid("remediation", "non-empty string", self.remediation, name=name)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native unsupported-combination record."""
        return {
            "selection": {key: value for key, value in self.selection},
            "reason": self.reason,
            "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class GeomodelCapabilityReport:
    """Stable pure-data description of one Geomodel public capability.

    Attributes:
        name / domain / maturity / algorithm: algorithm identity.
        dimensions: supported spatial dimensions.
        property_kinds / property_units / coordinate_unit: data contract.
        dtypes / devices / backends: execution axes.
        conditioning: supported conditioning modes.
        deterministic_scope: reproducibility declaration.
        differentiability: gradient support declaration.
        optional_dependencies: extra packages the algorithm needs.
        checkpoint_required: whether a model checkpoint must be supplied.
    """

    name: str
    domain: Literal["geomodel"]
    maturity: Literal["production", "experimental"]
    algorithm: str
    dimensions: tuple[int, ...]
    property_kinds: tuple[str, ...]
    property_units: tuple[str | None, ...]
    coordinate_unit: Literal["m"]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    backends: tuple[str, ...]
    conditioning: tuple[str, ...]
    deterministic_scope: str
    differentiability: str
    optional_dependencies: tuple[str, ...]
    checkpoint_required: bool
    resource_estimate_supported: bool
    unsupported: tuple[GeomodelUnsupportedCombination, ...]

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        for field_name in ("name", "algorithm", "deterministic_scope", "differentiability"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                _invalid(field_name, "non-empty string", value, name=object_name)
        if self.domain != "geomodel":
            _invalid("domain", "geomodel", self.domain, name=object_name)
        if self.maturity not in ("production", "experimental"):
            _invalid(
                "maturity",
                "'production' or 'experimental'",
                self.maturity,
                name=object_name,
            )
        if isinstance(self.dimensions, (str, bytes, bytearray)) or not isinstance(
            self.dimensions, Sequence
        ):
            _invalid(
                "dimensions",
                "unique non-empty sequence containing 2 and/or 3",
                self.dimensions,
                name=object_name,
            )
        dimensions = tuple(self.dimensions)
        if (
            not dimensions
            or any(type(item) is not int or item not in (2, 3) for item in dimensions)
            or len(set(dimensions)) != len(dimensions)
        ):
            _invalid(
                "dimensions",
                "unique non-empty sequence containing 2 and/or 3",
                dimensions,
                name=object_name,
            )
        object.__setattr__(self, "dimensions", dimensions)
        if self.coordinate_unit != "m":
            _invalid("coordinate_unit", "m", self.coordinate_unit, name=object_name)
        for field_name in (
            "property_kinds",
            "dtypes",
            "devices",
            "backends",
            "conditioning",
            "optional_dependencies",
        ):
            object.__setattr__(
                self,
                field_name,
                _strings(getattr(self, field_name), field=field_name, name=object_name),
            )
        if any(
            kind not in ("continuous", "categorical", "probability") for kind in self.property_kinds
        ):
            _invalid(
                "property_kinds",
                "continuous/categorical/probability values",
                self.property_kinds,
                name=object_name,
            )
        property_units = _property_units(self.property_units, name=object_name)
        if len(property_units) != len(self.property_kinds):
            _invalid(
                "property_units",
                f"{len(self.property_kinds)} units paired with property_kinds",
                property_units,
                name=object_name,
            )
        expected_units = {
            "continuous": "explicit non-empty unit string ('1' if dimensionless)",
            "categorical": "None",
            "probability": "'1'",
        }
        for index, (kind, unit) in enumerate(zip(self.property_kinds, property_units)):
            valid = (
                (kind == "continuous" and isinstance(unit, str) and bool(unit.strip()))
                or (kind == "categorical" and unit is None)
                or (kind == "probability" and unit == "1")
            )
            if not valid:
                _invalid(
                    "property_units",
                    expected_units[kind],
                    {"index": index, "kind": kind, "unit": unit},
                    name=object_name,
                )
        object.__setattr__(self, "property_units", property_units)
        for field_name in ("checkpoint_required", "resource_estimate_supported"):
            if not isinstance(getattr(self, field_name), bool):
                _invalid(field_name, "bool", getattr(self, field_name), name=object_name)
        if isinstance(self.unsupported, (str, bytes, bytearray)) or not isinstance(
            self.unsupported, Sequence
        ):
            _invalid(
                "unsupported",
                "sequence of GeomodelUnsupportedCombination",
                self.unsupported,
                name=object_name,
            )
        unsupported = tuple(self.unsupported)
        if not all(isinstance(item, GeomodelUnsupportedCombination) for item in unsupported):
            _invalid(
                "unsupported",
                "sequence of GeomodelUnsupportedCombination",
                self.unsupported,
                name=object_name,
            )
        object.__setattr__(self, "unsupported", unsupported)

    def to_dict(self) -> dict[str, object]:
        """Return a closed strict JSON-native capability record."""
        return {
            "name": self.name,
            "domain": self.domain,
            "maturity": self.maturity,
            "algorithm": self.algorithm,
            "dimensions": list(self.dimensions),
            "property_kinds": list(self.property_kinds),
            "property_units": list(self.property_units),
            "coordinate_unit": self.coordinate_unit,
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "backends": list(self.backends),
            "conditioning": list(self.conditioning),
            "deterministic_scope": self.deterministic_scope,
            "differentiability": self.differentiability,
            "optional_dependencies": list(self.optional_dependencies),
            "checkpoint_required": self.checkpoint_required,
            "resource_estimate_supported": self.resource_estimate_supported,
            "unsupported": [item.to_dict() for item in self.unsupported],
        }


def discover_geomodel_capabilities(
    include_experimental: bool = False,
) -> tuple[GeomodelCapabilityReport, ...]:
    """Return a stable name-ordered, production-only-by-default catalogue."""
    if not isinstance(include_experimental, bool):
        _invalid(
            "include_experimental",
            "bool",
            include_experimental,
            name="discover_geomodel_capabilities",
        )
    rows = (
        GeomodelCapabilityReport(
            "classical_simulation", "geomodel", "production",
            "indexed sequential, dense-factor, spectral, and training-image simulation",
            (2, 3),
            ("continuous", "categorical"),
            ("1", None),
            "m", ("float64",), ("cpu",),
            ("indexed", "exhaustive", "dense", "fft", "training_image"),
            ("hard", "soft"),
            "fixed seed, backend, worker count, and declared policies",
            "not differentiable",
            (), False, True, (),
        ),
        GeomodelCapabilityReport(
            "generative_diffusion", "geomodel", "experimental",
            "optional-provider latent diffusion",
            (3,), ("categorical",), (None,), "m",
            ("float32", "float64"), ("cpu", "cuda", "mps"), ("torch",),
            ("optional well constraints",),
            "fixed checkpoint, model card, seed, dtype, and device",
            "decoder only; label generation is non-differentiable",
            ("LDMC",), True, True, (),
        ),
        GeomodelCapabilityReport(
            "implicit_model", "geomodel", "production",
            "universal cokriging implicit geology",
            (2, 3), ("categorical",), (None,), "m",
            ("float32", "float64"), ("cpu", "cuda", "mps"), ("torch",),
            ("surface points", "orientations", "faults"),
            "fixed tensors, dtype, and device",
            "scalar fields and soft blocks; hard block is non-differentiable",
            (), False, True, (),
        ),
        GeomodelCapabilityReport(
            "kriging", "geomodel", "production",
            "simple, ordinary, universal, indicator, block, and collocated cokriging",
            (2, 3),
            ("continuous", "categorical", "probability"),
            ("1", None, "1"),
            "m", ("float64",), ("cpu",), ("indexed", "exhaustive"),
            ("hard", "soft"),
            "deterministic for fixed data and declared numerical policy",
            "not differentiable",
            (), False, True, (),
        ),
        GeomodelCapabilityReport(
            "neural_gan_vae", "geomodel", "experimental",
            "checkpoint-backed GAN and variational autoencoder",
            (3,), ("categorical",), (None,), "m",
            ("float32", "float64"), ("cpu", "cuda", "mps"), ("torch",),
            ("model dependent",),
            "fixed verified checkpoint, seed, dtype, and device",
            "soft decoder only; label generation is non-differentiable",
            ("model checkpoint",), True, True, (),
        ),
    )
    selected = rows if include_experimental else tuple(
        report for report in rows if report.maturity == "production"
    )
    return tuple(sorted(selected, key=lambda report: report.name))


__all__ = [
    "GeomodelCapabilityReport",
    "GeomodelUnsupportedCombination",
    "discover_geomodel_capabilities",
]
