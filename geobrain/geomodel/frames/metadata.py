"""Immutable property metadata for continuous, categorical, and probability data.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NoReturn

import numpy as np

from ..errors import GeomodelContractError

PropertyKind = Literal["continuous", "categorical", "probability"]


def _invalid(field: str, expected: object, actual: object, *, name: str) -> NoReturn:
    raise GeomodelContractError(
        "invalid Geomodel property metadata",
        object_name=name,
        field=field,
        expected=expected,
        actual=actual,
    )


@dataclass(frozen=True, slots=True)
class Category:
    """One exact integer categorical code and its human-readable label.

    Attributes:
        code: integer category code.
        label: human-readable category name.
    """

    code: int
    label: str

    def __post_init__(self) -> None:
        name = type(self).__name__
        if isinstance(self.code, bool) or not isinstance(self.code, int):
            _invalid("code", "integer (not bool)", self.code, name=name)
        if not isinstance(self.label, str) or not self.label.strip():
            _invalid("label", "non-empty string", self.label, name=name)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native category record."""
        return {"code": self.code, "label": self.label}


@dataclass(frozen=True, slots=True)
class PropertyMetadata:
    """Scientific identity, unit, and optional vocabulary of one property.

    Attributes:
        name: data-column / property name.
        kind: ``'continuous'`` or ``'categorical'``.
        unit: unit string.
        categories: category table for categorical properties.
    """

    name: str
    kind: PropertyKind
    unit: str | None
    categories: tuple[Category, ...] = ()

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        if not isinstance(self.name, str) or not self.name.strip():
            _invalid("name", "non-empty string", self.name, name=object_name)
        if self.kind not in ("continuous", "categorical", "probability"):
            _invalid(
                "kind",
                "'continuous', 'categorical', or 'probability'",
                self.kind,
                name=object_name,
            )
        try:
            categories = tuple(self.categories)
        except TypeError as exc:
            raise GeomodelContractError(
                "invalid Geomodel property metadata",
                object_name=object_name,
                field="categories",
                expected="sequence of Category records",
                actual=self.categories,
            ) from exc
        if not all(isinstance(category, Category) for category in categories):
            _invalid(
                "categories",
                "sequence of Category records",
                self.categories,
                name=object_name,
            )
        object.__setattr__(self, "categories", categories)

        if self.kind == "continuous":
            if not isinstance(self.unit, str) or not self.unit.strip():
                _invalid(
                    "unit",
                    "explicit non-empty unit string ('1' if dimensionless)",
                    self.unit,
                    name=object_name,
                )
            if categories:
                _invalid(
                    "categories", "empty for continuous property", categories, name=object_name
                )
        elif self.kind == "probability":
            if self.unit != "1":
                _invalid("unit", "'1' for probability property", self.unit, name=object_name)
            if categories:
                _invalid(
                    "categories", "empty for probability property", categories, name=object_name
                )
        else:
            if self.unit is not None:
                _invalid("unit", "None for categorical property", self.unit, name=object_name)
            if not categories:
                _invalid(
                    "categories",
                    "non-empty categorical vocabulary",
                    categories,
                    name=object_name,
                )
            codes = tuple(category.code for category in categories)
            labels = tuple(category.label for category in categories)
            if len(set(codes)) != len(codes):
                _invalid("categories", "unique category codes", codes, name=object_name)
            try:
                encoded_codes = tuple(np.float64(code) for code in codes)
            except (OverflowError, ValueError) as exc:
                raise GeomodelContractError(
                    "categorical codes must be exactly representable in float64 storage",
                    object_name=object_name,
                    field="categories",
                    expected="integer codes with exact finite float64 representations",
                    actual=codes,
                ) from exc
            if any(
                not np.isfinite(encoded) or int(encoded) != code
                for code, encoded in zip(codes, encoded_codes)
            ):
                _invalid(
                    "categories",
                    "integer codes with exact finite float64 representations",
                    codes,
                    name=object_name,
                )
            if len(set(labels)) != len(labels):
                _invalid("categories", "unique category labels", labels, name=object_name)

    @property
    def category_codes(self) -> tuple[int, ...]:
        """Return the exact categorical code vocabulary in declared order."""
        return tuple(category.code for category in self.categories)

    def validate_values(self, values: np.ndarray, *, object_name: str) -> None:
        """Validate one float64 property array against this metadata."""
        if not np.isfinite(values).all():
            raise GeomodelContractError(
                "property values must be finite",
                object_name=object_name,
                field=self.name,
                expected="finite values",
                actual="contains NaN or infinity",
            )
        if self.kind == "categorical":
            integral = np.equal(values, np.floor(values))
            vocabulary = np.asarray(self.category_codes, dtype=np.float64)
            known = np.isin(values, vocabulary)
            if not np.all(integral & known):
                bad = float(values[np.flatnonzero(~(integral & known))[0]])
                raise GeomodelContractError(
                    "categorical property contains a value outside its vocabulary",
                    object_name=object_name,
                    field=self.name,
                    expected={"vocabulary": list(self.category_codes)},
                    actual=bad,
                )
        elif self.kind == "probability" and np.any((values < 0.0) | (values > 1.0)):
            bad = float(values[np.flatnonzero((values < 0.0) | (values > 1.0))[0]])
            raise GeomodelContractError(
                "probability property values must lie in [0, 1]",
                object_name=object_name,
                field=self.name,
                expected="[0, 1]",
                actual=bad,
            )

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native property record."""
        return {
            "name": self.name,
            "kind": self.kind,
            "unit": self.unit,
            "categories": [category.to_dict() for category in self.categories],
        }


__all__ = ["Category", "PropertyKind", "PropertyMetadata"]
