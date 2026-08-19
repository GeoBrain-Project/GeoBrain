"""Deterministic family-local resource records and pre-allocation budget gate.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .domain_contract import DomainContract
from .errors import GeomodelContractError, GeomodelResourceError

_T = TypeVar("_T")


def _non_negative_integer(value: object, *, field: str, object_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GeomodelContractError(
            "invalid Geomodel resource record",
            object_name=object_name,
            field=field,
            expected="non-negative integer",
            actual=value,
        )
    return value


@dataclass(frozen=True, slots=True)
class GeomodelResourceRequest:
    """Algorithm-independent resource-selection inputs.

    Attributes:
        domain: the :class:`DomainContract` sized against.
        n_realizations / workers: run shape.
        autograd: whether gradients are requested.
        budget_bytes: optional budget the estimate is checked against.
    """

    domain: DomainContract
    n_realizations: int = 1
    workers: int = 1
    autograd: bool = False
    budget_bytes: int | None = None

    def __post_init__(self) -> None:
        name = type(self).__name__
        if not isinstance(self.domain, DomainContract):
            raise GeomodelContractError(
                "invalid Geomodel resource request domain",
                object_name=name,
                field="domain",
                expected="DomainContract",
                actual=type(self.domain).__name__,
            )
        for field_name in ("n_realizations", "workers"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GeomodelContractError(
                    "invalid Geomodel resource request",
                    object_name=name,
                    field=field_name,
                    expected="positive integer",
                    actual=value,
                )
        if not isinstance(self.autograd, bool):
            raise GeomodelContractError(
                "invalid Geomodel resource request",
                object_name=name,
                field="autograd",
                expected="bool",
                actual=type(self.autograd).__name__,
            )
        if self.budget_bytes is not None:
            _non_negative_integer(self.budget_bytes, field="budget_bytes", object_name=name)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native request record."""
        return {
            "domain": self.domain.to_dict(),
            "n_realizations": self.n_realizations,
            "workers": self.workers,
            "autograd": self.autograd,
            "budget_bytes": self.budget_bytes,
        }


@dataclass(frozen=True, slots=True)
class GeomodelResourceEstimate:
    """Deterministic work and byte accounting for one configured algorithm.

    Attributes:
        total_bytes / component_bytes: memory prediction.
        distance_checks / candidate_comparisons /
            covariance_evaluations: work counters.
        factorization_order / factorization_flops: solve cost.
        index_rebuilds / workers: execution facts.
        assumptions: estimate assumptions.
    """

    total_bytes: int
    component_bytes: tuple[tuple[str, int], ...]
    distance_checks: int
    candidate_comparisons: int
    covariance_evaluations: int
    factorization_order: int | None
    factorization_flops: int
    index_rebuilds: int
    workers: int
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        name = type(self).__name__
        total = _non_negative_integer(self.total_bytes, field="total_bytes", object_name=name)
        if isinstance(self.component_bytes, (str, bytes, bytearray)) or not isinstance(
            self.component_bytes, Sequence
        ):
            raise GeomodelContractError(
                "invalid Geomodel resource component accounting",
                object_name=name,
                field="component_bytes",
                expected="sequence of (unique name, non-negative bytes)",
                actual=self.component_bytes,
            )
        parts: list[tuple[str, int]] = []
        for item in self.component_bytes:
            if (
                not isinstance(item, Sequence)
                or isinstance(item, (str, bytes, bytearray))
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
            ):
                raise GeomodelContractError(
                    "invalid Geomodel resource component accounting",
                    object_name=name,
                    field="component_bytes",
                    expected="sequence of (unique name, non-negative bytes)",
                    actual=item,
                )
            parts.append(
                (
                    item[0],
                    _non_negative_integer(item[1], field=item[0], object_name=name),
                )
            )
        if len({part_name for part_name, _ in parts}) != len(parts):
            raise GeomodelContractError(
                "resource component names must be unique",
                object_name=name,
                field="component_bytes",
                expected="unique names",
                actual=[part_name for part_name, _ in parts],
            )
        if sum(value for _, value in parts) != total:
            raise GeomodelContractError(
                "resource total does not equal component bytes",
                object_name=name,
                field="total_bytes",
                expected=sum(value for _, value in parts),
                actual=total,
            )
        object.__setattr__(self, "component_bytes", tuple(parts))
        _non_negative_integer(self.distance_checks, field="distance_checks", object_name=name)
        _non_negative_integer(
            self.candidate_comparisons,
            field="candidate_comparisons",
            object_name=name,
        )
        _non_negative_integer(
            self.covariance_evaluations,
            field="covariance_evaluations",
            object_name=name,
        )
        if self.factorization_order is not None:
            _non_negative_integer(
                self.factorization_order,
                field="factorization_order",
                object_name=name,
            )
        _non_negative_integer(
            self.factorization_flops,
            field="factorization_flops",
            object_name=name,
        )
        _non_negative_integer(
            self.index_rebuilds,
            field="index_rebuilds",
            object_name=name,
        )
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers <= 0:
            raise GeomodelContractError(
                "invalid Geomodel resource worker count",
                object_name=name,
                field="workers",
                expected="positive integer",
                actual=self.workers,
            )
        if isinstance(self.assumptions, (str, bytes, bytearray)) or not isinstance(
            self.assumptions, Sequence
        ):
            raise GeomodelContractError(
                "invalid Geomodel resource assumptions",
                object_name=name,
                field="assumptions",
                expected="sequence of non-empty strings",
                actual=self.assumptions,
            )
        assumptions = tuple(self.assumptions)
        if not all(isinstance(item, str) and item for item in assumptions):
            raise GeomodelContractError(
                "invalid Geomodel resource assumptions",
                object_name=name,
                field="assumptions",
                expected="sequence of non-empty strings",
                actual=self.assumptions,
            )
        object.__setattr__(self, "assumptions", assumptions)

    def to_dict(self) -> dict[str, object]:
        """Return a strict JSON-native estimate record."""
        return {
            "total_bytes": self.total_bytes,
            "component_bytes": [
                {"name": name, "bytes": value} for name, value in self.component_bytes
            ],
            "distance_checks": self.distance_checks,
            "candidate_comparisons": self.candidate_comparisons,
            "covariance_evaluations": self.covariance_evaluations,
            "factorization_order": self.factorization_order,
            "factorization_flops": self.factorization_flops,
            "index_rebuilds": self.index_rebuilds,
            "workers": self.workers,
            "assumptions": list(self.assumptions),
        }


def enforce_budget(
    estimate: GeomodelResourceEstimate,
    budget_bytes: int | None,
    allocation: Callable[[], _T] | None = None,
) -> _T | None:
    """Reject an over-budget estimate before invoking an optional allocator.

    Args:
        estimate: the resource estimate to check.
        budget_bytes: budget to enforce (``None`` = no limit).
        allocation: optional callable run only when the budget fits.
    """
    if not isinstance(estimate, GeomodelResourceEstimate):
        raise GeomodelContractError(
            "invalid Geomodel resource estimate",
            object_name="enforce_budget",
            field="estimate",
            expected="GeomodelResourceEstimate",
            actual=type(estimate).__name__,
        )
    if budget_bytes is not None:
        budget = _non_negative_integer(
            budget_bytes,
            field="budget_bytes",
            object_name="enforce_budget",
        )
        if estimate.total_bytes > budget:
            raise GeomodelResourceError(
                "Geomodel resource estimate exceeds the configured budget",
                object_name="enforce_budget",
                field="budget_bytes",
                expected=f">= {estimate.total_bytes}",
                actual=budget,
                hint="reduce the domain/workers/realizations or raise budget_bytes",
            )
    if allocation is None:
        return None
    if not callable(allocation):
        raise GeomodelContractError(
            "allocation callback must be callable",
            object_name="enforce_budget",
            field="allocation",
            expected="callable",
            actual=type(allocation).__name__,
        )
    return allocation()


__all__ = [
    "GeomodelResourceEstimate",
    "GeomodelResourceRequest",
    "enforce_budget",
]
