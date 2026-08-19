"""Bounded value-keyed LRU cache for immutable Potential plans.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import Callable

import torch

from ..config import PotentialExecutionConfig
from ..errors import PotentialContractError
from .plan import PrismKernelPlan


def _key_payload(value: object, active: set[int] | None = None) -> object:
    if active is None:
        active = set()
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PotentialContractError(
                "Cache keys cannot contain non-finite floats.",
                object_name="PotentialPlanCache",
                field="key",
                expected="finite deterministic key values",
                actual=value,
                hint="Replace NaN or infinity with a finite explicit policy value.",
            )
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, torch.dtype):
        return ["torch.dtype", str(value)]
    if isinstance(value, torch.device):
        return ["torch.device", str(value)]
    if isinstance(value, torch.layout):
        return ["torch.layout", str(value)]
    if isinstance(value, PrismKernelPlan):
        return ["PrismKernelPlan", value.fingerprint]
    if isinstance(value, PotentialExecutionConfig):
        return [
            "PotentialExecutionConfig",
            {
                "strategy": value.strategy,
                "budget_bytes": value.budget_bytes,
                "observation_tile_size": value.observation_tile_size,
                "cell_tile_size": value.cell_tile_size,
            },
        ]
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise PotentialContractError(
                "Cache keys cannot contain recursive mappings.",
                object_name="PotentialPlanCache",
                field="key",
                expected="acyclic deterministic key data",
                actual="recursive mapping",
                hint="Replace recursive key data with an immutable value description.",
            )
        active.add(identity)
        try:
            entries = [
                [_key_payload(key, active), _key_payload(item, active)]
                for key, item in value.items()
            ]
            entries.sort(
                key=lambda entry: json.dumps(
                    entry[0],
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return ["mapping", entries]
        finally:
            active.remove(identity)
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise PotentialContractError(
                "Cache keys cannot contain recursive sequences.",
                object_name="PotentialPlanCache",
                field="key",
                expected="acyclic deterministic key data",
                actual="recursive sequence",
                hint="Replace recursive key data with an immutable value description.",
            )
        active.add(identity)
        try:
            return ["sequence", [_key_payload(item, active) for item in value]]
        finally:
            active.remove(identity)
    raise PotentialContractError(
        "Cache key contains an unsupported value type.",
        object_name="PotentialPlanCache",
        field="key",
        expected="JSON-like values, PotentialExecutionConfig, or PrismKernelPlan",
        actual=value,
        hint="Use the plan fingerprint, execution policy, and immutable operator physics values.",
    )


def _canonical_key(value: object) -> bytes:
    payload = _key_payload(value)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


class PotentialPlanCache:
    """Bounded LRU cache whose keys and plans are immutable by public value."""

    __slots__ = ("_entries", "_evictions", "_hits", "_max_entries", "_misses")

    def __init__(self, max_entries: int) -> None:
        if type(max_entries) is not int or max_entries <= 0:
            raise PotentialContractError(
                "Cache capacity must be a positive non-Boolean integer.",
                object_name=type(self).__name__,
                field="max_entries",
                expected="positive int; bool is forbidden",
                actual=max_entries,
                hint="Provide the maximum number of immutable plans retained by the cache.",
            )
        self._max_entries = max_entries
        self._entries: OrderedDict[bytes, PrismKernelPlan] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def evictions(self) -> int:
        return self._evictions

    def stats(self) -> dict[str, int]:
        """Return stable JSON-native cache counters without internal keys or tensors."""
        return {
            "max_entries": self.max_entries,
            "size": self.size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }

    def get_or_create(
        self,
        key: object,
        factory: Callable[[], PrismKernelPlan],
    ) -> PrismKernelPlan:
        """Return and promote an LRU hit, or create exactly once on a miss."""
        canonical_key = _canonical_key(key)
        if not callable(factory):
            raise PotentialContractError(
                "Cache factory must be callable.",
                object_name=type(self).__name__,
                field="factory",
                expected="zero-argument callable returning PrismKernelPlan",
                actual=factory,
                hint="Pass a callable that builds one validated immutable plan.",
            )
        cached = self._entries.pop(canonical_key, None)
        if cached is not None:
            self._entries[canonical_key] = cached
            self._hits += 1
            return cached
        self._misses += 1
        created = factory()
        if not isinstance(created, PrismKernelPlan):
            raise PotentialContractError(
                "Cache factory returned the wrong value type.",
                object_name=type(self).__name__,
                field="factory",
                expected="PrismKernelPlan",
                actual=created,
                hint="Return a plan built by PrismKernelPlan.build().",
            )
        self._entries[canonical_key] = created
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        return created


__all__ = ["PotentialPlanCache"]
