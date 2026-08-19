"""Owned, recursively immutable metadata for decision results.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch

from geobrain.core.errors import GeoBrainError


class FrozenMetadata(Mapping[str, Any]):
    """A small pickle-safe immutable mapping with recursively frozen values."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, values: Mapping[str, Any]) -> None:
        items = tuple((key, _freeze_value(value)) for key, value in values.items())
        self._items = items
        self._lookup = dict(items)

    def __getitem__(self, key: str) -> Any:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._lookup)

    def __len__(self) -> int:
        return len(self._lookup)

    def __repr__(self) -> str:
        return f"FrozenMetadata({self._lookup!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __reduce__(self) -> tuple[type[FrozenMetadata], tuple[dict[str, Any]]]:
        return FrozenMetadata, (dict(self._items),)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenMetadata):
        return value
    if isinstance(value, Mapping):
        return FrozenMetadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, torch.Tensor):
        return _freeze_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _freeze_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(f"metadata value {type(value).__name__!r} is not freezeable")


def freeze_metadata(
    metadata: Mapping[str, Any],
    *,
    object_name: str,
) -> FrozenMetadata:
    """Validate, detach, and recursively freeze result metadata."""
    if not isinstance(metadata, Mapping):
        raise GeoBrainError(
            "metadata must be a mapping",
            object_name=object_name,
            field="metadata",
            expected="mapping[str, immutable value]",
            actual=type(metadata),
        )
    if not all(isinstance(key, str) for key in metadata):
        raise GeoBrainError(
            "metadata keys must be strings",
            object_name=object_name,
            field="metadata",
            expected="mapping with string keys",
            actual=tuple(type(key).__name__ for key in metadata),
        )
    try:
        return FrozenMetadata(metadata)
    except TypeError as exc:
        raise GeoBrainError(
            "metadata contains an unsupported mutable value",
            object_name=object_name,
            field="metadata",
            expected="recursively freezeable values",
            actual=str(exc),
        ) from exc


def thaw_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached mutable plain-dict/list representation."""

    def thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [thaw(item) for item in value]
        if isinstance(value, frozenset):
            return {thaw(item) for item in value}
        return value

    return {key: thaw(value) for key, value in metadata.items()}


__all__ = ["FrozenMetadata", "freeze_metadata", "thaw_metadata"]
