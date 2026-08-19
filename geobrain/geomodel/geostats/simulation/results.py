"""Immutable JSON-safe result records for simulation ensembles.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, NoReturn, cast

import numpy as np

from ...frames import GeoFrame, PropertyMetadata
from ...errors import GeomodelContractError
from .execution import SimulationExecutionConfig


class _ImmutableJSONDict(dict[str, object]):
    """A recursively frozen JSON object that remains directly serializable."""

    @staticmethod
    def _reject_mutation() -> NoReturn:
        raise TypeError("simulation diagnostics are immutable")

    def __setitem__(self, key: str, value: object) -> None:
        del key, value
        self._reject_mutation()

    def __delitem__(self, key: str) -> None:
        del key
        self._reject_mutation()

    def __ior__(self, value: object) -> _ImmutableJSONDict:  # type: ignore[override,misc]
        del value
        self._reject_mutation()

    def clear(self) -> None:
        self._reject_mutation()

    def pop(self, key: str, default: object = None) -> object:
        del key, default
        self._reject_mutation()

    def popitem(self) -> tuple[str, object]:
        self._reject_mutation()

    def setdefault(self, key: str, default: object = None) -> object:
        del key, default
        self._reject_mutation()

    def update(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self._reject_mutation()


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return _ImmutableJSONDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _owned_json_mapping(values: Mapping[str, object], *, object_name: str) -> Mapping[str, object]:
    """Copy diagnostics through strict JSON so callers cannot retain aliases."""
    try:
        payload = json.dumps(dict(values), allow_nan=False, sort_keys=True)
        decoded = cast(object, json.loads(payload))
    except (TypeError, ValueError) as exc:
        raise GeomodelContractError(
            "simulation diagnostics must be strict JSON values",
            object_name=object_name,
            field="diagnostics",
            expected="JSON object without NaN or infinity",
            actual=type(values).__name__,
        ) from exc
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise GeomodelContractError(
            "simulation diagnostics must be a JSON object",
            object_name=object_name,
            field="diagnostics",
            expected="mapping with string keys",
            actual=type(values).__name__,
        )
    return cast(_ImmutableJSONDict, _freeze_json(decoded))


@dataclass(frozen=True, slots=True)
class SimulationRealization:
    """One indexed, seeded simulation frame and its owned diagnostics.

    Attributes:
        index: realisation index.
        seed: the derived per-realisation seed.
        frame: the realised :class:`GeoFrame` (column ``simulation``).
        diagnostics: per-realisation diagnostics.
    """

    index: int
    seed: int
    frame: GeoFrame
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if isinstance(self.index, (bool, np.bool_)) or not isinstance(
            self.index, (int, np.integer)
        ) or int(self.index) < 0:
            raise GeomodelContractError(
                "realization index must be a non-negative exact integer",
                object_name=type(self).__name__,
                field="index",
                expected="non-negative int",
                actual=self.index,
            )
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)):
            raise GeomodelContractError(
                "realization seed must be an exact integer",
                object_name=type(self).__name__,
                field="seed",
                expected="int",
                actual=self.seed,
            )
        if not isinstance(self.frame, GeoFrame):
            raise GeomodelContractError(
                "realization frame must be a GeoFrame",
                object_name=type(self).__name__,
                field="frame",
                expected="GeoFrame",
                actual=type(self.frame).__name__,
            )
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "diagnostics",
            _owned_json_mapping(self.diagnostics, object_name=type(self).__name__),
        )


@dataclass(frozen=True, slots=True)
class SimulationEnsemble:
    """A complete ordered set of simulation realizations.

    Attributes:
        property: the simulated :class:`PropertyMetadata`.
        realizations: the :class:`SimulationRealization` records.
        execution: the execution config that produced them.
        diagnostics: run-level diagnostics.
    """

    property: PropertyMetadata
    realizations: tuple[SimulationRealization, ...]
    execution: SimulationExecutionConfig
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.property, PropertyMetadata):
            raise GeomodelContractError(
                "ensemble property must be PropertyMetadata",
                object_name=type(self).__name__,
                field="property",
                expected="PropertyMetadata",
                actual=type(self.property).__name__,
            )
        realizations = tuple(self.realizations)
        if any(not isinstance(item, SimulationRealization) for item in realizations):
            raise GeomodelContractError(
                "ensemble realizations must be SimulationRealization records",
                object_name=type(self).__name__,
                field="realizations",
                expected="tuple[SimulationRealization, ...]",
                actual=type(self.realizations).__name__,
            )
        if tuple(item.index for item in realizations) != tuple(range(len(realizations))):
            raise GeomodelContractError(
                "ensemble realization indices must be ordered and contiguous",
                object_name=type(self).__name__,
                field="realizations",
                expected="indices 0..n-1",
                actual=[item.index for item in realizations],
            )
        if not isinstance(self.execution, SimulationExecutionConfig):
            raise GeomodelContractError(
                "ensemble execution must be SimulationExecutionConfig",
                object_name=type(self).__name__,
                field="execution",
                expected="SimulationExecutionConfig",
                actual=type(self.execution).__name__,
            )
        if len(realizations) != self.execution.n_realizations:
            raise GeomodelContractError(
                "ensemble realization count must match execution configuration",
                object_name=type(self).__name__,
                field="realizations",
                expected=self.execution.n_realizations,
                actual=len(realizations),
            )
        object.__setattr__(self, "realizations", realizations)
        object.__setattr__(
            self,
            "diagnostics",
            _owned_json_mapping(self.diagnostics, object_name=type(self).__name__),
        )


__all__ = ["SimulationEnsemble", "SimulationRealization"]
