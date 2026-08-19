"""Strict SI Earth-field and execution configuration for Potential.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .errors import PotentialContractError


PotentialStrategy = Literal["auto", "dense", "tiled", "store"]
_STRATEGIES = frozenset({"auto", "dense", "tiled", "store"})


def _require_float(value: object, *, object_name: str, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PotentialContractError(
            f"{field} must be a finite float.",
            object_name=object_name,
            field=field,
            expected="finite float",
            actual=value,
            hint=f"Provide {field} explicitly as a finite Python float.",
        )
    return value


def _require_positive_integer_or_none(
    value: object,
    *,
    object_name: str,
    field: str,
) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise PotentialContractError(
            f"{field} must be a positive non-Boolean integer or None.",
            object_name=object_name,
            field=field,
            expected="positive int or None; bool is forbidden",
            actual=value,
            hint=f"Use None for budget resolution or provide a positive integer {field}.",
        )
    return value


@dataclass(frozen=True, slots=True)
class EarthField:
    """Earth field in tesla and degrees without implicit normalization.

    Attributes:
        intensity_tesla: ambient field strength [T].
        inclination_deg: field inclination [deg, down-positive].
        declination_deg: field declination [deg, east of north].
    """

    intensity_tesla: float
    inclination_deg: float
    declination_deg: float

    def __post_init__(self) -> None:
        intensity = _require_float(
            self.intensity_tesla,
            object_name=type(self).__name__,
            field="intensity_tesla",
        )
        inclination = _require_float(
            self.inclination_deg,
            object_name=type(self).__name__,
            field="inclination_deg",
        )
        _require_float(
            self.declination_deg,
            object_name=type(self).__name__,
            field="declination_deg",
        )
        if intensity <= 0.0:
            raise PotentialContractError(
                "Earth-field intensity must be positive.",
                object_name=type(self).__name__,
                field="intensity_tesla",
                expected="> 0 T",
                actual=intensity,
                hint="Provide a positive SI magnetic flux density in tesla.",
            )
        if not -90.0 <= inclination <= 90.0:
            raise PotentialContractError(
                "Earth-field inclination is outside its physical range.",
                object_name=type(self).__name__,
                field="inclination_deg",
                expected="[-90, 90] degrees",
                actual=inclination,
                hint="Provide inclination in degrees within the closed interval [-90, 90].",
            )


@dataclass(frozen=True, slots=True)
class PotentialExecutionConfig:
    """Explicit bounded execution policy for Potential kernels.

    Attributes:
        strategy: evaluation strategy id (``auto`` picks by budget).
        budget_bytes: memory budget for the tiling planner.
        observation_tile_size / cell_tile_size: manual tile overrides.
    """

    strategy: PotentialStrategy = "auto"
    budget_bytes: int = 536_870_912
    observation_tile_size: int | None = None
    cell_tile_size: int | None = None

    def __post_init__(self) -> None:
        object_name = type(self).__name__
        if type(self.strategy) is not str or self.strategy not in _STRATEGIES:
            raise PotentialContractError(
                "Unknown Potential execution strategy.",
                object_name=object_name,
                field="strategy",
                expected=["auto", "dense", "tiled", "store"],
                actual=self.strategy,
                hint="Select one exact supported strategy name.",
            )
        if type(self.budget_bytes) is not int or self.budget_bytes <= 0:
            raise PotentialContractError(
                "Execution budget must be a positive non-Boolean integer.",
                object_name=object_name,
                field="budget_bytes",
                expected="positive int; bool is forbidden",
                actual=self.budget_bytes,
                hint="Provide the available memory budget as a positive byte count.",
            )
        observation_tile_size = _require_positive_integer_or_none(
            self.observation_tile_size,
            object_name=object_name,
            field="observation_tile_size",
        )
        cell_tile_size = _require_positive_integer_or_none(
            self.cell_tile_size,
            object_name=object_name,
            field="cell_tile_size",
        )
        if self.strategy in {"dense", "store"} and observation_tile_size is not None:
            raise PotentialContractError(
                "Dense and stored execution do not accept observation tiles.",
                object_name=object_name,
                field="observation_tile_size",
                expected="None for dense/store",
                actual=observation_tile_size,
                hint="Remove tile sizes or select auto/tiled execution.",
            )
        if self.strategy in {"dense", "store"} and cell_tile_size is not None:
            raise PotentialContractError(
                "Dense and stored execution do not accept cell tiles.",
                object_name=object_name,
                field="cell_tile_size",
                expected="None for dense/store",
                actual=cell_tile_size,
                hint="Remove tile sizes or select auto/tiled execution.",
            )


__all__ = ["EarthField", "PotentialExecutionConfig", "PotentialStrategy"]
