"""Immutable execution choices for deterministic geostatistical simulation.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...errors import GeomodelContractError, GeomodelResourceError

WorkerBackend = Literal["serial", "process", "thread"]
NeighbourhoodBackend = Literal["indexed", "exhaustive"]


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GeomodelContractError(
            f"{field} must be a positive exact integer",
            object_name="SimulationExecutionConfig",
            field=field,
            expected="positive non-boolean int",
            actual=value,
        )
    resolved = int(value)
    if resolved < 1:
        raise GeomodelContractError(
            f"{field} must be a positive exact integer",
            object_name="SimulationExecutionConfig",
            field=field,
            expected="positive non-boolean int",
            actual=value,
        )
    return resolved


@dataclass(frozen=True, slots=True)
class SimulationExecutionConfig:
    """Owned execution configuration shared by all classical simulators.

    Attributes:
        n_realizations: realisation count.
        seed: master seed (per-realisation seeds derive from it).
        workers / worker_backend: parallel execution policy.
        neighbourhood_backend: search-index backend id.
        budget_bytes: optional memory budget.
    """

    n_realizations: int = 1
    seed: int | None = None
    workers: int = 1
    worker_backend: WorkerBackend = "serial"
    neighbourhood_backend: NeighbourhoodBackend = "indexed"
    budget_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "n_realizations",
            _positive_int(self.n_realizations, field="n_realizations"),
        )
        object.__setattr__(self, "workers", _positive_int(self.workers, field="workers"))
        if self.seed is not None:
            if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
                self.seed, (int, np.integer)
            ):
                raise GeomodelContractError(
                    "seed must be an integer or None",
                    object_name=type(self).__name__,
                    field="seed",
                    expected="int or None",
                    actual=self.seed,
                )
            object.__setattr__(self, "seed", int(self.seed))
        if self.worker_backend not in ("serial", "process", "thread"):
            raise GeomodelContractError(
                "worker backend is unsupported",
                object_name=type(self).__name__,
                field="worker_backend",
                expected="serial, process, or thread",
                actual=self.worker_backend,
            )
        if self.neighbourhood_backend not in ("indexed", "exhaustive"):
            raise GeomodelContractError(
                "neighbourhood backend is unsupported",
                object_name=type(self).__name__,
                field="neighbourhood_backend",
                expected="indexed or exhaustive",
                actual=self.neighbourhood_backend,
            )
        if self.budget_bytes is not None:
            object.__setattr__(
                self,
                "budget_bytes",
                _positive_int(self.budget_bytes, field="budget_bytes"),
            )

    def require_budget(self, required_bytes: int, *, component: str) -> None:
        """Reject an allocation before it is attempted when a budget is set."""
        if required_bytes < 0:
            raise ValueError("required_bytes must not be negative")
        if self.budget_bytes is not None and required_bytes > self.budget_bytes:
            raise GeomodelResourceError(
                "simulation allocation exceeds the configured resource budget",
                object_name=type(self).__name__,
                field="budget_bytes",
                expected=f">= {required_bytes} bytes for {component}",
                actual=self.budget_bytes,
            )


__all__ = ["NeighbourhoodBackend", "SimulationExecutionConfig", "WorkerBackend"]
