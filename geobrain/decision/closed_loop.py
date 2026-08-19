"""Generic closed-loop ensemble update and decision orchestration.

The decision layer owns orchestration, status, and bounded history. Concrete
inversion or posterior-update implementations are injected through
``EnsembleUpdater`` and remain owned by their respective packages.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePath
from types import MappingProxyType
from typing import Any

import torch

from geobrain.core.validation import validate_non_negative_int
from geobrain.core.context import ForwardContext
from geobrain.core.errors import GeoBrainError
from geobrain.decision.protocols import (
    CancellationCheck,
    EnsembleUpdater,
    HistoryPolicy,
)
from geobrain.decision.status import DecisionRunStatus

logger = logging.getLogger(__name__)


def _freeze_payload(
    value: Any,
    *,
    field_name: str,
    active: set[int],
) -> Any:
    """Own supported nested payloads or reject an unsafe caller alias."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            Enum,
            range,
            torch.dtype,
            torch.device,
            PurePath,
        ),
    ):
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise GeoBrainError(
                "ClosedLoopStep payloads must not contain recursive mappings",
                object_name="ClosedLoopStep",
                field=field_name,
                expected="acyclic owned payload",
                actual="recursive mapping",
            )
        active.add(identity)
        try:
            return MappingProxyType(
                {
                    _freeze_payload(
                        key,
                        field_name=f"{field_name}.key",
                        active=active,
                    ): _freeze_payload(
                        item,
                        field_name=f"{field_name}[{key!r}]",
                        active=active,
                    )
                    for key, item in value.items()
                }
            )
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise GeoBrainError(
                "ClosedLoopStep payloads must not contain recursive sequences",
                object_name="ClosedLoopStep",
                field=field_name,
                expected="acyclic owned payload",
                actual="recursive sequence",
            )
        active.add(identity)
        try:
            return tuple(
                _freeze_payload(
                    item,
                    field_name=f"{field_name}[{index}]",
                    active=active,
                )
                for index, item in enumerate(value)
            )
        finally:
            active.remove(identity)
    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in active:
            raise GeoBrainError(
                "ClosedLoopStep payloads must not contain recursive sets",
                object_name="ClosedLoopStep",
                field=field_name,
                expected="acyclic owned payload",
                actual="recursive set",
            )
        active.add(identity)
        try:
            return frozenset(
                _freeze_payload(
                    item,
                    field_name=field_name,
                    active=active,
                )
                for item in value
            )
        finally:
            active.remove(identity)
    try:
        copied = copy.deepcopy(value)
    except Exception as exc:
        raise GeoBrainError(
            "ClosedLoopStep payload leaf could not be defensively copied",
            object_name="ClosedLoopStep",
            field=field_name,
            expected="copyable owned payload leaf",
            actual=type(value),
        ) from exc
    if copied is value:
        raise GeoBrainError(
            "ClosedLoopStep payload leaf could not be copied into owned state",
            object_name="ClosedLoopStep",
            field=field_name,
            expected="owned payload copy with distinct identity",
            actual=type(value),
        )
    return copied


def _thaw_payload(value: Any) -> Any:
    """Convert frozen built-in containers to pickle-compatible owned values."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {
            _thaw_payload(key): _thaw_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_thaw_payload(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw_payload(item) for item in value)
    return copy.deepcopy(value)


def _restore_closed_loop_step(state: Mapping[str, Any]) -> "ClosedLoopStep":
    """Restore a pickled step through the public ownership contract."""
    return ClosedLoopStep(**state)


@dataclass(frozen=True)
class ClosedLoopStep:
    """Record of one completed update and decision.

    ``ensemble`` is owned by the record only when selected by the history
    policy. ``HistoryPolicy.NONE`` records metadata with ``ensemble=None``.

    Attributes:
        step: 0-based loop index.
        ensemble: post-update ensemble snapshot.
        observed: data acquired this step.
        decision: the decision taken.
        decision_info: extras from ``decision_fn``.
        elapsed: wall time of the step.
    """

    step: int
    ensemble: torch.Tensor | None
    observed: Any
    decision: Any
    decision_info: Mapping[str, Any]
    elapsed: float

    def __post_init__(self) -> None:
        """Recursively own every retained caller-provided payload."""
        ensemble = self.ensemble
        if ensemble is not None and not isinstance(ensemble, torch.Tensor):
            raise GeoBrainError(
                "ClosedLoopStep ensemble must be a tensor or None",
                object_name="ClosedLoopStep",
                field="ensemble",
                expected="torch.Tensor or None",
                actual=type(ensemble),
            )
        object.__setattr__(
            self,
            "ensemble",
            None if ensemble is None else ensemble.detach().clone(),
        )
        object.__setattr__(
            self,
            "observed",
            _freeze_payload(
                self.observed,
                field_name="observed",
                active=set(),
            ),
        )
        object.__setattr__(
            self,
            "decision",
            _freeze_payload(
                self.decision,
                field_name="decision",
                active=set(),
            ),
        )
        object.__setattr__(
            self,
            "decision_info",
            _freeze_payload(
                self.decision_info,
                field_name="decision_info",
                active=set(),
            ),
        )

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Serialize plain owned state and restore through validation."""
        state = {
            "step": self.step,
            "ensemble": (
                None
                if self.ensemble is None
                else self.ensemble.detach().clone()
            ),
            "observed": _thaw_payload(self.observed),
            "decision": _thaw_payload(self.decision),
            "decision_info": _thaw_payload(self.decision_info),
            "elapsed": self.elapsed,
        }
        return _restore_closed_loop_step, (state,)


@dataclass(frozen=True)
class ClosedLoopRunResult:
    """Owned terminal result for one :meth:`ClosedLoopManager.run` call.

    Attributes:
        steps: the per-step :class:`ClosedLoopStep` records.
        completed_updates: update steps actually run.
        status: completed / stopped marker.
    """

    steps: tuple[ClosedLoopStep, ...]
    completed_updates: int
    status: DecisionRunStatus

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        completed = validate_non_negative_int(
            self.completed_updates,
            owner="ClosedLoopRunResult",
            field="completed_updates",
        )
        if completed != len(steps):
            raise GeoBrainError(
                "completed_updates must match the number of steps",
                object_name="ClosedLoopRunResult",
                field="completed_updates",
                expected=len(steps),
                actual=completed,
            )
        if not isinstance(self.status, DecisionRunStatus):
            raise GeoBrainError(
                "status must be a DecisionRunStatus",
                object_name="ClosedLoopRunResult",
                field="status",
                expected=DecisionRunStatus,
                actual=self.status,
            )
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "completed_updates", completed)


def _validate_callback(value: object, *, field: str, expected: str) -> None:
    if not callable(value):
        raise GeoBrainError(
            f"ClosedLoopManager {field} must be callable",
            object_name="ClosedLoopManager",
            field=field,
            expected=expected,
            actual=type(value).__name__,
        )


def _validate_ensemble(
    value: object,
    *,
    field: str,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise GeoBrainError(
            f"{field} must be a tensor",
            object_name="ClosedLoopManager",
            field=field,
            expected="torch.Tensor ensemble",
            actual=type(value).__name__,
        )
    if reference is not None and (
        value.shape != reference.shape
        or value.dtype != reference.dtype
        or value.device != reference.device
    ):
        raise GeoBrainError(
            f"{field} changed the ensemble structure",
            object_name="ClosedLoopManager",
            field=field,
            expected={
                "shape": tuple(reference.shape),
                "dtype": str(reference.dtype),
                "device": str(reference.device),
            },
            actual={
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            },
        )
    return value


class ClosedLoopManager:
    """Run observe-update-decide cycles through an injected updater.

    ``completed_updates`` and ``status`` describe the latest run boundary.
    The current ensemble is always read from the updater. History retains:

    - ``NONE``: every decision record, with no ensemble tensors;
    - ``LATEST``: only the most recent record and its owned ensemble; and
    - ``ALL``: every record, each with an owned ensemble.

    Args:
        updater: ensemble updater applied after each acquisition.
        problem_fn: builds the forward/assimilation problem per step.
        decision_fn: maps the current ensemble to the next decision.
        n_update_steps: updater iterations per loop step.
        ctx: optional shared :class:`~geobrain.core.ForwardContext`.
        history_policy: what per-step state to retain.
    """

    def __init__(
        self,
        updater: EnsembleUpdater,
        problem_fn: Callable[[Any], Any],
        decision_fn: Callable[[torch.Tensor], Any],
        *,
        n_update_steps: int = 200,
        ctx: ForwardContext | None = None,
        history_policy: HistoryPolicy = HistoryPolicy.LATEST,
        cancellation: CancellationCheck | None = None,
    ) -> None:
        update = getattr(updater, "update", None)
        if not callable(update):
            raise GeoBrainError(
                "ClosedLoopManager updater must implement update",
                object_name="ClosedLoopManager",
                field="updater",
                expected="EnsembleUpdater (ensemble property and update method)",
                actual=type(updater).__name__,
            )
        try:
            initial = updater.ensemble
        except (AttributeError, TypeError):
            raise GeoBrainError(
                "ClosedLoopManager updater must expose an ensemble tensor",
                object_name="ClosedLoopManager",
                field="updater",
                expected="EnsembleUpdater (ensemble property and update method)",
                actual=type(updater).__name__,
            ) from None
        _validate_ensemble(initial, field="updater")
        _validate_callback(
            problem_fn,
            field="problem_fn",
            expected="callable observed -> target",
        )
        _validate_callback(
            decision_fn,
            field="decision_fn",
            expected="callable ensemble -> decision or (decision, info)",
        )
        if not isinstance(history_policy, HistoryPolicy):
            raise GeoBrainError(
                "history_policy must be a HistoryPolicy",
                object_name="ClosedLoopManager",
                field="history_policy",
                expected=HistoryPolicy,
                actual=history_policy,
            )
        if cancellation is not None:
            _validate_callback(
                cancellation,
                field="cancellation",
                expected="callable () -> bool or None",
            )

        self.updater = updater
        self.problem_fn = problem_fn
        self.decision_fn = decision_fn
        self.n_update_steps = validate_non_negative_int(
            n_update_steps,
            owner="ClosedLoopManager",
            field="n_update_steps",
        )
        self.ctx = ctx
        self.history_policy = history_policy
        self.cancellation = cancellation
        self.history: list[ClosedLoopStep] = []
        self.completed_updates = 0
        self._total_updates = 0
        self._running = False
        self.status = DecisionRunStatus.COMPLETED

    @property
    def ensemble(self) -> torch.Tensor:
        """Return the updater's current ensemble without making a history copy."""
        return _validate_ensemble(
            self.updater.ensemble,
            field="updater.ensemble",
        ).detach()

    def _cancelled(self) -> bool:
        if self.cancellation is None:
            return False
        result = self.cancellation()
        if not isinstance(result, bool):
            raise GeoBrainError(
                "cancellation callback must return bool",
                object_name="ClosedLoopManager",
                field="cancellation return",
                expected=bool,
                actual=type(result).__name__,
            )
        return result

    def _retain(self, record: ClosedLoopStep) -> None:
        if self.history_policy is HistoryPolicy.LATEST:
            self.history[:] = [record]
        else:
            self.history.append(record)

    def step(
        self,
        observed: Any,
        n_update_steps: int | None = None,
        verbose: bool = False,
    ) -> ClosedLoopStep:
        """Complete one updater call and decision without mid-call cancellation."""
        if n_update_steps is None:
            n_steps = self.n_update_steps
        else:
            n_steps = validate_non_negative_int(
                n_update_steps,
                owner="ClosedLoopManager.step",
                field="n_update_steps",
            )
        cycle = self._total_updates
        before = self.ensemble
        target = self.problem_fn(observed)
        t0 = time.perf_counter()
        output = _validate_ensemble(
            self.updater.update(target, n_steps=n_steps, ctx=self.ctx),
            field="updater.update return",
            reference=before,
        )
        current = _validate_ensemble(
            self.updater.ensemble,
            field="updater.update return",
            reference=before,
        )
        if not torch.equal(output.detach(), current.detach()):
            raise GeoBrainError(
                "updater output must describe its current ensemble",
                object_name="ClosedLoopManager",
                field="updater.update return",
                expected="same values as updater.ensemble",
                actual="stale or inconsistent updater output",
            )

        raw = self.decision_fn(current)
        if isinstance(raw, tuple):
            if len(raw) != 2:
                raise GeoBrainError(
                    "decision_fn tuple must be exactly (decision, info)",
                    object_name="ClosedLoopManager",
                    field="decision_fn return",
                    expected="2-tuple or bare decision",
                    actual=f"tuple of length {len(raw)}",
                )
            decision, info = raw
            if not isinstance(info, dict):
                raise GeoBrainError(
                    "decision_fn info must be a dict",
                    object_name="ClosedLoopManager",
                    field="decision_fn return",
                    expected="(decision, dict) or bare decision",
                    actual=type(info).__name__,
                )
        else:
            decision, info = raw, {}

        retained = (
            None
            if self.history_policy is HistoryPolicy.NONE
            else current
        )
        record = ClosedLoopStep(
            step=cycle,
            ensemble=retained,
            observed=observed,
            decision=decision,
            decision_info=dict(info),
            elapsed=time.perf_counter() - t0,
        )
        self._total_updates += 1
        if self._running:
            self.completed_updates += 1
        else:
            self.completed_updates = 1
        self.status = DecisionRunStatus.COMPLETED
        self._retain(record)
        if verbose:
            logger.info(
                "Closed-loop update %d completed: decision=%r",
                cycle,
                decision,
            )
        return record

    def run(
        self,
        observations: Iterable[Any],
        n_update_steps: int | None = None,
        verbose: bool = False,
    ) -> ClosedLoopRunResult:
        """Run updates until inputs are exhausted or cancellation is requested."""
        records: list[ClosedLoopStep] = []
        self.completed_updates = 0
        self.status = DecisionRunStatus.COMPLETED
        self._running = True
        try:
            for observed in observations:
                if self._cancelled():
                    self.status = DecisionRunStatus.CANCELLED
                    break
                record = self.step(
                    observed,
                    n_update_steps=n_update_steps,
                    verbose=verbose,
                )
                if (
                    self.history_policy is HistoryPolicy.LATEST
                    and records
                    and records[-1].ensemble is not None
                ):
                    records[-1] = replace(records[-1], ensemble=None)
                records.append(record)
        finally:
            self._running = False
        return ClosedLoopRunResult(
            steps=tuple(records),
            completed_updates=self.completed_updates,
            status=self.status,
        )

    @property
    def decisions(self) -> list[Any]:
        """Return decisions retained by the selected history policy."""
        return [record.decision for record in self.history]

    @property
    def n_cycles(self) -> int:
        """Return the number of successfully completed updater calls."""
        return self._total_updates

    def summary(self) -> str:
        """Return a compact status and retained-history summary."""
        lines = [
            "=== Closed-Loop History ===",
            f"Total updates   : {self.n_cycles}",
            f"Latest status   : {self.status.value}",
            f"Ensemble shape  : {list(self.ensemble.shape)}",
        ]
        for record in self.history:
            lines.append(
                f"  Update {record.step}: decision={record.decision} "
                f"({record.elapsed:.1f}s)",
            )
        return "\n".join(lines)


__all__ = ["ClosedLoopManager", "ClosedLoopRunResult", "ClosedLoopStep"]
