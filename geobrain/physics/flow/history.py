"""Bounded accepted-state history for Flow execution.

Rejected attempts are diagnostics, never accepted history entries.  The writer
does not detach tensors: the execution configuration, rather than bookkeeping,
owns the selected autograd behavior.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

import torch

from .config import FlowHistoryConfig
from .errors import FlowContractError
from .solvers.diagnostics import FlowConvergenceDiagnostics


@dataclass(frozen=True, slots=True)
class FlowExecutionAccounting:
    """Measured work and retained tensor bytes for one Flow execution."""

    accepted_steps: int
    rejected_steps: int
    residual_evaluations: int
    jacobian_assemblies: int
    linear_solves: int
    recomputed_steps: int
    retained_state_bytes: int

    def to_dict(self) -> dict[str, int]:
        """Return strict-JSON-compatible accounting."""
        return {
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "residual_evaluations": self.residual_evaluations,
            "jacobian_assemblies": self.jacobian_assemblies,
            "linear_solves": self.linear_solves,
            "recomputed_steps": self.recomputed_steps,
            "retained_state_bytes": self.retained_state_bytes,
        }


@dataclass(frozen=True, slots=True)
class FlowCheckpoint:
    """Restartable accepted-state snapshot and deterministic input schedule."""

    accepted_step: int
    time_s: float
    state: Mapping[str, torch.Tensor]
    accepted_dt_s: tuple[float, ...]
    control_schedule: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class FlowHistory:
    """Immutable view of retained accepted states and rejected diagnostics."""

    times_s: tuple[float, ...]
    states: tuple[Mapping[str, torch.Tensor], ...]
    checkpoints: tuple[FlowCheckpoint, ...]
    rejected: tuple[FlowConvergenceDiagnostics, ...]
    accounting: FlowExecutionAccounting
    retained_step_indices: tuple[int, ...]
    accepted_dt_s: tuple[float, ...]
    control_schedule: tuple[Mapping[str, object], ...]

    def checkpoint_at_or_before(self, accepted_step: int) -> FlowCheckpoint:
        """Return the nearest restart checkpoint at or before ``accepted_step``."""
        eligible = [
            checkpoint
            for checkpoint in self.checkpoints
            if checkpoint.accepted_step <= accepted_step
        ]
        if not eligible:
            raise FlowContractError(
                "no retained checkpoint precedes the requested step",
                object_name=type(self).__name__,
                field="accepted_step",
                expected="an index at or after the first checkpoint",
                actual=accepted_step,
            )
        return eligible[-1]


def _owned_state(state: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise FlowContractError(
            "history state must be a non-empty tensor mapping",
            object_name="FlowHistoryWriter",
            field="state",
            expected="non-empty Mapping[str, Tensor]",
            actual=type(state).__name__,
        )
    owned: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise FlowContractError(
                "history state entries must have non-empty names and tensor values",
                object_name="FlowHistoryWriter",
                field="state",
                expected="Mapping[non-empty str, Tensor]",
                actual=(key, type(tensor).__name__),
            )
        # Keep the selected graph while taking ownership of storage. A plain
        # reference lets callers mutate a recorded checkpoint and break replay.
        owned[key] = tensor.clone()
    return MappingProxyType(owned)


def _owned_control(control: Mapping[str, object] | None) -> Mapping[str, object]:
    if control is None:
        return MappingProxyType({})
    if not isinstance(control, Mapping):
        raise FlowContractError(
            "control schedule entry must be a mapping",
            object_name="FlowHistoryWriter",
            field="control",
            expected="Mapping[str, object]",
            actual=type(control).__name__,
        )

    def freeze(value: object, *, path: str) -> object:
        if isinstance(value, Mapping):
            owned: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise FlowContractError(
                        "control schedule keys must be non-empty strings",
                        object_name="FlowHistoryWriter",
                        field="control",
                        expected="nested Mapping[non-empty str, JSON value]",
                        actual={"path": path, "key": key},
                    )
                owned[key] = freeze(item, path=f"{path}.{key}")
            return MappingProxyType(owned)
        if isinstance(value, (tuple, list)):
            return tuple(freeze(item, path=f"{path}[{index}]") for index, item in enumerate(value))
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise FlowContractError(
            "control schedule values must be finite JSON data",
            object_name="FlowHistoryWriter",
            field="control",
            expected="nested mappings/sequences and finite JSON scalars",
            actual={"path": path, "type": type(value).__name__},
        )

    frozen = freeze(control, path="control")
    assert isinstance(frozen, Mapping)
    return frozen


def _state_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


class FlowHistoryWriter:
    """Online writer implementing all five bounded history policies."""

    def __init__(
        self,
        config: FlowHistoryConfig,
        *,
        accepted_step_bound: int,
    ) -> None:
        if not isinstance(config, FlowHistoryConfig):
            raise FlowContractError(
                "history writer requires FlowHistoryConfig",
                object_name=type(self).__name__,
                field="config",
                expected=FlowHistoryConfig,
                actual=type(config),
            )
        if (
            isinstance(accepted_step_bound, bool)
            or not isinstance(accepted_step_bound, int)
            or accepted_step_bound < 0
        ):
            raise FlowContractError(
                "accepted_step_bound must be a non-negative integer",
                object_name=type(self).__name__,
                field="accepted_step_bound",
                expected="integer >= 0",
                actual=accepted_step_bound,
            )
        self.config = config
        self.accepted_step_bound = accepted_step_bound
        self._accepted_steps = 0
        self._retained: dict[int, tuple[float, Mapping[str, torch.Tensor]]] = {}
        self._rejected: list[FlowConvergenceDiagnostics] = []
        self._dt_schedule: list[float] = []
        self._control_schedule: list[Mapping[str, object]] = []
        self._residual_evaluations = 0
        self._jacobian_assemblies = 0
        self._linear_solves = 0
        self._recomputed_steps = 0
        self._initial_recorded = False
        self._rolling_final_index: int | None = None

    @property
    def accepted_steps(self) -> int:
        return self._accepted_steps

    def record_initial(self, *, time_s: float, state: Mapping[str, torch.Tensor]) -> None:
        if self._initial_recorded:
            raise FlowContractError(
                "initial history state was already recorded",
                object_name=type(self).__name__,
                field="state",
                expected="exactly one initial state",
                actual="duplicate",
            )
        time_value = self._time(time_s)
        self._retained[0] = (time_value, _owned_state(state))
        self._initial_recorded = True

    def record_accepted(
        self,
        *,
        time_s: float,
        state: Mapping[str, torch.Tensor],
        dt_s: float,
        control: Mapping[str, object] | None = None,
        residual_evaluations: int = 0,
        jacobian_assemblies: int = 0,
        linear_solves: int = 0,
    ) -> None:
        if not self._initial_recorded:
            raise FlowContractError(
                "record_initial must precede accepted states",
                object_name=type(self).__name__,
                field="state",
                expected="initial state first",
                actual="missing",
            )
        if self._accepted_steps >= self.accepted_step_bound:
            raise FlowContractError(
                "accepted execution exceeded its declared step bound",
                object_name=type(self).__name__,
                field="accepted_step_bound",
                expected=self.accepted_step_bound,
                actual=self._accepted_steps + 1,
            )
        time_value = self._time(time_s)
        dt_value = self._positive_float(dt_s, field="dt_s")
        last_time = max(value[0] for value in self._retained.values())
        if time_value <= last_time:
            raise FlowContractError(
                "accepted history times must be strictly increasing",
                object_name=type(self).__name__,
                field="time_s",
                expected=f"> {last_time}",
                actual=time_value,
            )
        self._accepted_steps += 1
        index = self._accepted_steps
        owned = _owned_state(state)
        self._dt_schedule.append(dt_value)
        self._control_schedule.append(_owned_control(control))
        self._add_work(
            residual_evaluations=residual_evaluations,
            jacobian_assemblies=jacobian_assemblies,
            linear_solves=linear_solves,
        )
        retain_by_policy = self._should_retain(index=index, time_s=time_value)
        # Final is not known online. Keep exactly one rolling candidate for
        # policies that do not retain every state, replacing the prior candidate.
        if self.config.mode == "final":
            self._retained = {index: (time_value, owned)}
            self._rolling_final_index = index
        elif self.config.mode == "all":
            self._retained[index] = (time_value, owned)
        else:
            if self._rolling_final_index is not None:
                self._retained.pop(self._rolling_final_index, None)
                self._rolling_final_index = None
            self._retained[index] = (time_value, owned)
            if not retain_by_policy:
                self._rolling_final_index = index

    def record_rejected(
        self,
        diagnostics: FlowConvergenceDiagnostics,
        *,
        residual_evaluations: int = 0,
        jacobian_assemblies: int = 0,
        linear_solves: int = 0,
    ) -> None:
        if not isinstance(diagnostics, FlowConvergenceDiagnostics):
            raise FlowContractError(
                "rejected step requires canonical convergence diagnostics",
                object_name=type(self).__name__,
                field="diagnostics",
                expected=FlowConvergenceDiagnostics,
                actual=type(diagnostics),
            )
        if diagnostics.converged:
            raise FlowContractError(
                "a converged diagnostic cannot be recorded as rejected",
                object_name=type(self).__name__,
                field="diagnostics.converged",
                expected=False,
                actual=True,
            )
        self._rejected.append(diagnostics)
        self._add_work(
            residual_evaluations=residual_evaluations,
            jacobian_assemblies=jacobian_assemblies,
            linear_solves=linear_solves,
        )

    def record_recomputed(
        self,
        steps: int,
        *,
        residual_evaluations: int = 0,
        jacobian_assemblies: int = 0,
        linear_solves: int = 0,
    ) -> None:
        self._recomputed_steps += self._nonnegative_int(steps, field="steps")
        self._add_work(
            residual_evaluations=residual_evaluations,
            jacobian_assemblies=jacobian_assemblies,
            linear_solves=linear_solves,
        )

    def finalize(
        self,
        *,
        require_complete_reports: bool = True,
    ) -> FlowHistory:
        if not self._initial_recorded:
            raise FlowContractError(
                "cannot finalize history without an initial state",
                object_name=type(self).__name__,
                field="state",
                expected="record_initial call",
                actual="missing",
            )
        entries = tuple(sorted(self._retained.items()))
        indices = tuple(index for index, _ in entries)
        times = tuple(entry[0] for _, entry in entries)
        states = tuple(entry[1] for _, entry in entries)
        if self.config.mode == "report" and require_complete_reports:
            missing = tuple(
                report
                for report in self.config.report_times_s
                if not any(
                    math.isclose(time_s, report, rel_tol=1e-12, abs_tol=1e-12) for time_s in times
                )
            )
            if missing:
                raise FlowContractError(
                    "requested report stations were not accepted",
                    object_name=type(self).__name__,
                    field="report_times_s",
                    expected=self.config.report_times_s,
                    actual=missing,
                )
        checkpoints: list[FlowCheckpoint] = []
        if self.config.mode in {"checkpoint", "recompute"}:
            for index, (time_s, state) in entries:
                checkpoints.append(
                    FlowCheckpoint(
                        accepted_step=index,
                        time_s=time_s,
                        state=state,
                        accepted_dt_s=tuple(self._dt_schedule[:index]),
                        control_schedule=tuple(self._control_schedule[:index]),
                    )
                )
        accounting = FlowExecutionAccounting(
            accepted_steps=self._accepted_steps,
            rejected_steps=len(self._rejected),
            residual_evaluations=self._residual_evaluations,
            jacobian_assemblies=self._jacobian_assemblies,
            linear_solves=self._linear_solves,
            recomputed_steps=self._recomputed_steps,
            retained_state_bytes=sum(_state_bytes(state) for state in states),
        )
        return FlowHistory(
            times_s=times,
            states=states,
            checkpoints=tuple(checkpoints),
            rejected=tuple(self._rejected),
            accounting=accounting,
            retained_step_indices=indices,
            accepted_dt_s=tuple(self._dt_schedule),
            control_schedule=tuple(self._control_schedule),
        )

    def _should_retain(self, *, index: int, time_s: float) -> bool:
        mode = self.config.mode
        if index == 0:
            return mode != "final"
        if mode == "all":
            return True
        if mode == "report":
            return any(
                math.isclose(time_s, report, rel_tol=1e-12, abs_tol=1e-12)
                for report in self.config.report_times_s
            )
        if mode == "checkpoint":
            return index % self.config.checkpoint_interval == 0
        if mode == "recompute":
            interval = max(
                1,
                math.ceil(self.accepted_step_bound / self.config.recompute_segments),
            )
            return index % interval == 0
        return False

    def _add_work(
        self,
        *,
        residual_evaluations: int,
        jacobian_assemblies: int,
        linear_solves: int,
    ) -> None:
        self._residual_evaluations += self._nonnegative_int(
            residual_evaluations, field="residual_evaluations"
        )
        self._jacobian_assemblies += self._nonnegative_int(
            jacobian_assemblies, field="jacobian_assemblies"
        )
        self._linear_solves += self._nonnegative_int(linear_solves, field="linear_solves")

    @staticmethod
    def _nonnegative_int(value: int, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FlowContractError(
                f"{field} must be a non-negative integer",
                object_name="FlowHistoryWriter",
                field=field,
                expected="integer >= 0",
                actual=value,
            )
        return value

    @staticmethod
    def _time(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FlowContractError(
                "time_s must be finite and non-negative",
                object_name="FlowHistoryWriter",
                field="time_s",
                expected="finite seconds >= 0",
                actual=value,
            )
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise FlowContractError(
                "time_s must be finite and non-negative",
                object_name="FlowHistoryWriter",
                field="time_s",
                expected="finite seconds >= 0",
                actual=value,
            )
        return result

    @staticmethod
    def _positive_float(value: float, *, field: str) -> float:
        result = FlowHistoryWriter._time(value)
        if result <= 0.0:
            raise FlowContractError(
                f"{field} must be positive",
                object_name="FlowHistoryWriter",
                field=field,
                expected="finite seconds > 0",
                actual=value,
            )
        return result


__all__ = [
    "FlowCheckpoint",
    "FlowExecutionAccounting",
    "FlowHistory",
    "FlowHistoryWriter",
]
