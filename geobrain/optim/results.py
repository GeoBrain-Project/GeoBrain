"""
Owned detached result snapshots for deterministic optimization and inversion.

The records are frozen against attribute replacement and their mappings are
read-only, but tensor storage is still mutable by whoever holds the snapshot.
Every tensor is detached and cloned, so such mutation cannot reach the caller's
original tensors or live optimizer state.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Sequence, TypeVar

import torch

from geobrain.core import GeoBrainError, ModelState
from geobrain.core.validation import (
    validate_mapping_key,
)

from ._validation import _coerce_integral_scalar, _coerce_real_scalar
from .execution import StopReason

__all__ = ["InversionResult", "OptimizationResult"]

_RESULT_SCHEMA_VERSION = 2
_ResultT = TypeVar("_ResultT", bound=object)


def _raise_unmaterialized_history(
    history: torch.Tensor,
    *,
    owner: str,
    field_name: str,
    cause: Exception | None = None,
) -> NoReturn:
    error = GeoBrainError(
        f"{owner}.{field_name} must be materialized",
        object_name=owner,
        field=field_name,
        expected="materialized one-dimensional real numeric tensor",
        actual={"type": type(history), "device": history.device},
    )
    if cause is None:
        raise error
    raise error from cause


def _snapshot_params(
    params: object,
    *,
    owner: str,
    field_name: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(params, MappingABC):
        raise GeoBrainError(
            f"{owner}.{field_name} must be a parameter mapping",
            object_name=owner,
            field=field_name,
            expected="non-empty mapping of string keys to torch.Tensor",
            actual=type(params),
        )
    if not params:
        raise GeoBrainError(
            f"{owner}.{field_name} must not be empty",
            object_name=owner,
            field=field_name,
            expected="non-empty mapping of string keys to torch.Tensor",
            actual={},
        )
    for name, tensor in params.items():
        validate_mapping_key(owner, field_name, name)
        if not isinstance(tensor, torch.Tensor):
            raise GeoBrainError(
                f"{owner}.{field_name} values must be torch.Tensor",
                object_name=owner,
                field=f"{field_name}[{name!r}]",
                expected=torch.Tensor,
                actual=type(tensor),
            )
    return MappingProxyType(
        {name: tensor.detach().clone() for name, tensor in params.items()}
    )


def _snapshot_matching_params(
    params: object,
    *,
    reference: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Own a best-parameter snapshot matching the final parameter schema."""
    snapshot = _snapshot_params(
        params,
        owner="InversionResult",
        field_name="best_params",
    )
    if set(snapshot) != set(reference):
        raise GeoBrainError(
            "InversionResult.best_params keys must match params exactly",
            object_name="InversionResult",
            field="best_params",
            expected=sorted(reference),
            actual=sorted(snapshot),
        )
    for name, tensor in snapshot.items():
        final_tensor = reference[name]
        if (
            tensor.shape != final_tensor.shape
            or tensor.dtype != final_tensor.dtype
            or tensor.device != final_tensor.device
        ):
            raise GeoBrainError(
                "InversionResult.best_params tensor metadata must match params",
                object_name="InversionResult",
                field=f"best_params[{name!r}]",
                expected={
                    "shape": tuple(final_tensor.shape),
                    "dtype": final_tensor.dtype,
                    "device": final_tensor.device,
                },
                actual={
                    "shape": tuple(tensor.shape),
                    "dtype": tensor.dtype,
                    "device": tensor.device,
                },
            )
    return snapshot


def _freeze_metadata_value(
    value: Any,
    *,
    owner: str,
    field_name: str,
    active: set[int],
) -> Any:
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
                f"{owner}.{field_name} must not contain recursive containers",
                object_name=owner,
                field=field_name,
                expected="acyclic metadata",
                actual="recursive mapping",
            )
        active.add(identity)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                validate_mapping_key(owner, field_name, key)
                frozen[key] = _freeze_metadata_value(
                    item,
                    owner=owner,
                    field_name=f"{field_name}[{key!r}]",
                    active=active,
                )
            return MappingProxyType(frozen)
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise GeoBrainError(
                f"{owner}.{field_name} must not contain recursive containers",
                object_name=owner,
                field=field_name,
                expected="acyclic metadata",
                actual="recursive sequence",
            )
        active.add(identity)
        try:
            return tuple(
                _freeze_metadata_value(
                    item,
                    owner=owner,
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
                f"{owner}.{field_name} must not contain recursive containers",
                object_name=owner,
                field=field_name,
                expected="acyclic metadata",
                actual="recursive set",
            )
        active.add(identity)
        try:
            return frozenset(
                _freeze_metadata_value(
                    item,
                    owner=owner,
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
            f"{owner}.{field_name} could not be defensively copied",
            object_name=owner,
            field=field_name,
            expected="copyable metadata value",
            actual=type(value),
        ) from exc
    if copied is value:
        raise GeoBrainError(
            f"{owner}.{field_name} could not be copied into owned metadata",
            object_name=owner,
            field=field_name,
            expected="owned metadata copy with distinct identity",
            actual=type(value),
        )
    return copied


def _freeze_metadata(
    metadata: Mapping[str, Any],
    *,
    owner: str,
) -> Mapping[str, Any]:
    """Deep-freeze built-in containers and detach tensors in metadata.

    Mappings become :class:`MappingProxyType`, sequences become tuples, sets
    become frozensets, and tensors are detached clones. Other leaf objects are
    deep-copied to break caller ownership, but their own classes may still be
    mutable; the result contract does not claim otherwise.
    """
    if not isinstance(metadata, Mapping):
        raise GeoBrainError(
            f"{owner}.metadata must be a mapping",
            object_name=owner,
            field="metadata",
            expected="mapping with non-empty string keys",
            actual=type(metadata),
        )
    frozen = _freeze_metadata_value(
        metadata,
        owner=owner,
        field_name="metadata",
        active=set(),
    )
    if not isinstance(frozen, MappingABC):  # defensive for exotic Mapping classes
        raise GeoBrainError(
            f"{owner}.metadata snapshot must remain a mapping",
            object_name=owner,
            field="metadata",
            expected="owned mapping snapshot",
            actual=type(frozen),
        )
    return frozen


def _snapshot_history(
    history: torch.Tensor,
    *,
    owner: str,
    field_name: str,
) -> torch.Tensor:
    if not isinstance(history, torch.Tensor):
        raise GeoBrainError(
            f"{owner}.{field_name} must be a torch.Tensor",
            object_name=owner,
            field=field_name,
            expected=torch.Tensor,
            actual=type(history),
        )
    if history.ndim != 1:
        raise GeoBrainError(
            f"{owner}.{field_name} must be one-dimensional",
            object_name=owner,
            field=field_name,
            expected="shape (completed_iters,)",
            actual=tuple(history.shape),
        )
    if history.dtype == torch.bool or history.is_complex():
        raise GeoBrainError(
            f"{owner}.{field_name} must contain real numeric losses",
            object_name=owner,
            field=field_name,
            expected="one-dimensional real numeric tensor",
            actual=history.dtype,
        )
    if history.device.type == "meta":
        _raise_unmaterialized_history(
            history,
            owner=owner,
            field_name=field_name,
        )
    try:
        return history.detach().clone()
    except (RuntimeError, ValueError) as exc:
        _raise_unmaterialized_history(
            history,
            owner=owner,
            field_name=field_name,
            cause=exc,
        )


def _validate_recorded_history(
    history: torch.Tensor,
    *,
    owner: str,
    field_name: str,
    completed_iters: int,
    stop_reason: StopReason,
    allow_unrecorded: bool,
) -> torch.Tensor:
    """Validate one history under the shared ordinary/non-finite stop policy."""
    snapshot = _snapshot_history(
        history,
        owner=owner,
        field_name=field_name,
    )
    if allow_unrecorded and snapshot.numel() == 0:
        return snapshot
    if snapshot.numel() != completed_iters:
        expected = (
            f"empty (not recorded) or length {completed_iters}"
            if allow_unrecorded
            else completed_iters
        )
        raise GeoBrainError(
            f"{owner}.{field_name} length must match completed_iters",
            object_name=owner,
            field=field_name,
            expected=expected,
            actual=snapshot.numel(),
        )
    finite = torch.isfinite(snapshot)
    if stop_reason is StopReason.NONFINITE:
        if not bool(finite[:-1].all()):
            raise GeoBrainError(
                f"{owner}.{field_name} may be non-finite only at termination",
                object_name=owner,
                field=field_name,
                expected="finite samples except possibly the terminal sample",
                actual=snapshot,
            )
    elif not bool(finite.all()):
        raise GeoBrainError(
            f"{owner}.{field_name} must be finite for {stop_reason.name}",
            object_name=owner,
            field=field_name,
            expected=f"all finite samples when stop_reason is {stop_reason.name}",
            actual=snapshot,
        )
    return snapshot


def _normalize_real_loss(
    value: object,
    *,
    owner: str,
    field_name: str,
) -> float:
    return float(
        _coerce_real_scalar(
            value,
            owner=owner,
            field=field_name,
            finite=False,
        )
    )


def _same_loss(left: float, right: float) -> bool:
    return (math.isnan(left) and math.isnan(right)) or left == right


def _snapshot_term_losses(
    term_losses: object,
    *,
    completed_iters: int,
    stop_reason: StopReason,
) -> Mapping[str, Sequence[float]] | None:
    if term_losses is None:
        return None
    if not isinstance(term_losses, MappingABC):
        raise GeoBrainError(
            "InversionResult.term_losses must be a mapping",
            object_name="InversionResult",
            field="term_losses",
            expected="mapping of term names to numeric loss histories",
            actual=type(term_losses),
        )
    frozen: dict[str, tuple[float, ...]] = {}
    for name, values in term_losses.items():
        validate_mapping_key("InversionResult", "term_losses", name)
        field_name = f"term_losses[{name!r}]"
        if not isinstance(values, SequenceABC) or isinstance(
            values, (str, bytes, bytearray)
        ):
            raise GeoBrainError(
                f"InversionResult.{field_name} must be a numeric sequence",
                object_name="InversionResult",
                field=field_name,
                expected=f"sequence of {completed_iters} real losses",
                actual=type(values),
            )
        normalized = tuple(
            _normalize_real_loss(
                item,
                owner="InversionResult",
                field_name=f"{field_name}[{index}]",
            )
            for index, item in enumerate(values)
        )
        if len(normalized) != completed_iters:
            raise GeoBrainError(
                f"InversionResult.{field_name} length must match completed_iters",
                object_name="InversionResult",
                field=field_name,
                expected=completed_iters,
                actual=len(normalized),
            )
        nonfinite_indices = [
            index for index, value in enumerate(normalized) if not math.isfinite(value)
        ]
        allowed = (
            stop_reason is StopReason.NONFINITE
            and all(index == completed_iters - 1 for index in nonfinite_indices)
        )
        if nonfinite_indices and not allowed:
            raise GeoBrainError(
                f"InversionResult.{field_name} has a non-terminal non-finite loss",
                object_name="InversionResult",
                field=field_name,
                expected=(
                    "finite samples except possibly the terminal sample "
                    "for NONFINITE"
                ),
                actual=normalized,
            )
        frozen[name] = normalized
    return MappingProxyType(frozen)


def _validate_run_semantics(
    *,
    owner: str,
    requested_iters: int,
    completed_iters: int,
    stop_reason: StopReason,
    loss_history: torch.Tensor,
) -> tuple[int, int, StopReason, torch.Tensor]:
    requested = _coerce_integral_scalar(
        requested_iters,
        owner=owner,
        field="requested_iters",
        minimum=0,
    )
    completed = _coerce_integral_scalar(
        completed_iters,
        owner=owner,
        field="completed_iters",
        minimum=0,
    )
    if completed > requested:
        raise GeoBrainError(
            f"{owner}.completed_iters must not exceed requested_iters",
            object_name=owner,
            field="completed_iters",
            expected=f"<= {requested}",
            actual=completed,
        )
    if not isinstance(stop_reason, StopReason):
        raise GeoBrainError(
            f"{owner}.stop_reason must be a StopReason",
            object_name=owner,
            field="stop_reason",
            expected=StopReason,
            actual=stop_reason,
        )
    if stop_reason is StopReason.COMPLETED and completed != requested:
        raise GeoBrainError(
            f"{owner} cannot report COMPLETED before all requested iterations",
            object_name=owner,
            field="stop_reason",
            expected="COMPLETED only when completed_iters == requested_iters",
            actual={
                "requested_iters": requested,
                "completed_iters": completed,
            },
        )
    history = _snapshot_history(
        loss_history,
        owner=owner,
        field_name="loss_history",
    )
    if history.numel() != completed:
        raise GeoBrainError(
            f"{owner}.loss_history length must equal completed_iters",
            object_name=owner,
            field="loss_history",
            expected=completed,
            actual=history.numel(),
        )
    if stop_reason in (StopReason.CALLBACK, StopReason.NONFINITE) and completed == 0:
        raise GeoBrainError(
            f"{owner}.{stop_reason.value} requires a completed iteration",
            object_name=owner,
            field="completed_iters",
            expected=f">= 1 when stop_reason is {stop_reason.name}",
            actual=completed,
        )
    finite = torch.isfinite(history)
    if stop_reason is StopReason.NONFINITE:
        prior_finite = bool(finite[:-1].all())
        terminal_nonfinite = not bool(finite[-1])
        if not prior_finite or not terminal_nonfinite:
            raise GeoBrainError(
                f"{owner}.loss_history must end at its only non-finite sample",
                object_name=owner,
                field="loss_history",
                expected=(
                    "finite samples before one terminal non-finite sample "
                    "when stop_reason is NONFINITE"
                ),
                actual=history,
            )
    elif not bool(finite.all()):
        raise GeoBrainError(
            f"{owner}.loss_history must be finite for {stop_reason.name}",
            object_name=owner,
            field="loss_history",
            expected=f"all finite samples when stop_reason is {stop_reason.name}",
            actual=history,
        )
    return requested, completed, stop_reason, history


def _plain_snapshot(value: Any) -> Any:
    """Convert owned snapshot containers into pickle-compatible plain values."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, MappingABC):
        return {key: _plain_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain_snapshot(item) for item in value)
    if isinstance(value, frozenset):
        return {_plain_snapshot(item) for item in value}
    return copy.deepcopy(value)


def _state_field(
    state: Mapping[str, Any],
    name: str,
    *,
    owner: str,
) -> Any:
    if name not in state:
        raise GeoBrainError(
            f"{owner} serialized state is missing {name!r}",
            object_name=owner,
            field="state",
            expected=f"serialized field {name!r}",
            actual=sorted(str(key) for key in state),
        )
    return state[name]


def _restore_result(
    result_type: type[_ResultT],
    payload: Mapping[str, Any],
    *,
    owner: str,
) -> _ResultT:
    """Re-enter constructor validation and structure incidental coercion errors."""
    try:
        return result_type(**payload)
    except GeoBrainError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise GeoBrainError(
            f"{owner} serialized state could not be validated",
            object_name=owner,
            field="state",
            expected="valid serialized constructor fields",
            actual=type(exc),
        ) from exc


def _legacy_stop_reason(
    history: torch.Tensor,
    *,
    callback_stopped: bool = False,
) -> StopReason:
    if callback_stopped:
        return StopReason.CALLBACK
    if history.numel() > 0:
        finite = torch.isfinite(history)
        if bool(finite[:-1].all()) and not bool(finite[-1]):
            return StopReason.NONFINITE
    return StopReason.COMPLETED


@dataclass(frozen=True)
class _PartialResult:
    """Typed exception-progress snapshot; deliberately has no stop reason."""

    params: Mapping[str, torch.Tensor]
    requested_iters: int
    completed_iters: int
    loss_history: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        requested = _coerce_integral_scalar(
            self.requested_iters,
            owner="_PartialResult",
            field="requested_iters",
            minimum=0,
        )
        completed = _coerce_integral_scalar(
            self.completed_iters,
            owner="_PartialResult",
            field="completed_iters",
            minimum=0,
        )
        if completed > requested:
            raise GeoBrainError(
                "_PartialResult.completed_iters must not exceed requested_iters",
                object_name="_PartialResult",
                field="completed_iters",
                expected=f"<= {requested}",
                actual=completed,
            )
        history = _snapshot_history(
            self.loss_history,
            owner="_PartialResult",
            field_name="loss_history",
        )
        if history.numel() != completed:
            raise GeoBrainError(
                "_PartialResult.loss_history length must equal completed_iters",
                object_name="_PartialResult",
                field="loss_history",
                expected=completed,
                actual=history.numel(),
            )
        if not bool(torch.isfinite(history).all()):
            raise GeoBrainError(
                "_PartialResult.loss_history must contain only completed finite samples",
                object_name="_PartialResult",
                field="loss_history",
                expected="all finite completed samples",
                actual=history,
            )
        object.__setattr__(
            self,
            "params",
            _snapshot_params(
                self.params,
                owner="_PartialResult",
                field_name="params",
            ),
        )
        object.__setattr__(self, "requested_iters", requested)
        object.__setattr__(self, "completed_iters", completed)
        object.__setattr__(self, "loss_history", history)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, owner="_PartialResult"),
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize through the validating factory without mapping proxies."""
        state = {
            "params": _plain_snapshot(self.params),
            "requested_iters": self.requested_iters,
            "completed_iters": self.completed_iters,
            "loss_history": self.loss_history.detach().clone(),
            "metadata": _plain_snapshot(self.metadata),
        }
        return _restore_partial_result, (state,)


def _make_partial_result(
    *,
    params: Mapping[str, torch.Tensor],
    requested_iters: int,
    completed_iters: int,
    loss_history: torch.Tensor,
    metadata: Mapping[str, Any] | None = None,
) -> _PartialResult:
    """Build an owned exception-progress snapshot for structured error wrapping."""
    return _PartialResult(
        params=params,
        requested_iters=requested_iters,
        completed_iters=completed_iters,
        loss_history=loss_history,
        metadata={} if metadata is None else metadata,
    )


def _restore_partial_result(state: Mapping[str, Any]) -> _PartialResult:
    """Rebuild a pickled partial snapshot through its public invariants."""
    return _make_partial_result(
        params=state["params"],
        requested_iters=state["requested_iters"],
        completed_iters=state["completed_iters"],
        loss_history=state["loss_history"],
        metadata=state["metadata"],
    )


@dataclass(frozen=True)
class OptimizationResult:
    """Owned snapshot returned by a bare deterministic optimizer.

    ``params`` remains a :class:`~geobrain.core.ModelState` so its state
    metadata has one canonical home. The state's tensor mapping, this result's
    metadata, and all tensor values are defensive snapshots. Read-only mappings
    prevent structural mutation, but a holder can still modify snapshot tensor
    storage.

    ``loss_history`` has exactly ``completed_iters`` real samples. Ordinary
    ``COMPLETED``, ``CALLBACK``, and ``CANCELLED`` results contain only finite
    samples. ``NONFINITE`` contains one terminal non-finite sample and only
    finite preceding samples. ``final_loss`` equals the final history sample
    exactly; a zero-iteration result instead uses ``NaN`` as the explicit
    no-loss sentinel.

    ``requested_iters`` and ``completed_iters`` are mandatory non-negative
    public constructor arguments, with completed not exceeding requested.
    ``stop_reason`` is also mandatory and ``COMPLETED`` requires equality.

    Pickle state is versioned and rebuilt through these validations. A
    pre-Task-5 state without run accounting migrates with
    ``requested_iters = completed_iters = len(loss_history)`` and records that
    rule in metadata.

    Attributes:
        params: owned final parameters.
        final_loss: objective at the final accepted iterate.
        requested_iters: iterations asked for.
        completed_iters: iterations actually run.
        stop_reason: why the loop ended.
        loss_history: per-iteration objective values.
        metadata: run extras (timings, config echo).
    """

    params: ModelState
    final_loss: float
    requested_iters: int
    completed_iters: int
    stop_reason: StopReason
    loss_history: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate iteration semantics and own all mutable input state."""
        if not isinstance(self.params, ModelState):
            raise GeoBrainError(
                "OptimizationResult.params must be a ModelState",
                object_name="OptimizationResult",
                field="params",
                expected=ModelState,
                actual=type(self.params),
            )
        params = ModelState(
            tensors=_snapshot_params(
                self.params.tensors,
                owner="OptimizationResult",
                field_name="params",
            ),
            metadata=_freeze_metadata(
                self.params.metadata,
                owner="OptimizationResult.params",
            ),
        )
        history = _snapshot_history(
            self.loss_history,
            owner="OptimizationResult",
            field_name="loss_history",
        )
        requested, completed, reason, history = _validate_run_semantics(
            owner="OptimizationResult",
            requested_iters=self.requested_iters,
            completed_iters=self.completed_iters,
            stop_reason=self.stop_reason,
            loss_history=history,
        )
        final_loss = _normalize_real_loss(
            self.final_loss,
            owner="OptimizationResult",
            field_name="final_loss",
        )
        if completed == 0:
            final_matches = math.isnan(final_loss)
            expected_final: object = "NaN when no iterations completed"
        else:
            terminal_loss = float(history[-1])
            final_matches = _same_loss(final_loss, terminal_loss)
            expected_final = terminal_loss
        if not final_matches:
            raise GeoBrainError(
                "OptimizationResult.final_loss must match the terminal history sample",
                object_name="OptimizationResult",
                field="final_loss",
                expected=expected_final,
                actual=self.final_loss,
            )
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "final_loss", final_loss)
        object.__setattr__(self, "loss_history", history)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, owner="OptimizationResult"),
        )
        object.__setattr__(self, "requested_iters", requested)
        object.__setattr__(self, "completed_iters", completed)
        object.__setattr__(self, "stop_reason", reason)

    def __getstate__(self) -> dict[str, Any]:
        """Return a versioned plain state without unpicklable mapping proxies."""
        return {
            "_schema_version": _RESULT_SCHEMA_VERSION,
            "params": {
                "tensors": _plain_snapshot(self.params.tensors),
                "metadata": _plain_snapshot(self.params.metadata),
            },
            "final_loss": self.final_loss,
            "loss_history": self.loss_history.detach().clone(),
            "metadata": _plain_snapshot(self.metadata),
            "requested_iters": self.requested_iters,
            "completed_iters": self.completed_iters,
            "stop_reason": self.stop_reason,
        }

    def __setstate__(self, state: object) -> None:
        """Restore new state or migrate the pre-Task-5 optimizer schema."""
        if not isinstance(state, MappingABC):
            raise GeoBrainError(
                "OptimizationResult serialized state must be a mapping",
                object_name="OptimizationResult",
                field="state",
                expected="mapping",
                actual=type(state),
            )
        schema_version = state.get("_schema_version")
        if schema_version == _RESULT_SCHEMA_VERSION:
            params_state = _state_field(
                state,
                "params",
                owner="OptimizationResult",
            )
            if not isinstance(params_state, MappingABC):
                raise GeoBrainError(
                    "OptimizationResult serialized params must be a mapping",
                    object_name="OptimizationResult",
                    field="state.params",
                    expected="mapping with tensors and metadata",
                    actual=type(params_state),
                )
            tensors_state = _state_field(
                params_state,
                "tensors",
                owner="OptimizationResult",
            )
            if not isinstance(tensors_state, MappingABC):
                raise GeoBrainError(
                    "OptimizationResult serialized tensors must be a mapping",
                    object_name="OptimizationResult",
                    field="state.params.tensors",
                    expected="mapping of string keys to torch.Tensor",
                    actual=type(tensors_state),
                )
            params_metadata_state = _state_field(
                params_state,
                "metadata",
                owner="OptimizationResult",
            )
            if not isinstance(params_metadata_state, MappingABC):
                raise GeoBrainError(
                    "OptimizationResult serialized state metadata must be a mapping",
                    object_name="OptimizationResult",
                    field="state.params.metadata",
                    expected="mapping",
                    actual=type(params_metadata_state),
                )
            params = ModelState(
                tensors=tensors_state,
                metadata=params_metadata_state,
            )
            payload = {
                "params": params,
                "final_loss": _state_field(
                    state,
                    "final_loss",
                    owner="OptimizationResult",
                ),
                "loss_history": _state_field(
                    state,
                    "loss_history",
                    owner="OptimizationResult",
                ),
                "metadata": _state_field(
                    state,
                    "metadata",
                    owner="OptimizationResult",
                ),
                "requested_iters": _state_field(
                    state,
                    "requested_iters",
                    owner="OptimizationResult",
                ),
                "completed_iters": _state_field(
                    state,
                    "completed_iters",
                    owner="OptimizationResult",
                ),
                "stop_reason": _state_field(
                    state,
                    "stop_reason",
                    owner="OptimizationResult",
                ),
            }
        elif schema_version is None:
            history = _state_field(
                state,
                "loss_history",
                owner="OptimizationResult",
            )
            if not isinstance(history, torch.Tensor):
                raise GeoBrainError(
                    "OptimizationResult legacy history must be a tensor",
                    object_name="OptimizationResult",
                    field="state.loss_history",
                    expected=torch.Tensor,
                    actual=type(history),
                )
            completed = int(history.numel())
            reason = _legacy_stop_reason(history)
            metadata_raw = _state_field(
                state,
                "metadata",
                owner="OptimizationResult",
            )
            if not isinstance(metadata_raw, MappingABC):
                raise GeoBrainError(
                    "OptimizationResult legacy metadata must be a mapping",
                    object_name="OptimizationResult",
                    field="state.metadata",
                    expected="mapping",
                    actual=type(metadata_raw),
                )
            metadata = dict(metadata_raw)
            metadata["_geobrain_result_migration"] = {
                "source_schema": "legacy OptimizationResult",
                "requested_iters_rule": "requested_iters = completed_iters",
                "stop_reason_rule": (
                    "NONFINITE for one terminal non-finite sample; "
                    "otherwise COMPLETED"
                ),
            }
            payload = {
                "params": _state_field(
                    state,
                    "params",
                    owner="OptimizationResult",
                ),
                "final_loss": _state_field(
                    state,
                    "final_loss",
                    owner="OptimizationResult",
                ),
                "loss_history": history,
                "metadata": metadata,
                "requested_iters": completed,
                "completed_iters": completed,
                "stop_reason": reason,
            }
        else:
            raise GeoBrainError(
                "OptimizationResult serialized schema is unsupported",
                object_name="OptimizationResult",
                field="state._schema_version",
                expected=_RESULT_SCHEMA_VERSION,
                actual=schema_version,
            )
        restored = _restore_result(
            type(self),
            payload,
            owner="OptimizationResult",
        )
        for name in (
            "params",
            "final_loss",
            "loss_history",
            "metadata",
            "requested_iters",
            "completed_iters",
            "stop_reason",
        ):
            object.__setattr__(self, name, getattr(restored, name))


@dataclass(frozen=True)
class InversionResult:
    """Owned inversion snapshot with explicit run and history semantics.

    ``params`` is the final detached parameter snapshot. ``best_params`` is
    ``None`` when no finite best sample exists; otherwise it is the detached
    snapshot associated with ``best_iter`` and has exactly the same keys and
    tensor shapes, dtypes, and devices as ``params``. If omitted for a finite
    history, it can be inferred from final ``params`` only when the final sample
    is a minimum. The mapping structures are read-only, while tensor storage
    remains modifiable by the snapshot holder without aliasing caller or
    optimizer tensors.

    ``loss_history`` always has length ``completed_iters``. It is finite for
    ``COMPLETED``, ``CALLBACK``, and ``CANCELLED``. ``NONFINITE`` requires at
    least one completed iteration, finite preceding samples, and one terminal
    non-finite sample. ``COMPLETED`` requires
    ``completed_iters == requested_iters``; cancellation may occur before the
    first step and therefore may have an empty history.

    ``data_loss_history`` and ``reg_loss_history`` are either empty to mean
    "not recorded" or have exactly ``completed_iters`` samples under the same
    finiteness policy. ``term_losses`` is ``None`` when unavailable; otherwise
    each named numeric sequence has exactly ``completed_iters`` samples under
    that policy.

    When at least one finite primary loss exists, ``best_iter`` is a valid
    zero-based history index and ``best_loss`` is exactly the minimum finite
    sample at that index. With zero iterations or no finite sample,
    ``best_iter`` is ``None`` and ``best_loss`` is ``NaN``.

    ``converged`` is the historical callback-stop view and is true exactly for
    ``StopReason.CALLBACK``; it is not a numerical convergence test.
    ``n_iters`` is the read-only compatibility view of ``completed_iters``.

    Pickle state is versioned. A legacy state containing only ``n_iters``
    migrates with ``requested_iters = completed_iters = n_iters``; its honest
    callback/non-finite evidence determines the stop reason, otherwise the
    result is ``COMPLETED``. A legacy run without a finite sample normalizes
    ``best_params`` and ``best_iter`` to ``None``. Metadata records the precise
    migration rules.

    Attributes:
        params: final parameters (physical space).
        requested_iters / completed_iters: iteration accounting.
        stop_reason: why the run ended.
        loss_history / data_loss_history / reg_loss_history: per-iteration
            total / data-term / regulariser-term losses.
        best_params / best_loss / best_iter: best-seen iterate.
        wall_clock_sec: total run time.
        converged: tolerance-based convergence flag.
        term_losses: named per-term loss breakdown.
        metadata: run extras.
    """

    params: Mapping[str, torch.Tensor]
    requested_iters: int
    completed_iters: int
    stop_reason: StopReason
    loss_history: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    best_params: Mapping[str, torch.Tensor] | None = None
    data_loss_history: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    reg_loss_history: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    best_loss: float = float("nan")
    best_iter: int | None = None
    wall_clock_sec: float = 0.0
    converged: bool | None = None
    term_losses: Mapping[str, Sequence[float]] | None = None

    def __post_init__(self) -> None:
        """Validate run accounting and own all mutable input state."""
        requested, completed, reason, history = _validate_run_semantics(
            owner="InversionResult",
            requested_iters=self.requested_iters,
            completed_iters=self.completed_iters,
            stop_reason=self.stop_reason,
            loss_history=self.loss_history,
        )
        params = _snapshot_params(
            self.params,
            owner="InversionResult",
            field_name="params",
        )
        provided_best_params = (
            None
            if self.best_params is None
            else _snapshot_matching_params(
                self.best_params,
                reference=params,
            )
        )
        data_history = _validate_recorded_history(
            self.data_loss_history,
            owner="InversionResult",
            field_name="data_loss_history",
            completed_iters=completed,
            stop_reason=reason,
            allow_unrecorded=True,
        )
        reg_history = _validate_recorded_history(
            self.reg_loss_history,
            owner="InversionResult",
            field_name="reg_loss_history",
            completed_iters=completed,
            stop_reason=reason,
            allow_unrecorded=True,
        )
        wall_clock_sec = _coerce_real_scalar(
            self.wall_clock_sec,
            owner="InversionResult",
            field="wall_clock_sec",
            finite=True,
            minimum=0.0,
        )
        expected_converged = reason is StopReason.CALLBACK
        if self.converged is not None and self.converged is not expected_converged:
            raise GeoBrainError(
                "InversionResult.converged must agree with stop_reason",
                object_name="InversionResult",
                field="converged",
                expected=expected_converged,
                actual=self.converged,
            )
        term_losses = _snapshot_term_losses(
            self.term_losses,
            completed_iters=completed,
            stop_reason=reason,
        )
        best_loss = _normalize_real_loss(
            self.best_loss,
            owner="InversionResult",
            field_name="best_loss",
        )
        finite_indices = torch.nonzero(
            torch.isfinite(history),
            as_tuple=False,
        ).flatten()
        if finite_indices.numel() == 0:
            if not math.isnan(best_loss):
                raise GeoBrainError(
                    "InversionResult.best_loss must be NaN without a finite sample",
                    object_name="InversionResult",
                    field="best_loss",
                    expected="NaN when loss_history has no finite sample",
                    actual=best_loss,
                )
            if self.best_iter is not None:
                raise GeoBrainError(
                    "InversionResult.best_iter must be None without a finite sample",
                    object_name="InversionResult",
                    field="best_iter",
                    expected="None",
                    actual=self.best_iter,
                )
            best_iter: int | None = None
            if provided_best_params is not None:
                raise GeoBrainError(
                    "InversionResult.best_params requires a finite best sample",
                    object_name="InversionResult",
                    field="best_params",
                    expected="None when best_iter is None",
                    actual="parameter snapshot without a finite best sample",
                )
            best_params: Mapping[str, torch.Tensor] | None = None
        else:
            if self.best_iter is None and math.isnan(best_loss):
                minimum_loss = min(
                    float(history[index]) for index in finite_indices
                )
                final_index = completed - 1
                if (
                    provided_best_params is None
                    and math.isfinite(float(history[final_index]))
                    and float(history[final_index]) == minimum_loss
                ):
                    best_iter = final_index
                else:
                    finite_values = history[finite_indices]
                    relative_index = int(torch.argmin(finite_values))
                    best_iter = int(finite_indices[relative_index])
                best_loss = float(history[best_iter])
            else:
                if self.best_iter is None:
                    raise GeoBrainError(
                        "InversionResult.best_iter must identify best_loss",
                        object_name="InversionResult",
                        field="best_iter",
                        expected=f"integer in [0, {completed})",
                        actual=type(None),
                    )
                best_iter = _coerce_integral_scalar(
                    self.best_iter,
                    owner="InversionResult",
                    field="best_iter",
                    minimum=0,
                )
                if best_iter >= completed or not bool(torch.isfinite(history[best_iter])):
                    raise GeoBrainError(
                        "InversionResult.best_iter must identify a finite history sample",
                        object_name="InversionResult",
                        field="best_iter",
                        expected=f"finite loss_history index in [0, {completed})",
                        actual=best_iter,
                    )
                expected_best = float(history[best_iter])
                if not math.isfinite(best_loss) or best_loss != expected_best:
                    raise GeoBrainError(
                        "InversionResult.best_loss must equal loss_history[best_iter]",
                        object_name="InversionResult",
                        field="best_loss",
                        expected=expected_best,
                        actual=best_loss,
                    )
                minimum_loss = min(float(history[index]) for index in finite_indices)
                if best_loss != minimum_loss:
                    raise GeoBrainError(
                        "InversionResult.best_loss must be the lowest finite loss",
                        object_name="InversionResult",
                        field="best_loss",
                        expected=minimum_loss,
                        actual=best_loss,
                    )
            if provided_best_params is None:
                if best_iter != completed - 1:
                    raise GeoBrainError(
                        "InversionResult.best_params is required for a non-final best",
                        object_name="InversionResult",
                        field="best_params",
                        expected=(
                            "owned snapshot for best_iter, or omitted only "
                            "when best_iter is the final history index"
                        ),
                        actual={"best_iter": best_iter, "final_iter": completed - 1},
                    )
                best_params = _snapshot_matching_params(
                    params,
                    reference=params,
                )
            else:
                best_params = provided_best_params
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "best_params", best_params)
        object.__setattr__(self, "requested_iters", requested)
        object.__setattr__(self, "completed_iters", completed)
        object.__setattr__(self, "stop_reason", reason)
        object.__setattr__(self, "loss_history", history)
        object.__setattr__(self, "data_loss_history", data_history)
        object.__setattr__(self, "reg_loss_history", reg_history)
        object.__setattr__(self, "best_loss", best_loss)
        object.__setattr__(self, "best_iter", best_iter)
        object.__setattr__(self, "wall_clock_sec", wall_clock_sec)
        object.__setattr__(self, "converged", expected_converged)
        object.__setattr__(self, "term_losses", term_losses)
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata(self.metadata, owner="InversionResult"),
        )

    def __getstate__(self) -> dict[str, Any]:
        """Return versioned pickle state containing only plain owned values."""
        return {
            "_schema_version": _RESULT_SCHEMA_VERSION,
            "params": _plain_snapshot(self.params),
            "requested_iters": self.requested_iters,
            "completed_iters": self.completed_iters,
            "stop_reason": self.stop_reason,
            "loss_history": self.loss_history.detach().clone(),
            "metadata": _plain_snapshot(self.metadata),
            "best_params": _plain_snapshot(self.best_params),
            "data_loss_history": self.data_loss_history.detach().clone(),
            "reg_loss_history": self.reg_loss_history.detach().clone(),
            "best_loss": self.best_loss,
            "best_iter": self.best_iter,
            "wall_clock_sec": self.wall_clock_sec,
            "converged": self.converged,
            "term_losses": _plain_snapshot(self.term_losses),
        }

    def __setstate__(self, state: object) -> None:
        """Restore new state or explicitly migrate the legacy ``n_iters`` state.

        Legacy snapshots cannot reveal the originally requested iteration count.
        Migration therefore sets ``requested_iters = completed_iters = n_iters``.
        Finite ordinary histories become ``COMPLETED``; an explicit legacy
        callback flag becomes ``CALLBACK``; and a single terminal non-finite
        sample becomes ``NONFINITE``. The applied rule is recorded in metadata.
        """
        if not isinstance(state, MappingABC):
            raise GeoBrainError(
                "InversionResult serialized state must be a mapping",
                object_name="InversionResult",
                field="state",
                expected="mapping",
                actual=type(state),
            )
        schema_version = state.get("_schema_version")
        if schema_version == _RESULT_SCHEMA_VERSION:
            payload = {
                name: _state_field(state, name, owner="InversionResult")
                for name in (
                    "params",
                    "requested_iters",
                    "completed_iters",
                    "stop_reason",
                    "loss_history",
                    "metadata",
                    "best_params",
                    "data_loss_history",
                    "reg_loss_history",
                    "best_loss",
                    "best_iter",
                    "wall_clock_sec",
                    "converged",
                    "term_losses",
                )
            }
        elif schema_version is None:
            history = _state_field(
                state,
                "loss_history",
                owner="InversionResult",
            )
            if not isinstance(history, torch.Tensor):
                raise GeoBrainError(
                    "InversionResult legacy history must be a tensor",
                    object_name="InversionResult",
                    field="state.loss_history",
                    expected=torch.Tensor,
                    actual=type(history),
                )
            n_iters = _coerce_integral_scalar(
                _state_field(state, "n_iters", owner="InversionResult"),
                owner="InversionResult",
                field="state.n_iters",
                minimum=0,
            )
            callback_stopped = _state_field(
                state,
                "converged",
                owner="InversionResult",
            )
            if not isinstance(callback_stopped, bool):
                raise GeoBrainError(
                    "InversionResult legacy converged flag must be bool",
                    object_name="InversionResult",
                    field="state.converged",
                    expected=bool,
                    actual=type(callback_stopped),
                )
            reason = _legacy_stop_reason(
                history,
                callback_stopped=callback_stopped,
            )
            has_finite_sample = bool(torch.isfinite(history).any())
            metadata_raw = _state_field(
                state,
                "metadata",
                owner="InversionResult",
            )
            if not isinstance(metadata_raw, MappingABC):
                raise GeoBrainError(
                    "InversionResult legacy metadata must be a mapping",
                    object_name="InversionResult",
                    field="state.metadata",
                    expected="mapping",
                    actual=type(metadata_raw),
                )
            metadata = dict(metadata_raw)
            if reason is StopReason.CALLBACK:
                stop_reason_rule = "CALLBACK from legacy converged=True"
            elif reason is StopReason.NONFINITE:
                stop_reason_rule = "NONFINITE from one terminal non-finite sample"
            else:
                stop_reason_rule = "COMPLETED for finite ordinary history"
            metadata["_geobrain_result_migration"] = {
                "source_schema": "legacy InversionResult with n_iters",
                "requested_iters_rule": (
                    "requested_iters = completed_iters = n_iters"
                ),
                "stop_reason_rule": stop_reason_rule,
                "legacy_converged": callback_stopped,
                "best_params_rule": (
                    "preserved for a finite best sample; "
                    "otherwise best_params = best_iter = None"
                ),
            }
            payload = {
                "params": _state_field(
                    state,
                    "params",
                    owner="InversionResult",
                ),
                "requested_iters": n_iters,
                "completed_iters": n_iters,
                "stop_reason": reason,
                "loss_history": history,
                "metadata": metadata,
                "best_params": _state_field(
                    state,
                    "best_params",
                    owner="InversionResult",
                ) if has_finite_sample else None,
                "data_loss_history": _state_field(
                    state,
                    "data_loss_history",
                    owner="InversionResult",
                ),
                "reg_loss_history": _state_field(
                    state,
                    "reg_loss_history",
                    owner="InversionResult",
                ),
                "best_loss": _state_field(
                    state,
                    "best_loss",
                    owner="InversionResult",
                ) if has_finite_sample else float("nan"),
                "best_iter": _state_field(
                    state,
                    "best_iter",
                    owner="InversionResult",
                ) if has_finite_sample else None,
                "wall_clock_sec": _state_field(
                    state,
                    "wall_clock_sec",
                    owner="InversionResult",
                ),
                "converged": reason is StopReason.CALLBACK,
                "term_losses": _state_field(
                    state,
                    "term_losses",
                    owner="InversionResult",
                ),
            }
        else:
            raise GeoBrainError(
                "InversionResult serialized schema is unsupported",
                object_name="InversionResult",
                field="state._schema_version",
                expected=_RESULT_SCHEMA_VERSION,
                actual=schema_version,
            )
        restored = _restore_result(
            type(self),
            payload,
            owner="InversionResult",
        )
        for name in (
            "params",
            "requested_iters",
            "completed_iters",
            "stop_reason",
            "loss_history",
            "metadata",
            "best_params",
            "data_loss_history",
            "reg_loss_history",
            "best_loss",
            "best_iter",
            "wall_clock_sec",
            "converged",
            "term_losses",
        ):
            object.__setattr__(self, name, getattr(restored, name))

    @property
    def n_iters(self) -> int:
        """Read-only compatibility view of :attr:`completed_iters`.

        Maintained experiment, validation, and example clients still consume
        this pre-release spelling. It stores no duplicate state and cannot be
        assigned on the frozen result. It is the number of recorded completed
        iterations, can be zero, and never means the original requested count.
        """
        return self.completed_iters
