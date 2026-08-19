"""Owned immutable result snapshots returned by Bayesian inference.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ..core.validation import validate_int
from ..core.errors import GeoBrainError
from .execution import SamplerStopReason


def _freeze_owned(
    value: Any,
    *,
    owner: str,
    field_name: str,
    active: set[int],
) -> Any:
    """Recursively own supported payloads or reject an unsafe leaf."""
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
                expected="acyclic owned payload",
                actual="recursive mapping",
            )
        active.add(identity)
        try:
            return MappingProxyType(
                {
                    _freeze_owned(
                        key,
                        owner=owner,
                        field_name=f"{field_name}.key",
                        active=active,
                    ): _freeze_owned(
                        item,
                        owner=owner,
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
                f"{owner}.{field_name} must not contain recursive containers",
                object_name=owner,
                field=field_name,
                expected="acyclic owned payload",
                actual="recursive sequence",
            )
        active.add(identity)
        try:
            return tuple(
                _freeze_owned(
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
                expected="acyclic owned payload",
                actual="recursive set",
            )
        active.add(identity)
        try:
            return frozenset(
                _freeze_owned(
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
            expected="copyable owned payload leaf",
            actual=type(value),
        ) from exc
    if copied is value:
        raise GeoBrainError(
            f"{owner}.{field_name} could not be copied into owned payload",
            object_name=owner,
            field=field_name,
            expected="owned payload copy with distinct identity",
            actual=type(value),
        )
    return copied


def _thaw_owned(value: Any) -> Any:
    """Convert frozen built-in containers to pickle-compatible owned values."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {_thaw_owned(key): _thaw_owned(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_owned(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_thaw_owned(item) for item in value)
    return copy.deepcopy(value)


def _restore_inference_result(state: Mapping[str, Any]) -> "InferenceResult":
    """Rebuild a pickled result through its public ownership invariants."""
    return InferenceResult(**state)


def _restore_partial_inference_result(
    state: Mapping[str, Any],
) -> "_PartialInferenceResult":
    """Rebuild a pickled private partial through its ownership invariants."""
    return _PartialInferenceResult(**state)


def _invalid(
    owner: str,
    message: str,
    field_name: str,
    expected: object,
    actual: object,
) -> None:
    raise GeoBrainError(
        message,
        object_name=owner,
        field=field_name,
        expected=expected,
        actual=actual,
    )


def _own_result_payload(
    *,
    owner: str,
    samples: Mapping[str, torch.Tensor],
    log_post_history: torch.Tensor,
    acceptance_rate: float,
    requested_iters: int,
    completed_iters: int,
    metadata: Mapping[str, Any],
) -> tuple[
    Mapping[str, torch.Tensor],
    torch.Tensor,
    float,
    int,
    int,
    Mapping[str, Any],
]:
    """Validate cross-field result semantics and return detached owned state."""
    requested = validate_int(
        requested_iters,
        owner=owner,
        field="requested_iters",
        minimum=0,
    )
    completed = validate_int(
        completed_iters,
        owner=owner,
        field="completed_iters",
        minimum=0,
    )
    if completed > requested:
        _invalid(
            owner,
            f"{owner}.completed_iters exceeds requested_iters",
            "completed_iters",
            f"<= {requested}",
            completed,
        )

    try:
        rate = float(acceptance_rate)
    except (TypeError, ValueError):
        _invalid(
            owner,
            f"{owner}.acceptance_rate must be a finite fraction",
            "acceptance_rate",
            "[0, 1]",
            acceptance_rate,
        )
        raise AssertionError("unreachable")
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        _invalid(
            owner,
            f"{owner}.acceptance_rate must be a finite fraction",
            "acceptance_rate",
            "[0, 1]",
            acceptance_rate,
        )
    if completed == 0 and rate != 0.0:
        _invalid(
            owner,
            f"{owner}.acceptance_rate is zero without sampling",
            "acceptance_rate",
            0.0,
            rate,
        )

    if not isinstance(metadata, Mapping):
        _invalid(
            owner,
            f"{owner}.metadata must be a mapping",
            "metadata",
            Mapping,
            type(metadata),
        )
    sample_layout = metadata.get("sample_layout")
    if sample_layout not in {"draws", "particles", "chains"}:
        _invalid(
            owner,
            f"{owner}.metadata must declare its sample layout",
            "sample_layout",
            "'draws', 'particles', or 'chains'",
            sample_layout,
        )
    stored_draws = validate_int(
        metadata.get("stored_draws"),
        owner=owner,
        field="stored_draws",
        minimum=0,
    )
    if sample_layout in {"draws", "chains"} and stored_draws > completed:
        _invalid(
            owner,
            f"{owner}.stored_draws exceeds completed transitions",
            "stored_draws",
            f"<= completed_iters ({completed})",
            stored_draws,
        )

    if not isinstance(samples, Mapping):
        _invalid(
            owner,
            f"{owner}.samples must be a mapping",
            "samples",
            "Mapping[str, torch.Tensor]",
            type(samples),
        )
    if not samples:
        _invalid(
            owner,
            f"{owner}.samples must contain at least one field",
            "samples",
            "non-empty Mapping[str, torch.Tensor]",
            {},
        )

    n_chains: int | None = None
    if sample_layout == "chains":
        n_chains = validate_int(
            metadata.get("n_chains"),
            owner=owner,
            field="n_chains",
            minimum=1,
        )

    owned_samples: dict[str, torch.Tensor] = {}
    for name, tensor in samples.items():
        if not isinstance(name, str) or not name:
            _invalid(
                owner,
                f"{owner}.samples keys must be non-empty strings",
                "samples",
                "non-empty string keys",
                name,
            )
        if not isinstance(tensor, torch.Tensor):
            _invalid(
                owner,
                f"{owner}.samples values must be tensors",
                f"samples[{name!r}]",
                torch.Tensor,
                type(tensor),
            )
        required_rank = 2 if sample_layout == "chains" else 1
        if tensor.ndim < required_rank:
            _invalid(
                owner,
                f"{owner}.samples tensor rank does not match sample layout",
                f"samples[{name!r}]",
                f">= {required_rank}-D for {sample_layout!r} layout",
                tuple(tensor.shape),
            )
        if sample_layout == "chains":
            assert n_chains is not None
            if tensor.shape[0] != n_chains or tensor.shape[1] != stored_draws:
                _invalid(
                    owner,
                    f"{owner}.samples chain/draw axes disagree with metadata",
                    f"samples[{name!r}]",
                    f"leading shape ({n_chains}, {stored_draws})",
                    tuple(tensor.shape[:2]),
                )
        elif tensor.shape[0] != stored_draws:
            _invalid(
                owner,
                f"{owner}.samples leading dimension disagrees with stored_draws",
                f"samples[{name!r}]",
                f"leading dimension {stored_draws}",
                tuple(tensor.shape),
            )
        owned_samples[name] = tensor.detach().clone()

    try:
        raw_history = torch.as_tensor(log_post_history)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GeoBrainError(
            f"{owner}.log_post_history must be tensor-like",
            object_name=owner,
            field="log_post_history",
            expected="real tensor-like history",
            actual=type(log_post_history),
        ) from exc
    if raw_history.is_complex():
        _invalid(
            owner,
            f"{owner}.log_post_history must be real",
            "log_post_history",
            "real tensor",
            raw_history.dtype,
        )
    history = raw_history.to(dtype=torch.float64).detach().clone()
    if not bool(torch.isfinite(history).all()):
        _invalid(
            owner,
            f"{owner}.log_post_history must contain only finite values",
            "log_post_history",
            "finite real values",
            history,
        )
    if sample_layout == "chains":
        assert n_chains is not None
        expected_history_shape = (n_chains, completed)
        if history.ndim != 2 or tuple(history.shape) != expected_history_shape:
            _invalid(
                owner,
                f"{owner}.log_post_history must be dense and chain-major",
                "log_post_history",
                expected_history_shape,
                tuple(history.shape),
            )
    elif history.ndim != 1 or history.numel() != completed:
        _invalid(
            owner,
            f"{owner}.log_post_history must be a dense 1-D transition history",
            "log_post_history",
            f"shape ({completed},)",
            tuple(history.shape),
        )

    return (
        MappingProxyType(owned_samples),
        history,
        rate,
        requested,
        completed,
        _freeze_owned(
            metadata,
            owner=owner,
            field_name="metadata",
            active=set(),
        ),
    )


@dataclass(frozen=True)
class _PartialInferenceResult:
    """Private owned snapshot attached only to an exceptional sampler exit.

    Unlike :class:`InferenceResult`, this record intentionally has no
    ``stop_reason``: exception phase and cause live on the structured
    :class:`GeoBrainError`, while this object reports only committed work.
    """

    samples: Mapping[str, torch.Tensor]
    log_post_history: torch.Tensor
    acceptance_rate: float
    requested_iters: int
    completed_iters: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate partial accounting and own all mutable inputs."""
        (
            samples,
            history,
            rate,
            requested,
            completed,
            metadata,
        ) = _own_result_payload(
            owner="_PartialInferenceResult",
            samples=self.samples,
            log_post_history=self.log_post_history,
            acceptance_rate=self.acceptance_rate,
            requested_iters=self.requested_iters,
            completed_iters=self.completed_iters,
            metadata=self.metadata,
        )
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "log_post_history", history)
        object.__setattr__(self, "acceptance_rate", rate)
        object.__setattr__(self, "requested_iters", requested)
        object.__setattr__(self, "completed_iters", completed)
        object.__setattr__(self, "metadata", metadata)

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Serialize only plain owned state and restore through validation."""
        state = {
            "samples": _thaw_owned(self.samples),
            "log_post_history": self.log_post_history.detach().clone(),
            "acceptance_rate": self.acceptance_rate,
            "requested_iters": self.requested_iters,
            "completed_iters": self.completed_iters,
            "metadata": _thaw_owned(self.metadata),
        }
        return _restore_partial_inference_result, (state,)


@dataclass(frozen=True)
class InferenceResult:
    """Owned snapshot of one successful or callback-terminated ``run`` call.

    ``metadata["sample_layout"]`` makes tensor axes explicit:

    - ``"draws"`` is one MCMC chain with ``n_draws == stored_draws``
      shaped ``(n_draws, *field_shape)``;
    - ``"particles"`` is the final SVGD ensemble shaped
      ``(n_particles, *field_shape)``; and
    - ``"chains"`` is the maintained :func:`run_chains` stack shaped
      ``(n_chains, stored_draws, *field_shape)``.

    A single sampler's ``log_post_history`` is always a finite dense 1-D
    history with one value per committed transition, independent of thinning
    or the final particle count. The explicit ``"chains"`` aggregate keeps
    the corresponding finite ``(n_chains, completed_iters)`` matrix.

    Exceptional exits never masquerade as callback returns. They attach a
    private :class:`_PartialInferenceResult` to ``GeoBrainError.partial_result``;
    that private record deliberately has no public ``stop_reason``.
    ``acceptance_rate`` is finite in ``[0, 1]`` and is exactly zero when no
    post-warmup transition completed.

    Attributes:
        samples: ``{param name: tensor}`` stored draws, laid out per
            ``metadata["sample_layout"]`` (see above).
        log_post_history: dense 1-D log-posterior per committed transition.
        acceptance_rate: fraction of accepted post-warmup transitions.
        requested_iters: sampling iterations requested for the run.
        completed_iters: sampling iterations actually committed.
        stop_reason: why the run ended (completed / callback / budget).
        metadata: layout tag plus sampler-specific extras.
    """

    samples: Mapping[str, torch.Tensor]
    log_post_history: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    acceptance_rate: float = 0.0
    requested_iters: int = 0
    completed_iters: int = 0
    stop_reason: SamplerStopReason = SamplerStopReason.COMPLETED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate returned-run semantics and own every mutable input."""
        (
            samples,
            history,
            rate,
            requested,
            completed,
            metadata,
        ) = _own_result_payload(
            owner="InferenceResult",
            samples=self.samples,
            log_post_history=self.log_post_history,
            acceptance_rate=self.acceptance_rate,
            requested_iters=self.requested_iters,
            completed_iters=self.completed_iters,
            metadata=self.metadata,
        )
        if not isinstance(self.stop_reason, SamplerStopReason):
            raise GeoBrainError(
                "InferenceResult.stop_reason must be SamplerStopReason",
                object_name="InferenceResult",
                field="stop_reason",
                expected=SamplerStopReason,
                actual=type(self.stop_reason),
            )
        if self.stop_reason is SamplerStopReason.COMPLETED and completed != requested:
            raise GeoBrainError(
                "COMPLETED requires all requested sampling transitions",
                object_name="InferenceResult",
                field="completed_iters",
                expected=requested,
                actual=completed,
            )
        callback_phase = metadata.get("callback_phase")
        callback_iteration = metadata.get("callback_iteration")
        if callback_phase not in {None, "warmup", "sampling"}:
            raise GeoBrainError(
                "callback_phase must identify the committed callback phase",
                object_name="InferenceResult",
                field="callback_phase",
                expected="'warmup', 'sampling', or None",
                actual=callback_phase,
            )
        if callback_iteration is not None and (
            not isinstance(callback_iteration, int)
            or isinstance(callback_iteration, bool)
        ):
            raise GeoBrainError(
                "callback_iteration must be an integer callback index or None",
                object_name="InferenceResult",
                field="callback_iteration",
                expected="int | None",
                actual=callback_iteration,
            )
        if callback_phase is None and callback_iteration is not None:
            raise GeoBrainError(
                "callback_iteration requires an explicit callback_phase",
                object_name="InferenceResult",
                field="callback_phase",
                expected="'warmup' or 'sampling'",
                actual=callback_phase,
            )
        if callback_phase is not None and callback_iteration is None:
            raise GeoBrainError(
                "callback_phase requires an explicit callback_iteration",
                object_name="InferenceResult",
                field="callback_iteration",
                expected="int",
                actual=None,
            )

        if self.stop_reason is SamplerStopReason.COMPLETED:
            if callback_phase is not None:
                raise GeoBrainError(
                    "COMPLETED cannot carry callback-stop metadata",
                    object_name="InferenceResult",
                    field="callback_phase",
                    expected=None,
                    actual=callback_phase,
                )
        elif callback_phase is None:
            raise GeoBrainError(
                "CALLBACK requires an explicit committed callback event",
                object_name="InferenceResult",
                field="stop_reason",
                expected="callback_phase and callback_iteration",
                actual={
                    "callback_phase": callback_phase,
                    "callback_iteration": callback_iteration,
                },
            )
        elif callback_phase == "sampling":
            expected_iteration = completed - 1
            if completed < 1 or callback_iteration != expected_iteration:
                raise GeoBrainError(
                    "sampling callback stop must commit its candidate transition",
                    object_name="InferenceResult",
                    field="callback_iteration",
                    expected=(
                        "completed_iters - 1 with at least one committed "
                        "sampling transition"
                    ),
                    actual={
                        "callback_iteration": callback_iteration,
                        "completed_iters": completed,
                    },
                )
        else:
            requested_warmup = validate_int(
                metadata.get("requested_warmup"),
                owner="InferenceResult",
                field="requested_warmup",
                minimum=0,
            )
            completed_warmup = validate_int(
                metadata.get("completed_warmup"),
                owner="InferenceResult",
                field="completed_warmup",
                minimum=0,
            )
            expected_iteration = completed_warmup - requested_warmup - 1
            valid_warmup_event = (
                completed == 0
                and requested_warmup >= 1
                and 1 <= completed_warmup <= requested_warmup
                and callback_iteration == expected_iteration
                and callback_iteration < 0
            )
            if not valid_warmup_event:
                raise GeoBrainError(
                    "warmup callback stop must identify its committed warmup event",
                    object_name="InferenceResult",
                    field="callback_iteration",
                    expected={
                        "completed_iters": 0,
                        "requested_warmup": ">= 1",
                        "completed_warmup": (
                            "between 1 and requested_warmup inclusive"
                        ),
                        "callback_iteration": expected_iteration,
                    },
                    actual={
                        "completed_iters": completed,
                        "requested_warmup": requested_warmup,
                        "completed_warmup": completed_warmup,
                        "callback_iteration": callback_iteration,
                    },
                )

        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "log_post_history", history)
        object.__setattr__(self, "acceptance_rate", rate)
        object.__setattr__(self, "requested_iters", requested)
        object.__setattr__(self, "completed_iters", completed)
        object.__setattr__(self, "metadata", metadata)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return per-field diagnostics from :mod:`geobrain.bayes.diagnostics`."""
        from .diagnostics import summarize

        summary: dict[str, dict[str, Any]] = summarize(self)
        return summary

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any]]]:
        """Serialize only plain owned state and restore through validation."""
        state = {
            "samples": _thaw_owned(self.samples),
            "log_post_history": self.log_post_history.detach().clone(),
            "acceptance_rate": self.acceptance_rate,
            "requested_iters": self.requested_iters,
            "completed_iters": self.completed_iters,
            "stop_reason": self.stop_reason,
            "metadata": _thaw_owned(self.metadata),
        }
        return _restore_inference_result, (state,)
