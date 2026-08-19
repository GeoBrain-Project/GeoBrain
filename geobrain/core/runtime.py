"""Runtime execution policy: one steerable record for how GeoBrain executes.

Consumed today: ``device``/``dtype`` resolution (:func:`resolve_device` /
:func:`resolve_dtype`) and ``torch.compile`` gating (:func:`maybe_compile`,
wave engine). The ``amp`` and ``deterministic`` fields are DECLARED policy
that no engine consults yet; they reserve the vocabulary so opting an
engine in later is a policy read, not an API change.

(Layer rule: this module imports **nothing from geobrain**, not even
``core.errors``: enforced by an architecture layer-contract test.)

Two divergent idioms motivated one steerable policy: the wave engine carried device/
dtype on its ``SolverConfig`` (defaulting to cpu/float32), while every other operator
derived device/dtype from its input model tensor. Neither could be steered globally.
This module adds one steerable policy that both idioms can consult, WITHOUT changing
default behavior. The default :class:`Policy` is deliberately *inert*:

* ``device is None`` / ``dtype is None`` mean "respect the input tensors" (the historical default),
* ``amp`` and ``compile`` are disabled,
* ``deterministic`` is off.

So importing and even threading the policy through changes nothing until a caller opts
in via :func:`set_default_policy` or the :func:`policy_scope` context manager. This is
what keeps the golden-value freeze bit-for-bit while the accelerated paths are built on
top of it.

``maybe_compile`` is the sanctioned way to apply ``torch.compile`` in GeoBrain: a
transparent no-op when the policy disables compilation (preserving byte-identity and
the golden freeze); when enabled it wraps the compiled callable with an automatic
eager fallback so a compilation failure degrades gracefully instead of crashing a
forward solve. Compile the small, hot *step* kernels, not the Python time loop.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""
from __future__ import annotations

import functools
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Callable, cast, Iterator, Optional, TypeVar, Union

import torch

DeviceLike = Union[str, torch.device]

F = TypeVar("F", bound=Callable[..., object])

__all__ = [
    "Policy",
    "AmpPolicy",
    "CompilePolicy",
    "current_policy",
    "set_default_policy",
    "policy_scope",
    "resolve_device",
    "resolve_dtype",
    "maybe_compile",
]


@dataclass(frozen=True)
class AmpPolicy:
    """Automatic mixed precision. ``enabled`` gates an ``autocast`` region; ``dtype`` is
    the autocast compute dtype (bf16 preferred over fp16 on A100 for its fp32-range
    exponent, avoiding the loss-scaling machinery fp16 needs)."""

    enabled: bool = False
    dtype: torch.dtype = torch.bfloat16


@dataclass(frozen=True)
class CompilePolicy:
    """``torch.compile`` options. ``mode`` is passed through ('default',
    'reduce-overhead', 'max-autotune'); ``fullgraph`` forbids graph breaks (used for
    the small step kernels that should compile whole)."""

    enabled: bool = False
    mode: str = "default"
    fullgraph: bool = False


@dataclass(frozen=True)
class Policy:
    """Immutable execution policy. Retrieve the active one with :func:`current_policy`."""

    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    # amp / deterministic are declared-but-not-yet-consumed policy (see the
    # module docstring); compile is consumed via maybe_compile.
    amp: AmpPolicy = field(default_factory=AmpPolicy)
    compile: CompilePolicy = field(default_factory=CompilePolicy)
    deterministic: bool = False

    def __post_init__(self) -> None:
        # Ergonomic coercion: accept "cuda"/"cpu" strings, always store torch.device.
        if self.device is not None and not isinstance(self.device, torch.device):
            object.__setattr__(self, "device", torch.device(self.device))


_INERT = Policy()
_policy_var: ContextVar[Policy] = ContextVar("geobrain_runtime_policy", default=_INERT)


def current_policy() -> Policy:
    """The policy in effect for the calling context (thread/async-safe)."""
    return _policy_var.get()


def set_default_policy(policy: Policy) -> None:
    """Set the default policy for the CURRENT context. Prefer :func:`policy_scope`
    for scoped overrides; call this once at app startup for a de-facto global.

    ContextVar semantics, honestly: ``asyncio`` tasks inherit the caller's
    context, but a fresh ``threading.Thread`` (DataLoader worker, executor
    pool) starts with an empty context and falls back to the inert default;
    set the policy inside each worker, or pass it explicitly."""
    if not isinstance(policy, Policy):
        raise TypeError(f"set_default_policy expects a Policy, got {type(policy)!r}")
    _policy_var.set(policy)


@contextmanager
def policy_scope(**overrides: object) -> Iterator[Policy]:
    """Temporarily override fields of the current policy within a ``with`` block.

    Keyword overrides replace individual :class:`Policy` fields (``device``, ``dtype``,
    ``amp``, ``compile``, ``deterministic``); untouched fields inherit the enclosing
    policy, so scopes nest and compose. The previous policy is restored on exit, including
    when the block raises.
    """
    new = replace(current_policy(), **cast(dict[str, Any], overrides))
    token = _policy_var.set(new)
    try:
        yield new
    finally:
        _policy_var.reset(token)


def resolve_device(
    explicit: Optional[DeviceLike] = None, *tensors: object
) -> torch.device:
    """Resolve the device to run on. Precedence: explicit argument > policy override >
    first tensor's device > cpu. This is the chokepoint operators call so a single policy
    can flip the whole platform cpu<->cuda without editing every ``.to(...)``."""
    if explicit is not None:
        return torch.device(explicit)
    policy = current_policy()
    if policy.device is not None:
        return policy.device
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.device
    return torch.device("cpu")


def resolve_dtype(
    explicit: Optional[torch.dtype] = None, *tensors: object
) -> torch.dtype:
    """Resolve the dtype. Precedence: explicit argument > policy override > first tensor's
    dtype > torch default. Note per-family dtype constraints (EM complex128, PF fp64) are
    enforced by the operators; this only supplies a *default* when nothing else pins it."""
    if explicit is not None:
        return explicit
    policy = current_policy()
    if policy.dtype is not None:
        return policy.dtype
    for t in tensors:
        if isinstance(t, torch.Tensor):
            return t.dtype
    return torch.get_default_dtype()


def maybe_compile(fn: F, *, policy: Optional[Policy] = None) -> F:
    """Return ``fn`` compiled per the active policy, or ``fn`` unchanged if disabled.

    When ``policy.compile.enabled`` is false (the default), this returns the *identical*
    callable, no wrapping, no overhead, byte-identical behavior. When enabled, it returns
    a wrapper around ``torch.compile(fn, mode=..., fullgraph=...)`` that, on the first
    exception from the compiled path, emits a warning and permanently falls back to eager
    ``fn`` for the life of the wrapper (so one un-compilable input can't break a run).
    """
    policy = policy if policy is not None else current_policy()
    cp = policy.compile
    if not cp.enabled:
        return fn

    compiled = torch.compile(fn, mode=cp.mode, fullgraph=cp.fullgraph)
    state = {"use_compiled": True}

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        if state["use_compiled"]:
            try:
                return compiled(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001, deliberate broad fallback
                warnings.warn(
                    f"torch.compile path for {getattr(fn, '__name__', fn)!r} failed "
                    f"({type(exc).__name__}: {exc}); falling back to eager execution.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                state["use_compiled"] = False
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
