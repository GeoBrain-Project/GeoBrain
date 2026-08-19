"""
Argument-validation and immutability helpers shared platform-wide.

Started as private helpers for the core containers, but the whole platform
now validates through here, the Bayes samplers, ``optim``, ``decision``,
``inverse`` and the core containers all import these directly, so the
module is addressable-tier: a
cross-cutting library, not core-internal machinery. Depends on
:mod:`geobrain.core.errors` only; every raise carries the structured
``GeoBrainError`` field/expected/actual payload.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .errors import DifferentiabilityError, FieldShapeError, GeoBrainError

__all__ = [
    "clamp_params_in_place",
    "freeze_mapping",
    "validate_bool",
    "validate_bound_endpoint",
    "validate_bound_pair",
    "validate_finite_float",
    "validate_finite_tensor",
    "validate_int",
    "validate_mapping_key",
    "validate_non_negative_int",
    "validate_param_mapping",
    "validate_param_name",
    "validate_positive_int",
    "validate_std_positive",
    "validate_tensor_mapping",
    "validate_trainable_params",
]


def freeze_mapping(d: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return a read-only view of ``d`` (empty when falsy). Shared by the frozen
    state containers (:class:`ModelState` / :class:`ForwardOutput`) and the typed
    contexts (:class:`ForwardContext`) to make their dict fields immutable.

    The freeze is **shallow**: the top-level mapping is read-only and the source
    dict is copied (so the caller's dict cannot alias in), but the *values* are
    shared by reference. Sharing the value tensors is intentional (autograd needs
    the same objects); the consequence is that a nested mutable container in a
    free-form ``metadata`` / ``extras`` value (e.g. a list or dict) stays mutable
    through the view. This is acceptable because ``metadata`` is documented as not
    used by autograd and stable operators must not read ``extras``; deep-freeze
    only if strict nested-immutability is ever required.
    """
    return MappingProxyType(dict(d)) if d else MappingProxyType({})


def validate_mapping_key(owner: str, field: str, name: object) -> None:
    """Validate that a mapping key is a non-empty string. Shared by the frozen
    state containers and the typed contexts."""
    if not isinstance(name, str) or not name:
        raise GeoBrainError(
            f"{owner}.{field} keys must be non-empty strings",
            object_name=owner,
            field=field,
            expected="non-empty string keys",
            actual=name,
        )


def validate_tensor_mapping(
    owner: str, field: str, mapping: Mapping[str, Any], *, message: str
) -> None:
    """Validate a name→tensor mapping: every key a non-empty string, every
    value a :class:`torch.Tensor`. The single implementation behind the frozen
    state containers' ``__post_init__`` checks (``ModelState.tensors``,
    ``ForwardOutput.data`` / ``.fields``); ``message`` is the exact error text
    raised for a non-tensor value, so each container keeps its own wording."""
    for name, t in mapping.items():
        validate_mapping_key(owner, field, name)
        if not isinstance(t, torch.Tensor):
            raise FieldShapeError(
                message,
                object_name=owner,
                field=name,
                expected=torch.Tensor,
                actual=type(t),
            )


def validate_std_positive(
    std: int | float | torch.Tensor, *, object_name: str, field: str
) -> None:
    """
    Validate that a standard-deviation value is real, finite, and strictly > 0.

    Shared by :class:`~geobrain.inverse.likelihood.GaussianLikelihood` and
    :class:`~geobrain.inverse.prior.GaussianPrior`.

    Args:
        std: The standard deviation to validate (scalar or tensor).
        object_name: Prefix used in the raised :class:`GeoBrainError` so each
            caller's diagnostics stay specific.
        field: The field name reported in the error.

    Raises:
        GeoBrainError: If ``std`` is complex, non-finite, or not strictly
            positive.
    """
    if isinstance(std, torch.Tensor):
        if std.is_complex():
            raise GeoBrainError(
                f"{object_name}.std must be real-valued",
                object_name=object_name,
                field=field,
                expected="real, finite, > 0",
                actual="complex tensor",
            )
        if std.numel() == 0:
            raise GeoBrainError(
                f"{object_name}.std must be non-empty",
                object_name=object_name,
                field=field,
                expected="at least one finite value > 0",
                actual="empty tensor",
            )
        if not bool(torch.isfinite(std).all()) or not bool((std > 0).all()):
            raise GeoBrainError(
                f"{object_name}.std must be finite and strictly positive",
                object_name=object_name,
                field=field,
                expected="finite values > 0",
                actual=std,
            )
        return

    value = float(std)
    if not math.isfinite(value) or value <= 0.0:
        raise GeoBrainError(
            f"{object_name}.std must be finite and strictly positive",
            object_name=object_name,
            field=field,
            expected="finite value > 0",
            actual=std,
        )


# --- Generic argument validators -------------------------------------------
#
# Small structured-``GeoBrainError``-raising validators shared by the two
# ``InverseProblem`` views: :mod:`geobrain.optim` (Inverter / Adam / LBFGS) and
# :mod:`geobrain.bayes` (samplers), so a single home holds the contract and its
# diagnostics instead of each view rolling its own (and drifting).


def validate_param_mapping(
    params: Mapping[str, torch.Tensor], owner: str, *, require_grad: bool = False,
) -> None:
    """Validate a parameter dict: non-empty, non-empty-string keys, ``torch.Tensor``
    values; optionally each value must have ``requires_grad=True``.

    Shared by the two ``InverseProblem`` views so the param-dict contract lives in
    one place. :func:`validate_trainable_params` is the ``require_grad=True``
    specialization used by the bare solvers (Adam / LBFGS) and the samplers, which
    consume leaf tensors directly; the :class:`~geobrain.optim.Inverter` wraps the
    values into :class:`torch.nn.Parameter` itself, so it calls this with the
    default ``require_grad=False``."""
    if not params:
        raise GeoBrainError(
            f"{owner} requires at least one"
            + (" trainable" if require_grad else "")
            + " parameter",
            object_name=owner, field="params",
            expected="non-empty mapping", actual={},
        )
    for name, tensor in params.items():
        if not isinstance(name, str) or not name:
            raise GeoBrainError(
                f"{owner} parameter names must be non-empty strings",
                object_name=owner,
                field="params",
                expected="non-empty string keys",
                actual=name,
            )
        if not isinstance(tensor, torch.Tensor):
            raise GeoBrainError(
                f"{owner} params values must be torch.Tensor",
                object_name=owner, field=f"params[{name!r}]",
                expected=torch.Tensor, actual=type(tensor),
            )
        if not (tensor.is_floating_point() or tensor.is_complex()):
            # Integer/bool params cannot carry gradients and break nn.Parameter
            # wrapping with a raw torch RuntimeError; reject with a structured
            # DifferentiabilityError instead (the same contract operators use).
            raise DifferentiabilityError(
                f"{owner} params must have a floating or complex dtype "
                "(integer/bool tensors cannot carry gradients)",
                object_name=owner, field=f"params[{name!r}]",
                expected="floating or complex dtype", actual=tensor.dtype,
            )
        if require_grad and not tensor.requires_grad:
            raise DifferentiabilityError(
                f"{owner} params must have requires_grad=True",
                object_name=owner, field=f"params[{name!r}]",
                expected="requires_grad=True", actual=tensor.requires_grad,
            )


def validate_trainable_params(
    params: Mapping[str, torch.Tensor], owner: str,
) -> None:
    """Validate a parameter dict: non-empty, all trainable (``requires_grad``) tensors."""
    validate_param_mapping(params, owner, require_grad=True)


def validate_finite_float(
    value: object,
    *,
    owner: str,
    field: str,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    """Coerce ``value`` to a finite float, optionally bounded below.

    Args:
        value: candidate value; anything ``float()`` accepts.
        owner: object name woven into the structured error.
        field: field name woven into the structured error.
        minimum: optional lower bound.
        minimum_inclusive: whether ``minimum`` itself is allowed.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise GeoBrainError(
            f"{owner}.{field} must be a finite float",
            object_name=owner,
            field=field,
            expected="finite float",
            actual=value,
        ) from None
    if not math.isfinite(out):
        raise GeoBrainError(
            f"{owner}.{field} must be finite",
            object_name=owner,
            field=field,
            expected="finite float",
            actual=value,
        )
    if minimum is not None:
        invalid = out < minimum if minimum_inclusive else out <= minimum
        if invalid:
            op = ">=" if minimum_inclusive else ">"
            raise GeoBrainError(
                f"{owner}.{field} must be {op} {minimum}",
                object_name=owner,
                field=field,
                expected=f"{op} {minimum}",
                actual=value,
            )
    return out


def validate_finite_tensor(
    tensor: torch.Tensor,
    *,
    owner: str,
    field: str,
) -> None:
    """Reject a physical-input tensor that contains any NaN or ±inf value.

    The scalar sibling :func:`validate_finite_float` guards constructor-time
    Python scalars; this guards a whole tensor at a forward's entry. A silent
    non-finite input (e.g. a NaN ``rho``) otherwise propagates through an
    analytic kernel and yields a NaN observation with no diagnostic, poisoning a
    downstream inversion. The check is fully **vectorised**
    (``torch.isfinite(tensor).all()``), no per-element Python, and imposes no
    sign/range bound, so legitimately signed physical quantities (e.g. a negative
    density *contrast* or a signed magnetisation component) are untouched.
    """
    # ``isfinite`` is False for NaN and ±inf; a complex tensor's element is
    # finite iff both parts are, so this covers complex inputs too.
    if not bool(torch.isfinite(tensor).all()):
        raise GeoBrainError(
            f"{owner}.{field} must be finite (contains NaN or inf)",
            object_name=owner,
            field=field,
            expected="all-finite tensor (no NaN/inf)",
            actual="tensor with non-finite entries",
        )


def validate_int(
    value: object,
    *,
    owner: str,
    field: str,
    minimum: int | None = None,
    minimum_inclusive: bool = True,
) -> int:
    """Validate ``value`` as a true int (bool rejected), optionally bounded below.

    Args:
        value: candidate value; must already be an ``int`` (bool rejected).
        owner: object name woven into the structured error.
        field: field name woven into the structured error.
        minimum: optional lower bound.
        minimum_inclusive: whether ``minimum`` itself is allowed.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeoBrainError(
            f"{owner}.{field} must be an integer",
            object_name=owner,
            field=field,
            expected="integer",
            actual=value,
        )
    if minimum is not None:
        invalid = value < minimum if minimum_inclusive else value <= minimum
        if invalid:
            op = ">=" if minimum_inclusive else ">"
            raise GeoBrainError(
                f"{owner}.{field} must be {op} {minimum}",
                object_name=owner,
                field=field,
                expected=f"{op} {minimum}",
                actual=value,
            )
    return value


def validate_positive_int(
    value: object,
    *,
    owner: str,
    field: str,
) -> int:
    """Validate ``value`` as an int ``>= 1``.

    Args:
        value: candidate value.
        owner: object name woven into the structured error.
        field: field name woven into the structured error.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeoBrainError(
            f"{owner}.{field} must be a positive integer",
            object_name=owner,
            field=field,
            expected="positive integer",
            actual=value,
        )
    if value <= 0:
        raise GeoBrainError(
            f"{owner}.{field} must be a positive integer",
            object_name=owner,
            field=field,
            expected="positive integer",
            actual=value,
        )
    return value


def validate_non_negative_int(
    value: object,
    *,
    owner: str,
    field: str,
) -> int:
    """Validate ``value`` as an int ``>= 0``.

    Args:
        value: candidate value.
        owner: object name woven into the structured error.
        field: field name woven into the structured error.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise GeoBrainError(
            f"{owner}.{field} must be a non-negative integer",
            object_name=owner,
            field=field,
            expected="non-negative integer",
            actual=value,
        )
    if value < 0:
        raise GeoBrainError(
            f"{owner}.{field} must be a non-negative integer",
            object_name=owner,
            field=field,
            expected="non-negative integer",
            actual=value,
        )
    return value


def validate_bool(value: object, *, owner: str, field: str) -> bool:
    """Validate ``value`` as a true bool (ints are rejected).

    Args:
        value: candidate value.
        owner: object name woven into the structured error.
        field: field name woven into the structured error.
    """
    if not isinstance(value, bool):
        raise GeoBrainError(
            f"{owner}.{field} must be a bool",
            object_name=owner,
            field=field,
            expected=bool,
            actual=type(value),
        )
    return value


def validate_param_name(name: object, *, owner: str) -> None:
    """Validate a parameter name as a non-empty string.

    Args:
        name: candidate parameter name.
        owner: object name woven into the structured error.
    """
    if not isinstance(name, str) or not name:
        raise GeoBrainError(
            f"{owner} parameter names must be non-empty strings",
            object_name=owner,
            field="params",
            expected="non-empty string keys",
            actual=name,
        )


# --- Bound (clamp interval) validators -------------------------------------
#
# A bound endpoint is ``None`` (one-sided / unconstrained) or a finite float; a
# bound pair is ``(lo, hi)`` with ``lo <= hi`` when both are present. Shared by
# the two clamp surfaces in :mod:`geobrain.optim`: the ``Inverter(bounds=...)``
# normalizer (:mod:`geobrain.optim._normalize`) and the standalone
# ``BoundsClamp`` step_projection (:mod:`geobrain.optim.processing`). ``owner``
# and ``field`` are threaded through so each caller keeps its own diagnostics.


def validate_bound_endpoint(
    value: object, *, owner: str, field: str,
) -> float | None:
    """Validate one clamp endpoint: ``None`` or a finite float."""
    if value is None:
        return None
    try:
        bound = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise GeoBrainError(
            f"{owner} endpoints must be finite floats or None",
            object_name=owner,
            field=field,
            expected="finite float or None",
            actual=value,
        ) from None
    if not math.isfinite(bound):
        raise GeoBrainError(
            f"{owner} endpoints must be finite floats or None",
            object_name=owner,
            field=field,
            expected="finite float or None",
            actual=value,
        )
    return bound


def validate_bound_pair(
    pair: object, *, owner: str, field: str, lo_field: str, hi_field: str,
) -> tuple[float | None, float | None]:
    """Validate a ``(lo, hi)`` clamp pair: unpackable, finite-or-None endpoints,
    ``lo <= hi`` when both are present."""
    try:
        lo_raw, hi_raw = pair  # type: ignore[misc]
    except (TypeError, ValueError):
        raise GeoBrainError(
            f"{owner} values must be (lo, hi) pairs",
            object_name=owner,
            field=field,
            expected="two endpoints: finite float or None",
            actual=pair,
        ) from None
    lo = validate_bound_endpoint(lo_raw, owner=owner, field=lo_field)
    hi = validate_bound_endpoint(hi_raw, owner=owner, field=hi_field)
    if lo is not None and hi is not None and lo > hi:
        raise GeoBrainError(
            f"{owner} lower endpoint must be <= upper endpoint",
            object_name=owner,
            field=field,
            expected="lo <= hi",
            actual=(lo, hi),
        )
    return lo, hi


def clamp_params_in_place(
    params: Mapping[str, torch.Tensor],
    bounds: Mapping[str, tuple[float | None, float | None]],
) -> None:
    """Project each named tensor into its ``(lo, hi)`` interval in-place.

    Runs under ``torch.no_grad()`` so the clamp never registers on the autograd
    tape (correct for post-step bound projection). Entries with both endpoints
    ``None`` are skipped. Callers are responsible for validating that ``bounds``
    keys exist in ``params``. Shared by ``Inverter._apply_bounds`` and
    ``BoundsClamp`` (:mod:`geobrain.optim.processing`) so the projection loop
    lives in one place."""
    if not bounds:
        return
    with torch.no_grad():
        for name, (lo, hi) in bounds.items():
            if lo is None and hi is None:
                continue
            params[name].clamp_(min=lo, max=hi)
