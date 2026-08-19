"""
Learning-rate and bounds normalization for :class:`Inverter`.

Pure validators that turn the loose ``lr`` (scalar or per-parameter mapping) and
``bounds`` (per-parameter ``(lo, hi)`` pairs) constructor arguments into the
canonical dict forms the inverter holds. Split out of ``inverter.py`` purely to
keep that facade at a tractable size (same rationale as :mod:`_factories`).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from geobrain.core import GeoBrainError
from geobrain.core.validation import validate_bound_pair


def _normalize_lr_mapping(
    lr: Mapping[str, float],
    params: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Validate and normalize per-parameter learning rates."""
    missing = set(params) - set(lr)
    unknown = set(lr) - set(params)
    if missing or unknown:
        raise GeoBrainError(
            "Inverter.lr mapping keys must exactly match params",
            object_name="Inverter",
            field="lr",
            expected=f"all and only {sorted(params)}",
            actual={
                "missing": sorted(missing),
                "unknown": sorted(unknown),
            },
        )
    return {
        name: _normalize_lr_value(f"lr[{name!r}]", lr[name])
        for name in params
    }


def _normalize_lr_value(field: str, value: float) -> float:
    try:
        lr = float(value)
    except (TypeError, ValueError):
        raise GeoBrainError(
            "Inverter.lr must be a finite non-negative float",
            object_name="Inverter",
            field=field,
            expected="finite non-negative float",
            actual=value,
        ) from None
    if not math.isfinite(lr) or lr < 0.0:
        raise GeoBrainError(
            "Inverter.lr must be a finite non-negative float",
            object_name="Inverter",
            field=field,
            expected="finite non-negative float",
            actual=value,
        )
    return lr


def _coerce_params(
    problem: Any,
    params: "Mapping[str, torch.Tensor] | Any | None",
) -> tuple[dict[str, torch.Tensor], Any | None]:
    """Resolve :class:`~geobrain.optim.Inverter`'s ``params=`` argument.

    Three accepted shapes (``params=model`` coercion):

    - A ``Mapping[str, Tensor]``: used AS-IS (returned as a plain ``dict``;
      no coercion, no identity check against ``problem.model``, an
      explicit dict is trusted to be whatever subset/superset of a model's
      trainables the caller intends, e.g. a partial inversion).
    - A model-like object: duck-typed via a callable ``.trainables()``
      (``hasattr``, NOT ``isinstance``: :mod:`geobrain.optim` must not
      import :mod:`geobrain.geomodel.earthmodel`, an architecture
      layer-contract test enforces this), coerced
      to ``params.trainables()``. When ``problem`` ALSO carries a ``.model``
      attribute (e.g. :class:`~geobrain.inverse.JointProblem`), that model
      must be the SAME object (``is``) as the one passed here, the
      identity contract: silently rebinding a different template would let
      the optimized leaves drift from the model ``problem``'s forward
      actually reads, an easy-to-miss bug with no other symptom than a
      stuck/wrong loss.
    - ``None`` (the default): defaults to ``problem.model.trainables()``
      when ``problem`` carries a ``.model``; raises when it does not (there
      is nothing to default from, and silently requiring an explicit
      ``params=`` would be a confusing ``TypeError``/``AttributeError``
      instead of a named, structured error).

    Args:
        problem: The :class:`~geobrain.optim.Inverter`'s ``problem=``
            argument, inspected ONLY via ``getattr(problem, "model",
            None)``, never imported/isinstance-checked against a concrete
            type.
        params: The raw ``params=`` argument (mapping, model, or ``None``).

    Returns:
        ``(params_dict, params_model)``: ``params_model`` is the model
        object ``params`` was coerced FROM (``problem.model`` in the
        ``None``-default case), stashed so :attr:`Inverter.params_model` can
        hand it back for the blessed physical-field recovery one-liner
        (``inv.params_model.resolve_from(result.params)``); ``None`` when
        ``params`` was passed as a bare mapping (nothing to stash).

    Raises:
        GeoBrainError: ``params=None`` with no usable ``problem.model``; a
            ``params=<model>`` whose identity disagrees with ``problem.model``;
            or a ``params=`` value that is neither a mapping, a
            trainables()-exposing model, nor ``None``.
    """
    problem_model = getattr(problem, "model", None)

    if params is None:
        if problem_model is None or not callable(getattr(problem_model, "trainables", None)):
            raise GeoBrainError(
                "Inverter params=None requires problem.model to be a "
                "trainables()-exposing model (e.g. JointProblem.model); "
                "pass params= explicitly otherwise",
                object_name="Inverter",
                field="params",
                expected="problem.model exposing trainables()",
                actual=None if problem_model is None else type(problem_model).__name__,
            )
        return dict(problem_model.trainables()), problem_model

    if isinstance(params, Mapping):
        return dict(params), None

    if callable(getattr(params, "trainables", None)):
        if problem_model is not None and problem_model is not params:
            raise GeoBrainError(
                "Inverter params=<model> identity mismatch: problem.model is "
                "a DIFFERENT model instance than params=, the identity "
                "contract forbids silently rebinding a different template",
                object_name="Inverter",
                field="params",
                expected=f"problem.model (id={id(problem_model)})",
                actual=f"params= object (id={id(params)})",
            )
        return dict(params.trainables()), params

    raise GeoBrainError(
        "Inverter params= must be a {name: Tensor} mapping, a model "
        "exposing trainables(), or None (defaulting to "
        "problem.model.trainables())",
        object_name="Inverter",
        field="params",
        expected="Mapping[str, Tensor] | <model with .trainables()> | None",
        actual=type(params).__name__,
    )


def _normalize_bounds(
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    params: Mapping[str, torch.Tensor],
) -> dict[str, tuple[float | None, float | None]]:
    """Validate and normalize per-parameter bound intervals."""
    if not bounds:
        return {}

    unknown = set(bounds) - set(params)
    if unknown:
        raise GeoBrainError(
            "Inverter.bounds contains unknown parameter names",
            object_name="Inverter",
            field="bounds",
            expected=f"keys in {sorted(params)}",
            actual=sorted(unknown),
        )

    normalized: dict[str, tuple[float | None, float | None]] = {}
    for name, pair in bounds.items():
        normalized[name] = validate_bound_pair(
            pair,
            owner="Inverter.bounds",
            field=f"bounds[{name!r}]",
            lo_field=f"bounds[{name!r}].lo",
            hi_field=f"bounds[{name!r}].hi",
        )

    return normalized
