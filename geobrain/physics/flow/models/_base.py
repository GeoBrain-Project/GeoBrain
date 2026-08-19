"""
The flow model contract.

Every reservoir model under :mod:`geobrain.physics.flow.models` (and
:mod:`geobrain.physics.flow.compositional`) is an ``nn.Module`` exposing the
same small duck-typed interface. That contract was historically documented only
in a docstring, which let the operator layer (:mod:`geobrain.physics.flow.operators`)
dispatch on ``isinstance`` ladders that recognised just the three TPFA models
(single-phase / oil-water / black-oil), so the richer MPFA / thermal /
compositional models were Newton-solvable through ``run_transient`` but had no
operator entry, and could not reach ``ModelState`` / ``ForwardOutput`` /
``InverseProblem`` / Bayes.

This module makes the contract explicit and machine-checkable. The operators now
dispatch **structurally** through :class:`FlowModel` and the helpers below. The
immutable :class:`~geobrain.physics.flow.contracts.FlowModelSchema` is the sole
authority for independent state names; model discovery never executes a
synthetic state probe.

The contract every model satisfies:

- ``n_cells``                                  -> ``int`` (property or attribute)
- ``state_size() -> int``                      flat state-vector length
- ``initial_state(*fields) -> Tensor``         pack per-variable fields, in
                                               state order, into the flat vector
- ``state_split(state) -> dict[str, Tensor]``  flat vector -> ``{internal key: field}``
- ``residual(state, state_old, dt, **sources) -> Tensor``
- ``jacobian(state, state_old, dt, **kw) -> Tensor``

The internal ``state_split`` keys (``p`` / ``sw`` / ``sg`` / ``T`` / ``z``) map
to the operator-facing ``ModelState`` variable names
(``pressure`` / ``sw`` / ``sg`` / ``temperature`` / ``composition``) through
:data:`SPLIT_TO_VARIABLE`: the single table where the two vocabularies meet.

A mandatory abstract base class is deliberately **not** imposed: the 18 models
are legitimately divergent ``nn.Module``\\s with different state layouts and
source-term kwargs, and the ``StencilInversionMixin`` already occupies the
inheritance slot on the thermal-MPFA family. A ``runtime_checkable`` Protocol
plus these free helper functions give the operator layer one structural contract
without a flag-day rewrite.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from ....core import ModelState
from ..contracts import FlowModelSchema
from ..errors import FlowContractError

# Internal state_split / initial_state-order key  ->  ModelState variable name.
# This one table is the single place the model-internal vocabulary and the
# operator-facing ModelState vocabulary meet.
SPLIT_TO_VARIABLE: dict[str, str] = {
    "p": "pressure",
    "sw": "sw",
    "sg": "sg",
    "T": "temperature",
    "z": "composition",
}


@runtime_checkable
class FlowModel(Protocol):
    """Structural contract shared by every flow reservoir model.

    ``runtime_checkable`` so the operator layer can assert ``isinstance(model,
    FlowModel)`` structurally (membership by attribute presence, not by
    inheritance), every TPFA / MPFA / thermal / compositional model already
    satisfies it. See the module docstring for the method semantics.
    """

    n_cells: int
    schema: FlowModelSchema

    def state_size(self) -> int: ...

    def initial_state(self, *fields: torch.Tensor) -> torch.Tensor: ...

    def state_split(self, state: torch.Tensor) -> dict[str, torch.Tensor]: ...

    def residual(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor: ...

    def jacobian(
        self,
        state: torch.Tensor,
        state_old: torch.Tensor,
        dt: float,
    ) -> torch.Tensor: ...


def state_variables(model: FlowModel) -> tuple[str, ...]:
    """The ordered ``ModelState`` variable names a model consumes and emits.

    The schema is inspected as immutable metadata. In particular, this function
    does not call ``state_size`` or ``state_split``: discovery cannot execute a
    model kernel, allocate a synthetic tensor, or infer state meaning from a
    value layout.
    """
    schema = getattr(model, "schema", None)
    if not isinstance(schema, FlowModelSchema):
        raise FlowContractError(
            "Flow model is missing its immutable schema",
            object_name=type(model).__qualname__,
            field="schema",
            expected="FlowModelSchema",
            actual=type(schema).__qualname__ if schema is not None else None,
        )
    return tuple(field.name for field in schema.primary_fields)


def pack_state(model: FlowModel, ms: ModelState) -> torch.Tensor:
    """Pack a :class:`ModelState` into the flat vector ``model.residual`` consumes.

    Fetches the model's state variables (in flat order) from ``ms`` and calls
    ``model.initial_state`` positionally, the structural generalisation of the
    old ``initial_state(p[, sw[, sg]])`` per-model calls.
    """
    fields = ms.fetch(*state_variables(model))
    return model.initial_state(*fields)


def unpack_state(model: FlowModel, state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a flat state into ``{ModelState variable: field}``.

    The inverse of :func:`pack_state`: ``model.state_split`` followed by the
    internal-key -> ModelState-name translation. Replaces the old
    ``_unpack_state`` that hard-coded the ``p`` -> ``pressure`` rename for the
    three TPFA models only.

    Mirrors :func:`state_variables`: derived diagnostic keys that ``state_split``
    may expose (black-oil ``so``; var-switch ``rs`` / ``so`` / ``saturated``) are
    **not** ModelState variables and are dropped, so the returned mapping is keyed
    by exactly the independent variables an operator emits.
    """
    return {
        SPLIT_TO_VARIABLE[key]: field
        for key, field in model.state_split(state).items()
        if key in SPLIT_TO_VARIABLE
    }
