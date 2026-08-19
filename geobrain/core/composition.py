"""
Operator composition: OperatorChain (sequential) and OperatorBundle (parallel).

- :class:`OperatorChain`: ``B @ A`` applies ``A`` first, then ``B`` (math
  convention ``B ∘ A``).
- :class:`OperatorBundle`: ``{name: op, ...}`` shares one :class:`ModelState`
  across operators and merges their :class:`ForwardOutput` outputs into one keyed by
  channel name.

A chain's differentiability is the weakest level among its members; its
``trainable_inputs`` are the first operator's (the ENTRY-POINT contract, a
non-first operator's ``trainable_inputs`` are capability metadata, presence-
checked by that operator at its own forward, not merged into the chain's
contract; the Inverter selects parameters explicitly via ``params=`` and
autograd flows regardless); its ``output_keys`` are the last
operator's. A joint observation's level is the weakest of its channels; its
``trainable_inputs`` are the union (first-seen order) of channel inputs; its
``output_keys`` are the channel names.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .differentiability import DifferentiabilityLevel, DifferentiabilitySpec
from .errors import GeoBrainError
from .operator import ForwardOperator, Operator
from .containers import ModelState, ForwardOutput
from .context import ForwardContext


_LEVEL_RANK: dict[DifferentiabilityLevel, int] = {
    DifferentiabilityLevel.NON_DIFFERENTIABLE: 0,
    DifferentiabilityLevel.FORWARD_ONLY: 1,
    DifferentiabilityLevel.IMPLICIT_VJP: 2,
    DifferentiabilityLevel.CUSTOM_VJP: 3,
    DifferentiabilityLevel.FULL_AUTOGRAD: 4,
}


def _weakest(levels: Sequence[DifferentiabilityLevel]) -> DifferentiabilityLevel:
    return min(levels, key=lambda lvl: _LEVEL_RANK[lvl])


def _weakest_level(operators: Sequence[Operator]) -> DifferentiabilityLevel:
    """The combined differentiability level of a set of operators: the weakest.

    Shared by :class:`OperatorChain` and :class:`OperatorBundle`, both derive
    their level identically (a composition is only as differentiable as its
    least-differentiable member). The ``trainable_inputs`` / ``output_keys``
    derivations genuinely differ between the two (chain = first/last operator;
    bundle = union / namespaced channel keys), so they are NOT unified here.
    """
    return _weakest([op.differentiability.level for op in operators])


def _terminates_in_observation(op: Operator) -> bool:
    # A OperatorBundle always produces a ForwardOutput (every channel is itself
    # observation-terminating, enforced in its __init__), so it counts as an
    # observation terminus: both for "may a chain end here" (yes) and for
    # "may it sit mid-chain feeding a transform" (no). ``OperatorBundle`` is
    # defined later in this module; the name resolves at call time (chains are
    # only ever built after import), so the forward reference is safe.
    if isinstance(op, (ForwardOperator, OperatorBundle)):
        return True
    if isinstance(op, OperatorChain):
        return op.is_observation
    return False


class OperatorChain(Operator):
    """
    Sequential composition. Build with ``B @ A`` or ``OperatorChain([A, B])``.

    Rules:
        - At least one operator.
        - At most one :class:`ForwardOperator`, and only in the last position.
        - Every other position must be a :class:`PropertyTransform` (or a chain
          that does not itself terminate in a :class:`ForwardOperator`).
    """

    _forward_label = "OperatorChain"

    def __init__(self, operators: Sequence[Operator]) -> None:
        super().__init__()
        ops = list(operators)
        if not ops:
            raise GeoBrainError(
                "OperatorChain requires at least one operator",
                object_name="OperatorChain",
                field="operators",
                expected="non-empty sequence",
                actual=ops,
            )
        for i, op in enumerate(ops):
            if not isinstance(op, Operator):
                raise GeoBrainError(
                    "OperatorChain element must be an Operator",
                    object_name="OperatorChain",
                    field=f"operators[{i}]",
                    expected=Operator,
                    actual=type(op),
                )
        for i, op in enumerate(ops[:-1]):
            if _terminates_in_observation(op):
                raise GeoBrainError(
                    "Only the last operator in a chain may be a ForwardOperator",
                    object_name="OperatorChain",
                    field=f"operators[{i}]",
                    expected="PropertyTransform (or chain of transforms)",
                    actual=type(op).__name__,
                )

        # The chain's spec takes operators[0]'s trainable_inputs (see
        # _build_spec): the ENTRY-POINT contract. Non-first operators'
        # trainable_inputs are deliberately NOT merged and NOT warned about:
        #
        # * ``trainable_inputs`` is CAPABILITY metadata ("differentiable
        #   w.r.t. these state keys; they must be present when I run"), not a
        #   demand that a chain train them. Class-level declarations exist on
        #   essentially every physics operator, so any heuristic that warns
        #   on non-first declarations flags the canonical composition pattern
        #   itself (transform first, physics last): measured: every chain in
        #   the test suite.
        # * Parameter selection is EXPLICIT at the Inverter (``params=``
        #   mapping); gradients flow wherever autograd allows, regardless of
        #   the chain's declared contract. Nothing is silently un-trainable.
        # * The runtime guarantee that matters survives per-operator: each
        #   member's forward presence-checks its OWN trainable_inputs against
        #   the threaded state (``_check_trainable_inputs``), so a downstream
        #   requirement that neither the caller nor an upstream stage supplies
        #   fails loud with MissingFieldError at the exact operator.
        #
        # (A construction-time warning was considered and rejected: any such
        # heuristic punishes deliberate fixed-parameter usage: e.g. inverting
        # Sw through Wood >> Gassmann with phi held fixed.)

        # Reconcile INPUT units across members that read the SAME field from the
        # shared EXTERNAL input state (see _check_input_unit_consistency for why a
        # chain's data-flow makes this subtler than the bundle's shared-state case).
        self._check_input_unit_consistency(ops)

        self._operators = nn.ModuleList(ops)
        self.differentiability = self._build_spec(ops)
        self._is_observation = _terminates_in_observation(ops[-1])

    @staticmethod
    def _check_input_unit_consistency(operators: Sequence[Operator]) -> None:
        """Reject a chain whose members read the SAME external-input field under
        conflicting input units.

        Data-flow distinction from :meth:`OperatorBundle._check_input_unit_consistency`;
        this is the subtle part. A **bundle** threads ONE shared ``ModelState``
        through every member, so *any* field two members declare with different
        units is unsatisfiable by a single tensor: a flat conflict. A **chain**
        instead threads each operator's OUTPUT into the next operator's INPUT, so
        a shared field name across two links can be one of two very different
        things:

        * a **shared external input**: a field both members read from the SAME
          input state, produced by neither: conflicting units here still cannot be
          satisfied by one tensor, so this is a real clash (the bundle rule
          applies), OR
        * a **produced intermediate**: a field an UPSTREAM member emits and a
          downstream member consumes (e.g. a converter transform that reads ``x``
          in ``foo`` and re-emits ``x`` in ``bar`` for the downstream stage). That
          is legitimate data flow *through the chain*, NOT a unit clash, the
          downstream unit describes the transformed intermediate, not the original
          input.

        A naive copy of the bundle check (group every ``input_units`` declaration
        by field) would FALSE-POSITIVE on the second case. The correct,
        input-side rule: reconcile a member's declared unit for field ``f`` ONLY
        when ``f`` is read from the external input, i.e. no EARLIER member in the
        chain produces ``f`` (``f`` not in any upstream member's ``output_keys``).
        Members whose read of ``f`` is a produced intermediate are excluded and
        never trigger a conflict. Among members that read ``f`` externally, two
        distinct unit strings raise a :class:`GeoBrainError`; undeclared fields are
        skipped (opt-in, no false positives).

        ``operators`` is in APPLICATION order (``operators[0]`` runs first), as
        stored by ``__init__`` / built by ``B @ A`` -> ``OperatorChain([A, B])``.
        """
        produced_upstream: set[str] = set()
        # field -> {unit_string -> [member labels reading it from the external input]}
        declared: dict[str, dict[str, list[str]]] = {}
        for idx, op in enumerate(operators):
            spec = op.differentiability
            units = getattr(spec, "input_units", {}) or {}
            for field_name, unit in units.items():
                # A field produced by an UPSTREAM member is an intermediate for
                # this consumer, NOT a shared external input: skip it.
                if field_name in produced_upstream:
                    continue
                label = f"operators[{idx}]:{type(op).__name__}"
                declared.setdefault(field_name, {}).setdefault(unit, []).append(label)
            # Register THIS member's produced fields only AFTER reading its inputs,
            # so a field it both reads and re-emits counts as an external read here
            # and an intermediate for everything downstream.
            produced_upstream.update(spec.output_keys)
        for field_name, by_unit in declared.items():
            if len(by_unit) < 2:
                continue  # 0 or 1 distinct external-input unit → nothing to reconcile
            detail = "; ".join(
                f"{unit!r} (declared by {', '.join(sorted(members))})"
                for unit, members in sorted(by_unit.items())
            )
            raise GeoBrainError(
                f"OperatorChain members declare conflicting input units for the "
                f"shared external-input field {field_name!r}: {detail}. These "
                f"members all read {field_name!r} from the same input state (none "
                f"is produced upstream in the chain), so a single tensor cannot "
                f"satisfy both conventions, convert to a common unit, or insert a "
                f"transform that re-emits {field_name!r} in the downstream unit.",
                object_name="OperatorChain",
                field=field_name,
                expected="one input unit across all members reading this external input",
                actual=sorted(by_unit),
            )

    @staticmethod
    def _build_spec(operators: Sequence[Operator]) -> DifferentiabilitySpec:
        """Derive the chain spec: weakest level, first op's inputs, last op's
        outputs, plus the introspection metadata a NESTED composition must not
        lose: the union of members' ``structural_params`` and the units of the
        chain's EXTERNAL inputs (fields no upstream member produces, exactly the
        set :meth:`_check_input_unit_consistency` reconciles). Without this, a
        chain nested inside an :class:`OperatorBundle` would advertise an empty
        ``input_units`` and silently drop out of the bundle's unit
        reconciliation."""
        produced_upstream: set[str] = set()
        external_units: dict[str, str] = {}
        structural: list[str] = []
        seen_structural: set[str] = set()
        for op in operators:
            spec = op.differentiability
            for field_name, unit in (getattr(spec, "input_units", {}) or {}).items():
                if field_name not in produced_upstream and field_name not in external_units:
                    external_units[field_name] = unit
            for name in spec.structural_params:
                if name not in seen_structural:
                    structural.append(name)
                    seen_structural.add(name)
            produced_upstream.update(spec.output_keys)
        return DifferentiabilitySpec(
            level=_weakest_level(operators),
            trainable_inputs=operators[0].differentiability.trainable_inputs,
            output_keys=operators[-1].differentiability.output_keys,
            structural_params=tuple(structural),
            input_units=external_units,
        )

    @property
    def operators(self) -> tuple[Operator, ...]:
        """The operators in application order (first applied first)."""
        return tuple(self._operators)

    @property
    def is_observation(self) -> bool:
        """True iff the terminal operator produces a ForwardOutput."""
        return self._is_observation

    def _forward(
        self, state: ModelState, ctx: ForwardContext
    ) -> ModelState | ForwardOutput:
        """
        Apply each operator in turn, threading each output into the next.

        Args:
            state: Input model parameters.
            ctx: Per-call configuration.

        Returns:
            The terminal operator's output, a :class:`ForwardOutput` for an
            observation chain, otherwise a :class:`ModelState`.
        """
        intermediate: Any = state
        last = len(self._operators) - 1
        for i, op in enumerate(self._operators):
            intermediate = op(intermediate, ctx)
            # Every non-terminal operator must hand a ModelState to the next.
            if i != last and not isinstance(intermediate, ModelState):
                raise GeoBrainError(
                    "Non-terminal operator in chain returned a non-ModelState",
                    object_name="OperatorChain",
                    field=f"operators[{i}]",
                    expected=ModelState,
                    actual=type(intermediate),
                )
        return intermediate

    def forward(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> ModelState | ForwardOutput:
        """
        Validate the input and run the chain. See :meth:`_forward`.

        Args:
            state: Input model parameters.
            ctx: Per-call configuration. Defaults to an empty
                :class:`ForwardContext`.

        Returns:
            The chain's terminal output.
        """
        ctx = self._validate_forward_input(state, ctx)
        return self._forward(state, ctx)

    def __repr__(self) -> str:
        names = " @ ".join(type(op).__name__ for op in reversed(list(self._operators)))
        return f"OperatorChain({names})"


class OperatorBundle(Operator):
    """
    Parallel observation: one :class:`ModelState`, many :class:`ForwardOperator`.

    Per-channel predictions are merged into a single :class:`ForwardOutput`. The
    three result key-spaces merge by **different**, deliberately-chosen rules:

    - ``data`` is *arity-dependent*: a single-output channel is lifted to the
      bare channel name, a multi-output channel is namespaced as
      ``f"{channel}/{subkey}"``. These are the observable channels tracked by
      ``output_keys`` and validated against ``InverseProblem`` observations.
    - ``fields`` (diagnostics, **not** tracked by ``output_keys``) is *always*
      namespaced ``f"{channel}/{subkey}"`` regardless of arity, so two channels'
      same-named diagnostics never collide.
    - ``metadata`` is *always* keyed by the bare ``channel``: the whole
      sub-prediction's metadata dict lives under its channel name.

    All channels run on the **shared** ``ModelState`` (that is the point, joint
    inversion of one model against several observations). They also share the
    call's :class:`ForwardContext` by default, but ``channel_configs`` lets a
    channel **overlay** its own config keys on top of the shared context. This
    is the seam for, e.g., time-lapse / multi-vintage joints where each channel
    sees the same model but a different survey / wavelet / mesh::

        OperatorBundle(
            {"v2020": Acoustic2D(...), "v2023": Acoustic2D(...)},
            channel_configs={"v2020": {"wavelet": w2020},
                             "v2023": {"wavelet": w2023}},
        )

    The overlay is shallow (channel keys win over the shared context); a channel
    with no entry uses the shared context unchanged, so the default behaviour is
    identical to before. A ``None`` value in a channel config does **not** set the
    key to ``None``; it *deletes* the inherited key so the channel falls back to
    that key's absence (see :meth:`ForwardContext.with_overrides`).

    Note:
        Channel names must not contain ``"/"``, that character is reserved for
        the namespacing separator used in multi-output channel keys
        (``f"{channel}/{subkey}"``).

    Args:
        channels: ``{channel name: operator}`` members; each must terminate
            in a :class:`ForwardOutput` and names must not contain ``'/'``.
        channel_configs: optional per-channel context overlays (see the
            merge rules above).
    """

    _forward_label = "OperatorBundle"

    def __init__(
        self,
        channels: Mapping[str, Operator],
        *,
        channel_configs: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        if not channels:
            raise GeoBrainError(
                "OperatorBundle requires at least one channel",
                object_name="OperatorBundle",
                field="channels",
                expected="non-empty mapping",
                actual=channels,
            )
        for name, op in channels.items():
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "OperatorBundle channel name must be a non-empty string",
                    object_name="OperatorBundle",
                    field="channels",
                    actual=name,
                )
            if "/" in name:
                raise GeoBrainError(
                    "OperatorBundle channel name must not contain '/'",
                    object_name="OperatorBundle",
                    field="channels",
                    expected="name without '/'",
                    actual=name,
                )
            if not isinstance(op, Operator):
                raise GeoBrainError(
                    "OperatorBundle channel value must be an Operator",
                    object_name="OperatorBundle",
                    field=f"channels[{name!r}]",
                    expected=Operator,
                    actual=type(op),
                )
            if not _terminates_in_observation(op):
                raise GeoBrainError(
                    "OperatorBundle channel must produce a ForwardOutput",
                    object_name="OperatorBundle",
                    field=f"channels[{name!r}]",
                    expected="ForwardOperator or OperatorChain terminating in one",
                    actual=type(op).__name__,
                )

        channel_configs = dict(channel_configs or {})
        unknown_cfg = set(channel_configs) - set(channels)
        if unknown_cfg:
            raise GeoBrainError(
                "channel_configs keys must be a subset of channel names",
                object_name="OperatorBundle",
                field="channel_configs",
                expected=f"subset of {sorted(channels)}",
                actual=sorted(unknown_cfg),
            )

        # Channels are held in an ``nn.ModuleDict``, which rejects a name
        # containing '.' or shadowing a Module attribute (``forward``, ``to``,
        # ``training``, ``keys``, …) with a raw KeyError/TypeError. Convert that
        # to a structured GeoBrainError so every channel-name failure stays on
        # one contract surface (matching nn's *actual* rule, version-robust).
        try:
            self._channels = nn.ModuleDict(dict(channels))
        except (KeyError, TypeError) as exc:
            raise GeoBrainError(
                "OperatorBundle channel name is not a valid nn.ModuleDict key "
                "(no '.', and must not shadow an nn.Module attribute such as "
                "'forward'/'to'/'training')",
                object_name="OperatorBundle",
                field="channels",
                expected="a valid nn.ModuleDict key",
                actual=sorted(channels),
            ) from exc
        # Reconcile INPUT units across members BEFORE building the spec: all
        # channels run on ONE shared ModelState, so a field two members read under
        # DIFFERENT unit conventions (e.g. one channel declaring ``x`` in ``foo``
        # vs. another declaring ``x`` in ``bar``) is silently physics-wrong in one
        # of them. Catch it loudly at construction rather than silently wrong in
        # the numbers.
        self._check_input_unit_consistency(channels)

        self._channel_order = tuple(channels.keys())
        self._channel_configs = channel_configs
        self.differentiability = self._build_spec(channels)

    @staticmethod
    def _check_input_unit_consistency(channels: Mapping[str, Operator]) -> None:
        """Reject a bundle whose members read a SHARED ModelState field under
        conflicting input units.

        Each member declares the units it expects for its input fields via
        ``differentiability.input_units`` (``{field: unit}``). Because the bundle
        threads ONE ``ModelState`` through every member, a field that two members
        declare with DIFFERENT unit strings cannot be satisfied by a single tensor;
        sharing it is silently wrong in one physics. This raises a
        :class:`GeoBrainError` naming the field, the conflicting units, and the
        members involved. Members that do not declare a unit for a field are
        skipped (opt-in, no false positives); agreeing declarations are fine.
        """
        # field -> {unit_string -> [channel names declaring it]}
        declared: dict[str, dict[str, list[str]]] = {}
        for name, op in channels.items():
            units = getattr(op.differentiability, "input_units", {}) or {}
            for field_name, unit in units.items():
                declared.setdefault(field_name, {}).setdefault(unit, []).append(name)
        for field_name, by_unit in declared.items():
            if len(by_unit) < 2:
                continue  # 0 or 1 distinct unit declared → nothing to reconcile
            detail = "; ".join(
                f"{unit!r} (declared by {', '.join(sorted(members))})"
                for unit, members in sorted(by_unit.items())
            )
            raise GeoBrainError(
                f"OperatorBundle members declare conflicting input units for the "
                f"shared ModelState field {field_name!r}: {detail}. All channels "
                f"run on one shared ModelState, so these units must be reconciled "
                f"before sharing a ModelState field (convert the field to a common "
                f"unit, or split the bundle).",
                object_name="OperatorBundle",
                field=field_name,
                expected="one input unit across all members sharing this field",
                actual=sorted(by_unit),
            )

    @staticmethod
    def _build_spec(channels: Mapping[str, Operator]) -> DifferentiabilitySpec:
        """Derive the joint spec: weakest level, union of inputs (trainable,
        structural, and reconciled ``input_units``), and output
        keys that MIRROR :meth:`_forward`'s merge rule, a single-output channel
        is lifted to the bare channel name, a multi-output channel is namespaced
        as ``f"{channel}/{subkey}"``. Declaring the bare channel name for a
        multi-output channel (the old behaviour) made ``output_keys`` disagree
        with the keys actually produced, so ``InverseProblem``'s observed-key
        validation rejected legitimate multi-output joint observations.
        """
        ops = list(channels.values())
        trainable: list[str] = []
        seen: set[str] = set()
        for op in ops:
            for name in op.differentiability.trainable_inputs:
                if name not in seen:
                    trainable.append(name)
                    seen.add(name)
        output_keys: list[str] = []
        for channel, op in channels.items():
            sub_keys = op.differentiability.output_keys
            if len(sub_keys) <= 1:
                output_keys.append(channel)
            else:
                output_keys.extend(f"{channel}/{subkey}" for subkey in sub_keys)
        structural: list[str] = []
        seen_structural: set[str] = set()
        merged_units: dict[str, str] = {}
        for op in ops:
            for name in op.differentiability.structural_params:
                if name not in seen_structural:
                    structural.append(name)
                    seen_structural.add(name)
            # Consistent by construction: _check_input_unit_consistency raised
            # on any conflicting declaration before this runs.
            merged_units.update(getattr(op.differentiability, "input_units", {}) or {})
        return DifferentiabilitySpec(
            level=_weakest_level(ops),
            trainable_inputs=tuple(trainable),
            output_keys=tuple(output_keys),
            structural_params=tuple(structural),
            input_units=merged_units,
        )

    @property
    def channel_names(self) -> tuple[str, ...]:
        """The channel names, in declaration order."""
        return self._channel_order

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ForwardOutput:
        """
        Run every channel on the shared ``state`` and merge their predictions.

        Args:
            state: Shared model parameters.
            ctx: Per-call configuration.

        Returns:
            A single :class:`ForwardOutput` keyed by channel name.
        """
        merged_data: dict[str, torch.Tensor] = {}
        merged_fields: dict[str, torch.Tensor] = {}
        merged_metadata: dict[str, Any] = {}
        # Per-channel units, re-keyed onto the merged data keys so the bundled
        # ForwardOutput carries a flat ``metadata["units"]`` the unit contract can
        # check (each member declares ``metadata["units"] = {sub_data_key: unit}``).
        merged_units: dict[str, str] = {}
        for name in self._channel_order:
            op = self._channels[name]
            overlay = self._channel_configs.get(name)
            ch_ctx = ctx.with_overrides(overlay) if overlay else ctx
            pred = op(state, ch_ctx)
            if not isinstance(pred, ForwardOutput):
                raise GeoBrainError(
                    "OperatorBundle channel produced a non-ForwardOutput",
                    object_name="OperatorBundle",
                    field=f"channels[{name!r}]",
                    expected=ForwardOutput,
                    actual=type(pred),
                )
            # The emitted data keys must be a subset of what the sub-operator
            # DECLARES: otherwise the merge (lift vs namespace below) keys off
            # `output_keys` while the data carries keys nobody declared, silently
            # dropping or misnaming channels. Use the instance's own spec, which
            # may be instance-level for dynamic operators.
            declared = set(op.differentiability.output_keys)
            emitted = set(pred.data.keys())
            undeclared = emitted - declared
            if undeclared:
                raise GeoBrainError(
                    "OperatorBundle channel emitted undeclared data keys",
                    object_name="OperatorBundle",
                    field=f"channels[{name!r}]",
                    expected=f"data keys subset of output_keys {sorted(declared)}",
                    actual=sorted(undeclared),
                )
            # Namespace on the channel's DECLARED output_keys count, matching
            # `_build_spec` above: keying both decisions off the same quantity
            # so the merged data keys always line up with the declared joint
            # output_keys (a sub-operator that conditionally emits a subset of
            # its declared channels, e.g. TransientFlow's well series, stays
            # consistent instead of flipping between the lifted/namespaced form).
            if len(op.differentiability.output_keys) <= 1:
                if not pred.data:
                    raise GeoBrainError(
                        "OperatorBundle channel emitted no data",
                        object_name="OperatorBundle",
                        field=f"channels[{name!r}]",
                        expected="exactly one data entry to lift to the channel name",
                        actual=f"{type(op).__name__} produced an empty ForwardOutput.data",
                    )
                ch_units = pred.metadata.get("units", {})
                subkey, value = next(iter(pred.data.items()))
                merged_data[name] = value
                if isinstance(ch_units, dict) and subkey in ch_units:
                    merged_units[name] = ch_units[subkey]
            else:
                ch_units = pred.metadata.get("units", {})
                for subkey, value in pred.data.items():
                    merged_data[f"{name}/{subkey}"] = value
                    if isinstance(ch_units, dict) and subkey in ch_units:
                        merged_units[f"{name}/{subkey}"] = ch_units[subkey]
            for subkey, value in pred.fields.items():
                merged_fields[f"{name}/{subkey}"] = value
            if pred.metadata:
                merged_metadata[name] = dict(pred.metadata)
        # Flat, merged-data-key-aligned units for the unit contract (a channel
        # literally named "units" is not expected for physics observables).
        if merged_units:
            merged_metadata["units"] = merged_units
        return ForwardOutput(
            data=merged_data,
            fields=merged_fields,
            metadata=merged_metadata,
        )

    def forward(
        self, state: ModelState, ctx: ForwardContext | None = None
    ) -> ForwardOutput:
        """
        Validate the input and run all channels. See :meth:`_forward`.

        Args:
            state: Shared model parameters.
            ctx: Per-call configuration. Defaults to an empty
                :class:`ForwardContext`.

        Returns:
            The merged :class:`ForwardOutput`.
        """
        ctx = self._validate_forward_input(state, ctx)
        return self._forward(state, ctx)

    def __repr__(self) -> str:
        parts = [f"{name!r}: {type(op).__name__}" for name, op in self._channels.items()]
        return f"OperatorBundle({{{', '.join(parts)}}})"
