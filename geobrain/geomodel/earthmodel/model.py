"""``EarthModel``: a typed, resolvable multi-field earth model.

Binds a mesh + a namespace of :class:`~geobrain.geomodel.earthmodel.Field` /
:class:`~geobrain.geomodel.earthmodel.Link` descriptors into one object that:

- validates its dependency DAG once, at construction (cycles, unknown args,
  name collisions, init-shape-vs-mesh);
- evaluates it functionally via :meth:`EarthModel.resolve`: a LOCAL memo per
  call (never instance state), so ``resolve() -> backward() -> resolve() ->
  backward()`` works without a "backward through the graph a second time"
  error;
- exposes its trainable leaves (namespaced by field/generator-param/link-param)
  via :meth:`EarthModel.trainables`;
- bridges into any physics chain via :meth:`EarthModel.as_transform`: a
  :class:`~geobrain.core.operator.PropertyTransform` so
  ``physics_op @ model.as_transform()`` drives the SAME ``Inverter`` / bayes
  machinery every other :class:`~geobrain.InverseProblem` uses (the
  optimization identity contract: gradients flow from a downstream loss,
  through ``resolve``, into the state-supplied leaves).

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

import copy
import os
import warnings
from collections import Counter
from typing import Any, Mapping, Sequence

import torch

from geobrain.core.containers import ModelState
from geobrain.core.transforms import InvertibleTransform
from geobrain.core.context import ForwardContext
from geobrain.core.differentiability import (
    DifferentiabilityLevel,
    DifferentiabilitySpec,
)
from geobrain.core.errors import GeoBrainError
from geobrain.core.operator import PropertyTransform

from .field import Field
from .field_specs import FIELD_SPECS
from .link import Link

__all__ = ["EarthModel"]

# Reserved: no field/link may be literally named "links" (the trainable
# namespace prefix for link params, "links.<name>.<param>"), and none may
# contain "." (the namespace separator for generator-param and link-param
# trainable names): see ``_validate_names_and_dag`` for the collision proof
# these two invariants make possible.
_RESERVED_NAME = "links"


class _ValidationState:
    """Shared, mutable first-resolve-validation bookkeeping.

    Deliberately SHARED (by reference) across every ``EarthModel`` produced
    by :meth:`EarthModel.with_values` from a common ancestor, otherwise a
    functional clone built fresh every forward pass (as
    ``as_transform()._forward`` and ``resolve_from`` both do) would see EVERY
    resolve as "first", permanently escalating in-loop drift warnings into
    training-killing raises. See ``EarthModel.resolve``'s docstring for the
    two-tier (raise-once / warn-once) semantics this implements.
    """

    __slots__ = ("seen", "warned", "validate_every")

    def __init__(self, validate_every: bool) -> None:
        self.seen: set[str] = set()
        self.warned: set[str] = set()
        self.validate_every = validate_every


def _validate_names_and_dag(fields: Mapping[str, Field], links: Mapping[str, Link]) -> None:
    """Construction-time DAG validation: collisions, reserved names, unknown
    args, and cycles (with the offending path named in the error)."""
    field_names = set(fields)
    link_names = set(links)

    collisions = field_names & link_names
    if collisions:
        raise GeoBrainError(
            "EarthModel field and link names must be disjoint",
            object_name="EarthModel",
            field="fields/links",
            expected="disjoint name sets",
            actual=sorted(collisions),
        )

    all_names = field_names | link_names
    for name in all_names:
        if "." in name:
            raise GeoBrainError(
                "EarthModel field/link names must not contain '.' "
                "(reserved as the trainable-name namespace separator)",
                object_name="EarthModel",
                field="fields/links",
                expected="name without '.'",
                actual=name,
            )
        if name == _RESERVED_NAME:
            raise GeoBrainError(
                f"EarthModel field/link name {name!r} is reserved "
                "(the 'links.<name>.<param>' trainable namespace prefix)",
                object_name="EarthModel",
                field="fields/links",
                expected=f"name != {_RESERVED_NAME!r}",
                actual=name,
            )

    for lname, link in links.items():
        missing = [a for a in link.args if a not in all_names]
        if missing:
            raise GeoBrainError(
                f"Link {lname!r} references unknown args {missing}",
                object_name="EarthModel",
                field=f"links[{lname!r}].args",
                expected=f"subset of available names {sorted(all_names)}",
                actual=missing,
            )

    # Cycle detection via DFS over the link->link subgraph (base fields are
    # always leaves: they have no ``args``). White/gray/black coloring names
    # the exact cycle path on failure.
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in link_names}
    path: list[str] = []

    def visit(n: str) -> None:
        color[n] = GRAY
        path.append(n)
        for a in links[n].args:
            if a not in link_names:
                continue
            if color[a] == GRAY:
                start = path.index(a)
                cycle = path[start:] + [a]
                raise GeoBrainError(
                    "EarthModel link DAG has a cycle: " + " -> ".join(cycle),
                    object_name="EarthModel",
                    field="links",
                    expected="acyclic link dependency graph",
                    actual=" -> ".join(cycle),
                )
            if color[a] == WHITE:
                visit(a)
        path.pop()
        color[n] = BLACK

    for n in link_names:
        if color[n] == WHITE:
            visit(n)


def _majority_dtype_device(fields: Mapping[str, Field]) -> tuple[torch.dtype, torch.device]:
    """The unified (dtype, device) every trainable leaf is cast to: the
    majority among ``init=``-carrying (non-generator) fields, falling back to
    the first generator's first parameter, then ``float32``/``cpu``."""
    dtypes: list[torch.dtype] = []
    devices: list[torch.device] = []
    for f in fields.values():
        if f.init is not None:
            dtypes.append(f.init.dtype)
            devices.append(f.init.device)
    if not dtypes:
        for f in fields.values():
            net = f.generator[0] if isinstance(f.generator, tuple) else f.generator
            if net is None:
                continue
            p = next(net.parameters(), None)
            if p is not None:
                dtypes.append(p.dtype)
                devices.append(p.device)
                break
    if not dtypes:
        return torch.float32, torch.device("cpu")
    dtype = Counter(dtypes).most_common(1)[0][0]
    device = Counter(devices).most_common(1)[0][0]
    return dtype, device


def _forward_chain(transforms: Sequence["InvertibleTransform"], leaf: torch.Tensor) -> torch.Tensor:
    value = leaf
    for t in transforms:
        value = t.forward(value)
    return value


def _inverse_chain(transforms: Sequence["InvertibleTransform"], physical: torch.Tensor) -> torch.Tensor:
    value = physical
    for t in reversed(transforms):
        value = t.inverse(value)
    return value


def _pin(value: torch.Tensor, where: Any, values: torch.Tensor) -> torch.Tensor:
    """``torch.where``-substitute ``value`` with DETACHED ``values`` at
    ``where`` (a boolean mask -> elementwise ``torch.where``; a slice/tuple of
    slices -> clone + index assignment). No gradient flows into ``value`` at
    the pinned positions."""
    values_d = values.detach().to(dtype=value.dtype, device=value.device)
    if isinstance(where, torch.Tensor) and where.dtype == torch.bool:
        return torch.where(where, values_d, value)
    out = value.clone()
    out[where] = values_d[where]
    return out


class EarthModel:
    """A typed, resolvable multi-field earth model.

    Args:
        mesh: The geometry every field lives on (an object exposing
            ``.shape``, e.g. :class:`~geobrain.mesh.TensorMesh`). Every
            ``init=``-carrying field's tensor must match ``mesh.shape``
            exactly.
        fields: ``{name: Field}``, the model's base unknowns.
        links: ``{name: Link}``, derived fields, evaluated topologically
            after their ``args`` dependencies. ``None`` (default), no links.
        frozen: ``{name: (where, values)}``, resolve-time region pinning
            (works for base AND derived names): in the ``where``-selected
            cells the resolved field is substituted with DETACHED ``values``
            (see :func:`_pin`), applied LAST (after transforms/generator for
            a base field, after ``fn`` for a link) so no gradient reaches the
            pinned cells and any downstream link sees the pinned value.
            ``None`` (default), no pinning.

    Raises:
        GeoBrainError: On any construction-time validation failure, field/
            link name collision or reservation, a link referencing an unknown
            arg, a cyclic link DAG, an ``init=`` shape mismatch against
            ``mesh.shape``, or an unknown ``frozen`` key.
    """

    def __init__(
        self,
        mesh: Any,
        fields: Mapping[str, Field],
        links: Mapping[str, Link] | None = None,
        frozen: Mapping[str, tuple[Any, torch.Tensor]] | None = None,
    ) -> None:
        fields = dict(fields)
        links = dict(links or {})
        frozen = dict(frozen or {})

        if not fields:
            raise GeoBrainError(
                "EarthModel requires at least one field",
                object_name="EarthModel",
                field="fields",
                expected="non-empty mapping",
                actual={},
            )
        for name, f in fields.items():
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "EarthModel field names must be non-empty strings",
                    object_name="EarthModel", field="fields", actual=name,
                )
            if not isinstance(f, Field):
                raise GeoBrainError(
                    "EarthModel fields values must be Field instances",
                    object_name="EarthModel", field=f"fields[{name!r}]",
                    expected=Field, actual=type(f),
                )
        for name, link in links.items():
            if not isinstance(name, str) or not name:
                raise GeoBrainError(
                    "EarthModel link names must be non-empty strings",
                    object_name="EarthModel", field="links", actual=name,
                )
            if not isinstance(link, Link):
                raise GeoBrainError(
                    "EarthModel links values must be Link instances",
                    object_name="EarthModel", field=f"links[{name!r}]",
                    expected=Link, actual=type(link),
                )

        _validate_names_and_dag(fields, links)

        mesh_shape = tuple(getattr(mesh, "shape", ()))
        for name, f in fields.items():
            if f.init is not None and tuple(f.init.shape) != mesh_shape:
                raise GeoBrainError(
                    "Field init= shape does not match mesh.shape",
                    object_name="EarthModel",
                    field=name,
                    expected=mesh_shape,
                    actual=tuple(f.init.shape),
                )

        unknown_frozen = set(frozen) - (set(fields) | set(links))
        if unknown_frozen:
            raise GeoBrainError(
                "EarthModel frozen= keys must be known field/link names",
                object_name="EarthModel",
                field="frozen",
                expected=f"subset of {sorted(set(fields) | set(links))}",
                actual=sorted(unknown_frozen),
            )
        for name, pin in frozen.items():
            if not (isinstance(pin, tuple) and len(pin) == 2):
                raise GeoBrainError(
                    "EarthModel frozen= values must be (where, values) pairs",
                    object_name="EarthModel",
                    field=f"frozen[{name!r}]",
                    expected="(where, values) tuple",
                    actual=pin,
                )
            _, values = pin
            if not isinstance(values, torch.Tensor):
                raise GeoBrainError(
                    "EarthModel frozen= values[1] must be a torch.Tensor",
                    object_name="EarthModel",
                    field=f"frozen[{name!r}]",
                    expected=torch.Tensor,
                    actual=type(values),
                )

        dtype, device = _majority_dtype_device(fields)

        field_leaf: dict[str, torch.Tensor] = {}
        generator_weights: dict[str, dict[str, torch.Tensor]] = {}
        for name, f in fields.items():
            if f.generator is not None:
                net = f.generator[0] if isinstance(f.generator, tuple) else f.generator
                net.to(dtype=dtype, device=device)
                generator_weights[name] = {pname: p for pname, p in net.named_parameters()}
            else:
                assert f.init is not None  # Field invariant: init xor generator
                init = f.init.to(dtype=dtype, device=device)
                field_leaf[name] = _inverse_chain(f.transforms, init)

        link_param: dict[str, dict[str, torch.Tensor]] = {}
        for name, link in links.items():
            params: dict[str, torch.Tensor] = {}
            for pname, v in link.params.items():
                if isinstance(v, torch.Tensor):
                    params[pname] = v.to(dtype=dtype, device=device)
                else:
                    params[pname] = torch.tensor(float(v), dtype=dtype, device=device)
            link_param[name] = params

        self.mesh = mesh
        self._fields = fields
        self._links = links
        self._frozen = frozen
        self._dtype = dtype
        self._device = device
        self._field_leaf = field_leaf
        self._generator_weights = generator_weights
        self._link_param = link_param
        self._validation_state = _ValidationState(
            validate_every=(os.environ.get("GEOBRAIN_VALIDATE") == "1")
        )

    def __repr__(self) -> str:
        return (
            f"EarthModel(fields={sorted(self._fields)}, "
            f"links={sorted(self._links)}, dtype={self._dtype}, device={self._device})"
        )

    # -----------------------------------------------------------------
    # Field access
    # -----------------------------------------------------------------

    def field(self, name: str) -> Field:
        """The :class:`~geobrain.geomodel.earthmodel.Field` descriptor for base field ``name``.

        The public per-field accessor: :meth:`trainables` hands back leaf
        TENSORS, but the descriptor itself (``transforms``/``generator``/
        ``unit``) lives on the ``Field`` object, consumers like
        ``geobrain.geomodel.bridge.ensemble_to_particles`` need it to map
        physical values into leaf space. Links are derived, not base fields,
        so a link name raises too (named as a link in the message).

        Args:
            name: A base-field name (a ``fields=`` key at construction).

        Returns:
            The ``Field`` object passed at construction (the shared
            descriptor itself, never a copy).

        Raises:
            KeyError: ``name`` is not a base field. The message lists the
                available field names, and flags ``name`` as a Link when it
                is one.
        """
        if name in self._fields:
            return self._fields[name]
        hint = " (it is a Link, not a Field)" if name in self._links else ""
        raise KeyError(
            f"EarthModel has no field {name!r}{hint}; "
            f"available fields: {sorted(self._fields)}"
        )

    # -----------------------------------------------------------------
    # Resolution
    # -----------------------------------------------------------------

    def _compute_base_field(self, name: str) -> torch.Tensor:
        field = self._fields[name]
        if field.generator is not None:
            net = field.generator[0] if isinstance(field.generator, tuple) else field.generator
            fixed_input = field.generator[1] if isinstance(field.generator, tuple) else None
            weights = self._generator_weights[name]
            args = (fixed_input,) if fixed_input is not None else ()
            # torch.func exports functional_call at runtime, but the bundled
            # stubs omit it from the module's explicit exports.
            out = torch.func.functional_call(  # type: ignore[attr-defined]
                net, weights, args
            )
            if not isinstance(out, torch.Tensor):
                raise GeoBrainError(
                    "Field generator forward must return a torch.Tensor",
                    object_name="EarthModel", field=name,
                    expected=torch.Tensor, actual=type(out),
                )
            mesh_shape = tuple(getattr(self.mesh, "shape", ()))
            if tuple(out.shape) != mesh_shape:
                raise GeoBrainError(
                    "Field generator output shape does not match mesh.shape",
                    object_name="EarthModel", field=name,
                    expected=mesh_shape, actual=tuple(out.shape),
                )
            return out
        return _forward_chain(field.transforms, self._field_leaf[name])

    def _unit_for(self, name: str) -> str | None:
        obj: Field | Link | None = self._fields.get(name)
        if obj is None:
            obj = self._links.get(name)
        if obj is not None and obj.unit is not None:
            return obj.unit
        spec = FIELD_SPECS.get(name)
        return spec[0] if spec is not None else None

    def _validate_value(self, name: str, value: torch.Tensor) -> None:
        if not bool(torch.isfinite(value).all()):
            raise GeoBrainError(
                f"EarthModel field {name!r} resolved to non-finite values (NaN/Inf)",
                object_name="EarthModel", field=name,
                expected="all-finite tensor", actual="contains NaN/Inf",
            )
        spec = FIELD_SPECS.get(name)
        if spec is None:
            return
        unit, (lo, hi) = spec
        in_range = bool(((value >= lo) & (value <= hi)).all())
        state = self._validation_state
        is_first = name not in state.seen
        state.seen.add(name)
        if in_range:
            return
        vmin = float(value.detach().min())
        vmax = float(value.detach().max())
        if is_first or state.validate_every:
            raise GeoBrainError(
                f"EarthModel field {name!r} is outside its plausible SI range",
                object_name="EarthModel", field=name,
                expected=f"[{lo}, {hi}] {unit}", actual=(vmin, vmax),
            )
        if name not in state.warned:
            state.warned.add(name)
            warnings.warn(
                f"EarthModel field {name!r} drifted outside its plausible SI "
                f"range [{lo}, {hi}] {unit} (observed [{vmin}, {vmax}]) on a "
                "later resolve, the model already passed its first-resolve "
                "check, so this is a warning (set GEOBRAIN_VALIDATE=1 to "
                "escalate every resolve to a hard error).",
                stacklevel=3,
            )

    def _resolve_all(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Topological evaluation with a memo LOCAL to this call (never
        instance state; see the module docstring's ``resolve() -> backward()
        -> resolve() -> backward()`` normative test)."""
        memo: dict[str, torch.Tensor] = {}

        def get(name: str) -> torch.Tensor:
            if name in memo:
                return memo[name]
            if name in self._fields:
                value = self._compute_base_field(name)
            elif name in self._links:
                link = self._links[name]
                args = [get(a) for a in link.args]
                params = self._link_param.get(name, {})
                value = link.fn(*args, **params)
                if not isinstance(value, torch.Tensor):
                    raise GeoBrainError(
                        "Link fn must return a torch.Tensor",
                        object_name="EarthModel", field=name,
                        expected=torch.Tensor, actual=type(value),
                    )
            else:
                raise GeoBrainError(
                    f"EarthModel.resolve: unknown name {name!r}",
                    object_name="EarthModel", field="names",
                    expected=f"one of {sorted(set(self._fields) | set(self._links))}",
                    actual=name,
                )
            if name in self._frozen:
                where, values = self._frozen[name]
                value = _pin(value, where, values)
            self._validate_value(name, value)
            memo[name] = value
            return value

        for n in names:
            get(n)
        return memo

    def resolve(self, *names: str) -> ModelState:
        """Evaluate the model's DAG and return the resolved
        :class:`~geobrain.core.ModelState`.

        Args:
            *names: Field/link names to resolve. Empty (the default) means
                ALL names: every base field plus every link.

        Returns:
            A :class:`~geobrain.core.ModelState` with one tensor per
            requested name (transforms applied, generators run, links
            evaluated, frozen regions pinned) and ``metadata["units"]``
            stamped for every name with a known unit (explicit
            ``Field``/``Link`` ``unit=``, or the
            :data:`~geobrain.geomodel.earthmodel.field_specs.FIELD_SPECS` default).

        Raises:
            GeoBrainError: An unrecognised name; a resolved value containing
                NaN/Inf (always); or a value outside its
                :data:`~geobrain.geomodel.earthmodel.field_specs.FIELD_SPECS` plausible
                range on the FIRST resolve of that (model, name) pair (a
                LATER out-of-range resolve instead warns once via ``warnings.warn``;
                see :envvar:`GEOBRAIN_VALIDATE` to escalate every resolve to
                a hard raise).

        Note:
            The evaluation memo is a plain local variable of this call,
            never stored on ``self``, so a fresh graph is built every call:
            ``resolve() -> loss.backward() -> resolve() -> loss.backward()``
            works (a shared instance-level cache would replay an
            already-freed graph on the second ``backward()``).
        """
        all_names = tuple(self._fields) + tuple(self._links)
        if not names:
            names = all_names
        else:
            known = set(self._fields) | set(self._links)
            unknown = [n for n in names if n not in known]
            if unknown:
                raise GeoBrainError(
                    f"EarthModel.resolve: unknown name(s) {unknown}",
                    object_name="EarthModel", field="names",
                    expected=f"subset of {sorted(known)}", actual=unknown,
                )
        memo = self._resolve_all(names)
        tensors = {n: memo[n] for n in names}
        units: dict[str, str] = {}
        for n in names:
            u = self._unit_for(n)
            if u is not None:
                units[n] = u
        metadata = {"units": units} if units else {}
        return ModelState(tensors=tensors, metadata=metadata)

    # -----------------------------------------------------------------
    # Trainables / functional update
    # -----------------------------------------------------------------

    def trainables(self) -> dict[str, torch.Tensor]:
        """The model's trainable leaves, namespaced by origin.

        - Base (non-generator) fields: bare field name, UNCONSTRAINED space
          (the raw leaf; apply the field's transform to get the physical
          value; :meth:`resolve` does this).
        - Generator fields: ``"<field>.<param_name>"`` per
          ``net.named_parameters()``.
        - Trainable link params (``requires_grad=True``): ``"links.<link>.<param>"``.

        All leaves share ONE dtype/device (the majority dtype among
        ``init=``-carrying fields, decided once at construction; see
        :func:`_majority_dtype_device`; a mismatched field/generator/param
        is cast to it there, not here).

        Returns:
            A fresh ``dict`` (safe to mutate; the underlying tensors are the
            model's own live leaves, mutate a returned TENSOR in place and
            you mutate the model's state; replace a key's VALUE and nothing
            changes until you pass it through :meth:`with_values`).
        """
        out: dict[str, torch.Tensor] = {}
        for name, t in self._field_leaf.items():
            out[name] = t
        for field_name, weights in self._generator_weights.items():
            for pname, t in weights.items():
                key = f"{field_name}.{pname}"
                if key in out:
                    raise GeoBrainError(
                        f"EarthModel trainable name collision: {key!r}",
                        object_name="EarthModel", field="trainables", actual=key,
                    )
                out[key] = t
        for link_name, params in self._link_param.items():
            for pname, t in params.items():
                if not t.requires_grad:
                    continue
                key = f"links.{link_name}.{pname}"
                if key in out:
                    raise GeoBrainError(
                        f"EarthModel trainable name collision: {key!r}",
                        object_name="EarthModel", field="trainables", actual=key,
                    )
                out[key] = t
        return out

    def _trainable_kind(self, name: str) -> tuple[str, ...]:
        if "." not in name:
            if name in self._field_leaf:
                return ("field", name)
            raise GeoBrainError(
                f"EarthModel.with_values: unknown trainable name {name!r}",
                object_name="EarthModel", field="updates",
                expected=f"one of {sorted(self.trainables())}", actual=name,
            )
        head, rest = name.split(".", 1)
        if head == _RESERVED_NAME:
            linkname, _, paramname = rest.partition(".")
            if (
                paramname
                and linkname in self._link_param
                and paramname in self._link_param[linkname]
            ):
                return ("link_param", linkname, paramname)
        elif head in self._generator_weights and rest in self._generator_weights[head]:
            return ("field_param", head, rest)
        raise GeoBrainError(
            f"EarthModel.with_values: unknown trainable name {name!r}",
            object_name="EarthModel", field="updates",
            expected=f"one of {sorted(self.trainables())}", actual=name,
        )

    def with_values(self, **updates: torch.Tensor) -> "EarthModel":
        """Functional replacement of trainable leaves BY NAME.

        Args:
            **updates: ``{trainable_name: new_tensor}``, any subset of
                :meth:`trainables`' keys (dotted names included, e.g.
                ``**{"vp.weight": w, "links.archie.a": a}}``).

        Returns:
            A NEW ``EarthModel`` sharing this model's immutable structure
            (mesh, ``Field``/``Link`` objects, ``frozen`` spec, and the
            shared first-resolve-validation bookkeeping) with only the named
            leaves replaced. The original model is untouched (functional
            purity).

        Raises:
            GeoBrainError: An update key is not a known trainable name.
        """
        valid = set(self.trainables())
        unknown = set(updates) - valid
        if unknown:
            raise GeoBrainError(
                f"EarthModel.with_values: unknown trainable name(s) {sorted(unknown)}",
                object_name="EarthModel", field="updates",
                expected=f"subset of {sorted(valid)}", actual=sorted(unknown),
            )

        new = copy.copy(self)
        new._field_leaf = dict(self._field_leaf)
        new._generator_weights = {k: dict(v) for k, v in self._generator_weights.items()}
        new._link_param = {k: dict(v) for k, v in self._link_param.items()}

        for name, tensor in updates.items():
            kind = self._trainable_kind(name)
            if kind[0] == "field":
                new._field_leaf[kind[1]] = tensor
            elif kind[0] == "field_param":
                new._generator_weights[kind[1]][kind[2]] = tensor
            elif kind[0] == "link_param":
                new._link_param[kind[1]][kind[2]] = tensor
        return new

    def resolve_from(self, params: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The blessed physical-value recovery one-liner:
        ``with_values(**params).resolve()``, detached.

        Args:
            params: ``{trainable_name: tensor}``, typically an
                :class:`~geobrain.optim.InversionResult` 's ``.params`` (or
                any subset of :meth:`trainables`' keys).

        Returns:
            ``{name: tensor.detach()}`` for every base field and link, the
            PHYSICAL fields, even when some are transform-/generator-backed
            (unlike ``result.params`` itself, which hands back raw
            unconstrained-space leaves / generator weights with no
            documented way back to physical space).
        """
        state = self.with_values(**dict(params)).resolve()
        return {k: v.detach() for k, v in state.tensors.items()}

    # -----------------------------------------------------------------
    # Chain bridge
    # -----------------------------------------------------------------

    def as_transform(self) -> PropertyTransform:
        """A chain-composable :class:`~geobrain.core.operator.PropertyTransform`
        bridging this model into any physics chain: ``physics_op @
        model.as_transform()``.

        Reads the trainable leaves BY NAME from the incoming
        :class:`~geobrain.core.ModelState` (never from this model's own
        stored leaves, the identity contract requires the leaves an
        upstream solver is actually optimizing, e.g. an
        :class:`~geobrain.optim.Inverter`'s cloned ``nn.Parameter`` s, to be
        the ones ``resolve`` differentiates through), runs the functional
        resolve, and emits the resolved physical fields (plus stamped
        ``metadata["units"]``) into the state, carrying through any other
        incoming tensors unchanged.

        The returned transform's ``differentiability.trainable_inputs`` is
        ``tuple(self.trainables())``: the chain's entry-point contract
        (``OperatorChain`` takes ``trainable_inputs`` from its FIRST member)
       , and ``output_keys`` is every field + link name, level
        ``FULL_AUTOGRAD``.
        """
        return _EarthAsTransform(self)


class _EarthAsTransform(PropertyTransform):
    """See :meth:`EarthModel.as_transform`."""

    def __init__(self, model: EarthModel) -> None:
        super().__init__()
        self._model = model
        trainable_names = tuple(model.trainables())
        output_names = tuple(model._fields) + tuple(model._links)
        self.differentiability = DifferentiabilitySpec(
            level=DifferentiabilityLevel.FULL_AUTOGRAD,
            trainable_inputs=trainable_names,
            output_keys=output_names,
        )

    def _forward(self, state: ModelState, ctx: ForwardContext) -> ModelState:
        leaves = {
            name: state.tensors[name] for name in self.differentiability.trainable_inputs
        }
        resolved = self._model.with_values(**leaves).resolve()

        metadata = dict(state.metadata)
        resolved_units = resolved.metadata.get("units", {})
        if resolved_units:
            units = dict(state.metadata.get("units", {}))
            units.update(resolved_units)
            metadata["units"] = units

        return ModelState(
            tensors={**state.tensors, **resolved.tensors},
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"EarthModel.as_transform({self._model!r})"
