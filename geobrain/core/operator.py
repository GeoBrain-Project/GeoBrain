"""
Operator ABC and its two specializations: PropertyTransform and ForwardOperator.

An :class:`Operator` is a node in the computation graph: it takes a
:class:`ModelState` (and a context) and produces either another
:class:`ModelState` (:class:`PropertyTransform`) or a :class:`ForwardOutput`
(:class:`ForwardOperator`). Every operator declares its differentiability via a
class-level ``differentiability`` attribute.

Subclasses implement ``_forward(state, ctx)``. The base ``forward`` validates the
input, delegates, then validates the output type.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)
Version: 0.2.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from torch import nn

from .differentiability import DifferentiabilitySpec
from .errors import DifferentiabilityError, GeoBrainError, MissingFieldError
from .containers import ModelState, ForwardOutput
from .context import ForwardContext
from ._require import require_capable_mesh


class Operator(nn.Module, ABC):
    """Abstract base. Subclass via PropertyTransform or ForwardOperator, not this.

    ``differentiability``: a plain, not ``ClassVar``,
    annotation. Most concrete operators satisfy it with a class attribute
    (``differentiability = DifferentiabilitySpec(...)``); operators whose spec
    depends on constructor arguments instead assign ``self.differentiability =
    DifferentiabilitySpec(...)`` once in ``__init__`` (a plain instance
    attribute, no property, no ``_dynamic_differentiability`` escape hatch).
    Both forms read identically off an instance. Nothing is enforced at class
    *definition* time any more (a per-instance spec cannot exist until
    ``__init__`` runs); the friendly check runs at first *use*; see
    :meth:`_check_trainable_inputs`. A subclass that defines neither raises
    :class:`AttributeError` on class-level access (bare annotation, no default)
    and the same, upgraded to a structured :class:`GeoBrainError`, on first
    ``forward()`` call.
    """

    differentiability: DifferentiabilitySpec
    # Capability markers the ctx mesh must declare. ENFORCED ON THE
    # ForwardOperator.forward PATH ONLY: it is inert on a PropertyTransform
    # (which never reads the ctx mesh; see the PropertyTransform docstring).
    requires_mesh_capabilities: ClassVar[tuple[type, ...]] = ()
    # The fixed name each forward family reports in its input-validation error
    # ("<label>.forward expects a ModelState"). Set on the four ``forward``-owning
    # classes (PropertyTransform/ForwardOperator/OperatorChain/OperatorBundle) and
    # inherited by their concretes, so the message is the FAMILY name, not the
    # concrete subclass name (that stays in ``object_name=type(self).__name__``).
    _forward_label: ClassVar[str] = "Operator"

    def _validate_forward_input(
        self, state: ModelState, ctx: ForwardContext | None
    ) -> ForwardContext:
        """Shared ``forward`` preamble for every operator family.

        Defaults ``ctx`` to an empty :class:`ForwardContext`, asserts ``state`` is
        a :class:`ModelState`, and presence/dtype-checks the declared trainable
        inputs. Returns the resolved ``ctx``. Behaviour-identical across the four
        forward families; the only per-family difference is the ``_forward_label``
        woven into the type-mismatch message.
        """
        if ctx is None:
            ctx = ForwardContext()
        if not isinstance(state, ModelState):
            raise GeoBrainError(
                f"{self._forward_label}.forward expects a ModelState",
                object_name=type(self).__name__,
                field="state",
                expected=ModelState,
                actual=type(state),
            )
        self._check_trainable_inputs(state)
        return ctx

    def __matmul__(self, other: "Operator") -> "Operator":
        """
        Compose two operators: ``B @ A`` applies ``A`` first, then ``B``.

        Follows the math convention ``(B ∘ A)(x) = B(A(x))``. If either side is
        already a chain it is flattened, so the result is one flat
        :class:`OperatorChain`.

        Args:
            other: The operator applied first (the right-hand side).

        Returns:
            An :class:`OperatorChain` representing the composition.
        """
        from .composition import OperatorChain  # local import to avoid cycle

        if not isinstance(other, Operator):
            return NotImplemented
        if isinstance(other, OperatorChain):
            right_ops = list(other.operators)
        else:
            right_ops = [other]
        if isinstance(self, OperatorChain):
            left_ops = list(self.operators)
        else:
            left_ops = [self]
        return OperatorChain(right_ops + left_ops)

    def _check_trainable_inputs(self, state: ModelState) -> None:
        """Verify every declared trainable input is present in ``state``
        and carries a floating or complex dtype.

        Only the DECLARED ``trainable_inputs`` are dtype-checked: integer
        tensors cannot ``requires_grad``, so an integer trainable input has no
        legitimate use and silently corrupts forwards (zero voltages, wrong
        kernels). Non-trainable auxiliary fields (index maps, masks, …) may
        legitimately be integer and are left alone.

        This is also the enforcement point: every concrete
        operator must have a usable ``differentiability`` spec by the time it
        runs. Static operators get this for free (a class attribute); dynamic
        ones must have assigned ``self.differentiability`` in ``__init__``, a
        subclass that does neither raises a friendly, structured error here
        instead of a bare ``AttributeError`` deep in ``spec.supports_gradients``.
        """
        spec = getattr(self, "differentiability", None)
        if not isinstance(spec, DifferentiabilitySpec):
            raise GeoBrainError(
                f"{type(self).__name__} must define differentiability "
                "(DifferentiabilitySpec) as a class attribute or set it in "
                "__init__",
                object_name=type(self).__name__,
                field="differentiability",
                expected=DifferentiabilitySpec,
                actual=type(spec),
            )
        if not spec.supports_gradients:
            return
        for name in spec.trainable_inputs:
            if name not in state.tensors:
                raise MissingFieldError(
                    "trainable input missing from ModelState",
                    object_name=type(self).__name__,
                    field=name,
                    expected=f"present in {sorted(state.tensors)}",
                )
            t = state.tensors[name]
            if not (t.is_floating_point() or t.is_complex()):
                raise DifferentiabilityError(
                    "trainable input must have a floating or complex dtype "
                    "(integer/bool tensors cannot carry gradients)",
                    object_name=type(self).__name__,
                    field=name,
                    expected="floating or complex dtype",
                    actual=t.dtype,
                )

    @abstractmethod
    def _forward(
        self, state: ModelState, ctx: ForwardContext
    ) -> ModelState | ForwardOutput:  # pragma: no cover
        """Subclass hook: compute the output from ``state`` and ``ctx``."""
        ...


class PropertyTransform(Operator):
    """ModelState → ModelState. For rock physics, parameterization, change-of-variable.

    Intended uses:
        - **Rock-physics transforms**: map one petrophysical state to another
          (e.g. Gardner ``rho = a·vp^b``, Gassmann fluid substitution).
        - **Model parameterization**: change-of-variable maps that re-express
          the inversion variables (e.g. log-velocity → velocity, or a
          coarse-grid basis → cell values) before a downstream forward.

    Unlike :class:`ForwardOperator`, a ``PropertyTransform`` does **not** enforce
    ``requires_mesh_capabilities``. The operative distinction is **not** whether a
    transform touches geometry; it is whether it reads the per-call *ctx* mesh.
    That enforcement governs only the ctx mesh (the one ``require_capable_mesh(ctx, …)``
    fetches), and a transform never reads it: a transform is a
    ``ModelState → ModelState`` map. A transform may still operate on a discretized
    geometry, e.g. :class:`~geobrain.mesh.projection.MeshProjection` reshapes
    a field between its own constructor-bound source/target meshes, but that
    geometry is self-managed, never threaded through ``ctx``, so there is no
    ctx-mesh contract to check. A transform whose *working* mesh would come from
    ``ctx`` belongs on the forward side. (This matches the in-tree contract test
    ``test_capability_contract``, which keys the same rule on ctx-mesh reads.)
    """

    _forward_label = "PropertyTransform"

    def forward(self, state: ModelState, ctx: ForwardContext | None = None) -> ModelState:
        """
        Validate the input, run :meth:`_forward`, and check the result type.

        Args:
            state: Input model parameters.
            ctx: Per-call configuration. Defaults to an empty
                :class:`ForwardContext`.

        Returns:
            The transformed :class:`ModelState`.
        """
        ctx = self._validate_forward_input(state, ctx)
        out = self._forward(state, ctx)
        if not isinstance(out, ModelState):
            raise GeoBrainError(
                "PropertyTransform._forward must return a ModelState",
                object_name=type(self).__name__,
                expected=ModelState,
                actual=type(out),
            )
        # A transform may carry through arbitrary input fields, so extra state
        # keys are allowed. But every field it declares as produced must be
        # present after the transform, otherwise chain specs lie about what the
        # next operator can consume.
        declared = set(self.differentiability.output_keys)
        missing = declared - set(out.tensors)
        if missing:
            raise GeoBrainError(
                "PropertyTransform did not emit declared output fields",
                object_name=type(self).__name__,
                field="output_keys",
                expected=f"declared fields present in ModelState {sorted(declared)}",
                actual=sorted(missing),
            )
        return out


class ForwardOperator(Operator):
    """ModelState → ForwardOutput. For forward modeling."""

    _forward_label = "ForwardOperator"

    def forward(self, state: ModelState, ctx: ForwardContext | None = None) -> ForwardOutput:
        """
        Validate the input, run :meth:`_forward`, and check the result type.

        Args:
            state: Input model parameters.
            ctx: Per-call configuration. Defaults to an empty
                :class:`ForwardContext`.

        Returns:
            The :class:`ForwardOutput` produced by the operator.
        """
        ctx = self._validate_forward_input(state, ctx)
        if self.requires_mesh_capabilities:
            require_capable_mesh(ctx, *self.requires_mesh_capabilities,
                                 owner=type(self).__name__)
        out = self._forward(state, ctx)
        if not isinstance(out, ForwardOutput):
            raise GeoBrainError(
                "ForwardOperator._forward must return a ForwardOutput",
                object_name=type(self).__name__,
                expected=ForwardOutput,
                actual=type(out),
            )
        # Emitted data keys must be a subset of the declared output_keys. This
        # mirrors OperatorBundle._forward's per-channel check: a forward that
        # writes a data key it never declared makes its DifferentiabilitySpec lie
        # about what it produces, which breaks downstream key-based wiring
        # (InverseProblem observed-key validation, bundle merge namespacing).
        # The contract governs `out.data` ONLY: `out.fields` are diagnostics and
        # are exempt by design.
        declared = set(self.differentiability.output_keys)
        undeclared = set(out.data) - declared
        if undeclared:
            raise GeoBrainError(
                "ForwardOperator emitted undeclared data keys",
                object_name=type(self).__name__,
                field="output_keys",
                expected=f"data keys subset of output_keys {sorted(declared)}",
                actual=sorted(undeclared),
            )
        return out
